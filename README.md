# nvidia-converge

[![CI](https://github.com/zeroecco/nvidia-converge/actions/workflows/ci.yml/badge.svg)](https://github.com/zeroecco/nvidia-converge/actions/workflows/ci.yml)

`nvidia-converge` is a node-level NVIDIA driver stack reconciler. It reads a desired state, audits the host, explains breakage, previews package/kernel/module changes, optionally converges the machine, validates the GPU stack, pins compatibility-sensitive packages, records rollback metadata, and emits a machine-readable compliance report.

The default desired state is:

```yaml
desired:
  role: compute
  driver: 580-open
  cuda_compat: none
  secure_boot: signed
  container_runtime: docker
  container_test_image: nvidia/cuda:13.1.2-devel-ubuntu22.04@sha256:d8332c008e2ef270e82d286e5245e771f839645683075bba21ad9e4fa59dbcbb
  fabric_manager: false
  mig: disabled
  mig_profile: none
  kernel_policy: pin-compatible
```

The container verification image must be an `nvidia/cuda` devel image pinned to a lowercase SHA256 digest. The verifier compiles its packaged, hash-bound CUDA Driver API probe inside that isolated image and calls `cuInit` and `cuDeviceGetCount` through the injected host driver. This release accepts only `cuda_compat: none`: the configured driver and container CUDA version must qualify for native CUDA minor-version compatibility, and forward-compatibility loader deployment is rejected because it is not reversibly modeled.

## Production prerequisites

Before allowing this tool to mutate a host:

- Drain the node and place it in a maintenance window. Package changes, Docker restarts, Fabric Manager transitions, module operations, and MIG changes can interrupt workloads.
- Start from a host whose current NVIDIA stack can enumerate every GPU with `nvidia-smi`, including current and pending MIG state and compute capability. First-driver bootstrap and recovery from an unenumerable or partially modeled GPU stack are deliberately out of scope and fail closed.
- Configure the distribution's signed NVIDIA driver/CUDA repository, NVIDIA Container Toolkit repository, and Docker CE repository using the vendor-supported signing-key mechanism. `nvidia-converge` consumes already-trusted package repositories; it does not bootstrap repository keys or trust policy.
- Confirm that those repositories contain the requested driver selector, Fabric Manager package when enabled, NVIDIA Container Toolkit, Docker CE, and running-kernel build dependencies. CUDA forward-compatibility packages are outside this release's supported desired state.
- Keep an independent host recovery path. Rollback restores the state modeled by its snapshot; it is not a complete system backup and does not replace image, volume, or configuration backups.
- Use a disposable GPU node to qualify the exact release and desired state before production rollout. See [Integration Testing](docs/integration-testing.md).

### Implemented recipes versus production qualification

The implemented convergence recipes are deliberately narrow. Recipe recognition means the code can audit and plan that package-manager shape; it is not, by itself, a production-support claim.

They repair and converge drift within an already selected driver branch and kernel-module flavor. First-driver bootstrap and in-place branch, exact-version, or open/closed flavor transitions are not qualified; use a pre-baked image or the vendor migration procedure, then rerun `doctor` and `plan`.

| Family | Package backend | Implemented releases and constraints |
| --- | --- | --- |
| Ubuntu / Debian | `apt-get` | Ubuntu 22, 24, and 26; Debian 12 and 13. Branch and exact-version selection use the vendor NVIDIA pinning package. |
| RHEL family | `dnf` | x86_64 RHEL, Rocky Linux, and AlmaLinux 8 or 9. Branch selection uses the NVIDIA DNF module stream; aarch64/64K kernels, Oracle Linux/UEK, yum-only hosts, and exact driver versions fail closed. |
| SUSE family | `zypper` | SLES and openSUSE Leap 15 or 16 with a derivable `default`, `azure`, or `64k` running-kernel variant. Exact driver versions fail closed. |

A release is production-qualified only for the exact OS ID/version, desired-state file, capabilities, and scenarios in its passing tag-specific evidence. Its machine-readable `qualified_platforms` array must exactly equal the distinct `(os_id, os_version, package_manager)` tuples from successful applied runs. The separate `recipe_path_coverage` booleans only prove that the `apt-get`, `dnf`, and `zypper` code paths ran; they never promote an unobserved family member. The current workflow targets Ubuntu 24.04, one exactly observed RHEL-family 9 release, and one exactly observed SLES/openSUSE Leap 15 release; recognized releases outside those observed tuples remain preview/dry-run targets. The evidence bundle, not the broader table above, is authoritative.

All implemented recipe targets require Python 3.10 or newer with `venv`/`ensurepip` support. CI exercises CPython 3.10 through 3.14. The unversioned `python3` on a base RHEL 8/9 or SLES 15 installation is not assumed to meet that requirement. Provision a supported interpreter before installing the release; Python 3.12 or 3.11 is preferred where available, while Ubuntu 22's Python 3.10 meets the implemented recipe's runtime floor. The GPU qualification workflow searches the fixed order `python3.12`, `python3.11`, `python3.10`, then `python3`, and rejects the node before privileged or mutating host work if none resolves to Python 3.10 or newer.

An unsupported distribution/release, incomplete package inventory, missing safe package recipe, or ambiguous security/runtime state blocks applied convergence. Run `nvidia-converge support` for the packaged support summary and `plan` on the target image before scheduling a change.

## Production installation

Install a release wheel only after verifying the complete release checksum inventory and the GitHub build-provenance attestation for every release asset. Place it in a versioned, root-owned environment that unprivileged accounts and deployment runners cannot modify. Do not run applied commands with `sudo python3 -m nvidia_converge` from a mutable checkout, and do not elevate a checkout-local virtual environment.

For example, download the wheel, source distribution, tag-specific GPU evidence bundle, and `SHA256SUMS` from the same GitHub release. Set `SOURCE_SHA` to the trusted full commit ID for that tag. The source distribution contains the release evidence tooling, checked desired-state profiles, and hashed Linux x86_64 CPython 3.12 release lock. Run the offline verifier from an unpacked, attested source distribution or from a trusted checkout of the same commit in an unprivileged verification environment:

```bash
set -Eeuo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin

TAG=v0.1.0
VERSION="${TAG#v}"
WHEEL="nvidia_converge-${VERSION}-py3-none-any.whl"
SDIST="nvidia_converge-${VERSION}.tar.gz"
EVIDENCE="gpu-integration-evidence-${TAG}.zip"
: "${SOURCE_SHA:?export the trusted 40-character commit for $TAG}"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]

validate_nvidia_python_path() {
  local kind="$1" path="$2" canonical current metadata owner mode next
  [[ -n "$path" && "$path" == /* && "$path" != *$'\n'* ]] || return 1
  canonical="$(realpath -m -- "$path")"
  [[ "$canonical" == "$path" ]] || return 1
  if [[ -e "$path" && ! -L "$path" ]]; then
    case "$kind" in
      file) [[ -f "$path" ]] || return 1 ;;
      directory) [[ -d "$path" ]] || return 1 ;;
      component) [[ -f "$path" || -d "$path" ]] || return 1 ;;
    esac
    current="$path"
  elif [[ "$kind" == component && ! -L "$path" ]]; then
    current="$(dirname -- "$path")"
    while [[ ! -e "$current" && ! -L "$current" ]]; do
      next="$(dirname -- "$current")"
      [[ "$next" != "$current" ]] || return 1
      current="$next"
    done
  else
    return 1
  fi
  while :; do
    [[ -e "$current" && ! -L "$current" ]] || return 1
    metadata="$(stat -Lc '%u %a' -- "$current")"
    owner="${metadata%% *}"
    mode="${metadata##* }"
    [[ "$owner" == 0 ]] && (( (8#$mode & 8#022) == 0 )) || return 1
    [[ "$current" == / ]] && break
    current="$(dirname -- "$current")"
  done
}

select_nvidia_converge_python() {
  local candidate_path index path trusted
  local -a trust_paths
  for candidate_path in \
    /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 /usr/bin/python3
  do
    [ -x "$candidate_path" ] || continue
    candidate_path="$(readlink -f -- "$candidate_path")"
    validate_nvidia_python_path file "$candidate_path" || continue
    if "$candidate_path" -I -S -c \
      'import ensurepip, sys, venv; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
      mapfile -d '' -t trust_paths < <(
        "$candidate_path" -I -S -c \
          'import ensurepip, sys, sysconfig, venv; from pathlib import Path; modules=sorted({path for module in sys.modules.values() if isinstance((path := getattr(module, "__file__", None)), str)}); packages=sorted(str(path) for package in (ensurepip, venv) for root in (Path(package.__file__).parent,) for path in (root, *root.rglob("*"))); values=[sys.base_prefix, sys.exec_prefix, sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib"), *sys.path, *modules, *packages]; sys.stdout.buffer.write(b"".join(value.encode("utf-8") + b"\0" for value in values))'
      )
      (( ${#trust_paths[@]} >= 5 )) || continue
      trusted=true
      for index in 0 1 2 3; do
        validate_nvidia_python_path directory "${trust_paths[$index]}" \
          || trusted=false
      done
      for path in "${trust_paths[@]:4}"; do
        validate_nvidia_python_path component "$path" || trusted=false
      done
      if [[ "$trusted" == true ]]; then
        printf '%s\n' "$candidate_path"
        return 0
      fi
    fi
  done
  echo "no import-path-safe, root-owned /usr/bin Python >=3.10 with venv/ensurepip support found; provision Python 3.11 or 3.12 (or 3.10)" >&2
  return 1
}
PYTHON_BIN="$(select_nvidia_converge_python)" || exit 2
VERIFY_PYTHON=/usr/bin/python3.12
if ! "$VERIFY_PYTHON" -I -c \
  'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "the hashed release verifier lock requires CPython 3.12 on Linux x86_64" >&2
  exit 2
fi
VERIFY_ENV="$(mktemp -d)"
trap 'rm -rf -- "$VERIFY_ENV"' EXIT

gh release verify "$TAG" --repo zeroecco/nvidia-converge
sha256sum --check --strict -- SHA256SUMS
for asset in SHA256SUMS "$WHEEL" "$SDIST" "$EVIDENCE"; do
  gh attestation verify "$asset" \
    --repo zeroecco/nvidia-converge \
    --signer-workflow zeroecco/nvidia-converge/.github/workflows/production-release.yml \
    --source-ref refs/heads/main \
    --source-digest "$SOURCE_SHA"
done
"$VERIFY_PYTHON" -m venv "$VERIFY_ENV"
"$VERIFY_ENV/bin/python" -m pip install \
  --require-hashes --no-deps \
  --requirement requirements/release.lock
"$VERIFY_ENV/bin/python" -m scripts.verify_release_evidence_bundle \
  --bundle "$EVIDENCE" \
  --release "$TAG" \
  --repository zeroecco/nvidia-converge \
  --qualification-wheel "$WHEEL"
WHEEL_SHA256="$(sha256sum "$WHEEL" | awk '{print $1}')"

validate_nvidia_python_path file "$PYTHON_BIN" || {
  echo "selected Python changed before privileged installation" >&2
  exit 2
}
sudo install -d -o root -g root -m 0700 /var/lib/nvidia-converge/packages
STAGED_WHEEL="/var/lib/nvidia-converge/packages/$WHEEL"
sudo install -o root -g root -m 0400 -- "$WHEEL" "$STAGED_WHEEL"
test "$(sudo sha256sum "$STAGED_WHEEL" | awk '{print $1}')" = "$WHEEL_SHA256"

RELEASE_DIR="/opt/nvidia-converge/releases/$VERSION"
for directory in /opt/nvidia-converge /opt/nvidia-converge/releases; do
  if sudo test -e "$directory" || sudo test -L "$directory"; then
    validate_nvidia_python_path directory "$directory" || {
      echo "unsafe production release ancestor: $directory" >&2
      exit 2
    }
  else
    sudo install -d -o root -g root -m 0755 -- "$directory"
    validate_nvidia_python_path directory "$directory"
  fi
done
validate_nvidia_python_path component "$RELEASE_DIR"
if sudo test -e "$RELEASE_DIR" || sudo test -L "$RELEASE_DIR"; then
  echo "release directory already exists: $RELEASE_DIR" >&2
  exit 2
fi
sudo "$PYTHON_BIN" -I -S -m venv "$RELEASE_DIR"
sudo "$RELEASE_DIR/bin/python" -I -m pip --isolated install \
  --no-index --no-deps "$STAGED_WHEEL"
sudo "$RELEASE_DIR/bin/python" -I -m compileall -q "$RELEASE_DIR/lib"
sudo chown -R root:root "$RELEASE_DIR"
sudo chmod -R go-w "$RELEASE_DIR"
validate_nvidia_python_path directory "$RELEASE_DIR"
validate_nvidia_python_path directory "$RELEASE_DIR/bin"
test -L "$RELEASE_DIR/bin/python"
test "$(readlink -f -- "$RELEASE_DIR/bin/python")" = "$PYTHON_BIN"
```

Recheck the staged wheel's SHA256 before installation whenever artifact staging and installation are separate deployment steps. A production image or configuration-management system may enforce stronger read-only or immutability controls around the release directory.

### Upgrade and uninstall lifecycle

Install upgrades into a new, verified versioned virtual environment beside the current release; never run `pip install --upgrade` or otherwise modify a production environment in place. Validate the new environment and its exact wheel first, then update the operator or configuration-management command to name that version explicitly. Keep the prior environment intact so recovery never depends on reconstructing executable code after an incident.

Retain the prior release environment, its exact wheel and release-verification artifacts, the desired-state input, and every original report, journal, applicable snapshot, paired package-payload bundle, and recovery report throughout every unresolved-operation and rollback window. The current rollback snapshot schema is exactly 2.6; a binary rejects other schema versions, so do not assume that a newer release can consume an older snapshot. Use the verified release that created the recovery material unless cross-version compatibility has been explicitly qualified, and never migrate recovery evidence while an operation is unresolved.

Before host mutation, `nvidia-converge` resolves the forward transaction and stages payloads for the exact baseline and forward package identities (name, architecture, epoch when applicable, and version) in a private, digest-addressed directory bound to the snapshot path. APT acquisition must authenticate repository metadata and payload hashes; RPM payloads must also pass the host's trusted signature check. The manifest records the SHA-256 of every retained file. This proves authenticated exact-identity restoration from the retained bytes; it does not claim that a freshly acquired archive is byte-for-byte identical to the archive that originally installed the package, because dpkg does not generally retain that provenance. Staging is size-bounded, holds an allocated emergency free-space reserve, and fails cleanly if every requested identity cannot be retained one-to-one. Applied forward and rollback package commands consume only those validated local artifacts with remote repositories or downloads disabled. Keep the snapshot and its paired payload directory together and private until the retention window closes; after staging, a reachable mutable repository is not rollback authority.

A release is safe to retire only after every applied operation either has a complete terminal report or has been terminally marked recovered by a complete, fully verified recovery report; no journal represents unresolved mutation; any required rollback has passed full verification; the rollback retention window is formally closed; and the replacement release has been independently verified for the node's desired state. Uninstall only a release that meets those criteria. Remove its versioned environment last; do not automatically delete `/var/lib/nvidia-converge`, exact-identity package recovery material, or other retained evidence as part of uninstall.

## Configuration and operation

Store the desired-state file in a root-controlled location and validate it before use. Applied commands require an explicit `--desired` file and enforce that the same opened file descriptor is root-owned and not group/world-writable; a path in a runner- or user-writable location is rejected even when the contents validate. Invoke the versioned virtual-environment interpreter with `-I -m nvidia_converge`; do not elevate the setuptools-generated console script because that script cannot force Python isolated mode before imports begin. A real applied invocation now rejects non-isolated Python before desired-state loading, report reservation, audit, or mutation. Isolated mode excludes several ambient Python inputs but does not prove interpreter or import-tree ownership, so the root-owned environment and full path validation in [Production installation](#production-installation) remain mandatory.

```bash
NVIDIA_CONVERGE=(
  /opt/nvidia-converge/releases/0.1.0/bin/python -I -m nvidia_converge
)
DESIRED=/etc/nvidia-converge/desired.yaml

sudo install -d -o root -g root -m 0755 /etc/nvidia-converge
sudo install -o root -g root -m 0644 desired.yaml "$DESIRED"

"${NVIDIA_CONVERGE[@]}" validate \
  --desired "$DESIRED" --out validation.json
"${NVIDIA_CONVERGE[@]}" doctor --desired "$DESIRED"
"${NVIDIA_CONVERGE[@]}" plan \
  --desired "$DESIRED" --out plan.json
```

If the plan reports `unsupported.package-policy-staging`, stage the observed vendor pin, module stream, or lock first, then re-audit and re-plan. This two-pass boundary ensures repository resolution and the applied transaction use the same active policy:

```bash
sudo "${NVIDIA_CONVERGE[@]}" lock --desired "$DESIRED" --apply \
  --allow-disruption
"${NVIDIA_CONVERGE[@]}" doctor --desired "$DESIRED"
"${NVIDIA_CONVERGE[@]}" plan \
  --desired "$DESIRED" --out plan-after-lock.json
```

Commands print a compact human summary by default. `--out PATH` writes the full machine-readable report atomically, while `--json` prints it to stdout. Applied commands reserve their evidence before host work begins: the report must be a new file directly under the root-owned `/var/lib/nvidia-converge/reports` directory, and a durable `<report>.journal.jsonl` command journal is retained beside it. The reservation also durably creates a small internal marker under the private `.active-journals` directory before mutation can begin. Recovery scans this bounded marker set instead of repeatedly loading the complete retained history; the tool retires a marker only after the operation has no possible host mutation or its matching terminal report or verified recovery report is durable. Marker retirement never deletes the report or journal. If `--out` is omitted, the tool chooses a unique path there automatically. Existing applied reports are never overwritten.

Applied operations also enforce a state-storage admission budget before every mutating command. Each distinct filesystem backing `/var/lib/nvidia-converge/reports` or `/var/lib/nvidia-converge/snapshots` must still have at least 64 MiB and 32 free inodes after a private 4 MiB report-filesystem emergency reserve is allocated. A journal append that encounters `ENOSPC` or quota exhaustion durably rolls back any partial line, releases that reserve, and retries once. The reserve is normally removed after the operation and stale reserves are reclaimed only while holding the global operation lock. This protects a small terminal or conservative recovery record; it is not a capacity guarantee for an arbitrarily large final report, package cache, another writer, or a failing filesystem. Monitor and retain substantially more space than the admission minimum.

```bash
sudo install -d -o root -g root -m 0700 /var/lib/nvidia-converge/reports
REPORT_DIR=/var/lib/nvidia-converge/reports
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"

sudo "${NVIDIA_CONVERGE[@]}" lock --desired "$DESIRED" --apply \
  --allow-disruption \
  --out "$REPORT_DIR/lock-$RUN_ID.json"
sudo "${NVIDIA_CONVERGE[@]}" snapshot --desired "$DESIRED" --apply \
  --out "$REPORT_DIR/snapshot-$RUN_ID.json"
sudo "${NVIDIA_CONVERGE[@]}" install --desired "$DESIRED" --apply --allow-disruption \
  --out "$REPORT_DIR/install-$RUN_ID.json"
sudo "${NVIDIA_CONVERGE[@]}" verify --desired "$DESIRED" --apply \
  --allow-disruption \
  --out "$REPORT_DIR/verify-$RUN_ID.json"
```

`install --apply` takes its own pre-mutation snapshot. If the plan contains disruptive actions, `--allow-disruption` is a required maintenance-window attestation. The command probes active GPU compute processes and fails closed when their state cannot be determined. If workloads are intentionally still active, both `--allow-disruption` and the stronger `--allow-active-workloads` acknowledgement are required:

```bash
sudo "${NVIDIA_CONVERGE[@]}" install --desired "$DESIRED" --apply \
  --allow-disruption --allow-active-workloads \
  --out "$REPORT_DIR/install-active-$RUN_ID.json"
```

Prefer draining the node instead. `--allow-active-workloads` acknowledges risk; it does not make module, service, package, or MIG transitions safe.

If an install mutation fails, the command re-audits the host, preflights exact rollback from its private pre-install snapshot, applies that rollback automatically, and verifies the modeled baseline invariants. The original install still reports failure even when compensation succeeds. Trusted GPU services are restored only after verified compensation; if compensation cannot be proven, they are re-quiesced or left stopped.

Commands that may mutate the host or persist rollback state (`install`, `verify`, `lock`, `snapshot`, and `rollback`) are dry-run unless `--apply` is supplied, and applied execution requires root. Without `--apply`, `verify` performs non-mutating checks but skips module loading and the Docker GPU execution check. Applied host operations are serialized with a root-owned operation lock. On DNF hosts, reserve an exclusive package-management maintenance window: the module apply helper also holds DNF4's native RPMDB process lock across its authoritative inventory, persistence, and reopen verification, but upstream DNF4 module-only commands do not acquire that lock and therefore must not run concurrently.

## Rollback snapshots

Applied snapshots use schema version 2.6 and are written as private files directly under `/var/lib/nvidia-converge/snapshots`. Each snapshot has a private sibling payload directory containing the authenticated, digest-addressed package files needed by its exact-identity local forward and rollback transactions. A snapshot is integrity-protected and bound to its original path, payload manifest, host identity, exact GPU UUID inventory, OS and version, architecture, package backend, and effective owner. Apply rollback from the original snapshot path on the same host; an exported artifact copy is evidence, not an applicable rollback input.

The snapshot records the scoped NVIDIA/kernel/Docker package inventory (including package architecture and RPM epoch where available), directly introduced packages, the running kernel, the exact loaded-module graph, loaded and on-disk NVIDIA module version/flavor/signature state, stable MIG mode and GI/CI geometry, Docker and Fabric Manager active/enabled state, and these managed policy/configuration files when relevant:

- `/etc/docker/daemon.json`
- `/etc/dnf/modules.d/nvidia-driver.module`
- `/var/lib/dnf/modulefailsafe/nvidia-driver:<stream>:<architecture>.yaml`
- `/etc/zypp/locks`

Rollback derives package-manager commands locally instead of trusting executable commands from the snapshot. Before convergence mutates the host, it authenticates and retains one payload for every recorded package identity, binds those files to the snapshot manifest, and proves the same offline solver transaction that apply will execute. It restores modeled package identities, files, modules, qualified MIG geometry, and service state, then audits that state. Exact automatic MIG restoration is intentionally limited to an empty baseline or one GPU instance with one compute instance consuming the entire GPU instance; other geometries fail closed before mutation. A reboot or external recovery may still be required.

Use the unique original path printed by `install --apply` or recorded in the install report:

```bash
SNAPSHOT_PATH=/var/lib/nvidia-converge/snapshots/<snapshot>.json
sudo "${NVIDIA_CONVERGE[@]}" rollback --desired "$DESIRED" --snapshot "$SNAPSHOT_PATH" --apply \
  --allow-disruption \
  --out "$REPORT_DIR/rollback-$RUN_ID.json"
```

Applied rollback requires the same maintenance-window attestation and active-workload safety check as disruptive install. Use `--allow-active-workloads` only for an explicitly accepted emergency rollback risk.

### Interrupted-operation recovery

Every applied command is crash-journaled before its first host mutation. At the start of the next applied command, `nvidia-converge` scans its bounded private active-journal index while holding the operation lock. Existing installations without the index receive a one-time, bounded, idempotent bootstrap from the preserved top-level journals; its completion sentinel is written only after every required marker is durable. A crash before that sentinel safely repeats the bootstrap. If the legacy directory exceeds 16,384 entries, 4,096 journals, or 64 MiB of journal data, bootstrap refuses rather than loading an unbounded history. Take the tool and node out of service and archive only confirmed terminal report/journal pairs under the same private evidence controls before retrying; never move, edit, or delete an unresolved or recovery-bound artifact. New applied work is limited to 1,024 simultaneously active markers.

If an indexed journal can represent an interrupted mutation, all new applied work is refused except a rollback using the one exact snapshot path/hash/creator/host binding recorded before that mutation. A torn final append is repaired by durably removing only its incomplete suffix after the complete prefix validates; malformed or conflicting durable records fail closed.

Keep the node drained while this gate is active. One or more launchers may be stopped and persistently masked. Do not manually start, enable, unmask, delete, rename, or edit the affected service units, report, journal, or snapshot. Preserve the original bytes and paths, then run the exact rollback command printed by the refusal, for example:

```bash
SNAPSHOT_PATH=/var/lib/nvidia-converge/snapshots/<exact-bound-snapshot>.json
sudo "${NVIDIA_CONVERGE[@]}" rollback --desired "$DESIRED" \
  --snapshot "$SNAPSHOT_PATH" --apply --allow-disruption \
  --out "$REPORT_DIR/recovery-$RUN_ID.json"
```

Add `--allow-active-workloads` only when the emergency disruption risk has been explicitly accepted. If rollback reports a reboot-pending MIG transition, leave every launcher masked, reboot the drained node, and rerun the same exact snapshot rollback. Earlier journals are marked `operation-recovered` only after full rollback verification succeeds. Confirm the recovery report is complete and all modeled rollback checks pass before returning workloads, then retain the original report, journal, snapshot, and recovery report together under the same private evidence policy.

There is no force/bypass flag for a missing, modified, ambiguous, wrong-host, or malformed recovery authority. Restore the exact protected recovery material through an approved out-of-band procedure, or reimage/recover the node from an independently trusted point.

## Development

Source checkouts and editable environments are for unprivileged development, dry-run inspection, and automated tests only:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[test]"
python3 -m nvidia_converge --version
python3 -m nvidia_converge validate --desired examples/compute-580-open.yaml
python3 -m nvidia_converge doctor
python3 -m nvidia_converge plan --out report.json
python3 -m compileall -q nvidia_converge scripts tests
python3 tests/run_tests.py
python3 -m ruff check .
python3 -m mypy nvidia_converge scripts/canonicalize_sdist.py scripts/check_repository_controls.py scripts/check_release_evidence.py scripts/export_integration_attestation.py scripts/bundle_release_evidence.py scripts/verify_release_evidence_bundle.py
python3 -m pytest -q
python3 -m build
```

The self-hosted GPU workflow deliberately runs a fixed checked-out commit on disposable, access-controlled test nodes. It qualifies a controlled missing-package convergence, cleanup-guarded live doctor faults, profile-specific capability observations, and exact rollback. It uploads only a sanitized attestation bundle; raw reports, host diagnostics, command output, and the applicable snapshot remain private on the runner. That integration-only mechanism is not a production deployment pattern.

## Releases

Production releases are created only by `.github/workflows/production-release.yml`, a `repository_dispatch` workflow that GitHub loads from the current default branch. Its first hosted job requires the run ref and run SHA to equal the live `main` head, derives the release tag from the trusted `pyproject.toml` version, and requires the matching tracked `integrations/results.<tag>.json`. It then executes the strict `repository-controls` -> `gates` -> `artifacts` -> `publish` chain. The gates and fresh artifact runner both require an exact `origin/main == GITHUB_SHA` checkout. The wheel is reproducibly built twice from the GPU-tested commit; the source distribution is independently built and canonicalized twice from the release commit, which may differ only by the matching evidence file. The evidence bundler validates the source Actions artifacts and binds their canonical sanitized copies to the exact qualification-wheel name and SHA-256. Canonicalization strips ZIP metadata and other unneeded channels rather than retaining the original archive bytes. The handoff contains exactly the wheel, source distribution, tag-specific evidence ZIP, and `SHA256SUMS`.

The protected `release` job is the sole visible automated release writer under the required external App-custody controls. Organization owners, Release Creator App managers, and holders of an unrevoked App private key remain explicitly trusted control-plane roots because GitHub cannot deny or completely inventory their release authority. After approval, the job proves the source is still live `main`, rechecks repository controls, owner-enforced immutable releases, and the visible writer inventory, then mints a short-lived installation token for a dedicated Release Creator GitHub App. The workflow constrains that token to this repository and verifies its returned repository inventory before it creates the previously nonexistent tag at the gated commit. It attests all four assets, creates one new draft through the REST API, captures its numeric release ID, uploads each asset to that exact ID, verifies server metadata and downloaded bytes, and repeats the live-main, tag, control, and byte checks immediately before publishing that same ID. It refuses to reuse an existing tag, draft, or release.

Administrators must configure these controls before dispatch:

- Every effective ruleset contributing required protection to `main` is active, applies to that exact branch, and has an empty bypass inventory. In particular, do not make the Release Creator App a main-branch bypass actor; its `Contents: write` authority is limited to creating the protected release tag through the narrowly scoped tag-creation ruleset below.
- Both `gpu-qualification` and `release` environments require non-self approval and exactly one custom deployment branch policy: the `main` branch.
- One active `refs/tags/v*` creation ruleset has exactly the Release Creator GitHub App as an `always` bypass actor. A separate active `v*` update/deletion ruleset has no bypass actors.
- [GitHub release immutability](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes) is enabled and enforced by the repository owner.
- The protected `release` environment contains `RELEASE_CREATOR_APP_ID` as a variable, `RELEASE_CREATOR_PRIVATE_KEY` as a secret, and `REPOSITORY_AUDIT_TOKEN` as a secret. The App is organization-owned, its installation uses selected-repository access, and `Contents: write` is its only write permission; each release mints and verifies a token narrowed to this repository alone. Before dispatch, an organization owner must verify out of band that [only organization owners manage the App](https://docs.github.com/en/organizations/managing-programmatic-access-to-your-organization/adding-and-removing-github-app-managers-in-your-organization), [GitHub App installation is restricted to owners](https://docs.github.com/en/organizations/managing-programmatic-access-to-your-organization/limiting-oauth-app-and-github-app-access-requests-and-installations), the environment secret contains the sole active private key, and every superseded or separately held key is revoked. GitHub's repository APIs do not expose a complete App-manager, private-key, or active-token inventory, so checker success does not prove this custody condition. The audit credential belongs to an organization owner and needs repository `Administration: read` and `Actions: read` plus organization `Members: read` and `Administration: read`. Those permissions expose ruleset bypasses, owners, effective collaborators, teams, repository and organization invitations, deploy keys, default Actions authority, nonterminal workflow runs, organization defaults, and organization App installations. The release check rejects non-owner human writers or custom roles, write teams and deploy keys, pending invitations, organization-wide write defaults, every alternate App with any write or administration permission, a non-read-only Actions default, and every queued or running Actions workflow except the exact current release run and attempt.
- Legacy `release.yml` and `gpu-integration.yml` workflow IDs are disabled through the Actions API after this migration is merged. Deleting the files from `main` alone is insufficient because an old tag can still contain an old privileged workflow.

Consumers verify the immutable release, its exact `SHA256SUMS`, and every GitHub artifact attestation while constraining the repository, signer workflow `.github/workflows/production-release.yml`, source ref `refs/heads/main`, and trusted release commit. They then run `scripts.verify_release_evidence_bundle` offline as shown in [Production installation](#production-installation). The source Actions artifacts and build handoff expire after 30 days; the durable release bundle inherits its GitHub API provenance from the release-time online checks. Offline verification proves retained structure, digests, manifest semantics, and sanitized evidence, not a fresh query of expired Actions artifacts.

Never create or push a production tag manually. First set the target version in `pyproject.toml`, merge it to protected `main`, and qualify that fixed commit. Then commit only the passing `integrations/results.<tag>.json` described in [the integration-testing guide](docs/integration-testing.md), merge that evidence-only release commit to protected `main`, verify that it is still the live head, and dispatch the trusted workflow:

```bash
REPOSITORY=OWNER/REPOSITORY
gh api --method POST "repos/$REPOSITORY/dispatches" \
  -f event_type=production-release
```

## Reports and schemas

Reports include the command and mode, tool and host identity, operation ID, timing, outcome and exit code, reboot/incomplete status, audit findings, proposed or applied actions, verification results, rollback metadata, and an SBOM-style package/module inventory. Applied reports are pre-reserved with `incomplete: true`; successful finalization atomically replaces that provisional record and appends an `operation-completed` journal event. An interrupted operation therefore leaves a conservative report and the last durably recorded command result.

JSON Schemas are available in `schemas/` and through the CLI:

```bash
nvidia-converge schema desired
nvidia-converge schema integration-results
nvidia-converge schema report
nvidia-converge schema validation
```
