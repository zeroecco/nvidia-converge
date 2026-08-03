# Integration Testing

`nvidia-converge` has unit, schema, packaging, and CLI smoke coverage in CI across CPython 3.10, 3.11, 3.12, 3.13, and 3.14. Host mutation is release-qualified separately on disposable GPU nodes. No applied path should be described as production-proven for a release until the promotion evidence below passes.

## Trusted GPU workflow

The production workflow `.github/workflows/production-gpu-qualification.yml` is `repository_dispatch`-only. GitHub loads it from the default branch, and its first hosted step requires the event action `production-gpu-qualification`, `GITHUB_REF == refs/heads/main`, and `GITHUB_SHA` equal to the current live `main` head returned by GitHub. It then runs an authenticated, read-only repository-control check before a qualification build or self-hosted runner is allocated. The checker rejects personal-account ownership, missing or ineffective `main` rules, and a missing, ruleless, self-approvable, or non-`main` `gpu-qualification` environment. A hosted runner builds one qualification wheel from the bound commit with the hashed `requirements/build.lock` toolchain, passes its exact name and SHA256 through job outputs, and retains the wheel plus an exact checksum manifest. Only then does the deterministic six-job GPU matrix begin. Every GPU runner must be routed through the exact `nvidia-converge-gpu` runner group and carry the core labels `self-hosted`, `nvidia-gpu`, and `disposable` in addition to its matrix-specific labels.

| Matrix job | Additional labels | Desired state | Coverage |
| --- | --- | --- | --- |
| `ubuntu-24.04` | `ubuntu-24.04` | `examples/compute-580-open.yaml` | Ubuntu LTS and `apt-get` |
| `rhel-9` | `rhel-9` | `examples/compute-580-open.yaml` | RHEL family and `dnf` |
| `sles-15` | `sles-15` | `examples/compute-580-open.yaml` | SUSE family and `zypper` |
| `secure-boot` | `ubuntu-24.04`, `secure-boot` | `examples/compute-580-open.yaml` | Secure Boot |
| `fabric-manager` | `ubuntu-24.04`, `fabric-manager` | `examples/compute-580-open-fabric-manager.yaml` | Fabric Manager required and active |
| `mig-toggle` | `ubuntu-24.04`, `mig-capable` | `examples/compute-580-open-mig.yaml` | MIG transition and restoration |

The base and MIG profiles set `fabric_manager: false`; only the Fabric Manager profile sets it to `true`. Capability labels describe independently observed host properties and may overlap on one runner. Each designated matrix job must carry its required label, and every capability label it carries must agree with retained host observations. The release gate compares the complete label set from GitHub's Jobs API with the manifest and derives capability claims again from retained evidence.

Production qualification is always applied: the workflow owns `QUALIFICATION_APPLY=true` and accepts neither a caller-selected ref, an `apply` input, nor `client_payload` control data. Desired-state paths are fixed in the checked-in matrix. Each matrix job:

1. Rejects a quarantined runner and proves every claimed capability independently from live audit fields.
2. Requires every checked-in profile to use `cuda_compat: none`; qualification neither deploys nor claims CUDA forward-compatibility libraries.
3. Applies the backend package policy first and retains the lock command's private pre-policy rollback snapshot.
4. Persists a second, canonical post-policy snapshot, re-plans, and proves the only desired-state drift is an absent `nvidia-container-toolkit` executable/package and Docker NVIDIA runtime configuration.
5. Requires successful, journaled package installation plus `nvidia-ctk` configuration and Docker restart, then proves the toolkit package closure, runtime, a digest-pinned and source/hash-bound container CUDA Driver API probe, and healthy-host state.
6. Exercises real, controlled doctor faults while holding `/run/nvidia-converge/operation.lock`, restores each fault, and records a same-host healthy doctor report after every restoration.
7. Runs a separate `always()` cleanup from persisted fault prestate. Cleanup failure leaves `/run/nvidia-converge/fault-cleanup-required`, fails promotion, and quarantines the runner.
8. Rolls convergence back to the canonical post-policy snapshot, then independently rolls package policy back to the lock-owned pre-policy snapshot. Both applied rollbacks are always attempted after their corresponding snapshot exists and must pass independent state verification.
9. Captures private post-state and uploads only a privacy-minimized attestation bundle with a unique SHA256-addressed artifact retained for 30 days. A dependent GitHub-hosted job installs the hashed validation lock, validates every retained report schema, and requires all six GPU jobs to have passed.

The workflow invokes a fixed checked-out main-branch source commit with privilege only on disposable, access-controlled integration runners. Production nodes must use a verified release wheel in a root-owned, non-writable deployment as described in the [README](../README.md#production-installation).

For every run, the GPU job downloads the hosted artifact, requires its exact two-file inventory, compares its checksum manifest with the immutable hosted job outputs, and installs the wheel without an index into an unprivileged tooling environment. For an applied run, it also hashes the checked-in desired input, copies both inputs into a unique root-owned directory under `/opt/nvidia-converge/qualification`, and installs the same wheel without an index into a root-owned, non-writable virtual environment. Privileged CLI and controlled-fault calls use that isolated interpreter with `-I`; they never import from the runner-writable checkout or its tooling environment. GPU runners perform no network Python dependency resolution or local package build. The qualification directory is removed after rollback and fault cleanup.

## Required GitHub controls

Promotion depends on repository configuration that cannot be encoded entirely in workflow YAML:

- Protect `main` so every effective ruleset contributing its required protections is active, applies to that exact branch, and has an empty bypass inventory. Together they must require at least one pull-request approval, dismiss stale approvals, require approval of the last push by someone else, require resolved review threads, bind every `Python 3.10` through `Python 3.14` status check to the GitHub Actions integration, test against the latest mainline, and block force pushes and deletions. Do not make the Release Creator App a main-branch bypass actor. The workflow rejects non-main dispatches, and the release evidence verifier independently requires GitHub's run API to report `head_branch: main`.
- Protect `v*` tags with two separate active repository rulesets. The creation ruleset must have exactly one bypass actor: the dedicated Release Creator GitHub App in `always` mode. The update/deletion lock must have no bypass actors. Do not grant the creator App update or deletion authority. The release workflow creates the previously nonexistent tag only after all gates and protected-environment approval, binds it to the gated commit, and rechecks it before draft creation, immediately before publication, and after publication.
- Configure both `gpu-qualification` and `release` as protected environments with a required reviewer who is not the workflow initiator. Do not permit self-approval or administrator bypass. Each environment must use custom deployment branch policies with exactly one policy: `main` with type `branch`.
- Transfer the repository to an organization before production qualification, update the canonical repository identity in evidence and consumer instructions, and create the exact `nvidia-converge-gpu` runner group. Restrict it to this repository and to `OWNER/REPOSITORY/.github/workflows/production-gpu-qualification.yml@refs/heads/main`; do not attach production nodes or general-purpose organization runners to its labels. GitHub does not provide runner groups for the repository's current personal-account ownership, and the workflow intentionally cannot route GPU jobs there.
- Enable immutable releases and require owner enforcement before release dispatch. Treat organization owners, Release Creator App managers, and holders of an unrevoked App private key as trusted control-plane roots; GitHub does not provide a release-specific deny role that can constrain them or a repository API that completely inventories App managers, private keys, and active tokens. Create a dedicated organization-owned GitHub App installation with selected-repository access and `Contents: write` as its only write permission; store its numeric App ID as the protected `release` environment variable `RELEASE_CREATOR_APP_ID` and its sole active private key as the environment secret `RELEASE_CREATOR_PRIVATE_KEY`. Restrict App installation to organization owners. Before every dispatch, an organization owner must verify out of band that only organization owners manage this App and that every superseded or separately held private key is revoked. Store a short-lived fine-grained credential belonging to an organization owner with repository `Administration: read` and `Actions: read` plus organization `Members: read` and `Administration: read` as `REPOSITORY_AUDIT_TOKEN`. The publish job uses the audit credential only for fail-closed reads, mints a short-lived App installation token explicitly narrowed to this repository, verifies that token's exact one-repository inventory, and uses it only for tag/release writes.
- After this migration reaches `main`, resolve the exact legacy `release.yml` and `gpu-integration.yml` workflow IDs from the read-only workflow inventory, [disable each ID through the Actions API](https://docs.github.com/en/rest/actions/workflows?apiVersion=2026-03-10#disable-a-workflow) with an `Actions: write` maintenance credential, and read each ID back to verify `state == disabled_manually`. Resolve before writing; never operate on an unverified ID or name match. Removing the files from the default branch does not revoke an old tagged workflow's authority.

The hosted checker uses documented GitHub read APIs with the workflow's read-only token and never runs on a GPU node. It proves organization ownership, effective active branch-rule details and applicability, exact custom-`main` environment policies, reviewer structure and self-review prevention, tag-rule structure, and an exact active workflow inventory consisting only of CI plus the two production workflows. If GitHub hides bypass inventories from that token, the hosted result explicitly leaves ruleset bypass actors unverified. If GitHub still lists either retired privileged workflow, its state must be `disabled_manually`. In the protected publish job, `REPOSITORY_AUDIT_TOKEN` additionally proves that every effective `main` ruleset has no bypass actors, owner-enforced immutable releases, the exact Release Creator App/no-bypass tag topology, read-only or no organization default access, no pending repository or organization invitation, and a visible writer inventory with no non-owner writer or custom role, write team, write deploy key, or alternate App with any write or administration permission. It also requires read-only default Actions permissions and proves that the exact current release run and attempt is the sole queued or running workflow, excluding still-live tokens from retired or older workflows. The creator token is explicitly narrowed to this repository and must report exactly this repository through the installation-token API. These are fail-closed point-in-time checks around a GitHub publish API that has no compare-and-swap precondition; durable least privilege, a change freeze during publication, and the trusted organization-owner/App-key-custody boundary remain essential. The APIs used by the checker do not expose a complete App-manager, private-key, or active-token inventory. The environment API does not expose administrator bypass, and runner-group repository/workflow scope requires organization-level runner administration access, so administrators must verify those controls separately; checker success never claims they were observed. Naming an environment in workflow YAML is not provisioning: GitHub can create a referenced missing environment without protection, which is why the earlier hosted job queries it and fails on 404 or an empty reviewer rule before sensitive work.

Run the same visible checks before qualification or release dispatch:

```bash
python scripts/check_repository_controls.py \
  --repository OWNER/REPOSITORY \
  --scope gpu

GITHUB_TOKEN=<administration-and-actions-read-token-with-bypass-visibility> \
python scripts/check_repository_controls.py \
  --repository OWNER/REPOSITORY \
  --scope release \
  --require-immutable-releases \
  --release-creator-app-id <positive-app-id> \
  --require-release-writer-isolation
```

Dispatch qualification only after the bound commit is the live protected `main` head and the runner group is restricted to the exact workflow identity above. Do not include a client payload:

```bash
REPOSITORY=OWNER/REPOSITORY
gh api --method POST "repos/$REPOSITORY/dispatches" \
  -f event_type=production-gpu-qualification
```

## Node prerequisites

Each runner must be a disposable or reimageable GPU node with out-of-band recovery and no production workloads. Before dispatch:

- Configure the signed NVIDIA driver/CUDA, NVIDIA Container Toolkit, and Docker CE repositories. At snapshot staging time, authenticated repository metadata and payloads must expose every required forward and baseline package identity; applied transactions use only the retained local files afterward.
- Ensure runner labels describe observed host capabilities rather than intended capabilities.
- Provision matrix targets that collectively cover Secure Boot, Fabric Manager, and MIG capability. A target may provide more than one capability when its labels and retained observations report that overlap truthfully.
- Drain all GPU compute work. The standard workflow supplies `--allow-disruption` but intentionally does not supply `--allow-active-workloads`.
- Prepare the controlled base fixture with the desired driver, Docker, module state, and services healthy, stable MIG disabled, and the `nvidia-container-toolkit` package/runtime tools absent. Recycled already-converged nodes are rejected. Toolkit dependency packages already present in the baseline are preserved; every closure member introduced by convergence is tracked for removal.
- Ensure the running-kernel `/lib/modules/$(uname -r)/build` entry is a single root-owned symlink to the distribution's root-owned, non-group/world-writable header tree under `/usr/src`. Its prepared-tree and release markers must bind exactly to `uname -r`, and the matching `linux-headers-*`, `kernel-devel`, or flavor-specific `kernel-*-devel` package must own the release marker. The loaded NVIDIA module graph may contain only the recognized `nvidia`, `nvidia_modeset`, `nvidia_drm`, `nvidia_uvm`, `nvidia_peermem`, and `nvidia_fs` modules needed by the host.
- Install `flock`, `unshare`, `mount`, `modprobe`, and systemd tooling used by the cleanup-guarded fault harness.
- Run GitHub Actions Runner 2.327.1 or newer. The workflow's commit-pinned JavaScript actions use the Node 24 runtime and fail before qualification on older self-hosted runners.
- Provision a root-owned, non-group/world-writable Python 3.10 or newer with working `venv`/`ensurepip` support. Do not rely on the base distribution's unversioned `python3` on RHEL 8/9 or SLES 15. The workflow pins a system-only tool `PATH` and deterministically tries `python3.12`, `python3.11`, `python3.10`, then `python3`; before any privileged use it requires a canonical nonsymlink executable and root-owned, non-group/world-writable ancestry, then applies the same checks to `sys.base_prefix`, `sys.exec_prefix`, the standard-library roots, every absolute `sys.path` component, every loaded `venv`/`ensurepip` module file, and their complete package-data trees (including bundled bootstrap wheels) observed under isolated no-site mode. Empty, relative, symlink-resolved, writable, or otherwise unsafe import paths fail closed before host mutation. The executable ancestry is rechecked immediately before first privilege, and root virtual-environment creation uses `-I -S` so system site customization cannot run as root.
- Ensure `/var/lib/nvidia-converge/snapshots` can retain the private, host-bound rollback snapshot until cleanup completes.

The minimum release matrix is exactly Ubuntu 24.04 with `apt-get`, one observed RHEL-family release at version 9 with `dnf`, and one observed SLES/openSUSE Leap release at version 15 with `zypper`, plus observed Secure Boot, Fabric Manager, and safe MIG-toggle capability. The manifest's `qualified_platforms` array carries the exact observed OS ID, OS version, and package-manager tuples; recipe-path coverage does not promote other recognized releases. A yum-only RHEL host does not satisfy the current convergence recipe.

The present applied fixture qualifies controlled repair of a healthy preselected driver stack whose deliberate drift is limited to the missing NVIDIA Container Toolkit package/runtime configuration. It does not qualify first-driver/bootstrap installation, driver or kernel-header package installation, an actual reboot boundary, install-failure compensation, or a second-pass idempotence claim. Those implemented paths remain unqualified until distinct retained scenarios exercise them.

## Required scenarios

Every run recorded as an apply run must independently pass every applicable scenario below. A non-apply run may not claim a mutating scenario as passed. `doctor_fabric_manager_inactive` is required only for the FM-enabled profile; every other profile must explicitly record it as `not_applicable` with no report.

| Evidence name | Required proof |
| --- | --- |
| `doctor` | A healthy desired-state host has no blocking findings. |
| `doctor_missing_headers` | Doctor detects missing running-kernel headers and gives actionable remediation. |
| `doctor_module_unloaded` | Doctor detects an unloaded NVIDIA module. |
| `doctor_driver_mismatch` | Doctor detects that the live module does not match a controlled desired branch that differs only in `driver`. |
| `doctor_runtime_missing` | Doctor detects a missing or unusable Docker NVIDIA runtime. |
| `doctor_fabric_manager_inactive` | Doctor detects inactive Fabric Manager when it is required. |
| `plan` | The report shows all required package, kernel, module, service, runtime, lock, MIG, and verification actions without mutation. |
| `snapshot` | A private schema-v2.6 canonical snapshot and its authenticated digest-addressed package-payload bundle, including stable MIG GI/CI geometry, are persisted after package-policy staging and before convergence mutation. |
| `install_apply` | Applied install converges the clean test image to the desired state. |
| `verify_apply` | Applied verification proves module load, `nvidia-smi`, isolated NVML import, the digest-pinned Docker CUDA Driver API probe, Fabric Manager version/state, and desired MIG mode, geometry, and device binding. |
| `lock_apply` | The package backend's branch policy is applied. |
| `rollback_apply` | Applied rollback completes using the original host-bound snapshot. |
| `rollback_state_verify` | Post-rollback audit matches recorded package versions/identities, running kernel, module version/load state, exact qualified MIG GI/CI geometry, Docker/Fabric Manager active and enabled state, and managed files. |
| `policy_rollback_apply` | A second applied rollback uses the lock command's private pre-policy snapshot. |
| `policy_rollback_state_verify` | Post-policy-rollback audit independently matches the pre-lock package inventory, backend policy file/selector, GPU UUID binding, and all other modeled state. |
| `report_schema` | All referenced reports validate against `schemas/report.schema.json`. |

The workflow performs four applicable faults on every applied matrix job and the Fabric Manager fault only on its FM-required job. It atomically hides the confirmed kernel-header symlink, unloads and reloads the exact captured NVIDIA module set after a fail-closed workload check, checks a live audit against a desired copy differing only in driver branch, and masks the resolved NVIDIA runtime tools inside a private mount namespace. The FM job also stops and restarts the captured active service. These reports come from normal CLI host probes; no audit object is patched or fabricated.

Rollback evidence is bound to the backend's exact safe transaction shape and retained package-identity manifest. APT combines local `.deb` restores and `name-` purges into one `apt-get install --no-download` solver transaction with held-package, downgrade, purge, and no-recommends controls. Zypper likewise combines local `.rpm` restores and `-name` removals into one repository-disabled, no-refresh, no-force-resolution transaction. DNF uses one fixed isolated Python/DNF transaction with available repositories absent, weak dependencies and autoremove disabled, retained RPM signatures rechecked, and the complete install/remove sets compared before execution. The DNF4 helper also requires an empty libdnf module-state delta, suppresses exactly one `updateFailSafeData` call in the process-local SWIG wrapper, restores that method, and proves both `modules.d` and the configured `persistdir/modulefailsafe` tree remained exact; an incompatible read-only SWIG binding therefore refuses safely before `do_transaction`. The check and apply paths reconstruct the same command. The release checker derives the applied package delta from the before-audit and retained baseline, then requires those exact successful commands and the applicable managed-file restores. Here, exactness means package identity and transaction membership; the retained-file SHA-256 proves which authenticated bytes were used but does not assert that a newly acquired archive is byte-identical to the archive that originally installed the package.

DNF module-policy qualification additionally requires the canonical root-controlled `/var/lib/dnf/modulefailsafe` directory to exist before the check, and requires the effective platform to be uniquely observable from an explicit `module_platform_id` or the latest `system-release` package's `base-module(platform:…)` provide. Clean images must create the standard fail-safe directory during image preparation; the policy helper deliberately will not create an un-snapshotted directory. The schema-v2.6 rollback snapshot captures the exact proof-authorized NVIDIA fail-safe YAML target even while that file is absent, and a fresh post-quarantine proof must resolve the same path before apply. The apply helper takes DNF4's native nonblocking RPMDB process lock before its first authoritative inventory and retains it through persistence and reopened-state proof; the check helper does not create DNF's PID lock file. Qualification also requires exclusive package-manager administration for the maintenance window because upstream DNF4's module-only transaction branch does not take that native lock; a concurrent privileged `dnf module` writer is outside the supported execution contract.

Every direct mutation has a shell trap, persisted prestate, and a distinct `restoration/doctor-*.json` healthy report. A separate cleanup step rechecks headers, exact module and service state, and the complete desired state before clearing the quarantine marker. A signal, cleanup error, missing restoration report, host mismatch, or invalid chronology makes the job and release evidence non-promotable. Runner loss still requires the disposable node to be quarantined or reimaged before reuse.

The container check must use the digest-pinned `container_test_image` from the desired state. Do not substitute a mutable tag. The verifier binds the audited physical GPU UUID when MIG is disabled and the unique audited MIG-device UUID when MIG is enabled. It streams its packaged C probe over standard input, verifies the source digest and the image's declared full CUDA version inside the container, compiles only in a bounded `tmpfs`, rejects any `libcuda.so.1` resolved from `/usr/local/cuda*/compat`, and requires successful `cuInit` plus `cuDeviceGetCount == 1` for the isolated device. The container runs without network access, read-only, with capabilities dropped, `no-new-privileges`, an explicit compute-only NVIDIA driver capability set, and bounded process/memory/CPU resources.

## Safety and artifact requirements

- Capture package logs, `dkms status`, `modinfo`, `uname`, and `nvidia-smi -q` only in the runner-private working directory. Never upload those raw files.
- Keep raw reports, journals, and the applicable rollback snapshot private on the disposable runner. They can contain registry/proxy configuration, host identity, device topology, command output, or package-manager details.
- The tool does not prune private reports, journals, or snapshots. Its `.active-journals` directory is an internal bounded recovery index, not exported evidence: do not edit or remove its entries, and let the tool retire each marker only after durable terminal evidence exists. The node owner must monitor `/var/lib/nvidia-converge`, archive evidence under the same access controls, and define retention appropriate to incident and rollback policy. Retire a snapshot only after every operation/report that references it is closed, no rollback can still be requested, and the node's independent recovery point is confirmed; delete its paired private evidence according to the same policy. If legacy-index bootstrap reaches its documented entry, journal-count, or byte limit, take the tool and node out of service and archive only independently confirmed terminal report/journal pairs before retrying. Never relocate unresolved or recovery-bound artifacts. Treat ordinary deletion on SSD or copy-on-write storage as logical retirement, not guaranteed physical erasure.
- Export only `attestation.json`, schema-valid sanitized reports and journals, same-host restoration reports, and sanitized canonical and pre-policy rollback snapshots plus checksums. Each export generates an in-memory random secret key that is never retained, then uses HMAC-SHA-256 to pseudonymize host and GPU UUID identity and attest managed-file content consistently within that matrix job. The export also removes device paths, clears private snapshot paths, and redacts command output and free-form evidence. Those secret-keyed content attestations prove policy-file change and unrelated-file stability without publishing configuration bytes or creating a public offline guessing oracle for low-entropy configuration.
- The public attestation manifest inventories every sanitized file, binds its exported SHA256 and canonical source-path mapping, and records the exact hosted qualification-wheel name and SHA256 used by that matrix job. It deliberately retains no digest of the private raw evidence, because an unkeyed raw-source commitment would provide an offline guessing oracle without being independently verifiable from the sanitized artifact. The GitHub artifact digest binds the complete sanitized archive. The release checker rejects missing inventory entries, wheel-binding disagreement, digest mismatches, raw sensitive fields, and every unexpected ZIP member.
- Keep the original applicable snapshot at its private path under `/var/lib/nvidia-converge/snapshots`; only that original path- and host-bound snapshot may be applied. The sanitized artifact copy is deliberately non-applicable.
- Reboot when the package manager, kernel, or module state requires it, then rerun applied verification. Record any reboot boundary in the run notes and retained evidence.
- Treat a timed-out or interrupted package operation, or a report with `incomplete: true`, as a failed scenario. Follow the README's interrupted-operation recovery runbook with the exact private snapshot authority; if that authority cannot be established, inspect through approved out-of-band recovery or reimage the disposable node.

## Promotion evidence

Record release evidence in `integrations/results.<tag>.json` using `schemas/integration-results.schema.json`. Start from `integrations/results.example.json`, which is intentionally blocked, and replace its placeholders only with observed results.

Each run binds evidence to:

- the repository, trusted workflow path, workflow run ID and attempt, workflow URL, and tested head SHA;
- the complete runner-label set, OS ID/version, package manager, fixed desired-state path, apply mode, timestamps, and observed capabilities;
- the exact qualification-wheel name and SHA256, which must equal the manifest-wide binding and the binding inside that job's sanitized attestation;
- a unique set of scenario results, each passing scenario naming a safe sanitized `reports/*.json` artifact path, with non-applicable scenarios carrying no path; and
- retained GitHub artifact ID, exact name, API archive URL, and lowercase SHA256 digest.

All recorded runs must pass. Each apply run must pass all applicable scenarios, prove the controlled pre-drift/package mutation/restoration chain, and retain a non-expired, non-placeholder sanitized artifact. `qualified_platforms` is the canonical, lexicographically sorted exact set of distinct `(os_id, os_version, package_manager)` tuples derived from those passed applied runs; no family-level boolean may expand that set. `recipe_path_coverage` records only whether the `apt-get`, `dnf`, and `zypper` convergence recipes were exercised, and is not a support claim for any other release in the same distribution family. All three recipe paths and all of `secure_boot`, `fabric_manager`, `mig_toggle`, `install_apply`, `rollback_apply`, `policy_rollback_apply`, and `policy_rollback_state_verify` must be true. Main rollback coverage also requires `rollback_state_verify`.

The release gate validates the manifest/schema, downloads the attestation ZIP, and checks GitHub's API metadata for each claimed workflow run, job, and artifact. The trusted run must be a successful `repository_dispatch` of `.github/workflows/production-gpu-qualification.yml` on `main` in the release repository at the tested SHA and recorded attempt. Workflow job ID/URL, complete runner-label set, start/completion timestamps, artifact ID/name/API URL/digest/expiry state, workflow run, head branch, and head SHA must match the claim. The upload action exposes artifact identifiers in the job summary; `workflow_job_id`, job URL, and final `completed_at` must be copied from the Jobs API only after the job completes.

After those online checks pass, the release job re-downloads each digest-bound sanitized source archive and writes a deterministic `gpu-integration-evidence-<tag>.zip`. It canonicalizes each validated inner ZIP to strip archive metadata and unneeded covert channels rather than retaining the original GitHub archive bytes. The outer bundle contains the checked evidence manifest, those canonical sanitized archives, a bundle manifest mapping every GitHub source digest to its canonical digest and retained path, the common qualification-wheel binding, and internal checksums. Promotion fails unless the release wheel rebuilt directly from the tested commit has that exact name and SHA256. The bundle is included in the release checksums, receives the same GitHub build-provenance attestation as the wheel and source distribution, and is published as a durable GitHub release asset. The source Actions artifacts and the build-to-publish handoff are retained for 30 days; they are required for promotion but are not the long-term evidence store. Raw reports and the applicable snapshots remain excluded.

The owner-enforced [immutable releases setting](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes) must be active before promotion. `.github/workflows/production-release.yml` is `repository_dispatch`-only and has exactly four dependent jobs: `repository-controls` -> `gates` -> `artifacts` -> `publish`. The first job binds the run to the exact live `main` head, derives the tag from `pyproject.toml`, and rejects missing visible production controls without an environment or write permission. The gate job pins CPython 3.12.13, installs the fully hashed `requirements/release.lock`, requires exact equality between the checkout, `origin/main`, and the dispatch SHA, and runs all release tests without uploading candidate files. The artifact job starts on a fresh hosted runner, repeats those checks, and rejects a dirty checkout. It builds two direct wheels from independent archives of the exact GPU-tested commit using that commit's `SOURCE_DATE_EPOCH`, builds two source distributions from independent archives of the release commit, canonicalizes them, and requires byte-identical pairs. The evidence bundler runs from another release-commit archive and requires the wheel name and SHA-256 to match the qualification binding retained across all accepted GPU attestations.

The publish job depends only on the exact artifact handoff, never on files or interpreter state from the gate runner. After protected `release` approval, it checks live `main`, executes the checksum-verified wheel's control checker with `REPOSITORY_AUDIT_TOKEN`, and requires empty bypass inventories for every effective `main` ruleset, owner-enforced immutability, the exact App/no-bypass tag rules, and the release-writer isolation described above. It then mints a short-lived Release Creator App token, proves that token can see exactly this repository, and creates the fresh tag at the gated commit. It attests all four files, creates exactly one new draft by REST, records that draft's numeric ID, and uploads assets only to that ID. Before publication it verifies exact metadata, downloads every asset by REST asset ID, byte-compares them, rechecks the downloaded checksum manifest, and repeats the live-main, control, tag, and asset checks. Publication overwrites the canonical release metadata while setting `draft: false`, then the final checks require that same ID, tag, target commit, exact asset inventory, and immutable state. GitHub does not expose an atomic conditional publish operation, so the isolation checks define organization owners as trusted and exclude every other visible writer during the release window. A failed draft must be inspected and removed explicitly before a one-shot retry. Protected approval must complete before the 30-day handoff expires.

Consumers verify in this order:

1. Run `gh release verify <tag> --repo zeroecco/nvidia-converge` to verify the immutable release attestation, download all four assets from that release, and run `sha256sum --check --strict -- SHA256SUMS`.
2. Run `gh attestation verify` for all four assets with `--repo zeroecco/nvidia-converge`, `--signer-workflow zeroecco/nvidia-converge/.github/workflows/production-release.yml`, `--source-ref refs/heads/main`, and `--source-digest <trusted-release-commit>`.
3. From the unpacked, attested source distribution or a trusted checkout of that source commit on Linux x86_64 with CPython 3.12, install `requirements/release.lock` with `--require-hashes --no-deps` in an unprivileged verification environment and run the offline bundle verifier:

   ```bash
   python -m scripts.verify_release_evidence_bundle \
     --bundle "gpu-integration-evidence-${TAG}.zip" \
     --release "$TAG" \
     --repository zeroecco/nvidia-converge \
     --qualification-wheel "nvidia_converge-${TAG#v}-py3-none-any.whl"
   ```

The release-time gate authenticates the workflow run, job, artifact metadata, and source archive digest through GitHub's API before canonicalization. Once the source Actions artifact expires, the durable bundle inherits that checked API provenance through its manifests and release attestation; the offline verifier validates the retained canonical evidence and bindings, including byte equality between the downloaded release wheel and the GPU qualification digest, but does not perform a fresh API authentication of expired artifacts.

To exercise the same evidence checks before release dispatch:

```bash
GITHUB_TOKEN=<read-actions-token> \
python scripts/check_release_evidence.py \
  --evidence integrations/results.v0.1.0.json \
  --release v0.1.0 \
  --commit "$(git rev-parse HEAD)" \
  --repository zeroecco/nvidia-converge \
  --verify-github \
  --qualification-wheel dist/nvidia_converge-0.1.0-py3-none-any.whl
```

The evidence `commit` must be the full lowercase SHA tested by every run and an ancestor of the release-dispatch commit. The only permitted path changed between the tested commit and that live `main` commit is the matching `integrations/results.<tag>.json` evidence file. Set the package version before qualification; after qualification, merge only that evidence file before dispatch. The protected workflow, not an operator, creates the matching tag.

Treat `install --apply` and `rollback --apply` as production-proven only for the exact release, desired state, platforms, and capabilities supported by a passing manifest. Evidence from another commit, a synthetic audit, an expired/unverifiable artifact, or a runner with incomplete cleanup does not promote a release. The checked-in example remains intentionally blocked; no release is promoted until genuine successful workflow/job/artifact identifiers and observed reports replace every placeholder.
