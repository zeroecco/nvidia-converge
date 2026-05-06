# Integration Testing

`nvidia-converge` has unit, schema, packaging, and CLI smoke coverage in CI. Host mutation still needs validation on disposable GPU nodes before an apply path should be treated as fully production-proven.

The manual GitHub Actions workflow `.github/workflows/gpu-integration.yml` runs the executable integration sequence on self-hosted runners labeled `self-hosted`, `nvidia-gpu`, and `disposable`. With `apply=false`, it performs non-mutating plan/report validation. With `apply=true`, it runs install, verify, lock, rollback, and post-rollback verify.

## Required Test Nodes

- Ubuntu LTS with apt and Docker.
- RHEL-family host with dnf or yum and Docker.
- SUSE-family host with zypper and Docker.
- At least one Secure Boot enabled host.
- At least one host with Fabric Manager capable GPUs.
- At least one host where MIG can be toggled safely.

## Required Scenarios

1. `doctor` on a healthy host reports no blocking findings.
2. `doctor` on a broken host explains the exact failure:
   - missing kernel headers
   - unloaded NVIDIA module
   - driver/userspace mismatch
   - Docker NVIDIA runtime missing
   - Fabric Manager inactive
3. `plan` shows package, kernel, module, service, runtime, lock, and verification actions without changing the host.
4. `install --apply` converges from a clean base image to the desired state.
5. `verify --apply` proves:
   - module can be loaded
   - `nvidia-smi` works
   - NVML loads
   - Docker GPU container can run `nvidia-smi`
   - Fabric Manager state matches the desired state
6. `lock --apply` pins or locks the expected packages for the package manager.
7. `snapshot` writes rollback metadata before mutation.
8. `rollback --apply` restores the previous package/module state after an applied install.
9. `--out report.json` emits a report that validates against `schemas/report.schema.json`.

## Safety Requirements

- Run only on disposable nodes or maintenance-window hosts.
- Capture `/var/log/apt`, `/var/log/dnf*`, `/var/log/zypp`, `dkms status`, `modinfo nvidia`, and `nvidia-smi -q` before and after mutation.
- Preserve generated reports and rollback snapshots as CI artifacts.
- Reboot when the package manager or kernel update path requires it, then rerun `verify --apply`.

## Promotion Gate

Treat `install --apply` and `rollback --apply` as production-proven only after the required scenarios pass on all required test nodes for the target release tag.

Record promotion evidence in `integrations/results.<tag>.json` using `schemas/integration-results.schema.json`. The schema is also available through:

```bash
nvidia-converge schema integration-results
```

Start from `integrations/results.example.json`, replace blocked statuses with actual workflow results, and include the retained artifact URI and SHA256 for each GPU integration run.
