# nvidia-converge

`nvidia-converge` is a node-level NVIDIA driver stack reconciler. It reads a desired state, audits the host, explains breakage, previews package/kernel/module changes, optionally converges the machine, validates the GPU stack, pins compatibility-sensitive packages, records rollback metadata, and emits a machine-readable compliance report.

Default desired state:

```yaml
desired:
  role: compute
  driver: 580-open
  cuda_compat: 13.0
  secure_boot: signed
  container_runtime: docker
  fabric_manager: true
  mig: disabled
  kernel_policy: pin-compatible
```

## Usage

From a source checkout, run the CLI as a module:

```bash
python3 -m nvidia_converge doctor
python3 -m nvidia_converge plan --out report.json
sudo python3 -m nvidia_converge install --apply --out report.json
python3 -m nvidia_converge verify --out verify.json
python3 -m nvidia_converge lock --apply --out lock.json
sudo python3 -m nvidia_converge rollback --snapshot /var/lib/nvidia-converge/snapshots/latest.json --apply
```

To install the console command in your current Python environment:

```bash
python3 -m pip install -e .
```

After installation, use the packaged command:

```bash
nvidia-converge doctor
nvidia-converge plan --out report.json
sudo nvidia-converge install --apply --out report.json
nvidia-converge verify --out verify.json
nvidia-converge lock --apply --out lock.json
sudo nvidia-converge rollback --snapshot /var/lib/nvidia-converge/snapshots/latest.json --apply
```

Host-mutating commands (`install`, `lock`, `rollback`) are dry-run unless `--apply` is supplied.

## Report

Every command can write a JSON report with audit findings, diagnostics, proposed or applied actions, verification results, rollback snapshot metadata, and an SBOM-style package/module inventory.
