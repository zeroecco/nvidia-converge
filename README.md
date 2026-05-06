# nvidia-converge

[![CI](https://github.com/zeroecco/nvidia-converge/actions/workflows/ci.yml/badge.svg)](https://github.com/zeroecco/nvidia-converge/actions/workflows/ci.yml)

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

Commands print a compact human summary by default. Use `--out report.json` to write the full machine-readable report, or `--json` to print that report to stdout.

From a source checkout, run the CLI as a module:

```bash
python3 -m nvidia_converge --version
python3 -m nvidia_converge doctor
python3 -m nvidia_converge plan --out report.json
sudo python3 -m nvidia_converge install --apply --out report.json
python3 -m nvidia_converge verify --out verify.json
python3 -m nvidia_converge lock --apply --out lock.json
sudo python3 -m nvidia_converge rollback --snapshot /var/lib/nvidia-converge/snapshots/latest.json --apply
```

Or use the checkout-local launcher:

```bash
./nvidia-converge doctor
./nvidia-converge plan --out report.json
sudo ./nvidia-converge install --apply --out report.json
```

On Ubuntu and other PEP 668 distributions, do not install into the system Python with `python3 -m pip install -e .`. Use a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
nvidia-converge doctor
```

After venv installation, use the packaged command while the venv is active:

```bash
nvidia-converge doctor
nvidia-converge plan --out report.json
sudo nvidia-converge install --apply --out report.json
nvidia-converge verify --out verify.json
nvidia-converge lock --apply --out lock.json
sudo nvidia-converge rollback --snapshot /var/lib/nvidia-converge/snapshots/latest.json --apply
```

Host-mutating commands (`install`, `lock`, `rollback`) are dry-run unless `--apply` is supplied.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[test]"
python3 -m compileall -q nvidia_converge tests
python3 tests/run_tests.py
python3 -m pytest -q
python3 -m build
```

## Releases

Tagged releases build and smoke-test wheel/sdist artifacts in GitHub Actions:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Report

Every command can write a JSON report with audit findings, diagnostics, proposed or applied actions, verification results, rollback snapshot metadata, and an SBOM-style package/module inventory.
