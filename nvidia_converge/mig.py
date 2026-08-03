from __future__ import annotations

import re

from .models import MigGpuInstance

MIG_DISABLED_PROFILE = "none"
MIG_FULL_PROFILE = "full"


def full_mig_geometry_matches(
    geometry: list[MigGpuInstance],
    gpu_uuids: list[str],
) -> bool:
    """Return whether geometry is one full-profile GI with one full CI."""
    return bool(
        len(gpu_uuids) == 1
        and len(geometry) == 1
        and geometry[0].gpu_uuid == gpu_uuids[0]
        and geometry[0].profile_id == 0
        and geometry[0].placement_start == 0
        and len(geometry[0].compute_instances) == 1
        and _compute_instance_uses_full_gpu_instance(geometry[0])
    )


def restorable_mig_geometry(
    geometry: list[MigGpuInstance],
    gpu_uuids: list[str],
) -> bool:
    """Recognize geometry that can be recreated exactly with nvidia-smi -C.

    Empty geometry is exact.  A populated baseline is qualified only when it is
    a single GPU instance with a single compute instance consuming that entire
    GPU instance.  This lets rollback recreate the recorded profile and
    placement without depending on ephemeral GI/CI IDs or MIG UUIDs.
    """
    if not geometry:
        return True
    return bool(
        len(gpu_uuids) == 1
        and len(geometry) == 1
        and geometry[0].gpu_uuid == gpu_uuids[0]
        and len(geometry[0].compute_instances) == 1
        and _compute_instance_uses_full_gpu_instance(geometry[0])
    )


def mig_geometry_create_command(
    gpu_uuid: str,
    geometry: list[MigGpuInstance],
) -> list[str] | None:
    """Build the exact supported GI/default-CI recreation command."""
    if not geometry:
        return None
    instance = geometry[0]
    if not restorable_mig_geometry(geometry, [gpu_uuid]):
        raise ValueError("MIG geometry is not safely restorable")
    return [
        "nvidia-smi",
        "mig",
        "-i",
        gpu_uuid,
        "-cgi",
        f"{instance.profile_id}:{instance.placement_start}",
        "-C",
    ]


def desired_full_mig_geometry_command(gpu_uuid: str) -> list[str]:
    """Create one full physical-GPU GI and its default full-size CI."""
    return [
        "nvidia-smi",
        "mig",
        "-i",
        gpu_uuid,
        "-cgi",
        "0:0",
        "-C",
    ]


def mig_geometry_destroy_commands(gpu_uuid: str) -> list[list[str]]:
    """Destroy CIs before their parent GIs on one UUID-bound GPU."""
    return [
        ["nvidia-smi", "mig", "-i", gpu_uuid, "-dci"],
        ["nvidia-smi", "mig", "-i", gpu_uuid, "-dgi"],
    ]


def _compute_instance_uses_full_gpu_instance(instance: MigGpuInstance) -> bool:
    compute = instance.compute_instances[0]
    gpu_profile = _normalize_profile(instance.profile)
    compute_profile = _normalize_profile(compute.profile)
    if compute_profile == gpu_profile:
        return True
    gpu_match = re.fullmatch(r"(?P<slices>[1-9]\d*)g\.(?P<rest>.+)", gpu_profile)
    compute_match = re.fullmatch(
        r"(?P<slices>[1-9]\d*)c\.(?P<gpu>[1-9]\d*g\..+)",
        compute_profile,
    )
    return bool(
        gpu_match
        and compute_match
        and compute_match.group("gpu") == gpu_profile
        and compute_match.group("slices") == gpu_match.group("slices")
    )


def _normalize_profile(profile: str) -> str:
    return profile.removeprefix("MIG ").strip().lower()
