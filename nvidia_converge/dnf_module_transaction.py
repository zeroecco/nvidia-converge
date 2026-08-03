from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .models import CommandResult

DNF_MODULE_NAME = "nvidia-driver"
DNF_MODULE_STATE_FILE = "nvidia-driver.module"
DNF_MODULE_FAILSAFE_DIRECTORY = "/var/lib/dnf/modulefailsafe"
DNF_MODULE_UNBOUND_PREFLIGHT = "unbound-preflight-proof"
_MODULE_STREAM_PATTERN = re.compile(r"[1-9]\d{2,3}-(?:open|dkms)")
_MODULE_FIELD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,255}")
_MODULE_VERSION_PATTERN = re.compile(r"[0-9]{1,32}")
_MODULE_PLATFORM_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}:[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}"
)
_SHA256_PATTERN = re.compile(r"[a-f0-9]{64}")
_MAX_PROOF_BYTES = 256 * 1024
_MAX_REQUIREMENTS = 256
_MAX_ACTIVE_MODULES = 4096
_MAX_REPOSITORIES = 4096


# This probe deliberately uses the system Python: on supported DNF 4 hosts,
# python3-dnf and libdnf are distribution-owned bindings.  Keeping the complete
# check and mutation in one fixed script lets applied execution repeat the
# dependency proof immediately before ModuleBase persists any state.
DNF_MODULE_ENABLE_SCRIPT = r'''
import configparser
import hashlib
import json
import os
import re
import stat
import sys

STATE_DIRECTORY = "/etc/dnf/modules.d"
PERSIST_DIRECTORY = "/var/lib/dnf"
FAILSAFE_DIRECTORY = "/var/lib/dnf/modulefailsafe"
TARGET_NAME = "nvidia-driver"
TARGET_FILE = "nvidia-driver.module"
DIRECTORY_RECORD = "__directory_metadata__"
MAX_STATE_FILES = 4096
MAX_STATE_DIRECTORY_ENTRIES = 8192
MAX_STATE_FILE_BYTES = 64 * 1024
MAX_STATE_TOTAL_BYTES = 8 * 1024 * 1024
MAX_FAILSAFE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MATCHED_TARGETS = 64
MAX_ACTIVE_TARGETS = 2
MAX_ACTIVE_MODULES = 4096
MAX_MODULE_PACKAGES = 131072
MAX_REQUIREMENTS = 256
MAX_REQUIREMENT_STREAMS = 64
MAX_MODULE_YAML_BYTES = 1024 * 1024
MAX_ACTIVE_MODULE_YAML_BYTES = 32 * 1024 * 1024
MAX_REPOSITORIES = 4096
MAX_PLATFORM_PACKAGES = 64
MAX_PACKAGE_PROVIDES = 4096
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,254}\.module")
SAFE_FAILSAFE_NAME = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}:"
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}:"
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}\.yaml"
)
SAFE_FIELD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,255}")
SAFE_REPOSITORY = re.compile(r"[A-Za-z0-9@][A-Za-z0-9_.+@-]{0,255}")
SAFE_STREAM = re.compile(r"[1-9]\d{2,3}-(?:open|dkms)")
SAFE_VERSION = re.compile(r"[0-9]{1,32}")
SAFE_SHA256 = re.compile(r"[a-f0-9]{64}")
SAFE_PLATFORM_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}:"
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}"
)
SAFE_BASE_MODULE_PROVIDE = re.compile(
    r"base-module\(("
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}:"
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}"
    r")\)"
)


def trusted_directory(path):
    if not path.startswith("/") or os.path.normpath(path) != path:
        raise RuntimeError("module state directory is not an exact absolute path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptor = os.open("/", flags)
    try:
        current = "/"
        for component in path.split("/")[1:]:
            metadata = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != 0
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise RuntimeError(
                    "module state ancestor is not a trusted root-owned directory: "
                    + os.path.join(current, component)
                )
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if (
                opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or not stat.S_ISDIR(opened.st_mode)
                or opened.st_uid != 0
                or stat.S_IMODE(opened.st_mode) & 0o022
            ):
                os.close(child)
                raise RuntimeError("module state ancestor changed while opening")
            os.close(descriptor)
            descriptor = child
            current = os.path.join(current, component)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def state_files(
    path=STATE_DIRECTORY,
    suffix=".module",
    filename_pattern=SAFE_NAME,
    max_file_bytes=MAX_STATE_FILE_BYTES,
    max_total_bytes=MAX_STATE_TOTAL_BYTES,
    reject_unexpected=False,
):
    directory = trusted_directory(path)
    try:
        opened_directory = os.fstat(directory)
        directory_names = os.listdir(directory)
        if len(directory_names) > MAX_STATE_DIRECTORY_ENTRIES:
            raise RuntimeError("module state directory inventory is too large")
        names = sorted(
            name for name in directory_names if name.endswith(suffix)
        )
        if reject_unexpected and len(names) != len(directory_names):
            raise RuntimeError("module state directory has an unexpected entry")
        if len(names) > MAX_STATE_FILES:
            raise RuntimeError("module state file inventory is too large")
        records = {}
        total = 0
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for name in names:
            if not filename_pattern.fullmatch(name):
                raise RuntimeError("module state contains an unsafe filename")
            metadata = os.stat(
                name,
                dir_fd=directory,
                follow_symlinks=False,
            )
            mode = stat.S_IMODE(metadata.st_mode)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != 0
                or mode & 0o022
                or metadata.st_nlink != 1
                or metadata.st_size < 0
                or metadata.st_size > max_file_bytes
            ):
                raise RuntimeError(
                    "module state file is not a bounded root-controlled regular file: "
                    + name
                )
            total += metadata.st_size
            if total > max_total_bytes:
                raise RuntimeError("module state file inventory is too large")
            descriptor = os.open(name, flags, dir_fd=directory)
            try:
                opened = os.fstat(descriptor)
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or opened.st_size != metadata.st_size
                    or opened.st_uid != metadata.st_uid
                    or opened.st_mode != metadata.st_mode
                    or opened.st_nlink != metadata.st_nlink
                ):
                    raise RuntimeError("module state file changed while opening")
                remaining = metadata.st_size + 1
                chunks = []
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                content = b"".join(chunks)
                if len(content) != metadata.st_size or remaining == 0:
                    raise RuntimeError("module state file changed while reading")
                final = os.fstat(descriptor)
                if (
                    final.st_dev != opened.st_dev
                    or final.st_ino != opened.st_ino
                    or final.st_size != opened.st_size
                    or final.st_uid != opened.st_uid
                    or final.st_mode != opened.st_mode
                    or final.st_nlink != opened.st_nlink
                    or final.st_mtime_ns != opened.st_mtime_ns
                    or final.st_ctime_ns != opened.st_ctime_ns
                ):
                    raise RuntimeError("module state file changed while reading")
            finally:
                os.close(descriptor)
            records[name] = {
                "content": content,
                "ctime_ns": opened.st_ctime_ns,
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "mode": mode,
                "mtime_ns": opened.st_mtime_ns,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        if sorted(os.listdir(directory)) != sorted(directory_names):
            raise RuntimeError("module state directory changed during inventory")
        final_directory = os.fstat(directory)
        if (
            final_directory.st_dev != opened_directory.st_dev
            or final_directory.st_ino != opened_directory.st_ino
            or final_directory.st_mode != opened_directory.st_mode
            or final_directory.st_uid != opened_directory.st_uid
            or final_directory.st_mtime_ns != opened_directory.st_mtime_ns
            or final_directory.st_ctime_ns != opened_directory.st_ctime_ns
        ):
            raise RuntimeError("module state directory changed during inventory")
        records[DIRECTORY_RECORD] = {
            "ctime_ns": opened_directory.st_ctime_ns,
            "device": opened_directory.st_dev,
            "inode": opened_directory.st_ino,
            "mode": stat.S_IMODE(opened_directory.st_mode),
            "mtime_ns": opened_directory.st_mtime_ns,
        }
        return records
    finally:
        os.close(directory)


def inventory_digest(records):
    public = {}
    for name, record in records.items():
        public[name] = {
            key: value
            for key, value in record.items()
            if key != "content"
        }
    encoded = json.dumps(
        public,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def failsafe_files():
    return state_files(
        FAILSAFE_DIRECTORY,
        ".yaml",
        SAFE_FAILSAFE_NAME,
        MAX_MODULE_YAML_BYTES,
        MAX_FAILSAFE_TOTAL_BYTES,
        True,
    )


def combined_state_digest(module_records, failsafe_records):
    return canonical_digest(
        {
            "module_failsafe_sha256": inventory_digest(failsafe_records),
            "module_state_sha256": inventory_digest(module_records),
        }
    )


def changed_files(before, after):
    names = (set(before) | set(after)) - {DIRECTORY_RECORD}
    return sorted(
        name
        for name in names
        if before.get(name) != after.get(name)
    )


def require_exact_directory_identity(before, after, label):
    before_directory = dict(before[DIRECTORY_RECORD])
    after_directory = dict(after[DIRECTORY_RECORD])
    before_directory.pop("mtime_ns")
    before_directory.pop("ctime_ns")
    after_directory.pop("mtime_ns")
    after_directory.pop("ctime_ns")
    if before_directory != after_directory:
        raise RuntimeError("DNF replaced or changed the " + label + " directory")


def safe_field(value, label, pattern=SAFE_FIELD):
    text = str(value)
    if not pattern.fullmatch(text):
        raise RuntimeError("module metadata contains an invalid " + label)
    return text


def canonical_digest(value):
    encoded = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def enabled_repositories(base):
    repositories = sorted(
        safe_field(repository.id, "repository")
        for repository in base.repos.iter_enabled()
    )
    if (
        len(repositories) > MAX_REPOSITORIES
        or len(repositories) != len(set(repositories))
    ):
        raise RuntimeError("enabled DNF repository inventory is invalid")
    return repositories


def platform_ids_from_packages(packages):
    if len(packages) > MAX_PLATFORM_PACKAGES:
        raise RuntimeError("DNF platform package inventory is too large")
    platform_ids = set()
    for package in packages:
        provides = list(package.provides)
        if len(provides) > MAX_PACKAGE_PROVIDES:
            raise RuntimeError("DNF platform package provides are too large")
        for provide in provides:
            match = SAFE_BASE_MODULE_PROVIDE.fullmatch(str(provide))
            if match is not None:
                platform_ids.add(match.group(1))
    return platform_ids


def effective_platform_id(base):
    configured = base.conf.module_platform_id
    if configured is not None:
        configured = str(configured)
        if not SAFE_PLATFORM_ID.fullmatch(configured):
            raise RuntimeError("configured DNF module platform ID is invalid")
        return configured

    # Mirror libdnf's preferred auto-detection inputs from the exact sack that
    # modular filtering used: latest available system-release packages first,
    # then latest installed ones.  A supported production host must expose one
    # unambiguous base-module(name:stream) provide; libdnf's os-release fallback
    # is intentionally rejected because it is not observable through the DNF
    # transaction API with equivalent race-resistant file provenance here.
    base_query = base.sack.query().filter(
        provides="system-release",
        latest=1,
    )
    for packages in (list(base_query.available()), list(base_query.installed())):
        platform_ids = platform_ids_from_packages(packages)
        if len(platform_ids) == 1:
            return next(iter(platform_ids))
    raise RuntimeError(
        "DNF effective module platform ID is not uniquely inspectable from "
        "system-release metadata"
    )


def module_identity(module):
    name = safe_field(module.getName(), "name")
    stream = safe_field(module.getStream(), "stream")
    version = safe_field(module.getVersion(), "version", SAFE_VERSION)
    context = safe_field(module.getContext(), "context")
    architecture = safe_field(module.getArch(), "architecture")
    repository = safe_field(
        module.getRepoID(),
        "repository",
        SAFE_REPOSITORY,
    )
    yaml = module.getYaml()
    if (
        not isinstance(yaml, str)
        or not yaml
        or len(yaml.encode("utf-8")) > MAX_MODULE_YAML_BYTES
    ):
        raise RuntimeError("active module metadata is not inspectable")
    return {
        "architecture": architecture,
        "context": context,
        "name": name,
        "repository": repository,
        "stream": stream,
        "version": version,
        "yaml_sha256": hashlib.sha256(yaml.encode("utf-8")).hexdigest(),
    }, yaml.encode("utf-8")


def active_modules(container):
    modules = container.getModulePackages()
    if len(modules) > MAX_MODULE_PACKAGES:
        raise RuntimeError("DNF module metadata inventory is too large")
    active = []
    total_yaml_bytes = 0
    for module in modules:
        if not container.isModuleActive(module.getId()):
            continue
        name = safe_field(module.getName(), "name")
        if name == "platform":
            continue
        record, yaml = module_identity(module)
        active.append(record)
        total_yaml_bytes += len(yaml)
        if (
            len(active) > MAX_ACTIVE_MODULES
            or total_yaml_bytes > MAX_ACTIVE_MODULE_YAML_BYTES
        ):
            raise RuntimeError("active module identity inventory is too large")
    active.sort(
        key=lambda item: (
            item["name"],
            item["stream"],
            item["version"],
            item["context"],
            item["architecture"],
            item["repository"],
            item["yaml_sha256"],
        )
    )
    if len(active) != len({tuple(sorted(record.items())) for record in active}):
        raise RuntimeError("active module identity inventory is ambiguous")
    return active


def require_stable_active_modules(before, after, active_target):
    before_unrelated = [
        record for record in before if record["name"] != TARGET_NAME
    ]
    after_unrelated = [
        record for record in after if record["name"] != TARGET_NAME
    ]
    after_target = [
        record for record in after if record["name"] == TARGET_NAME
    ]
    if before_unrelated != after_unrelated:
        raise RuntimeError(
            "DNF module enable changed an unrelated active module identity"
        )
    if after_target != active_target:
        raise RuntimeError("DNF active module inventory disagrees on the target")


def failsafe_filename(record):
    filename = (
        record["name"]
        + ":"
        + record["stream"]
        + ":"
        + record["architecture"]
        + ".yaml"
    )
    if not SAFE_FAILSAFE_NAME.fullmatch(filename):
        raise RuntimeError("active module has an unsafe fail-safe filename")
    return filename


def active_enabled_failsafe(container):
    expected = {}
    modules = container.getModulePackages()
    if len(modules) > MAX_MODULE_PACKAGES:
        raise RuntimeError("DNF module metadata inventory is too large")
    for module in modules:
        if not container.isModuleActive(module.getId()):
            continue
        if safe_field(module.getName(), "name") == "platform":
            continue
        record, yaml = module_identity(module)
        if record["repository"] == "@System" or not container.isEnabled(
            record["name"],
            record["stream"],
        ):
            continue
        filename = failsafe_filename(record)
        if filename in expected:
            raise RuntimeError(
                "DNF has multiple active enabled identities for one fail-safe slot"
            )
        expected[filename] = {"identity": record, "yaml": yaml}
    return expected


def require_failsafe_baseline(records, expected, container, target_filename):
    for filename, item in expected.items():
        if item["identity"]["name"] == TARGET_NAME:
            raise RuntimeError("NVIDIA target is already active and enabled")
        record = records.get(filename)
        if record is None or record["content"] != item["yaml"]:
            raise RuntimeError(
                "an unrelated active module lacks exact fail-safe metadata"
            )
    for filename, record in records.items():
        if filename == DIRECTORY_RECORD:
            continue
        name, stream, _architecture_yaml = filename.split(":", 2)
        if name == TARGET_NAME:
            raise RuntimeError(
                "NVIDIA target fail-safe metadata already exists before enable"
            )
        if filename in expected:
            continue
        if not container.isEnabled(name, stream):
            raise RuntimeError(
                "DNF would delete unrelated stale module fail-safe metadata"
            )
    if target_filename in records:
        raise RuntimeError("NVIDIA target fail-safe metadata baseline is not absent")


def require_exact_failsafe_resolution(before, after, active_target):
    target_filename = failsafe_filename(active_target[0])
    before_unrelated = {
        filename: item
        for filename, item in before.items()
        if item["identity"]["name"] != TARGET_NAME
    }
    after_unrelated = {
        filename: item
        for filename, item in after.items()
        if item["identity"]["name"] != TARGET_NAME
    }
    target = {
        filename: item
        for filename, item in after.items()
        if item["identity"]["name"] == TARGET_NAME
    }
    if before_unrelated != after_unrelated or list(target) != [target_filename]:
        raise RuntimeError("DNF fail-safe resolution expanded beyond the target")
    target_yaml = target[target_filename]["yaml"]
    if hashlib.sha256(target_yaml).hexdigest() != active_target[0]["yaml_sha256"]:
        raise RuntimeError("DNF target fail-safe YAML identity is inconsistent")
    return target_filename, target_yaml


def require_exact_failsafe_postcondition(
    before,
    after,
    target_filename,
    target_yaml,
):
    if changed_files(before, after) != [target_filename]:
        raise RuntimeError("DNF changed fail-safe metadata outside the exact target")
    target = after.get(target_filename)
    if (
        target is None
        or target["content"] != target_yaml
        or target["mode"] != 0o644
    ):
        raise RuntimeError("DNF did not persist exact target fail-safe metadata")
    require_exact_directory_identity(before, after, "fail-safe")


def target_evidence(module_base, container, target):
    modules, nsvcap = module_base.get_modules(target)
    if (
        nsvcap is None
        or str(nsvcap.name or "") != TARGET_NAME
        or str(nsvcap.stream or "") != target.split(":", 1)[1]
        or nsvcap.version not in (None, -1)
        or nsvcap.context not in (None, "")
        or nsvcap.arch not in (None, "")
        or nsvcap.profile not in (None, "")
    ):
        raise RuntimeError("DNF did not parse the exact requested module stream")
    if len(modules) > MAX_MATCHED_TARGETS:
        raise RuntimeError("DNF target module metadata inventory is too large")
    active = []
    requirements = set()
    for module in modules:
        if not container.isModuleActive(module.getId()):
            continue
        record, _yaml_bytes = module_identity(module)
        name = record["name"]
        stream = safe_field(record["stream"], "stream", SAFE_STREAM)
        version = record["version"]
        context = record["context"]
        architecture = record["architecture"]
        repository = record["repository"]
        if name != TARGET_NAME or target != name + ":" + stream:
            raise RuntimeError("DNF activated a different target module stream")
        active.append(record)
        if len(active) > MAX_ACTIVE_TARGETS:
            raise RuntimeError("DNF activated too many target module identities")
        for dependency in module.getModuleDependencies():
            for block in dependency.getRequires():
                for dependency_name, dependency_streams in block.items():
                    dependency_name = safe_field(
                        dependency_name,
                        "dependency name",
                    )
                    if len(dependency_streams) > MAX_REQUIREMENT_STREAMS:
                        raise RuntimeError("module dependency stream set is too large")
                    streams = tuple(
                        sorted(
                            safe_field(stream, "dependency stream")
                            for stream in dependency_streams
                        )
                    )
                    if not streams or len(streams) != len(set(streams)):
                        raise RuntimeError("module dependency stream set is ambiguous")
                    requirements.add((dependency_name, streams))
                    if len(requirements) > MAX_REQUIREMENTS:
                        raise RuntimeError("module dependency inventory is too large")
    active.sort(
        key=lambda item: (
            item["name"],
            item["stream"],
            item["version"],
            item["context"],
            item["architecture"],
            item["repository"],
            item["yaml_sha256"],
        )
    )
    if len(active) != 1:
        raise RuntimeError(
            "DNF target stream does not resolve to one exact active "
            "version/context/architecture/repository identity"
        )
    return active, [
        {"name": name, "streams": list(streams)}
        for name, streams in sorted(requirements)
    ]


def pending_changes(container):
    return {
        "disable": sorted(str(name) for name in container.getDisabledModules()),
        "enable": {
            str(name): str(stream)
            for name, stream in sorted(container.getEnabledStreams().items())
        },
        "install_profiles": {
            str(name): sorted(str(profile) for profile in profiles)
            for name, profiles in sorted(container.getInstalledProfiles().items())
        },
        "remove_profiles": {
            str(name): sorted(str(profile) for profile in profiles)
            for name, profiles in sorted(container.getRemovedProfiles().items())
        },
        "reset": sorted(str(name) for name in container.getResetModules()),
        "switch": {
            str(name): [str(streams[0]), str(streams[1])]
            for name, streams in sorted(container.getSwitchedStreams().items())
        },
    }


def require_exact_enable(before, after, stream):
    empty = {
        "disable": [],
        "enable": {},
        "install_profiles": {},
        "remove_profiles": {},
        "reset": [],
        "switch": {},
    }
    if before != empty:
        raise RuntimeError(
            "DNF module container has a pre-existing pending state change"
        )
    expected = dict(empty)
    expected["enable"] = {TARGET_NAME: stream}
    if after != expected:
        raise RuntimeError("DNF did not plan the exact target-only module enable")
    return expected


def require_target_file(records, stream):
    record = records.get(TARGET_FILE)
    if record is None:
        raise RuntimeError("DNF did not persist the target module state file")
    try:
        text = record["content"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("DNF target module state is not UTF-8") from exc
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.read_string(text)
    if parser.sections() != [TARGET_NAME]:
        raise RuntimeError("DNF target module state has an unexpected section")
    section = parser[TARGET_NAME]
    if set(section) != {"name", "profiles", "state", "stream"}:
        raise RuntimeError("DNF target module state has unexpected fields")
    if (
        section.get("name", "") != TARGET_NAME
        or section.get("profiles", "").strip()
        or section.get("state", "").strip().lower() != "enabled"
        or section.get("stream", "").strip() != stream
    ):
        raise RuntimeError("DNF target module state does not match the exact request")


def proof_preflight_sha256(payload):
    normalized = {
        key: payload[key]
        for key in (
            "active_modules_count",
            "active_modules_sha256",
            "active_target",
            "changes",
            "failsafe_target",
            "module_failsafe_before_sha256",
            "module_platform_id",
            "module_state_before_sha256",
            "repositories_count",
            "repositories_sha256",
            "requirements",
            "schema",
            "state_before_sha256",
            "target",
        )
    }
    normalized.update(
        {
            "applied": False,
            "failsafe_changed_files": [],
            "module_changed_files": [],
            "module_failsafe_after_sha256": payload[
                "module_failsafe_before_sha256"
            ],
            "module_state_after_sha256": payload["module_state_before_sha256"],
            "state_after_sha256": payload["state_before_sha256"],
        }
    )
    return canonical_digest(normalized)


rpmdb_lock = None
try:
    if os.geteuid() != 0:
        raise RuntimeError("DNF module transaction requires root")
    if len(sys.argv) < 2:
        raise RuntimeError("invalid DNF module transaction arguments")
    mode = sys.argv[1]
    if mode not in {"--check", "--apply"}:
        raise RuntimeError("invalid DNF module transaction mode")
    expected_arguments = 4 if mode == "--check" else 5
    if len(sys.argv) != expected_arguments:
        raise RuntimeError("invalid DNF module transaction arguments")
    name, stream = sys.argv[2:4]
    supplied_preflight = sys.argv[4] if mode == "--apply" else None
    if name != TARGET_NAME or not SAFE_STREAM.fullmatch(stream):
        raise RuntimeError("invalid exact NVIDIA DNF module stream")
    if mode == "--apply" and not SAFE_SHA256.fullmatch(supplied_preflight or ""):
        raise RuntimeError("DNF module apply lacks a bound preflight proof")
    target = name + ":" + stream

    import dnf
    import dnf.lock
    from dnf.module.module_base import ModuleBase, STATE_ENABLED

    with dnf.Base() as base:
        base.conf.read()
        if str(base.conf.installroot) != "/":
            raise RuntimeError("DNF installroot is not the supported exact path")
        base.read_all_repos()
        if str(base.conf.persistdir) != PERSIST_DIRECTORY:
            raise RuntimeError("DNF persistdir is not the supported exact path")
        # Apply recomputes the accepted proof while serialized with every normal
        # DNF RPM transaction, retaining that lock through persistence and the
        # reopen proof.  Check remains strictly non-mutating because DNF's lock
        # creates a PID file.  DNF4's upstream module-only path does not take
        # this lock itself, so callers must also provide exclusive package-
        # manager administration; exact tree inventories detect overlapping
        # non-cooperating module writers rather than silently accepting them.
        if mode == "--apply":
            candidate_lock = dnf.lock.build_rpmdb_lock(PERSIST_DIRECTORY, True)
            candidate_lock.__enter__()
            rpmdb_lock = candidate_lock
        before_files = state_files()
        before_failsafe_files = failsafe_files()
        module_state_before_sha256 = inventory_digest(before_files)
        module_failsafe_before_sha256 = inventory_digest(before_failsafe_files)
        before_digest = combined_state_digest(before_files, before_failsafe_files)
        configured_repositories = enabled_repositories(base)
        if not configured_repositories:
            raise RuntimeError("DNF has no configured enabled repository")
        base.fill_sack_from_repos_in_cache(load_system_repo=True)
        loaded_repositories = enabled_repositories(base)
        if loaded_repositories != configured_repositories:
            raise RuntimeError(
                "DNF did not load every configured enabled repository from cache"
            )
        module_platform_id = effective_platform_id(base)
        repositories_sha256 = canonical_digest(loaded_repositories)
        container = base._moduleContainer
        baseline_changes = pending_changes(container)
        baseline_active_modules = active_modules(container)
        baseline_failsafe_expected = active_enabled_failsafe(container)
        module_base = ModuleBase(base)
        module_base.enable([target])
        observed_changes = pending_changes(container)
        changes = require_exact_enable(
            baseline_changes,
            observed_changes,
            stream,
        )
        active_target, requirements = target_evidence(
            module_base,
            container,
            target,
        )
        resolved_active_modules = active_modules(container)
        require_stable_active_modules(
            baseline_active_modules,
            resolved_active_modules,
            active_target,
        )
        resolved_failsafe_expected = active_enabled_failsafe(container)
        target_failsafe_filename, target_failsafe_yaml = (
            require_exact_failsafe_resolution(
                baseline_failsafe_expected,
                resolved_failsafe_expected,
                active_target,
            )
        )
        require_failsafe_baseline(
            before_failsafe_files,
            baseline_failsafe_expected,
            container,
            target_failsafe_filename,
        )
        active_modules_sha256 = canonical_digest(resolved_active_modules)
        unchanged_files = state_files()
        unchanged_failsafe_files = failsafe_files()
        if (
            before_files != unchanged_files
            or before_failsafe_files != unchanged_failsafe_files
        ):
            raise RuntimeError("module state changed during dependency preflight")
        preflight_payload = {
            "active_modules_count": len(resolved_active_modules),
            "active_modules_sha256": active_modules_sha256,
            "active_target": active_target,
            "applied": False,
            "changes": changes,
            "failsafe_changed_files": [],
            "failsafe_target": {
                "filename": target_failsafe_filename,
                "yaml_sha256": hashlib.sha256(target_failsafe_yaml).hexdigest(),
            },
            "module_changed_files": [],
            "module_failsafe_after_sha256": module_failsafe_before_sha256,
            "module_failsafe_before_sha256": module_failsafe_before_sha256,
            "module_platform_id": module_platform_id,
            "module_state_after_sha256": module_state_before_sha256,
            "module_state_before_sha256": module_state_before_sha256,
            "repositories_count": len(loaded_repositories),
            "repositories_sha256": repositories_sha256,
            "requirements": requirements,
            "schema": 2,
            "state_after_sha256": before_digest,
            "state_before_sha256": before_digest,
            "target": {"name": name, "stream": stream},
        }
        preflight_sha256 = proof_preflight_sha256(preflight_payload)
        if mode == "--apply":
            if supplied_preflight != preflight_sha256:
                raise RuntimeError(
                    "DNF module apply does not match the accepted preflight proof"
            )
            previous_umask = os.umask(0o022)
            try:
                container.save()
                container.updateFailSafeData()
            finally:
                os.umask(previous_umask)

    after_files = state_files()
    after_failsafe_files = failsafe_files()
    observed_changed_files = changed_files(before_files, after_files)
    observed_failsafe_changed_files = changed_files(
        before_failsafe_files,
        after_failsafe_files,
    )
    if mode == "--apply":
        if observed_changed_files != [TARGET_FILE]:
            raise RuntimeError(
                "DNF changed module state outside the exact target file"
            )
        require_exact_directory_identity(
            before_files,
            after_files,
            "module state",
        )
        require_target_file(after_files, stream)
        require_exact_failsafe_postcondition(
            before_failsafe_files,
            after_failsafe_files,
            target_failsafe_filename,
            target_failsafe_yaml,
        )
        with dnf.Base() as verify_base:
            verify_base.conf.read()
            if str(verify_base.conf.installroot) != "/":
                raise RuntimeError("DNF installroot changed during apply")
            verify_base.read_all_repos()
            if str(verify_base.conf.persistdir) != PERSIST_DIRECTORY:
                raise RuntimeError("DNF persistdir changed during apply")
            verify_configured_repositories = enabled_repositories(verify_base)
            verify_base.fill_sack_from_repos_in_cache(load_system_repo=True)
            verify_loaded_repositories = enabled_repositories(verify_base)
            if (
                verify_configured_repositories != configured_repositories
                or verify_loaded_repositories != loaded_repositories
                or effective_platform_id(verify_base) != module_platform_id
            ):
                raise RuntimeError(
                    "DNF repository or platform inventory changed during apply"
                )
            verify_container = verify_base._moduleContainer
            if (
                verify_container.getModuleState(TARGET_NAME) != STATE_ENABLED
                or str(verify_container.getEnabledStream(TARGET_NAME)) != stream
                or list(verify_container.getInstalledProfiles(TARGET_NAME))
            ):
                raise RuntimeError("persisted DNF module state failed exact re-observation")
            verify_module_base = ModuleBase(verify_base)
            observed_active_target, observed_requirements = target_evidence(
                verify_module_base,
                verify_container,
                target,
            )
            observed_active_modules = active_modules(verify_container)
            observed_failsafe_expected = active_enabled_failsafe(
                verify_container
            )
            if (
                observed_active_target != active_target
                or observed_requirements != requirements
                or observed_active_modules != resolved_active_modules
                or observed_failsafe_expected != resolved_failsafe_expected
            ):
                raise RuntimeError("persisted DNF module resolution changed")
        verified_files = state_files()
        verified_failsafe_files = failsafe_files()
        if (
            verified_files != after_files
            or verified_failsafe_files != after_failsafe_files
        ):
            raise RuntimeError("module state changed during postcondition proof")
    elif (
        after_files != before_files
        or after_failsafe_files != before_failsafe_files
    ):
        raise RuntimeError("DNF module check unexpectedly changed persistent state")

    module_state_after_sha256 = inventory_digest(after_files)
    module_failsafe_after_sha256 = inventory_digest(after_failsafe_files)
    payload = {
        "active_modules_count": len(resolved_active_modules),
        "active_modules_sha256": active_modules_sha256,
        "active_target": active_target,
        "applied": mode == "--apply",
        "changes": changes,
        "failsafe_changed_files": observed_failsafe_changed_files,
        "failsafe_target": {
            "filename": target_failsafe_filename,
            "yaml_sha256": hashlib.sha256(target_failsafe_yaml).hexdigest(),
        },
        "module_changed_files": observed_changed_files,
        "module_failsafe_after_sha256": module_failsafe_after_sha256,
        "module_failsafe_before_sha256": module_failsafe_before_sha256,
        "module_platform_id": module_platform_id,
        "module_state_after_sha256": module_state_after_sha256,
        "module_state_before_sha256": module_state_before_sha256,
        "preflight_sha256": preflight_sha256,
        "repositories_count": len(loaded_repositories),
        "repositories_sha256": repositories_sha256,
        "requirements": requirements,
        "schema": 2,
        "state_after_sha256": combined_state_digest(
            after_files,
            after_failsafe_files,
        ),
        "state_before_sha256": before_digest,
        "target": {"name": name, "stream": stream},
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(2)
finally:
    if rpmdb_lock is not None:
        rpmdb_lock.__exit__(None, None, None)
'''.strip()


@dataclass(frozen=True)
class DnfModuleIdentity:
    name: str
    stream: str
    version: str
    context: str
    architecture: str
    repository: str
    yaml_sha256: str


@dataclass(frozen=True)
class DnfModuleRequirement:
    name: str
    streams: tuple[str, ...]


@dataclass(frozen=True)
class DnfModuleEnableProof:
    applied: bool
    target: DnfModuleIdentity
    requirements: tuple[DnfModuleRequirement, ...]
    preflight_sha256: str
    active_modules_count: int
    active_modules_sha256: str
    repositories_count: int
    repositories_sha256: str
    module_platform_id: str
    failsafe_filename: str
    failsafe_yaml_sha256: str
    module_state_before_sha256: str
    module_state_after_sha256: str
    module_failsafe_before_sha256: str
    module_failsafe_after_sha256: str
    state_before_sha256: str
    state_after_sha256: str


def dnf_module_enable_command(
    *,
    apply: bool,
    stream: str,
    preflight_sha256: str | None = None,
) -> list[str]:
    if _MODULE_STREAM_PATTERN.fullmatch(stream) is None:
        raise ValueError("invalid exact NVIDIA DNF module stream")
    if not apply and preflight_sha256 is not None:
        raise ValueError("DNF module check cannot accept an apply binding")
    if (
        apply
        and preflight_sha256 is not None
        and _SHA256_PATTERN.fullmatch(preflight_sha256) is None
    ):
        raise ValueError("invalid DNF module preflight binding")
    command = [
        "python3",
        "-I",
        "-c",
        DNF_MODULE_ENABLE_SCRIPT,
        "--apply" if apply else "--check",
        DNF_MODULE_NAME,
        stream,
    ]
    if apply:
        command.append(preflight_sha256 or DNF_MODULE_UNBOUND_PREFLIGHT)
    return command


def parse_dnf_module_enable_proof(
    result: CommandResult,
    *,
    applied: bool,
    stream: str,
    preflight_sha256: str | None = None,
) -> DnfModuleEnableProof | None:
    """Parse the fixed script's bounded, exact module-state proof."""

    if (
        result.returncode != 0
        or len(result.stdout.encode("utf-8")) > _MAX_PROOF_BYTES
        or "[output truncated:" in result.stdout
        or "[output truncated:" in result.stderr
        or _MODULE_STREAM_PATTERN.fullmatch(stream) is None
    ):
        return None
    try:
        payload = json.loads(
            result.stdout,
            object_pairs_hook=_object_without_duplicates,
        )
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "active_modules_count",
        "active_modules_sha256",
        "active_target",
        "applied",
        "changes",
        "failsafe_changed_files",
        "failsafe_target",
        "module_changed_files",
        "module_failsafe_after_sha256",
        "module_failsafe_before_sha256",
        "module_platform_id",
        "module_state_after_sha256",
        "module_state_before_sha256",
        "preflight_sha256",
        "repositories_count",
        "repositories_sha256",
        "requirements",
        "schema",
        "state_after_sha256",
        "state_before_sha256",
        "target",
    }:
        return None
    if (
        type(payload["schema"]) is not int
        or payload["schema"] != 2
        or payload["applied"] is not applied
    ):
        return None
    target = payload["target"]
    if (
        not isinstance(target, dict)
        or target != {"name": DNF_MODULE_NAME, "stream": stream}
    ):
        return None
    expected_changes = {
        "disable": [],
        "enable": {DNF_MODULE_NAME: stream},
        "install_profiles": {},
        "remove_profiles": {},
        "reset": [],
        "switch": {},
    }
    if payload["changes"] != expected_changes:
        return None
    expected_module_changed = [DNF_MODULE_STATE_FILE] if applied else []
    if payload["module_changed_files"] != expected_module_changed:
        return None
    before_digest = payload["state_before_sha256"]
    after_digest = payload["state_after_sha256"]
    proof_binding = payload["preflight_sha256"]
    active_modules_count = payload["active_modules_count"]
    active_modules_sha256 = payload["active_modules_sha256"]
    repositories_count = payload["repositories_count"]
    repositories_sha256 = payload["repositories_sha256"]
    module_platform_id = payload["module_platform_id"]
    module_state_before = payload["module_state_before_sha256"]
    module_state_after = payload["module_state_after_sha256"]
    module_failsafe_before = payload["module_failsafe_before_sha256"]
    module_failsafe_after = payload["module_failsafe_after_sha256"]
    if (
        not isinstance(before_digest, str)
        or not isinstance(after_digest, str)
        or not isinstance(proof_binding, str)
        or not isinstance(active_modules_sha256, str)
        or not isinstance(repositories_sha256, str)
        or not isinstance(module_platform_id, str)
        or _MODULE_PLATFORM_PATTERN.fullmatch(module_platform_id) is None
        or _SHA256_PATTERN.fullmatch(before_digest) is None
        or _SHA256_PATTERN.fullmatch(after_digest) is None
        or _SHA256_PATTERN.fullmatch(proof_binding) is None
        or _SHA256_PATTERN.fullmatch(active_modules_sha256) is None
        or _SHA256_PATTERN.fullmatch(repositories_sha256) is None
        or not all(
            isinstance(digest, str)
            and _SHA256_PATTERN.fullmatch(digest) is not None
            for digest in (
                module_state_before,
                module_state_after,
                module_failsafe_before,
                module_failsafe_after,
            )
        )
        or type(active_modules_count) is not int
        or not 1 <= active_modules_count <= _MAX_ACTIVE_MODULES
        or type(repositories_count) is not int
        or not 1 <= repositories_count <= _MAX_REPOSITORIES
        or (before_digest == after_digest) is applied
        or (module_state_before == module_state_after) is applied
        or (module_failsafe_before == module_failsafe_after) is applied
        or (
            preflight_sha256 is not None
            and proof_binding != preflight_sha256
        )
    ):
        return None
    identities = _parse_identities(payload["active_target"], stream)
    requirements = _parse_requirements(payload["requirements"])
    if identities is None or len(identities) != 1 or requirements is None:
        return None
    failsafe_target = payload["failsafe_target"]
    expected_failsafe_filename = (
        f"{DNF_MODULE_NAME}:{stream}:{identities[0].architecture}.yaml"
    )
    if (
        not isinstance(failsafe_target, dict)
        or failsafe_target
        != {
            "filename": expected_failsafe_filename,
            "yaml_sha256": identities[0].yaml_sha256,
        }
        or payload["failsafe_changed_files"]
        != ([expected_failsafe_filename] if applied else [])
        or before_digest
        != _combined_state_digest(
            module_state_before,
            module_failsafe_before,
        )
        or after_digest
        != _combined_state_digest(
            module_state_after,
            module_failsafe_after,
        )
    ):
        return None
    if proof_binding != _proof_preflight_sha256(payload):
        return None
    return DnfModuleEnableProof(
        applied=applied,
        target=identities[0],
        requirements=tuple(requirements),
        preflight_sha256=proof_binding,
        active_modules_count=active_modules_count,
        active_modules_sha256=active_modules_sha256,
        repositories_count=repositories_count,
        repositories_sha256=repositories_sha256,
        module_platform_id=module_platform_id,
        failsafe_filename=expected_failsafe_filename,
        failsafe_yaml_sha256=identities[0].yaml_sha256,
        module_state_before_sha256=module_state_before,
        module_state_after_sha256=module_state_after,
        module_failsafe_before_sha256=module_failsafe_before,
        module_failsafe_after_sha256=module_failsafe_after,
        state_before_sha256=before_digest,
        state_after_sha256=after_digest,
    )


def _proof_preflight_sha256(payload: dict[str, Any]) -> str:
    normalized = {
        key: payload[key]
        for key in (
            "active_modules_count",
            "active_modules_sha256",
            "active_target",
            "changes",
            "failsafe_target",
            "module_failsafe_before_sha256",
            "module_platform_id",
            "module_state_before_sha256",
            "repositories_count",
            "repositories_sha256",
            "requirements",
            "schema",
            "state_before_sha256",
            "target",
        )
    }
    normalized.update(
        {
            "applied": False,
            "failsafe_changed_files": [],
            "module_changed_files": [],
            "module_failsafe_after_sha256": payload[
                "module_failsafe_before_sha256"
            ],
            "module_state_after_sha256": payload["module_state_before_sha256"],
            "state_after_sha256": payload["state_before_sha256"],
        }
    )
    encoded = json.dumps(
        normalized,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _combined_state_digest(module_digest: str, failsafe_digest: str) -> str:
    encoded = json.dumps(
        {
            "module_failsafe_sha256": failsafe_digest,
            "module_state_sha256": module_digest,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_identities(
    value: Any,
    stream: str,
) -> list[DnfModuleIdentity] | None:
    if not isinstance(value, list) or len(value) > 1:
        return None
    identities: list[DnfModuleIdentity] = []
    for record in value:
        if not isinstance(record, dict) or set(record) != {
            "architecture",
            "context",
            "name",
            "repository",
            "stream",
            "version",
            "yaml_sha256",
        }:
            return None
        fields = (
            record["architecture"],
            record["context"],
            record["name"],
            record["repository"],
            record["stream"],
            record["version"],
            record["yaml_sha256"],
        )
        if not all(isinstance(field, str) for field in fields):
            return None
        if (
            record["name"] != DNF_MODULE_NAME
            or record["stream"] != stream
            or any(
                _MODULE_FIELD_PATTERN.fullmatch(record[field]) is None
                for field in ("architecture", "context", "repository")
            )
            or _MODULE_VERSION_PATTERN.fullmatch(record["version"]) is None
            or _SHA256_PATTERN.fullmatch(record["yaml_sha256"]) is None
        ):
            return None
        identities.append(
            DnfModuleIdentity(
                name=record["name"],
                stream=record["stream"],
                version=record["version"],
                context=record["context"],
                architecture=record["architecture"],
                repository=record["repository"],
                yaml_sha256=record["yaml_sha256"],
            )
        )
    return identities


def _parse_requirements(value: Any) -> list[DnfModuleRequirement] | None:
    if not isinstance(value, list) or len(value) > _MAX_REQUIREMENTS:
        return None
    requirements: list[DnfModuleRequirement] = []
    previous: tuple[str, tuple[str, ...]] | None = None
    for record in value:
        if not isinstance(record, dict) or set(record) != {"name", "streams"}:
            return None
        name = record["name"]
        streams = record["streams"]
        if (
            not isinstance(name, str)
            or _MODULE_FIELD_PATTERN.fullmatch(name) is None
            or not isinstance(streams, list)
            or not streams
            or not all(
                isinstance(item, str)
                and _MODULE_FIELD_PATTERN.fullmatch(item) is not None
                for item in streams
            )
        ):
            return None
        normalized = (name, tuple(streams))
        if list(normalized[1]) != sorted(set(normalized[1])):
            return None
        if previous is not None and normalized <= previous:
            return None
        previous = normalized
        requirements.append(
            DnfModuleRequirement(name=name, streams=normalized[1])
        )
    return requirements
