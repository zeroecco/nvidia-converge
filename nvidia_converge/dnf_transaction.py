from __future__ import annotations

DNF_LOCAL_TRANSACTION_SCRIPT = r"""
import hashlib
import json
import os
import re
import stat
import sys

MODULE_STATE_DIRECTORY = "/etc/dnf/modules.d"
MAX_STATE_FILES = 4096
MAX_DIRECTORY_ENTRIES = 8192
MAX_STATE_FILE_BYTES = 4 * 1024 * 1024
MAX_STATE_TOTAL_BYTES = 64 * 1024 * 1024
SAFE_MODULE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,254}\.module")
SAFE_FAILSAFE_FILE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}:"
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}:"
    r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}\.yaml"
)


def trusted_directory(path):
    if not path.startswith("/") or os.path.normpath(path) != path:
        raise RuntimeError("DNF state directory is not an exact absolute path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptor = os.open("/", flags)
    try:
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
                    "DNF state ancestor is not a trusted root-owned directory"
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
                raise RuntimeError("DNF state ancestor changed while opening")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def state_tree(path, filename_pattern):
    parent = os.path.dirname(path)
    leaf = os.path.basename(path)
    if (
        not leaf
        or not path.startswith("/")
        or os.path.normpath(path) != path
    ):
        raise RuntimeError("DNF state path is not an exact absolute path")
    parent_descriptor = trusted_directory(parent)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        try:
            metadata = os.stat(
                leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return {"exists": False, "records": {}}
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError("DNF state path is not a trusted root-owned directory")
        directory = os.open(
            leaf,
            directory_flags,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    try:
        opened_directory = os.fstat(directory)
        if (
            opened_directory.st_dev != metadata.st_dev
            or opened_directory.st_ino != metadata.st_ino
            or opened_directory.st_mode != metadata.st_mode
            or opened_directory.st_uid != metadata.st_uid
        ):
            raise RuntimeError("DNF state directory changed while opening")
        directory_names = os.listdir(directory)
        if len(directory_names) > MAX_DIRECTORY_ENTRIES:
            raise RuntimeError("DNF state directory inventory is too large")
        names = sorted(name for name in directory_names if name.endswith(
            ".module" if filename_pattern is SAFE_MODULE_FILE else ".yaml"
        ))
        if len(names) > MAX_STATE_FILES:
            raise RuntimeError("DNF state file inventory is too large")
        records = {}
        total = 0
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for name in names:
            if not filename_pattern.fullmatch(name):
                raise RuntimeError("DNF state contains an unsafe filename")
            file_metadata = os.stat(
                name,
                dir_fd=directory,
                follow_symlinks=False,
            )
            mode = stat.S_IMODE(file_metadata.st_mode)
            if (
                not stat.S_ISREG(file_metadata.st_mode)
                or file_metadata.st_uid != 0
                or mode & 0o022
                or file_metadata.st_nlink != 1
                or file_metadata.st_size < 0
                or file_metadata.st_size > MAX_STATE_FILE_BYTES
            ):
                raise RuntimeError(
                    "DNF state file is not a bounded root-controlled regular file"
                )
            total += file_metadata.st_size
            if total > MAX_STATE_TOTAL_BYTES:
                raise RuntimeError("DNF state file inventory is too large")
            descriptor = os.open(name, file_flags, dir_fd=directory)
            try:
                opened = os.fstat(descriptor)
                if (
                    opened.st_dev != file_metadata.st_dev
                    or opened.st_ino != file_metadata.st_ino
                    or opened.st_size != file_metadata.st_size
                    or opened.st_uid != file_metadata.st_uid
                    or opened.st_mode != file_metadata.st_mode
                    or opened.st_nlink != file_metadata.st_nlink
                ):
                    raise RuntimeError("DNF state file changed while opening")
                digest = hashlib.sha256()
                remaining = opened.st_size
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 64 * 1024))
                    if not chunk:
                        raise RuntimeError("DNF state file changed while reading")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if os.read(descriptor, 1):
                    raise RuntimeError("DNF state file changed while reading")
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
                    raise RuntimeError("DNF state file changed while reading")
            finally:
                os.close(descriptor)
            records[name] = {
                "ctime_ns": opened.st_ctime_ns,
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "mode": mode,
                "mtime_ns": opened.st_mtime_ns,
                "sha256": digest.hexdigest(),
                "size": opened.st_size,
            }
        if sorted(os.listdir(directory)) != sorted(directory_names):
            raise RuntimeError("DNF state directory changed during inventory")
        final_directory = os.fstat(directory)
        if (
            final_directory.st_dev != opened_directory.st_dev
            or final_directory.st_ino != opened_directory.st_ino
            or final_directory.st_mode != opened_directory.st_mode
            or final_directory.st_uid != opened_directory.st_uid
            or final_directory.st_mtime_ns != opened_directory.st_mtime_ns
            or final_directory.st_ctime_ns != opened_directory.st_ctime_ns
        ):
            raise RuntimeError("DNF state directory changed during inventory")
        return {
            "ctime_ns": opened_directory.st_ctime_ns,
            "device": opened_directory.st_dev,
            "exists": True,
            "inode": opened_directory.st_ino,
            "mode": stat.S_IMODE(opened_directory.st_mode),
            "mtime_ns": opened_directory.st_mtime_ns,
            "records": records,
        }
    finally:
        os.close(directory)


def module_pending_changes(container):
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


def require_no_module_changes(container):
    if module_pending_changes(container) != {
        "disable": [],
        "enable": {},
        "install_profiles": {},
        "remove_profiles": {},
        "reset": [],
        "switch": {},
    }:
        raise RuntimeError("local RPM transaction has a pending DNF module-state delta")
    is_changed = getattr(container, "isChanged", None)
    if not callable(is_changed) or is_changed() is not False:
        raise RuntimeError(
            "local RPM transaction has an authoritative DNF module-state delta"
        )


def do_transaction_without_failsafe_update(base):
    container = base._moduleContainer
    container_class = type(container)
    original = getattr(container_class, "updateFailSafeData", None)
    if not callable(original):
        raise RuntimeError(
            "this DNF4/libdnf binding cannot suppress module fail-safe updates"
        )
    calls = [0]

    def suppressed_update(_container):
        calls[0] += 1

    try:
        setattr(container_class, "updateFailSafeData", suppressed_update)
    except Exception as exc:
        raise RuntimeError(
            "this DNF4/libdnf binding cannot suppress module fail-safe updates"
        ) from exc
    transaction_error = None
    restoration_error = None
    try:
        bound = getattr(container, "updateFailSafeData", None)
        if (
            getattr(container_class, "updateFailSafeData", None)
            is not suppressed_update
            or not callable(bound)
            or getattr(bound, "__func__", None) is not suppressed_update
            or getattr(bound, "__self__", None) is not container
            or base._moduleContainer is not container
        ):
            raise RuntimeError(
                "DNF4/libdnf module fail-safe interception could not be verified"
            )
        base.do_transaction()
    except Exception as exc:
        transaction_error = exc
    finally:
        try:
            setattr(container_class, "updateFailSafeData", original)
            if getattr(container_class, "updateFailSafeData", None) is not original:
                raise RuntimeError("DNF4/libdnf fail-safe updater restoration failed")
        except Exception as exc:
            restoration_error = exc
    if restoration_error is not None:
        raise RuntimeError("DNF4/libdnf fail-safe updater restoration failed") from restoration_error
    if transaction_error is not None:
        raise transaction_error
    if calls[0] != 1:
        raise RuntimeError(
            "DNF transaction did not invoke the suppressed fail-safe updater exactly once"
        )


try:
    import dnf
    import hawkey
    import rpm

    mode = sys.argv[1]
    remove_marker = sys.argv.index("--remove", 2)
    install_marker = sys.argv.index("--expect-install", remove_marker + 1)
    expected_remove_marker = sys.argv.index("--expect-remove", install_marker + 1)
    if mode not in {"--check", "--apply"}:
        raise RuntimeError("invalid local RPM transaction mode")
    if sys.argv.count("--remove") != 1 or sys.argv.count("--expect-install") != 1 or sys.argv.count("--expect-remove") != 1:
        raise RuntimeError("ambiguous local RPM transaction markers")
    restore_paths = sys.argv[2:remove_marker]
    remove_specs = sys.argv[remove_marker + 1:install_marker]
    expected_installs = set(sys.argv[install_marker + 1:expected_remove_marker])
    expected_removals = set(sys.argv[expected_remove_marker + 1:])
    if len(restore_paths) != len(set(restore_paths)):
        raise RuntimeError("duplicate local RPM path")
    if len(remove_specs) != len(set(remove_specs)):
        raise RuntimeError("duplicate exact RPM removal")
    if len(expected_installs) != len(sys.argv[install_marker + 1:expected_remove_marker]):
        raise RuntimeError("duplicate expected RPM installation")
    if len(expected_removals) != len(sys.argv[expected_remove_marker + 1:]):
        raise RuntimeError("duplicate expected RPM removal")
    if not restore_paths and not remove_specs:
        raise RuntimeError("empty local RPM transaction")
    if any(not os.path.isabs(path) or "://" in path for path in restore_paths):
        raise RuntimeError("local RPM transaction contains a non-local path")
    before_module_state = state_tree(
        MODULE_STATE_DIRECTORY,
        SAFE_MODULE_FILE,
    )

    def nevra(package):
        epoch = str(package.epoch or "")
        epoch_prefix = "" if epoch in {"", "0", "None"} else epoch + ":"
        return (
            f"{package.name}-{epoch_prefix}{package.version}-"
            f"{package.release}.{package.arch}"
        )

    with dnf.Base() as base:
        base.conf.clean_requirements_on_remove = False
        base.conf.install_weak_deps = False
        base.conf.gpgcheck = True
        base.conf.localpkg_gpgcheck = True
        base.fill_sack(load_system_repo=True, load_available_repos=False)
        module_container = base._moduleContainer
        require_no_module_changes(module_container)
        persistdir = str(base.conf.persistdir)
        if (
            not persistdir.startswith("/")
            or os.path.normpath(persistdir) != persistdir
        ):
            raise RuntimeError("DNF persistdir is not an exact absolute path")
        failsafe_directory = os.path.join(persistdir, "modulefailsafe")
        before_failsafe_state = state_tree(
            failsafe_directory,
            SAFE_FAILSAFE_FILE,
        )
        local_packages = base.add_remote_rpms(restore_paths, strict=True)
        if len(local_packages) != len(restore_paths):
            raise RuntimeError("not every retained RPM was loaded locally")
        local_nevras = [nevra(package) for package in local_packages]
        if len(local_nevras) != len(set(local_nevras)):
            raise RuntimeError("duplicate retained RPM identity")
        if set(local_nevras) != expected_installs:
            raise RuntimeError("retained RPM headers do not match expected installs")
        installed_by_slot = {}
        for installed in base.sack.query().installed():
            installed_by_slot.setdefault(
                (str(installed.name), str(installed.arch)),
                [],
            ).append(installed)
        for package in local_packages:
            signature_result, signature_error = base.package_signature_check(package)
            if signature_result != 0:
                raise RuntimeError(
                    "retained RPM signature verification failed: "
                    + str(signature_error)
                )
            installed_slot = installed_by_slot.get(
                (str(package.name), str(package.arch)),
                [],
            )
            if not installed_slot:
                base.package_install(package)
                continue
            if len(installed_slot) != 1:
                raise RuntimeError("local RPM target slot is not singular")
            installed = installed_slot[0]
            comparison = rpm.labelCompare(
                (
                    str(package.epoch or "0"),
                    str(package.version),
                    str(package.release),
                ),
                (
                    str(installed.epoch or "0"),
                    str(installed.version),
                    str(installed.release),
                ),
            )
            if comparison < 0:
                base.package_downgrade(package, strict=True)
            elif comparison > 0:
                base.package_upgrade(package)
            else:
                package_reinstall = getattr(base, "package_reinstall", None)
                if not callable(package_reinstall):
                    raise RuntimeError(
                        "this DNF version cannot express an exact same-NEVRA reinstall"
                    )
                package_reinstall(package)
        for spec in remove_specs:
            base.remove(spec, forms=[hawkey.FORM_NEVRA])
        base.resolve(allow_erasing=False)
        require_no_module_changes(module_container)
        observed_installs = {nevra(package) for package in base.transaction.install_set}
        observed_removals = {nevra(package) for package in base.transaction.remove_set}
        if observed_installs != expected_installs:
            raise RuntimeError("resolved RPM install set changed")
        if observed_removals != expected_removals:
            raise RuntimeError("resolved RPM removal set changed")
        if state_tree(MODULE_STATE_DIRECTORY, SAFE_MODULE_FILE) != before_module_state:
            raise RuntimeError("DNF module state changed during local RPM preflight")
        if state_tree(failsafe_directory, SAFE_FAILSAFE_FILE) != before_failsafe_state:
            raise RuntimeError("DNF module fail-safe state changed during local RPM preflight")
        if mode == "--apply":
            do_transaction_without_failsafe_update(base)
        if state_tree(MODULE_STATE_DIRECTORY, SAFE_MODULE_FILE) != before_module_state:
            raise RuntimeError("local RPM transaction changed DNF module state")
        if state_tree(failsafe_directory, SAFE_FAILSAFE_FILE) != before_failsafe_state:
            raise RuntimeError("local RPM transaction changed DNF module fail-safe state")
        payload = {
            "install": sorted(observed_installs),
            "remove": sorted(observed_removals),
        }
    if state_tree(MODULE_STATE_DIRECTORY, SAFE_MODULE_FILE) != before_module_state:
        raise RuntimeError("DNF module state changed while closing local RPM transaction")
    if state_tree(failsafe_directory, SAFE_FAILSAFE_FILE) != before_failsafe_state:
        raise RuntimeError(
            "DNF module fail-safe state changed while closing local RPM transaction"
        )
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(2)
""".strip()


def dnf_local_transaction_command(
    *,
    apply: bool,
    restore_paths: list[str],
    remove_specs: list[str],
    expected_installs: list[str],
    expected_removals: list[str],
) -> list[str]:
    return [
        "python3",
        "-I",
        "-c",
        DNF_LOCAL_TRANSACTION_SCRIPT,
        "--apply" if apply else "--check",
        *restore_paths,
        "--remove",
        *remove_specs,
        "--expect-install",
        *expected_installs,
        "--expect-remove",
        *expected_removals,
    ]
