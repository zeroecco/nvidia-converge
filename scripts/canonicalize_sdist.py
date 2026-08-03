"""Canonicalize a locally built source distribution for byte-stable release use."""

from __future__ import annotations

import argparse
import gzip
import os
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import IO


def _validate_member(member: tarfile.TarInfo, root: str | None) -> str:
    name = member.name
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe source-distribution member: {name!r}")
    if not (member.isdir() or member.isfile()):
        raise ValueError(
            f"unsupported source-distribution member type: {name!r}"
        )
    member_root = path.parts[0]
    if root is not None and member_root != root:
        raise ValueError("source distribution has more than one top-level root")
    return member_root


def _canonical_member(member: tarfile.TarInfo, epoch: int) -> tarfile.TarInfo:
    canonical = tarfile.TarInfo(member.name)
    canonical.type = member.type
    canonical.size = member.size if member.isfile() else 0
    canonical.mode = (
        0o755
        if member.isdir() or (member.isfile() and member.mode & 0o111)
        else 0o644
    )
    canonical.uid = 0
    canonical.gid = 0
    canonical.uname = ""
    canonical.gname = ""
    canonical.mtime = epoch
    canonical.pax_headers = {}
    return canonical


def _write_canonical_archive(
    source: tarfile.TarFile,
    members: Sequence[tarfile.TarInfo],
    output: IO[bytes],
    epoch: int,
) -> None:
    with gzip.GzipFile(
        filename="", mode="wb", compresslevel=9, fileobj=output, mtime=epoch
    ) as compressed, tarfile.open(
        fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
    ) as destination:
        for member in sorted(members, key=lambda item: item.name):
            canonical = _canonical_member(member, epoch)
            payload = source.extractfile(member) if member.isfile() else None
            if member.isfile() and payload is None:
                raise ValueError(
                    f"could not read source-distribution member: {member.name!r}"
                )
            try:
                destination.addfile(canonical, payload)
            finally:
                if payload is not None:
                    payload.close()


def canonicalize_sdist(path: Path, *, epoch: int) -> None:
    """Rewrite *path* in place with stable ordering, ownership, modes, and times."""

    if epoch < 0:
        raise ValueError("source epoch must be non-negative")
    stat_result = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise ValueError("source distribution must be a regular nonsymlink file")

    temporary_path: Path | None = None
    try:
        with tarfile.open(path, mode="r:gz") as source:
            members = source.getmembers()
            if not members:
                raise ValueError("source distribution is empty")
            root: str | None = None
            names: set[str] = set()
            for member in members:
                root = _validate_member(member, root)
                if member.name in names:
                    raise ValueError(
                        f"duplicate source-distribution member: {member.name!r}"
                    )
                names.add(member.name)

            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                _write_canonical_archive(source, members, output.file, epoch)
                output.flush()
                os.fsync(output.fileno())

        os.chmod(temporary_path, stat_result.st_mode & 0o777)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    canonicalize_sdist(args.archive, epoch=args.source_date_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
