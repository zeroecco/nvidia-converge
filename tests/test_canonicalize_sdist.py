from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import pytest

from scripts.canonicalize_sdist import canonicalize_sdist


def _write_archive(
    path: Path,
    *,
    gzip_mtime: int,
    member_mtime: int,
    reverse: bool,
) -> None:
    entries = [
        ("example-1.0/", None, 0o775),
        ("example-1.0/module.py", b"value = 1\n", 0o664),
        ("example-1.0/bin/tool", b"#!/bin/sh\n", 0o775),
    ]
    if reverse:
        entries.reverse()
    with path.open("wb") as raw, gzip.GzipFile(
        filename=path.name,
        mode="wb",
        fileobj=raw,
        mtime=gzip_mtime,
    ) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
        for name, payload, mode in entries:
            member = tarfile.TarInfo(name)
            member.mtime = member_mtime
            member.uid = 1000
            member.gid = 100
            member.uname = "builder"
            member.gname = "builders"
            member.mode = mode
            if payload is None:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def test_canonicalize_sdist_is_byte_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_archive(first, gzip_mtime=1, member_mtime=2, reverse=False)
    _write_archive(second, gzip_mtime=3, member_mtime=4, reverse=True)

    canonicalize_sdist(first, epoch=1_700_000_000)
    canonicalize_sdist(second, epoch=1_700_000_000)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, mode="r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == sorted(
        member.name for member in members
    )
    assert {member.mtime for member in members} == {1_700_000_000}
    assert {member.uid for member in members} == {0}
    assert {member.gid for member in members} == {0}
    modes = {member.name: member.mode for member in members}
    assert modes["example-1.0"] == 0o755
    assert modes["example-1.0/module.py"] == 0o644
    assert modes["example-1.0/bin/tool"] == 0o755


def test_canonicalize_sdist_rejects_link_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("example-1.0/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        archive.addfile(member)

    with pytest.raises(ValueError, match="unsupported.*member type"):
        canonicalize_sdist(archive_path, epoch=1)
