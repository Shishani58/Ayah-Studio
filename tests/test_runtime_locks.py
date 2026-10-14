"""Ensure shipped optional runtime locks stay tied to their declared inputs."""

import re
from pathlib import Path

from surah_cutter.core import model_runtime

LOCKS = Path(model_runtime.__file__).with_name("runtime_locks")


def _name(requirement: str) -> str:
    return re.split(r"\[|==", requirement, maxsplit=1)[0].replace("_", "-").lower()


def test_every_runtime_input_is_locked_with_distribution_hashes():
    for runtime, packages in model_runtime._PACKAGES.items():
        declared = (LOCKS / f"{runtime}.in").read_text().splitlines()
        assert declared == packages
        content = (LOCKS / f"{runtime}.txt").read_text()
        entries = re.split(r"(?=^[A-Za-z0-9][A-Za-z0-9_.-]*==)", content, flags=re.M)
        pinned: dict[str, str] = {}
        for entry in entries[1:]:
            first = entry.splitlines()[0]
            assert re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*==[^\s;]+", first)
            assert "--hash=sha256:" in entry
            pinned[_name(first)] = first
        assert {_name(package) for package in packages} <= pinned.keys()
        for package in packages:
            assert package.split("==", 1)[1] in pinned[_name(package)]
        assert "--hash=sha256:" in content


def test_universal_locks_include_platform_markers():
    for runtime in model_runtime._PACKAGES:
        content = (LOCKS / f"{runtime}.txt").read_text()
        assert "sys_platform == 'linux'" in content
        assert "sys_platform == 'win32'" in content
