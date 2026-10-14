"""Race regressions execute the actual browser functions through Node VM."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_browser_async_races():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for browser JavaScript race regressions")
    subprocess.run([node, str(Path(__file__).with_name("webgui_async.cjs"))],
                   check=True, timeout=20)
