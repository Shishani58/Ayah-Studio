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


def test_player_preferences_constant_speed_and_navigation():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for player regressions")
    subprocess.run([node, str(Path(__file__).with_name("player_preferences.cjs"))],
                   check=True, timeout=20)


def test_finite_audio_preview_ranges_and_clock():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for player regressions")
    subprocess.run([node, str(Path(__file__).with_name("playback_ranges.cjs"))],
                   check=True, timeout=20)
