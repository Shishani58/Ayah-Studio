import json
from pathlib import Path
import sys

from tools import bench_full_windows as bench


def test_scoring_uses_selected_recording_and_correct_surah_ayah():
    item = {"surah": 80, "rows": [(1, 1100, .5), (2, 2000, .9)],
            "windows": {"windows": [{"end_ms": 1000}, {"end_ms": 3000}]}}
    bench.score_surahs([item], {(80, 1): 1000, (80, 2): 2100, (81, 1): 0})
    assert item["errors_ms"] == [100, -100]
    assert item["stats"]["median_ms"] == 100
    assert item["seam_errors_ms"] == [100, -100]


def test_case_process_receives_matching_reference_and_production_guards(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    reference = tmp_path / "other-reciter.db"
    monkeypatch.setattr(sys, "argv", ["bench_full_windows", "--cases", "strong:20:3:0",
                                     "--reference", str(reference), "--production-guards",
                                     "--output", str(output)])
    monkeypatch.setattr(bench, "sha256", lambda _: "test-sha")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        child = Path(command[command.index("--output") + 1])
        child.write_text(json.dumps({"case": "strong:20:3:0", "full_s": 1,
                                     "peak_rss_mib": 100, "errors": {}, "failures": []}))

    monkeypatch.setattr(bench.subprocess, "run", run)
    bench.main()
    assert len(commands) == 1
    assert commands[0][commands[0].index("--reference") + 1] == str(reference)
    assert "--production-guards" in commands[0]
    settings = json.loads(output.read_text())["settings"]
    assert settings["reference"] == str(reference) and settings["production_guards"] is True
