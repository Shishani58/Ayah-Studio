from surah_cutter.webgui.server import _fast_trace_summary


def test_fast_trace_summary_exposes_forecast_and_atomic_fallbacks():
    summary = _fast_trace_summary(
        {
            "ayahs": [
                {"ayah": 1, "reliable": True},
                {"ayah": 2, "repaired": True},
            ],
            "tempo_forecast": {
                "rate_units_per_ms": 0.12,
                "confidence": 0.8,
                "observation_count": 3,
            },
            "pause_sequence_events": [
                {"action": "unresolved_current", "ayah": 2},
                {"action": "unresolved_current", "ayah": 2},
            ],
            "model_reads": [
                {"phase": "wasl-run", "atomic": True},
                {"phase": "wasl-tiled-fallback", "atomic": True},
                {"phase": "wasl-run", "atomic": False},
            ],
        }
    )

    assert summary["tempo_rate"] == 0.12
    assert summary["tempo_confidence"] == 0.8
    assert summary["tempo_observations"] == 3
    assert summary["unresolved"] == 1
    assert summary["atomic_wasl_reads"] == 2
