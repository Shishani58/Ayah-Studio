from types import SimpleNamespace

from tools import bench_fast
from tools.bench_fast import pause_contract_violations, sequence_contract_violations


def _add_onset_proofs(rows, trace):
    trace["total_ms"] = 10000
    trace["acoustic_reads"] = []
    trace["boundaries"] = []
    for ayah, start_ms, confidence in rows:
        match = next((m for m in trace["pause_matches"] if m["ayah"] == ayah), {})
        evidence = match.get("boundary_evidence")
        if evidence not in {"wasl", "first_onset"}:
            continue
        read_id = len(trace["acoustic_reads"])
        trace["acoustic_reads"].append({
            "outcome": "completed", "window_ms": [0, 5000],
            "observed": {"units": ["tail", "start"], "positions_ms": [0, start_ms],
                         "unit_quality": [confidence, confidence], "truncated": False},
        })
        trace["boundaries"].append({
            "ayah": ayah, "start_ms": start_ms, "confidence": confidence,
            "evidence_kind": evidence, "read_id": read_id,
            "proof": {"complete": True, "competitors_checked": True, "phrase": ["start"],
                      "offset": 1, "previous_tail": ["tail"]},
        })


def test_select_suras_keeps_medium_and_full_corpus_presets_distinct():
    assert bench_fast.select_suras(["long"])[0] == "078"
    assert bench_fast.select_suras(["all"])[0] == "078"
    assert bench_fast.select_suras(["first77"]) == [f"{n:03d}" for n in range(1, 78)]
    corpus = bench_fast.select_suras(["corpus"])
    assert corpus[0] == "001"
    assert corpus[-1] == "114"
    assert len(corpus) == 114


def test_extract_ayah_audio_dir_keeps_regular_benchmark_args():
    assert bench_fast._extract_ayah_audio_dir([
        "026",
        "fast-only",
        "--ayah-audio-dir=/clips",
        "--audit-sequence",
    ]) == (["026", "fast-only", "--audit-sequence"], "/clips")


def test_benchmark_forwards_exact_clip_directory(tmp_path, monkeypatch):
    (tmp_path / "001.mp3").touch()
    seen: dict[str, object] = {}
    monkeypatch.setattr(bench_fast, "AUDIO", tmp_path)
    monkeypatch.setattr(bench_fast, "REF", {(1, 1): 0})
    monkeypatch.setattr(bench_fast.align, "_get_onnx_bundle", lambda: object())
    monkeypatch.setattr(bench_fast.decode, "decode", lambda _path: object())
    monkeypatch.setattr(
        bench_fast.quran_text,
        "surah_words",
        lambda _surah: [SimpleNamespace(ayah=1)],
    )

    def fake_fast(_pcm, _sr, _words, **kwargs):
        seen["ayah_audio_dir"] = kwargs["ayah_audio_dir"]
        kwargs["trace"].update(total_ms=1_000, n_pause=0, n_wasl=0, coverage=1.0)
        return [(1, 0, 0.9)]

    monkeypatch.setattr(bench_fast.align_fast, "align_surah_fast", fake_fast)

    assert bench_fast.main(["001", "fast-only", "--ayah-audio-dir", "/clips"]) == 0
    assert seen["ayah_audio_dir"] == "/clips"


def test_pause_contract_accepts_real_pause_and_explicit_wasl():
    rows = [(2, 1200, 0.8), (3, 2400, 0.7)]
    trace = {
        "pause_matches": [
            {
                "ayah": 2,
                "from_pause": True,
                "boundary_evidence": "pause",
                "pause_start_ms": 1100,
                "pause_end_ms": 1300,
            },
            {"ayah": 3, "from_pause": False, "boundary_evidence": "wasl"},
        ]
    }

    _add_onset_proofs(rows, trace)
    assert pause_contract_violations(rows, trace) == []


def test_pause_contract_rejects_off_center_owner_and_disguised_wasl_pause():
    rows = [(2, 1110, 0.8), (3, 2400, 0.7)]
    trace = {
        "phase1_pauses": [{"center_ms": 1200}, {"center_ms": 2400}],
        "pause_matches": [
            {
                "ayah": 2,
                "from_pause": True,
                "boundary_evidence": "pause",
                "pause_start_ms": 1100,
                "pause_end_ms": 1300,
            },
            {"ayah": 3, "from_pause": False, "boundary_evidence": "wasl"},
        ],
    }

    _add_onset_proofs(rows, trace)
    assert pause_contract_violations(rows, trace) == [
        "ayah 2: pause-backed start 1110 is not centre 1200 of 1100-1300",
        "ayah 3: wasl start 2400 equals a detected pause centre",
    ]


def test_pause_contract_ignores_rejected_non_owner_pause_center():
    rows = [(2, 2400, 0.7)]
    trace = {
        "phase1_pauses": [{"center_ms": 2400, "quality": "rejected"}],
        "pause_matches": [
            {"ayah": 2, "from_pause": False, "boundary_evidence": "wasl"},
        ],
    }

    _add_onset_proofs(rows, trace)
    assert pause_contract_violations(rows, trace) == []


def test_pause_contract_rejects_legacy_fingerprint_exemption():
    trace = {"pause_matches": [{
        "ayah": 2, "from_pause": True, "boundary_evidence": "exact_fingerprint",
    }]}
    assert pause_contract_violations([(2, 1350, 0.95)], trace)


def test_pause_contract_rejects_stale_pause_and_unclassified_non_pause_start():
    rows = [(2, 1500, 0.8), (3, 2400, 0.7)]
    trace = {
        "pause_matches": [
            {
                "ayah": 2,
                "from_pause": True,
                "boundary_evidence": "pause",
                "pause_start_ms": 1100,
                "pause_end_ms": 1300,
            },
            {"ayah": 3, "from_pause": False, "boundary_evidence": "recovered"},
        ]
    }

    _add_onset_proofs(rows, trace)
    assert pause_contract_violations(rows, trace) == [
        "ayah 2: start 1500 outside pause 1100-1300",
        "ayah 3: non-pause start 2400 has unknown evidence 'recovered'",
    ]


def test_sequence_contract_rejects_missing_ayah_and_equal_starts():
    assert sequence_contract_violations(
        [(1, 0, 0.9), (2, 100, 0.8), (4, 100, 0.7)],
        [1, 2, 3, 4],
    ) == [
        "ayah sequence [1, 2, 4]… does not equal expected [1, 2, 3, 4]…",
        "ayah 2->4: non-increasing start 100->100",
    ]


def test_outlier_diagnoses_classifies_weak_recovered_pause(monkeypatch):
    monkeypatch.setattr(bench_fast, "REF", {(84, 10): 53_417})
    rows = [(10, 54_320, 0.37)]
    trace = {
        "pause_matches": [
            {
                "ayah": 10,
                "from_pause": True,
                "boundary_evidence": "pause",
                "match_action": "recovered",
                "pause_sim": 0.37,
                "pause_start_ms": 53_840,
                "pause_end_ms": 54_800,
            }
        ]
    }

    assert bench_fast.outlier_diagnoses(rows, trace, 84) == [
        "ayah 10: error=903ms cause=weak-recovered-pause-needs-edge-analysis "
        "action=recovered sim=0.37 timing=None pause=53840-54800 start/ref=54320/53417"
    ]


def test_candidate_diagnoses_reports_ranked_pauses_near_reference(monkeypatch):
    monkeypatch.setattr(bench_fast, "REF", {(2, 30): 581_328})
    rows = [(30, 570_300, 0.3)]
    trace = {
        "anchor_first": {
            "pause_rankings": [{
                "pause_ms": 581_250.0,
                "accepted": False,
                "rejection_reasons": ["low_margin"],
                "candidates": [
                    {"ayah": 30, "similarity": 0.61, "prefix_chars": 5},
                    {"ayah": 31, "similarity": 0.58, "prefix_chars": 2},
                ],
            }]
        }
    }

    assert bench_fast.candidate_diagnoses(rows, trace, 2) == [
        "ayah 30: result/ref=570300/581328 pause=581250 accepted=False "
        "reasons=low_margin top=[30:0.610/p5,31:0.580/p2]"
    ]


def test_estimate_boundary_requires_zero_confidence_and_review():
    trace = {"review_required_ayahs": [2], "pause_matches": [
        {"ayah": 2, "from_pause": False, "match_action": "insufficient_audio",
         "boundary_evidence": "estimate"},
    ]}
    assert pause_contract_violations([(2, 1000, 0.0)], trace) == []
    assert pause_contract_violations([(2, 1000, 0.9)], trace)
    trace["review_required_ayahs"] = []
    assert pause_contract_violations([(2, 1000, 0.0)], trace)


def test_first_onset_requires_a_matching_completed_local_read():
    rows = [(1, 500, 0.9)]
    trace = {"pause_matches": [{"ayah": 1, "from_pause": False, "boundary_evidence": "first_onset"}]}
    _add_onset_proofs(rows, trace)
    assert pause_contract_violations(rows, trace) == []
    trace["acoustic_reads"][0]["observed"]["positions_ms"][1] += 100
    assert "onset_proof_mismatch" in pause_contract_violations(rows, trace)[0]
    trace["acoustic_reads"][0]["outcome"] = "failed"
    assert "incomplete_acoustic_proof" in pause_contract_violations(rows, trace)[0]


def test_wasl_evidence_label_alone_is_not_proof():
    rows = [(2, 1000, 0.9)]
    trace = {"pause_matches": [{"ayah": 2, "from_pause": False, "boundary_evidence": "wasl"}]}
    assert pause_contract_violations(rows, trace)
    _add_onset_proofs(rows, trace)
    trace["boundaries"][0]["proof"]["previous_tail"] = []
    assert "unidentified_wasl_context" in pause_contract_violations(rows, trace)[0]


def test_pause_contract_requires_one_evidence_row_per_output():
    assert pause_contract_violations([(1, 0, 0.0)], {}) == ["ayah 1: output has no boundary evidence"]
    match = {"ayah": 1, "from_pause": False, "boundary_evidence": "estimate"}
    assert "duplicate boundary evidence" in pause_contract_violations(
        [(1, 0, 0.0)], {"review_required_ayahs": [1], "pause_matches": [match, match]},
    )
