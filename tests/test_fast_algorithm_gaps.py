"""Recovery regressions: actual canonical ownership, synthetic acoustics."""

from dataclasses import replace

import numpy as np
import pytest

from surah_cutter.core.fast_executor import (
    Boundary,
    _noisy_opening,
    pause_geometry,
    plan_phrase,
    plan_seam,
    validate_boundaries,
)
from test_fast_executor import Fake, observation, pause, run


def test_missing_onset_requires_independent_additional_complete_word():
    groups = {1: tuple("abcdefghi"), 2: tuple("jklmnop")}
    ends = {1: [6, 9], 2: [3, 7]}
    assert _noisy_opening(observation(tuple("bcdef")), groups, ends, 1, 0.25) == (None, True)
    proof, expand = _noisy_opening(observation(tuple("bcdefghi")), groups, ends, 1, 0.25)
    assert expand and proof["phrase"] == groups[1]
    assert not proof["opening_recovery"]["onset_supported"]
    for bad in (
        observation(tuple("xcdefghi")),
        observation(tuple("bcdefwrong")),
        observation(tuple("bcdefWRONGghi")),
        observation(tuple("bcdefgh")),
        observation(tuple("bcdefghi"), quality=0.2),
        replace(observation(tuple("bcdefghi")), truncated=True),
    ):
        assert _noisy_opening(bad, groups, ends, 1, 0.25)[0] is None
    # A strong opening and a weak distinguishing extra word cannot be averaged.
    weak = replace(observation(tuple("bcdefghi")), unit_quality=(0.99,) * 5 + (0.1,) * 3)
    assert _noisy_opening(weak, groups, ends, 1, 0.25)[0] is None


@pytest.mark.parametrize(
    "other,ends2", [("xbcdefuvw", [6, 9]), ("zzzabcdefuvw", [3, 9, 12]), ("bcdefuvw", [5, 8])]
)
def test_missing_onset_checks_wrong_owner_and_internal_starts(other, ends2):
    groups = {1: tuple("abcdefghi"), 2: tuple(other)}
    assert _noisy_opening(
        observation(tuple("bcdefghi")), groups, {1: [6, 9], 2: ends2}, 1, 0.25
    ) == (None, False)


def test_missing_onset_executor_expands_once_then_validates_identity_only():
    fake = Fake(lambda *a: observation(()))
    fake.metric = "ctc_greedy_probability"
    rendered = list(fake.rendered)
    rendered[0], rendered[1] = tuple("abcdef"), tuple("ghi")
    fake.rendered = tuple(rendered)
    fake.responder = lambda f, s, d: (
        observation(tuple("bcdef" if d < 1800 else "bcdefghi")) if s == 1000 else observation(())
    )
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][1] == 1100 and rows[0][2] > 0
    assert trace["boundaries"][0]["proof"]["opening_recovery"]["missing_initial_units"] == 1
    assert sum(r["reason"].startswith("ayah=1:pause") for r in trace["acoustic_reads"]) == 2
    assert trace["confidence_semantics"]["calibrated_probability"] is False
    groups, ends = {}, {}
    for w, units in zip(fake.words, fake.rendered):
        groups[w.ayah] = groups.get(w.ayah, ()) + units
        ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
    boundary = Boundary(**trace["boundaries"][0])
    corrupt = {
        **boundary.proof,
        "opening_recovery": {**boundary.proof["opening_recovery"], "onset_supported": True},
    }
    with pytest.raises(ValueError, match="unheard_noisy_opening_identity"):
        validate_boundaries(
            [replace(boundary, proof=corrupt)],
            [1],
            40000,
            trace["acoustic_reads"],
            groups=groups,
            word_ends=ends,
            threshold=0.25,
        )
    with pytest.raises(ValueError, match="unsupported_noisy_opening_proof"):
        validate_boundaries(
            [replace(boundary, evidence_kind="wasl")],
            [1],
            40000,
            trace["acoustic_reads"],
            groups=groups,
            word_ends=ends,
            threshold=0.25,
        )


def contextual_fake(tail_kind="correct"):
    fake = Fake(lambda *a: observation(()))
    fake.timing = False
    rendered = list(fake.rendered)
    for ayah, label in ((1, "first"), (2, "shared"), (3, "shared"), (4, "last")):
        i = next(i for i, w in enumerate(fake.words) if w.ayah == ayah)
        rendered[i] = (label,)
    fake.rendered = tuple(rendered)
    groups, ends = {}, {}
    for w, u in zip(fake.words, fake.rendered):
        groups[w.ayah] = groups.get(w.ayah, ()) + u
        ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
    assert not plan_phrase(groups, 2, ends)["ambiguous"]
    assert len(plan_seam(groups, 2, ends)["units"]) < len(plan_phrase(groups, 2, ends)["units"])

    def respond(f, s, d):
        if s == 1000:
            return observation(("first",), positions=False)
        if s == 6000:
            return observation(("shared",), positions=False)
        if s + d == 6000:
            if tail_kind == "wrong":
                return observation(("wrong",), positions=False)
            obs = observation(
                groups[1], positions=False, quality=0.59 if tail_kind == "weak" else 0.71
            )
            return replace(obs, truncated=True) if tail_kind == "truncated" else obs
        return observation((), positions=False)

    fake.responder = respond
    return fake


def test_incomplete_distinct_opening_uses_shorter_pause_seam_without_timing():
    rows, trace = run(contextual_fake(), [pause(1000), pause(6000)])
    assert rows[1][1] == 6100 and rows[1][2] == pytest.approx(0.71)
    proof = trace["boundaries"][1]["proof"]
    assert proof["contextual_owner"]
    assert proof["previous_context_read_id"] != trace["boundaries"][1]["read_id"]
    assert trace["acoustic_reads"][proof["previous_context_read_id"]]["window_ms"][1] == 6000
    assert any(
        e.get("reason") == "incomplete_opening_contextual_pause" for e in trace["transitions"]
    )


@pytest.mark.parametrize("kind", ["wrong", "weak", "truncated"])
def test_shorter_pause_seam_requires_previous_tail(kind):
    rows, trace = run(contextual_fake(kind), [pause(1000), pause(6000)])
    assert rows[1][2] == 0
    assert any("previous_tail" in r["reason"] for r in trace["acoustic_reads"])


@pytest.mark.parametrize("width", [200, 860, 1500, 3000])
def test_geometry_checks_short_and_long_quiet_and_decay_controls(width):
    interval = pause(1000, width).pause
    pcm = np.full(16000 * 6, 0.001, np.float32)
    quiet = pause_geometry(pcm, 16000, interval)
    assert quiet["accepted"] and quiet["temporal_probability"] is None
    left, right = 16000, 16000 + width * 16
    pcm[left:right] = 0.04 * np.exp(-np.arange(right - left) / (width * 4))
    assert not pause_geometry(pcm, 16000, interval)["accepted"]


def test_unmeasurable_geometry_never_silently_passes():
    assert not pause_geometry(np.zeros(32000, np.float32), 16000, pause(1000, 100).pause)[
        "accepted"
    ]


def test_repeated_contextual_seam_cannot_prove_owner():
    groups = {1: ("tail",), 2: ("shared", "lateA"), 3: ("tail",), 4: ("shared", "lateB")}
    ends = {a: list(range(1, len(u) + 1)) for a, u in groups.items()}
    seam = plan_seam(groups, 2, ends)
    assert seam is not None
    # tail+shared is repeated; the planner must retain the distinguishing lateA.
    assert seam["units"] == ("shared", "lateA")
    assert len(seam["units"]) > 1


def test_short_intervening_word_quality_is_essential_too():
    groups = {1: tuple("abcdefxyghi"), 2: tuple("jklmnop")}
    ends = {1: [6, 8, 11], 2: [3, 7]}
    obs = observation(tuple("bcdefxyghi"))
    assert _noisy_opening(obs, groups, ends, 1, 0.25)[0] is not None
    weak = replace(obs, unit_quality=(0.99,) * 5 + (0.1,) * 2 + (0.99,) * 3)
    assert _noisy_opening(weak, groups, ends, 1, 0.25)[0] is None


@pytest.mark.parametrize("kind", ["weak", "wrong", "truncated"])
def test_unproved_alternative_tail_preserves_original_opening_expansion(kind):
    fake = contextual_fake(kind)
    original = fake.responder

    def respond(f, start, duration):
        if start == 6000 and duration > 2000:
            return observation(f.phrase(2), quality=0.99, positions=False)
        return original(f, start, duration)

    fake.responder = respond
    rows, trace = run(fake, [pause(1000), pause(6000), pause(18000)])
    assert rows[1] == (2, 6100, 0.99)
    proof = trace["boundaries"][1]["proof"]
    assert not proof.get("contextual_owner")
    read_id = trace["boundaries"][1]["read_id"]
    assert trace["acoustic_reads"][read_id]["window_ms"] == [6000.0, 8700.0]
    assert any(
        e.get("reason") == "alternative_context_unproved_resume_opening"
        for e in trace["transitions"]
    )
    # Expansion stays at the original pause, with each failed tail charged.
    tail_reads = [r for r in trace["acoustic_reads"] if "previous_tail" in r["reason"]]
    assert len(tail_reads) == 3
    assert all(r["window_ms"][1] == 6000.0 for r in tail_reads)
    assert not any(
        r["reason"].startswith("ayah=2:pause") and r["window_ms"][0] == 18000
        for r in trace["acoustic_reads"]
    )


def test_failed_alternative_tail_does_not_replace_future_opening_provenance():
    from surah_cutter.core.fast_executor import Policy

    fake = contextual_fake("weak")
    original = fake.responder

    def respond(f, start, duration):
        if start == 6000 and duration > 2000:
            return observation(f.phrase(3), quality=0.99, positions=False)
        return original(f, start, duration)

    fake.responder = respond
    rows, trace = run(fake, [pause(1000), pause(6000)], policy=Policy(sparse_anchors=True))
    assert rows[1][2] == 0 and rows[2] == (3, 6100, 0.99)
    anchor = trace["boundaries"][2]
    read = trace["acoustic_reads"][anchor["read_id"]]
    assert read["window_ms"] == [6000.0, 8700.0]
    assert "previous_tail" not in read["reason"]
    assert anchor["proof"]["sparse_anchor"]


def test_correct_tail_cannot_hide_strong_future_continuation():
    fake = contextual_fake("correct")
    original = fake.responder
    fake.responder = lambda f, s, d: (
        observation(f.phrase(3), positions=False) if s == 6000 else original(f, s, d)
    )
    rows, trace = run(fake, [pause(1000), pause(6000)])
    assert rows[1][2] == 0
    assert not trace["boundaries"][1]["proof"]
    assert not any("previous_tail" in r["reason"] for r in trace["acoustic_reads"])


@pytest.mark.parametrize("quality", [0.99, 0.59])
def test_shorter_seam_distinguishes_strong_wrong_and_weak_continuation(quality):
    fake = contextual_fake("correct")
    original = fake.responder

    def respond(f, s, d):
        if s == 6000:
            return replace(
                observation(("shared", "wrong"), positions=False), unit_quality=(0.99, quality)
            )
        return original(f, s, d)

    fake.responder = respond
    rows, trace = run(fake, [pause(1000), pause(6000)])
    assert (rows[1][2] > 0) == (quality < 0.6)


def test_contextual_validator_rejects_hidden_strong_wrong_continuation():
    from copy import deepcopy

    fake = contextual_fake("correct")
    rows, trace = run(fake, [pause(1000), pause(6000)])
    groups, ends = {}, {}
    for w, u in zip(fake.words, fake.rendered):
        groups[w.ayah] = groups.get(w.ayah, ()) + u
        ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
    boundary = Boundary(**trace["boundaries"][1])
    reads = deepcopy(trace["acoustic_reads"])
    observed = reads[boundary.read_id]["observed"]
    observed["units"] = ("shared", "wrong")
    observed["unit_quality"] = (0.99, 0.99)
    with pytest.raises(ValueError, match="contradictory_contextual_continuation"):
        validate_boundaries(
            [boundary], [2], 40000, reads, groups=groups, word_ends=ends, threshold=0.6
        )
