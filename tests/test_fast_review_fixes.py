"""Regressions for acoustic evidence/ownership, not a substitute for audio QA."""
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pytest

from surah_cutter.core.fast_adapters import Observation, attributed_asr_units, arabic_units, TarteelAdapter
from surah_cutter.core.fast_ctc import independent_beam, prefix_scores
from surah_cutter.core.fast_executor import _beam_view, plan_seam, plan_phrase
from surah_cutter.data import quran_text
from test_fast_executor import Fake, observation, pause, run


def asr_result(words, probabilities):
    tokens = [{"id": i, "bytes": list((" " if i else "").encode() + word.encode()),
               "log_probability": math.log(p)} for i, (word, p) in enumerate(zip(words, probabilities))]
    return {"text": " ".join(words), "selected_tokens": tokens}


def test_strong_proof_word_not_poisoned_by_unneeded_weak_tail():
    units, quality, attribution = attributed_asr_units(asr_result(["قل", "هو", "الله", "احد"], [.99, .98, .97, .1]))
    assert units == ("قل", "هو", "الله", "احد")
    assert quality == pytest.approx([.99, .98, .97, .1])
    assert attribution[0]["token_ids"] == [0]


def test_utf8_split_and_token_spanning_words_have_all_owners():
    text = "قُلْ هُوَ"
    raw = text.encode()
    # One token ends inside Arabic UTF-8; a later token includes the space.
    tokens = [{"id": i, "bytes": list(part), "log_probability": math.log(p)}
              for i, (part, p) in enumerate(zip([raw[:1], raw[1:6], raw[6:]], [.95, .2, .99]))]
    units, quality, attribution = attributed_asr_units({"text": text, "selected_tokens": tokens})
    assert units == ("قل", "هو")
    assert quality[0] == pytest.approx(.2)
    assert quality[1] == pytest.approx(.99)
    assert attribution[0]["selected_indices"] == [0, 1, 2]


def test_missing_or_non_reversible_mapping_is_review():
    result = asr_result(["قل"], [.99])
    result["text"] = "هو"
    assert attributed_asr_units(result)[1] == (0.,)
    assert attributed_asr_units({"text": "قل"})[1] == (0.,)


@pytest.mark.parametrize("surah,expected", [(2, ("الف", "لام", "ميم")), (19, ("كاف", "ها", "يا", "عين", "صاد")),
    (20, ("طا", "ها")), (36, ("يا", "سين")), (50, ("قاف",))])
def test_spoken_letter_units_retain_canonical_word(surah, expected):
    adapter = TarteelAdapter(SimpleNamespace(emitter=SimpleNamespace(model_id="whisper-tarteel-tiny",
                                                                    runtime_identity={"test": True})))
    words = quran_text.surah_words(surah)
    rendered = adapter.represent(words)
    assert rendered[0] == expected and len(rendered) == len(words)


def test_noisy_first_word_uses_exact_later_anchor_for_finite_expansion():
    fake = Fake(lambda *args: observation(()))
    render = list(fake.rendered)
    render[0] = ("common",)
    first_second = next(i for i, w in enumerate(fake.words) if w.ayah == 2)
    render[first_second] = ("common",)
    fake.rendered = tuple(render)
    fake.responder = lambda f, start, duration: observation(("noise", f.phrase(1)[1]) if duration < 2000 else f.phrase(1))
    rows, trace = run(fake, [pause(1000)])
    assert rows[0][2] > 0
    assert any(e.get("acoustic_anchor") == "exact_later_word_noisy_opening" for e in trace["transitions"])
    assert all(r["window_ms"][1] - r["window_ms"][0] <= 6000 for r in trace["acoustic_reads"])


def test_speech_edge_is_finite_context_reason_without_identity_proof():
    from surah_cutter.core.fast_executor import _incomplete_speech_edge

    pcm = np.full(16000 * 5, .05, np.float32)
    obs = observation(("x",))
    groups, ends = {1: ("a", "b", "c"), 2: ("d", "e")}, {1: [3], 2: [2]}
    proof = _incomplete_speech_edge(pcm, 16000, 1500, obs, groups[1], groups, ends, 1, .25)
    assert proof and proof["required_units"] == 3
    assert not _incomplete_speech_edge(pcm, 16000, 1500, observation(()), groups[1], groups, ends, 1, .25)
    assert not _incomplete_speech_edge(pcm, 16000, 1500, observation(("d", "e")), groups[1], groups, ends, 1, .25)
    assert not _incomplete_speech_edge(pcm, 16000, 1500, replace(obs, unit_quality=(.1,)), groups[1], groups, ends, 1, .25)
    assert not _incomplete_speech_edge(pcm, 16000, 1500, replace(obs, positions_ms=None), groups[1], groups, ends, 1, .25)
    pcm[1500 * 16:] = 0
    assert not _incomplete_speech_edge(pcm, 16000, 1500, obs, groups[1], groups, ends, 1, .25)


def test_beam_is_text_free_ctc_mass_and_preserves_blank_separated_repeat():
    logits = np.log(np.array([[.4, .35, .25]] * 3))
    hypotheses, null = independent_beam(logits, 0, {1: "a", 2: "b"}, 20, 0)
    assert hypotheses[0]["units"] == ["a"]  # Greedy is three blanks.
    assert null == pytest.approx(3 * math.log(.4))
    emissions = np.full((3, 2), -100.)
    emissions[np.arange(3), [1, 0, 1]] = 0.
    hypotheses, _ = independent_beam(emissions, 0, {1: "a"}, 20, 0)
    assert hypotheses[0]["units"] == ["a", "a"]


def test_independent_alternative_requires_rival_and_null_margin():
    h = {"units": ["proof"], "positions_ms": [20.], "unit_quality": [.99], "log_probability": -1.}
    competitor = {**h, "units": ["rival"], "log_probability": -2.}
    obs = Observation(("noise",), (0.,), .99, unit_quality=(.99,), hypotheses=(h, competitor), null_log_probability=-10.)
    assert not _beam_view(obs, ("proof",), .6, math.log(9.))[1]
    good = replace(obs, hypotheses=(h, {**competitor, "log_probability": -5.}))
    selected, proof = _beam_view(good, ("proof",), .6, math.log(9.))
    assert selected.units == ("proof",) and proof["negative_log_scores"] == [-5., -10.]
    assert not _beam_view(replace(good, null_log_probability=-1.), ("proof",), .6, math.log(9.))[1]


def canonical_fake(surah):
    words = quran_text.surah_words(surah)
    fake = Fake(lambda *args: observation(()), words)
    fake.rendered = tuple(arabic_units(w.text) for w in words)
    groups, ends = {}, {}
    for word, units in zip(words, fake.rendered):
        groups.setdefault(word.ayah, []).extend(units)
        ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
    return fake, {a: tuple(u) for a, u in groups.items()}, ends


def test_repeated_109_owner_requires_local_previous_tail():
    fake, groups, ends = canonical_fake(109)
    assert plan_phrase(groups, 3, ends)["ambiguous"]
    assert plan_seam(groups, 3, ends) and plan_seam(groups, 5, ends)
    starts = {1000 + 5000 * (a - 1): a for a in groups}
    def respond(f, start, duration):
        if start in starts:
            return observation(groups[starts[start]])
        for ayah in (3, 5):
            if start + duration == 1000 + 5000 * (ayah - 1):
                return observation(groups[ayah - 1])
        return observation(())
    fake.responder = respond
    rows, trace = run(fake, [pause(p) for p in starts])
    assert all(r[2] > 0 for r in rows)
    assert trace["boundaries"][2]["proof"]["contextual_owner"]
    assert trace["boundaries"][4]["proof"]["contextual_owner"]
    fake.responder = lambda f, start, duration: observation(groups[starts[start]]) if start in starts else observation(groups[4])
    rows, trace = run(fake, [pause(p) for p in starts])
    assert rows[2][2] == 0


def test_context_confidence_is_minimum_of_both_essential_reads():
    from surah_cutter.core.fast_executor import Boundary, validate_boundaries

    fake, groups, ends = canonical_fake(109)
    starts = {1000 + 5000 * (a - 1): a for a in groups}
    def respond(f, start, duration):
        if start in starts:
            return observation(groups[starts[start]], quality=.99)
        for ayah in (3, 5):
            if start + duration == 1000 + 5000 * (ayah - 1):
                return observation(groups[ayah - 1], quality=.61)
        return observation(())
    fake.responder = respond
    rows, trace = run(fake, [pause(p) for p in starts])
    assert rows[2][2] == pytest.approx(.61) and rows[4][2] == pytest.approx(.61)
    boundaries = [Boundary(**b) for b in trace["boundaries"]]
    boundaries[2] = replace(boundaries[2], confidence=.99)
    with pytest.raises(ValueError, match="unheard_previous_context"):
        validate_boundaries(boundaries, list(groups), 40000, trace["acoustic_reads"],
                            groups=groups, word_ends=ends, threshold=.6)


@pytest.mark.parametrize("surah,ayah", [(55, 13), (77, 19)])
def test_repeated_refra_ins_have_distinguishing_seam(surah, ayah):
    _, groups, ends = canonical_fake(surah)
    assert plan_phrase(groups, ayah, ends)["ambiguous"]
    assert plan_seam(groups, ayah, ends) is not None


def test_long_pause_keeps_full_dsp_center_with_right_edge_crop():
    fake = Fake(lambda f, start, duration: observation(f.phrase(1)) if start == 7800 else observation(()))
    rows, trace = run(fake, [pause(1000, 7000)])
    assert rows[0][1] == 4500 and rows[0][2] > 0
    read = trace["acoustic_reads"][trace["boundaries"][0]["read_id"]]
    assert read["window_ms"] == [7800., 9300.]
    assert trace["boundaries"][0]["proof"]["pause_ms"] == [1000., 8000.]


def test_long_current_wasl_recovers_tail_opening_after_future_pause():
    fake = Fake(lambda *args: observation(()))
    groups, ends = {}, {}
    for w, units in zip(fake.words, fake.rendered):
        groups.setdefault(w.ayah, []).extend(units)
        ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
    groups = {a: tuple(u) for a, u in groups.items()}
    seam = plan_seam(groups, 2, ends)
    def respond(f, start, duration):
        if start == 1000:
            return observation(f.phrase(1))
        if start == 8000:
            return observation(f.phrase(3))
        return observation(seam["combined"])
    fake.responder = respond
    rows, trace = run(fake, [pause(1000), pause(8000)])
    assert rows[1][2] > 0 and trace["boundaries"][1]["evidence_kind"] == "wasl"
    assert any(r["reason"].endswith("anchored_wasl_forecast") for r in trace["acoustic_reads"])


def test_marginal_prefix_probabilities_match_exhaustive_paths():
    import itertools
    probabilities = np.array([[.4, .35, .25], [.5, .3, .2], [.6, .25, .15]])
    targets = [(1,), (2,), (1, 1), (1, 2), (2, 1)]
    totals = [0.] * len(targets)
    for path in itertools.product(range(3), repeat=3):
        collapsed, previous = [], -1
        for token in path:
            if token != 0 and token != previous:
                collapsed.append(token)
            previous = token
        probability = np.prod([probabilities[i, token] for i, token in enumerate(path)])
        for i, target in enumerate(targets):
            if tuple(collapsed[:len(target)]) == target:
                totals[i] += probability
    assert np.exp(prefix_scores(np.log(probabilities), targets, 0)) == pytest.approx(totals)


def test_residual_mass_blocks_confident_expected_prefix_but_allows_finite_retry():
    from surah_cutter.core.fast_executor import _competing_prefix, Policy
    adapter = SimpleNamespace(vocab={"proof": 1, "rival": 2}, blank_id=0, lag_ms=0, model_id="fake")
    emissions = np.log(np.array([[.1, .26, .64]]))
    obs = Observation(("rival",), (0.,), .64, unit_quality=(.64,), posterior=emissions,
                      stride_ms=20., null_log_probability=math.log(.1), posterior_sha256="test")
    record = {}
    selected, proof, expand = _competing_prefix(obs, ("proof",), {1:("proof",),2:("rival",)}, 1,
        {1:[1],2:[1]}, adapter, record, Policy(), .25)
    assert not proof and selected.units == ("rival",) and not expand
    # Closed canonical rivals are weak, but noncanonical remainder is .74.
    adapter.vocab = {"proof":1, "canonical_rival":3}
    emissions = np.log(np.array([[.01,.26,.72,.01]]))
    obs = replace(obs, posterior=emissions, null_log_probability=math.log(.01))
    selected, proof, expand = _competing_prefix(obs, ("proof",), {1:("proof",),2:("canonical_rival",)}, 1,
        {1:[1],2:[1]}, adapter, {}, Policy(), .25)
    assert not proof and expand
    emissions = np.log(np.array([[.01,.96,.02,.01]]))
    obs = replace(obs, posterior=emissions)
    selected, proof, _ = _competing_prefix(obs, ("proof",), {1:("proof",),2:("canonical_rival",)}, 1,
        {1:[1],2:[1]}, adapter, {}, Policy(), .25)
    assert proof and selected.units == ("proof",)


def test_sparse_anchor_preserves_unresolved_owner_and_blocks_late_repeat():
    from surah_cutter.core.fast_executor import Policy
    fake = Fake(lambda f, start, duration: observation(f.phrase(3)) if start==1000 else observation(f.phrase(1)))
    rows, trace = run(fake, [pause(1000),pause(10000)], policy=Policy(sparse_anchors=True))
    assert rows[0][2] == rows[1][2] == 0 and rows[2][2] > 0
    assert trace["boundaries"][0]["review_reason"] == "unresolved_before_unique_anchor"
    assert trace["boundaries"][2]["proof"]["unresolved_ayahs"] == [1,2]
    assert not any(b["ayah"]==1 and b["confidence"]>0 for b in trace["boundaries"])


def test_sparse_anchor_never_accepts_global_repeat_or_far_future():
    from surah_cutter.core.fast_executor import Policy
    fake = Fake(lambda f, start, duration: observation(f.phrase(4)))
    rows, _ = run(fake, [pause(1000)], policy=Policy(sparse_anchors=True,max_unresolved_ayahs=2))
    assert not any(conf>0 for _,_,conf in rows)
    fake, groups, _ = canonical_fake(109)
    fake.responder = lambda *args: observation(groups[3])
    rows, _ = run(fake, [pause(1000)], policy=Policy(sparse_anchors=True))
    assert not any(conf>0 for _,_,conf in rows)


def test_spoken_partial_owner_rejects_explicit_contradictory_names():
    class Spoken(Fake):
        def proof_word_ends(self, words, rendered):
            return tuple(tuple(range(1,len(u)+1)) for u in rendered)
    fake = Spoken(lambda *args: observation(("الف","لام","راء")))
    fake.rendered = (("الف","لام","ميم"),)+fake.rendered[1:]
    rows, _ = run(fake,[pause(1000)])
    assert rows[0][2] == 0
    fake.responder = lambda *args: observation(("الف",))
    rows, trace = run(fake,[pause(1000)])
    assert rows[0][2] > 0 and trace["boundaries"][0]["proof"]["spoken_owner_units"] == ("الف","لام","ميم")


def test_compact_decoded_letters_preserve_token_quality_and_reject_wrong_alias():
    result = asr_result(["طه"],[.98])
    result.update(token_confidence=.98,truncated=False)
    emitter=SimpleNamespace(model_id="whisper-tarteel-tiny",runtime_identity={"test": True},recognize_local=lambda *a,**kw: result)
    observation = TarteelAdapter(SimpleNamespace(emitter=emitter)).observe(np.ones(16000),16000,total_samples=32000)
    assert observation.units == ("طا","ها") and observation.unit_quality == pytest.approx([.98,.98])
    assert observation.attribution[0]["source_units"] == ("طا","ها")


def test_posterior_is_cached_with_real_bytes_but_never_copied_to_ledger():
    from surah_cutter.core.fast_reads import LocalEvidenceReads
    posterior=np.ones((1000,1000),np.float32)
    fake=Fake(lambda *args: replace(observation(("proof",)), posterior=posterior))
    reads=LocalEvidenceReads(np.ones(16000*40,np.float32),16000,fake,cache_limit_bytes=100000)
    reads.read(0,1000,reason="bounded_posterior")
    assert reads.cache_evictions==1 and reads.cache_bytes==0
    assert "posterior" not in reads.calls[0]["observed"]


def test_future_partial_word_expands_locally_before_sparse_anchor():
    from surah_cutter.core.fast_executor import Policy
    fake = Fake(lambda *args: observation(()))
    rendered = list(fake.rendered)
    index = next(i for i,w in enumerate(fake.words) if w.ayah==2)
    rendered[index] = ("anchorword",)
    fake.rendered = tuple(rendered)
    fake.responder = lambda f,start,duration: observation(("anchor",) if duration<2000 else ("anchorword",))
    rows, trace = run(fake,[pause(1000)],policy=Policy(sparse_anchors=True))
    assert rows[0][2]==0 and rows[1][2]>0
    assert any(e.get("acoustic_anchor","").startswith("provisional_unique_anchor_partial") for e in trace["transitions"])


def test_decaying_long_interval_is_review_but_stable_silence_keeps_geometry():
    from surah_cutter.core.fast_executor import pause_geometry
    interval=pause(1000,3000).pause
    stable=np.full(16000*6,.001,np.float32)
    assert pause_geometry(stable,16000,interval)["accepted"]
    decaying=stable.copy()
    decaying[16000:64000]=.04*np.exp(-np.arange(48000)/16000)
    result=pause_geometry(decaying,16000,interval)
    assert not result["accepted"] and result["reason"]=="decaying_tail_midpoint_unproved"
    assert [interval.start_ms,interval.end_ms]==[1000.,4000.]


def test_ctc_segmentation_projection_preserves_apostrophe_and_repeated_letters(monkeypatch):
    from surah_cutter.core import align
    from surah_cutter.core.fast_adapters import CTCAdapter
    vocab={"a":0,"-":1,"'":2,"[PAD]":3,"|":4," ":5}
    bundle=SimpleNamespace(emitter=SimpleNamespace(runtime_identity={"weights_sha256":"test"}),vocab=vocab,blank_id=3,lag_ms=0)
    sequence=[0,1,0,2,0]
    emissions=np.full((5,6),-100.,np.float32)
    emissions[np.arange(5),sequence]=0.
    calls=[]
    def infer(*args):
        calls.append(args[1].shape)
        return emissions
    monkeypatch.setattr(align,"_emissions",infer)
    observed=CTCAdapter(bundle).observe(np.ones(16000,np.float32),16000,total_samples=32000)
    assert observed.units==("a","a","'","a") and len(calls)==1


def test_failed_future_partial_does_not_block_current_after_intro():
    from surah_cutter.core.fast_executor import Policy
    fake=Fake(lambda *args: observation(()))
    rendered=list(fake.rendered)
    index=next(i for i,w in enumerate(fake.words) if w.ayah==2)
    rendered[index]=("anchorword",)
    fake.rendered=tuple(rendered)
    def respond(f,start,duration):
        if start==1000:
            return observation(("anchor",) if duration<2000 else ("intro",))
        if start==6000:
            return observation(f.phrase(1))
        return observation(())
    fake.responder=respond
    rows,trace=run(fake,[pause(1000),pause(6000)],policy=Policy(sparse_anchors=True))
    assert rows[0][2]>0 and rows[0][1]==6100
    assert any(e.get("acoustic_anchor","").startswith("provisional_unique_anchor_partial") for e in trace["transitions"])


def test_loaded_code_fingerprint_stable_and_changes_for_running_helper(monkeypatch):
    from surah_cutter.core.fast_adapters import loaded_implementation_sha
    from surah_cutter.core import fast_ctc
    before=loaded_implementation_sha()
    assert loaded_implementation_sha()==before
    def changed(*args,**kwargs):
        return ()
    changed.__module__=fast_ctc.__name__
    monkeypatch.setattr(fast_ctc,"prefix_scores",changed)
    assert loaded_implementation_sha()!=before


def test_far_unique_anchor_does_not_read_or_confirm_the_unresolved_gap():
    from surah_cutter.core.fast_executor import Policy
    fake=Fake(lambda f,start,duration: observation(f.phrase(4)))
    rows,trace=run(fake,[pause(1000)],policy=Policy(sparse_anchors=True,max_unresolved_ayahs=None))
    assert all(r[2]==0 for r in rows[:3]) and rows[3][2]>0
    assert len(fake.calls)==1
    assert trace["boundaries"][3]["proof"]["unresolved_ayahs"]==[1,2,3]
    fake.responder=lambda f,start,duration: observation(f.phrase(4),quality=.1)
    rows,_=run(fake,[pause(1000)],policy=Policy(sparse_anchors=True,max_unresolved_ayahs=None))
    assert not any(r[2]>0 for r in rows)


def test_hallucinated_wrong_or_repeated_far_identity_cannot_be_an_anchor():
    from surah_cutter.core.fast_executor import Policy
    fake=Fake(lambda f,start,duration: observation((f.phrase(4)[0],"wrong")))
    # Make one-word opening insufficient: the first word also occurs internally.
    render=list(fake.rendered)
    target=next(i for i,w in enumerate(fake.words) if w.ayah==4)
    render[1]=render[target]
    fake.rendered=tuple(render)
    rows,_=run(fake,[pause(1000)],policy=Policy(sparse_anchors=True,max_unresolved_ayahs=None))
    assert not any(r[2]>0 for r in rows)


def test_loaded_code_fingerprint_includes_effective_beam_default(monkeypatch):
    from surah_cutter.core.fast_adapters import loaded_implementation_sha
    from surah_cutter.core.fast_ctc import independent_beam
    before=loaded_implementation_sha()
    monkeypatch.setitem(independent_beam.__kwdefaults__,"width",9)
    assert loaded_implementation_sha()!=before
