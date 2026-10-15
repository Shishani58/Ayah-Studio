"""Raw Arabic spelling must decide aliases without changing BPE ownership."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from surah_cutter.core.fast_adapters import TarteelAdapter,attributed_asr_units


def result(text):
    payload = text.encode("utf-8")
    return {"text":text,"token_confidence":.7,"truncated":False,"selected_tokens":[
        {"id":1,"bytes":list(payload[:1]),"log_probability":math.log(.9)},
        {"id":2,"bytes":list(payload[1:]),"log_probability":math.log(.6)}]}


def observe(value):
    emitter = SimpleNamespace(model_id="whisper-tarteel-base",runtime_identity={},
                              recognize_local=lambda pcm,**kwargs:value)
    return TarteelAdapter(SimpleNamespace(emitter=emitter)).observe(np.zeros(1000),16000,total_samples=1000)


@pytest.mark.parametrize("text,units,alias",[
    ("أَلَمْ",("الم",),False),
    ("الٓمٓ",("الف","لام","ميم"),True),
    ("الم",("الف","لام","ميم"),True),
])
def test_raw_hamza_distinguishes_question_from_named_letters(text,units,alias):
    observed = observe(result(text))
    assert observed.units==units
    assert observed.unit_quality==pytest.approx((.6,)*len(units))
    assert all(a["selected_indices"]==[0,1] for a in observed.attribution)
    assert all(("spoken_alias" in a)==alias for a in observed.attribution)
    assert all(a["raw_word"]==text for a in observed.attribution)


def test_raw_words_and_shared_token_ownership_remain_separate():
    units,quality,attribution = attributed_asr_units(result("أَلَمْ تَرَ"))
    assert units==("الم","تر")
    assert quality==pytest.approx((.6,.6))
    assert [a["raw_word"] for a in attribution]==["أَلَمْ","تَرَ"]
    assert [a["selected_indices"] for a in attribution]==[[0,1],[1]]


def test_nonreversible_single_word_stays_unsupported_without_crashing():
    value = result("أَلَمْ")
    value["selected_tokens"][1]["bytes"]=list(b"wrong")
    observed = observe(value)
    assert observed.unit_quality and all(q==0 for q in observed.unit_quality)
    assert all(a["unsupported"]=="ambiguous_token_word_attribution" for a in observed.attribution)
