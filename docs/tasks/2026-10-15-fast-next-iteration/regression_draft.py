"""Prepared during research; run only at the final verification stage."""
from types import SimpleNamespace
import pytest
from surah_cutter.core.fast_executor import _split_opening

GROUPS = {1: ('انما', 'توعدون', 'لواقع'), 2: ('فاذا', 'النجوم', 'طمست')}
ENDS = {a: list(range(1, len(u) + 1)) for a, u in GROUPS.items()}

def observed(units=('ان', 'ما', 'توعدون'), quality=None):
    qs = quality or (.9,) * len(units)
    return SimpleNamespace(units=units, unit_quality=qs, truncated=False,
        attribution=tuple(dict(word=u, selected_indices=[i], probabilities=[q], token_ids=[100+i])
                          for i,(u,q) in enumerate(zip(units,qs))))

def test_split_keeps_all_original_selected_tokens():
    proof, expansion = _split_opening(observed(), GROUPS, ENDS, 1, .6)
    assert expansion and proof['phrase'] == GROUPS[1][:2]
    assert proof['segmented_opening']['source_count'] == 3
    assert proof['segmented_opening']['selected_indices'] == [0,1,2]
    assert proof['segmented_opening']['onset_supported'] is False

@pytest.mark.parametrize('units,quality,expand', [
    (('ان','ما'), None, True),  # incomplete continuation may expand, never confirms
    (('ان','ما','تريد'), (.9,.9,.4), True),
    (('ان','ما','تريد'), (.9,.9,.9), False),
    (('ان','ما','ثم','توعدون'), None, False),
    (('ان','ما','لواقع'), None, False),
    (('ان','م','توعدون'), None, False),
    (('ان','ما','توع'), None, True),
    (('ان','ما','توعدون'), (.9,.59,.9), False),
    (('انماتوعدون',), None, False), # merge intentionally unsupported
])
def test_partial_inserted_missing_weak_and_merge_do_not_confirm(units,quality,expand):
    assert _split_opening(observed(units,quality), GROUPS, ENDS, 1, .6) == (None,expand)

@pytest.mark.parametrize('other', [
    ('ان','ما','توعدون'), ('قال','ان','ما','توعدون'), ('انما','توعدون','لواقع'),
])
def test_symmetric_owner_internal_start_and_refrain_rejected(other):
    groups={**GROUPS,3:other};ends={a:list(range(1,len(u)+1)) for a,u in groups.items()}
    assert _split_opening(observed(),groups,ends,1,.6)==(None,False)

def test_wrong_owner_rejected():
    assert _split_opening(observed(),GROUPS,ENDS,2,.6)==(None,False)

@pytest.mark.parametrize('fault', ['unsupported','alias','probabilities','indices','word','truncated'])
def test_original_attribution_cannot_be_bypassed(fault):
    obs=observed()
    if fault=='unsupported':obs.attribution[1]['unsupported']='ambiguous'
    if fault=='alias':obs.attribution[1]['spoken_alias']='named_letter'
    if fault=='probabilities':obs.attribution[1]['probabilities']=[.4,.9]
    if fault=='indices':obs.attribution[1]['selected_indices']=[]
    if fault=='word':obs.attribution[1]['word']='foreign'
    if fault=='truncated':obs.truncated=True
    assert _split_opening(obs,GROUPS,ENDS,1,.6)==(None,False)


def test_validator_rechecks_source_count_and_json_roundtrip():
    import json
    from surah_cutter.core.fast_executor import Boundary, validate_boundaries
    obs=observed()
    proof,_=_split_opening(obs,GROUPS,ENDS,1,.6)
    proof.update(complete=True,ambiguous=False,competitors_checked=True,offset=0,pause_ms=[0,200])
    read={'outcome':'completed','window_ms':[0,1500],'observed':{
        'units':obs.units,'unit_quality':obs.unit_quality,'quality':.9,
        'truncated':False,'attribution':obs.attribution}}
    serialized=json.loads(json.dumps({'proof':proof,'read':read}))
    boundary=Boundary(1,100,'pause',.9,read_id=0,proof=serialized['proof'])
    validate_boundaries([boundary],[1],5000,[serialized['read']],groups=GROUPS,word_ends=ENDS,threshold=.6)
    boundary.proof['segmented_opening']['source_count']=2
    with pytest.raises(ValueError,match='unheard_segmented_phrase'):
        validate_boundaries([boundary],[1],5000,[serialized['read']],groups=GROUPS,word_ends=ENDS,threshold=.6)
