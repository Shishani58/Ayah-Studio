"""An independently heard body may authorize a retry, never a guessed onset."""
from test_fast_executor import Fake,observation,pause,run
from surah_cutter.core.fast_executor import Policy


def scenario(body="strong",head="headx"):
    def respond(fake,start,duration):
        ayah={1000:1,6000:2,12000:3}.get(start)
        if ayah:
            return observation(fake.phrase(ayah),positions=False)
        if start==20000:
            if duration<2000:
                return observation((head,),quality=.7,positions=False)
            return observation(("head","wrong" if duration<3000 else "wanted"),positions=False)
        if start==24000:
            return observation(("body",) if body=="partial" else ("body","suffix"),
                               quality=.59 if body=="weak" else .99,positions=False)
        return observation((),positions=False)

    fake=Fake(respond)
    fake.model_id="whisper-tarteel-tiny"
    fake.timing=False
    rendered=list(fake.rendered)
    # Real112 has the same «ولم» in3 and4; full4opening needs «يكن» too.
    rendered[8],rendered[9]=("head",),("rival",)
    rendered[10],rendered[11]=("head",),("wanted",)
    rendered[12],rendered[13]=("body",),("suffix",)
    fake.rendered=tuple(rendered)
    return run(fake,[pause(t) for t in [1000,6000,12000,20000,24000]],
               policy=Policy(sparse_anchors=True))


def test_body_cue_authorizes_full_head_retry_at_same_measured_pause():
    rows,trace=scenario()
    assert rows[3][1:]==(20100,.99)
    rechecks=[e for e in trace["transitions"] if e["state"]=="HEAD_BOUNDARY_RECHECK"]
    assert len(rechecks)==2
    assert all(e["candidate"]==3 for e in rechecks)
    boundary=trace["boundaries"][3]
    assert boundary["proof"]["head_boundary_recheck"]
    read=trace["acoustic_reads"][boundary["read_id"]]
    assert tuple(read["observed"]["units"][:2])==("head","wanted")
    assert boundary["start_ms"]==20100


def test_weak_body_cannot_authorize_an_unread_head():
    rows,trace=scenario("weak")
    assert rows[3][2]==0
    assert not any(e["state"]=="HEAD_BOUNDARY_RECHECK" for e in trace["transitions"])


def test_partial_body_cannot_authorize_an_unread_head():
    rows,trace=scenario("partial")
    assert rows[3][2]==0
    assert not any(e["state"]=="HEAD_BOUNDARY_RECHECK" for e in trace["transitions"])


def test_unrelated_initial_word_is_not_replaced_using_body_identity():
    rows,trace=scenario(head="unrelated")
    assert rows[3][2]==0
    assert not any(e["state"]=="HEAD_BOUNDARY_RECHECK" for e in trace["transitions"])
