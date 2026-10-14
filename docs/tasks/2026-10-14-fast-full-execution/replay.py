"""Replay frozen observations after an acceptance-only fix; NO text-model inference; DSP/VAD is recalculated."""
import hashlib
import json
from pathlib import Path

import numpy as np

from surah_cutter.core import candidates, decode
from surah_cutter.core.fast_adapters import Observation, arabic_units
from surah_cutter.core.fast_executor import Policy, align_surah_sequential
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.data import quran_text, transliteration

ROOT=Path(__file__).parent
frozen=json.loads((ROOT/"corpus-final.json").read_text())
summary=json.loads((ROOT/"corpus-final.summary.json").read_text())

class Replay:
    def __init__(self,item):
        self.item=item
        self.reads=iter(zip(item["actual_dispatches"],item["trace"]["acoustic_reads"]))
        self.count=0
        snap=item["trace"]["run_snapshot"]
        self.version=snap["adapter"]
        self.model_id=snap["model"]
        self.revision=snap["revision"]
        self.lag_ms=snap["lag_ms"]
        self.metric=item["trace"]["backend"]
        self.timing=self.model_id=="quran-phonetics"
    def represent(self,words):
        if self.timing:
            return tuple(tuple(text) for text in transliteration.surah_word_tr(words[0].surah))
        return tuple(arabic_units(text) for text in inference_words(words,self.model_id))
    def observe(self,pcm,sr,*,total_samples):
        dispatch,record=next(self.reads)
        assert dispatch["samples"]==len(pcm)
        assert dispatch["sha256"]==hashlib.sha256(memoryview(np.ascontiguousarray(pcm,dtype=np.float32))).hexdigest()
        self.count+=1
        obs=dict(record["observed"])
        for key in ("units","positions_ms","unit_quality"):
            if obs[key] is not None:
                obs[key]=tuple(obs[key])
        return Observation(**obs)

results=[]
for surah in sorted({i["surah"] for i in frozen}):
    pcm=decode.decode(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3")
    decisions=[]
    for original in summary["initial_pauses"][str(surah)]["decisions"]:
        d=dict(original)
        d["pause"]=candidates.Pause(**d["pause"])
        decisions.append(candidates.PauseDecision(**d))
    for item in [i for i in frozen if i["surah"]==surah]:
        replay=Replay(item)
        trace={}
        rows=align_surah_sequential(pcm,16000,quran_text.surah_words(surah),adapter=replay,
               pause_decisions=decisions,trace=trace,policy=Policy(**item["trace"]["run_snapshot"]["policy"]))
        assert [list(r) for r in rows]==item["rows"]
        assert replay.count==len(item["actual_dispatches"])
        assert [r["window_ms"] for r in trace["acoustic_reads"]]==[r["window_ms"] for r in item["trace"]["acoustic_reads"]]
        results.append({"model":item["model"],"surah":surah,"identical_rows":True,
                        "identical_windows":True,"replayed_observations":replay.count})
source=Path("surah_cutter/core/fast_executor.py")
out={"kind":"observation_replay_not_new_inference","frozen_source_sha256":summary["source_sha256"][str(source)],
     "current_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
     "current_algorithm":trace["fast_version"],"current_dsp_models":trace["run_snapshot"].get("dsp_models"),
     "results":results}
(ROOT/"replay.json").write_text(json.dumps(out,indent=2)+"\n")
print(len(results),"recordings: identical rows/windows; no new text-model inference")
