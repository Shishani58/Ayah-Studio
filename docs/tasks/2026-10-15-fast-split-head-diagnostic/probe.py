"""Independent local head recheck; word text and quality remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
from surah_cutter.core import align,decode,model_store
from surah_cutter.core.fast_adapters import TarteelAdapter
from surah_cutter.core.fast_executor import Policy,plan_phrase
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy,verify_dispatch

parser = argparse.ArgumentParser()
parser.add_argument("surah",type=int,choices=[77])
args = parser.parse_args()
surah=args.surah
ayah,pause_id,windows = {77:(7,6,[(200,2500),(200,4000)])}[surah]
task=Path(__file__).parent.resolve()
prior=task.parent / "2026-10-15-fast-head-word-boundary"
case=next(c for c in json.loads((prior / "whisper-tarteel-tiny-after.json").read_text()) if c["surah"]==surah)
audio=Path(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03}.mp3")
assert hashlib.sha256(audio.read_bytes()).hexdigest()==case["audio_sha256"]
pcm=decode.decode(str(audio))
model_store.ACTIVE_ID="whisper-tarteel-tiny"
bundle=align._get_onnx_bundle()
adapter=TarteelAdapter(bundle)
words=quran_text.surah_words(surah)
rendered=adapter.represent(words)
proof_ends=adapter.proof_word_ends(words,rendered)
groups,ends={},{}
for w,u,e in zip(words,rendered,proof_ends):
    base=len(groups.get(w.ayah,()))
    groups.setdefault(w.ayah,[]).extend(u)
    ends.setdefault(w.ayah,[]).extend(base+x for x in e)
groups={a:tuple(u) for a,u in groups.items()}
plan=plan_phrase(groups,ayah,ends)
phrase=plan["units"]
pause=case["trace"]["phase1_pauses"][pause_id]
threshold=Policy().asr_quality
reads=LocalEvidenceReads(pcm,16000,adapter,cache_limit_bytes=32*1024*1024)
out={"scope":"headdiagnostic only; no boundaryaccepted","surah":surah,"ayah":ayah,
     "phrase":phrase,"ambiguous":plan["ambiguous"],"pause":pause,"threshold":threshold,
     "audio_sha256":case["audio_sha256"],"runtime_identity":adapter.runtime_identity,"evaluations":[]}
try:
    with dispatch_spy(bundle,pcm) as dispatches:
        for preroll,speech in windows:
            obs=reads.read(max(pause["start_ms"],pause["end_ms"]-preroll),min(len(pcm)/16,pause["end_ms"]+speech),reason=f"ayah={ayah}:head_crop_diagnostic")
            if obs is None:
                break
            complete=tuple(obs.units[:len(phrase)])==phrase
            q=min(obs.unit_quality[:len(phrase)],default=0)
            owner=rendered[next(i for i,w in enumerate(words) if w.ayah==ayah)]
            conflict=any("spoken_alias" in a and tuple(a["source_units"])!=owner for a in obs.attribution[:len(phrase)])
            proof=complete and not obs.truncated and not conflict and not plan["ambiguous"] and q>=threshold
            out["evaluations"].append({"preroll_ms":preroll,"speech_ms":speech,"units":obs.units,
                                       "unit_quality":obs.unit_quality,"complete":complete,"proof":proof})
            out.update(reads=reads.calls,actual_dispatches=dispatches)
            (task / f"head-{surah}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
            if proof:
                break
    verify_dispatch(dispatches,pcm,{"acoustic_reads":reads.calls})
finally:
    release_runtimes()
print(out["evaluations"])
