"""Whole-corpus DSP benchmark, no alignment models and no pytest."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("label",choices=["before","after"])
args = parser.parse_args()
task = Path(__file__).parent.resolve()
source = Path(json.loads((task / "snapshot-paths.json").read_text())[args.label]).resolve()
manifest = json.loads((task / "snapshot-manifests.json").read_text())[args.label]
assert all(hashlib.sha256((source / k).read_bytes()).hexdigest()==v for k,v in manifest.items())
sys.path.insert(0,str(source))
from surah_cutter.core import candidates,decode  # noqa: E402
assert Path(candidates.__file__).resolve() == source / "surah_cutter/core/candidates.py"
result = []
for surah in [1,2,32,55,59,65,73,77,79,80,98,112]:
    audio = Path(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03}.mp3")
    pcm = decode.decode(str(audio))
    start = time.perf_counter()
    decisions = candidates.pause_pipeline_decisions(pcm,16000,surah=surah)
    elapsed = time.perf_counter()-start
    result.append({"surah":surah,"audio_sha256":hashlib.sha256(audio.read_bytes()).hexdigest(),
                   "samples":len(pcm),"seconds":elapsed,"decisions":[asdict(d) for d in decisions]})
    (task / f"dsp-{args.label}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(args.label,surah,len(decisions),elapsed,flush=True)
assert all(hashlib.sha256((source / k).read_bytes()).hexdigest()==v for k,v in manifest.items())
