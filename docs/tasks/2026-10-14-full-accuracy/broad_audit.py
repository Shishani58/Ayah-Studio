"""Same-emissions before/after on Shatri78–114 and67, no DB writes."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
sys.path.insert(0,str(ROOT))
from surah_cutter.core import align, decode, emissions, full_windows  # noqa: E402
from surah_cutter.data import quran_text, references  # noqa: E402

emitter=emissions.load_emitter(ROOT/"models/wav2vec2-quran-phonetics.onnx")
bundle=align._OnnxBundle(emitter)
CACHE=Path.home()/".cache/surah-cutter/full-accuracy"
real_check=align._check_ayah_onsets
results=[]
for surah in [67,*range(78,115)]:
    path=Path.home()/f"Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3"
    pcm=decode.decode(str(path))
    stem=f"abu_bakr_ash_shatri-{surah}"
    matrix_path=CACHE/(stem+".npy")
    window_path=CACHE/(stem+"-windows.json")
    started=time.perf_counter()
    if matrix_path.exists():
        matrix=np.load(matrix_path)
        windows=json.loads(window_path.read_text())
        cached=True
    else:
        windows={}
        matrix=full_windows.generate_pause_emissions(emitter,pcm,window_s=20,workers=3,
                         strong_pauses=True,max_window_s=60,trace=windows)
        np.save(matrix_path,matrix)
        window_path.write_text(json.dumps(windows))
        cached=False
    inference_s=time.perf_counter()-started
    def cached_full(_bundle,_pcm,_np,*,trace=None):
        trace["full_windows"]=copy.deepcopy(windows)
        trace["full_windows"].update(algorithm=full_windows.FULL_POLICY,
                          review_policy=full_windows.FULL_REVIEW_POLICY,
                          onset_policy=full_windows.FULL_ONSET_POLICY)
        return matrix
    align._full_emissions=cached_full
    def baseline_check(*args,**kwargs):
        kwargs.pop("pcm",None)
        return real_check(*args,**kwargs)
    before_trace={}
    align._check_ayah_onsets=baseline_check
    started=time.perf_counter()
    before=align.align_surah(pcm,16000,quran_text.surah_words(surah),model=bundle,surah=surah,trace=before_trace)
    before_s=time.perf_counter()-started
    after_trace={}
    align._check_ayah_onsets=real_check
    started=time.perf_counter()
    after=align.align_surah(pcm,16000,quran_text.surah_words(surah),model=bundle,surah=surah,trace=after_trace)
    after_s=time.perf_counter()-started
    assert [r[0] for r in after]==list(range(1,quran_text.ayah_count(surah)+1))
    assert all(b[1]>a[1] for a,b in zip(after,after[1:]))
    assert [r[2] for r in before]==[r[2] for r in after]
    assert [c["reliable"] for c in before_trace["ayahs"]]==[c["reliable"] for c in after_trace["ayahs"]]
    gt=references.load_reference("abu_bakr_ash_shatri")
    result={"surah":surah,"audio_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "emissions_sha256":hashlib.sha256(matrix.tobytes()).hexdigest(),"cached_emissions":cached,
            "inference_s":inference_s,"before_alignment_s":before_s,"after_alignment_s":after_s,
            "before":before,"after":after,"before_errors_ms":[r[1]-gt[surah,r[0]] for r in before],
            "after_errors_ms":[r[1]-gt[surah,r[0]] for r in after],"checks":after_trace["ayahs"]}
    results.append(result)
    (TASK/"broad-audit.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    print(surah,"changes",[(a[0],a[1],b[1]) for a,b in zip(before,after) if a[1]!=b[1]],flush=True)
