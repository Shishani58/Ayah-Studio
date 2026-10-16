"""Save predictions/dispatches before any external reference evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time

from surah_cutter.core import align, candidates, decode, fast_executor, model_store
from surah_cutter.core.model_runtime import release_runtimes
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch

P = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('--surahs', nargs='+', type=int, default=[2, 78, 80, 112])
parser.add_argument('--models', nargs='+', default=['whisper-tarteel-base', 'quran-phonetics'])
parser.add_argument('--reciter', default='shatri')
args = parser.parse_args()
audio = (Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri') if args.reciter == 'shatri' else
         Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / args.reciter)
output = P / (args.label + '.jsonl')
assert not output.exists()
source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('surah_cutter').rglob('*.py')}
for model in args.models:
 model_store.ACTIVE_ID = model
 align._ONNX_CACHE.clear()
 release_runtimes()
 started = time.monotonic()
 bundle = align.default_bundle()
 if hasattr(bundle.emitter, 'probe'):
  bundle.emitter.probe()
 load = time.monotonic() - started
 for surah in args.surahs:
  path = audio / f'{surah:03}.mp3'
  started = time.monotonic()
  pcm = decode.decode(str(path))
  decode_s = time.monotonic() - started
  started = time.monotonic()
  decisions = candidates.pause_pipeline_decisions(pcm, 16000, surah=surah)
  dsp_s = time.monotonic() - started
  trace = {}
  started = time.monotonic()
  dispatches = []
  try:
   with dispatch_spy(bundle, pcm) as dispatches:
    rows = fast_executor.align_surah_sequential(pcm, 16000, quran_text.surah_words(surah),
         bundle=bundle, surah=surah, trace=trace, pause_decisions=decisions)
  except BaseException as error:
   # SIGINT/GenerationCancelled/model failures must keep their already charged
   # local intervals and independent actual PCM observations. Failed cases are
   # separate from successful prediction JSONL, never fed to the evaluator.
   failed = dict(model=model, reciter=args.reciter, surah=surah, trace=trace,
                 dispatches=dispatches, error=type(error).__name__+': '+str(error),
                 seconds=time.monotonic()-started, source_sha256=source,
                 outcome='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
   (P/(args.label+f'-failed-{surah}-{model}.json')).write_text(json.dumps(failed,
                 ensure_ascii=False, indent=2)+'\n')
   release_runtimes()
   raise
  seconds = time.monotonic() - started
  verify_dispatch(dispatches, pcm, trace)
  item = dict(model=model, reciter=args.reciter, surah=surah, rows=rows, trace=trace,
       seconds=seconds, decode_seconds=decode_s, dsp_seconds=dsp_s, load_seconds=load,
       dispatches=dispatches, decisions=[asdict(d) for d in decisions],
       audio_path=str(path), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       source_sha256=source, peak_rss_self=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       peak_rss_children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
  with output.open('a') as log:
   log.write(json.dumps(item, ensure_ascii=False) + '\n')
  print(args.label, model, surah, 'positive', sum(r[2] > 0 for r in rows),
        'calls', len(dispatches), 'seconds', round(seconds, 2), flush=True)
release_runtimes()
