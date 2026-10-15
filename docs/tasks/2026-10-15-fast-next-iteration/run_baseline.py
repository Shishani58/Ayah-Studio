import json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).parent.resolve()
root=Path(json.loads((p/'snapshot-paths.json').read_text())['baseline']).resolve()
assert root != Path.cwd().resolve()
args=[sys.executable,'-u',str(root/'tools/bench_fast_models.py'),'--models','quran-phonetics','quran-phonetics-fp32','whisper-tarteel-tiny','whisper-tarteel-base','--surahs',*map(str,[1,32,55,59,65,73,77,80,98,112,2,79]),'--audio-dir','/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri','--reference',str(p.parents[2]/'data/references/abu_bakr_ash_shatri.db'),'--output',str(p/'baseline-raw.json')]
(p/'baseline-command.json').write_text(json.dumps({'argv':args,'cwd':str(root),'PYTHONPATH':str(root)},indent=2))
with (p/'baseline.log').open('w') as log:
 r=subprocess.run(args,cwd=root,env={**os.environ,'PYTHONPATH':str(root)},stdout=log,stderr=subprocess.STDOUT)
print('baseline exit',r.returncode,flush=True)
raise SystemExit(r.returncode)
