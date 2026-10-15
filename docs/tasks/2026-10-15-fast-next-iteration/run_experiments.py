import argparse,json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).parent.resolve();repo=p.parents[2]
args=argparse.ArgumentParser();args.add_argument('kind',choices=['local','after','transfer','timing']);kind=args.parse_args().kind
paths=json.loads((p/'snapshot-paths.json').read_text())
def run(label,source,models,surahs,audio,reference):
 root=Path(paths[source]).resolve()
 command=[sys.executable,'-u',str(root/'tools/bench_fast_models.py'),'--models',*models,'--surahs',*map(str,surahs),'--audio-dir',str(audio),'--reference',str(reference),'--output',str(p/f'{label}.json')]
 (p/f'{label}-command.json').write_text(json.dumps({'argv':command,'cwd':str(root),'PYTHONPATH':str(root)},indent=2))
 print('start',label,flush=True)
 with (p/f'{label}.log').open('w') as f:r=subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':str(root)},stdout=f,stderr=subprocess.STDOUT)
 print('exit',label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
models=['quran-phonetics','quran-phonetics-fp32','whisper-tarteel-tiny','whisper-tarteel-base']
if kind=='local':run('split-local','prototype',['whisper-tarteel-tiny'],[77],'/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri',repo/'data/references/abu_bakr_ash_shatri.db')
if kind=='after':run('after-raw','prototype',models,[1,32,55,59,65,73,77,80,98,112,2,79],'/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri',repo/'data/references/abu_bakr_ash_shatri.db')
if kind=='transfer':
 manifest=json.loads((p/'transfer-selection.json').read_text())
 for variant in ['baseline','prototype']:
  run('transfer-'+variant,variant,models,manifest['surahs'],manifest['audio_dir'],manifest['reference'])

if kind=='timing':
 for variant in ['baseline','prototype']:
  run('timing-'+variant,variant,['whisper-tarteel-tiny'],[77,77,77],'/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri',repo/'data/references/abu_bakr_ash_shatri.db')
