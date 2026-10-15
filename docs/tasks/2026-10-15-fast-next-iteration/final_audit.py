import ast,copy,hashlib,json,math,re
from pathlib import Path
from types import SimpleNamespace
from surah_cutter.core.fast_executor import _split_opening,Boundary,validate_boundaries,Policy
from surah_cutter.core.fast_adapters import arabic_units
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.core.model_worker import whisper_alignment_text
from surah_cutter.data import quran_text
p=Path(__file__).parent;paths=json.loads((p/'snapshot-paths.json').read_text())
file=Path(paths['prototype'])/'surah_cutter/core/fast_executor.py'
tree=ast.parse(file.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_split_opening')
namespace={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),namespace)
old=namespace['_split_opening'];replayed=validated=0
for label in ['baseline-raw','after-raw','transfer-baseline','transfer-prototype']:
 assert (p/f'{label}.summary.json').exists()
 for c in json.loads((p/f'{label}.json').read_text()):
  if not c['model'].startswith('whisper'):continue
  groups={}
  for w,t in zip(quran_text.surah_words(c['surah']),inference_words(quran_text.surah_words(c['surah']),c['model'])):
   groups.setdefault(w.ayah,[]).extend(arabic_units(whisper_alignment_text(t)))
  groups={a:tuple(u) for a,u in groups.items()};ends={a:list(range(1,len(u)+1)) for a,u in groups.items()}
  for read in c['trace']['acoustic_reads']:
   a=int(re.search(r'ayah=(\d+):',read['reason']).group(1));o=read['observed']
   obs=SimpleNamespace(units=tuple(o['units']),unit_quality=tuple(o['unit_quality']),attribution=tuple(o['attribution']),truncated=o['truncated'])
   assert json.dumps(old(obs,groups,ends,a,.6),sort_keys=True)==json.dumps(_split_opening(obs,groups,ends,a,.6),sort_keys=True)
   replayed+=1
  for b in c['trace']['boundaries']:
   if not (b.get('proof') or {}).get('segmented_opening'):continue
   pauses=[SimpleNamespace(**p) for p in c['trace']['phase1_pauses']]
   validate_boundaries([Boundary(**b)],[b['ayah']],c['duration_ms'],c['trace']['acoustic_reads'],groups=groups,word_ends=ends,threshold=.6,pauses=pauses,pause_geometry_data=c['trace']['pause_geometry'])
   validated+=1
protected=json.loads((p/'protected-sha.json').read_text())
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in protected.items())
initial=json.loads((p/'initial-sha.json').read_text());changed=[f for f,h in initial.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
assert changed==['surah_cutter/core/fast_executor.py'],changed
assert Path('AGENTS.md').read_bytes()==Path('CLAUDE.md').read_bytes()
out=dict(source_files_changed=changed,protected_db_sha_unchanged=len(protected),asr_observations_equivalent=replayed,segmented_json_proofs_validated=validated,agents_claude_equal=True)
(p/'final-audit.json').write_text(json.dumps(out,indent=2));print(out)
