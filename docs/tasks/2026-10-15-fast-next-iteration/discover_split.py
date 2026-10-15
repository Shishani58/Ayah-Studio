import json,sys
from pathlib import Path
from surah_cutter.core.fast_adapters import arabic_units
from surah_cutter.core.model_runtime import inference_words
from surah_cutter.core.model_worker import whisper_alignment_text
from surah_cutter.data import quran_text
p=Path(__file__).parent
out=[]
for model in ['whisper-tarteel-tiny','whisper-tarteel-base']:
 for c in json.loads((p.parent/'2026-10-15-fast-head-word-boundary'/f'{model}-after.json').read_text()):
  words=quran_text.surah_words(c['surah'])
  groups={}
  for w,t in zip(words,inference_words(words,model)):
   groups.setdefault(w.ayah,[]).extend(arabic_units(whisper_alignment_text(t)))
  for r in c['trace']['acoustic_reads']:
   o=r.get('observed',{}); units=o.get('units',[])
   if not units:continue
   candidates=[]
   for a,g in groups.items():
    joined=''
    for n,u in enumerate(units[:5],1):
     joined+=u
     if joined==g[0] and n>1:
      candidates.append({'ayah':a,'first':g[0],'source_count':n,'min_quality':min(o['unit_quality'][:n])})
     if not g[0].startswith(joined):break
   if candidates:out.append({'model':model,'surah':c['surah'],'read_id':r['id'],'reason':r['reason'],'window_ms':r['window_ms'],'units':units,'candidates':candidates})
(p/'split-discovery.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('split reads',len(out));print(json.dumps(out,ensure_ascii=False,indent=2))
