import csv,json,collections
from pathlib import Path
import numpy as np
p=Path(__file__).parent
cases=json.loads((p/'baseline-raw.json').read_text());assert len(cases)==48 and (p/'baseline-raw.summary.json').exists()
rows=[];totals=[]
for c in cases:
 t=c['trace'];conf=[e for r,e in zip(c['rows'],c['errors_ms']) if r[2]>0]
 first=next((b for b in t['boundaries'] if b['confidence']==0),None)
 blocks=[x for x in t['transitions'] if x['state']=='RESOLVE_BLOCK']
 stop=next((x for x in t['transitions'] if x['state'] in ('RESOLVE_BLOCK','REVIEW_REQUIRED')),None)
 reads=[r for r in t['acoustic_reads'] if stop and r['reason'].startswith(f"ayah={stop['ayah']}:")]
 r=reads[-1] if reads else None
 def stats(es):return [float(np.median(es)),float(np.percentile(es,95)),max(es)] if es else [None]*3
 allmed,allp95,allmax=stats(c['errors_ms']);med,p95,mx=stats(conf)
 over=[{'ayah':r[0],'error_ms':e,'confirmed':r[2]>0,'review':r[0] in t['review_required_ayahs']} for r,e in zip(c['rows'],c['errors_ms']) if e>500]
 row=dict(model=c['model'],surah=c['surah'],version=t['fast_version'],ayahs=len(c['rows']),confirmed=len(conf),review=len(t['review_required_ayahs']),first_unresolved=first['ayah'] if first else None,first_unresolved_reason=first['review_reason'] if first else None,first_block=stop['ayah'] if stop else None,first_block_reason=stop['reason'] if stop else None,last_local_units=r['observed']['units'] if r else None,median_ms=allmed,p95_ms=allp95,max_ms=allmax,confirmed_median_ms=med,confirmed_p95_ms=p95,confirmed_max_ms=mx,over500=over,calls=len(t['acoustic_reads']),coverage=t['coverage'],input_ratio=t['audio_ms']/c['duration_ms'],decode_s=c['decode_seconds'],dsp_s=c['dsp_seconds']+t['dsp_seconds'],inference_s=t['inference_seconds'],executor_s=c['seconds'],total_excluding_load_s=c['decode_seconds']+c['dsp_seconds']+c['seconds'],peak_rss_self_bytes=c['peak_rss_self'],peak_rss_children_bytes=c['peak_rss_children'],denied=t['denied_acoustic_reads'])
 rows.append(row)
for model in dict.fromkeys(c['model'] for c in cases):
 cs=[c for c in cases if c['model']==model];rs=[r for r in rows if r['model']==model]
 totals.append(dict(model=model,confirmed=sum(r['confirmed'] for r in rs),review=sum(r['review'] for r in rs),calls=sum(r['calls'] for r in rs),inference_s=sum(r['inference_s'] for r in rs),executor_s=sum(r['executor_s'] for r in rs),confirmed_over500=[(r['surah'],e) for r in rs for e in r['over500'] if e['confirmed']],first_blocks=collections.Counter(r['first_block_reason'] for r in rs)))
(p/'baseline.json').write_text(json.dumps({'rows':rows,'totals':totals},ensure_ascii=False,indent=2))
with (p/'baseline.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Актуальный baseline','', 'Свежий frozen source sparse11. 4 модели ×12 полных сур =48 случаев, 607 аятов на модель. Все ошибки ниже относятся к соответствующему эталону; confirmed и estimates считаются отдельно. RSS — high-water процесса, children не включает живой worker; не сумма пиков. Total исключает загрузку; load отдельно в summary. DSP — внешняя подготовка + внутренняя фаза.','', '| Модель | Подтверждено | Ревью | Вызовы | Inference, с | Executor, с | Confirmed >500 мс |','|---|---:|---:|---:|---:|---:|---|']
for r in totals:lines.append(f"|{r['model']}|{r['confirmed']}|{r['review']}|{r['calls']}|{r['inference_s']:.3f}|{r['executor_s']:.3f}|{r['confirmed_over500']}|")
lines+=['','## Первые блоки','', '| Модель | Сура | Первый block | Причина | Последнее local decode |','|---|---:|---:|---|---|']
for r in rows:lines.append(f"|{r['model']}|{r['surah']}|{r['first_block']}|{r['first_block_reason']}|{r['last_local_units']}|")
lines+=['','Приоритеты: Tiny77:7 — наблюдаемый split (отдельно от многих прочих отказов); Base79:26 — текст подтверждён, но timing +508 мс; CTC/FP32 имеют иные слабые/неполные независимые чтения и не исправляются ASR split. Review после block — каскад, а не число независимых модельных ошибок. Причины first_block сохранены без переименования в уверенные акустические диагнозы. Для всех случаев ledger проверен runner spy; интервалы и runtime identity в raw JSON.']
(p/'baseline.md').write_text('\n'.join(lines)+'\n')
print(totals)
