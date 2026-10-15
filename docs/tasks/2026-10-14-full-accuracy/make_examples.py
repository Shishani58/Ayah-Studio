"""Standalone waveform/audio review page. No claim of human listening."""
import base64
import html
import io
import json
from pathlib import Path
import sys
import wave
import numpy as np

TASK=Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parents[2]))
from surah_cutter.core import decode  # noqa: E402
data=json.loads((TASK/"diagnostics.json").read_text())
selection=[("mahmoud_khalil_al_hussary",80,26),("abdul_basit_mujawwad",98,1),
           ("abdul_basit_mujawwad",98,7),("abdul_basit_mujawwad",80,41),
           ("abu_bakr_ash_shatri",42,1),("abu_bakr_ash_shatri",83,10),
           ("abu_bakr_ash_shatri",95,8),("saad_al_ghamidi",98,1),
           ("mohammed_siddiq_al_minshawi",98,1),("abu_bakr_ash_shatri",112,1)]
parts=["<!doctype html><meta charset=utf-8><title>FULL: проверка границ</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#172033}section{margin:40px 0}svg{width:100%;background:#f4f6fa}button{margin:8px;padding:8px}audio{width:100%}</style><h1>FULL: спорные начала и контрольные примеры</h1><p>Синий: RMS по20мс; зелёный: GT; красный: прежний FULL; оранжевый: локальное исправление42:1; серый: DSP-пауза. Это сигнальный/CTC-аудит, слуховая проверка агентом не выполнена. Метки GT не передавались модели.</p>"]
features=[]
for index,(name,surah,ayah) in enumerate(selection):
    s=next(s for s in data if s["reciter"]==name and s["surah"]==surah)
    row=s["rows"][ayah-1]
    gt=row[1]-s["errors_ms"][ayah-1]
    d=s["details"][ayah-1]
    tokens=d["tokens"][:d["first_word_tokens"]]
    first=tokens[0]["start_ms"]
    strong=next((t["start_ms"] for t in tokens if t["conf"]>=.45),first)
    lo=max(0,min(gt,row[1],first)-1500)
    hi=max(gt,row[1],strong)+1800
    pcm=decode.decode_range(s["audio"],start_ms=lo,end_ms=hi)
    hi=lo+len(pcm)/16
    rms=np.sqrt(np.mean(pcm[:len(pcm)//320*320].reshape(-1,320)**2,axis=1))
    scale=max(float(np.percentile(rms,98)),1e-6)
    def x(ms):
        return 40+(ms-lo)/(hi-lo)*1000
    pause=next((p for p in s["pauses"] if p["start_ms"]<=gt<=p["end_ms"]),None)
    svg=['<svg viewBox="0 0 1080 240" role="img" aria-label="RMS и метки времени">']
    for p in s["pauses"]:
        if p["end_ms"]>lo and p["start_ms"]<hi:
            svg.append(f'<rect x="{x(max(lo,p["start_ms"])):.1f}" y="40" width="{(min(hi,p["end_ms"])-max(lo,p["start_ms"]))*1000/(hi-lo):.1f}" height="130" fill="#dde2e8"/>')
    points=" ".join(f"{x(lo+i*20):.1f},{170-min(float(r)/scale,1)*120:.1f}" for i,r in enumerate(rms))
    svg.append(f'<polyline points="{points}" fill="none" stroke="#345ea8" stroke-width="1.5"/>')
    marks=[("GT",gt,"#168448"),("FULL",row[1],"#bb223d")]
    if surah==42:
        marks.append(("FIX",6220,"#a65c00"))
    for j,(label,ms,color) in enumerate(marks):
        svg.append(f'<line x1="{x(ms):.1f}" x2="{x(ms):.1f}" y1="30" y2="185" stroke="{color}" stroke-width="2"/><text x="{x(ms):.1f}" y="{15+j*14}" font-size="12" fill="{color}">{label} {ms/1000:.3f}</text>')
    for sec in range(int(lo//1000)+1,int(hi//1000)+1):
        svg.append(f'<text x="{x(sec*1000):.1f}" y="205" font-size="12">{sec}s</text>')
    svg.append('</svg>')
    buffer=io.BytesIO()
    with wave.open(buffer,"wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(pcm,-1,1)*32767).astype("<i2").tobytes())
    wav_path=Path.home()/f".cache/surah-cutter/full-accuracy/{name}-{surah}-{ayah}.wav"
    wav_path.write_bytes(buffer.getvalue())
    audio_id=f"a{index}"
    buttons="".join(f'<button onclick="let a=document.getElementById(\'{audio_id}\');a.currentTime={(ms-lo)/1000:.3f};a.play()">С {label}</button>' for label,ms,_ in marks)
    note=f'GT {gt}мс; FULL {row[1]}мс; разница {row[1]-gt:+d}мс. Первый токен {tokens[0]["token"]}: confidence={tokens[0]["conf"]}.'
    if pause:
        note+=f' DSP-пауза {pause["start_ms"]:.0f}–{pause["end_ms"]:.0f}мс.'
    parts.append(f'<section><h2>{html.escape(name)} {surah}:{ayah} — {html.escape(d["first_word"])}</h2><p>{note}</p>'+"".join(svg)+f'<audio id="{audio_id}" controls src="data:audio/wav;base64,{base64.b64encode(buffer.getvalue()).decode()}"></audio>'+buttons+"</section>")
    features.append({"reciter":name,"surah":surah,"ayah":ayah,"gt_ms":gt,"full_ms":row[1],
                     "first_token_ms":first,"first_strong_token_ms":strong,"gt_pause":pause,
                     "clip":str(wav_path),"clip_start_ms":lo,"clip_end_ms":hi,
                     "first_word_tokens":tokens})
(TASK/"examples.html").write_text("\n".join(parts))
(TASK/"examples.json").write_text(json.dumps(features,ensure_ascii=False,indent=2)+"\n")
print("10 waveform/audio examples saved")
