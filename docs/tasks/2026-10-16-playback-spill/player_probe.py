"""Actual shipped ayah selection/stop functions with native finite sources."""
from pathlib import Path
import json
import re

path = Path(__file__).with_name("webkit_probe.py")
source = path.read_text()
a = source.index('javascript = """')
b = source.index('result = None', a)
app = (path.parents[3] / "surah_cutter/webgui/static/app.js").read_text()
names = ["currentAyahByTime", "ayahEndMs", "seekAyah", "scheduleAyahStop", "finishAyahPlayback"]
functions = "\n".join(re.search(r"function " + n + r"\([^\n]*[\s\S]*?\n}", app)[0] for n in names)
rows = json.loads(path.with_name("user-full.json").read_text())["trace"]["ayahs"]
script = r"""
HELPER
const R={audio:new Audio(),dataReady:true,audioReady:true,ayahs:ROWS.map(r=>({time:r.onset_ms})),
  duration:59011,playbackMode:'single',loop:false,playbackAyah:0,current:0};
const W={mode:false};
const setCurrent=i=>{R.current=i;},drawWave=()=>{},scrollWaveToMs=()=>{},syncPlaybackRate=()=>{};
FUNCTIONS
(async()=>{
  setAudioSource(R.audio,'http://127.0.0.1:8769/api/audio/1');
  await new Promise(resolve=>{R.audio.onloadedmetadata=resolve;R.audio.load();});
  for(const event of ['play','seeking','seeked'])R.audio.addEventListener(event,()=>scheduleAyahStop(true));
  for(const event of ['pause','ratechange'])R.audio.addEventListener(event,()=>scheduleAyahStop());
  R.audio.addEventListener('ended',()=>finishAyahPlayback(true));
  R.audio.addEventListener('timeupdate',()=>finishAyahPlayback(R.audio.ended));
  const results=[];let finish;
  const pause=R.audio.pause.bind(R.audio);
  R.audio.pause=()=>{const end=mediaTimeMs(R.audio);pause();if(end>=ayahEndMs(R.playbackAyah)-1)finish?.(end);};
  for(let i=0;i<6;i++){
    R.audio.playbackRate=2;
    const done=new Promise(resolve=>{finish=resolve;});
    seekAyah(i,true);
    const end=await Promise.race([done,new Promise((_,reject)=>setTimeout(()=>reject(Error('timeout '+i)),10000))]);
    results.push({ayah:i+1,cut_ms:ayahEndMs(i),actual_end_ms:end,selected:R.current,
      clip_start_ms:R.audio._clipStartMs,clip_end_ms:R.audio._clipEndMs});
  }
  window.webkit.messageHandlers.result.postMessage(JSON.stringify(results));
})().catch(e=>window.webkit.messageHandlers.result.postMessage(JSON.stringify({error:String(e)})));
"""
script = script.replace("HELPER", (path.parents[3] / "surah_cutter/webgui/static/playback.js").read_text())
script = script.replace("ROWS", json.dumps(rows)).replace("FUNCTIONS", functions)
source = source[:a] + "javascript = " + repr(script) + "\n" + source[b:]
source = source.replace('("webkit-guard.json" if args.guard else "webkit-blocked.json")', '"player-webkit.json"')
exec(compile(source, str(path), "exec"))
