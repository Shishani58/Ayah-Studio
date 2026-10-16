"""Test the shipped finite-source helper in the native desktop media engine."""
from pathlib import Path
import json

path = Path(__file__).with_name("webkit_probe.py")
source = path.read_text()
a = source.index('javascript = """')
b = source.index('result = None', a)
script = r"""
HELPER
(async()=>{
  const results=[];
  for(const rate of [1,2])for(const cut of [6483,11460,15937,21577,27740,33395]){
    const audio=new Audio();setAudioSource(audio,'http://127.0.0.1:8769/api/audio/1');
    await new Promise(resolve=>{audio.onloadedmetadata=resolve;audio.load();});
    seekAudioMs(audio,cut-800);setAudioPlaybackEnd(audio,cut);
    await new Promise(resolve=>audio.addEventListener('loadedmetadata',resolve,{once:true}));
    audio.playbackRate=rate;
    const done=new Promise(resolve=>audio.addEventListener('ended',resolve,{once:true}));
    await audio.play();
    setTimeout(()=>{const until=performance.now()+350;while(performance.now()<until){};},300/rate);
    await done;
    results.push({cut_ms:cut,rate,clip_start_ms:audio._clipStartMs,clip_end_ms:audio._clipEndMs,
      duration_ms:audio.duration*1000,absolute_end_ms:mediaTimeMs(audio),paused:audio.paused,ended:audio.ended});
  }
  window.webkit.messageHandlers.result.postMessage(JSON.stringify(results));
})().catch(e=>window.webkit.messageHandlers.result.postMessage(JSON.stringify({error:String(e)})));
"""
script = script.replace("HELPER", (path.parents[3] / "surah_cutter/webgui/static/playback.js").read_text())
source = source[:a] + "javascript = " + repr(script) + "\n" + source[b:]
source = source.replace('("webkit-guard.json" if args.guard else "webkit-blocked.json")', '"bounded-webkit.json"')
exec(compile(source, str(path), "exec"))
