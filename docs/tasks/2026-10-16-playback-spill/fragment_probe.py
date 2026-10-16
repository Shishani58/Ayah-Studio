"""Check native temporal-fragment support without product mutations."""
from pathlib import Path

path = Path(__file__).with_name("webkit_probe.py")
source = path.read_text()
a = source.index('javascript = """')
b = source.index('result = None', a)
source = source[:a] + '''javascript = """
(async()=>{
  const result=[];
  for(const cut of [6483,11460,15937,21577,27740,33395]){
    const audio=new Audio('http://127.0.0.1:8769/api/audio/1#t='+((cut-600)/1000)+','+(cut/1000));
    await new Promise(resolve=>{audio.onloadedmetadata=resolve;audio.load();});
    const stopped=new Promise(resolve=>{audio.onpause=()=>resolve('pause');audio.onended=()=>resolve('ended');});
    await audio.play();
    const state=await Promise.race([stopped,new Promise(resolve=>setTimeout(()=>resolve('timeout'),2000))]);
    result.push({cut_ms:cut,state,current_ms:audio.currentTime*1000,paused:audio.paused});
    audio.pause();
  }
  window.webkit.messageHandlers.result.postMessage(JSON.stringify(result));
})().catch(e=>window.webkit.messageHandlers.result.postMessage(JSON.stringify({error:String(e)})));
"""
''' + source[b:]
source = source.replace('("webkit-guard.json" if args.guard else "webkit-blocked.json")', '"fragment.json"')
exec(compile(source, str(path), "exec"))
