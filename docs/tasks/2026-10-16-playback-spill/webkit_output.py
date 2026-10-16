"""Record pre/post-guard output in an isolated WKWebView, not just its clock."""
from pathlib import Path

path = Path(__file__).with_name("webkit_probe.py")
source = path.read_text()
a = source.index('javascript = """')
b = source.index('result = None', a)
script = r"""
const helper=HELPER;
(async()=>{
  const NativeContext=window.AudioContext||window.webkitAudioContext;
  const results=[];
  for(const guarded of [false,true])for(const cut of [6483,11460,15937,21577,27740,33395]){
    const ctx=new NativeContext({sampleRate:16000});
    const reference=(await ctx.decodeAudioData(await (await fetch('http://127.0.0.1:8769/api/audio/1')).arrayBuffer())).getChannelData(0);
    const audio=new Audio('http://127.0.0.1:8769/api/audio/1');
    await new Promise(resolve=>{audio.onloadedmetadata=resolve;audio.load();});
    let node,gain;
    const createSource=ctx.createMediaElementSource.bind(ctx),createGain=ctx.createGain.bind(ctx);
    ctx.createMediaElementSource=a=>{node=createSource(a);return node;};
    ctx.createGain=()=>{gain=createGain();return gain;};
    window.AudioContext=function(){return ctx;};
    (0,eval)(helper);
    if(guarded)setAudioPlaybackEnd(audio,cut);
    else {node=ctx.createMediaElementSource(audio);gain=ctx.createGain();node.connect(gain);gain.connect(ctx.destination);}
    const merge=ctx.createChannelMerger(2),proc=ctx.createScriptProcessor(1024,2,1);
    node.connect(merge,0,0);gain.connect(merge,0,1);merge.connect(proc);proc.connect(ctx.destination);
    const chunks=[];
    proc.onaudioprocess=e=>chunks.push([new Float32Array(e.inputBuffer.getChannelData(0)),new Float32Array(e.inputBuffer.getChannelData(1))]);
    await ctx.resume();
    audio.currentTime=(cut-1800)/1000;
    await new Promise(resolve=>audio.addEventListener('seeked',resolve,{once:true}));
    await audio.play();
    await new Promise(resolve=>{
      const stop=()=>{if(audio.currentTime*1000<cut){setTimeout(stop,2);return;}audio.pause();resolve();};stop();
    });
    const stopped=audio.currentTime*1000;
    await new Promise(resolve=>setTimeout(resolve,200));
    const pre=new Float32Array(chunks.length*1024),post=new Float32Array(pre.length);
    chunks.forEach((c,i)=>{pre.set(c[0],i*1024);post.set(c[1],i*1024);});
    let last=post.length-1;while(last>0&&Math.abs(post[last])<1e-7)last--;
    const length=6400,from=Math.max(0,last-length);
    let best={score:-1};
    for(let base=Math.round((cut-1400)*16);base<Math.round((cut+400)*16);base++){
      let dot=0,a=0,b=0;
      for(let j=0;j<length;j+=8){const u=pre[from+j]||0,v=reference[base+j]||0;dot+=u*v;a+=u*u;b+=v*v;}
      const score=dot/Math.sqrt(a*b||1);
      if(score>best.score)best={score,source_end_ms:(base+last-from)/16};
    }
    results.push({guarded,cut_ms:cut,stopped_ms:stopped,sample_rate:ctx.sampleRate,
      ...best,output_offset_ms:best.source_end_ms-cut,nonzero_samples:last});
    proc.disconnect();merge.disconnect();node.disconnect();gain.disconnect();await ctx.close();
  }
  window.webkit.messageHandlers.result.postMessage(JSON.stringify(results));
})().catch(e=>window.webkit.messageHandlers.result.postMessage(JSON.stringify({error:String(e)})));
"""
import json
script = script.replace("HELPER", json.dumps(path.with_name("gain-prototype.js").read_text()))
source = source[:a] + "javascript = " + repr(script) + "\n" + source[b:]
source = source.replace('("webkit-guard.json" if args.guard else "webkit-blocked.json")', '"webkit-output.json"')
source = source.replace('time.monotonic() + 55', 'time.monotonic() + 55')
exec(compile(source, str(path), "exec"))
