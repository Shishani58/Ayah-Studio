// Correlate actual media-element output after MP3 seeks with decoded PCM.
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('/Applications/ChatGPT.app/Contents/Resources/cua_node/lib/node_modules/playwright');
(async()=>{
  const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless:true,args:['--autoplay-policy=no-user-gesture-required']});
  try {
    const page=await browser.newPage();
    await page.goto('http://127.0.0.1:8769/api/project');
    const result=await page.evaluate(async()=>{
      const ctx=new AudioContext({sampleRate:16000});
      const data=await (await fetch('/api/audio/1')).arrayBuffer();
      const ref=(await ctx.decodeAudioData(data)).getChannelData(0);
      const audio=new Audio('/api/audio/1');
      await new Promise(resolve=>{audio.onloadedmetadata=resolve;audio.load();});
      const node=ctx.createMediaElementSource(audio), proc=ctx.createScriptProcessor(1024,1,1);
      node.connect(proc);proc.connect(ctx.destination); await ctx.resume();
      let chunks=[];
      proc.onaudioprocess=e=>{chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));};
      const result=[];
      for(const cut of [6483,11460,15937,21577,27740,33395]) {
        audio.currentTime=(cut-1200)/1000;
        await new Promise(resolve=>audio.addEventListener('seeked',resolve,{once:true}));
        chunks=[];await audio.play();
        await new Promise(resolve=>{
          const stop=()=>{
            if(audio.currentTime*1000<cut) {setTimeout(stop,1);return;}
            audio.pause();resolve();
          };stop();
        });
        const stopped=audio.currentTime*1000;
        // Drain only this isolated test's graph; no product sleeps or waits.
        await new Promise(resolve=>setTimeout(resolve,200));
        const pcm=new Float32Array(chunks.length*1024); chunks.forEach((c,i)=>pcm.set(c,i*1024));
        let last=pcm.length-1;while(last>0&&Math.abs(pcm[last])<1e-7)last--;
        // Correlate an audible 500 ms slice ending 100 ms before output stops.
        const end=Math.max(8000,last-1600), start=end-8000;
        let best={score:-1};
        for(let shift=-16000;shift<16000;shift++) {
          const base=Math.round((cut-600)*16)+shift;
          let dot=0,a=0,b=0;
          for(let j=0;j<8000;j+=8) {
            const u=pcm[start+j],v=ref[base+j]||0;dot+=u*v;a+=u*u;b+=v*v;
          }
          const score=dot/Math.sqrt(a*b||1);
          if(score>best.score)best={score,end_in_source_ms:(base+last-start)/16};
        }
        result.push({cut_ms:cut,stopped_ms:stopped,last_nonzero_output_sample:last,
          ...best,output_past_cut_ms:best.end_in_source_ms-cut});
      }
      audio.pause();proc.disconnect();node.disconnect();await ctx.close();return result;
    });
    fs.writeFileSync(path.join(__dirname,'output-before.json'),JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify(result,null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
