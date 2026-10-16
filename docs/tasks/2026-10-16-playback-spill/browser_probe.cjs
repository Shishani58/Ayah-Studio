// Isolated headless regression with the shipped playback functions and real MP3.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('/Applications/ChatGPT.app/Contents/Resources/cua_node/lib/node_modules/playwright');
const task = __dirname;
const app = fs.readFileSync(path.join(task, 'timer-player.js'), 'utf8');
function extract(name) {
  const a = app.search(new RegExp(`function ${name}\\(`));
  if (a < 0) throw Error(name);
  return app.slice(a, app.indexOf('\n}', a) + 2);
}
(async () => {
  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true, args: ['--autoplay-policy=no-user-gesture-required'],
  });
  try {
    const page = await browser.newPage();
    // The local application is only read: project settings and DB stay untouched.
    await page.goto('http://127.0.0.1:8769/api/project');
    const rows = JSON.parse(fs.readFileSync(path.join(task, 'user-full.json'))).trace.ayahs;
    const probes = await page.evaluate(async ({rows, code}) => {
      const R = window.R = { audio: new Audio('/api/audio/1'), dataReady: true,
        ayahs: rows.map(r => ({time:r.onset_ms})), duration: 59010.625,
        playbackMode:'single', loop:false, playbackAyah:0 };
      window.W = { mode: false };
      window.setCurrent = () => {}; window.drawWave = () => {};
      // Script evaluation belongs to this isolated test page, not the user's GUI.
      (0, eval)(code);
      await new Promise((resolve, reject) => {
        R.audio.onloadedmetadata = resolve; R.audio.onerror = reject; R.audio.load();
      });
      const stops = [];
      let finish;
      const pause = R.audio.pause.bind(R.audio);
      R.audio.pause = () => {
        stops.push({actual_ms:R.audio.currentTime*1000});
        pause(); finish?.();
      };
      for (const rate of [1, 2]) for (let i=0;i<6;i++) {
        const cut = R.ayahs[i+1].time;
        R.audio.playbackRate = rate; R.playbackAyah = i;
        R.audio.currentTime = (cut-350)/1000;
        await new Promise(resolve => R.audio.addEventListener('seeked', resolve, {once:true}));
        const ended = new Promise(resolve => { finish=resolve; });
        await R.audio.play(); scheduleAyahStop();
        await ended;
        const sample = stops.at(-1);
        Object.assign(sample, {ayah:i+1,rate,cut_ms:cut,overshoot_ms:sample.actual_ms-cut});
      }
      return {stops,duration_ms:R.audio.duration*1000};
    }, {rows,code:['ayahEndMs','scheduleAyahStop','finishAyahPlayback'].map(extract).join('\n')});
    fs.writeFileSync(path.join(task, 'browser-before.json'), JSON.stringify(probes,null,2)+'\n');
    console.log(JSON.stringify(probes,null,2));
    const windows=JSON.parse(fs.readFileSync(path.join(task,'decoder-windows.json')));
    const offsets=await page.evaluate(async windows=>{
      const context=new AudioContext({sampleRate:16000});
      const data=await (await fetch('/api/audio/1')).arrayBuffer();
      const buffer=await context.decodeAudioData(data);
      const pcm=buffer.getChannelData(0);
      const offsets=windows.map(w=>{
        let best={score:-1};
        for(let shift=-3200;shift<=3200;shift++) {
          let dot=0,a=0,b=0;
          for(let i=0;i<w.samples.length;i+=4) {
            const v=pcm[w.start_ms*16+i+shift]||0, u=w.samples[i];
            dot+=v*u;a+=v*v;b+=u*u;
          }
          const score=dot/Math.sqrt(a*b||1);
          if(score>best.score) best={score,shift_ms:shift/16};
        }
        return {start_ms:w.start_ms,...best};
      });
      await context.close();return {duration_ms:buffer.duration*1000,offsets};
    },windows);
    fs.writeFileSync(path.join(task,'decoder-offset.json'),JSON.stringify(offsets,null,2)+'\n');
    console.log(JSON.stringify(offsets,null,2));
    await page.setContent(fs.readFileSync(path.join(task,'waveforms.svg'),'utf8'));
    await page.setViewportSize({width:1400,height:1200});
    await page.screenshot({path:path.join(task,'waveforms.png'),fullPage:true});
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
