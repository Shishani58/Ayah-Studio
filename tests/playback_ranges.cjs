const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../surah_cutter/webgui/static');
const c = {};
vm.createContext(c);vm.runInContext(fs.readFileSync(path.join(root,'playback.js'),'utf8'),c);
const a={currentTime:2,duration:60,paused:true,playbackRate:1.5,pause(){this.paused=true;},
  listeners:{},addEventListener(e,f){(this.listeners[e] ||= []).push(f);},
  emit(e){for(const f of this.listeners[e] || []) f();}};
function ready(){a.emit('loadedmetadata');a.emit('seeked');}
c.setAudioSource(a,'/api/audio/1');
ready();c.seekAudioMs(a,2000);a.emit('seeked');
c.setAudioPlaybackEnd(a,6483);
assert.equal(a.src,'/api/audio/1/clip?start_ms=2000&end_ms=6483');
// A source replacement resets native time to zero, just as the browser does.
a.currentTime=99;assert.equal(c.mediaTimeMs(a),2000); // WebKit may retain the old clock until metadata.
ready();assert.equal(a.currentTime,0);assert.equal(c.mediaDurationMs(a),60000);
c.seekAudioMs(a,3000);assert.equal(a.currentTime,1);assert.equal(c.mediaTimeMs(a),3000);
const original=a.src;c.setAudioPlaybackEnd(a,6483);assert.equal(a.src,original);
c.setAudioPlaybackEnd(a,null);assert.equal(a.src,'/api/audio/1');assert.equal(a.currentTime,3);
assert.equal(c.mediaDurationMs(a),60000);
ready();
assert.equal(a.playbackRate,1.5);
c.setAudioPlaybackEnd(a,6483);ready();
c.seekAudioMs(a,120);assert.equal(a.src,'/api/audio/1');assert.equal(a.currentTime,.12);
c.setAudioPlaybackEnd(a,6483);ready();a.currentTime=6.363;
assert.equal(c.mediaTimeMs(a),6483);
c.seekAudioMs(a,120);assert.equal(a.currentTime,0); // repeat reuses the same finite clip
c.setAudioSource(a,'/api/audio/112');assert.equal(c.mediaTimeMs(a),0);assert.equal(a._clipEndMs,null);
ready();

// The product's selected ayah / words use absolute time despite the clip offset.
const source=fs.readFileSync(path.join(root,'app.js'),'utf8');
function extract(name){const start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}
Object.assign(c,{R:{audio:a,dataReady:true,playbackAyah:0,playbackMode:'single',
  ayahs:[{ayah:1,time:1000},{ayah:2,time:5000}],duration:8000},W:{mode:false},
  clearTimeout(){},setTimeout:()=>1,currentAyahByTime:()=>0,finishWord(){},setCurrent(){},drawWave(){}});
vm.runInContext(['ayahEndMs','scheduleAyahStop','finishAyahPlayback','scheduleWordStop'].map(extract).join('\n'),c);
a.currentTime=1;c.scheduleAyahStop();assert.equal(a.src,'/api/audio/112/clip?start_ms=1000&end_ms=5000');
ready();
a.currentTime=4;assert.equal(c.finishAyahPlayback(true),true);assert.equal(c.mediaTimeMs(a),4998);
c.R.playbackMode='continuous';c.scheduleAyahStop();assert.equal(a.src,'/api/audio/112');
c.W.mode=true;c.W.stop={end:6000};c.scheduleWordStop();
assert.equal(a._clipEndMs,6000);c.scheduleAyahStop();assert.equal(a._clipEndMs,6000);
// A previous clip's native duration can survive a rapid source replacement.
a.duration=.5;c.setAudioPlaybackEnd(a,null);c.setAudioPlaybackEnd(a,7000);
assert.equal(c.mediaDurationMs(a),60000);
console.log('Finite preview ranges, clocks and playback integration checks passed');
