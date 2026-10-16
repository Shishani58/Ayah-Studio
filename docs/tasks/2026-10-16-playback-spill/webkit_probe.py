"""Isolated native WKWebView test, the same engine as pywebview on macOS.

Creates its own offscreen window; never attaches to or edits the user's GUI.
"""
import argparse
import json
from pathlib import Path
import re
import time

import AppKit
import Foundation
import WebKit

TASK = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--guard", action="store_true")
args = parser.parse_args()
source = (TASK / "timer-player.js").read_text()
names = ["ayahEndMs", "scheduleAyahStop", "finishAyahPlayback"]
functions = "\n".join(re.search(r"function " + n + r"\([^\n]*[\s\S]*?\n}", source)[0] for n in names)
if args.guard:
    functions = (TASK / "gain-prototype.js").read_text() + functions
else:
    functions = "const setAudioPlaybackEnd=()=>{};\n" + functions
rows = json.loads((TASK / "user-full.json").read_text())["trace"]["ayahs"]
javascript = """
const NativeContext=window.AudioContext || window.webkitAudioContext;
window.AudioContext=function(){
  const ctx=new NativeContext();window.probeContext=ctx;const createGain=ctx.createGain.bind(ctx);
  ctx.createGain=()=>{window.probeGain=createGain();return window.probeGain;};
  return ctx;
};
const R={audio:new Audio('http://127.0.0.1:8769/api/audio/1'),dataReady:true,
  ayahs:ROWS.map(r=>({time:r.onset_ms})),duration:59010.625,playbackMode:'single',
  loop:false,playbackAyah:0};
const W={mode:false}; const setCurrent=()=>{},drawWave=()=>{};
FUNCTIONS
(async()=>{
  await new Promise((resolve,reject)=>{R.audio.onloadedmetadata=resolve;R.audio.onerror=reject;R.audio.load();});
  const stops=[],busy=[];let finish;
  const pause=R.audio.pause.bind(R.audio);
  R.audio.pause=()=>{stops.push({actual_ms:R.audio.currentTime*1000,
    gain_at_late_stop:window.probeGain?.gain.value ?? null});pause();finish?.();};
  for(const rate of [1,2]) for(let i=0;i<6;i++) {
    const cut=R.ayahs[i+1].time;
    R.audio.playbackRate=rate;R.playbackAyah=i;R.audio.currentTime=(cut-1500)/1000;
    await new Promise(resolve=>R.audio.addEventListener('seeked',resolve,{once:true}));
    const done=new Promise(resolve=>{finish=resolve;});
    await R.audio.play();setAudioPlaybackEnd(R.audio,cut);scheduleAyahStop();
    // Force an ordinary long UI task to straddle the boundary.
    const left=(cut-R.audio.currentTime*1000)/rate;
    setTimeout(()=>{
      const start=R.audio.currentTime*1000,until=performance.now()+350;
      while(performance.now()<until){};
      busy.push({cut,start_ms:start,end_ms:R.audio.currentTime*1000,gain:window.probeGain?.gain.value??null});
    },Math.max(0,left-100));
    await done;
    const s=stops[stops.length-1];Object.assign(s,{ayah:i+1,rate,cut_ms:cut,overshoot_ms:s.actual_ms-cut});
  }
  window.webkit.messageHandlers.result.postMessage(JSON.stringify({stops,busy,duration_ms:R.audio.duration*1000}));
})().catch(e=>window.webkit.messageHandlers.result.postMessage(JSON.stringify({error:String(e)})));
""".replace("ROWS", json.dumps(rows)).replace("FUNCTIONS", functions)
result = None


class Receiver(Foundation.NSObject):
    def userContentController_didReceiveScriptMessage_(self, controller, message):
        global result
        result = json.loads(str(message.body()))


app = AppKit.NSApplication.sharedApplication()
app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyProhibited)
config = WebKit.WKWebViewConfiguration.alloc().init()
config.setMediaTypesRequiringUserActionForPlayback_(0)
receiver = Receiver.alloc().init()
config.userContentController().addScriptMessageHandler_name_(receiver, "result")
view = WebKit.WKWebView.alloc().initWithFrame_configuration_(Foundation.NSMakeRect(0, 0, 800, 600), config)
window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
    Foundation.NSMakeRect(-2000, -2000, 800, 600), 0, AppKit.NSBackingStoreBuffered, False)
window.setContentView_(view)
view.loadHTMLString_baseURL_("<html><body><script>" + javascript + "</script></body></html>",
                            Foundation.NSURL.URLWithString_("http://127.0.0.1:8769/"))
deadline = time.monotonic() + 55
while result is None and time.monotonic() < deadline:
    Foundation.NSRunLoop.currentRunLoop().runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(.05))
if result is None:
    result = {"error": "WKWebView offscreen probe timed out"}
(TASK / ("webkit-guard.json" if args.guard else "webkit-blocked.json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
config.userContentController().removeScriptMessageHandlerForName_("result")
view.stopLoading()
window.orderOut_(None)
