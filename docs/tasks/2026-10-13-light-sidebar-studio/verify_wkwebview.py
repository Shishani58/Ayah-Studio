# Run with the isolated copy server on 8879; no edits to the source project.
import time
import json
import threading
from pathlib import Path
import webview

window = webview.create_window(
    "Ayah Studio — UI verification", "http://127.0.0.1:8879", width=1440, height=940
)
out = Path(__file__).resolve().parent


def wait_js(script):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if window.evaluate_js(script):
            return
        time.sleep(0.1)
    raise AssertionError("timeout: " + script)


def verify():
    try:
        wait_js("dashboardSuras.length === 114")
        window.evaluate_js(
            "$('suraSearch').value='114'; renderSuras(); $('grid').querySelector('button').click()"
        )
        wait_js("R.dataReady && R.waveReady && R.audioReady")
        assert window.evaluate_js("R.ayahs.length") == 6
        old_width = window.evaluate_js("$('review').clientWidth")
        window.evaluate_js("$('sidebarToggle').click()")
        wait_js(f"$('review').clientWidth > {old_width}")
        assert window.evaluate_js("$('review').clientWidth") > old_width, (
            old_width,
            window.evaluate_js(
                "({width:$('review').clientWidth,hidden:$('suraSidebar').hidden,body:document.body.scrollWidth,inner:innerWidth})"
            ),
        )
        window.evaluate_js(
            "$('sidebarToggle').click(); $('rate').value='1.5'; $('rate').dispatchEvent(new Event('change')); $('playBtn').click()"
        )
        time.sleep(0.3)
        assert window.evaluate_js("R.audio.paused") is False
        window.evaluate_js("$('playBtn').click(); setMode('words')")
        wait_js("W.data !== null && !W.loading")
        window.evaluate_js("setWaveZoom(8); selectWord(1,false)")
        assert window.evaluate_js("R.zoom") == 8
        window.evaluate_js(
            "setMode('full'); setWaveZoom(1); $('rate').value='1'; $('rate').dispatchEvent(new Event('change')); $('suraSearch').value=''; renderSuras(); $('grid').querySelector('[data-surah=\"114\"]').scrollIntoView({block:'end'}); $('ayahList').scrollTop=0; R.audio.currentTime=11; setCurrent(1); setStatus(''); drawWave(); document.activeElement.blur()"
        )
        time.sleep(0.4)
        report = window.evaluate_js(
            "({engine:navigator.userAgent,rows:R.ayahs.length,sidebar:$('suraSidebar').getBoundingClientRect().width,overflow:document.documentElement.scrollWidth>innerWidth,summaryAboveWave:$('reviewTitle').getBoundingClientRect().bottom<$('wave').getBoundingClientRect().top,firstRow:$('ayahList').firstElementChild.getBoundingClientRect().top})"
        )
        assert not report["overflow"]
        assert report["summaryAboveWave"]
        (out / "wkwebview.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        from webview.platforms.cocoa import BrowserView
        from PyObjCTools import AppHelper
        import AppKit

        done = threading.Event()

        def snapshot():
            view = BrowserView.instances[window.uid].webview

            def complete(image, error):
                if error:
                    print(error)
                else:
                    rep = AppKit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
                    data = rep.representationUsingType_properties_(
                        AppKit.NSBitmapImageFileTypePNG, {}
                    )
                    data.writeToFile_atomically_(str(out / "implementation-wkwebview.png"), True)
                done.set()

            view.takeSnapshotWithConfiguration_completionHandler_(None, complete)

        AppHelper.callAfter(snapshot)
        assert done.wait(15)
        print(
            "PASS WKWebView: canonical six rows, streaming waveform, sidebar geometry, playback/speed, words, zoom persistence, actual audio and copied project DB; screenshot saved.",
            flush=True,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("FAIL", repr(e), flush=True)
    finally:
        window.destroy()


webview.start(verify, gui="cocoa")
