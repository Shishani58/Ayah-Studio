# Native UI check on disposable fixture server 8878, no user data.
import time
import threading
import json
from pathlib import Path
import webview

window = webview.create_window(
    "Ayah Studio — UI verification", "http://127.0.0.1:8878", width=1440, height=940
)
out = Path(__file__).resolve().parent
failed = []


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
        window.evaluate_js("$('grid').querySelector('[data-surah=\"114\"]').click()")
        wait_js("R.dataReady && R.audioReady && R.waveReady")
        assert window.evaluate_js("document.querySelector('.ayah-table-head') === null")
        assert window.evaluate_js("$('reviewProject').textContent.includes('ayah-studio-verify')")
        window.evaluate_js(
            "applyAppearance('dark',30); $('ayahList').children[1].querySelector('.a-num').click()"
        )
        assert window.evaluate_js("$('ayahTime2').open")
        assert window.evaluate_js("$('ayahTime2').contains(document.activeElement)")
        window.evaluate_js(
            "const input=$('ayahTime2').querySelector('input'); input.value='00:02.160'; input.dispatchEvent(new Event('change'))"
        )
        wait_js("R.ayahs[1].time === 2160 && !editBusy")
        report = window.evaluate_js(
            "({engine:navigator.userAgent,rows:R.ayahs.length,modal:$('ayahTime2').open,overflow:document.documentElement.scrollWidth>innerWidth,time:R.ayahs[1].time})"
        )
        assert not report["overflow"]
        (out / "wkwebview.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        from webview.platforms.cocoa import BrowserView
        from PyObjCTools import AppHelper
        import AppKit

        done = threading.Event()

        def snapshot():
            view = BrowserView.instances[window.uid].webview

            def complete(image, error):
                if error:
                    failed.append(str(error))
                else:
                    rep = AppKit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
                    data = rep.representationUsingType_properties_(
                        AppKit.NSBitmapImageFileTypePNG, {}
                    )
                    data.writeToFile_atomically_(str(out / "popover-wkwebview.png"), True)
                done.set()

            view.takeSnapshotWithConfiguration_completionHandler_(None, complete)

        AppHelper.callAfter(snapshot)
        assert done.wait(15)
        window.evaluate_js("$('ayahTime2').close()")
        wait_js("document.activeElement === $('ayahList').children[1].querySelector('.a-num')")
        print(
            "PASS WKWebView: compact rows, project summary, native dialog/focus, timing API save, no overflow; screenshot saved.",
            flush=True,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        failed.append(str(e))
    finally:
        window.destroy()


webview.start(verify, gui="cocoa")
if failed:
    raise SystemExit(str(failed))
