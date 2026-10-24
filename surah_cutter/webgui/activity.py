"""Keep an explicitly requested calculation out of macOS App Nap."""

from contextlib import contextmanager
import logging
import sys


@contextmanager
def generation_activity():
    """Balance the native activity on success, failure and cancellation.

    The display may sleep, but an in-flight calculation must keep running even
    when the user hides the window. PyObjC is already supplied by macOS pywebview;
    browser-only installations and other platforms do not require it.
    """
    if sys.platform != "darwin":
        yield
        return
    try:
        from Foundation import NSActivityUserInitiated, NSProcessInfo
        from objc import autorelease_pool
    except ImportError:
        logging.getLogger(__name__).warning("macOS calculation activity unavailable: PyObjC is not installed")
        yield
        return
    with autorelease_pool():
        process = NSProcessInfo.processInfo()
        # This is a long CPU calculation, not latency-critical media I/O.
        # Keep App Nap protection without requesting the strongest urgency.
        token = process.beginActivityWithOptions_reason_(
            NSActivityUserInitiated, "Ayah Studio calculation",
        )
        try:
            yield
        finally:
            process.endActivity_(token)
