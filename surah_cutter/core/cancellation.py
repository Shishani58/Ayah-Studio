"""Thread-local generation cancellation, including blocking native calls."""

from contextlib import contextmanager
import threading


class GenerationCancelled(BaseException):
    """Control flow, deliberately bypassing alignment error/fallback handlers."""


_LOCAL = threading.local()


def current_event():
    return getattr(_LOCAL, "event", None)


def check_cancelled():
    event = current_event()
    if event is not None and event.is_set():
        raise GenerationCancelled()


@contextmanager
def cancellation_scope(event):
    previous = current_event()
    _LOCAL.event = event
    try:
        yield
    finally:
        _LOCAL.event = previous


@contextmanager
def interruptible(abort):
    """Abort a blocking call on cancellation; retire watcher before reuse."""
    event = current_event()
    if event is None:
        yield
        return
    if event.is_set():
        abort()
        raise GenerationCancelled()
    done = threading.Event()

    def watch():
        while not done.wait(0.05):
            if event.is_set():
                abort()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join()
        check_cancelled()
