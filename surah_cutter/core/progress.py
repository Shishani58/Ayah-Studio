"""Optional structured work progress, scoped to the generation thread."""

from contextlib import contextmanager
import threading
import time


_LOCAL = threading.local()


@contextmanager
def progress_scope(callback):
    previous = vars(_LOCAL).copy()
    _LOCAL.reporter = callback
    _LOCAL.stage = None
    try:
        yield
    finally:
        vars(_LOCAL).clear()
        vars(_LOCAL).update(previous)


def current_stage():
    return getattr(_LOCAL, "stage", None)


def report(stage, done=None, total=None):
    """Report actual completed units; unknown totals stay indeterminate."""
    callback = getattr(_LOCAL, "reporter", None)
    if callback is None:
        return
    now = time.monotonic()
    new = current_stage() != stage or (done == 0 and getattr(_LOCAL, "done", None) != 0)
    if new:
        _LOCAL.stage = stage
        _LOCAL.started = now
        _LOCAL.started_ms = int(time.time() * 1000)
        _LOCAL.sent = -float("inf")
    _LOCAL.done = done
    # Frame-level CTC updates must not fill the bounded SSE history.
    if not new and done != total and now - _LOCAL.sent < .25:
        return
    _LOCAL.sent = now
    elapsed = now - _LOCAL.started
    measurable = total is not None and total > 0 and done is not None
    eta = elapsed * (total - done) / done if measurable and 0 < done < total and elapsed >= 2 else None
    callback({"stage": stage, "stage_done": done, "stage_total": total,
              "stage_started_at_ms": _LOCAL.started_ms,
              "stage_reported_at_ms": int(time.time() * 1000), "stage_eta_seconds": eta})


def advance(stage, done, total):
    """Only update the active stage; local rechecks cannot replace its total."""
    if current_stage() == stage:
        report(stage, done, total)
