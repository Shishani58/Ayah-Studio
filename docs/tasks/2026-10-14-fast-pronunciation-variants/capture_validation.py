"""Persist rejected proof evidence, then re-raise the original validation error."""
from dataclasses import asdict
import json
from pathlib import Path
import runpy

from surah_cutter.core import fast_executor

task = Path(__file__).parent.resolve()
original = fast_executor.validate_boundaries


def capture(boundaries, ayahs, duration, reads, **kwargs):
    try:
        return original(boundaries, ayahs, duration, reads, **kwargs)
    except ValueError as error:
        (task / 'raw-tempo-rejected-proof.json').write_text(json.dumps({
            'error': str(error), 'boundaries': [asdict(b) for b in boundaries],
            'reads': reads, 'ayahs': ayahs, 'duration': duration,
        }, ensure_ascii=False, indent=2, default=str) + '\n')
        raise


fast_executor.validate_boundaries = capture
runpy.run_path(str(task / 'collapse_raw_tempo_probe.py'), run_name='__main__')
