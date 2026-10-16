"""Measurement-only worker decorator; greedy inputs/outputs stay untouched."""
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
decorator = '''
def _profile_local_read(method):
    def measured(*args, **kwargs):
        import cProfile
        from pathlib import Path
        import os
        directory = Path(os.environ["FAST_PROFILE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        index = len(list(directory.glob("*.prof")))
        profile = cProfile.Profile()
        try:
            return profile.runcall(method, *args, **kwargs)
        finally:
            profile.dump_stats(str(directory / f"{index:05}.prof"))
    return measured


'''
for source in ('A', 'B3'):
    root = Path(roots[source]).with_name(source + 'P')
    assert not root.exists()
    shutil.copytree(roots[source], root, ignore=shutil.ignore_patterns('__pycache__'))
    file = root / 'surah_cutter/core/model_worker.py'
    code = file.read_text()
    # Defining the decorator before any class requires no extra runtime import.
    at = code.index('\nclass ')
    code = code[:at] + '\n' + decorator + code[at:]
    code = code.replace('    def whisper_local(self, pcm, total_samples):',
                        '    @_profile_local_read\n    def whisper_local(self, pcm, total_samples):', 1)
    compile(code, str(file), 'exec')
    file.write_text(code)
    roots[source + 'P'] = str(root)
(P / 'snapshot-paths.json').write_text(json.dumps(roots, indent=2) + '\n')
