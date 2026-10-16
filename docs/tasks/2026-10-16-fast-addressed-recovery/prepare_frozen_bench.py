"""Reuse hashed baseline PCM/DSP inputs while obtaining genuinely fresh model reads."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
roots = json.loads((P / 'snapshot-paths.json').read_text())
root = Path(roots['PF'])
code = (root / 'tools/bench_fast_models.py').read_text()
start = code.index('        for surah in args.surahs:')
end = code.index('        for model_id in args.models:', start)
inputs = '''        manifest = json.loads(Path(INPUT_MANIFEST).read_text())['recordings']
        pcm_paths = {}
        for surah in args.surahs:
            info = manifest[str(surah)]
            audio = args.audio_dir / f"{surah:03d}.mp3"
            assert _sha(audio) == info['audio_sha256']
            cached_path = None
            for cache in Path(tempfile.gettempdir()).glob('local-fast-bench-*'):
                path = cache / f"{surah}.npy"
                if path.exists():
                    pcm = np.load(path, mmap_mode='r', allow_pickle=False)
                    if hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest() == info['pcm_sha256']:
                        cached_path = path
                        break
            if cached_path is None:
                pcm = decode.decode(str(audio))
                assert hashlib.sha256(memoryview(np.ascontiguousarray(pcm))).hexdigest() == info['pcm_sha256']
                cached_path = folder / f"{surah}.npy"
                np.save(cached_path, pcm)
            del pcm
            pcm_paths[surah] = cached_path
            decisions = [candidates.PauseDecision(**{**d, 'pause': candidates.Pause(**d['pause'])})
                         for d in info['decisions']]
            recordings[surah] = {**info, 'decisions': decisions,
                'decode_seconds': 0., 'dsp_seconds': 0., 'frozen_pcm_dsp_inputs': True,
                'input_manifest': INPUT_MANIFEST}
'''
code = code[:start] + inputs + code[end:]
code = code.replace('    parser = argparse.ArgumentParser(',
                    '    INPUT_MANIFEST = ' + repr(str(P / 'frozen-input-manifest.json')) + '\n    parser = argparse.ArgumentParser(', 1)
assert 'INPUT_MANIFEST = ' in code
code = code.replace('np.load(folder / f"{surah}.npy", allow_pickle=False)',
                    'np.load(pcm_paths[surah], mmap_mode="r", allow_pickle=False)')
code = code.replace('paths.extend([root / "tools/bench_fast_models.py",',
                    'paths.extend([Path(__file__).resolve(), root / "tools/bench_fast_models.py",', 1)
compile(code, str(root / 'tools/frozen_bench_fast.py'), 'exec')
(root / 'tools/frozen_bench_fast.py').write_text(code)
(P / 'frozen_bench_fast.py').write_text(code)
