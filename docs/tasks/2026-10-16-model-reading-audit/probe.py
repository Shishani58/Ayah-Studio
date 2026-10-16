"""Local Base decoding alternatives; diagnostics only, no production writes."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sys.path.insert(0, str(ROOT))


def dump(name, data):
    (P / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare():
    from surah_cutter.core import decode, model_runtime, model_store
    from surah_cutter.data import quran_text

    assert not (P / 'inputs.json').exists()
    arrays, cases = {}, []
    # Intervals fixed from earlier acoustic diagnostics, never from reference DB.
    for surah, ayah, start in [(1, 1, 200), (80, 1, 6180),
                                (98, 2, 16940), (112, 1, 6120)]:
        path = Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri') / f'{surah:03}.mp3'
        pcm = decode.decode(str(path))
        words = quran_text.surah_words(surah)
        rendered = model_runtime.inference_words(words, 'whisper-tarteel-base')
        groups = {}
        for word, text in zip(words, rendered):
            groups.setdefault(word.ayah, []).append(text)
        candidates = [{'role': 'target', 'text': ' '.join(groups[ayah][:2])}]
        for a in [ayah - 1, ayah + 1]:
            if a in groups:
                candidates.append({'role': f'other-ayah-{a}', 'text': ' '.join(groups[a][:2])})
        candidates.append({'role': 'internal', 'text': ' '.join(groups[ayah][1:3])})
        for duration in [1500, 4000]:
            key = f's{surah}-a{ayah}-{duration}'
            crop = pcm[start * 16:(start + duration) * 16].copy()
            assert len(crop) == duration * 16 and len(crop) < len(pcm)
            arrays[key] = crop
            cases.append(dict(key=key, surah=surah, ayah=ayah,
                              start_ms=start, end_ms=start + duration,
                              duration_ms=len(pcm) / 16, candidates=candidates,
                              pcm_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                              audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    np.savez(P / 'inputs.npz', **arrays)
    spec = model_store.spec('whisper-tarteel-base')
    dump('inputs.json', dict(cases=cases, weights=str(model_store.installed_path(spec)),
                            model=spec, worker_sha256=hashlib.sha256(
                                (ROOT / 'surah_cutter/core/model_worker.py').read_bytes()).hexdigest()))


def run():
    import torch
    import transformers
    from surah_cutter.core.model_worker import Acoustic

    assert not (P / 'results.json').exists()
    config = json.loads((P / 'inputs.json').read_text())
    arrays = np.load(P / 'inputs.npz')
    acoustic = Acoustic('whisper-tarteel-base', Path(config['weights']))
    model, processor = acoustic.model, acoustic.processor
    results = []
    for case in config['cases']:
        for control in ['audio', 'zero']:
            pcm = arrays[case['key']]
            if control == 'zero':
                pcm = np.zeros_like(pcm)
            started = time.perf_counter()
            features = processor(pcm, sampling_rate=16000, return_tensors='pt',
                                 return_attention_mask=True)
            with torch.inference_mode():
                encoded = model.model.encoder(features.input_features, return_dict=True)
                readings = {}
                for beams in [1, 5]:
                    generated = model.generate(
                        features.input_features, encoder_outputs=encoded,
                        attention_mask=features.attention_mask,
                        language='ar', task='transcribe', return_timestamps=False,
                        max_new_tokens=64, do_sample=False, num_beams=beams,
                        num_return_sequences=beams, use_cache=True)
                    readings[str(beams)] = processor.batch_decode(generated, skip_special_tokens=True)
                scores = []
                prefix = processor.tokenizer.prefix_tokens
                for candidate in case['candidates']:
                    tokens = processor.tokenizer.encode(' ' + candidate['text'], add_special_tokens=False)
                    ids = torch.tensor([[*prefix, *tokens]])
                    out = model(encoder_outputs=encoded, decoder_input_ids=ids, use_cache=False)
                    logits = out.logits[0, len(prefix) - 1:len(prefix) - 1 + len(tokens)]
                    logp = logits.log_softmax(-1)[torch.arange(len(tokens)), torch.tensor(tokens)]
                    scores.append({**candidate, 'mean_logp': float(logp.mean()),
                                   'min_logp': float(logp.min()), 'token_count': len(tokens)})
            result = dict(key=case['key'], control=control,
                          encoder_input_samples=len(pcm),
                          encoder_input_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                          readings=readings, scores=scores,
                          seconds=time.perf_counter() - started)
            results.append(result)
            dump('results.json', dict(torch=torch.__version__, transformers=transformers.__version__,
                                     model='whisper-tarteel-base', results=results))
            print(case['key'], control, readings, scores, flush=True)


if __name__ == '__main__':
    prepare() if sys.argv[1] == 'prepare' else run()
