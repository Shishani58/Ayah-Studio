"""Relate Base n-best to actual FAST phrase/quality gates, not text alone."""
import copy
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
import sys
import time

import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sys.path.insert(0, str(ROOT))


def dump(name, data):
    (P / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def representation(surah, ayah):
    from surah_cutter.core.fast_adapters import arabic_units
    from surah_cutter.core.fast_executor import plan_phrase
    from surah_cutter.core.model_runtime import inference_words
    from surah_cutter.data import quran_text

    words = quran_text.surah_words(surah)
    groups, ends = {}, {}
    for word, text in zip(words, inference_words(words, 'whisper-tarteel-base')):
        groups.setdefault(word.ayah, []).extend(arabic_units(text))
        ends.setdefault(word.ayah, []).append(len(groups[word.ayah]))
    groups = {a: tuple(u) for a, u in groups.items()}
    plan = plan_phrase(groups, ayah, ends)
    return dict(phrase=plan['units'], ambiguous=plan['ambiguous'], expected=groups[ayah])


def prepare():
    from surah_cutter.core import decode, model_store

    assert not (P / 'deep-inputs.json').exists()
    configs, arrays, pcms, skipped = [], {}, {}, []
    prior = json.loads((P / 'inputs.json').read_text())
    for case in prior['cases']:
        if case['end_ms'] - case['start_ms'] == 1500:
            configs.append(dict(key=f"original-{case['key']}", reciter='shatri',
                                surah=case['surah'], ayah=case['ayah'],
                                window_ms=[case['start_ms'], case['end_ms']], role='prior-example'))
    baseline_path = ROOT / 'docs/tasks/2026-10-15-fast-systematic-execution/baseline-a2.json'
    baseline = json.loads(baseline_path.read_text())
    for surah, ayah in [(32, 1), (65, 1), (80, 12), (98, 8), (2, 4), (79, 27)]:
        row = next(r for r in baseline if r['model'] == 'whisper-tarteel-base' and r['surah'] == surah)
        reads = [r for r in row['trace']['acoustic_reads']
                 if r.get('reason', '').startswith(f'ayah={ayah}:pause:')]
        phrase = representation(surah, ayah)['phrase']
        # Choose suspected owner using historical text only, never reference times.
        read = max(reads, key=lambda r: SequenceMatcher(None, ' '.join(phrase),
                    ' '.join(r.get('observed', {}).get('units', [])[:len(phrase)])).ratio())
        for extended in [False, True]:
            start, end = read['window_ms']
            if extended:
                end = max(end, start + 4000)
            configs.append(dict(key=f'blocked-{surah}-{ayah}-{int(extended)}', reciter='shatri',
                                surah=surah, ayah=ayah, window_ms=[start, end],
                                role='blocked-expanded' if extended else 'blocked-original',
                                historical_text=read['observed']['text'],
                                selection_read_id=read['id']))
    for reciter in ['saad_al_ghamidi', 'mahmoud_khalil_al_hussary', 'mohammed_siddiq_al_minshawi']:
        path = ROOT / f'docs/tasks/2026-10-15-fast-production-improvement/corpus-{reciter}-baseline.json'
        rows = json.loads(path.read_text())
        for surah, ayah in [(80, 1), (98, 2), (112, 1)]:
            row = next(r for r in rows if r['model'] == 'whisper-tarteel-base' and r['surah'] == surah)
            phrase = representation(surah, ayah)['phrase']
            reads = [r for r in row['trace']['acoustic_reads']
                     if r.get('reason', '').startswith(f'ayah={ayah}:pause:')]
            if not reads:
                skipped.append(dict(reciter=reciter, surah=surah, ayah=ayah,
                                    reason='no_historical_target_pause_read'))
                continue
            read = max(reads, key=lambda r: SequenceMatcher(None, ' '.join(phrase),
                        ' '.join(r.get('observed', {}).get('units', [])[:len(phrase)])).ratio())
            configs.append(dict(key=f'transfer-{reciter}-{surah}', reciter=reciter,
                                surah=surah, ayah=ayah, window_ms=read['window_ms'],
                                role='transfer-selected', historical_text=read['observed']['text']))
    for cfg in configs:
        reciter, surah = cfg['reciter'], cfg['surah']
        path = ((Path('/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri') if reciter == 'shatri'
                 else Path('/Users/ayub/.cache/surah-cutter/full-reciter-audit') / reciter)
                / f'{surah:03}.mp3')
        if str(path) not in pcms:
            pcms[str(path)] = decode.decode(str(path))
        pcm = pcms[str(path)]
        lo, hi = cfg['window_ms']
        hi = min(hi, len(pcm) / 16)
        crop = pcm[round(lo * 16):round(hi * 16)].copy()
        assert 4800 <= len(crop) <= 96000 and len(crop) < len(pcm)
        arrays[cfg['key']] = crop
        cfg.update(representation(surah, cfg['ayah']))
        cfg.update(audio_path=str(path), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   pcm_sha256=hashlib.sha256(crop.tobytes()).hexdigest(), samples=len(crop))
    for name, source, surah, ayah in [
        ('zero80', 'original-s80-a1-1500', 80, 1),
        ('wrong80', 'original-s98-a2-1500', 80, 1),
        ('wrong98', 'original-s80-a1-1500', 98, 2),
    ]:
        arrays[name] = np.zeros(24000, np.float32) if name.startswith('zero') else arrays[source]
        configs.append(dict(key=name, role='negative', surah=surah, ayah=ayah,
                            **representation(surah, ayah), source=source,
                            pcm_sha256=hashlib.sha256(arrays[name].tobytes()).hexdigest(),
                            samples=len(arrays[name])))
    dump('deep-inputs.json', dict(cases=configs, skipped=skipped,
         weights=str(model_store.installed_path(model_store.spec('whisper-tarteel-base'))),
         source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ROOT / 'surah_cutter/core/fast_executor.py',
                                  ROOT / 'surah_cutter/core/fast_adapters.py',
                                  ROOT / 'surah_cutter/core/model_worker.py']}))
    np.savez(P / 'deep-inputs.npz', **arrays)
    print('prepared', len(configs))


def run(prefix='deep'):
    import torch
    from transformers.generation.utils import GenerationMixin
    from surah_cutter.core.fast_adapters import attributed_asr_units
    from surah_cutter.core.fast_executor import _strong_continuation_conflict
    from surah_cutter.core.model_worker import Acoustic
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode

    assert not (P / f'{prefix}-results.json').exists()
    config = json.loads((P / f'{prefix}-inputs.json').read_text())
    arrays = np.load(P / f'{prefix}-inputs.npz')
    acoustic = Acoustic('whisper-tarteel-base', Path(config['weights']))
    model, processor = acoustic.model, acoustic.processor
    tokenizer = processor.tokenizer
    byte_decoder = {c: v for v, c in bytes_to_unicode().items()}
    results = []
    for case in config['cases']:
        pcm = arrays[case['key']]
        features = processor(pcm, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        started = time.perf_counter()
        outputs, elapsed = {}, {}
        with torch.inference_mode():
            encoded = model.model.encoder(features.input_features, return_dict=True)
            encoder_seconds = time.perf_counter() - started
            for beams in [1, 5]:
                started = time.perf_counter()
                generation = copy.deepcopy(model.generation_config)
                generation.forced_decoder_ids = None
                generated = GenerationMixin.generate(
                    model, encoder_outputs=encoded, attention_mask=features.attention_mask,
                    decoder_input_ids=torch.tensor([tokenizer.prefix_tokens]),
                    generation_config=generation, max_new_tokens=64, do_sample=False,
                    num_beams=beams, num_return_sequences=beams,
                    return_dict_in_generate=True, output_scores=True)
                transition = model.compute_transition_scores(
                    generated.sequences, generated.scores,
                    getattr(generated, 'beam_indices', None), normalize_logits=True)
                hypotheses = []
                for i, sequence in enumerate(generated.sequences):
                    selected = sequence[-transition.shape[1]:]
                    tokens = []
                    for token, logp in zip(selected, transition[i]):
                        token_id = int(token)
                        if token_id in tokenizer.all_special_ids:
                            continue
                        encoded_token = tokenizer.convert_ids_to_tokens(token_id)
                        tokens.append(dict(id=token_id, log_probability=float(logp),
                            bytes=list(bytes(byte_decoder[c] for c in encoded_token))))
                    text = tokenizer.decode(sequence, skip_special_tokens=True)
                    units, quality, attribution = attributed_asr_units(dict(text=text, selected_tokens=tokens))
                    phrase = tuple(case['phrase'])
                    complete = tuple(units[:len(phrase)]) == phrase
                    q = min(quality[:len(phrase)], default=0.)
                    truncated = tokenizer.eos_token_id not in sequence[len(tokenizer.prefix_tokens):].tolist()
                    conflict = _strong_continuation_conflict(units, quality, case['expected'], len(phrase), .6)
                    hypotheses.append(dict(text=text, units=units, unit_quality=quality,
                        attribution=attribution, complete=complete, phrase_quality=q,
                        strong_continuation_conflict=conflict, truncated=truncated,
                        necessary_opening_checks=complete and q >= .6 and not truncated and not conflict,
                        selected_tokens=tokens))
                outputs[str(beams)] = hypotheses
                elapsed[str(beams)] = time.perf_counter() - started
        result = dict(key=case['key'], encoder_seconds=encoder_seconds,
                      decoder_seconds=elapsed, readings=outputs)
        results.append(result)
        dump(f'{prefix}-results.json', results)
        g, bs = outputs['1'][0], outputs['5']
        print(case['key'], 'phrase', case['phrase'], 'greedy', g['text'], g['necessary_opening_checks'],
              'beam', [(b['text'], round(b['phrase_quality'], 4), b['necessary_opening_checks']) for b in bs], flush=True)


if __name__ == '__main__':
    prepare() if sys.argv[1] == 'prepare' else run(sys.argv[2] if len(sys.argv) > 2 else 'deep')
