"""Research-only exact CTC substring graph, with explicit unrestricted complement.

Unlike prefix scoring, noncanonical/earlier speech is allowed before a known
phrase. This may establish identity inside a crop, never that the crop begins
at the ayah boundary. No timestamp is produced by this experiment.
"""
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from surah_cutter.core import align, decode, fast_executor, model_store
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_ctc import prefix_scores
from surah_cutter.data import quran_text

P = Path(__file__).resolve().parent


def substring_probability(logp, target, blank):
    """Probability that collapsed CTC contains target anywhere (KMP DFA)."""
    length, vocab = len(target), logp.shape[1]
    table = np.zeros((length, vocab), dtype=np.int32)
    for k in range(length):
        for token in range(vocab):
            stream = tuple(target[:k]) + (token,)
            for n in range(min(length, len(stream)), -1, -1):
                if stream[len(stream)-n:] == tuple(target[:n]):
                    table[k, token] = n
                    break
    mass = np.zeros((length, vocab), dtype=np.float64)
    mass[0, blank] = 1.
    accepted = 0.
    for row in np.exp(logp.astype(np.float64)):
        total = mass.sum(axis=1)
        following = mass * row  # repeated nonblank emits no new label
        following[:, blank] = total * row[blank]
        changes = np.maximum(0., total[:, None] - mass) * row
        changes[:, blank] = 0
        matched = table == length
        accepted = accepted * row.sum() + changes[matched].sum()
        k, token = np.nonzero((~matched) & (changes > 0))
        np.add.at(following, (table[k, token], token), changes[k, token])
        mass = following
    return min(1., float(accepted))


def check_graph():
    rng = np.random.default_rng(7)
    probabilities = rng.dirichlet([1, 1, 1], size=4)
    for target in [(1,), (1, 1), (1, 2), (2, 1, 2)]:
        expected = 0.
        for path in itertools.product(range(3), repeat=4):
            decoded = tuple(t for i, t in enumerate(path) if t and (not i or t != path[i-1]))
            if any(decoded[i:i+len(target)] == target for i in range(len(decoded))):
                expected += math.prod(probabilities[i, t] for i, t in enumerate(path))
        assert abs(substring_probability(np.log(probabilities), target, 0) - expected) < 1e-12


def main():
    check_graph()
    model_store.ACTIVE_ID = 'quran-phonetics'
    adapter = CTCAdapter(align.default_bundle())
    inputs = []
    with (P / 'baseline.jsonl').open() as f:
        for line in f:
            item = json.loads(line)
            if item['model'] != 'whisper-tarteel-base' or item['surah'] not in (2, 78, 80):
                continue
            used = set()
            for read in item['trace']['acoustic_reads']:
                a = int(read['reason'].split(':')[0].split('=')[1])
                if a in used or len(used) >= 3:
                    continue
                used.add(a)
                inputs.append((item['surah'], a, read['window_ms'], item['audio_path']))
    # Positive boundary crops, not just the first (often basmala) hypothesis.
    with (P / 'baseline.jsonl').open() as f:
        for line in f:
            item = json.loads(line)
            if item['model'] != 'whisper-tarteel-base' or item['surah'] not in (2, 80):
                continue
            for boundary in [b for b in item['trace']['boundaries'] if b['confidence'] > 0][:3]:
                read = item['trace']['acoustic_reads'][boundary['read_id']]
                left = read['window_ms'][0]
                right = min(item['trace']['total_ms'], left + 4200)
                inputs.append((item['surah'], boundary['ayah'], [left, right], item['audio_path']))
    results = []
    for index, (surah, ayah, (left, right), path) in enumerate(inputs):
        pcm = decode.decode_range(path, start_ms=left, end_ms=right)
        assert 0 < len(pcm) <= 96000
        cases = [('audio', pcm), ('silence', np.zeros_like(pcm))] if index in (0, 3) else [('audio', pcm)]
        words = quran_text.surah_words(surah)
        rendered = adapter.represent(words)
        groups, ends = {}, {}
        for w, units in zip(words, rendered):
            groups.setdefault(w.ayah, []).extend(units)
            ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
        groups = {a: tuple(units) for a, units in groups.items()}
        plan = fast_executor.plan_phrase(groups, ayah, ends)
        phrase = plan['units']
        target = [adapter.vocab[u] for u in phrase]
        for kind, local in cases:
            started = time.monotonic()
            observation = adapter.observe(local, 16000, total_samples=len(local)+1)
            np.savez_compressed(P / f'graph-v2-{index}-{kind}.npz', posterior=observation.posterior)
            inference = time.monotonic() - started
            started = time.monotonic()
            prefix = prefix_scores(observation.posterior, [target], adapter.blank_id)[0]
            probability = substring_probability(observation.posterior, target, adapter.blank_id)
            # Run globally competing canonical word-start phrases only if the
            # proposed phrase could possibly meet even the expansion gate.
            rivals = []
            if probability >= .1:
                for owner, units in groups.items():
                    stream = units + groups.get(owner+1, ())
                    for offset in [0] + ends[owner][:-1]:
                        if (owner, offset) == (ayah, 0):
                            continue
                        other = stream[offset:offset+len(phrase)]
                        if len(other) != len(phrase):
                            continue
                        score = substring_probability(observation.posterior,
                                  [adapter.vocab[u] for u in other], adapter.blank_id)
                        rivals.append((score, owner, offset))
            rival = max(rivals, default=(0., None, None))
            unknown = 1 - probability
            accepted = probability >= .9 and probability >= 9*max(unknown, rival[0]) and not plan['ambiguous']
            result = dict(surah=surah, ayah=ayah, window_ms=[left,right], kind=kind,
                heard=''.join(observation.units), target=''.join(phrase), ambiguous=plan['ambiguous'],
                prefix_probability=math.exp(prefix), substring_probability=probability,
                unknown_probability=unknown, largest_rival=rival, identity_candidate=accepted,
                inference_seconds=inference, graph_seconds=time.monotonic()-started,
                published_timestamp=None, posterior_sha256=observation.posterior_sha256)
            results.append(result)
            (P / 'graph-results-v2.json').write_text(json.dumps(results, indent=2) + '\n')
            print(result, flush=True)


if __name__ == '__main__':
    main()
