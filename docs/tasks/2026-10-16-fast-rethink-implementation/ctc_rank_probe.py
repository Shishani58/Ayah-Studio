"""External research: forced local CTC likelihood versus length-aware rivals.

No inference, references, timestamp or production threshold. Uses saved local
posteriors. A rank among Quran phrases is not a calibrated probability of owner.
"""
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_executor import plan_phrase
from surah_cutter.data import quran_text

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
vocab = json.loads((ROOT/'tests/fixtures/quran_phonetics_vocab.json').read_text())
blank = vocab['[PAD]']
adapter = CTCAdapter(SimpleNamespace(emitter=SimpleNamespace(runtime_identity={'weights_sha256':'research'}),
                     vocab=vocab, blank_id=blank, lag_ms=0))


def ctc_likelihood(logp, targets, blank_id=blank):
    """Sum all full-window CTC paths; batch targets of equal token length."""
    length = len(targets[0])
    labels = np.full((len(targets), 2*length+1), blank_id, dtype=int)
    labels[:, 1::2] = targets
    allow = np.zeros_like(labels, dtype=bool)
    allow[:, 2:] = (labels[:, 2:] != blank_id) & (labels[:, 2:] != labels[:, :-2])
    state = np.full(labels.shape, -np.inf)
    state[:, 0] = 0
    for row in logp:
        one = np.full_like(state, -np.inf)
        two = np.full_like(state, -np.inf)
        one[:, 1:] = state[:, :-1]
        two[:, 2:] = state[:, :-2]
        state = np.logaddexp(np.logaddexp(state, one), np.where(allow, two, -np.inf)) + row[labels]
    return np.logaddexp(state[:, -1], state[:, -2])


# Exhaustively check repeated symbols and blank transitions on a tiny model.
rng = np.random.default_rng(42)
prob = rng.dirichlet([1,1,1], size=4)
for target in [(1,), (1,1), (1,2)]:
    mass = sum(math.prod(prob[i,t] for i,t in enumerate(path))
        for path in itertools.product(range(3), repeat=4)
        if tuple(t for i,t in enumerate(path) if t and (not i or t!=path[i-1])) == target)
    assert abs(float(np.exp(ctc_likelihood(np.log(prob), [target], blank_id=0))[0]) - mass) < 1e-12

posteriors = {}
for path in P.glob('graph-v2-*.npz'):
    array = np.load(path)['posterior']
    digest = __import__('hashlib').sha256(memoryview(np.ascontiguousarray(array))).hexdigest()
    posteriors[digest] = array
results = []
cases = json.loads((P/'graph-results-v2.json').read_text())
for index, case in enumerate(cases):
    logp = posteriors[case['posterior_sha256']]
    words = quran_text.surah_words(case['surah'])
    groups, ends = {}, {}
    for w, units in zip(words, adapter.represent(words)):
        groups.setdefault(w.ayah, []).extend(units)
        ends.setdefault(w.ayah, []).append(len(groups[w.ayah]))
    groups = {a:tuple(u) for a,u in groups.items()}
    phrase = plan_phrase(groups, case['ayah'], ends)['units']
    length = len(phrase)
    # All canonical word starts, extended to the nearest complete word around
    # target's length. Raw likelihood and per-token / per-frame are retained.
    alternatives = [(case['ayah'], 0, phrase, 'expected')]
    for a, text in groups.items():
        for offset in [0]+ends[a][:-1]:
            if (a,offset)==(case['ayah'],0):
                continue
            sizes = [end-offset for end in ends[a] if end>offset]
            size = min(sizes, key=lambda n:abs(n-length))
            if .5*length <= size <= 2*length:
                alternatives.append((a, offset, text[offset:offset+size], 'canonical_rival'))
    phone_ids = [t for c,t in vocab.items() if len(c)==1 and c not in (' ','-','|')]
    for _ in range(8):
        random = tuple(next(c for c,t in vocab.items() if t==token) for token in rng.choice(phone_ids,length))
        alternatives.append((None,None,random,'same_length_garbage'))
    scores = [None]*len(alternatives)
    batches = defaultdict(list)
    for i,(_,_,units,_) in enumerate(alternatives):
        batches[len(units)].append(i)
    started = monotonic()
    for _, indices in batches.items():
        values = ctc_likelihood(logp, [tuple(vocab[u] for u in alternatives[i][2]) for i in indices])
        for i,value in zip(indices,values):
            scores[i]=float(value)
    # Only the exact same token length supports a directly comparable rank.
    equal = [i for i,a in enumerate(alternatives) if len(a[2])==length and i]
    rival = max(equal,key=lambda i:scores[i], default=None)
    normalized = max(range(1,len(alternatives)),key=lambda i:scores[i]/len(alternatives[i][2]))
    null = float(logp[:,blank].sum())
    unrestricted_viterbi = float(logp.max(axis=1).sum())
    result = dict(surah=case['surah'],ayah=case['ayah'],kind=case['kind'],window_ms=case['window_ms'],
        target=case['target'],heard=case['heard'],frames=len(logp),target_tokens=length,
        expected_log_likelihood=scores[0],expected_per_token=scores[0]/length,
        expected_per_frame=scores[0]/len(logp),same_length_candidates=len(equal),
        same_length_rank=1+sum(scores[i]>scores[0] for i in equal),
        best_same_length=dict(owner=alternatives[rival][:2], text=''.join(alternatives[rival][2]),
                              kind=alternatives[rival][3],log_likelihood=scores[rival]) if rival is not None else None,
        best_per_token=dict(owner=alternatives[normalized][:2],text=''.join(alternatives[normalized][2]),
                            per_token=scores[normalized]/len(alternatives[normalized][2])),
        blank_log_likelihood=null,unrestricted_best_path=unrestricted_viterbi,
        expected_minus_blank=scores[0]-null,expected_minus_unrestricted=scores[0]-unrestricted_viterbi,
        ranking_seconds=monotonic()-started,published_owner=None,published_timestamp=None)
    results.append(result)
(P/'rank-results.json').write_text(json.dumps(results,indent=2)+'\n')
for r in results:
 print(r['surah'],r['ayah'],r['kind'],'rank',r['same_length_rank'], '/',r['same_length_candidates']+1,
    'exp/null',round(r['expected_minus_blank'],2),'exp/unrestricted',round(r['expected_minus_unrestricted'],2),flush=True)
