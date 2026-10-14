"""Target-independent bounded CTC prefix beam over already charged local frames."""
from __future__ import annotations

import math

import numpy as np


def independent_beam(emissions, blank, inverse, stride_ms, lag_ms, *, width=8):
    # Each state is (blank/nonblank summed log masses, best blank/nonblank
    # paths). Paths contain only first token frames; no known text is supplied.
    empty = (0., ())
    beams = {(): [0., -math.inf, empty, (-math.inf, ())]}
    for frame, scores in enumerate(emissions):
        choices = set(np.argsort(scores)[-width:].tolist()) | {blank}
        next_beams = {}

        def add(prefix, kind, mass, path):
            if not math.isfinite(mass):
                return
            state = next_beams.setdefault(prefix, [-math.inf, -math.inf,
                                                   (-math.inf, ()), (-math.inf, ())])
            state[kind] = float(np.logaddexp(state[kind], mass))
            if path[0] > state[kind + 2][0]:
                state[kind + 2] = path

        for prefix, (pb, pn, vb, vn) in beams.items():
            total = float(np.logaddexp(pb, pn))
            best = max(vb, vn, key=lambda p: p[0])
            for token in choices:
                score = float(scores[token])
                if token == blank:
                    add(prefix, 0, total + score, (best[0] + score, best[1]))
                elif prefix and token == prefix[-1]:
                    add(prefix, 1, pn + score, (vn[0] + score, vn[1]))
                    add(prefix + (token,), 1, pb + score,
                        (vb[0] + score, vb[1] + ((frame, math.exp(score)),)))
                else:
                    add(prefix + (token,), 1, total + score,
                        (best[0] + score, best[1] + ((frame, math.exp(score)),)))
        beams = dict(sorted(next_beams.items(),
                            key=lambda item: np.logaddexp(item[1][0], item[1][1]),
                            reverse=True)[:width])
    result = []
    for prefix, (pb, pn, vb, vn) in beams.items():
        path = max(vb, vn, key=lambda p: p[0])[1]
        if len(path) != len(prefix):
            raise ValueError("incomplete independent CTC path")
        units, starts, quality = [], [], []
        for token, (frame, probability) in zip(prefix, path):
            for char in inverse.get(token, ""):
                if char != "|":
                    units.append(char)
                    starts.append(max(0., frame * stride_ms - lag_ms))
                    quality.append(probability)
        result.append({"units": units, "positions_ms": starts, "unit_quality": quality,
                       "log_probability": float(np.logaddexp(pb, pn))})
    result.sort(key=lambda h: h["log_probability"], reverse=True)
    # Null is an independently scored all-blank competitor, even after pruning.
    return tuple(result), float(emissions[:, blank].sum())


def prefix_scores(emissions, prefixes, blank):
    """Exact CTC first-entry probabilities for independent text alternatives.

    A shared prefix trie avoids one Python DP per canonical competitor. Each
    accepting event is the FIRST completed prefix, then future frames are free.
    The missing/noncanonical probability mass is an explicit negative too.
    """
    parents, tokens, nodes, leaves = [0], [blank], {}, []
    for prefix in prefixes:
        node = 0
        for token in prefix:
            key = (node, token)
            if key not in nodes:
                nodes[key] = len(tokens)
                parents.append(node)
                tokens.append(token)
            node = nodes[key]
        leaves.append(node)
    parents, tokens = np.asarray(parents), np.asarray(tokens)
    different = tokens != tokens[parents]
    pb = np.full(len(tokens), -math.inf)
    pn = np.full(len(tokens), -math.inf)
    done = np.full(len(tokens), -math.inf)
    pb[0] = 0.
    for row in emissions:
        entry = np.logaddexp(pb[parents], np.where(different, pn[parents], -math.inf))
        entry[0] = -math.inf
        score = row[tokens]
        done = np.logaddexp(done, entry + score)
        next_pb = np.logaddexp(pb, pn) + row[blank]
        pn = np.logaddexp(pn, entry) + score
        pb = next_pb
    return tuple(min(0., float(done[node])) for node in leaves)


def prefix_path(emissions, targets, blank, stride_ms, lag_ms):
    """Best supporting prefix path; never used alone as identity evidence."""
    states = [(-math.inf, ()) for _ in range(len(targets) * 2 - 1)]
    states[0] = (0., ())
    symbols = [blank]
    for token in targets[:-1]:
        symbols.extend((token, blank))
    complete = (-math.inf, ())
    for frame, row in enumerate(emissions):
        result = []
        for i, token in enumerate(symbols):
            sources = [(states[i][0], states[i][1], False)]
            if i:
                sources.append((*states[i-1], token != blank))
            if i > 1 and token != blank and token != symbols[i-2]:
                sources.append((*states[i-2], True))
            score, path, append = max(sources, key=lambda s: s[0])
            if append:
                path += ((max(0., frame * stride_ms - lag_ms), math.exp(float(row[token]))),)
            result.append((score + float(row[token]), path))
        sources = [states[-1]]
        if len(targets) > 1 and targets[-1] != targets[-2]:
            sources.append(states[-2])
        score, path = max(sources, key=lambda s: s[0])
        candidate = (score + float(row[targets[-1]]), path +
                     ((max(0., frame * stride_ms - lag_ms), math.exp(float(row[targets[-1]]))),))
        complete = max(complete, candidate, key=lambda s: s[0])
        states = result
    if len(complete[1]) != len(targets) or not math.isfinite(complete[0]):
        return None
    return {"positions_ms": [p[0] for p in complete[1]], "unit_quality": [p[1] for p in complete[1]]}
