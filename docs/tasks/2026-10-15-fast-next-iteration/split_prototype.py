"""Research only: exact characters, symmetric ownership, original token evidence."""
import math


def split_opening(obs, groups, ends, ayah, threshold):
    target = groups[ayah]
    # Only plain one-unit canonical words. Named letters need their own proof.
    if ends[ayah] != list(range(1, len(target) + 1)) or len(target) < 2 or obs.truncated:
        return None, False
    units, qualities, attrs = obs.units, obs.unit_quality, obs.attribution
    if len(units) != len(qualities) or len(attrs) != len(units):
        return None, False
    joined, first_count = '', None
    for i, unit in enumerate(units):
        joined += unit
        if joined == target[0]:
            first_count = i + 1
            break
        if not target[0].startswith(joined):
            return None, False
    if first_count is None or first_count < 2 or len(target[0]) < 3:
        return None, False
    def strong(count):
        for u, q, a in zip(units[:count], qualities[:count], attrs[:count]):
            ps = a.get('probabilities', ())
            if (not math.isfinite(q) or q < threshold or a.get('word') != u
                    or 'spoken_alias' in a or 'unsupported' in a or not ps
                    or len(a.get('selected_indices', ())) != len(ps)
                    or any(not math.isfinite(p) or not threshold <= p <= 1 for p in ps)
                    or q != min(ps)):
                return False
        return True
    if not strong(first_count):
        return None, False
    starts = [(a, i, ''.join(text[i:])) for a, text in groups.items()
              for i in [0] + ends[a][:-1] if (a, i) != (ayah, 0)]
    # No partial first-word identity can authorize scanning repeated owners.
    if any(text.startswith(target[0]) for _, _, text in starts):
        return None, False
    # At least TWO whole canonical words; exact source unit endpoints only.
    for count in range(2, len(target) + 1):
        text = ''.join(target[:count])
        if any(other.startswith(text) for _, _, other in starts):
            continue
        observed = ''
        for n, unit in enumerate(units, 1):
            observed += unit
            if observed == text and strong(n):
                return {'phrase': target[:count], 'segmented_opening': {
                    'source_count': n, 'first_source_count': first_count,
                    'source_units': units[:n], 'selected_indices': sorted({i for a in attrs[:n] for i in a['selected_indices']}),
                    'competitor_count': len(starts), 'onset_supported': False}}, True
            if not text.startswith(observed):
                break
    # A strong contradictory next unit is not excused as word segmentation.
    rest = ''.join(units[first_count:])
    expected = ''.join(target[1:])
    if rest and not expected.startswith(rest):
        if any(q >= threshold for q in qualities[first_count:]):
            return None, False
    return None, True
