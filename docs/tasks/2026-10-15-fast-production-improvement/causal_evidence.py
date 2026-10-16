"""Classify direct extra-read evidence separately from later cursor recovery."""
MARKER = 'whole_spoken_prefix_incomplete_canonical_word'

def cue_evidence(case, prior, ayah):
    trace = case['trace']
    boundary = next(b for b in trace['boundaries'] if b['ayah'] == ayah)
    proof_read = trace['acoustic_reads'][boundary['read_id']]
    observations = []
    for i, event in enumerate(trace['transitions']):
        if not event.get('acoustic_anchor', '').endswith(MARKER):
            continue
        read_event = next(e for e in reversed(trace['transitions'][:i])
                          if e['state'] == 'READ_LOCAL' and e['ayah'] == event['ayah'])
        for read in trace['acoustic_reads']:
            if read.get('requested_window_ms', read['window_ms']) == read_event['window_ms']:
                observations.append({'id': read['id'], 'window_ms': read['window_ms'],
                    'requested_for_ayah': event['ayah'], 'anchor': event['acoustic_anchor'],
                    'observed': read['observed']['units'],
                    'unit_quality': read['observed']['unit_quality']})
    phrase = boundary['proof']['phrase']
    same_window = [r for r in observations if r['id'] < boundary['read_id']
        and r['window_ms'][0] == proof_read['window_ms'][0]
        and r['window_ms'][1] < proof_read['window_ms'][1]
        and r['observed'] and r['observed'][0] == phrase[0]]
    proof_hash = case['actual_dispatches'][boundary['read_id']]['sha256']
    new_input = proof_hash not in {r['sha256'] for r in prior['actual_dispatches']}
    return {'proof_read': proof_read, 'cue_reads': same_window,
            'direct': bool(same_window) and new_input, 'proof_input_new': new_input}
