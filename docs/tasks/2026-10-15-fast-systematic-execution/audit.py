"""Static text/source analysis; never calls an acoustic model or test suite."""
import collections
import hashlib
import json
from pathlib import Path
import unicodedata

from surah_cutter.core import fast_executor as ex
from surah_cutter.core.fast_adapters import arabic_units
from surah_cutter.data import quran_text, transliteration

P = Path(__file__).resolve().parent
REPO = P.parents[2]


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def representations():
    vocab = json.loads((REPO / 'models/wav2vec2-quran-phonetics.onnx.vocab.json').read_text())
    rows = []
    counts = collections.Counter()
    for surah in range(1, 115):
        words = quran_text.surah_words(surah)
        targets = transliteration.surah_word_tr(surah)
        assert len(words) == len(targets)
        for index, (word, raw) in enumerate(zip(words, targets)):
            ctc = ''.join(c for c in raw if c not in ('-', '|', ' '))
            unknown = sorted(set(ctc) - vocab.keys())
            non_nfc = raw != unicodedata.normalize('NFC', raw)
            counts.update(unknown)
            rows.append({'surah': surah, 'ayah': word.ayah, 'word_index': index,
                'canonical': word.text, 'source_target': raw, 'ctc_target': ctc,
                'asr_target': arabic_units(word.text), 'unknown': unknown, 'non_nfc': non_nfc,
                'repeated_adjacent': any(a == b for a, b in zip(ctc, ctc[1:]))})
    dump('representation-audit.json', {'kind': 'static-text-no-listening', 'vocab': vocab,
         'blank': 37, 'policy': vars(ex.Policy()), 'words': rows,
         'unknown': dict(counts), 'non_nfc_count': sum(r['non_nfc'] for r in rows),
         'counts': {'words': len(rows), 'ayahs': 6236, 'surahs': 114},
         'helper_sha256': hashlib.sha256(transliteration.CACHE_PATH.read_bytes()).hexdigest()})
    print('representation', len(rows), 'unknown', dict(counts), 'NFC', sum(r['non_nfc'] for r in rows))


def causes():
    source = P / 'baseline-a2.json'
    if not (P / 'baseline-a2.summary.json').exists():
        source = P.parent / '2026-10-15-fast-next-iteration/after-raw.json'
    cases = json.loads(source.read_text())
    output = []
    for case in cases:
        trace = case['trace']
        stop = next((e for e in trace['transitions'] if e['state'] in ('RESOLVE_BLOCK', 'REVIEW_REQUIRED')), None)
        ayah = stop['ayah'] if stop else None
        reads = []
        for r in trace['acoustic_reads']:
            obs = r.get('observed', {})
            reads.append({k: r.get(k) for k in ('id', 'reason', 'window_ms', 'samples', 'outcome',
                         'prefix_rejections', 'prefix_comparisons')} | {
                 'text': obs.get('text'), 'units': obs.get('units'),
                 'unit_quality': obs.get('unit_quality'), 'attribution': obs.get('attribution'),
                 'hypotheses': obs.get('hypotheses'), 'truncated': obs.get('truncated')})
        anchors = [row[0] for row in case['rows'] if row[2] > 0]
        output.append({'model': case['model'], 'surah': case['surah'], 'first_stop': stop,
            'confirmed': len(anchors), 'anchors': anchors,
            'unresolved_at_or_after_stop': sum(r[0] >= ayah and r[2] == 0 for r in case['rows']) if ayah else 0,
            'anchors_after_stop': [a for a in anchors if a > ayah] if ayah else [],
            'planning_and_evaluation': [e for e in trace['transitions'] if e.get('ayah') == ayah],
            'reads': reads, 'denied': trace['denied_acoustic_reads']})
    dump('baseline-causes.json', {'source': str(source), 'historical': source.parent != P,
         'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'cases': output})
    for model in dict.fromkeys(c['model'] for c in output):
        subset = [c for c in output if c['model'] == model]
        print(model, sum(c['confirmed'] for c in subset),
              'zero-anchor', sum(not c['anchors'] for c in subset),
              'potential cascade', sum(c['unresolved_at_or_after_stop'] for c in subset))


if __name__ == '__main__':
    representations()
    causes()
