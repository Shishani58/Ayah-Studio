"""Declared whole alif-name proof within its unchanged canonical opening word."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

task = Path(__file__).parent.resolve()
parent = task.parent / '2026-10-14-fast-muaalem-local-analysis'
previous = task.parent / '2026-10-14-fast-pronunciation-variants'
sys.path.insert(0, str(parent))
import adapter  # noqa: E402

# Reuse exactly the established pure normalization, without running its driver.
module = ast.parse((previous / 'collapse_raw_tempo_probe.py').read_text())
function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'collapse')
namespace = {'LONG': {'ا', 'ۦ', 'ۥ', 'ـ'}}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<retained-collapse>', 'exec'), namespace)
collapse = namespace['collapse']
declarations_path = task / 'named-letter-mappings.json'
declarations = {c['surah']: c for c in json.loads(declarations_path.read_text())['cases']}
original_init = adapter.MuaalemLocal.__init__


def initialize(self, selected):
    original_init(self, selected)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    self.version += '-declared-alif-' + digest
    self.runtime_identity['declared_named_openings'] = {
        'source_sha256': digest,
        'mapping_sha256': hashlib.sha256(declarations_path.read_bytes()).hexdigest(),
        'scope': 'experimental surahs 2/32; whole first alif-name, not arbitrary subword',
    }


def proof_word_ends(self, words, rendered):
    result = []
    for word, units in zip(words, rendered):
        ends = (len(units),)
        if word.index == 0 and word.ayah == 1 and word.surah in declarations:
            declaration = declarations[word.surah]
            first = declaration['mapping'][0]
            assert declaration['canonical_words'] == [word.text]
            assert first['char'] == 'ا' and not first['deleted'] and first['pos'][0] == 0
            # Verify the complete representation and the original-character
            # mapping before exposing any internal proof endpoint.
            assert collapse(tuple(declaration['phones']))[0] == tuple(units)
            name = collapse(tuple(first['phones']))[0]
            assert name == tuple('ءَلِف') and tuple(units[:len(name)]) == name
            ends = (len(name), len(units))
        result.append(ends)
    return tuple(result)


adapter.MuaalemLocal.__init__ = initialize
adapter.MuaalemLocal.proof_word_ends = proof_word_ends
os.environ.setdefault('MUA_PROBE_SURAHS', '32')
os.environ.setdefault('MUA_PROBE_OUTPUT', str(task / 'named-alif-32.json'))
os.environ.setdefault('MUA_AUDIT_DIR', str(task / 'named-alif-32-audit'))
runpy.run_path(str(previous / 'collapse_raw_tempo_probe.py'), run_name='__main__')
