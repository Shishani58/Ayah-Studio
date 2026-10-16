"""True n-best below Whisper's num_return_sequences replication wrapper."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers.generation.utils import GenerationMixin

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[2]))
from surah_cutter.core.model_worker import Acoustic  # noqa: E402

assert not (P / 'beam-results.json').exists()
config = json.loads((P / 'inputs.json').read_text())
previous = json.loads((P / 'results.json').read_text())['results']
arrays = np.load(P / 'inputs.npz')
acoustic = Acoustic('whisper-tarteel-base', Path(config['weights']))
model, processor = acoustic.model, acoustic.processor
results = []
for case in config['cases']:
    if case['end_ms'] - case['start_ms'] != 1500:
        continue
    for control in ['audio', 'zero']:
        pcm = arrays[case['key']]
        if control == 'zero':
            pcm = np.zeros_like(pcm)
        features = processor(pcm, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        with torch.inference_mode():
            encoded = model.model.encoder(features.input_features, return_dict=True)
            outputs = {}
            for beams in [1, 5]:
                generation = copy.deepcopy(model.generation_config)
                generation.forced_decoder_ids = None
                result = GenerationMixin.generate(
                    model, encoder_outputs=encoded,
                    attention_mask=features.attention_mask,
                    decoder_input_ids=torch.tensor([processor.tokenizer.prefix_tokens]),
                    generation_config=generation, max_new_tokens=64,
                    do_sample=False, num_beams=beams, num_return_sequences=beams,
                    return_dict_in_generate=True, output_scores=True)
                outputs[str(beams)] = processor.batch_decode(result.sequences, skip_special_tokens=True)
        baseline = next(r for r in previous if r['key'] == case['key'] and r['control'] == control)
        row = dict(key=case['key'], control=control, readings=outputs,
                   raw_greedy_equals_wrapper=outputs['1'] == baseline['readings']['1'],
                   encoder_input_samples=len(pcm))
        results.append(row)
        (P / 'beam-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(row, flush=True)
