from types import SimpleNamespace

import numpy as np

from surah_cutter.core.emissions import OnnxEmitter, generate_emissions


class SingleWindowSession:
    def __init__(self):
        self.batch_sizes = []

    def get_inputs(self):
        return [SimpleNamespace(name="input_values")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, _outputs, inputs):
        batch = inputs["input_values"]
        self.batch_sizes.append(len(batch))
        assert len(batch) == 1, "Neighbouring windows must not share INT8 scales"
        # Match wav2vec2's frame count so context/tail trimming is exercised.
        return [np.zeros((len(batch), batch.shape[1] // 320 - 1, 3), np.float32)]


def test_windowed_int8_inference_is_single_window_by_default():
    session = SingleWindowSession()
    pcm = np.ones(36_000, np.float32)  # 2.25 seconds, including a padded tail.
    result = generate_emissions(session, pcm, window_s=1, context_s=1)

    assert session.batch_sizes == [1, 1, 1]
    assert result.shape == (113, 3)
    np.testing.assert_allclose(np.exp(result).sum(axis=1), 1.0)


def test_emitter_uses_single_window_default_too():
    session = SingleWindowSession()
    emitter = OnnxEmitter(session, {"[PAD]": 0, "a": 1, "|": 2}, blank_id=0)
    result = emitter.emissions(np.ones(32_000, np.float32), window_s=1, context_s=1)

    assert session.batch_sizes == [1, 1]
    assert result.shape == (100, 3)
