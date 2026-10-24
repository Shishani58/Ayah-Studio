"""Exact alignment must preserve the original stable Viterbi tie order."""

import numpy as np
import pytest

from surah_cutter.core._ctc import forced_align


@pytest.mark.parametrize("targets, expected", [
    ([1, 2], [1, 2, 0, 0, 0, 0]),
    ([1, 1], [1, 0, 1, 0, 0, 0]),
    ([2, 1, 2], [2, 1, 2, 0, 0, 0]),
])
def test_equal_scores_keep_stay_then_step_then_skip(targets, expected):
    path, scores = forced_align(np.zeros((1, 6, 3)), np.array([targets]))
    np.testing.assert_array_equal(path[0], expected)
    np.testing.assert_array_equal(scores, np.zeros((1, 6)))


def test_negative_infinity_does_not_become_a_plausible_token():
    matrix = np.full((1, 6, 3), -np.inf)
    matrix[0, np.arange(6), [0, 1, 1, 0, 2, 0]] = 0
    path, scores = forced_align(matrix, np.array([[1, 2]]))
    np.testing.assert_array_equal(path[0], [0, 1, 1, 0, 2, 0])
    np.testing.assert_array_equal(scores, np.zeros((1, 6)))
