"""Probability parts never offer an option outside [0, 1]."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mathgen import make


def _options(part):
    for d in part.distractors:
        text = str(getattr(d, "text", d))
        yield text, Fraction(text)


@pytest.mark.parametrize("topic", ["exd_x_r2l_bernoulli_chain", "exd_x_r2l_relfreq_chain"])
def test_fraction_options_of_probability_parts_are_possible_probabilities(topic):
    checked = 0
    for seed in range(1, 121):
        q = make(topic, seed=seed, difficulty="exam")
        for part in q.parts:
            if part.answer_kind != "fraction" or part.sensible_range != (0, 1):
                continue
            assert len(part.distractors) >= 2, (topic, seed, part.label)
            for text, value in _options(part):
                assert 0 <= value <= 1, (topic, seed, part.label, text)
                checked += 1
    assert checked > 200
