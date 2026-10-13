"""The correct option of an estimate MCQ must not sit at one fixed rank in value order.

Round-2 scope section (d): estimate length/area/mass/capacity always put the key 2nd of 4 (distractors x10, /10, x100),
`p12c_estimate_limit_ah` and `interval_estimate_proportion_margin` fixed it per tier. Over 100 seeds the key's rank
among the four options sorted by value must take at least 3 values on every tier.
"""

import re

import pytest

from mathgen import make

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
TOPICS = [
    "estimate_length_reference",
    "estimate_area_reference",
    "estimate_mass_reference",
    "estimate_volume_capacity_reference",
    "p12c_estimate_limit_ah",
    "interval_estimate_proportion_margin",
]


def _value(option: str) -> float:
    return float(_NUM.search(option.replace(",", "")).group())


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("topic", TOPICS)
def test_key_rank_varies_with_seed(topic, difficulty):
    ranks = set()
    for seed in range(100):
        q = make(topic, seed=seed, difficulty=difficulty)
        options = [q.answer_text, *q.distractors]
        assert len(options) == 4, (topic, difficulty, seed, options)
        key = _value(q.answer_text)
        ranks.add(1 + sum(_value(o) < key for o in options))
    assert len(ranks) >= 3, f"{topic}/{difficulty}: key rank only took {sorted(ranks)}"
