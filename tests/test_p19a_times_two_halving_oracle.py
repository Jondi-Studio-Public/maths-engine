"""Seed count is `N_SEEDS` (default 250 per tier; three-tier skill, so the
override runs at 400/tier = 1200), overridable by `P19A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p19a_times_two_halving"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("P19A_ORACLE_SEEDS", "250"))

_TWO_FACTS_RE = re.compile(r"You know 2 × (\d+) = (\d+) and 2 × (\d+) = (\d+)\. Use the right fact to find (\d+) ÷ 2")
_PRODUCT_RE = re.compile(r"(?:= |is |gives )(\d+)")


def _oracle_half(text: str) -> int:
    """Re-derive the halved value from `text` alone.

    The two-known-facts hard shape names both products explicitly and asks
    for a specific one, divided by 2 -- parsed with its own regex, never
    reusing the generic single-product path. Every other shape (easy,
    medium, and the hard sharing-between-two shape) states exactly one
    product; the answer is always that product // 2.
    """
    m = _TWO_FACTS_RE.search(text)
    if m is not None:
        pa, pb, target = int(m.group(2)), int(m.group(4)), int(m.group(5))
        assert target in (pa, pb), f"oracle: asked-about product {target} matches neither fact in {text!r}"
        assert target % 2 == 0
        return target // 2

    m = _PRODUCT_RE.search(text)
    assert m is not None, f"oracle: no product found in {text!r}"
    product = int(m.group(1))
    assert product % 2 == 0, f"oracle: odd product {product} in {text!r}"
    return product // 2


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer_text(tier):
    mismatches = []
    for q in _all(tier):
        oracle = _oracle_half(q.text)
        actual = int(q.answer_text)
        if oracle != actual:
            mismatches.append((q.seed, q.text, oracle, q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _all(tier, seeds=range(30)):
        oracle = _oracle_half(q.text)
        for d in q.distractors:
            assert int(d) != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"
