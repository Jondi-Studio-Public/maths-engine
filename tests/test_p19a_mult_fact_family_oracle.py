"""Seed count is `N_SEEDS` (default 250 per tier, >= 1000 questions over the
three tiers -- since this is a three-tier skill, three-tier convention runs
the override at 400/tier = 1200), overridable by `P19A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p19a_mult_fact_family"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("P19A_ORACLE_SEEDS", "250"))

_FACT_RE = re.compile(r"(\d+) × (\d+) = \?")
_ANSWER_RE = re.compile(r"^product = (\d+), quotient = (\d+)$")
_DIVISOR_RE_QUOTITIVE = re.compile(r"How many groups of (\d+)")
_DIVISOR_RE_PARTITIVE = re.compile(r"(?:Share|Split) this product equally (?:into|between) (\d+) groups")
_DIVISOR_RE_EXPLICIT = re.compile(r"[÷/] (\d+)")
_DIVISOR_RE_DIVIDE_BY = re.compile(r"[Dd]ivide (?:the result |it )?by (\d+)")


def _oracle_pair(text: str) -> tuple[int, int]:
    """Re-derive (product, quotient) from the question text alone.

    The multiplication fact gives (first, second) -- product = first *
    second. The divisor is whichever number the text names as the thing to
    divide the product by (explicit "? ÷ D", quotitive "groups of D", or
    partitive "into/between D groups"); the quotient is product // divisor.
    """
    m = _FACT_RE.search(text)
    assert m is not None, f"oracle: no multiplication fact found in {text!r}"
    first, second = int(m.group(1)), int(m.group(2))
    product = first * second

    m = _DIVISOR_RE_QUOTITIVE.search(text)
    if m is None:
        m = _DIVISOR_RE_PARTITIVE.search(text)
    if m is None:
        m = _DIVISOR_RE_DIVIDE_BY.search(text)
    if m is None:
        m = _DIVISOR_RE_EXPLICIT.search(text)
    assert m is not None, f"oracle: no divisor found in {text!r}"
    divisor = int(m.group(1))
    assert product % divisor == 0, f"oracle: {product} not divisible by {divisor} in {text!r}"
    return product, product // divisor


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer_text(tier):
    mismatches = []
    for q in _all(tier):
        oracle = _oracle_pair(q.text)
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None, q.answer_text
        actual = (int(m.group(1)), int(m.group(2)))
        if oracle != actual:
            mismatches.append((q.seed, q.text, oracle, q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_answer_shape_is_recognised(tier):
    for q in _all(tier, seeds=range(20)):
        assert _ANSWER_RE.match(q.answer_text), q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _all(tier, seeds=range(30)):
        oracle = _oracle_pair(q.text)
        for d in q.distractors:
            m = _ANSWER_RE.match(d)
            assert m is not None, d
            actual = (int(m.group(1)), int(m.group(2)))
            assert actual != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"
