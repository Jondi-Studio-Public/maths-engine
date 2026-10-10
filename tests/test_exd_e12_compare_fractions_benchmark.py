"""Tests the hard-tier `benchmark` mode of `compare_common_fractions`, which asks which of two fractions is
closer to one half.

`benchmark` mode is a different task from the symbol comparison the easy/medium tiers ask, and it never
changes the default (non-benchmark) rendering.

The two fractions must not be drawn independently with no constraint relating them: ties (equidistant from 1/2)
and equal-value pairs would then be possible, the generator would silently answer with the first-listed
fraction in a tie, and it would ALSO offer "They are equally close" as a distractor -- the mathematically
correct option, marked wrong.

This test asserts, over >=1000 seeds of the hard source's own params, that:
  1. no rendered question has its two fractions equidistant from one half, and
  2. no rendered question shows two fractions of equal value.

It is written against `question.text` and `question.answer_text` only, never against the generator's
internals, so it is independent of how the generator draws its fractions.
"""

import re
from fractions import Fraction

import mathgen

# The skill's own hard source params (skills/primary-number/compare-common-fractions.yaml)
HARD_PARAMS = {"max_den": 12, "include_equal": True, "benchmark": True}
SEEDS = 1000


def _parse_two_fractions(text):
    pairs = re.findall(r"(\d+)/(\d+)", text)
    assert len(pairs) == 2, f"expected exactly 2 fractions in text, got {pairs!r} from {text!r}"
    return [Fraction(int(n), int(d)) for n, d in pairs]


def test_no_tie_and_no_equal_value_pair_over_1000_seeds():
    half = Fraction(1, 2)
    tie_failures = []
    equal_value_failures = []
    for seed in range(SEEDS):
        q = mathgen.make("compare_common_fractions", seed=seed, difficulty="hard", **HARD_PARAMS)
        left, right = _parse_two_fractions(q.text)
        if abs(left - half) == abs(right - half):
            tie_failures.append(seed)
        if left == right:
            equal_value_failures.append(seed)
    assert not tie_failures, (
        f"{len(tie_failures)}/{SEEDS} seeds are equidistant from one half (a tie): {tie_failures[:20]}..."
    )
    assert not equal_value_failures, (
        f"{len(equal_value_failures)}/{SEEDS} seeds show two fractions of equal value: {equal_value_failures[:20]}..."
    )


def test_answer_is_one_of_the_two_rendered_fractions():
    for seed in range(SEEDS):
        q = mathgen.make("compare_common_fractions", seed=seed, difficulty="hard", **HARD_PARAMS)
        left, right = _parse_two_fractions(q.text)
        left_str, right_str = re.findall(r"\d+/\d+", q.text)
        assert q.answer_text in (left_str, right_str), (
            f"seed {seed}: answer_text {q.answer_text!r} is not one of the two rendered "
            f"fractions {left_str!r}, {right_str!r}"
        )


def test_default_is_non_benchmark_and_unchanged():
    """The default (non-benchmark) rendering must stay byte-identical."""
    a = mathgen.make("compare_common_fractions", seed=5, max_den=12, include_equal=True)
    b = mathgen.make("compare_common_fractions", seed=5, max_den=12, include_equal=True, benchmark=False)
    assert a.text == b.text
    assert a.answer == b.answer
