"""Tests the exam source of `p8c_confidence_interval_mean` (`exd_x_s3_ci`): the key is re-derived from the rendered
prompt alone, with exact fractions."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s3_ci"
HARD = "p8c_confidence_interval_mean"
Z = {"90%": Fraction("1.645"), "95%": Fraction("1.96"), "99%": Fraction("2.576")}


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _round1(v: Fraction) -> Fraction:
    scaled = v * 10
    return Fraction(int(scaled + Fraction(1, 2)) if scaled >= 0 else -int(-scaled + Fraction(1, 2)), 10)


def _oracle(text: str):
    new = re.search(r"find the (\d+%) confidence interval", text).group(1)
    interval = re.search(r"\(([\d.]+), ([\d.]+)\)", text)
    if interval:
        old = re.match(r"A (\d+%)", text).group(1)
        lo, hi = Fraction(interval.group(1)), Fraction(interval.group(2))
        centre, half = (lo + hi) / 2, (hi - lo) / 2
    else:
        old = re.search(r"lower bound of the (\d+%)", text).group(1)
        centre = Fraction(re.search(r"sample mean of ([\d.]+)", text).group(1))
        half = centre - Fraction(re.search(r"interval for the population mean is ([\d.]+)", text).group(1))
    se = half / Z[old]
    return _round1(centre - Z[new] * se), _round1(centre + Z[new] * se)


def _bounds(answer_text: str):
    lo, hi = re.findall(r"[\d.]+", answer_text)
    return Fraction(lo), Fraction(hi)


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        if _bounds(q.answer_text) != _oracle(q.text):
            bad.append((seed, q.text, q.answer_text, _oracle(q.text)))
    assert bad == [], bad[:3]


def test_both_framings_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("lower bound of" in t for t in texts) and any("interval for the population mean of" in t for t in texts)


def test_grader_accepts_key_and_rejects_every_distractor_over_100_seeds():
    grader = GRADERS["pair"]
    for seed in range(100):
        q = _make(seed)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) == 3
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, d, q.answer_text)


def test_prompt_says_not_to_round_intermediate_values():
    for seed in range(300):
        assert "do not round any intermediate value" in _make(seed).text


def test_rounding_the_half_width_midway_is_not_needed_for_the_key():
    # The key is exact-then-rounded, so it is derivable from the prompt with the stated z values and no early rounding.
    for seed in range(300):
        q = _make(seed)
        assert _bounds(q.answer_text) == _oracle(q.text)
