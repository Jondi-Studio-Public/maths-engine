"""Exam source of the proportion interval-estimate skill: `exd_x_s4_interval`."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s4_interval"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make("interval_estimate_proportion_margin", seed=seed, difficulty="hard", variant="solve_for_n")


def _oracle(text):
    """Smallest n with z*sqrt(p(1-p)/n) <= half-width, from the two printed end points and z."""
    z = Fraction(re.search(r"z = (\d+\.\d+)", text).group(1))
    nums = re.findall(r"(\d+\.\d+)(%?)", text)
    ends = [Fraction(v) for v, _pct in nums if Fraction(v) != z][:2]
    if any(pct for _v, pct in nums):
        ends = [e / 100 for e in ends]
    lo, hi = ends
    p, e = (lo + hi) / 2, (hi - lo) / 2
    n = max(1, math.floor(z * z * p * (1 - p) / (e * e)) - 2)
    while z * z * p * (1 - p) / n > e * e:
        n += 1
    return n


def test_independent_oracle_matches_key():
    bad = []
    for seed in range(400):
        q = _exam(seed)
        if _oracle(q.text) != q.answer:
            bad.append((seed, q.text, q.answer, _oracle(q.text)))
    assert bad == []


def test_percent_and_decimal_forms_both_occur():
    texts = [_exam(s).text for s in range(60)]
    assert any("%)" in t or "%." in t or "% and" in t for t in texts if "%" in t.split("confidence")[1])
    assert any("%" not in t.split("confidence interval")[1] for t in texts)


def test_grader_accepts_key_and_rejects_every_distractor():
    grader = GRADERS["number"]
    bad = []
    for seed in range(150):
        q = _exam(seed)
        assert len(q.distractors) >= 3
        if not grader(q, q.answer_text).correct:
            bad.append(("key rejected", seed))
        for d in q.distractors:
            if grader(q, d).correct:
                bad.append(("distractor accepted", seed, d))
    assert bad == []
