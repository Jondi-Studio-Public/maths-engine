"""Exam source of the Pascal's triangle skill: `exd_x_s4_pascal`."""

from __future__ import annotations

import math
import re

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s4_pascal"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make("pascals_triangle_properties", seed=seed, difficulty="hard", tier="hard")


def _oracle(text):
    """Find the row from the printed clue by brute force, then answer the printed question."""
    value = max(int(v) for v in re.findall(r"\d+", text))  # the clue is the largest number printed
    if "largest" in text or "greatest" in text or "biggest" in text:
        row = next(n for n in range(1, 40) if 2**n == value)
        return max(math.comb(row, k) for k in range(row + 1))
    if "sum of all" in text or "total of all" in text or "add up to" in text:
        row = next(n for n in range(2, 40) if math.comb(n, 2) == value)
        return sum(math.comb(row, k) for k in range(row + 1))
    row = next(n for n in range(2, 40) if math.comb(n, 2) == value)
    return [math.comb(row, k) for k in range(row + 1)][3]


def test_independent_oracle_matches_key():
    bad = []
    for seed in range(400):
        q = _exam(seed)
        if _oracle(q.text) != q.answer:
            bad.append((seed, q.text, q.answer, _oracle(q.text)))
    assert bad == []


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
