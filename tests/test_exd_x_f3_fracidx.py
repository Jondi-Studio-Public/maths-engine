"""Exam tier of senior-use-radicals-and-convert-to-and-from-fractional.

The oracle re-derives the integer key from the rendered prompt alone and never imports the generator.
"""

from __future__ import annotations

import re

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f3_fracidx"
_GRADER = GRADERS["number"]


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


_SOLVE = re.compile(r"x\^\((-?)(\d+)/(\d+)\) = (?:1/)?(\d+)\.")
_MIX = re.compile(r"Evaluate (\d+)\^\((\d+)/(\d+)\) ([÷×]) (\d+)\^\((-?)(\d+)/(\d+)\)\.")


def _oracle(text):
    m = _SOLVE.search(text)
    if m:
        neg, mm, nn, big = m.groups()
        mm, nn, big = int(mm), int(nn), int(big)
        # x^(m/n) = N  (or x^(-m/n) = 1/N) means x^m = N^n exactly; search the integers.
        hits = [x for x in range(1, 20000) if x**mm == big**nn]
        assert len(hits) == 1, (text, hits)
        return hits[0]
    m = _MIX.search(text)
    assert m, text
    b1, m1, n1, op, b2, neg, m2, n2 = m.groups()
    b1, m1, n1, b2, m2, n2 = map(int, (b1, m1, n1, b2, m2, n2))
    e2 = -m2 / n2 if neg else m2 / n2
    val = b1 ** (m1 / n1) / b2**e2 if op == "÷" else b1 ** (m1 / n1) * b2**e2
    assert abs(val - round(val)) < 1e-6, (text, val)
    return round(val)


def test_key_matches_independent_oracle_over_400_seeds():
    bad = []
    for s in range(400):
        q = _q(s)
        if _oracle(q.text) != q.answer:
            bad.append((s, q.text, q.answer))
    assert bad == [], bad[:3]


def test_both_forms_occur():
    texts = [_q(s).text for s in range(40)]
    assert any(t.startswith("Find") for t in texts) and any(t.startswith("Evaluate") for t in texts)


def test_grader_accepts_key_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for s in range(200):
        q = _q(s)
        assert len(q.distractors) >= 2
        assert _GRADER(q, q.answer_text).correct, s
        bad += [(s, d) for d in q.distractors if _GRADER(q, d).correct]
    assert bad == []


def test_grader_rejects_fixed_wrong_values():
    q = _q(1)
    k = q.answer
    for wrong in (str(k + 1), str(k - 1), str(-k), "0"):
        assert not _GRADER(q, wrong).correct, wrong
