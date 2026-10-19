"""Exam tier of senior-use-recursion-to-generate-a-geometric-sequence.

The oracle re-derives the four generated terms from the rendered prompt alone with exact fractions and
never imports the generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f3_georec"
_GRADER = GRADERS["ordered_list"]


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


_TWO = re.compile(r"has t(\d+) = (-?\d+) and t(\d+) = (-?\d+), and its common ratio r is (positive|negative)")
_PAR = re.compile(r"are k, k ([+-]) (\d+) and k ([+-]) (\d+), where k")


def _oracle(text):
    m = _TWO.search(text)
    if m:
        p, tp, q, tq, sign = m.groups()
        p, tp, q, tq = map(int, (p, tp, q, tq))
        ratio = Fraction(tq, tp)
        roots = [Fraction(c) for c in range(-50, 51) if c and (c < 0) == (sign == "negative") and c ** (q - p) == ratio]
        assert len(roots) == 1, (text, roots)
        r = roots[0]
        t1 = tp / r ** (p - 1)
    else:
        m = _PAR.search(text)
        assert m, text
        s1, a, s2, b = m.groups()
        a = int(a) * (1 if s1 == "+" else -1)
        b = int(b) * (1 if s2 == "+" else -1)
        t1 = Fraction(a * a, b - 2 * a)  # from (k + a)^2 = k (k + b)
        r = (t1 + a) / t1
        assert (t1 + b) / (t1 + a) == r
    return [int(t1 * r**n) for n in range(1, 5)]


def test_key_matches_independent_oracle_over_400_seeds():
    bad = [(s, _q(s).text) for s in range(400) if _oracle(_q(s).text) != _q(s).answer]
    assert bad == [], bad[:3]


def test_both_forms_occur():
    texts = [_q(s).text for s in range(40)]
    assert any("common ratio" in t for t in texts) and any("k is a non-zero" in t or "where k" in t for t in texts)


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for s in range(150):
        q = _q(s)
        assert len(q.distractors) == 3
        assert _GRADER(q, q.answer_text).correct, s
        bad += [(s, d) for d in q.distractors if _GRADER(q, d).correct]
    assert bad == []


def test_grader_rejects_fixed_wrong_values():
    q = _q(2)
    t = q.answer
    for wrong in (", ".join(map(str, t[::-1])), ", ".join(map(str, t[:-1])), ", ".join(str(v + 1) for v in t), ""):
        assert not _GRADER(q, wrong).correct, wrong
