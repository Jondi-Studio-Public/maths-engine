"""Exam tier of senior-solve-equations-involving-indices-using-logarithms.

The oracle re-derives x from the rendered prompt alone and never imports the generator.
"""

from __future__ import annotations

import math
import re

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f3_logeq"
_GRADER = GRADERS["number"]


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_variety_and_new_shapes_over_hard():
    shapes = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("solve_indices_with_logarithms", seed=s, difficulty="hard")) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard


def test_no_text_defects_and_deterministic():
    for s in range(60):
        assert find_text_defects(_q(s)) == [], s
    assert _q(5).to_json() == _q(5).to_json()


_BASES = re.compile(r"Solve (\d+)\^\(x \+ (\d+)\) = (\d+)\^(?:x|\(x ([+-]) (\d+)\)) for x\.")
_QUAD = re.compile(r"Solve (\d+)\^\(2x\)(?: ([+-]) (\d+)\(\d+\^x\))? - (\d+) = 0 for x\.")


def _oracle(text):
    m = _BASES.search(text)
    if m:
        p, a, q, sgn, bb = m.groups()
        p, a, q = int(p), int(a), int(q)
        b = 0 if bb is None else (int(bb) if sgn == "+" else -int(bb))
        # (x + a) ln p = (x + b) ln q is linear in x: bisect on it.
        f = lambda x: (x + a) * math.log(p) - (x + b) * math.log(q)  # noqa: E731
        lo, hi = -100.0, 100.0
        increasing = f(hi) > f(lo)
        for _ in range(200):
            mid = (lo + hi) / 2
            if (f(mid) > 0) == increasing:
                hi = mid
            else:
                lo = mid
        return lo
    m = _QUAD.search(text)
    assert m, text
    p, sgn, c, k = m.groups()
    p, k = int(p), int(k)
    c = 0 if c is None else (int(c) if sgn == "+" else -int(c))
    y = (-c + math.sqrt(c * c + 4 * k)) / 2  # the positive root of y^2 + c y - k = 0
    return math.log(y) / math.log(p)


def test_key_matches_independent_oracle_over_400_seeds():
    bad = []
    for s in range(400):
        q = _q(s)
        x = _oracle(q.text)
        if abs(x - float(q.answer_text)) > 0.0005 + 1e-9:
            bad.append((s, q.text, x, q.answer_text))
    assert bad == [], bad[:3]


def test_both_forms_occur():
    kinds = {("x +" in _q(s).text) for s in range(40)}
    assert kinds == {True, False}


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for s in range(150):
        q = _q(s)
        assert len(q.distractors) >= 2
        assert _GRADER(q, q.answer_text).correct, s
        for d in q.distractors:
            if _GRADER(q, d).correct:
                bad.append((s, d, q.answer_text))
    assert bad == []


def test_grader_rejects_fixed_wrong_values():
    q = _q(2)
    x = float(q.answer_text)
    for wrong in (f"{x + 0.01:.3f}", f"{x - 0.01:.3f}", f"{-x:.3f}", f"{x + 1:.3f}"):
        assert not _GRADER(q, wrong).correct, wrong
