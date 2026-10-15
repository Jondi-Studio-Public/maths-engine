"""Exam tier of senior-recognise-and-use-the-recursive-definition-of-an.

The oracle re-derives (first term, d) from the RENDERED PROMPT alone, with
exact fractions, and never imports the generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f3_recdef"
HARD = "arithmetic_recursive_definition_state"
_GRADER = GRADERS["pair"]


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


_TWO = re.compile(r"has t(\d+) = (-?\d+) and t(\d+) = (-?\d+)\.")
_SUM = re.compile(r"t(\d+) \+ t(\d+) = (-?\d+) and t(\d+) = (-?\d+)\.")
_PAR = re.compile(r"are (.+?), (.+?) and (.+?), where k")


def _lin(expr: str):
    """(coefficient of k, constant) from text like '2k + 2', '-k - 3', 'k', '-5'."""
    expr = expr.replace(" ", "")
    coef = const = 0
    for sign, body in re.findall(r"([+-]?)([^+-]+)", expr):
        mult = -1 if sign == "-" else 1
        if body.endswith("k"):
            coef += mult * (int(body[:-1]) if body[:-1] else 1)
        else:
            const += mult * int(body)
    return Fraction(coef), Fraction(const)


def _oracle(text):
    m = _TWO.search(text)
    if m:
        p, tp, q, tq = map(int, m.groups())
        d = Fraction(tq - tp, q - p)
        return tp - (p - 1) * d, d
    m = _SUM.search(text)
    if m:
        p, q, s, r, tr = map(int, m.groups())
        # t1 + (n-1)d: solve [[2, p+q-2], [1, r-1]] [a, d] = [s, tr]
        det = Fraction(2 * (r - 1) - (p + q - 2))
        return (s * (r - 1) - (p + q - 2) * tr) / det, (2 * tr - s) / det
    m = _PAR.search(text)
    assert m, text
    (c1, e1), (c2, e2), (c3, e3) = (_lin(g) for g in m.groups())
    k = (e1 + e3 - 2 * e2) / (2 * c2 - c1 - c3)
    t = [c * k + e for c, e in ((c1, e1), (c2, e2), (c3, e3))]
    assert t[1] - t[0] == t[2] - t[1]
    return t[0], t[1] - t[0]


def test_key_matches_independent_oracle_over_400_seeds():
    bad = []
    for s in range(400):
        q = _q(s)
        a, d = _oracle(q.text)
        if (a, d) != (q.answer["first_term"], q.answer["d"]):
            bad.append((s, q.text, (a, d), q.answer))
    assert bad == [], bad[:3]


def test_all_three_forms_occur():
    kinds = set()
    for s in range(60):
        t = _q(s).text
        kinds.add("two" if _TWO.search(t) else "sum" if _SUM.search(t) else "par")
    assert kinds == {"two", "sum", "par"}


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for s in range(150):
        q = _q(s)
        assert len(q.distractors) >= 2
        if not _GRADER(q, q.answer_text).correct:
            bad.append(("key", s))
        for d in q.distractors:
            if _GRADER(q, d).correct:
                bad.append(("distractor", s, d))
    assert bad == [], bad


def test_grader_rejects_fixed_wrong_values():
    q = _q(3)
    a, d = q.answer["first_term"], q.answer["d"]
    for wrong in (f"first_term = {a + 1}, d = {d}", f"first_term = {a}, d = {d + 1}", f"first_term = {a}, d = {-d}"):
        assert not _GRADER(q, wrong).correct, wrong
