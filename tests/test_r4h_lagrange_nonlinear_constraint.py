"""Tests `uc2-lagrange-multiplier-nonlinear-constraint`: `r4h_lagrange_nonlinear_constraint` and its exam chain.

The oracle reads the objective and the constraint out of the text and solves the Lagrange system ``∇f = λ∇g``,
``g = c`` with SymPy, keeping the real solution that gives the largest value of ``f``.
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-lagrange-multiplier-nonlinear-constraint",
    "r4h_lagrange_nonlinear_constraint",
    "exd_x_r4h_lagrange_nonlinear_constraint",
)
SKILL = sup.skill_of(SKILL_ID)
X, Y, LAM = sympy.symbols("x y lam", real=True)
R = sympy.Rational


def read_problem(text):
    """(objective, constraint expression, constant) from the stem."""
    text = sup.clean(text)
    cons = re.search(r"(?:(\d*)x² \+ (?:(xy) \+ )?(\d*)y²) = (\d+(?:\.\d+)?)", text)
    p = int(cons.group(1) or 1)
    q = int(cons.group(3) or 1)
    g = p * X**2 + (X * Y if cons.group(2) else 0) + q * Y**2
    head = text[: cons.start()] + " " + text[cons.end() :]
    if re.search(r"\bxy\b", head.replace(cons.group(0), "")) and not re.search(r"x [+-] \d*y", head):
        f = X * Y
    else:
        m = re.search(r"(-?\d*)x ([+-]) (\d*)y", head)
        a = -1 if m.group(1) == "-" else int(m.group(1) or 1)
        b = int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
        f = a * X + b * Y
    return f, g, R(cons.group(4))


def maximise(f, g, c):
    sols = sympy.solve(
        [sympy.diff(f, X) - LAM * sympy.diff(g, X), sympy.diff(f, Y) - LAM * sympy.diff(g, Y), g - c],
        [X, Y, LAM],
        dict=True,
    )
    real = [s for s in sols if all(v.is_real for v in s.values())]
    best = max(real, key=lambda s: sympy.nsimplify(f.subs({X: s[X], Y: s[Y]})))
    return best, sympy.nsimplify(f.subs({X: best[X], Y: best[Y]}))


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_keys_are_the_lagrange_maximum():
    for tier in sup.TIERS:
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            f, g, c = read_problem(q.text)
            assert R(q.answer_text) == maximise(f, g, c)[1], (tier, seed, q.text, q.answer_text)


def test_tiers_differ_in_shape():
    for seed in range(40):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        assert "x² + y² =" in easy and "xy +" not in easy
        assert "x² + xy + y²" in hard and "x² + xy + y²" not in easy + medium
        assert ("ellipse" in medium) or ("constraint" in medium) or ("xy" in medium)


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=15, repeats=15, by="text")


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["fraction"] * 4, [1, 1, 1, 2], [False, True, True, True])
    for seed in range(50):
        q = sup.exam(EXAM, seed)
        f, g, c = read_problem(q.stem)
        best, value = maximise(f, g, c)
        values = [R(p.answer_text) for p in q.parts]
        assert values == [best[LAM], best[X], best[Y], value], (seed, q.stem, values)
        assert values[0] > 0  # the multiplier of the maximum, for grad f = lam * grad g


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=20)
    sup.check_exam_variety(EXAM, distinct=15)
