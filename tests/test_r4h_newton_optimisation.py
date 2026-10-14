"""Tests `uc2-newton-method-optimisation-step`: `r4h_newton_optimisation` and `exd_x_r4h_newton_optimisation`.

The oracle reads the function and the starting point back out of the text and does the Newton step with SymPy
(``x - H⁻¹ ∇f`` with SymPy's Hessian and matrix inverse).
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-newton-method-optimisation-step",
    "r4h_newton_optimisation",
    "exd_x_r4h_newton_optimisation",
)
SKILL = sup.skill_of(SKILL_ID)
X, Y = sympy.symbols("x y")
R = sympy.Rational
RUN = re.compile(r"[0-9xy⁰-⁹¹²³+\- ]*⁴[0-9xy⁰-⁹¹²³+\- ]*")
SUP = {"²": "2", "³": "3", "⁴": "4"}


def read_function(text):
    run = RUN.search(sup.clean(text)).group(0).strip()
    run = re.sub(r"([²³⁴])", lambda m: "**" + SUP[m.group(1)] + "*", run)
    run = re.sub(r"\*(\s|$)", r"\1", run)
    run = re.sub(r"(\d)([xy])", r"\1*\2", run)
    run = re.sub(r"([xy])([xy])", r"\1*\2", run)
    run = re.sub(r"\*\*(\d)([xy])", r"**\1*\2", run)
    return sympy.sympify(run, locals={"x": X, "y": Y})


def start_point(text):
    text = sup.clean(text)
    m = re.search(r"\((-?\d+), (-?\d+)\)", text)
    return R(m.group(1)), R(m.group(2))


def newton(f, x0, y0):
    grad = sympy.Matrix([sympy.diff(f, X), sympy.diff(f, Y)]).subs({X: x0, Y: y0})
    hess = sympy.hessian(f, (X, Y)).subs({X: x0, Y: y0})
    step = hess.inv() * grad
    return hess, (x0 - step[0], y0 - step[1])


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_key_is_the_one_variable_step():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        text = sup.clean(q.text)
        f = sup.sympy_expr(RUN.search(text).group(0).strip())
        x0 = R(re.search(r"(?:x₀ = |guess )(-?\d+)", text).group(1))
        x1 = x0 - sympy.diff(f, X).subs(X, x0) / sympy.diff(f, X, 2).subs(X, x0)
        assert R(q.answer_text) == x1, (seed, q.text, q.answer_text)


def test_medium_key_is_one_coordinate_of_the_step():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        f = read_function(q.text)
        x0, y0 = start_point(q.text)
        _h, (x1, y1) = newton(f, x0, y0)
        which = re.search(r"([xy])-coordinate", q.text).group(1)
        assert R(q.answer_text) == (x1 if which == "x" else y1), (seed, q.text, q.answer_text)


def test_hard_key_is_f_at_the_new_point():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        f = read_function(q.text)
        x0, y0 = start_point(q.text)
        _h, (x1, y1) = newton(f, x0, y0)
        assert R(q.answer_text) == f.subs({X: x1, Y: y1}), (seed, q.text, q.answer_text)


def test_tiers_differ_in_shape():
    for seed in range(30):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        assert "y" not in easy.split("f(x)")[-1][:40] or "f(x)" in easy
        assert "coordinate" in medium and "coordinate" not in hard
        assert "f" in hard and ("value" in hard or "What is f" in hard or "evaluate" in hard)


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=20, repeats=12, by="text")


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["fraction", "fraction", "fraction", "number"], [1, 1, 1, 2], [False, True, True, True])
    for seed in range(50):
        q = sup.exam(EXAM, seed)
        f = read_function(q.stem)
        x0, y0 = start_point(q.stem)
        hess, (x1, y1) = newton(f, x0, y0)
        values = [R(p.answer_text) for p in q.parts[:3]]
        assert values == [hess.det(), x1, y1], (seed, q.stem, values)
        change = f.subs({X: x1, Y: y1}) - f.subs({X: x0, Y: y0})
        assert abs(float(q.parts[3].answer_text) - float(change)) <= 0.00005 + 1e-12, (seed, q.stem)


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=20)
    sup.check_exam_variety(EXAM, distinct=30)
