"""Tests `uc2-kkt-conditions-check`: `r4h_kkt_conditions` and `exd_x_r4h_kkt_conditions`.

The oracle reads the objective, the point and the constraint gradients out of the text and solves the stationarity
equations with SymPy (a linear solve for the multipliers; the constraint-and-stationarity system for the exam).
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = "uc2-kkt-conditions-check", "r4h_kkt_conditions", "exd_x_r4h_kkt_conditions"
SKILL = sup.skill_of(SKILL_ID)
X, Y = sympy.symbols("x y")
R = sympy.Rational
SQUARE = re.compile(r"(\d*)\(([xy]) ([+−]) ([\d.]+)\)²|(\d*)([xy])²")


def read_objective(text):
    """The sum of weighted squares in the text, in order, as a SymPy expression."""
    total = 0
    for m in SQUARE.finditer(text.split(" subject to ")[0].split(" and g")[0]):
        if m.group(2):
            w = int(m.group(1) or 1)
            c = R(m.group(4)) * (1 if m.group(3) == "+" else -1)
            total += w * (sympy.Symbol(m.group(2)) + c) ** 2
        else:
            total += int(m.group(5) or 1) * sympy.Symbol(m.group(6)) ** 2
    return total


LINEAR = re.compile(r"(?<![\w.)])(−?\d*)x(?: ([+−]) (\d*)y)?")


def read_linear(chunk):
    """(p, q) from text such as '2x − y', '-x - 3y', '3x' or 'y'."""
    chunk = sup.clean(chunk)
    m = re.search(r"(-?\d*)x ([+-]) (\d*)y", chunk)
    if m:
        p = -1 if m.group(1) == "-" else int(m.group(1) or 1)
        return p, int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
    m = re.search(r"(-?\d*)y", chunk)
    return 0, (-1 if m.group(1) == "-" else int(m.group(1) or 1))


def point(text):
    a, b = re.search(r"P = \((−?[\d.]+), (−?[\d.]+)\)", text).groups()
    return R(a.replace("−", "-")), R(b.replace("−", "-"))


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_key_solves_stationarity():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        f = read_objective(q.text)
        px, py = point(q.text)
        grad = sympy.Matrix([sympy.diff(f, X), sympy.diff(f, Y)]).subs({X: px, Y: py})
        gtext = re.search(r"(?:g\(x, y\) = |subject to |g\(x, y\) = )(−?\d*x [+−] \d*y)", q.text).group(1)
        p, q_ = read_linear(gtext)
        lam = sympy.Symbol("lam")
        sol = sympy.solve(list(grad + lam * sympy.Matrix([p, q_])), lam, dict=True)
        assert sol and R(q.answer_text) == sol[0][lam], (seed, q.text, q.answer_text)


def test_medium_key_solves_the_two_by_two_system():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        f = read_objective(q.text)
        px, py = point(q.text)
        grad = sympy.Matrix([sympy.diff(f, X), sympy.diff(f, Y)]).subs({X: px, Y: py})
        chunks = re.findall(r"(-?\d*x [+-] \d*y)", sup.clean(q.text))
        gradients = [read_linear(c) for c in chunks[:2]]
        l1, l2 = sympy.symbols("l1 l2")
        eqs = grad + l1 * sympy.Matrix(gradients[0]) + l2 * sympy.Matrix(gradients[1])
        sol = sympy.solve(list(eqs), [l1, l2], dict=True)[0]
        name = re.search(r"multiplier «?(λ[₁₂])|(λ[₁₂]) (?:in|as)|give (λ[₁₂])|find (λ[₁₂])|(λ[₁₂])", q.text)
        which = [g for g in name.groups() if g][0]
        assert R(q.answer_text) == sol[l1 if which == "λ₁" else l2], (seed, q.text, q.answer_text)


def test_hard_key_is_the_circle_multiplier():
    lam, x, y = sympy.symbols("lam x y")
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        text = sup.clean(q.text)
        m = re.search(r"(-?\d*)x ([+-]) (\d*)y", text)
        a = -1 if m.group(1) == "-" else int(m.group(1) or 1)
        b = int(m.group(3) or 1) * (1 if m.group(2) == "+" else -1)
        r2 = int(re.search(r"x² \+ y² (?:≤|-) ?(\d+)", text).group(1))
        sols = sympy.solve([a + lam * 2 * x, b + lam * 2 * y, x**2 + y**2 - r2], [x, y, lam], dict=True)
        positive = [s[lam] for s in sols if s[lam] > 0]
        assert positive and R(q.answer_text) == positive[0], (seed, q.text, q.answer_text)


def test_tiers_differ_in_shape():
    for seed in range(30):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        assert (
            "λ₁" not in easy
            and "λ" in medium
            and "λ₁" in medium.replace("λ₂", "λ₁")
            and "x² + y²" in hard
            and "x² + y²" not in easy + medium
        )


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=25, repeats=12, by="text")


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["fraction"] * 4, [1, 1, 1, 2], [False, True, True, True])
    lam = sympy.Symbol("lam")
    for seed in range(50):
        q = sup.exam(EXAM, seed)
        f = read_objective(q.stem)
        cons = re.search(r"(-?\d*x [+-] \d*y) ≤ (-?[\d.]+)", sup.clean(q.stem))
        p, q_ = read_linear(cons.group(1))
        c = R(cons.group(2))
        sol = sympy.solve(
            [sympy.diff(f, X) + lam * p, sympy.diff(f, Y) + lam * q_, p * X + q_ * Y - c], [X, Y, lam], dict=True
        )[0]
        values = [R(part.answer_text) for part in q.parts]
        assert values == [sol[lam], sol[X], sol[Y], f.subs({X: sol[X], Y: sol[Y]})], (seed, q.stem, values)
        assert values[0] > 0


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=20)
    sup.check_exam_variety(EXAM, distinct=15)
