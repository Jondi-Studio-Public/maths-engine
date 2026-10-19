"""Tests the five circle skills (`w5_circle_*` topics and their exam chains).

The oracles read the circle, the point or the condition back out of the question TEXT and re-derive the key with SymPy
(completing the square symbolically, solving the three-point system, solving for the unknown), never from the
generator's own helpers.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen.generators.w5_circle_equations_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import w5_support as sup

X, Y, F_, T = sympy.symbols("x y F t")
_TR = standard_transformations + (implicit_multiplication_application, convert_xor)

FORMS, FORMS_X = "w5_circle_forms", "exd_x_w5_circle_forms"
SQ, SQ_X = "w5_circle_complete_square", "exd_x_w5_circle_complete_square"
POS, POS_X = "w5_circle_point_position", "exd_x_w5_circle_point_position"
UNK, UNK_X = "w5_circle_unknown", "exd_x_w5_circle_unknown"
TAN, TAN_X = "w5_circle_tangent", "exd_x_w5_circle_tangent"
SKILLS = {
    FORMS: sup.skill_of("circle-equation-centre-radius-form"),
    SQ: sup.skill_of("circle-complete-the-square"),
    POS: sup.skill_of("circle-point-position"),
    UNK: sup.skill_of("circle-find-unknown-constant"),
    TAN: sup.skill_of("circle-tangent-at-a-point"),
}
EXAMS = {FORMS: FORMS_X, SQ: SQ_X, POS: POS_X, UNK: UNK_X, TAN: TAN_X}
ANSWER_KINDS = {FORMS: "ordered_list", SQ: "ordered_list", POS: "mcq", UNK: "number", TAN: "number"}


def _expr(text: str):
    return parse_expr(text, local_dict={"x": X, "y": Y, "F": F_, "t": T}, transformations=_TR)


def circle_of(equation: str):
    """(h, k, r^2) of the circle written as `lhs = rhs`, in any form: completes the square symbolically."""
    left, right = equation.split("=")
    poly = sympy.expand(_expr(left) - _expr(right))
    m = poly.coeff(X, 2)
    assert m == poly.coeff(Y, 2) and m != 0, equation
    d, e = poly.coeff(X, 1), poly.coeff(Y, 1)
    f = poly.subs({X: 0, Y: 0})
    h, k = -d / (2 * m), -e / (2 * m)
    rr = h * h + k * k - f / m
    return h, k, rr


def _nums(text: str):
    return [Fraction(v) for v in re.findall(r"-?\d+(?:/\d+)?", text)]


def _tuple(sympy_values):
    return [Fraction(str(v)) for v in sympy_values]


def _check_key(q, expected):
    assert [Fraction(v) for v in q.answer] == [Fraction(str(v)) for v in expected], (q.text, q.answer_text, expected)


# -- skill wiring and the generic checks ----------------------------------------------------------------------------


@pytest.mark.parametrize("topic", list(SKILLS))
def test_wiring(topic):
    skill = SKILLS[topic]
    sup.check_wiring(skill, topic, EXAMS[topic], skill.answer)
    assert skill.answer == ANSWER_KINDS[topic]
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
@pytest.mark.parametrize("topic", list(SKILLS))
def test_key_accepted_and_distractors_rejected(topic, tier):
    sup.check_single_tier(SKILLS[topic], topic, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
@pytest.mark.parametrize("topic", [FORMS, SQ, UNK, TAN])
def test_typed_answers_are_unguessable(topic, tier):
    assert len(sup.answer_spread(topic, tier, 60)) >= 12, (topic, tier)


@pytest.mark.parametrize("topic", list(SKILLS))
def test_pools_and_shapes(topic):
    sup.check_pools(topic, min_prompts=20)


@pytest.mark.parametrize("topic", list(SKILLS))
def test_exam_chain_grades_and_distractors(topic):
    skill = SKILLS[topic]
    q = sup.exam(EXAMS[topic], 0)
    marks = [p.marks for p in q.parts]
    kinds = [{p.answer_kind} for p in q.parts]
    # the kind of a numeric part follows its value (number or fraction)
    kinds = [{"number", "fraction"} if k <= {"number", "fraction"} else k for k in kinds]
    sup.check_exam_chain_kinds(skill, EXAMS[topic], marks, kinds, seeds=40)


# -- circle forms ---------------------------------------------------------------------------------------------------


def _forms_expected(text: str):
    if "(x + a)" not in text and (m := re.search(r"((?:\(x|x\^2)[^=]*= \d+)\.", text)):
        h, k, rr = circle_of(m.group(1))
        return [h, k, sympy.sqrt(rr)]  # read: centre and radius
    if m := re.search(r"centre \((-?\d+), (-?\d+)\) and radius (\d+)\.", text):
        h, k, r = map(int, m.groups())
        return [-h, -k, r * r]
    if m := re.search(r"x\^2 \+ y\^2 = (\d+) is moved (\d+) (right|left) and (\d+) (up|down)", text):
        rr, dx, hd, dy, vd = m.groups()
        h = int(dx) * (1 if hd == "right" else -1)
        k = int(dy) * (1 if vd == "up" else -1)
        return [-h, -k, int(rr)]
    if m := re.search(r"centre \((-?\d+), (-?\d+)\) and passes through the point \((-?\d+), (-?\d+)\)", text):
        h, k, px, py = map(int, m.groups())
        return [-h, -k, (px - h) ** 2 + (py - k) ** 2]
    if m := re.search(r"A\((-?\d+), (-?\d+)\) and B\((-?\d+), (-?\d+)\)\. Its equation", text):
        x1, y1, x2, y2 = map(int, m.groups())
        h, k = Fraction(x1 + x2, 2), Fraction(y1 + y2, 2)
        return [-h, -k, (x1 - h) ** 2 + (y1 - k) ** 2]
    if m := re.search(r"O\(0, 0\), A\((-?\d+), 0\) and B\(0, (-?\d+)\)", text):
        a, b = map(int, m.groups())
        h, k, rr = sympy.symbols("h k rr")
        sol = sympy.solve(
            [h**2 + k**2 - rr, (a - h) ** 2 + k**2 - rr, h**2 + (b - k) ** 2 - rr], [h, k, rr], dict=True
        )[0]
        return [-sol[h], -sol[k], sol[rr]]
    if m := re.search(r"radius (\d+) lies in the (\w+) quadrant and touches both axes", text):
        r = int(m.group(1))
        sx, sy = {"first": (1, 1), "second": (-1, 1), "third": (-1, -1), "fourth": (1, -1)}[m.group(2)]
        return [-sx * r, -sy * r, r * r]
    if m := re.search(r"centre \((-?\d+), (-?\d+)\) and touches the (x|y)-axis", text):
        h, k = int(m.group(1)), int(m.group(2))
        r = abs(k) if m.group(3) == "x" else abs(h)
        return [-h, -k, r * r]
    raise AssertionError(text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_forms_keys_by_independent_derivation(tier):
    seen = set()
    for seed in range(120):
        q = sup.single(FORMS, tier, seed)
        _check_key(q, _forms_expected(q.text))
        seen.add(re.sub(r"-?\d+", "N", q.text)[:40])
    assert len(seen) >= (2 if tier == "easy" else 3)


def test_forms_hard_covers_every_shape_and_medium_every_shape():
    hard = " ".join(sup.single(FORMS, "hard", s).text for s in range(120))
    assert "passes through O(0, 0)" in hard and "touches both axes" in hard and "touches the" in hard
    medium = " ".join(sup.single(FORMS, "medium", s).text for s in range(120))
    assert "is moved" in medium and "passes through the point" in medium and "diameter" in medium


def test_forms_exam_chain_by_independent_derivation():
    for seed in range(60):
        q = sup.exam(FORMS_X, seed)
        x1, y1, x2, y2 = map(
            int, re.findall(r"-?\d+", q.text.split(" (a)")[0])
        )  # the only numbers in the stem are the endpoints
        a, b, c, d = q.parts
        h, k = Fraction(x1 + x2, 2), Fraction(y1 + y2, 2)
        r2 = (x1 - h) ** 2 + (y1 - k) ** 2
        assert [Fraction(v) for v in a.answer] == [h, k]
        assert Fraction(b.answer_text) ** 2 == r2
        assert [Fraction(v) for v in c.answer] == [-h, -k, r2]
        centre_to_origin = sympy.sqrt(h * h + k * k)
        assert Fraction(d.answer_text) == abs(Fraction(str(centre_to_origin)) - Fraction(b.answer_text))


# -- complete the square --------------------------------------------------------------------------------------------


def _sq_expected(q):
    text = q.text
    if m := re.search(r"circle ([^=]*= \d+) is written as x\^2 \+ y\^2 \+ Dx \+ Ey \+ F = 0", text):
        left, right = m.group(1).split("=")
        poly = sympy.expand(_expr(left) - _expr(right))
        return [poly.coeff(X, 1), poly.coeff(Y, 1), poly.subs({X: 0, Y: 0})]
    m = re.search(r"equation ([^.]*? = -?\d+|[^.]*? = 0)\.", text) or re.search(
        r"equation (x\^2 \+ y\^2 = [^.]*)\.", text
    )
    if m is None:
        m = re.search(r"equation ([^.]*)\. (?:Complete|Enter)", text)
    h, k, rr = circle_of(m.group(1))
    return [h, k, rr if "radius squared" in text else sympy.sqrt(rr)]


@pytest.mark.parametrize("tier", sup.TIERS)
def test_complete_square_keys_by_independent_derivation(tier):
    kinds = set()
    for seed in range(100):
        q = sup.single(SQ, tier, seed)
        _check_key(q, _sq_expected(q))
        kinds.add("expand" if "Dx + Ey + F" in q.text else "radius squared" if "radius squared" in q.text else "radius")
    assert (
        kinds
        == {"easy": {"radius"}, "medium": {"radius squared", "expand", "radius"}, "hard": {"radius squared"}}[tier]
    )


def test_complete_square_exam_chain_by_independent_derivation():
    for seed in range(60):
        q = sup.exam(SQ_X, seed)
        h, k, rr = circle_of(re.search(r"equation ([^.]*? = 0)\.", q.text).group(1))
        a, b, c, d = q.parts
        assert [Fraction(v) for v in a.answer] == [h, k]
        assert Fraction(b.answer_text) ** 2 == rr
        r = sympy.sqrt(rr)
        assert sorted(Fraction(v) for v in c.answer) == sorted([Fraction(str(k - r)), Fraction(str(k + r))])
        chord = 2 * sympy.sqrt(rr - k * k)
        assert Fraction(d.answer_text) == Fraction(str(chord))


# -- point position -------------------------------------------------------------------------------------------------


def _side(rr, h, k, point):
    d2 = (point[0] - h) ** 2 + (point[1] - k) ** 2
    return "inside" if d2 < rr else "on" if d2 == rr else "outside"


def _points(options):
    return [tuple(int(v) for v in re.findall(r"-?\d+", o)) for o in options]


@pytest.mark.parametrize("tier", sup.TIERS)
def test_point_position_exactly_one_option_has_the_property(tier):
    for seed in range(100):
        q = sup.single(POS, tier, seed)
        side = re.search(r"lies (inside|on|outside) the circle", q.text).group(1)
        h, k, rr = circle_of(re.search(r"circle (.+?= -?\d+)[?.]", q.text).group(1))
        pts = _points([q.answer_text, *q.distractors])
        assert len(set(pts)) == len(pts) == 5
        assert [_side(rr, h, k, p) == side for p in pts].count(True) == 1
        assert _side(rr, h, k, pts[0]) == side
        assert len({q.text for q in (sup.single(POS, tier, s) for s in range(60))}) >= 20


def test_point_position_hard_uses_the_expanded_form_and_all_three_sides():
    sides, expanded = set(), 0
    for seed in range(120):
        q = sup.single(POS, "hard", seed)
        sides.add(re.search(r"lies (inside|on|outside)", q.text).group(1))
        expanded += "Dx" in q.text or "x^2 + y^2 +" in q.text or "x^2 + y^2 -" in q.text
    assert sides == {"inside", "on", "outside"} and expanded == 120


def test_point_position_exam_chain_by_independent_derivation():
    for seed in range(60):
        q = sup.exam(POS_X, seed)
        h, k, rr = circle_of(re.search(r"equation ([^.]*?)\. The point", q.text).group(1))
        px, py = map(int, re.search(r"P is \((-?\d+), (-?\d+)\)", q.text).groups())
        a, b, c, d = q.parts
        dist = sympy.sqrt((px - h) ** 2 + (py - k) ** 2)
        r = sympy.sqrt(rr)
        assert Fraction(a.answer_text) == Fraction(str(dist))
        side = re.search(r"lies (inside|outside)", b.prompt).group(1)
        assert _side(rr, h, k, _points([b.answer_text])[0]) == side
        assert all(_side(rr, h, k, p) != side for p in _points(b.distractors))
        assert Fraction(c.answer_text) == abs(Fraction(str(dist - r)))
        assert Fraction(d.answer_text) == Fraction(str(dist + r))


# -- unknown constants ----------------------------------------------------------------------------------------------


def _unk_expected(q):
    text = q.text
    if m := re.search(
        r"The point \(([^)]*)\) lies on the circle ([^.]*?) and is (above|below|to the right of|to the left of)", text
    ):
        px, py = (_expr(v) for v in m.group(1).split(", "))
        h, k, rr = circle_of(m.group(2))
        eq = (px - h) ** 2 + (py - k) ** 2 - rr
        sols = sympy.solve(eq, T)
        word = m.group(3)
        centre = k if "t" in str(py) else h
        keep = [s for s in sols if (s > centre) == (word in ("above", "to the right of"))]
        assert len(keep) == 1
        return keep[0]
    if m := re.search(r"equation ([^=]*F) = 0 describes a circle of radius (\d+)", text):
        h, k, rr = circle_of(m.group(1).replace("+ F", "+ 0") + " = 0")
        base = h * h + k * k
        return base - int(m.group(2)) ** 2
    if m := re.search(r"The circle ([^=]*F) = 0 passes through the point \((-?\d+), (-?\d+)\)", text):
        px, py = int(m.group(2)), int(m.group(3))
        expr = _expr(m.group(1)).subs({X: px, Y: py})
        return sympy.solve(expr, F_)[0]
    if m := re.search(r"equation ([^=]*F) = 0 describes a circle\. Find the greatest whole", text):
        h, k, _ = circle_of(m.group(1).replace("+ F", "+ 0") + " = 0")
        return sympy.ceiling(h * h + k * k) - 1
    if m := re.search(r"The circle ([^=]*F) = 0 touches the (x|y)-axis", text):
        h, k, _ = circle_of(m.group(1).replace("+ F", "+ 0") + " = 0")
        f = F_
        r2 = h * h + k * k - f
        target = k * k if m.group(2) == "x" else h * h
        return sympy.solve(sympy.Eq(r2, target), f)[0]
    raise AssertionError(text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_unknown_keys_by_independent_derivation(tier):
    for seed in range(100):
        q = sup.single(UNK, tier, seed)
        assert Fraction(q.answer_text) == Fraction(str(_unk_expected(q))), (q.text, q.answer_text)


def test_unknown_exam_chain_by_independent_derivation():
    for seed in range(60):
        q = sup.exam(UNK_X, seed)
        h, k, _ = circle_of(
            re.search(r"equation ([^.]*?F) = 0 describes", q.text).group(1).replace("+ F", "+ 0") + " = 0"
        )
        a, b, c, d = q.parts
        assert Fraction(a.answer_text) == Fraction(str(h))
        assert Fraction(b.answer_text) == Fraction(str(h * h + k * k - 1))
        r = int(re.search(r"radius (\d+)", c.prompt).group(1))
        assert Fraction(c.answer_text) == Fraction(str(h * h + k * k - r * r))
        assert Fraction(d.answer_text) == Fraction(str(h * h))


# -- tangent --------------------------------------------------------------------------------------------------------


def _tan_setup(text: str):
    px, py = map(int, re.search(r"P\((-?\d+), (-?\d+)\)", text).groups())
    h, k, rr = circle_of(re.search(r"circle ([^.]*?\d+)\.", text).group(1))
    assert (px - h) ** 2 + (py - k) ** 2 == rr
    return px, py, h, k, rr


def test_tangent_keys_by_independent_derivation():
    for tier in sup.TIERS:
        for seed in range(100):
            q = sup.single(TAN, tier, seed)
            px, py, h, k, rr = _tan_setup(q.text)
            # the tangent is the line whose normal is the radius: (px - h)(x - h) + (py - k)(y - k) = rr
            line = sympy.Eq((px - h) * (X - h) + (py - k) * (Y - k), rr)
            slope = sympy.solve(line, Y)[0].coeff(X, 1)
            y_int = sympy.solve(line.subs(X, 0), Y)[0]
            x_int = sympy.solve(line.subs(Y, 0), X)[0]
            if "gradient of the radius" in q.text:
                want = Fraction(py - k, px - h)
            elif "gradient of the tangent" in q.text:
                want = Fraction(str(slope))
            elif "y-coordinate of that point" in q.text:
                want = Fraction(str(y_int))
            else:
                assert "x-coordinate of that point" in q.text, q.text
                want = Fraction(str(x_int))
            assert Fraction(q.answer_text) == want, (tier, seed, q.text, q.answer_text)


def test_tangent_exam_chain_by_independent_derivation_and_follow_through():
    for seed in range(60):
        q = sup.exam(TAN_X, seed)
        px, py, h, k, rr = _tan_setup(q.text)
        a, b, c, d = q.parts
        line = sympy.Eq((px - h) * (X - h) + (py - k) * (Y - k), rr)
        assert Fraction(a.answer_text) == Fraction(py - k, px - h)
        assert Fraction(b.answer_text) == Fraction(str(sympy.solve(line, Y)[0].coeff(X, 1)))
        assert Fraction(c.answer_text) == Fraction(str(sympy.solve(line.subs(X, 0), Y)[0]))
        assert Fraction(d.answer_text) == Fraction(str(sympy.solve(line.subs(Y, 0), X)[0]))
    # a wrong radius gradient carries through the later parts
    done = 0
    for seed in range(40):
        q = sup.exam(TAN_X, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        px, py, h, k, rr = _tan_setup(q.text)
        ga = Fraction(canon["a"]) + 1
        if ga in (0, -1):
            continue
        gb = -1 / ga
        c = py - gb * px
        d = px - py / gb if gb else None
        reply = {**canon, "a": str(ga), "b": str(gb), "c": str(c), "d": str(d)}
        result = grade(SKILLS[TAN], q, reply)
        for label in "bcd":
            assert result.parts[label].marks == next(p for p in q.parts if p.label == label).marks, (seed, label)
        done += 1
    assert done >= 25
