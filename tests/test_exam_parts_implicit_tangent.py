"""The multi-part exam tier `exam_parts_implicit_tangent` of
senior-use-implicit-differentiation-to-determine-the-gradient-of: dy/dx, the gradient at P, the tangent, its
x-intercept.

The oracle below re-derives every answer from the question text alone (the curve and P are parsed out of the
stem, the answers worked out with SymPy's partial derivatives); it never imports the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from mathgen.skills import load_skills

TOPIC = "exam_parts_implicit_tangent"
SKILL_ID = "senior-use-implicit-differentiation-to-determine-the-gradient-of"
SEEDS = range(100)
FEWER = range(40)  # the follow-through and oracle-heavy tests: grading each reply costs tens of milliseconds
x, y = sympy.symbols("x y")


@pytest.fixture(scope="module")
def skill():
    _, skills = load_skills()
    return next(s for s in skills if s.id == SKILL_ID)


def q_for(seed):
    return make(TOPIC, seed=seed)


def parse_stem(text: str):
    """(F, x0, y0): F(x, y) = lhs - rhs from the stem's equation, and the point P."""
    stem = text.split("(a)")[0]
    eq = re.search(r"(?:equation|by) ([^=]+?) = (-?[\dxy]+)[.,]", stem)
    lhs_text, rhs_text = eq.group(1), eq.group(2)

    def to_expr(s):
        s = s.replace("²", "**2").replace("³", "**3").replace(" ", "")
        s = re.sub(r"(?<=[\dxy])(?=[xy])", "*", s)
        return sympy.sympify(s, locals={"x": x, "y": y})

    point = re.search(r"P\((-?\d+), (-?\d+)\)", stem)
    return to_expr(lhs_text) - to_expr(rhs_text), int(point.group(1)), int(point.group(2))


def oracle(text):
    F, x0, y0 = parse_stem(text)
    assert F.subs({x: x0, y: y0}) == 0
    dydx = -sympy.diff(F, x) / sympy.diff(F, y)
    m = sympy.Rational(dydx.subs({x: x0, y: y0}))
    tangent = m * x + (y0 - m * x0)
    return dydx, m, tangent, x0 - sympy.Rational(y0) / m, x0, y0


def reply(q, **overrides):
    """Canonical replies for every part, with `overrides` (label -> reply) replacing some."""
    out = {p.label: p.answer_text for p in q.parts}
    out.update(overrides)
    return out


class TestFullMarks:
    def test_composed_answer_text_earns_full_marks(self, skill):
        for seed in SEEDS:
            q = q_for(seed)
            result = grade(skill, q, q.answer_text)
            assert result.correct, (seed, q.answer_text)
            assert result.marks == result.max_marks == 6

    def test_shape(self):
        for seed in SEEDS:
            q = q_for(seed)
            assert [p.label for p in q.parts] == list("abcd")
            assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "expression", "fraction"]
            assert [p.marks for p in q.parts] == [2, 1, 2, 1]
            assert q.parts[0].follow_through is None
            assert q.parts[1].follow_through.startswith("Subs(part_a, (x, y), (")
            assert "part_b" in q.parts[2].follow_through and "part_b" in q.parts[3].follow_through
            assert "*" not in q.text and "**" not in q.text

    def test_variety(self):
        assert len({q_for(s).text for s in range(60)}) >= 40


class TestDistractors:
    def test_every_distractor_is_marked_wrong(self, skill):
        for seed in SEEDS:
            q = q_for(seed)
            for part in q.parts:
                assert part.distractors, (seed, part.label)
                for d in part.distractors:
                    result = grade(skill, q, reply(q, **{part.label: d}))
                    assert result.parts[part.label].marks == 0, (seed, part.label, d)
                    others = [p for p in q.parts if p.label != part.label]
                    # the other parts, answered canonically, still earn their marks
                    assert all(result.parts[p.label].marks == p.marks for p in others), (seed, part.label, d)


class TestOracle:
    def test_oracle_agrees_with_every_part(self, skill):
        for seed in SEEDS:
            q = q_for(seed)
            dydx, m, tangent, x_int, x0, y0 = oracle(q.text)
            a, b, c, d = q.parts
            a_expr = sympy.sympify(a.answer_text.replace("^", "**"), locals={"x": x, "y": y})
            assert sympy.simplify(a_expr - dydx) == 0, (seed, a.answer_text)
            assert Fraction(str(m)) == Fraction(b.answer_text), (seed, b.answer_text)
            c_expr = sympy.sympify(c.answer_text.replace("y =", "").replace("^", "**"), locals={"x": x})
            assert sympy.expand(c_expr - tangent) == 0, (seed, c.answer_text)
            assert Fraction(str(x_int)) == Fraction(d.answer_text), (seed, d.answer_text)
            assert m != 0 and y0 != 0

    def test_equivalent_forms_are_accepted(self, skill):
        for seed in range(30):
            q = q_for(seed)
            dydx, m, tangent, x_int, x0, y0 = oracle(q.text)
            other_a = str(sympy.together(dydx)).replace("**", "^")
            other_c = f"y - {y0} = ({m})*(x - {x0})"
            for label, text in (("a", other_a), ("c", "y = " + str(sympy.expand(tangent)).replace("**", "^"))):
                assert (
                    grade(skill, q, reply(q, **{label: text})).parts[label].marks == q.parts["abcd".index(label)].marks
                )
            # a numeric decimal is not an exact fraction for a fraction part, but "m" as a/b is
            assert grade(skill, q, reply(q, b=str(m))).parts["b"].marks == 1
            assert other_c  # the unexpanded point-slope form is not asked for, only built to be inspected


class TestFollowThrough:
    def test_wrong_a_carries_into_b_c_d(self, skill):
        checked = 0
        for seed in FEWER:
            q = q_for(seed)
            dydx, m, tangent, x_int, x0, y0 = oracle(q.text)
            wrong_a = q.parts[0].distractors[0]
            wa = sympy.sympify(wrong_a.replace("^", "**"), locals={"x": x, "y": y})
            wb = wa.subs({x: x0, y: y0})
            if not wb.is_Rational or wb == 0:
                continue
            wc = wb * (x - x0) + y0
            wd = x0 - sympy.Rational(y0) / wb
            replies = {"a": wrong_a, "b": str(wb), "c": "y = " + str(sympy.expand(wc)).replace("**", "^"), "d": str(wd)}
            result = grade(skill, q, replies)
            assert result.parts["a"].marks == 0
            for label in "bcd":
                part = result.parts[label]
                assert part.marks == q.parts["abcd".index(label)].marks, (seed, label, replies)
            assert result.parts["b"].follow_through is True
            # (c) and (d) name only part_b, which itself earned its marks by following through, so the core
            # flags them follow_through only when (b) is wrong (see test_wrong_b_carries_into_c_and_d).
            assert result.marks == 4 < result.max_marks
            assert not result.correct
            checked += 1
        assert checked >= 25

    def test_wrong_b_carries_into_c_and_d(self, skill):
        for seed in FEWER:
            q = q_for(seed)
            dydx, m, tangent, x_int, x0, y0 = oracle(q.text)
            wb = m + sympy.Rational(3, 7)
            if wb == 0:
                continue
            wc = wb * (x - x0) + y0
            wd = x0 - sympy.Rational(y0) / wb
            replies = reply(q, b=str(wb), c="y = " + str(sympy.expand(wc)).replace("**", "^"), d=str(wd))
            result = grade(skill, q, replies)
            assert result.parts["a"].marks == 2 and result.parts["b"].marks == 0
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True
            assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True
            assert result.marks == 5 < result.max_marks

    def test_wrong_c_does_not_carry_into_d(self, skill):
        for seed in FEWER:
            q = q_for(seed)
            dydx, m, tangent, x_int, x0, y0 = oracle(q.text)
            wrong_d = x_int + 1
            result = grade(skill, q, reply(q, c="y = 0", d=str(wrong_d)))
            assert result.parts["c"].marks == 0
            assert result.parts["d"].marks == 0


class TestNoFollowThroughTarget:
    def test_undefined_target_earns_nothing_and_does_not_crash(self, skill):
        for seed in range(30):
            q = q_for(seed)
            dydx, m, tangent, x_int, x0, y0 = oracle(q.text)
            # (a) infinite at P: (b)'s follow-through is undefined, so only the canonical reply earns (b).
            blow_up = f"1/(x - {x0})"
            result = grade(skill, q, reply(q, a=blow_up, b="5"))
            assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
            assert grade(skill, q, reply(q, a=blow_up)).parts["b"].marks == 1  # canonical (b) still earns
            # (b) zero: (d)'s follow-through divides by zero, so a non-canonical (d) earns nothing.
            result = grade(skill, q, reply(q, b="0", d="5"))
            assert result.parts["b"].marks == 0 and result.parts["d"].marks == 0
            # a missing, empty or unreadable (a) is no crash either
            for junk in ("", "banana", "(((", "1/0"):
                result = grade(skill, q, {"a": junk, "b": junk, "c": junk, "d": junk})
                assert result.marks == 0


class TestWrongEverywhere:
    def test_wrong_everywhere_earns_zero(self, skill):
        for seed in SEEDS:
            q = q_for(seed)
            result = grade(skill, q, {"a": "x", "b": "17/3", "c": "y = 5", "d": "11/3"})
            assert result.marks == 0, seed
            assert not result.correct
