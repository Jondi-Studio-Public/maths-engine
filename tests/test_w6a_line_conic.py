"""Tests `line-meets-parabola-circle` (topic `w6a_line_conic`) and its exam chain `exd_x_w6a_line_conic`.

The oracle reads the parabola, circle and line back out of the question TEXT, parses them with SymPy and re-derives the
key by solving the equations, never through the generator's helpers.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.w6a_line_conic_build  # noqa: F401  (registers the topics)
from tests import w6a_support as sup

TOPIC = "w6a_line_conic"
EXAM = "exd_x_w6a_line_conic"
SKILL = sup.skill_of("line-meets-parabola-circle")
X, Y, K, C = sympy.symbols("x y k C")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_LINE_RE = re.compile(r"(\d*)y = (-?[0-9xkC²]+(?: [+-] [0-9xkC²]+)*)")
_CIRCLE_RE = re.compile(r"x² \+ y² = (\d+)")


def _expr(text: str):
    return parse_expr(text.replace("²", "**2"), local_dict={"x": X, "k": K, "C": C}, transformations=_TRANSFORMS)


def _curves(text: str):
    """(parabola RHS or None, circle r² or None, [(q, RHS expression)] lines in the text)."""
    parabola, lines = None, []
    for q, rhs in _LINE_RE.findall(text):
        if "x²" in rhs:
            parabola = _expr(rhs)
        else:
            lines.append((int(q or 1), _expr(rhs)))
    circle = _CIRCLE_RE.search(text)
    return parabola, int(circle.group(1)) if circle else None, lines


def _xs_parabola(parabola, line):
    return sorted(sympy.solve(sympy.Eq(parabola, line), X), key=float)


def _xs_circle(r2, q, rhs):
    y = rhs / q
    return sorted(sympy.solve(sympy.Eq(X**2 + y**2, r2), X), key=float)


def _frac(value):
    return sympy.Rational(value)


def _answer(q):
    return sympy.nsimplify(q.answer_text)


def _count_two(poly_of_k, lo, hi):
    n = 0
    for k in range(lo, hi + 1):
        if sympy.discriminant(poly_of_k(k), X) > 0:
            n += 1
    return n


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_easy_keys_by_solving_the_text():
    seen = set()
    for seed in range(120):
        q = sup.single(TOPIC, "easy", seed)
        parabola, _circle, lines = _curves(q.text)
        (_q, line), = lines
        xs = _xs_parabola(parabola, line)
        assert len(xs) == 2 and all(v.is_integer for v in xs), (seed, q.text)
        if "larger" in q.text or "right-hand" in q.text:
            want = xs[1]
            seen.add("larger")
        elif "y-coordinate" in q.text:
            want = line.subs(X, xs[0])
            seen.add("y")
        else:
            want = sum(xs)
            seen.add("sum")
        assert _answer(q) == want, (seed, q.text)
    assert seen == {"larger", "y", "sum"}


def test_medium_keys_by_solving_the_text():
    seen = set()
    for seed in range(150):
        q = sup.single(TOPIC, "medium", seed)
        parabola, r2, lines = _curves(q.text)
        if "midpoint" in q.text:
            (_q, line), = lines
            xs = _xs_parabola(parabola, line)
            assert len(xs) == 2 and not any(v.is_rational for v in xs)
            assert _answer(q) == sympy.simplify(sum(xs) / 2), (seed, q.text)
            seen.add("midpoint")
        elif r2 is not None:
            (qq, rhs), = lines
            assert _answer(q) == _xs_circle(r2, qq, rhs)[1], (seed, q.text)
            seen.add("circle")
        else:
            (_q, line), = lines
            assert line.has(K)
            sols = sympy.solve(sympy.discriminant(sympy.expand(parabola - line), X), K)
            assert len(sols) == 1 and _answer(q) == sols[0], (seed, q.text)
            seen.add("tangent")
    assert seen == {"midpoint", "circle", "tangent"}


def test_hard_keys_by_solving_the_text():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "hard", seed)
        parabola, r2, lines = _curves(q.text)
        if "(0," in q.text:
            point = int(re.search(r"\(0, (-?\d+)\)", q.text).group(1))
            m = sympy.Symbol("m")
            if r2 is not None or "circle" in q.text or "x² + y²" in q.text:
                r2 = int(_CIRCLE_RE.search(q.text).group(1))
                disc = sympy.discriminant(sympy.expand(X**2 + (m * X + point) ** 2 - r2), X)
            else:
                disc = sympy.discriminant(sympy.expand(parabola - (m * X + point)), X)
            assert _answer(q) == max(sympy.solve(disc, m)), (seed, q.text)
            seen.add("tangents")
        elif "whole" in q.text or "integers" in q.text:
            lo, hi = (int(v) for v in re.findall(r"-?\d+", re.search(r"(?:from|with) (-?\d+)(?: <= k <=|) (?:to )?(-?\d+)", q.text).group(0)))
            (_q, line), = lines
            m = line.coeff(X)
            n = 0
            for kv in range(lo, hi + 1):
                if r2 is not None:
                    poly = sympy.expand(X**2 + (m * X + kv) ** 2 - r2)
                else:
                    poly = sympy.expand(parabola - (m * X + kv))
                n += bool(sympy.discriminant(poly, X) > 0)
            assert _answer(q) == n, (seed, q.text)
            seen.add("count")
        else:
            (qq, rhs), = lines
            poly = sympy.expand(X**2 + (rhs / qq) ** 2 - r2)
            sols = [s for s in sympy.solve(sympy.discriminant(poly, X), C) if s > 0]
            assert len(sols) == 1 and _answer(q) == sols[0], (seed, q.text)
            seen.add("circle tangent")
    assert seen == {"tangents", "count", "circle tangent"}


def test_exam_chain_shape():
    kinds = set()
    for seed in range(60):
        q = sup.exam(EXAM, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        kinds.add(tuple(p.answer_kind for p in q.parts))
        assert [p.follow_through is not None for p in q.parts] in ([False, False, False, True], [False, False, True, True])
        full = sup.grade(SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                assert sup.grade(SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0
    assert len(kinds) == 2


def test_exam_keys_by_solving_the_text():
    seen = set()
    for seed in range(120):
        q = sup.exam(EXAM, seed)
        parabola, r2, lines = _curves(q.stem)
        a, b, c, d = q.parts
        if r2 is None:
            (_q, line), = lines
            xs = _xs_parabola(parabola, line)
            assert _answer(a) == xs[1]
            m = line.coeff(X)
            sols = sympy.solve(sympy.discriminant(sympy.expand(parabola - (m * X + K)), X), K)
            assert _answer(b) == sols[0]
            contact = sympy.solve(sympy.expand(parabola - (m * X + sols[0])), X)
            assert len(contact) == 1 and _answer(c) == contact[0]
            assert _answer(d) == abs(line.subs(X, 0) - sols[0])
            seen.add("parabola")
        else:
            (qq, rhs), = lines
            assert _answer(a) == _xs_circle(r2, qq, rhs)[1]
            p = rhs.coeff(X)
            poly = sympy.expand(X**2 + ((p * X + C) / qq) ** 2 - r2)
            cc = [s for s in sympy.solve(sympy.discriminant(poly, X), C) if s > 0][0]
            assert _answer(b) == cc
            contact = sympy.solve(sympy.expand(poly.subs(C, cc)), X)
            assert len(contact) == 1 and _answer(c) == contact[0]
            assert _answer(d) == (p * contact[0] + cc) / qq
            seen.add("circle")
    assert seen == {"parabola", "circle"}


def test_exam_follow_through_from_a_wrong_tangent_intercept():
    # parabola chain: the gap uses the learner's own k; circle chain: x then y use the learner's own C.
    checked = 0
    for seed in range(60):
        q = sup.exam(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        _par, r2, lines = _curves(q.stem)
        if r2 is None:
            (_q, line), = lines
            c0 = line.subs(X, 0)
            k_wrong = _frac(canon["b"]) + 3
            if k_wrong == 0:
                continue
            reply = {"b": str(k_wrong), "d": str(abs(c0 - k_wrong))}
            if reply["d"] in q.parts[3].distractors or reply["d"] == reply["b"]:
                continue  # a reply that repeats an earlier reply is not given follow-through
            res = sup.grade(SKILL, q, {**canon, **reply})
            assert res.parts["d"].marks == q.parts[3].marks and res.parts["d"].follow_through is True, (seed, res.parts["d"])
        else:
            (qq, rhs), = lines
            p = rhs.coeff(X)
            den = p * p + qq * qq
            c_wrong = _frac(canon["b"]) + 5
            x_wrong = -p * c_wrong / den
            if str(x_wrong) in q.parts[2].distractors:
                continue
            res = sup.grade(SKILL, q, {**canon, "b": str(c_wrong), "c": str(x_wrong)})
            assert res.parts["c"].marks == q.parts[2].marks and res.parts["c"].follow_through is True, (seed, res.parts["c"])
        checked += 1
    assert checked >= 50


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier)
        assert len(spread) >= 12, (tier, spread)
        assert max(spread.values()) <= 12, (tier, spread.most_common(3))
