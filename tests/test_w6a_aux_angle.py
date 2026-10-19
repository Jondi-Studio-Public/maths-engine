"""Tests `auxiliary-angle-max-min-solve` (topic `w6a_aux_angle`) and its exam chain `exd_x_w6a_aux_angle`.

The oracle parses the function out of the question TEXT, reads its exact cos and sin coefficients by evaluating it at
0, π/2, π and 3π/2, and re-derives R, α, the extremes and the solutions with SymPy (not the generator's helpers).
"""

from __future__ import annotations

import math
import re

import pytest
import sympy
from sympy import pi, sqrt
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.w6a_aux_angle_build  # noqa: F401  (registers the topics)
from tests import w6a_support as sup

TOPIC = "w6a_aux_angle"
EXAM = "exd_x_w6a_aux_angle"
SKILL = sup.skill_of("auxiliary-angle-max-min-solve")
X = sympy.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    text = text.replace("π", "pi").replace("·", "").replace("^", "**")
    text = re.sub(r"√(\d+)", r"sqrt(\1)", text)
    text = re.sub(r"(cos|sin) x", r"\1(x)", text)
    return parse_expr(text, local_dict={"x": X, "pi": pi}, transformations=_T)


def _coeffs(fn):
    """(A, B, d) with fn(x) = A cos x + B sin x + d, by evaluating at four points."""
    f = lambda v: sympy.simplify(fn.subs(X, v))  # noqa: E731
    A = sympy.simplify((f(0) - f(pi)) / 2)
    B = sympy.simplify((f(pi / 2) - f(3 * pi / 2)) / 2)
    d = sympy.simplify((f(0) + f(pi)) / 2)
    assert sympy.simplify(f(pi / 3) - (A * sympy.cos(pi / 3) + B * sympy.sin(pi / 3) + d)) == 0
    return A, B, d


def _alpha(A, B):
    v = sympy.atan2(B, A)
    return sympy.simplify(v + 2 * pi if v < 0 else v)


def _function_in(text: str, marker: str):
    m = re.search(
        marker + r" ((?:[-+]?\s*[\d√()\s]*(?:cos x|sin x))[^.?]*?)(?= for all| takes| reaches|\.|\?|,|$)", text
    )
    return m


def _ans(q):
    return _parse(q.answer_text)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "expression")


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_easy_keys_are_the_exact_extremes():
    seen = set()
    for seed in range(120):
        q = sup.single(TOPIC, "easy", seed)
        m = re.search(r"(?:y|f\(x\)) = (.+?)(?: for all real x\.|\.| What| \?)", q.text)
        expr = _parse(m.group(1))
        A, B, d = _coeffs(expr)
        R = sqrt(A * A + B * B)
        want = d + R if re.search(r"maximum|greatest", q.text) else d - R
        assert sympy.simplify(_ans(q) - want) == 0, (seed, q.text, q.answer_text, want)
        seen.add("max" if re.search(r"maximum|greatest", q.text) else "min")
    assert seen == {"max", "min"}


def test_medium_keys_are_the_positions_of_the_extremes():
    for seed in range(120):
        q = sup.single(TOPIC, "medium", seed)
        m = re.search(r"f\(x\) = (.+?)(?: takes| reaches)", q.text) or re.search(
            r"is (.+?) (?:largest|smallest)\?", q.text
        )
        A, B, d = _coeffs(_parse(m.group(1)))
        alpha = _alpha(A, B)
        top = bool(re.search(r"greatest|largest", q.text))
        far = alpha + pi if alpha + pi < 2 * pi else alpha - pi
        assert sympy.simplify(_ans(q) - (alpha if top else far)) == 0, (seed, q.text, q.answer_text)
        # and the position is where the function really is extreme
        fn = A * sympy.cos(X) + B * sympy.sin(X)
        v = float(fn.subs(X, _ans(q)))
        R = float(sqrt(A * A + B * B))
        assert abs(v - (R if top else -R)) < 1e-9


def test_hard_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "hard", seed)
        if "no solution" in q.text or "cannot be solved" in q.text:
            m = re.search(r"(?:for which|equation|that) (.+?) = c", q.text)
            A, B, d = _coeffs(_parse(m.group(1)))
            R = float(sqrt(A * A + B * B))
            want = math.floor(R) + 1
            assert int(q.answer_text) == want, (seed, q.text)
            seen.add("threshold")
        else:
            m = re.search(r"(?:Solve|equation|root of the equation) (.+?) = (.+?)(?: for| has|\. | on|\.$)", q.text)
            A, B, d = _coeffs(_parse(m.group(1)))
            c = _parse(m.group(2))
            R = sqrt(A * A + B * B)
            alpha = _alpha(A, B)
            delta = sympy.acos(sympy.simplify(c / R))
            sols = []
            for v in (alpha + delta, alpha - delta):
                v = sympy.simplify(v)
                while v < 0:
                    v += 2 * pi
                while v >= 2 * pi:
                    v -= 2 * pi
                sols.append(v)
            sols = sorted(sols, key=float)
            larger = bool(re.search(r"larger", q.text))
            want = sols[1] if larger else sols[0]
            assert sympy.simplify(_ans(q) - want) == 0, (seed, q.text, q.answer_text, sols)
            seen.add("solve")
    assert seen == {"threshold", "solve"}


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 2, 2, 1], ["expression"] * 4)


def test_exam_keys_by_independent_derivation():
    for seed in range(120):
        q = sup.exam(EXAM, seed)
        m = re.search(r"f\(x\) = (.+?), for", q.stem)
        A, B, d = _coeffs(_parse(m.group(1)))
        R = sqrt(A * A + B * B)
        alpha = _alpha(A, B)
        a, b, c_part, d_part = q.parts
        assert sympy.simplify(_ans(a) - R) == 0
        assert sympy.simplify(_ans(b) - alpha) == 0
        c = _parse(re.search(r"f\(x\) = (.+?), and", c_part.prompt).group(1))
        delta = sympy.acos(sympy.simplify(c / R))
        assert sympy.simplify(_ans(c_part) - (alpha + delta)) == 0
        shift = int(re.search(r"g\(x\) = (\d+) \+ f", d_part.prompt).group(1))
        assert sympy.simplify(_ans(d_part) - (shift + R)) == 0
        assert float(_ans(c_part)) < 2 * math.pi
        fn = A * sympy.cos(X) + B * sympy.sin(X)
        assert abs(float(fn.subs(X, _ans(c_part)) - c)) < 1e-9


def test_exam_follow_through():
    def wrong(q, canon):
        # a wrong alpha earns the later solution marks; a wrong R earns the later maximum marks
        return None

    checked = 0
    for seed in range(60):
        q = sup.exam(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        alpha = _ans(q.parts[1])
        bad_alpha = sympy.simplify(alpha + pi / 12)
        delta = _parse(q.parts[2].follow_through.split("+", 1)[1])
        reply = sympy.simplify(bad_alpha + delta)
        if float(reply) >= 2 * math.pi or str(reply) == canon["c"]:
            continue
        res = sup.grade(SKILL, q, {**canon, "b": str(bad_alpha).replace("pi", "π"), "c": str(reply).replace("pi", "π")})
        assert res.parts["c"].marks == q.parts[2].marks and res.parts["c"].follow_through is True, (
            seed,
            res.parts["c"],
        )
        checked += 1
        shift = int(re.search(r"g\(x\) = (\d+) \+ f", q.parts[3].prompt).group(1))
        wrong_r = sympy.simplify(_ans(q.parts[0]) + 1)
        res = sup.grade(SKILL, q, {**canon, "a": str(wrong_r), "d": str(sympy.simplify(shift + wrong_r))})
        assert res.parts["d"].marks == q.parts[3].marks and res.parts["d"].follow_through is True, (
            seed,
            res.parts["d"],
        )
    assert checked >= 30


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier)
        assert len(spread) >= 8, (tier, spread)
        assert max(spread.values()) <= 20, (tier, spread.most_common(3))
