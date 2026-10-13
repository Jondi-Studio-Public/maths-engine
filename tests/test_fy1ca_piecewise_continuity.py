"""Tests `ucv1-piecewise-continuity` (topics `fy1ca_piecewise_continuity` and `fy1ca_piecewise_continuity_exam`).

The oracle reads the pieces back out of the question TEXT and re-derives each key with SymPy: it solves the junction
equations for the unknown constants, or takes the limit of the quotient piece.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade
from scripts.measure_tiers import classify_skill
from tests.fy1ca_support import TIERS, check_exam, check_follow_through, check_single_tier, check_wiring, load_skill

TOPIC = "fy1ca_piecewise_continuity"
EXAM = "fy1ca_piecewise_continuity_exam"
SKILL = load_skill("ucv1-piecewise-continuity")
x, k, a, b = sympy.symbols("x k a b")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_LOCALS = {"x": x, "k": k, "a": a, "b": b, "e": sympy.E}
_PIECE = re.compile(r"[fghp]\(x\) = (.+?) for (-?\d+ ≤ x ≤ -?\d+|x [<≤>≥] -?\d+)")


def _expr(text: str):
    text = text.replace("²", "**2").replace("^", "**").replace("e**(", "exp(")
    return parse_expr(text, local_dict=_LOCALS, transformations=_TRANSFORMS)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def test_skill_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "fraction")
    assert "limit-removable" in SKILL.prerequisites


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single_tier(SKILL, TOPIC, tier)


def test_easy_key_makes_the_pieces_meet():
    for seed in range(150):
        q = _q("easy", seed)
        pieces = _PIECE.findall(q.text)
        assert len(pieces) == 2, q.text
        c = int(re.findall(r"x = (-?\d+)", q.text)[-1])
        values = [_expr(e).subs(x, c) for e, _ in pieces]
        solution = sympy.solve(sympy.Eq(*values), k)
        assert solution == [sympy.Rational(q.answer_text)], (seed, q.text, solution)


def test_medium_key_is_the_limit_or_the_matching_slope():
    seen = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        if "for x ≠" in q.text:
            quotient = re.search(r"\(x\) = (.+?) for x ≠ (-?\d+)", q.text)
            point = int(quotient.group(2))
            limit = sympy.limit(_expr(quotient.group(1)), x, point)
            assert limit == sympy.Rational(q.answer_text), (seed, q.text)
            seen["limit"] += 1
        else:
            left, right = _PIECE.findall(q.text)
            point = int(re.search(r"x [<≤>≥] (-?\d+)", left[1]).group(1))
            limit = sympy.limit(_expr(left[0]), x, point)
            solution = sympy.solve(sympy.Eq(_expr(right[0]).subs(x, point), limit), k)
            assert solution == [sympy.Rational(q.answer_text)], (seed, q.text)
            seen["match"] += 1
    assert seen["limit"] >= 60 and seen["match"] >= 15, seen


def test_hard_key_needs_both_junctions():
    for seed in range(150):
        q = _q("hard", seed)
        pieces = _PIECE.findall(q.text)
        assert len(pieces) == 3, q.text
        p = int(re.search(r"x < (-?\d+)", pieces[0][1]).group(1))
        qq = int(re.search(r"x > (-?\d+)", pieces[2][1]).group(1))
        f1, f2, f3 = (_expr(e) for e, _ in pieces)
        solution = sympy.solve([f1.subs(x, p) - f2.subs(x, p), f2.subs(x, qq) - f3.subs(x, qq)], [a, b], dict=True)
        assert len(solution) == 1 and solution[0][b] == sympy.Rational(q.answer_text), (seed, q.text)
        # the second constant really does depend on the first: with a unconstrained, b is not fixed by junction 2 alone
        assert a in sympy.solve(f2.subs(x, qq) - f3.subs(x, qq), b)[0].free_symbols


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 8, (tier, Counter(answers).most_common(4))


def test_exam_shape_and_marks():
    check_exam(SKILL, EXAM, ["a", "b", "c", "d"], [2, 2, 1, 1], ["fraction", "fraction", "mcq", "fraction"])


def test_exam_keys_match_independent_derivation():
    for seed in range(120):
        q = _exam(seed)
        stem = q.stem
        m = re.search(r"([fghp])\(x\) = (.+?) for x < (\d+), \1\(\3\) = (-?\d+), and \1\(x\) = (.+?) for x > \3", stem)
        left, right, p, value = _expr(m.group(2)), _expr(m.group(5)), int(m.group(3)), int(m.group(4))
        limit = right.subs(x, p)
        assert limit == sympy.Rational(q.parts[0].answer_text)
        slope = sympy.solve(sympy.Eq(left.subs(x, p), limit), a)[0]
        assert slope == sympy.Rational(q.parts[1].answer_text)
        assert value != limit  # a stated value that is not the limit: removable, never continuous
        assert q.parts[2].answer_text.startswith("removable")
        r = int(re.search(rf"evaluate \w\((-?\d+)\)", q.parts[3].prompt).group(1))
        assert r < p
        assert left.subs({a: slope, x: r}) == sympy.Rational(q.parts[3].answer_text)


def test_exam_follow_through_from_a_wrong_earlier_part():
    check_follow_through(SKILL, EXAM)


def test_exam_slope_follows_through_from_a_wrong_limit():
    credited = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        wrong_limit = int(q.parts[0].answer_text) + 3
        # the left limit is p*a + u, so a wrong limit L' gives a = (L' - u)/p; the grader works that out itself
        result = grade(SKILL, q, {**canon, "a": str(wrong_limit)})
        assert result.parts["a"].marks == 0
        slope_reply = sympy.Rational(q.parts[1].answer_text) + sympy.Rational(
            3, int(re.search(r"x < (\d+)", q.stem).group(1))
        )
        text = str(slope_reply.p) if slope_reply.q == 1 else f"{slope_reply.p}/{slope_reply.q}"
        follow = grade(SKILL, q, {**canon, "a": str(wrong_limit), "b": text})
        credited += follow.parts["b"].marks == 2 and follow.parts["b"].follow_through is True
    assert credited >= 50


def test_last_tier_adds_a_shape():
    assert classify_skill(SKILL)[0] == "escalates"
