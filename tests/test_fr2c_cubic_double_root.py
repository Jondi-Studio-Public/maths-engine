"""`f10-cubic-double-root-graph`: topic `fr2c_cubic_equation` (easy, medium, hard) and the exam chain `fr2c_cubic_features_exam`.

The oracle finds the zeros in the drawn samples (a crossing changes sign, a touch does not), rebuilds the equation, and for
the exam differentiates the cubic with SymPy.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy import Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import figures, make
from tests.fr2c_checks import check_follow_through, check_parts, check_single, skill

SKILL = "f10-cubic-double-root-graph"
TOPIC = "fr2c_cubic_equation"
EXAM = "fr2c_cubic_features_exam"
TIERS = ("easy", "medium", "hard")
X = Symbol("x")
_T = standard_transformations + (implicit_multiplication_application, convert_xor)


def _pts(q):
    return [p for piece in q.figure["curves"][0]["pieces"] for p in piece]


def _zeros(q):
    by_x = {round(x, 4): y for x, y in _pts(q)}
    cross, touch = [], []
    for x, y in by_x.items():
        if abs(y) < 1e-9 and float(x).is_integer():
            l, r = by_x.get(round(x - 0.125, 4)), by_x.get(round(x + 0.125, 4))
            (touch if l * r > 0 else cross).append(int(x))
    return cross, touch


def _expr(q):
    return parse_expr(q.answer_text[4:], transformations=_T)


def test_skill_wiring():
    sk = skill(SKILL)
    assert sk.answer == "expression" and sk.needs_figure is True and sk.capability_gates == ["figure_stimulus"]
    assert [(s.topic, s.difficulty) for s in sk.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_wrong_equation_rejected(tier):
    check_single(SKILL, TOPIC, tier, min_options=3)


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_valid_and_the_reader_agrees(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert not figures.validate(q.figure)
        assert figures.get_reader(TOPIC)(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("tier", TIERS)
def test_one_crossing_one_touch_and_the_key_squares_the_touch(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        cross, touch = _zeros(q)
        assert len(cross) == 1 and len(touch) == 1, (seed, cross, touch)
        e = _expr(q)
        assert sympy.degree(e, X) == 3
        assert sympy.roots(sympy.Poly(e, X)) == {cross[0]: 1, touch[0]: 2}


def test_k_by_tier():
    ks = {t: set() for t in TIERS}
    for tier in TIERS:
        for seed in range(100):
            q = make(TOPIC, seed=seed, difficulty=tier)
            ks[tier].add(sympy.Poly(_expr(q), X).LC())
    assert ks["easy"] == {1}
    assert ks["medium"] == {1, -1}
    assert len(ks["hard"]) >= 4 and not ks["hard"] & {1, -1}


def test_hard_marked_point_fixes_k():
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="hard")
        px, py = q.figure["points"][0]["at"]
        assert _expr(q).subs(X, px) == py


def test_end_behaviour_matches_the_sign_of_k():
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="medium")
        pts = sorted(_pts(q))
        right = pts[-1][1]
        assert (right > 0) == (sympy.Poly(_expr(q), X).LC() > 0) or abs(right) > 9


def test_exam_chain_shape():
    check_parts(
        SKILL, EXAM, marks=[2, 1, 2, 2], kinds=["mcq", "number", "number", "number"], follow=[False, False, False, True]
    )


def test_exam_answers_recomputed_with_sympy():
    for seed in range(80):
        q = make(EXAM, seed=seed, difficulty="exam")
        e = parse_expr(re.search(r"y = (.*?)\. \(a\)", q.text).group(1), transformations=_T)
        roots = sympy.roots(sympy.Poly(e, X))
        touch = next(r for r, m in roots.items() if m == 2)
        cross = next(r for r, m in roots.items() if m == 1)
        p = {x.label: x for x in q.parts}
        assert f"crosses the x-axis at x = {cross} and touches it at x = {touch}" in p["a"].answer_text
        assert int(p["b"].answer_text) == e.subs(X, 0)
        crit = [r for r in sympy.solve(sympy.diff(e, X), X) if r != touch]
        assert len(crit) == 1
        assert int(p["c"].answer_text) == crit[0]
        assert int(p["d"].answer_text) == e.subs(X, crit[0])
        assert p["d"].follow_through.startswith(f"{sympy.Poly(e, X).LC()}*(part_c")


def test_exam_follow_through_from_a_wrong_turning_point():
    def replace(q, parts):
        wrong = int(parts["c"].distractors[0])
        e = parse_expr(re.search(r"y = (.*?)\. \(a\)", q.text).group(1), transformations=_T)
        return str(wrong), str(e.subs(X, wrong))

    check_follow_through(SKILL, EXAM, "c", replace, "d", seeds=range(60))


def test_exam_covers_both_signs_and_gaps():
    seen_k, seen_d = set(), set()
    for seed in range(120):
        q = make(EXAM, seed=seed, difficulty="exam")
        e = parse_expr(re.search(r"y = (.*?)\. \(a\)", q.text).group(1), transformations=_T)
        seen_k.add(sympy.Poly(e, X).LC())
        roots = sympy.roots(sympy.Poly(e, X))
        seen_d.add(abs(list(roots)[0] - list(roots)[1]))
    assert seen_k == {1, -1, 2, -2} and seen_d == {3, 6}
