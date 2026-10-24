"""Tests the graph-sqrt-of-function-domain-range skill: keys re-derived from the drawn graph or the rule, grading, exam."""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from mathgen.grading.inequality import read_solution_set
from tests import sr2b_support as sup

SKILL = "graph-sqrt-of-function-domain-range"
TOPIC = "gt_sqrt_function"
EXAM = "gt_sqrt_function_exam"
X, Y = sp.symbols("x y")
_T = standard_transformations + (implicit_multiplication_application,)


def _rule(stem: str):
    m = re.search(r"f\(x\) = ([^.?]*?)(?:\. |\.$|, )", stem + " ")
    return parse_expr(m.group(1).replace("^", "**"), local_dict={"x": X}, transformations=_T)


def _set(text: str):
    got = read_solution_set(text)
    assert got is not None, text
    return got[0]


def _has(s: sp.Set, v: float) -> bool:
    return bool(s.contains(sp.Float(v)))


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "inequality", ["function-notation", "domain-range-standard-functions"])


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_drawn_keys_are_where_the_graph_is_not_below_the_axis(tier):
    for seed in range(80):
        q = sup.tier_question(TOPIC, tier, seed)
        curve = next(c for c in q.figure["curves"] if c["label"] == "f")
        pts = np.array([p for piece in curve["pieces"] for p in piece])
        key = _set(q.answer_text)
        checked = 0
        for x, y in pts[::7]:
            if abs(y) > 0.1:
                assert _has(key, float(x)) == (y > 0), (seed, q.text, q.answer_text, x, y)
                checked += 1
        assert checked >= 5


def test_medium_covers_every_shape_of_domain():
    texts = {sup.tier_question(TOPIC, "medium", s).answer_text for s in range(200)}
    assert "all real numbers" in texts and "no solution" in texts
    assert any(" or " in t for t in texts) and any(re.match(r"-?\d+ ≤ x ≤ -?\d+", t) for t in texts)


def test_hard_keys_match_an_independent_derivation():
    seen = set()
    for seed in range(120):
        q = sup.tier_question(TOPIC, "hard", seed)
        f = _rule(q.text)
        key = _set(q.answer_text)
        if "range" in q.text:
            seen.add("range")
            roots = sp.solve(sp.diff(f, X), X)
            (r,) = roots
            extreme = f.subs(X, r)
            lead = sp.Poly(f, X).LC()
            values = sp.Interval(extreme, sp.oo) if lead > 0 else sp.Interval(-sp.oo, extreme)
            rng = sp.Intersection(values, sp.Interval(0, sp.oo))
            lo, hi = sp.sqrt(rng.inf), (sp.sqrt(rng.sup) if rng.sup != sp.oo else sp.oo)
            want = sp.Interval(lo, hi)
            assert key == want, (seed, q.text, q.answer_text, want)
        else:
            seen.add("domain")
            want = sp.solve_univariate_inequality(f >= 0, X, relational=False)
            assert key == want, (seed, q.text, q.answer_text, want)
    assert seen == {"range", "domain"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8, top=0.3)


def test_equivalent_spellings_are_accepted():
    sk = sup.skill_object(SKILL)
    q = next(q for q in (sup.tier_question(TOPIC, "easy", s) for s in range(40)) if "≤" in q.answer_text)
    assert grade(sk, q, q.answer_text.replace("≤", "<=")).correct
    assert (
        grade(sk, q, "{x : " + q.answer_text + "}").correct or grade(sk, q, q.answer_text.replace("x ", "x  ")).correct
    )


def test_strict_and_closed_end_points_differ():
    sk = sup.skill_object(SKILL)
    q = sup.tier_question(TOPIC, "easy", 1)
    assert not grade(sk, q, q.answer_text.replace("≤", "<").replace("≥", ">")).correct


def test_tiers_differ():
    assert all(q.figure is not None for q in (sup.tier_question(TOPIC, "easy", s) for s in range(20)))
    assert all(q.figure is not None for q in (sup.tier_question(TOPIC, "medium", s) for s in range(20)))
    assert all(q.figure is None for q in (sup.tier_question(TOPIC, "hard", s) for s in range(20)))
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=2)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        f = _rule(q.text)
        a, b, c, d = q.parts
        (r,) = sp.solve(sp.diff(f, X), X)
        top = f.subs(X, r)
        assert sp.Rational(a.answer_text) == top, (seed, q.text)
        assert sp.Rational(b.answer_text) == sp.sqrt(top)
        assert _set(c.answer_text) == sp.solve_univariate_inequality(f >= 0, X, relational=False), (seed, q.text)
        got = parse_expr(d.answer_text.replace("^", "**"), transformations=_T)
        assert sp.simplify(got - sp.sqrt(f.subs(X, 0))) == 0, (seed, q.text, d.answer_text)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["number", "number", "inequality", "expression"], [2, 1, 2, 1], [False, True, False, False]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.1)


def test_exam_part_b_follows_through_from_a():
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "a": "49", "b": "7"}
        assert sup.part_marks(SKILL, q, replies, "b") == q.parts[1].marks
        assert sup.part_marks(SKILL, q, {**replies, "b": "49"}, "b") == 0


def test_exam_half_distractor_is_only_offered_for_a_whole_half():
    odd = even_offered = 0
    for seed in range(300):
        q = sup.exam_question("gt_sqrt_function_exam", seed)
        part_b = q.parts[1]
        t = int(part_b.answer_text) ** 2
        notes = part_b.option_notes
        assert notes, seed  # the tags are on the part, so the checks below can see them
        halves = [text for text, tag in notes.items() if tag == "gtf-root-as-half"]
        if t % 2:
            odd += 1
            assert not halves, (seed, halves)
        elif halves:
            even_offered += 1
            assert halves == [str(t // 2)], (seed, halves)
    assert odd >= 20 and even_offered >= 20, (odd, even_offered)


def test_skill_declares_every_misconception_the_generator_emits():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skill = next(x for x in skills if x.id == "graph-sqrt-of-function-domain-range")
    declared = {m.id for m in skill.misconceptions}
    emitted = set()
    for tier in ("easy", "medium", "hard"):
        for seed in range(150):
            emitted |= set(sup.tier_question("gt_sqrt_function", tier, seed).option_notes.values())
    for seed in range(150):
        for p in sup.exam_question("gt_sqrt_function_exam", seed).parts:
            emitted |= set(p.option_notes.values())
    assert {"gtf-root-as-half", "gtf-domain-wrong-side", "gtf-root-not-taken"} <= emitted, emitted
    assert emitted <= declared, emitted - declared
