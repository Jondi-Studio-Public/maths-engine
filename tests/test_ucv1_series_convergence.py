"""Tests for `ucv1-series-convergence`: the three single-answer tiers and the four-part exam chains.

The oracle reads only the question text: it parses the printed series back into SymPy and sums it, takes the limit
of its ratio, or solves for its radius by a route different from the generator's (the generator checks itself with
exact partial sums and `math.lgamma`). It imports nothing from the generator.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects
from scripts import skill_contract
from ucv1_calc_support import N, X, parse_math, value_of

SKILL_ID = "ucv1-series-convergence"
TOPIC = "ucv1_series_convergence"
EXAM_TOPIC = "exd_x_uc_series_convergence"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
K = sympy.Symbol("k", integer=True, positive=True)
_SERIES = re.compile(r"Σ_\(n=(?P<start>\d+)\)\^∞ (?P<term>.+?)(?: when x = (?P<x>-?\d+))?\.$")


def _harmonic(s: int) -> sympy.Rational:
    return sum((sympy.Rational(1, i) for i in range(1, s + 1)), sympy.Integer(0))


def telescoping_sum(term, var, start: int = 1) -> sympy.Rational:
    """The sum of a rational term with simple poles, from its partial fractions: -sum(A_j * H(s_j)) when sum(A_j) = 0."""
    total_coefficient = 0
    harmonic = 0
    for piece in sympy.Add.make_args(sympy.apart(term, var)):
        num, den = sympy.fraction(sympy.together(piece))
        poly = sympy.Poly(den, var)
        assert poly.degree() == 1, piece
        a, b = poly.all_coeffs()
        coefficient, shift = num / a, b / a
        total_coefficient += coefficient
        harmonic += coefficient * _harmonic(int(shift) + start - 1)
    assert total_coefficient == 0
    return -harmonic


def oracle(q):
    """The key, from the printed question by SymPy."""
    text = q.text
    if text.startswith("Find the radius of convergence"):
        return radius_oracle(text)
    m = re.search(r"a_n = (?P<term>.+), find L", text)
    if m:
        a_n = parse_math(m["term"])
        ratio = sympy.gammasimp(a_n.subs(N, N + 1) / a_n)
        return sympy.limit(ratio, N, sympy.oo)
    m = _SERIES.search(text)
    assert m, text
    start, term_text = int(m["start"]), m["term"]
    if m["x"] is not None:
        term = parse_math(term_text.replace("x", f"({m['x']})"))
    else:
        term = parse_math(term_text)
    if (
        "(n + " in term_text
        or term_text.startswith(("2/(n", "3/(n", "4/(n", "5/(n", "6/(n", "1/(n"))
        or re.search(r"/\(n\(", term_text)
    ):
        return telescoping_sum(term, N, start)
    return sympy.summation(term, (N, start, sympy.oo))


def radius_oracle(text):
    body = re.search(r"Σ_\(n=1\)\^∞ (?P<term>.+)\.$", text)["term"]
    centre = re.search(r"\(x - (\d+)\)", body)
    shift = int(centre[1]) if centre else 0
    term = parse_math(body)
    ratio = sympy.limit(sympy.gammasimp(term.subs(N, N + 1) / term), N, sympy.oo)
    t = sympy.Symbol("t", positive=True)
    roots = sympy.solve(sympy.Eq(sympy.simplify(ratio.subs(X, shift + t)), 1), t)
    (root,) = [r for r in roots if r.is_positive]
    return root


def _same(a, b) -> bool:
    return abs(float(sympy.N(a - b, 30))) < 1e-9


def test_source_wiring():
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM_TOPIC, "exam"),
    ]
    assert [s.stages for s in _SKILL.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert _SKILL.answer == "fraction"


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_sympy_derivation_of_the_printed_question(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert _same(oracle(q), value_of(q.answer_text)), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_every_distractor_rejected_and_diagnosed(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2
        for d in q.distractors:
            result = grade(_SKILL, q, d)
            assert not result.correct, (seed, tier, q.text, d)
            assert result.misconception == q.option_notes[d], (seed, d)


def test_equivalent_spellings_of_the_key_are_accepted():
    q = make(TOPIC, seed=3, difficulty="easy")
    value = Fraction(q.answer_text)
    for spelling in (f"{value.numerator}/{value.denominator}", f"{value.numerator * 2}/{value.denominator * 2}"):
        assert grade(_SKILL, q, spelling).correct, spelling


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [make(TOPIC, seed=s, difficulty=tier).answer_text for s in range(60)]
    counts = Counter(answers)
    assert len(counts) >= 12, counts
    assert max(counts.values()) <= 12, counts


@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_asks_for_distinct_questions_and_is_deterministic(tier):
    texts = {make(TOPIC, seed=s, difficulty=tier).text for s in range(60)}
    assert len(texts) >= 40
    assert make(TOPIC, seed=7, difficulty=tier).to_json() == make(TOPIC, seed=7, difficulty=tier).to_json()


def test_the_tiers_escalate_in_form():
    easy = " ".join(make(TOPIC, seed=s, difficulty="easy").text for s in range(60))
    medium = " ".join(make(TOPIC, seed=s, difficulty="medium").text for s in range(60))
    hard = " ".join(make(TOPIC, seed=s, difficulty="hard").text for s in range(60))
    assert "(n + " not in easy and "lim" not in easy and "x" not in easy.replace("exact", "")
    assert "lim(n→∞)" in medium and "(n + " in medium and "radius" not in medium and "when x" not in medium
    assert "radius of convergence" in hard and "when x" in hard and "(n + 2)(n + " in hard


@pytest.mark.parametrize("tier", TIERS)
def test_no_text_defects(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


# -- the exam chains -----------------------------------------------------------------------------------------------


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def is_telescoping(q) -> bool:
    return q.stem.startswith("Let a_k")


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _a_k(q):
    return parse_math(re.search(r"a_k = (?P<term>.+?) for k ≥ 1", q.stem)["term"], k=K)


def _direct_partial_sum(term, m: int) -> sympy.Rational:
    return sum((term.subs(K, i) for i in range(1, m + 1)), sympy.Integer(0))


def _tolerance(q) -> sympy.Rational:
    return sympy.Rational(re.search(r"S - S_n < (?P<eps>\S+)\.$", q.parts[3].prompt)["eps"])


def telescoping_exam_oracle(q):
    term = _a_k(q)
    pieces = sympy.Add.make_args(sympy.apart(term, K))
    first = min(pieces, key=lambda t: sympy.fraction(sympy.together(t))[1].subs(K, 0))  # the 1/(k + r) piece
    amount = sympy.fraction(sympy.together(first))[0] / sympy.Poly(sympy.fraction(sympy.together(first))[1], K).LC()
    total = telescoping_sum(term, K, 1)
    eps = _tolerance(q)
    n, partial = 0, sympy.Integer(0)
    while True:
        n += 1
        partial += term.subs(K, n)
        if total - partial < eps:
            break
        assert n < 5000
    return {"a": amount, "c": total, "d": sympy.Integer(n)}, term


def geometric_exam_oracle(q):
    m = re.search(r"S\(x\) = Σ_\(n=0\)\^∞ (?P<base>\(\(.+\)/\d+\))\^n for", q.stem)
    assert m, q.stem
    ratio = parse_math(m["base"])
    x0 = int(re.search(r"when x = (-?\d+)\.", q.parts[0].prompt)[1])
    target = sympy.Rational(re.search(r"S\(x\) = (\S+)\.$", q.parts[3].prompt)[1])
    r0 = ratio.subs(X, x0)
    total = sympy.summation(r0**N, (N, 0, sympy.oo))
    tail = sympy.summation(r0**N, (N, 1, sympy.oo))
    (x_value,) = sympy.solve(sympy.Eq(1 / (1 - ratio), target), X)
    return {"a": r0, "b": total, "c": tail, "d": x_value}


def exam_oracle(q):
    if is_telescoping(q):
        values, term = telescoping_exam_oracle(q)
        for m in range(1, 13):
            closed = value_of(q.parts[1].answer_text).subs(N, m)
            assert closed == _direct_partial_sum(term, m), (m, q.parts[1].answer_text)
        return values
    return geometric_exam_oracle(q)


def test_exam_source_wired_last():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages) == (EXAM_TOPIC, "exam", ["exam"])


def test_exam_shape_marks_and_follow_through():
    for seed in range(80):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert sum(p.marks for p in q.parts) == 10
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, False]
        assert find_text_defects(q, include_working=True) == []
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_exam_covers_both_chains():
    assert {is_telescoping(_exam(s)) for s in range(40)} == {True, False}


def test_exam_keys_match_independent_derivation():
    for seed in range(60):
        q = _exam(seed)
        expected = exam_oracle(q)
        for part in q.parts:
            if part.label in expected:
                assert _same(expected[part.label], value_of(part.answer_text)), (seed, part.label, q.text)


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(60):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 10
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)


def _reply(value) -> str:
    value = sympy.sympify(value)
    if value.is_Rational:
        return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"
    return sympy.sstr(value)


def test_exam_follow_through_from_a_wrong_part_a():
    """A learner whose part (a) is wrong but who carries it correctly into the later parts keeps those marks."""
    checked = {True: 0, False: 0}
    for seed in range(80):
        q = _exam(seed)
        tele = is_telescoping(q)
        floor = 0 if tele else -1
        wrong_text = next(
            (d for d in q.parts[0].distractors if sympy.Rational(d) > floor and abs(sympy.Rational(d)) < 1 + 10 * tele),
            None,
        )
        if wrong_text is None:
            continue
        wrong = sympy.Rational(wrong_text)
        expected = exam_oracle(q)
        if tele:
            scale = wrong / expected["a"]
            replies = {"b": scale * value_of(q.parts[1].answer_text), "c": scale * expected["c"]}
            ours = {"b", "c"}
        else:
            if wrong == 0 or abs(wrong) >= 1:
                continue
            total = sympy.summation(wrong**N, (N, 0, sympy.oo))
            replies = {"b": total, "c": total - 1}
            ours = {"b", "c"}
        if any(v.is_number and abs(v) < sympy.Rational(1, 100) for v in replies.values()):
            continue  # the engine withholds follow-through when a carried value rounds to zero
        canon = _canonical(q)
        result = grade(_SKILL, q, {**canon, "a": wrong_text, **{k: _reply(v) for k, v in replies.items()}})
        assert result.parts["a"].marks == 0
        for label in sorted(ours):
            part = result.parts[label]
            assert part.marks == q.parts[ord(label) - 97].marks and (part.follow_through or part.correct), (
                seed,
                label,
                part,
            )
        checked[tele] += 1
    assert checked[True] >= 10 and checked[False] >= 10, checked


def test_exam_least_n_has_no_follow_through_from_a_wrong_part():
    for seed in range(40):
        q = _exam(seed)
        if is_telescoping(q):
            assert q.parts[3].follow_through is None
            return
    raise AssertionError("no telescoping exam found")


def test_exam_answer_space_and_determinism():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(Counter(answers).values()) <= 15
    assert _exam(11).to_json() == _exam(11).to_json()


# A multi-part question has no question-level distractors (each part carries its own, checked above), so the
# shared `distractors` family, which reads `question.distractors`, cannot apply to the exam source.
@pytest.mark.parametrize("family", sorted(set(skill_contract.FAMILIES) - {"distractors"}))
def test_skill_contract_family(family):
    skill_contract.FAMILIES[family](SKILL_ID)
