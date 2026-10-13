"""Tests for `ucv1-lhopital-rule`: the three single-answer tiers and the four-part exam chain.

The oracle reads only the question text: it parses the printed limit back into SymPy and takes `sympy.limit`,
a different route from the generator's own 80-digit numeric check. It imports nothing from the generator.
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
from ucv1_calc_support import X, parse_math, value_of

SKILL_ID = "ucv1-lhopital-rule"
TOPIC = "ucv1_lhopital"
EXAM_TOPIC = "exd_x_uc_lhopital"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_LIMIT = re.compile(r"lim\(x→(?P<point>[^)]+)\) (?P<body>.+)\.$")


def _point(text: str):
    text = text.replace("⁺", "").replace("⁻", "")
    return sympy.oo if text == "∞" else sympy.sympify(text)


def oracle(q):
    """The limit printed in the question, by SymPy."""
    m = _LIMIT.search(q.text)
    assert m, q.text
    return sympy.limit(parse_math(m["body"]), X, _point(m["point"]), "+")


def _same(a, b) -> bool:
    return sympy.simplify(a - b) == 0


def test_source_wiring():
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM_TOPIC, "exam"),
    ]
    assert [s.stages for s in _SKILL.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert _SKILL.answer == "expression"


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_sympy_limit_of_the_printed_question(tier):
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


def test_alternative_spellings_of_the_key_are_accepted():
    q = make(TOPIC, seed=5, difficulty="hard")  # a power of e
    value = value_of(q.answer_text)
    for spelling in (str(value).replace("**", "^"), sympy.sstr(value), f"exp({sympy.sstr(sympy.log(value))})"):
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
    assert "∞" not in easy and "cos" not in easy
    assert "x→∞" in medium and "cos" in medium
    assert "^(1/x" in hard or "^(x/" in hard or "^(2x)" in hard  # a 1^∞ form appears
    assert not any(t in medium for t in ("^(1/x", "x(e^("))


@pytest.mark.parametrize("tier", TIERS)
def test_no_text_defects(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


# -- the exam chain ------------------------------------------------------------------------------------------------


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _exam_givens(q):
    stem = q.stem
    m = re.search(r"f\(x\) = \((?P<g>.+) - qx\)/x(?P<n>[²³]?) for", stem)
    assert m, stem
    n = {"": 1, "²": 2, "³": 3}[m["n"]]
    g = parse_math(m["g"])
    return g, n


def exam_oracle(q) -> dict[str, sympy.Basic]:
    g, n = _exam_givens(q)
    q_sym = sympy.Symbol("q")
    numerator = g - q_sym * X
    # (a) the q for which the limit exists: the first-order coefficient of the numerator must vanish.
    first_order = sympy.series(g, X, 0, 2).removeO().coeff(X, 1)
    q_value = sympy.solve(sympy.Eq(first_order - q_sym, 0), q_sym)[0]
    f = (numerator / X**n).subs(q_sym, q_value)
    limit_b = sympy.limit(f, X, 0, "+")
    # (c) and (d) are read from their own prompts.
    prompt_c = q.parts[2].prompt
    body_c = re.search(r"evaluate lim\(x→0\) (.+)\.$", prompt_c)[1].replace("qx", "q*x")
    expr_c = parse_math(body_c, q=q_value)
    prompt_d = q.parts[3].prompt
    body_d = re.search(r"evaluate lim\(x→0\) (.+)\.$", prompt_d)[1]
    expr_d = parse_math(body_d)
    return {
        "a": q_value,
        "b": limit_b,
        "c": sympy.limit(expr_c, X, 0, "+"),
        "d": sympy.limit(expr_d, X, 0, "+"),
    }


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_source_wired_last():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages) == (EXAM_TOPIC, "exam", ["exam"])


def test_exam_shape_marks_and_follow_through():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 2, 3]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert find_text_defects(q, include_working=True) == []
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_exam_keys_match_independent_derivation():
    for seed in range(60):
        q = _exam(seed)
        expected = exam_oracle(q)
        for part in q.parts:
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


def test_exam_follow_through_from_a_wrong_part_b():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        r = int(re.search(r"/\((\d)x[²³]?\)", q.parts[2].prompt)[1])
        p_value = Fraction(q.parts[0].answer_text)
        wrong_b = Fraction(q.parts[1].distractors[0])
        if wrong_b == 0:
            continue
        canon = _canonical(q)
        c_reply = (
            str(wrong_b / r)
            if (wrong_b / r).denominator == 1
            else f"{(wrong_b / r).numerator}/{(wrong_b / r).denominator}"
        )
        d_value = sympy.exp(
            sympy.Rational(wrong_b.numerator, wrong_b.denominator)
            / sympy.Rational(p_value.numerator, p_value.denominator)
        )
        result = grade(_SKILL, q, {**canon, "b": q.parts[1].distractors[0], "c": c_reply, "d": sympy.sstr(d_value)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, result.parts["c"])
        assert result.parts["d"].marks == 3 and result.parts["d"].follow_through is True, (seed, result.parts["d"])
        checked += 1
    assert checked >= 50


def test_exam_a_wrong_part_a_does_not_pay_later_parts_unless_they_are_right():
    q = _exam(3)
    canon = _canonical(q)
    result = grade(_SKILL, q, {**canon, "a": q.parts[0].distractors[0]})
    assert result.parts["a"].marks == 0
    assert result.parts["b"].marks == q.parts[1].marks  # (b) does not depend on the learner's q


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
