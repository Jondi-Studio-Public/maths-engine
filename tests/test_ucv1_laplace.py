"""Tests `ucv1-laplace-transform`: the easy, medium and hard topic `laplace_transform` and the exam chain
`exd_x_uc_laplace`.

The oracle reads only the question text. It parses the function, the transform or the differential equation back
out of the stem and re-derives every key with SymPy's own `laplace_transform`, `inverse_laplace_transform`,
`dsolve` and `residue`, a route that shares nothing with the generator's table, its numerical integration or its
exponent-by-exponent substitution.
"""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.ucv1_laplace_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "ucv1-laplace-transform"
TOPIC = "laplace_transform"
EXAM_TOPIC = "exd_x_uc_laplace"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
t = sympy.Symbol("t", positive=True)
s = sympy.Symbol("s")
y = sympy.Function("y")

SEEDS = range(40)
TIERS = ("easy", "medium", "hard")


def _read(text: str, **local):
    """The learner's notation (e^(2t), t², 3sin(2t), 2s/(s^2 + 4)) as a SymPy expression."""
    text = text.replace("²", "**2").replace("³", "**3").replace("^", "**")
    text = re.sub(r"e\*\*\(([^()]*)\)", r"exp(\1)", text)
    text = re.sub(r"(?<![A-Za-z])e\*\*t\b", "exp(t)", text)
    return parse_expr(text, transformations=_TRANSFORMS, local_dict={"t": t, "s": s, **local})


def _agree(left, right, var, points) -> bool:
    return all(abs(complex(left.subs(var, p).evalf()) - complex(right.subs(var, p).evalf())) < 1e-8 for p in points)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _oracle_single(question) -> sympy.Expr:
    """The key, as a SymPy function of s (forward) or of t (inverse), from the stem alone."""
    text = question.text
    if "L⁻¹" in text or "inverse" in text or "whose Laplace transform" in text:
        match = re.search(r"F\(s\) = (.+?)(?:, find the function|\.$)", text)
        assert match, text
        F = _read(match.group(1))
        return sympy.inverse_laplace_transform(F, s, t), t
    f = _read(text.rsplit("f(t) = ", 1)[1].rstrip("."))
    return sympy.laplace_transform(f, t, s, noconds=True), s


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages, last.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert [src.difficulty for src in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_sympy_transform(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        expected, var = _oracle_single(q)
        given = _read(q.answer_text)
        points = (0.7, 1.9) if var is t else (3.3, 5.1)
        assert _agree(expected, given, var, points), (seed, q.text, q.answer_text, expected)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(30):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 2
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_wrong_answers(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert len(q.distractor_values) >= 2
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == len(q.distractors)


def test_a_differently_written_but_equal_answer_is_accepted():
    q = _q("medium", 2)  # an inverse with two simple poles
    expected, _ = _oracle_single(q)
    assert grade(_SKILL, q, q.answer_text.replace(" ", "")).correct
    forward = _q("easy", 1)
    combined = str(sympy.together(_read(forward.answer_text))).replace("**", "^")
    assert grade(_SKILL, forward, combined).correct


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 12
    assert max(answers.count(a) for a in set(answers)) <= 6


def test_hard_tier_is_never_a_simple_table_lookup():
    shapes = {"repeat": 0, "square": 0}
    for seed in range(60):
        q = _q("hard", seed)
        match = re.search(r"F\(s\) = (.+?)(?:, find the function|\.$)", q.text)
        denominator = match.group(1).split("/", 1)[1]
        # a repeated root, or a quadratic with no real root
        assert "²" in denominator or "s³" in denominator or re.search(r"s² [+-] \d+s [+-] \d+", denominator)
        if "e^" in q.answer_text and "sin" in q.answer_text:
            shapes["square"] += 1
        else:
            shapes["repeat"] += 1
    assert min(shapes.values()) >= 10


def test_medium_mixes_the_shift_theorem_and_partial_fractions():
    forward = inverse = 0
    for seed in range(60):
        q = _q("medium", seed)
        if q.text.startswith(("Find the inverse", "Given", "Find f(t)")):
            inverse += 1
        else:
            forward += 1
    assert forward >= 15 and inverse >= 15


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _parse_ode(question):
    """(ode as a SymPy equation, y(0), y'(0) or None) read from the stem."""
    match = re.search(r"((?:y''|y').*?|y' .*?) = (.+?), y\(0\) = (-?\d+)(?:, y'\(0\) = (-?\d+))?", question.stem)
    assert match, question.stem
    left_text, right_text, y0, v0 = match.groups()
    left = left_text.replace("y''", "D2").replace("y'", "D1")
    left = re.sub(r"(?<![A-Za-z])y(?![A-Za-z(])", "Y0", left)
    expr = _read(left, D2=y(t).diff(t, 2), D1=y(t).diff(t), Y0=y(t))
    return sympy.Eq(expr, _read(right_text)), int(y0), None if v0 is None else int(v0)


def _pole(factor: str) -> int:
    if factor == "s":
        return 0
    match = re.fullmatch(r"\(s ([+-]) (\d+)\)", factor)
    return -int(match[2]) if match[1] == "+" else int(match[2])


def _oracle_exam(question) -> dict:
    ode, y0, v0 = _parse_ode(question)
    ics = {y(0): y0} if v0 is None else {y(0): y0, y(t).diff(t).subs(t, 0): v0}
    solution = sympy.dsolve(ode, y(t), ics=ics).rhs
    Y = sympy.laplace_transform(solution, t, s, noconds=True)
    b_prompt = question.parts[1].prompt
    letter = re.search(r"([AB])\.$", b_prompt).group(1)
    poles = [_pole(f) for f in re.findall(r"[AB]/(\(s [+-] \d+\)|s)", b_prompt)]
    assert len(poles) == 2
    pole_value = poles[0 if letter == "A" else 1]
    coefficient = sympy.residue(Y, s, pole_value)
    t0 = sympy.Rational(re.search(r"Find y\(([\d.]+)\)", question.parts[3].prompt).group(1))
    return dict(Y=Y, y=solution, coef=coefficient, pole=pole_value, y0=y0, t0=t0, letter=letter, poles=poles)


def _facts(question) -> dict:
    """What the follow-through tests need from the stem and the prompts, without solving the equation again."""
    b_prompt = question.parts[1].prompt
    letter = re.search(r"([AB])\.$", b_prompt).group(1)
    poles = [_pole(f) for f in re.findall(r"[AB]/(\(s [+-] \d+\)|s)", b_prompt)]
    y0 = int(re.search(r"y\(0\) = (-?\d+)", question.stem).group(1))
    t0 = sympy.Rational(re.search(r"Find y\(([\d.]+)\)", question.parts[3].prompt).group(1))
    Y = _read(question.parts[0].answer_text)
    pole = poles[0 if letter == "A" else 1]
    return dict(Y=Y, coef=sympy.Rational(question.parts[1].answer_text), pole=pole, poles=poles, y0=y0, t0=t0)


def _half_up(value) -> str:
    return str(Decimal(str(float(value))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 2, 2, 1]
        assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "expression", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q) == []
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_exam_keys_match_independent_derivation():
    for seed in range(30):
        q = _exam(seed)
        o = _oracle_exam(q)
        a, b, c, d = q.parts
        assert _agree(o["Y"], _read(a.answer_text), s, (3.7, 6.1)), (seed, q.text, a.answer_text, o["Y"])
        assert sympy.nsimplify(o["coef"]) == sympy.nsimplify(sympy.Rational(b.answer_text)), (seed, b.answer_text)
        assert _agree(o["y"], _read(c.answer_text), t, (0.6, 1.7)), (seed, c.answer_text, o["y"])
        assert _half_up(o["y"].subs(t, o["t0"]).evalf(30)) == d.answer_text, (seed, d.answer_text)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(25):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 8
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)


def test_exam_expanded_denominator_and_exp_spelling_accepted():
    q = _exam(3)
    canon = _canonical(q)
    expanded = str(sympy.together(_read(canon["a"]))).replace("**", "^")
    assert grade(_SKILL, q, {**canon, "a": expanded}).parts["a"].marks == 3


def test_follow_through_from_a_wrong_Y():
    """A wrong Y(s) with the same poles: (b) is marked on the learner's own Y."""
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        o = _facts(q)
        canon = _canonical(q)
        for wrong in q.parts[0].distractors:
            wrong_Y = _read(wrong)
            poles_ok = sympy.simplify(sympy.denom(sympy.together(wrong_Y)) - sympy.denom(sympy.together(o["Y"]))) == 0
            if not poles_ok:
                continue
            coef = sympy.residue(wrong_Y, s, o["pole"])
            if coef in (0, o["coef"]) or coef.q > 12:
                continue
            reply = {**canon, "a": wrong, "b": str(coef)}
            result = grade(_SKILL, q, reply)
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (
                seed,
                wrong,
                coef,
                result,
            )
            checked += 1
    assert checked >= 15


def test_follow_through_from_a_wrong_coefficient_into_y_of_t():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        o = _facts(q)
        canon = _canonical(q)
        other_pole = [v for v in o["poles"] if v != o["pole"]][0]
        for wrong in q.parts[1].distractors:
            w = sympy.Rational(wrong)
            if w == 0:
                continue
            c_text = f"({w})e^({o['pole']}t) + ({o['y0']} - ({w}))e^({other_pole}t)"
            reply = {**canon, "b": wrong, "c": c_text}
            result = grade(_SKILL, q, reply)
            assert result.parts["b"].marks == 0
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 80


def test_follow_through_from_a_wrong_y_of_t_into_the_value():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        o = _facts(q)
        canon = _canonical(q)
        for wrong in q.parts[2].distractors:
            value = _read(wrong).subs(t, o["t0"]).evalf(30)
            if abs(value) < 0.01 or abs(value) > 1e4 or _half_up(value) == canon["d"]:
                continue
            if not _read(wrong).has(t):  # a constant: the engine gives no credit for just repeating it
                continue
            reply = {**canon, "c": wrong, "d": _half_up(value)}
            result = grade(_SKILL, q, reply)
            assert result.parts["c"].marks == 0
            assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 40


def test_a_correct_chain_typed_with_exp_or_expanded_forms_still_scores_fully():
    for seed in range(20):
        q = _exam(seed)
        canon = _canonical(q)
        assert grade(_SKILL, q, canon).marks == 8


def test_kinds_all_appear():
    stems = {("y''" in _exam(seed).stem, "e^" in _exam(seed).stem) for seed in range(60)}
    assert stems == {(True, False), (False, True), (False, False)}


def test_exam_answer_space():
    answers = [_exam(seed).answer_text for seed in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_exam_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()


def test_value_to_three_places_is_not_a_coin_flip():
    for seed in range(60):
        q = _exam(seed)
        o = _oracle_exam(q) if seed < 10 else None
        value = float(q.parts[3].answer_text)
        assert not math.isnan(value)
        if o:
            exact = float(o["y"].subs(t, o["t0"]))
            scaled = abs(exact) * 1000
            assert abs(scaled - math.floor(scaled) - 0.5) > 0.05
