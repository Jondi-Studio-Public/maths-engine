"""Tests for `ucv1-improper-integral`: the three single-answer tiers and the four-part exam chain.

The oracle reads only the question text: it parses the printed integral (or the printed density function, or the
curve and its area) back into SymPy and integrates it exactly, a different route from the generator's own
`mpmath.quad` check. It imports nothing from the generator.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects
from scripts import skill_contract
from ucv1_calc_support import X, parse_math, value_of

SKILL_ID = "ucv1-improper-integral"
TOPIC = "ucv1_improper_integral"
EXAM_TOPIC = "exd_x_uc_improper_integral"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_INTEGRAL = re.compile(r"∫_(?P<lo>\S+?)\^(?P<hi>\S+) (?P<body>.+) dx")
_PARAMETER = re.compile(r"Find the value of (?P<sym>[kp]) > [01] for which (?P<integral>∫.+) = (?P<value>\S+)\.$")


def _limit(text: str):
    text = text.strip("()")
    return sympy.oo if text == "∞" else -sympy.oo if text == "-∞" else sympy.sympify(text)


def _integrate(integral_text: str, **symbols):
    m = _INTEGRAL.search(integral_text)
    assert m, integral_text
    body = parse_math(m["body"].replace("kx", "k*x"), **symbols)
    return sympy.integrate(body, (X, _limit(m["lo"]), _limit(m["hi"])))


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
    assert _SKILL.answer == "expression"


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_sympy_integral_of_the_printed_question(tier):
    parameter_questions = 0
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        key = value_of(q.answer_text)
        m = _PARAMETER.search(q.text)
        if m:
            # a parameter question: put the key back into the printed integral and integrate
            parameter_questions += 1
            assert key.is_positive
            assert _same(_integrate(m["integral"], **{m["sym"]: key}), sympy.Rational(m["value"])), (seed, q.text)
        else:
            assert _same(_integrate(q.text), key), (seed, q.text, q.answer_text)
    if tier == "hard":
        assert parameter_questions >= 8


def test_a_parameter_question_has_exactly_one_admissible_value():
    """For the printed integral as a function of its parameter, the key is the only positive root."""
    seen = 0
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = _PARAMETER.search(q.text)
        if not m:
            continue
        sym = sympy.Symbol(m["sym"], positive=True)
        body = _INTEGRAL.search(m["integral"])
        expr = parse_math(body["body"].replace("kx", "k*x"), **{m["sym"]: sym})
        lo, hi = _limit(body["lo"]), _limit(body["hi"])
        total = sympy.integrate(expr, (X, lo, hi), conds="none")
        target = sympy.Rational(m["value"])
        roots = [r for r in sympy.solve(sympy.Eq(total, target), sym) if r.is_positive and (m["sym"] != "p" or r > 1)]
        assert roots == [value_of(q.answer_text)], (seed, q.text, roots)
        seen += 1
    assert seen >= 8


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
    q = make(TOPIC, seed=2, difficulty="medium")
    while "ln" not in q.answer_text:  # a logarithm key
        q = make(TOPIC, seed=q.seed + 1, difficulty="medium")
    value = value_of(q.answer_text)
    for spelling in (sympy.sstr(value), sympy.sstr(value).replace("log", "ln"), q.answer_text.replace("ln(", "log(")):
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
    assert len(texts) >= 35
    assert make(TOPIC, seed=7, difficulty=tier).to_json() == make(TOPIC, seed=7, difficulty=tier).to_json()


def test_the_tiers_escalate_in_form():
    easy = " ".join(make(TOPIC, seed=s, difficulty="easy").text for s in range(60))
    medium = " ".join(make(TOPIC, seed=s, difficulty="medium").text for s in range(60))
    hard = " ".join(make(TOPIC, seed=s, difficulty="hard").text for s in range(60))
    assert "ln" not in easy and "for which" not in easy and "(x +" not in easy
    assert "(x +" in medium and "x² + " in medium and "for which" not in medium and "ln x" not in medium
    assert "ln x" in hard and "for which" in hard


def test_no_standalone_convergence_question():
    for tier in TIERS:
        for seed in range(60):
            text = make(TOPIC, seed=seed, difficulty=tier).text.lower()
            assert "converge" not in text and "diverge" not in text


@pytest.mark.parametrize("tier", TIERS)
def test_no_text_defects(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


# -- the exam chain ------------------------------------------------------------------------------------------------


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _density(q):
    """The density function printed in the stem: ``(g, lo, hi, k)`` with f = k g."""
    m = re.search(r"f\(x\) = (?P<g>.+?) for (?P<dom>.+?) and f\(x\) = 0", q.stem)
    assert m, q.stem
    g = m["g"]
    g = "1/" + g[2:] if g.startswith("k/") else g[2:] if g.startswith("k ") else None
    assert g is not None, m["g"]
    dom = m["dom"]
    if dom.startswith("x ≥ "):
        lo, hi = sympy.Integer(dom[4:]), sympy.oo
    else:
        lo, hi = sympy.Integer(0), sympy.Integer(re.fullmatch(r"0 < x ≤ (\d+)", dom)[1])
    gx = parse_math(g)
    k = 1 / sympy.integrate(gx, (X, lo, hi))
    return gx, lo, hi, k


def _probability_range(q, lo, hi):
    m = re.search(r"P\(X (?P<op>[>≤]) (?P<t>\d+)\)", q.parts[1].prompt)
    assert m, q.parts[1].prompt
    t = sympy.Integer(m["t"])
    return (t, hi) if m["op"] == ">" else (lo, t)


def _density_moments(q):
    gx, lo, hi, k = _density(q)
    a, b = _probability_range(q, lo, hi)
    return dict(
        k=k,
        tail=sympy.integrate(gx, (X, a, b)),
        first=sympy.integrate(X * gx, (X, lo, hi)),
        second=sympy.integrate(X**2 * gx, (X, lo, hi)),
    )


def _region(q):
    m = re.search(r"y = (?P<c>\d+)/x\^p .*area of R is (?P<area>\S+) square", q.stem)
    assert m, q.stem
    c = sympy.Integer(m["c"])
    area = sympy.Rational(m["area"])
    p = sympy.Symbol("p", positive=True)
    area_of_p = sympy.integrate(c * X ** (-p), (X, 1, sympy.oo), conds="none")
    (p_value,) = [r for r in sympy.solve(sympy.Eq(area_of_p, area), p) if r > 2]
    return c, p_value


def _volumes(c, p):
    v_x = sympy.pi * sympy.integrate((c * X**-p) ** 2, (X, 1, sympy.oo))
    v_y = 2 * sympy.pi * sympy.integrate(X * c * X**-p, (X, 1, sympy.oo))
    return v_x, v_y


def is_region_text(stem: str) -> bool:
    return stem.startswith("The region R")


def is_region(q) -> bool:
    return is_region_text(q.stem)


def exam_oracle(q) -> dict[str, sympy.Basic]:
    if is_region(q):
        c, p = _region(q)
        v_x, v_y = _volumes(c, p)
        return {"a": p, "b": v_x, "c": v_y, "d": v_x / v_y}
    mo = _density_moments(q)
    mean = mo["first"] * mo["k"]
    return {
        "a": mo["k"],
        "b": mo["k"] * mo["tail"],
        "c": mean,
        "d": mo["k"] * mo["second"] - mean**2,
    }


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_source_wired_last():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages) == (EXAM_TOPIC, "exam", ["exam"])


def test_exam_shape_marks_and_follow_through():
    for seed in range(80):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert sum(p.marks for p in q.parts) == 10
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q, include_working=True) == []
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_exam_covers_all_five_families():
    def family(stem: str) -> str:
        if is_region_text(stem):
            return "region"
        return "root" if "/√x" in stem else "power" if "k/x^" in stem else "x-exp" if "k x e^" in stem else "exp"

    assert {family(_exam(s).stem) for s in range(80)} == {"region", "root", "power", "x-exp", "exp"}


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


def _reply(value) -> str:
    if value.is_Rational:
        return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"
    return sympy.sstr(value)


def test_exam_follow_through_from_a_wrong_part_a():
    """A learner whose part (a) is wrong but who carries it correctly into (b) to (d) keeps those marks."""
    checked = 0
    for seed in range(80):
        q = _exam(seed)
        floor = 2 if is_region(q) else 0
        wrong_text = next((d for d in q.parts[0].distractors if sympy.Rational(d) > floor), None)
        if wrong_text is None:
            continue
        wrong = sympy.Rational(wrong_text)
        canon = _canonical(q)
        if is_region(q):
            c, _ = _region(q)
            v_x, v_y = _volumes(c, wrong)
            replies = {"b": v_x, "c": v_y, "d": v_x / v_y}
        else:
            gx, lo, hi, _ = _density(q)
            a, b = _probability_range(q, lo, hi)
            first = sympy.integrate(X * gx, (X, lo, hi))
            second = sympy.integrate(X**2 * gx, (X, lo, hi))
            replies = {
                "b": wrong * sympy.integrate(gx, (X, a, b)),
                "c": wrong * first,
                "d": wrong * second - (wrong * first) ** 2,
            }
        if min(abs(v) for v in replies.values() if v.is_number) < sympy.Rational(1, 100):
            continue  # the engine withholds follow-through when a carried value rounds to zero
        result = grade(_SKILL, q, {**canon, "a": wrong_text, **{k: _reply(v) for k, v in replies.items()}})
        assert result.parts["a"].marks == 0
        for label in "bcd":
            part = result.parts[label]
            if label == "b" and q.parts[1].sensible_range and not 0 < replies["b"] < 1:
                continue  # an impossible probability carries no follow-through
            assert part.marks == q.parts[ord(label) - 97].marks and (part.follow_through or part.correct), (
                seed,
                label,
                part,
            )
        checked += 1
    assert checked >= 30


def test_exam_an_unrelated_wrong_reply_earns_nothing_later():
    q = _exam(4)
    canon = _canonical(q)
    result = grade(_SKILL, q, {**canon, "c": q.parts[2].distractors[0]})
    assert result.parts["c"].marks == 0
    assert result.parts["a"].marks == q.parts[0].marks


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
