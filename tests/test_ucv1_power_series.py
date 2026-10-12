"""Tests `ucv1-power-series-convergence` (topic `ucv1_power_series_convergence`) and the chain `exd_x_uc_power_series`.

The oracle reads the series back out of the question TEXT, parses it into a SymPy term, and re-derives the interval
from scratch: the limit of |t(n+1)/t(n)| solved for x gives the open interval, and `Sum(...).is_convergent()` decides
each endpoint. That shares no code with the generator, which classifies endpoints by the p-series rule and checks
them numerically.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen.generators.ucv1_power_series_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.grading.inequality import read_solution_set
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_power_series_convergence"
EXAM = "exd_x_uc_power_series"
SKILL_ID = "ucv1-power-series-convergence"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_X = sympy.symbols("x", real=True)
_N = sympy.symbols("n", integer=True, positive=True)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _series(text):
    return re.search(r"Σ (.*?), (?:summed|with|where)", text).group(1)


def _term(series):
    s = series.replace("·", "*").replace("n²", "n**2").replace("^", "**")
    s = re.sub(r"\)\s+\(", ")*(", s)
    s = re.sub(r"(\*\*n)\s+n", r"\1*n", s)
    return parse_expr(s, local_dict={"x": _X, "n": _N}, transformations=_TRANSFORMS)


def _intervals(term):
    """(open interval, full interval of convergence) by the ratio test and SymPy's convergence tests."""
    ratio = sympy.simplify(term.subs(_N, _N + 1) / term)
    limit = sympy.limit(sympy.Abs(ratio), _N, sympy.oo)
    opened = sympy.solve_univariate_inequality(limit < 1, _X, relational=False)

    def converges(endpoint):
        t = sympy.simplify(term.subs(_X, endpoint))
        return bool(sympy.Sum(t, (_N, 1, sympy.oo)).is_convergent())

    lo, hi = opened.start, opened.end
    return opened, sympy.Interval(lo, hi, not converges(lo), not converges(hi))


def _q(difficulty, seed):
    return make(TOPIC, seed=seed, difficulty=difficulty)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _set(text):
    read = read_solution_set(text)
    assert read is not None, text
    return read[0]


# -- wiring ---------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "inequality"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    src = _SKILL.sources[-1]
    assert (src.stages, src.params) == (["exam"], {})
    assert _SKILL.prerequisites


# -- single-answer tiers ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier,full", [("easy", True), ("medium", False), ("hard", True)])
def test_key_matches_independent_interval(tier, full):
    for seed in range(80):
        q = _q(tier, seed)
        opened, interval = _intervals(_term(_series(q.text)))
        expected = interval if full else opened
        assert q.answer_kind == "inequality"
        assert _set(q.answer) == expected, (seed, q.text, q.answer_text, expected)
        if tier == "easy":
            assert interval == opened, (seed, q.text)  # both endpoints diverge, so the open interval is the answer


def test_hard_tier_exercises_every_endpoint_pattern():
    patterns = Counter()
    for seed in range(120):
        q = _q("hard", seed)
        s = _set(q.answer)
        patterns[(s.left_open, s.right_open)] += 1
    assert set(patterns) == {(True, True), (False, True), (True, False), (False, False)}, patterns


def test_medium_never_asks_for_endpoints_and_is_always_shifted():
    for seed in range(80):
        q = _q("medium", seed)
        s = _set(q.answer)
        assert s.left_open and s.right_open
        assert (s.start + s.end) / 2 != 0, (seed, q.text)
        assert "open interval" in q.text or "ratio test" in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_in_other_spellings_and_distractors_rejected(tier):
    for seed in range(80):
        q = _q(tier, seed)
        s = _set(q.answer)
        interval_text = ("(" if s.left_open else "[") + f"{s.start}, {s.end}" + (")" if s.right_open else "]")
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert grade(_SKILL, q, interval_text).correct, (seed, interval_text)
        assert not grade(_SKILL, q, "all real numbers").correct
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert _set(text) != s
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


def test_hard_endpoint_slips_are_offered():
    tags = Counter()
    for seed in range(120):
        tags.update(_q("hard", seed).option_notes.values())
    assert {"series-endpoints-both-included", "series-endpoints-both-excluded", "series-endpoints-swapped"} <= set(tags)
    assert {"series-interval-centred-at-zero", "series-centre-sign-error", "series-radius-reciprocal"} <= set(tags)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    answers = [_q(tier, s).answer_text for s in range(60)]
    assert len(set(answers)) >= 12, Counter(answers).most_common(5)
    assert max(Counter(answers).values()) <= 12, Counter(answers).most_common(3)


# -- exam chain --------------------------------------------------------------------------------------------------


def _x0(q):
    return int(re.search(r"f'\((-?\d+)\)", q.parts[3].prompt).group(1))


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 3, 2]
        assert [p.answer_kind for p in q.parts] == ["fraction", "inequality", "inequality", "fraction"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert find_text_defects(q) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(100):
        q = _exam(seed)
        term = _term(_series(q.stem))
        opened, full = _intervals(term)
        assert sympy.Rational(q.parts[0].answer) == (opened.end - opened.start) / 2, seed
        assert _set(q.parts[1].answer) == opened
        assert _set(q.parts[2].answer) == full
        x0 = _x0(q)
        assert opened.start < x0 < opened.end
        derivative = sympy.diff(term, _X).subs(_X, x0)
        total = sympy.summation(sympy.simplify(derivative), (_N, 1, sympy.oo))
        key = q.parts[3].answer
        assert total == (sympy.Rational(key["p"], key["q"]) if isinstance(key, dict) else key), seed


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 8
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_radius():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        term = _term(_series(q.stem))
        opened, _ = _intervals(term)
        radius = (opened.end - opened.start) / 2
        centre = (opened.end + opened.start) / 2
        t = _x0(q) - centre
        sign = 1 if "(-1)^n" not in q.stem else -1
        for wrong in (1 / radius, 2 * radius, radius + 1):
            if wrong - sign * t == 0:
                continue
            value = sympy.Rational(sign) / (wrong - sign * t)
            canon = {p.label: p.answer_text for p in q.parts}
            reply = {"a": str(wrong), "d": str(value)}
            result = grade(_SKILL, q, {**canon, **reply})
            assert result.parts["a"].marks == 0
            if value != sympy.Rational(q.parts[3].answer_text) and value != wrong:
                assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, wrong, result)
                checked += 1
    assert checked >= 100


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(Counter(answers).values()) <= 15
