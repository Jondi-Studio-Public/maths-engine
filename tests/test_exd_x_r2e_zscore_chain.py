"""Tests for the multi-part exam source of `z-score` (`exd_x_r2e_zscore_chain`): the standard deviation and mean
from two z-scores, a third z-score, and a normal tail probability."""

from __future__ import annotations

import collections
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from statistics import NormalDist

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2e_zscore_chain"
SKILL_ID = "z-score"
HARD = ("z_score", {})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _dp(x: float, places: int) -> str:
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _facts(q):
    (x1, z1), (x2, z2) = re.findall(r"of (\d+)(?: g| mm| minutes)? has a z-score of (-?\d+\.\d\d)", q.stem)
    x3 = int(re.search(r"of (\d+)", q.parts[2].prompt).group(1))
    upper = "greater than" in q.parts[3].prompt
    return int(x1), Fraction(z1), int(x2), Fraction(z2), x3, upper


def _tail(z: float, upper: bool) -> str:
    return _dp(1 - NormalDist().cdf(z) if upper else NormalDist().cdf(z), 4)


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem and the part prompts alone."""
    x1, z1, x2, z2, x3, upper = _facts(q)
    sigma = (x2 - x1) / (z2 - z1)
    mu = x1 - z1 * sigma
    z3 = (x3 - mu) / sigma
    return {
        "a": str(sigma),
        "b": str(mu),
        "c": _dp(float(z3), 2),
        "d": _tail(float(z3), upper),
    }


def test_wired_to_the_skill():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam" and s.effective_key == "exam"]
    assert [(s.topic, s.params, s.stages) for s in exam] == [(TOPIC, {"tier": "exam"}, ["exam"])]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_add_to_the_hard_tier():
    shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_answer_space(questions):
    tuples = collections.Counter(tuple(p.answer_text for p in q.parts) for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_distractors_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_a_wrong_sd(questions):
    """A wrong standard deviation carried into the mean, the z-score and the probability."""
    checked = 0
    for q in questions[:150]:
        x1, z1, _x2, _z2, x3, upper = _facts(q)
        sigma_wrong = 2 * int(q.parts[0].answer_text)
        mu_wrong = x1 - z1 * sigma_wrong
        if mu_wrong <= 0:
            continue
        z_wrong = _dp(float((x3 - mu_wrong) / sigma_wrong), 2)
        reply = {"a": str(sigma_wrong), "b": str(mu_wrong), "c": z_wrong, "d": _tail(float(z_wrong), upper)}
        if len({reply["a"], reply["b"], reply["c"]}) < 3 or Fraction(z_wrong) == 0:
            continue  # a zero z-score is a trivial reply and earns no follow-through
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)
        assert result.marks == 5 and not result.correct
        checked += 1
    assert checked >= 60


def test_follow_through_into_the_probability(questions):
    for q in questions[:100]:
        canon = _canonical(q)
        upper = _facts(q)[5]
        z_wrong = Fraction(canon["c"]) + Fraction(1, 2)
        if z_wrong == 0:
            continue
        result = grade(_SKILL, q, {**canon, "c": str(float(z_wrong)), "d": _tail(float(z_wrong), upper)})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, q.text


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "7777", "b": "7777", "c": "9999", "d": "99999"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert make(TOPIC, seed=11).to_dict() == make(TOPIC, seed=11).to_dict()
