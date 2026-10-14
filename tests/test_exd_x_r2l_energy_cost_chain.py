"""Tests for the exam source of senior-use-units-of-energy-to-describe-consumption-of.

The exam tier is a three-part chain: daily energy in kilowatt-hours from a power in watts and a time in hours and
minutes, the cost over a stated number of days at a tariff in cents per kilowatt-hour, and the saving from a
lower-power replacement. Every key is re-derived from the rendered prompt with exact fractions.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import round_half_up
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2l_energy_cost_chain"
SKILL_ID = "senior-use-units-of-energy-to-describe-consumption-of"
SEEDS = range(320)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _two_places(v: Fraction) -> str:
    cents = round_half_up(v, 2) * 100
    return f"{int(cents) // 100}.{int(cents) % 100:02d}"


def _facts(q):
    power, hours, minutes = (
        int(v) for v in re.search(r"rated at (\d+) W for (\d+) hours? (\d+) minutes", q.text).groups()
    )
    tariff = Fraction(re.search(r"costs (\d+\.\d) cents", q.text).group(1))
    days = int(re.search(r"for (\d+) days, in dollars", q.parts[1].prompt).group(1))
    new_power = int(re.search(r"rated at (\d+) W, used", q.parts[2].prompt).group(1))
    return power, Fraction(hours) + Fraction(minutes, 60), tariff, days, new_power


def oracle(q) -> dict[str, str]:
    power, t, tariff, days, new_power = _facts(q)
    energy = Fraction(power, 1000) * t
    cost = energy * days * tariff / 100
    saving = Fraction(power - new_power, 1000) * t * days * tariff / 100
    return {"a": _two_places(energy), "b": _two_places(cost), "c": _two_places(saving)}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_oracle_agrees_over_300_seeds(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == []


def test_new_appliance_uses_less_power_and_amounts_are_plausible(questions):
    for q in questions:
        power, t, tariff, days, new_power = _facts(q)
        assert 400 <= new_power < power <= 3000
        assert 0 < float(q.parts[0].answer_text) < 40
        assert 0 < float(q.parts[1].answer_text) < 2500


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_every_distractor_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) == 3 and len(set(p.distractors)) == 3
            assert all(p.option_notes.get(d) in CATALOGUE for d in p.distractors), (p.label, p.option_notes)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_daily_energy(questions):
    checked = 0
    for q in questions[:150]:
        power, t, tariff, days, new_power = _facts(q)
        wrong_a = Fraction(q.parts[0].answer_text) + Fraction(1, 2)
        wrong_b = round_half_up(wrong_a * days * tariff / 100, 2)
        wrong_c = round_half_up(wrong_b * Fraction(power - new_power, power), 2)
        result = grade(_SKILL, q, {"a": _two_places(wrong_a), "b": _two_places(wrong_b), "c": _two_places(wrong_c)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True
        assert result.marks == 4 and not result.correct
        checked += 1
    assert checked == 150


def test_correct_rounded_cost_carries_into_the_saving(questions):
    for q in questions[:100]:
        power, t, tariff, days, new_power = _facts(q)
        b = Fraction(q.parts[1].answer_text)
        c = round_half_up(b * Fraction(power - new_power, power), 2)
        result = grade(_SKILL, q, {**_canonical(q), "c": _two_places(c)})
        assert result.parts["c"].marks == 2


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "999.99", "b": "1.11", "c": "2.22"})
        assert result.marks == 0
