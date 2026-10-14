"""Tests `f10-check-given-answer-reasonable` (topic `fr2b_check_reasonable`), a two-part question: an estimate, then a
verdict on a stated answer.

The oracle reads the numbers and the stated answer out of the question text, rounds them its own way, and decides the
verdict from the exact answer.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.fr2b_check_reasonable_build  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "fr2b_check_reasonable"
SKILL_ID = "f10-check-given-answer-reasonable"
TIERS = ("easy", "medium", "hard")
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

OK, HIGH, LOW = "reasonable", "too high", "too low"


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _nearest(value, unit):
    return int(value / unit + 0.5) * unit


def _whole_numbers(q):
    nums = [int(n.replace(" ", "")) for n in re.findall(r"\d[\d ]*\d|\d", q.stem)]
    a, b, stated = nums[0], nums[1], nums[2]
    subtract = "subtract" in q.parts[0].prompt
    unit = 10 if "ten" in q.parts[0].prompt else 100
    return a, b, stated, subtract, unit


def _verdict(stated, estimate, tolerance):
    if abs(stated - estimate) <= tolerance:
        return OK
    return HIGH if stated > estimate else LOW


def test_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [(TOPIC, t) for t in TIERS]


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_whole_number_estimate_and_verdict(tier):
    verdicts = Counter()
    for seed in range(200):
        q = _q(tier, seed)
        a, b, stated, subtract, unit = _whole_numbers(q)
        estimate = _nearest(a, unit) - _nearest(b, unit) if subtract else _nearest(a, unit) + _nearest(b, unit)
        exact = a - b if subtract else a + b
        assert int(q.parts[0].answer) == estimate, (seed, q.stem)
        verdict = _verdict(stated, estimate, unit)
        # The exact answer is always inside the rounding bound, so a stated answer is judged the same way the
        # generator judged it, and a wrong one is clear of the bound by more than three times its size.
        assert abs(exact - estimate) <= unit
        assert verdict == OK or abs(stated - estimate) > 3 * unit
        assert (stated == exact) == (verdict == OK)
        assert q.parts[1].answer_text.split(":")[0].lower().endswith(verdict.split()[-1]) or verdict == OK
        verdicts[verdict] += 1
    assert set(verdicts) == {OK, HIGH, LOW}, verdicts


def test_hard_money_estimate_and_verdict():
    shapes = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        quantity = int(re.search(r"buys (\d+)", q.stem).group(1))
        price = int(round(float(re.search(r"\$(\d+\.\d\d) each", q.stem).group(1)) * 100))
        note = int(re.search(r"\$(\d+) note", q.stem).group(1))
        stated = float(re.search(r"\$(\d+\.\d\d)\.", q.stem).group(1))
        estimate = note - quantity * _nearest(price / 100, 1)
        assert int(q.parts[0].answer) == estimate, (seed, q.stem)
        exact = note - quantity * price / 100
        assert abs(exact - estimate) <= quantity / 2
        verdict = _verdict(stated, estimate, quantity / 2)
        assert (abs(stated - exact) < 0.005) == (verdict == OK)
        assert verdict == OK or abs(stated - estimate) > 3 * quantity / 2
        shapes[verdict] += 1
    assert set(shapes) == {OK, HIGH, LOW}, shapes


@pytest.mark.parametrize("tier", TIERS)
def test_parts_shape_marks_and_options(tier):
    for seed in range(100):
        q = _q(tier, seed)
        assert [p.answer_kind for p in q.parts] == ["number", "mcq"]
        assert [p.marks for p in q.parts] == [2, 1]
        assert len(q.parts[1].distractors) == 3 and len(set(q.parts[1].distractors)) == 3
        assert q.parts[0].sensible_range == (0, None)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(100):
        q = _q(tier, seed)
        full = grade(SKILL, q, q.answer_text)
        canon = {p.label: p.answer_text for p in q.parts}
        assert full.correct and full.marks == full.max_marks == 3
        for p in q.parts:
            assert len(p.distractors) >= 3
            for text in p.distractors:
                result = grade(SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE
        # A number part never has a decimal point in it, so no distractor differs in places.
        assert all("." not in d for d in q.parts[0].distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_variety():
    for tier in TIERS:
        assert len({_q(tier, s).text for s in range(80)}) >= 60
