"""Tests `f10-written-division` (topic `fr2b_written_division`).

The oracle reads the dividend and divisor out of the question text and rebuilds the key with `divmod` and exact
Fractions.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.fr2b_written_division_build  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "fr2b_written_division"
SKILL_ID = "f10-written-division"
TIERS = ("easy", "medium", "hard")
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_NUMBERS = re.compile(r"\d[\d ]*\d|\d")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _dividend_and_divisor(text):
    """The two numbers in the setting sentence: the quantity (spaces between thousands) and the divisor."""
    nums = [int(n.replace(" ", "")) for n in _NUMBERS.findall(text)]
    return nums[0], nums[1]


def test_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [(TOPIC, t) for t in TIERS]


def test_easy_quotient_has_a_zero_digit_and_no_remainder():
    for seed in range(150):
        q = _q("easy", seed)
        n, d = _dividend_and_divisor(q.text)
        assert divmod(n, d) == (q.answer, 0), (seed, q.text)
        assert "0" in str(q.answer)[1:]
        assert str(q.answer).replace("0", "", 1) in q.distractors, "the zero-dropped answer is offered"


def test_medium_key_is_the_remainder_over_the_divisor():
    for seed in range(150):
        q = _q("medium", seed)
        n, d = _dividend_and_divisor(q.text)
        quotient, remainder = divmod(n, d)
        assert remainder
        value = Fraction(n, d)
        whole, _, frac = q.answer_text.partition(" ")
        assert int(whole) == quotient and Fraction(frac) == Fraction(remainder, d), (seed, q.answer_text)
        assert q.answer_kind == "fraction"
        # An unreduced remainder is the same number and is accepted; a swapped one is not.
        assert grade(SKILL, q, f"{quotient} {remainder * 2}/{d * 2}").correct
        assert grade(SKILL, q, f"{n}/{d}").correct and value == Fraction(n, d)
        assert not grade(SKILL, q, f"{quotient} {d}/{remainder}").correct


def test_hard_key_is_the_exact_decimal_and_distractors_share_its_places():
    divisors = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        n, d = _dividend_and_divisor(q.text)
        divisors[d] += 1
        assert Fraction(q.answer_text) == Fraction(n, d), (seed, q.text)
        places = len(q.answer_text.split(".")[1])
        assert 1 <= places <= 3
        assert all(len(t.split(".")[1]) == places for t in q.distractors), (q.answer_text, q.distractors)
    assert set(divisors) == {2, 4, 5, 8}, divisors


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    tags = set()
    for seed in range(120):
        q = _q(tier, seed)
        assert grade(SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            tags.add(q.option_notes[text])
    assert len(tags) >= 3, tags


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_variety_and_settings_are_plausible():
    for tier in TIERS:
        assert len({_q(tier, s).text for s in range(80)}) >= 60
    for tier in ("medium", "hard"):
        for seed in range(80):
            q = _q(tier, seed)
            share = int(q.answer_text.split()[0].split(".")[0])
            if "ribbon" in q.text:
                assert 25 <= share <= 800, q.text
            if "cycle path" in q.text:
                assert 25 <= share <= 1500, q.text
