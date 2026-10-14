"""Tests `f10-fraction-decimal-links` (topic `fr2b_fraction_decimal_links`).

The oracle reads the decimal or the fractions back out of the question text and rebuilds the key in exact Fractions.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.fr2b_fraction_decimal_links_build  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "fr2b_fraction_decimal_links"
SKILL_ID = "f10-fraction-decimal-links"
TIERS = ("easy", "medium", "hard")
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def test_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [(TOPIC, t) for t in TIERS]


def test_easy_key_is_the_decimal_as_a_fraction():
    denominators = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        decimal = Fraction(re.search(r"\d\.\d+", q.text).group(0))
        num, den = (int(x) for x in q.answer_text.split("/"))
        assert Fraction(num, den) == decimal, (seed, q.text, q.answer_text)
        assert den in (10, 100) and q.answer_kind == "fraction"
        denominators[den] += 1
    assert set(denominators) == {10, 100}, denominators


def test_medium_key_is_the_sum_or_difference_as_a_decimal():
    ops = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        t, op, h = re.search(r"(\d+)/10 ([+-]) (\d+)/100", q.text).groups()
        expected = Fraction(int(t), 10) + (1 if op == "+" else -1) * Fraction(int(h), 100)
        assert Fraction(q.answer_text) == expected and 0 < expected < 1, (seed, q.text)
        assert len(q.answer_text.split(".")[1]) == 2
        ops[op] += 1
    assert set(ops) == {"+", "-"}, ops


def test_hard_key_counts_hundredths():
    for seed in range(150):
        q = _q("hard", seed)
        t = int(re.search(r"(\d+)/10", q.text).group(1))
        h = int(re.search(r"(\d+)/100", q.text).group(1))
        d = Fraction(re.search(r"0\.\d+", q.text).group(0))
        assert q.answer == t * 10 + h + int(d * 100) == int((Fraction(t, 10) + Fraction(h, 100) + d) * 100), q.text


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(120):
        q = _q(tier, seed)
        assert grade(SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 3
        places = {len(d.split(".")[1]) if "." in d else None for d in q.distractors}
        assert places == {len(q.answer_text.split(".")[1]) if "." in q.answer_text else None}, (
            q.answer_text,
            q.distractors,
        )
        for text in q.distractors:
            assert not grade(SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE


def test_easy_fraction_accepts_an_equivalent_form_but_not_a_place_slip():
    for seed in range(60):
        q = _q("easy", seed)
        num, den = (int(x) for x in q.answer_text.split("/"))
        assert grade(SKILL, q, f"{num * 10}/{den * 10}").correct
        wrong = f"{num}/{den * 10 if den == 10 else den // 10}"
        assert not grade(SKILL, q, wrong).correct


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_variety():
    assert len({_q("easy", s).text for s in range(80)}) >= 40
    assert len({_q("medium", s).text for s in range(80)}) >= 60
    assert len({_q("hard", s).answer for s in range(80)}) >= 30
