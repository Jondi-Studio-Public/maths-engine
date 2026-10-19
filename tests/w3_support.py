"""Helpers shared by the `test_w3_*` files: parsing amounts back out of question text, and the checks every w3 skill
repeats (wiring, key accepted and distractors rejected, pools, answer spread, option shape)."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.misconceptions import CATALOGUE
from tests import w7b_support as sup

MONEY = re.compile(r"\$([\d,]+(?:\.\d+)?)")


def dollars(text):
    return [Fraction(m.replace(",", "")) for m in MONEY.findall(text)]


def percents(text):
    return [Fraction(m) for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]


def ints_before(text, word):
    return [int(m) for m in re.findall(rf"(\d+) {word}", text)]


def cents(x: Fraction) -> Fraction:
    """Round half up to the cent with integer arithmetic."""
    return Fraction(int(x * 100 + Fraction(1, 2)), 100)


def check_catalogue(mod):
    for ident, description, _area in mod.MISCONCEPTIONS:
        assert CATALOGUE[ident].description == description, ident


def shared_checks(skill, topic, exam, spread_min=12, repeat_max=10):
    """A pytest class with the checks every w3 skill repeats, bound to this skill."""

    class Shared:
        def test_wiring(self):
            sup.check_wiring(skill, topic, exam, "number")
            steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
            assert steps == sorted(steps) and len(set(steps)) == 4

        @pytest.mark.parametrize("tier", sup.TIERS)
        def test_key_accepted_and_distractors_rejected(self, tier):
            sup.check_single_tier(skill, topic, tier)

        def test_pools(self):
            sup.check_pools(topic)

        def test_answers_are_not_guessable(self):
            for tier in sup.TIERS:
                spread = sup.answer_spread(topic, tier, 60)
                assert len(spread) >= spread_min, (tier, len(spread))
                assert max(spread.values()) <= repeat_max, (tier, spread.most_common(3))

        def test_distractors_match_the_answer_shape(self):
            for tier in sup.TIERS:
                for seed in range(60):
                    q = sup.single(topic, tier, seed)
                    places = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
                    for d in q.distractors:
                        assert (len(d.split(".")[1]) if "." in d else 0) == places, (tier, seed, d, q.answer_text)

    return Shared
