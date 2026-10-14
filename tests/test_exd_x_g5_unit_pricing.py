"""Tests for the `unit-pricing` exam source (topic `exd_x_g5_unit_pricing`): a better-buy gap across grams and kilograms, and a multi-buy offer.

The oracle in this file re-derives every answer from the RENDERED PROMPT alone with exact `Fraction`
arithmetic; it does not import the generator module.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g5_unit_pricing"
BASE_TOPIC = "unit_pricing"
BASE_PARAMS = {}
_GRADE = GRADERS["number"]


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _money(text):
    return Fraction(text)


def _cents(value):
    """Round half up to the cent."""
    return Fraction(math.floor(value * 100 + Fraction(1, 2)), 100)


def test_determinism():
    for seed in (0, 1, 2, 3, 100, 12345):
        assert _make(seed).to_json() == _make(seed).to_json()


def test_key_is_accepted_and_every_distractor_rejected_by_the_real_grader():
    bad = []
    for seed in range(150):
        q = _make(seed)
        if not _GRADE(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        if len(q.distractors) != 3 or len(set(q.distractors)) != 3:
            bad.append(("distractor count", seed, q.distractors))
        for d in q.distractors:
            if d == q.answer_text or _GRADE(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]


def test_values_one_unit_away_in_the_last_place_are_rejected():
    for seed in range(60):
        q = _make(seed)
        places = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
        unit = Fraction(1, 10**places)
        for delta in (unit, -unit):
            wrong = Fraction(q.answer_text) + delta
            text = f"{float(wrong):.{places}f}"
            assert not _GRADE(q, text).correct, (seed, q.answer_text, text)


def test_oracle_agrees_with_the_key_over_400_seeds():
    mismatches = []
    for seed in range(400):
        q = _make(seed)
        expected = _oracle(q.text)
        if Fraction(q.answer_text) != expected:
            mismatches.append((seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches[:3]


def test_every_variant_appears():
    seen = {_variant(_make(s).text) for s in range(120)}
    assert seen == {"compare", "offer"}, seen


# -- the independent oracle ---------------------------------------------------------------------------------


def _variant(text):
    if "two packs" in text:
        return "compare"
    assert "special offer" in text, text
    return "offer"


def _grams(amount, unit):
    value = Fraction(amount)
    return value * 1000 if unit == "kg" else value


def _oracle(text):
    if _variant(text) == "compare":
        m = re.search(
            r"packs: (\d+(?:\.\d+)?) (g|kg) for \$(\d+(?:\.\d+)?) and (\d+(?:\.\d+)?) (g|kg) for \$(\d+(?:\.\d+)?)\.",
            text,
        )
        g1, g2 = _grams(m.group(1), m.group(2)), _grams(m.group(4), m.group(5))
        c1, c2 = _money(m.group(3)) * 100, _money(m.group(6)) * 100
        return abs(c1 / (g1 / 100) - c2 / (g2 / 100))
    m = re.search(r"pack of (\d+) .*? costs \$(\d+(?:\.\d+)?)\. A special offer: (.*?)\. If you buy (\d+) packs", text)
    count, price, clause, packs = int(m.group(1)), _money(m.group(2)) * 100, m.group(3), int(m.group(4))
    if "pay for only 2" in clause:
        paid = Fraction(2)
    elif "half price" in clause:
        paid = Fraction(3, 2)
    elif "25% off" in clause:
        paid = packs * Fraction(3, 4)
    else:
        assert "one of them free" in clause, clause
        paid = Fraction(packs - 1)
    return price * paid / (packs * count)
