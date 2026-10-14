"""Tests for the `compound-interest` exam source (topic `exd_x_g5_compound_interest`): two-stage balance, rate from a final balance, periodic compounding.

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

TOPIC = "exd_x_g5_compound_interest"
BASE_TOPIC = "financial_interest"
BASE_PARAMS = {"kind": "compound"}
_GRADE = GRADERS["number"]


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _money(text):
    return Fraction(text)


def _cents(value):
    """Round half up to the cent."""
    return Fraction(math.floor(value * 100 + Fraction(1, 2)), 100)


def test_prompt_variety_and_new_shapes_over_the_existing_tier():
    exam = {prompt_shape(_make(s)) for s in range(60)}
    base = {prompt_shape(make(BASE_TOPIC, seed=s, difficulty="easy", **BASE_PARAMS)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - base, "the exam source adds no prompt shape beyond the existing tier"


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
    assert seen == {"phases", "rate", "periods"}, seen


# -- the independent oracle ---------------------------------------------------------------------------------


def _variant(text):
    if "rate changes" in text:
        return "phases"
    if "annual interest rate" in text:
        return "rate"
    assert "nominal rate" in text, text
    return "periods"


def _oracle(text):
    kind = _variant(text)
    if kind == "phases":
        m = re.search(
            r"invests \$(\d+) at (\d+)% p\.a.*?After (\d+) years the rate changes to (\d+)% p\.a.*?"
            r"adds a further \$(\d+).*?balance (\d+) years after",
            text,
        )
        p, r1, n1, r2, dep, n2 = (int(g) for g in m.groups())
        bal = Fraction(p) * Fraction(100 + r1, 100) ** n1
        return _cents((bal + dep) * Fraction(100 + r2, 100) ** n2)
    if kind == "periods":
        m = re.search(
            r"invests \$(\d+) at a nominal rate of (\d+)% p\.a., compounded (half-yearly|quarterly).*?after (\d+) years",
            text,
        )
        p, nominal, word, years = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        per = 2 if word == "half-yearly" else 4
        return _cents(Fraction(p) * (1 + Fraction(nominal, 100 * per)) ** (per * years))
    m = re.search(r"puts \$(\d+) .*?After (\d+) years the balance is \$(\d+(?:\.\d+)?)", text)
    p, years, final = int(m.group(1)), int(m.group(2)), _money(m.group(3))
    hits = [r for r in range(1, 60) if _cents(Fraction(p) * Fraction(100 + r, 100) ** years) == final]
    assert len(hits) == 1, (text, hits)
    return Fraction(hits[0])
