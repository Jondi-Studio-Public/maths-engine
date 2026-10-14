"""Tests for the `simple-interest` exam source (topic `exd_x_g5_simple_interest`): a sum split between two accounts, and an annual rate from interest over months.

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

TOPIC = "exd_x_g5_simple_interest"
BASE_TOPIC = "financial_interest"
BASE_PARAMS = {"kind": "simple"}
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
    assert seen == {"split", "months"}, seen


# -- the independent oracle ---------------------------------------------------------------------------------


def _variant(text):
    if "divides" in text:
        return "split"
    assert "months" in text, text
    return "months"


def _oracle(text):
    if _variant(text) == "split":
        m = re.search(
            r"divides \$(\d+) between two accounts\. The first pays (\d+)% p\.a.*?second pays (\d+)% p\.a.*?"
            r"After (\d+) years the total interest earned is \$(\d+(?:\.\d+)?)",
            text,
        )
        total, r1, r2, years = (int(g) for g in m.groups()[:4])
        interest = _money(m.group(5))
        hits = [x for x in range(0, total + 1) if Fraction(x * r1 + (total - x) * r2, 100) * years == interest]
        assert len(hits) == 1, (text, hits)
        return Fraction(hits[0])
    m = re.search(r"invests \$(\d+) for (\d+) months .*? earns \$(\d+(?:\.\d+)?) in interest", text)
    p, months, interest = int(m.group(1)), int(m.group(2)), _money(m.group(3))
    hits = [r for r in range(1, 100) if Fraction(p * r * months, 1200) == interest]
    assert len(hits) == 1, (text, hits)
    return Fraction(hits[0])
