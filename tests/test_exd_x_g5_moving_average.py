"""Tests for the `moving-average` exam source (topic `exd_x_g5_moving_average`): a missing observation from a window average, and a centred even-window average.

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

TOPIC = "exd_x_g5_moving_average"
BASE_TOPIC = "moving_average"
BASE_PARAMS = {"window": 3}
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
    assert seen == {"missing", "centred"}, seen


# -- the independent oracle ---------------------------------------------------------------------------------


def _variant(text):
    if "missing" in text:
        return "missing"
    assert "centred" in text, text
    return "centred"


def _oracle(text):
    if _variant(text) == "centred":
        listed = re.search(r"are ([\d, ]+)\. A 4-point", text).group(1)
        x = [Fraction(int(v)) for v in listed.split(", ")]
        assert len(x) == 5
        weighted = x[0] / 2 + x[1] + x[2] + x[3] + x[4] / 2
        return _cents(weighted / 4)
    listed = re.search(r"are ([\d?, ]+) \(the", text).group(1)
    items = listed.split(", ")
    m = re.search(r"The (\d+)-point moving average of \w+ (\d+) to (\d+) is (\d+(?:\.\d+)?)\.", text)
    window, lo, hi, avg = int(m.group(1)), int(m.group(2)), int(m.group(3)), _money(m.group(4))
    assert hi - lo + 1 == window
    known = sum(int(items[i - 1]) for i in range(lo, hi + 1) if items[i - 1] != "?")
    assert any(items[i - 1] == "?" for i in range(lo, hi + 1))
    return window * avg - known
