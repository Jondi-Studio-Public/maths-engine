"""Exam source for `angle-degrees-to-radians-decimal`: the angle must first be found (polygon interior angle,
clock hands, triangle exterior angle) and then converted. Keys are re-derived from the rendered prompt."""

from __future__ import annotations

import math
import re

import mathgen
import mathgen.generators.exd_x_f4_deg_to_rad  # noqa: F401  (registers the topic)
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f4_deg_to_rad"


def _make(seed):
    return mathgen.make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return mathgen.make("t4_degrees_to_radians_decimal", seed=seed, difficulty="hard", variant="dms_forward")


def _round(x: float, places: int) -> str:
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _agrees(q) -> bool:
    text = q.text
    if "polygon" in text:
        n = int(re.search(r"\d+", text).group(0))
        value = (n - 2) * math.pi / n
    elif "clock" in text:
        h, m = (int(x) for x in re.search(r"(\d+):(\d\d)", text).groups())
        gap = abs((30 * h + m / 2) - 6 * m)
        value = math.radians(min(gap, 360 - gap))
    else:
        a, b = (int(x) for x in re.findall(r"(\d+) degrees", text))
        value = math.radians(a + b)
    return _round(value, 3) == q.answer_text and float(q.answer_text) == q.answer


def test_all_three_settings_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("polygon" in t for t in texts)
    assert any("clock" in t for t in texts)
    assert any("triangle" in t for t in texts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_make(s)) for s in range(80)}
    hard = {prompt_shape(_hard(s)) for s in range(80)}
    assert len(exam) >= 3
    assert len(exam - hard) >= 3


def test_key_matches_independent_oracle_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        if not _agrees(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["number"]
    for seed in range(150):
        q = _make(seed)
        assert grade(q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, q.text, d)
