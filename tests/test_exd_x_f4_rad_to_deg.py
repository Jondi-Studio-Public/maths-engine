"""Exam source for `angle-radians-to-degrees-decimal`: the radian value must first be found or reduced
(triangle third angle, arc subtending an angle, whole turns removed) and then converted. Keys are re-derived
from the rendered prompt."""

from __future__ import annotations

import math
import re

import mathgen
import mathgen.generators.exd_x_f4_rad_to_deg  # noqa: F401  (registers the topic)
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f4_rad_to_deg"


def _make(seed):
    return mathgen.make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return mathgen.make("t4_radians_to_degrees_decimal", seed=seed, difficulty="hard", variant="pi_fraction_input")


def _round(x: float, places: int) -> str:
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _agrees(q) -> bool:
    text = q.text
    if "triangle" in text:
        a, b = (float(x) for x in re.findall(r"\d\.\d\d", text)[:2])
        value = math.degrees(math.pi - a - b)
    elif "arc" in text:
        r = float(re.search(r"radius (\d+)", text).group(1))
        length = float(re.search(r"(?:length|is) (\d+\.\d+) cm", text).group(1))
        value = math.degrees(length / r)
    else:
        w = float(re.search(r"(?:at|rotates at) (\d+\.\d+) rad", text).group(1))
        t = int(re.search(r"(?:for) (\d+) seconds", text).group(1))
        value = math.degrees(w * t) % 360
    return _round(value, 1) == q.answer_text and float(q.answer_text) == q.answer


def test_all_three_settings_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("triangle" in t for t in texts)
    assert any("arc" in t for t in texts)
    assert any("rad/s" in t or "radians per second" in t for t in texts)


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
