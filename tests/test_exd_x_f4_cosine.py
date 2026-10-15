"""Exam source for `cosine-rule`: the included angle must be found from bearings, a change of course or a
parallelogram, and the cosine rule chosen. Keys are re-derived from the rendered prompt by vector geometry."""

from __future__ import annotations

import math
import re

import mathgen
import mathgen.generators.exd_x_f4_cosine  # noqa: F401  (registers the topic)
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f4_cosine"


def _make(seed):
    return mathgen.make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return mathgen.make("trigonometry", seed=seed, difficulty="hard", kinds=("cosine_rule",))


def _vec(r, bearing):
    b = math.radians(bearing)
    return (r * math.sin(b), r * math.cos(b))


def _round(x: float, places: int) -> str:
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _agrees(q) -> bool:
    text = q.text.replace("1 decimal place", "")
    if "parallelogram" in text:
        a, b, t = (int(x) for x in re.findall(r"\d+", text))
        which = "longer" if "longer" in text else "shorter"
        u = (a, 0.0)
        v = (b * math.cos(math.radians(t)), b * math.sin(math.radians(t)))
        d1 = math.hypot(u[0] - v[0], u[1] - v[1])
        d2 = math.hypot(u[0] + v[0], u[1] + v[1])
        value = max(d1, d2) if which == "longer" else min(d1, d2)
    elif "lighthouse" in text or "camp" in text:
        r1, b1, r2, b2 = (int(x) for x in re.findall(r"\d+", text))
        p, q2 = _vec(r1, b1), _vec(r2, b2)
        value = math.hypot(p[0] - q2[0], p[1] - q2[1])
    else:
        r1, b1, b2, r2 = (int(x) for x in re.findall(r"\d+", text))
        p, q2 = _vec(r1, b1), _vec(r2, b2)
        value = math.hypot(p[0] + q2[0], p[1] + q2[1])
    return _round(value, 1) == q.answer_text and float(q.answer_text) == q.answer


def test_all_three_settings_occur():
    texts = [_make(s).text for s in range(60)]
    assert any("parallelogram" in t for t in texts)
    assert any("lighthouse" in t or "camp" in t for t in texts)
    assert any("further" in t or "more" in t for t in texts)


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
