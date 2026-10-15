"""Tests for the natural-logarithm exam source (`exd_x_f1_ln_exponential_equation`)."""

from __future__ import annotations

import math
import re

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f1_ln_exponential_equation"
SKILL_ID = "senior-define-the-natural-logarithm-ln-x-log-ex"


def _skill():
    return {sk.id: sk for sk in load_skills()[1]}[SKILL_ID]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    src = [s for s in _skill().sources if s.difficulty == "hard"][0]
    return make(src.topic, seed=seed, difficulty="hard", **src.params)


def _key_from_prompt(text):
    m = re.search(r"e\^\(2x\) - (\d+)e\^x \+ (\d+) = 0", text)
    if m:
        s, prod = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"e\^x \+ (\d+)e\^\(-x\) = (\d+)", text)
        prod, s = int(m.group(1)), int(m.group(2))
    # t = e^x solves t^2 - s t + prod = 0 (the second form after multiplying by t).
    disc = s * s - 4 * prod
    root = math.sqrt(disc)
    xs = sorted(math.log((s + sign * root) / 2) for sign in (-1, 1))
    if "larger" in text:
        return round(xs[1], 3)
    if "smaller" in text:
        return round(xs[0], 3)
    assert "sum" in text
    return round(sum(xs), 3)


def test_key_rederived_from_prompt():
    bad = [s for s in range(300) if abs(_key_from_prompt(_exam(s).text) - _exam(s).answer) > 1e-9]
    assert bad == []


def test_grader_accepts_key_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        assert GRADERS["number"](q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) == 3
        for dist in q.distractors:
            assert not GRADERS["number"](q, dist).correct, (seed, dist, q.answer_text)


def test_skill_has_exam_source_last():
    src = _skill().sources[-1]
    assert (src.topic, src.difficulty, list(src.stages)) == (TOPIC, "exam", ["exam"])
