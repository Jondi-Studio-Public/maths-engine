"""Tests for the logarithm-definition exam source (`exd_x_f1_log_nested_base`)."""

from __future__ import annotations

import re
from fractions import Fraction

from sympy import integer_log

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f1_log_nested_base"
SKILL_ID = "senior-define-logarithms-as-indices-a-x-b-is"


def _skill():
    return {sk.id: sk for sk in load_skills()[1]}[SKILL_ID]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    src = [s for s in _skill().sources if s.difficulty == "hard"][0]
    return make(src.topic, seed=seed, difficulty="hard", **src.params)


def _key_from_prompt(text):
    nested = re.search(r"log_(\d+)\(log_(\d+)\(x\)\) = (\d+)", text)
    if nested:
        a, b, c = (int(v) for v in nested.groups())
        # x = b ** (a ** c): rebuild by exact repeated multiplication, not the generator's formula.
        inner = 1
        for _ in range(c):
            inner *= a
        x = 1
        for _ in range(inner):
            x *= b
        assert integer_log(x, b) == (inner, True)
        return x
    m = re.search(r"log_x\((?:1/)?(\d+)\) = (-?\d+)", text)
    n, c = int(m.group(1)), int(m.group(2))
    target = Fraction(1, n) if "1/" in m.group(0) else Fraction(n)
    found = [b for b in range(2, 400) if Fraction(b) ** c == target]
    assert len(found) == 1
    return found[0]


def test_key_rederived_from_prompt():
    bad = [s for s in range(300) if _key_from_prompt(_exam(s).text) != _exam(s).answer]
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
