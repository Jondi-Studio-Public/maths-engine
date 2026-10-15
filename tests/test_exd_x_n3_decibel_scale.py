"""Exam source of `f10-logarithmic-scales`: a decibel scale read as a ratio, an intensity and a level."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n3_decibel_scale"
SKILL_ID = "f10-logarithmic-scales"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _read(q):
    l1, c = re.search(r"is (\d+) dB and has an intensity of ([\d.]+) microwatts", q.stem).groups()
    l2 = re.search(r"is (\d+) dB\.$", q.stem).group(1)
    times = int(re.search(r"is (\d+) times as intense", q.parts[2].prompt).group(1))
    return int(l1), Fraction(c), int(l2), times


def oracle(q):
    l1, c, l2, times = _read(q)
    steps = Fraction(l2 - l1, 10)
    assert steps.denominator == 1
    ratio = 10 ** int(steps)
    j = len(str(times)) - 1
    assert times == 10**j
    return [ratio, c * ratio, l2 + 10 * j]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert [Fraction(p.answer_text) for p in q.parts] == oracle(q), (seed, q.text)


def test_grader_accepts_key_and_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert p.distractors
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_follow_through_from_wrong_ratio():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        _, c, _, _ = _read(q)
        wrong = int(q.parts[0].distractors[0])
        b_wrong = c * wrong
        reply = str(int(b_wrong)) if b_wrong.denominator == 1 else f"{float(b_wrong):.1f}"
        result = grade(_SKILL, q, {"a": str(wrong), "b": reply})
        assert result.parts["a"].marks == 0
        if reply != q.parts[1].answer_text and wrong != int(b_wrong):
            assert result.parts["b"].marks == result.parts["b"].max_marks and result.parts["b"].follow_through, (
                seed,
                q.text,
                wrong,
                reply,
            )
            checked += 1
    assert checked >= 80
