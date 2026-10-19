"""Exam source of `hypergeometric-probability`: an acceptance rule applied to a sample drawn without replacement."""

from __future__ import annotations

import re
from fractions import Fraction
from math import comb

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u6_hypergeometric_acceptance"
SKILL_ID = "hypergeometric-probability"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(q):
    N, D, n = (int(v) for v in re.findall(r"\d+", q.stem))
    total = comb(N, n)
    # Count samples by the number of faulty items directly, not by the formulas used in the generator.
    by_k = {k: Fraction(comb(D, k) * comb(N - D, n - k), total) for k in range(0, min(D, n) + 1)}
    assert sum(by_k.values()) == 1
    rejected = sum(v for k, v in by_k.items() if k >= 2)
    return {"a": by_k[0], "b": by_k[1], "c": rejected}


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q)
        for p in q.parts:
            assert Fraction(p.answer_text) == expected[p.label], (seed, q.text, p.label)


def test_every_option_is_a_probability():
    for seed in range(100):
        for p in _exam(seed).parts:
            assert p.sensible_range == (0, 1)
            assert 0 < Fraction(p.answer_text) < 1
            for d in p.distractors:
                assert 0 <= Fraction(d) <= 1, (seed, d)


def test_grader_accepts_key_and_rejects_distractors():
    shown = 0
    for seed in range(100):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 300


def test_follow_through_from_a_wrong_part():
    checked = 0
    for seed in range(100):
        q = _exam(seed)
        a_wrong = Fraction(q.parts[0].distractors[0])
        b = Fraction(q.parts[1].answer_text)
        c_follow = 1 - a_wrong - b
        if not 0 < c_follow < 1 or c_follow in (a_wrong, b):
            continue
        reply = {"a": q.parts[0].distractors[0], "b": q.parts[1].answer_text, "c": str(c_follow)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (seed, reply)
        assert result.parts["c"].follow_through
        checked += 1
    assert checked >= 30
