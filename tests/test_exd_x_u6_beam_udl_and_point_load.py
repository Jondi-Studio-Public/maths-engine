"""Exam source of `bending-moment-point-load`: reaction and bending moments under a UDL plus a point load."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u6_beam_udl_and_point_load"
SKILL_ID = "bending-moment-point-load"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _loads(q):
    L = int(re.search(r"\d+", q.stem).group())
    w = int(re.search(r"(\d+) kN/m", q.stem).group(1))
    W = int(re.search(r"(\d+) kN(?!/)", q.stem).group(1))
    a = int(re.search(r"(\d+) m from", q.stem).group(1))
    return L, w, W, a


def _oracle(q):
    L, w, W, a = _loads(q)
    b = L - a
    # Moments about B for the reactions, then each section moment from the B side of the cut.
    r_a = Fraction(W * b + Fraction(w * L * L, 2), L)
    r_b = W + w * L - r_a
    return {
        "a": r_a,
        "b": r_b * b - Fraction(w * b * b, 2),
        "c": r_b * Fraction(L, 2) - Fraction(w * L * L, 8),
    }


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q)
        for p in q.parts:
            assert Fraction(p.answer_text) == expected[p.label], (seed, q.text, p.label)


def test_grader_accepts_key_and_rejects_distractors():
    shown = 0
    for seed in range(100):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 7
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 250


def test_follow_through_from_a_wrong_reaction():
    checked = 0
    for seed in range(200):
        q = _exam(seed)
        L, w, W, a = _loads(q)
        wrong = Fraction(q.parts[0].distractors[0])
        b_ft = wrong * a - Fraction(w * a * a, 2)
        c_ft = wrong * Fraction(L, 2) - Fraction(w * L * L, 8) - W * (Fraction(L, 2) - a)
        if b_ft <= 0 or c_ft <= 0 or len({wrong, b_ft, c_ft}) < 3:
            continue
        reply = {"a": q.parts[0].distractors[0], "b": f"{float(b_ft):.1f}", "c": f"{float(c_ft):.1f}"}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (seed, label, reply)
        checked += 1
    assert checked >= 30
