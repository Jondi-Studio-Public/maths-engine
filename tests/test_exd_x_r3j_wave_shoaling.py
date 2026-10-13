"""Exam source of `ucv1-wave-parameter-relation`: frequency, deep-water speed, shallow wavelength, wave count."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from exd_r3j_support import (
    canonical,
    check_answer_space,
    check_deterministic,
    check_distractors,
    check_full_marks,
    check_shapes,
    check_source_wired,
    skill,
)

from mathgen import make
from mathgen.engine import grade

TOPIC = "exd_x_r3j_wave_shoaling"
SKILL_ID = "ucv1-wave-parameter-relation"


def _read(q):
    n, t, spacing = (
        int(v) for v in re.search(r"falls (\d+) times in (\d+) seconds.*?crests are (\d+) m apart", q.stem).groups()
    )
    v2 = Fraction(re.search(r"speed falls to ([\d.]+) m/s", q.stem).group(1))
    section = int(re.search(r"section is (\d+) m long", q.stem).group(1))
    return n, t, spacing, v2, section


def _dp2(x: Fraction) -> str:
    scaled = math.floor(x * 100 + Fraction(1, 2))
    return f"{scaled // 100}.{scaled % 100:02d}"


def _oracle(q):
    n, t, spacing, v2, section = _read(q)
    f = Fraction(n, t)
    v1 = f * spacing
    lam2 = v2 / f  # the frequency is unchanged, so lambda = v / f in the second medium
    return {"a": _dp2(f), "b": _dp2(v1), "c": _dp2(lam2), "d": str(math.floor(section / lam2))}


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.marks for p in q.parts] == [1, 1, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p14f_wave_parameter_relation", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_shallow_speed_is_below_deep_speed():
    for s in range(300):
        q = make(TOPIC, seed=s)
        n, t, spacing, v2, _ = _read(q)
        assert 0 < v2 < Fraction(n, t) * spacing


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_frequency():
    sk = skill(SKILL_ID)
    done = 0
    for s in range(100):
        q = make(TOPIC, seed=s)
        n, t, spacing, v2, section = _read(q)
        f_typed = Fraction(_dp2(Fraction(n, t))) + Fraction(1, 20)
        v1_typed = Fraction(_dp2(f_typed * spacing))
        lam_typed = Fraction(_dp2(spacing * v2 / v1_typed))
        count = math.floor(section / lam_typed)
        reply = {"a": _dp2(f_typed), "b": _dp2(v1_typed), "c": _dp2(lam_typed), "d": str(count)}
        if len(set(reply.values())) < 4:
            continue
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (s, label, reply)
            if reply[label] != canonical(q)[label]:
                assert result.parts[label].follow_through, (s, label)
        done += 1
    assert done >= 60


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "9.99", "b": "8.88", "c": "7.77", "d": "99"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
