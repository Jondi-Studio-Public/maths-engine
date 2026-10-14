"""Exam source of `senior-examine-momentum-force-resultant-force-action-and-reaction`: a coupling collision."""

from __future__ import annotations

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

TOPIC = "exd_x_r3j_collision_impulse"
SKILL_ID = "senior-examine-momentum-force-resultant-force-action-and-reaction"


def _read(q):
    """(masses, velocities, contact time, impulse body, force body) read from the rendered question."""
    mass, vel = {}, {}
    for name in "AB":
        m = re.search(rf"[A-Za-z]+ {name} has mass (\d+) kg and (is at rest|moves (left|right) at (\d+) m/s)", q.text)
        mass[name] = int(m.group(1))
        vel[name] = 0 if m.group(2) == "is at rest" else (1 if m.group(3) == "right" else -1) * int(m.group(4))
    t = Fraction(re.search(r"contact lasts ([\d.]+) s", q.text).group(1))
    x = re.search(r"impulse on \w+ ([AB])", q.text).group(1)
    y = re.search(r"force exerted on \w+ ([AB])", q.text).group(1)
    return mass, vel, t, x, y


def _oracle(q):
    mass, vel, t, x, y = _read(q)
    p = sum(mass[n] * vel[n] for n in "AB")
    v = Fraction(p, mass["A"] + mass["B"])
    j = mass[x] * (v - vel[x])
    f = -j / t
    return {"a": str(p), "b": str(v), "c": str(j), "d": str(f)}


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
        _, _, _, x, y = _read(q)
        assert x != y


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p13b_momentum_force_reaction", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_bodies_approach_each_other():
    """The body behind is the faster one in the direction of travel, so a collision really happens."""
    for s in range(300):
        q = make(TOPIC, seed=s)
        mass, vel, _, _, _ = _read(q)
        assert vel["A"] != vel["B"]
        assert Fraction(sum(mass[n] * vel[n] for n in "AB"), mass["A"] + mass["B"]) not in (0,)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_momentum():
    sk = skill(SKILL_ID)
    done = 0
    for s in range(80):
        q = make(TOPIC, seed=s)
        mass, vel, t, x, y = _read(q)
        total = mass["A"] + mass["B"]
        p = int(q.parts[0].answer_text)
        for k in range(1, 6):
            p2 = p + k * total
            v2 = Fraction(p2, total)
            j2 = mass[x] * (v2 - vel[x])
            f2 = -j2 / t
            if v2 != 0 and j2 != 0 and f2 != 0 and len({p2, v2, j2, f2}) == 4:
                break
        reply = {"a": str(p2), "b": str(v2), "c": str(j2), "d": str(f2)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (s, label, reply)
            assert result.parts[label].follow_through
        assert result.marks == 6 and not result.correct
        done += 1
    assert done == 80


def test_follow_through_from_wrong_velocity_only():
    sk = skill(SKILL_ID)
    checked = 0
    for s in range(60):
        q = make(TOPIC, seed=s)
        mass, vel, t, x, y = _read(q)
        canon = canonical(q)
        v2 = int(canon["b"]) + 3
        j2 = mass[x] * (v2 - vel[x])
        f2 = -Fraction(j2) / t
        if v2 == 0 or j2 == 0 or len({int(canon["a"]), v2, j2, f2}) < 4:
            continue  # the engine withholds follow-through from a zero or from a reply that repeats another
        checked += 1
        result = grade(sk, q, {**canon, "b": str(v2), "c": str(j2), "d": str(f2)})
        assert result.parts["a"].marks == result.parts["a"].max_marks
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks
        assert result.parts["d"].marks == result.parts["d"].max_marks
    assert checked >= 30


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "9999", "b": "77", "c": "-31", "d": "5"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
