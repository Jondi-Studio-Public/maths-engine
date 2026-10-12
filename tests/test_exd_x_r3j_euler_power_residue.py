"""Exam source of `ucv1-number-theory-computation`: factor n, find phi(n), reduce the exponent, then the power."""

from __future__ import annotations

import re

import sympy
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

TOPIC = "exd_x_r3j_euler_power_residue"
SKILL_ID = "ucv1-number-theory-computation"


def _read(q):
    n = int(re.search(r"n = (\d+)", q.stem).group(1))
    m, e = (int(v) for v in re.search(r"m = (\d+) and e = (\d+)", q.stem).groups())
    return n, m, e


def _oracle(q):
    n, m, e = _read(q)
    factors = sympy.factorint(n)
    assert len(factors) == 2 and set(factors.values()) == {1}
    phi = int(sympy.totient(n))
    return {
        "a": str(max(factors)),
        "b": str(phi),
        "c": str(e % phi),
        "d": str(pow(m, e, n)),  # straight from the definition, not through the reduced exponent
    }


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p14b_number_theory_computation", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_base_is_coprime_and_exponent_large():
    for s in range(300):
        q = make(TOPIC, seed=s)
        n, m, e = _read(q)
        assert sympy.gcd(m, n) == 1 and e > 100


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_totient():
    sk = skill(SKILL_ID)
    done = 0
    for s in range(120):
        q = make(TOPIC, seed=s)
        n, m, e = _read(q)
        canon = canonical(q)
        phi2 = int(canon["b"]) - 7
        r2 = e % phi2
        if r2 < 2:
            continue
        final = pow(m, r2, n)
        if len({int(canon["a"]), phi2, r2, final}) < 4 or final == 0:
            continue
        result = grade(sk, q, {**canon, "b": str(phi2), "c": str(r2), "d": str(final)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through
        done += 1
    assert done >= 50


def test_follow_through_from_wrong_prime_factor():
    """Taking the smaller prime as the 'larger' factor leaves phi(n) unchanged, so (b) stays right."""
    sk = skill(SKILL_ID)
    for s in range(40):
        q = make(TOPIC, seed=s)
        n, m, e = _read(q)
        canon = canonical(q)
        small = n // int(canon["a"])
        result = grade(sk, q, {**canon, "a": str(small)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["c"].marks == 1 and result.parts["d"].marks == 2


def test_wild_reduced_exponent_is_not_evaluated():
    """A huge reply to (c) must neither hang the grader nor earn follow-through marks in (d)."""
    sk = skill(SKILL_ID)
    q = make(TOPIC, seed=2)
    result = grade(sk, q, {**canonical(q), "c": "999999999", "d": "5"})
    assert result.parts["d"].marks == 0


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "1", "b": "2", "c": "3", "d": "4"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
