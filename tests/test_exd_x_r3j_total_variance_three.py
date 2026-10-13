"""Exam source of `ucv1-total-variance-finite`: E[Y], E[Var], Var(E), then Var(Y) over three unequal cases."""

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

TOPIC = "exd_x_r3j_total_variance_three"
SKILL_ID = "ucv1-total-variance-finite"


def _read(q):
    weights = [Fraction(int(v), 100) for v in re.findall(r"(\d+)%", q.stem)]
    pairs = [(int(m), int(v)) for m, v in re.findall(r"Y has mean (\d+) and variance (\d+)", q.stem)]
    assert len(weights) == 3 and len(pairs) == 3 and sum(weights) == 1
    return weights, [m for m, _ in pairs], [v for _, v in pairs]


def _oracle(q):
    w, mu, var = _read(q)
    e_y = sum(a * b for a, b in zip(w, mu))
    # Var(Y) from the second moment, not from the law being tested
    second = sum(a * (v + m * m) for a, m, v in zip(w, mu, var))
    total = second - e_y**2
    e_var = sum(a * v for a, v in zip(w, var))
    return {"a": _t(e_y), "b": _t(e_var), "c": _t(total - e_var), "d": _t(total)}


def _t(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.marks for p in q.parts] == [1, 1, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p14e_total_variance_two_case", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_probabilities_are_unequal():
    for s in range(300):
        weights, means, _ = _read(make(TOPIC, seed=s))
        assert len(set(weights)) >= 2 and len(set(means)) >= 2


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_mean():
    """A wrong E[Y] used correctly in Var(E[Y|X]) earns (c), and (d) then follows from (b) and the learner's (c)."""
    sk = skill(SKILL_ID)
    done = 0
    for s in range(80):
        q = make(TOPIC, seed=s)
        w, mu, _ = _read(q)
        canon = canonical(q)
        s2 = sum(a * m * m for a, m in zip(w, mu))
        for shift in (Fraction(-1, 3), Fraction(1, 3), Fraction(-1, 7), Fraction(1, 7)):
            bad_mean = Fraction(canon["a"]) + shift
            c2 = s2 - bad_mean**2
            d2 = Fraction(canon["b"]) + c2
            if c2 > 0 and len({bad_mean, c2, d2, Fraction(canon["b"])}) == 4:
                break
        else:
            continue
        result = grade(sk, q, {**canon, "a": _t(bad_mean), "c": _t(c2), "d": _t(d2)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through
        done += 1
    assert done >= 40


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "1", "b": "2/7", "c": "3", "d": "4"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
