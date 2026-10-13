"""Exam source of `ucv1-gram-schmidt-2d`: the QR factorisation of a 2 x 2 matrix, entry by entry."""

from __future__ import annotations

import re
from fractions import Fraction

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

TOPIC = "exd_x_r3j_gram_schmidt_qr"
SKILL_ID = "ucv1-gram-schmidt-2d"
_INT = r"(-?\d+)"


def _oracle(q) -> dict[str, str]:
    """Re-derive every part from the stem with sympy's QR decomposition."""
    a, b, p, q2 = (int(v) for v in re.search(rf"u = \({_INT}, {_INT}\).*?v = \({_INT}, {_INT}\)", q.stem).groups())
    comp = re.search(r"Find the ([xy])-component of e2", q.text).group(1)
    big_q, big_r = sympy.Matrix([[a, p], [b, q2]]).QRdecomposition()
    assert big_r[0, 0] > 0 and big_r[1, 1] > 0
    e2 = big_q[0 if comp == "x" else 1, 1]
    return {"a": str(big_r[0, 0]), "b": str(big_r[0, 1]), "c": str(big_r[1, 1]), "d": str(e2)}


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p12f_gram_schmidt_2d", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_length():
    """A wrong r11 carried through (b), (c) and (d) earns those parts."""
    sk = skill(SKILL_ID)
    checked = 0
    for s in range(60):
        q = make(TOPIC, seed=s)
        a, b, p, q2 = (int(v) for v in re.search(rf"u = \({_INT}, {_INT}\).*?v = \({_INT}, {_INT}\)", q.stem).groups())
        bad = Fraction(q.parts[0].answer_text) + 1
        dot, det = a * p + b * q2, a * q2 - b * p
        r12 = Fraction(dot) / bad
        r22 = Fraction(abs(det)) / bad
        use_x = "x-component" in q.parts[3].prompt
        v_c, u_c = (p, a) if use_x else (q2, b)
        e2 = (v_c - r12 * u_c / bad) / r22
        reply = {"a": str(bad), "b": str(r12), "c": str(r22), "d": str(e2)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (s, label, reply)
            assert result.parts[label].follow_through
        assert result.marks == 6 and not result.correct
        checked += 1
    assert checked == 60


def test_all_wrong_earns_nothing():
    sk = skill(SKILL_ID)
    q = make(TOPIC, seed=3)
    assert grade(sk, q, {"a": "1000", "b": "1/7", "c": "2/9", "d": "5/11"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
