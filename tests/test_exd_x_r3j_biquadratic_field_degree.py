"""Exam source of `ucv1-field-extension-degree`: discriminant, root, then two field degrees for q(x^2)."""

from __future__ import annotations

import functools
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

TOPIC = "exd_x_r3j_biquadratic_field_degree"
SKILL_ID = "ucv1-field-extension-degree"
_X = sympy.Symbol("x")
_Y = sympy.Symbol("y")


@functools.lru_cache(maxsize=None)
def _degrees(u1: int, u2: int) -> tuple[int, int]:
    """[Q(sqrt u1):Q] and [Q(sqrt u1, sqrt u2):Q], the latter as the degree of the primitive element sqrt u1 + sqrt u2."""
    alpha = sympy.degree(sympy.minimal_polynomial(sympy.sqrt(u1), _X), _X)
    field = sympy.degree(sympy.minimal_polynomial(sympy.sqrt(u1) + sympy.sqrt(u2), _X), _X)
    return int(alpha), int(field)


def _oracle(q):
    s, t = (int(v) for v in re.search(r"x\^4 − (\d+)x\^2 \+ (\d+)", q.stem).groups())
    quadratic = _Y**2 - s * _Y + t
    roots = sorted(sympy.roots(quadratic, _Y), reverse=True)
    assert len(roots) == 2 and all(r.is_Integer and r > 0 for r in roots)
    u1, u2 = (int(r) for r in roots)
    alpha, field = _degrees(u1, u2)
    return {
        "a": str(sympy.discriminant(quadratic, _Y)),
        "b": str(u1),
        "c": str(alpha),
        "d": str(field),
    }


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, False]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p17a_field_extension_degree", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_every_degree_pattern_occurs():
    patterns = {(canonical(make(TOPIC, seed=s))["c"], canonical(make(TOPIC, seed=s))["d"]) for s in range(300)}
    # (1, 1) and (2, 2) are never drawn: (c) would equal (d) and one answer would serve both parts.
    assert patterns == {("1", "2"), ("2", "4")}


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_discriminant():
    """A wrong discriminant that is a perfect square gives a root by the quadratic formula."""
    sk = skill(SKILL_ID)
    done = 0
    for s in range(100):
        q = make(TOPIC, seed=s)
        coef = int(re.search(r"x\^4 − (\d+)x", q.stem).group(1))
        canon = canonical(q)
        wrong_disc = (int(canon["a"]) ** 0.5 + 2) ** 2
        wrong_disc = int(round(wrong_disc))
        root = (coef + (wrong_disc**0.5)) / 2
        if root != int(root) or int(root) in (int(canon["b"]), wrong_disc):
            continue
        result = grade(sk, q, {**canon, "a": str(wrong_disc), "b": str(int(root))})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through, (s, wrong_disc, root)
        done += 1
    assert done >= 30


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "7", "b": "9", "c": "3", "d": "5"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
