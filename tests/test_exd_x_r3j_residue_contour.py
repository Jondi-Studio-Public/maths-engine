"""Exam source of `ucv1-residue-simple-pole`: three residues, then the contour integral by the residue theorem."""

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

TOPIC = "exd_x_r3j_residue_contour"
SKILL_ID = "ucv1-residue-simple-pole"
z = sympy.Symbol("z")


def _oracle(q):
    """Residues as limits of (z - p) f(z), and k by the poles inside the circle read from the stem."""
    body = re.search(r"f\(z\) = (.*?) has simple poles", q.stem).group(1)
    f = sympy.parse_expr(body.replace("^", "**"), transformations="all")
    poles = [int(t) for t in re.findall(r"-?\d+", re.search(r"poles at z = (.*?)\.", q.stem).group(1))]
    circle = re.search(r"\|(.*?)\| = ([\d.]+)", q.stem)
    centre = sympy.sympify(circle.group(1).replace(" ", "")).subs(z, 0)
    centre = -centre  # |z - 3| has centre 3: the expression at z = 0 is -3
    radius = Fraction(circle.group(2))
    residues = [sympy.cancel(f * (z - p)).subs(z, p) for p in poles]
    k = sum(r for p, r in zip(poles, residues) if abs(Fraction(p) - Fraction(int(centre))) < radius)
    return {"a": str(residues[0]), "b": str(residues[1]), "c": str(residues[2]), "d": str(k)}


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.marks for p in q.parts] == [1, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p13e_residue_simple_pole", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_circle_separates_the_poles():
    for s in range(300):
        q = make(TOPIC, seed=s)
        names = q.parts[3].follow_through.split(" + ")
        assert 1 <= len(names) <= 2


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_residues():
    """A wrong residue used correctly in the sum earns the contour-integral marks."""
    sk = skill(SKILL_ID)
    done = 0
    for s in range(80):
        q = make(TOPIC, seed=s)
        names = [n[-1] for n in q.parts[3].follow_through.split(" + ")]
        canon = canonical(q)
        reply = dict(canon)
        first = names[0]
        wrong = Fraction(canon[first]) + Fraction(1, 7)
        reply[first] = str(wrong)
        k = sum(Fraction(reply[n]) for n in names)
        if k == 0 or str(k) in (reply["a"], reply["b"], reply["c"]):
            continue
        reply["d"] = str(k)
        result = grade(sk, q, reply)
        assert result.parts[first].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (s, reply)
        done += 1
    assert done >= 50


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "100", "b": "200", "c": "300", "d": "400"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
