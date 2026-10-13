"""Tests the skill `proof-by-induction-inequality-or-recurrence`.

The oracle reads the statement back out of the rendered stem and finds the thresholds by brute force with Python
integers (the first n from which L(n) > R(n) for ever, the first k from which the key inequality holds for ever), and
for a recurrence it recovers A, p and r by solving a 3 x 3 system on the first three terms of the sequence the stem
defines, not by using the generator's construction.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from tests.r4f_support import (
    canonical,
    check_answer_space_single,
    check_exam_clean,
    check_exam_oracle,
    check_exam_shape,
    check_single_clean,
    check_wired,
    exam_questions,
    skill,
)

SKILL_ID = "proof-by-induction-inequality-or-recurrence"
TOPIC = "r4f_induction_inequality"
EXAM = "exd_x_r4f_induction_inequality"
_SUP = {"²": 2, "³": 3, "⁴": 4, "⁵": 5}
_LIMIT = 300


def parse_side(text: str):
    """A function of n from '2^n', 'n!', '3n²', '5n + 3', 'n⁴'."""
    text = text.strip()
    if text == "n!":
        return lambda n: math.factorial(n)
    m = re.fullmatch(r"(\d+)\^n", text)
    if m:
        return lambda n, b=int(m.group(1)): b**n
    m = re.fullmatch(r"(\d*)n([²³⁴⁵])", text)
    if m:
        return lambda n, c=int(m.group(1) or 1), p=_SUP[m.group(2)]: c * n**p
    m = re.fullmatch(r"(\d+)n(?: \+ (\d+))?", text)
    assert m, text
    return lambda n, c=int(m.group(1)), e=int(m.group(2) or 0): c * n + e


def parse_statement(text: str):
    m = re.search(r"(?:P\(n\):|statement) (\S+) > ([^.]+?)\.", text)
    assert m, text
    return parse_side(m.group(1)), parse_side(m.group(2)), m.group(1)


def thresholds(text: str):
    left, right, left_text = parse_statement(text)
    n0 = max(n for n in range(1, _LIMIT) if not left(n) > right(n)) + 1
    mult = (lambda kk: int(left_text.split("^")[0])) if "^" in left_text else (lambda kk: kk + 1)
    k0 = max(kk for kk in range(1, _LIMIT) if not mult(kk) * right(kk) >= right(kk + 1)) + 1
    # the step is genuinely L(k + 1) = mult(k) L(k)
    assert all(left(kk + 1) == mult(kk) * left(kk) for kk in range(1, 12))
    return left, right, mult, n0, k0


def test_wired():
    check_wired(SKILL_ID, TOPIC, EXAM)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_single_tiers_grade_cleanly(tier):
    check_single_clean(SKILL_ID, TOPIC, tier, n=50)


def test_easy_key_is_the_last_failure_plus_one():
    seen_not_first = False
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="easy")
        left, right, _mult, n0, _k0 = thresholds(q.text)
        assert Fraction(q.answer_text) == n0, q.text
        assert all(left(n) > right(n) for n in range(n0, n0 + 40))
        assert not left(n0 - 1) > right(n0 - 1)
        seen_not_first |= left(1) > right(1) and n0 > 1
    assert seen_not_first, "no easy statement is true at n = 1 yet starts later"


def test_medium_key_is_the_step_threshold():
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="medium")
        left, right, mult, n0, k0 = thresholds(q.text)
        assert Fraction(q.answer_text) == k0, q.text
        assert k0 <= n0 and k0 != n0  # the two thresholds are different numbers


def recurrence(text: str):
    m = re.search(r"u\(1\) = (-?\d+) and u\(n \+ 1\) = (\d+)u\(n\) ([+-]) ([^.]+?)\. ", text + " ")
    assert m, text
    u1, a, sign, forcing = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    mm = re.fullmatch(r"(-?\d*)n(?: ([+-]) (\d+))?|(\d+)", forcing)
    assert mm, forcing
    if mm.group(4) is not None:
        c1, c0 = 0, int(mm.group(4))
    else:
        c1 = {"": 1, "-": -1}.get(mm.group(1), None)
        c1 = int(mm.group(1)) if c1 is None else c1
        c0 = 0 if mm.group(3) is None else int(mm.group(3)) * (1 if mm.group(2) == "+" else -1)
    if sign == "-":  # the sign before the first term of the forcing expression
        if mm.group(4) is not None:
            c0 = -c0
        else:
            c1 = -c1
    return u1, a, c1, c0


def solve_closed_form(u1, a, c1, c0):
    """A, p, r with u(n) = A a^n + p n + r, from the first three terms of the stated sequence."""
    us = [u1]
    for n in range(1, 3):
        us.append(a * us[-1] + c1 * n + c0)
    A, p, r = sympy.symbols("A p r")
    sol = sympy.solve([A * a**n + p * n + r - us[n - 1] for n in (1, 2, 3)], [A, p, r])
    A_, p_, r_ = (Fraction(int(sol[x].p), int(sol[x].q)) for x in (A, p, r))
    # the closed form really satisfies the recurrence for many more terms
    u = u1
    for n in range(1, 12):
        assert u == A_ * a**n + p_ * n + r_
        u = a * u + c1 * n + c0
    return A_, p_, r_


def test_hard_key_is_r_or_A():
    asked = set()
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="hard")
        A, p, r = solve_closed_form(*recurrence(q.text))
        if q.text.endswith("Find r."):
            asked.add("r")
            assert Fraction(q.answer_text) == r, q.text
        else:
            assert q.text.endswith("Find the integer A.")
            asked.add("A")
            assert Fraction(q.answer_text) == A, q.text
        assert A.denominator == p.denominator == r.denominator == 1
    assert asked == {"r", "A"}


def test_tiers_use_different_prompts():
    texts = {
        t: {make(TOPIC, seed=s, difficulty=t).text.split(".")[0][:12] for s in range(40)}
        for t in ("easy", "medium", "hard")
    }
    assert "A sequence" in "".join(texts["hard"]) and not any(
        "A sequence" in x for x in texts["easy"] | texts["medium"]
    )


@pytest.mark.parametrize(("tier", "floor"), [("easy", 20), ("medium", 20), ("hard", 25)])
def test_answer_space(tier, floor):
    answers = {(make(TOPIC, seed=s, difficulty=tier).text) for s in range(300)}
    assert len(answers) >= floor


def oracle(q):
    stem = q.stem
    if stem.startswith("A sequence"):
        u1, a, c1, c0 = recurrence(stem)
        A, p, r = solve_closed_form(u1, a, c1, c0)
        u3 = a * (a * u1 + c1 + c0) + 2 * c1 + c0
        return {"a": u3, "b": p, "c": r, "d": A}
    left, right, mult, n0, k0 = thresholds(stem)
    return {
        "a": n0,
        "b": left(n0) - right(n0),
        "c": k0,
        "d": mult(n0) * right(n0) - right(n0 + 1),
    }


def test_exam_clean_and_graded():
    check_exam_clean(SKILL_ID, EXAM, n=40)


def test_exam_key_rederived_from_stem():
    check_exam_oracle(EXAM, oracle, n=300)


def test_exam_answer_space():
    check_exam_shape(EXAM, floor_distinct=60, n=300, cap=0.12)


def test_exam_has_both_shapes():
    kinds = {q.stem.startswith("A sequence") for q in exam_questions(EXAM, 60)}
    assert kinds == {True, False}


def test_exam_parts_shape():
    for q in exam_questions(EXAM, 30):
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.marks for p in q.parts] in ([2, 1, 2, 1], [1, 2, 2, 1])
        assert sum(p.follow_through is not None for p in q.parts) >= 2


def test_inequality_follow_through_from_a_wrong_n0():
    """A wrong n0 carried correctly into the base-case difference and the key-inequality margin is marked as follow-through."""
    sk = skill(SKILL_ID)
    done = 0
    for q in exam_questions(EXAM, 120):
        if q.stem.startswith("A sequence"):
            continue
        left, right, mult, n0, k0 = thresholds(q.stem)
        wrong = n0 + 1
        reply = canonical(q) | {
            "a": str(wrong),
            "b": str(left(wrong) - right(wrong)),
            "d": str(mult(wrong) * right(wrong) - right(wrong + 1)),
        }
        got = grade(sk, q, reply)
        assert got.parts["a"].marks == 0
        for label in "bd":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, (
                q.text,
                label,
            )
        done += 1
    assert done >= 20


def test_recurrence_follow_through_from_a_wrong_p():
    sk = skill(SKILL_ID)
    done = 0
    for q in exam_questions(EXAM, 120):
        if not q.stem.startswith("A sequence"):
            continue
        u1, a, c1, c0 = recurrence(q.stem)
        A, p, r = solve_closed_form(u1, a, c1, c0)
        wrong_p = int(q.parts[1].distractors[0])
        r2 = Fraction(c0 - wrong_p, 1 - a)
        if r2.denominator != 1:
            continue
        A2 = Fraction(u1 - wrong_p - r2, a)
        if A2.denominator != 1:
            continue
        got = grade(sk, q, canonical(q) | {"b": str(wrong_p), "c": str(int(r2)), "d": str(int(A2))})
        assert got.parts["b"].marks == 0
        for label in "cd":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, (
                q.text,
                label,
            )
        done += 1
    assert done >= 5
