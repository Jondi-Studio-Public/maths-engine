"""Tests the skill `proof-by-induction-divisibility`.

The oracle reads the statement f(n) = r1^(m1 n + s1) +/- r2^(m2 n + s2) and the divisor back out of the rendered stem,
then recomputes every key with plain integers: f(1) / d, f(k + 1) - v f(k) and f(k + 1) - c f(k) over a range of k, and
the exam chain's four values. It never reuses the generator's construction.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy import Symbol

from mathgen import make
from mathgen.engine import grade
from tests.r4f_support import (
    canonical,
    check_answer_space_single,
    check_exam_clean,
    check_exam_oracle,
    check_exam_shape,
    check_single_clean,
    check_tiers_differ,
    check_wired,
    exam_questions,
    parse,
    skill,
)

SKILL_ID = "proof-by-induction-divisibility"
TOPIC = "r4f_induction_divisibility"
EXAM = "exd_x_r4f_induction_divisibility"
k = Symbol("k")
_POWER = r"(\d+)\^(?:\(([^)]*)\)|n)"
_STATEMENT = re.compile(rf"f\(n\) = {_POWER} ([+-]) {_POWER} is divisible by (\d+)")


def _exponent(text):
    if text is None:
        return 1, 0
    m = re.fullmatch(r"(\d*)n(?: \+ (\d+))?", text)
    assert m, text
    return int(m.group(1) or 1), int(m.group(2) or 0)


def statement(text):
    """(f as a function of n, divisor, (r1, m1, s1), (r2, m2, s2), sign)."""
    m = _STATEMENT.search(text)
    assert m, text
    r1, e1, sign, r2, e2, d = m.groups()
    (m1, s1), (m2, s2) = _exponent(e1), _exponent(e2)
    sgn = 1 if sign == "+" else -1
    f = lambda n: int(r1) ** (m1 * n + s1) + sgn * int(r2) ** (m2 * n + s2)  # noqa: E731
    return f, int(d), (int(r1), m1, s1), (int(r2), m2, s2), sgn


def test_wired():
    check_wired(SKILL_ID, TOPIC, EXAM)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_single_tiers_grade_cleanly(tier):
    check_single_clean(SKILL_ID, TOPIC, tier, n=50)


def test_tiers_differ_in_shape():
    check_tiers_differ(TOPIC)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    check_answer_space_single(TOPIC, tier, n=200, floor=25)


def test_statements_are_true_and_not_the_plain_forms():
    """Every statement is divisible for n = 1..15, and each term carries a shift or a multiplier in its exponent."""
    for tier in ("easy", "medium", "hard"):
        for q in (make(TOPIC, seed=s, difficulty=tier) for s in range(80)):
            f, d, (r1, m1, s1), (r2, m2, s2), _ = statement(q.text)
            assert d >= 3
            assert all(f(n) % d == 0 and f(n) > 0 for n in range(1, 16)), q.text
            assert (m1, s1) != (1, 0) and (m2, s2) != (1, 0), q.text


def test_easy_key_is_f1_over_d():
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="easy")
        f, d, *_ = statement(q.text)
        assert f(1) % d == 0
        assert Fraction(q.answer_text) == f(1) // d, q.text


def test_medium_key_is_the_remainder():
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="medium")
        f, d, *_ = statement(q.text)
        v = int(re.search(r"f\(k \+ 1\) = (\d+)f\(k\) \+ R", q.text).group(1))
        key = parse(q.answer_text)
        for kk in range(1, 7):
            assert key.subs(k, kk) == f(kk + 1) - v * f(kk), (q.text, kk)


def test_hard_key_is_the_multiple():
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="hard")
        f, d, *_ = statement(q.text)
        c, dd = (int(x) for x in re.search(r"f\(k \+ 1\) = (\d+)f\(k\) \+ (\d+)M", q.text).groups())
        assert dd == d
        key = parse(q.answer_text)
        for kk in range(1, 7):
            assert key.subs(k, kk) * d == f(kk + 1) - c * f(kk), (q.text, kk)
            assert (f(kk + 1) - c * f(kk)) % d == 0


def test_hard_uses_both_routes():
    cs = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        f, d, (r1, m1, s1), (r2, m2, s2), _ = statement(q.text)
        c = int(re.search(r"f\(k \+ 1\) = (\d+)f\(k\)", q.text).group(1))
        cs.add("first" if c == r1**m1 else "second" if c == r2**m2 else "other")
    assert cs == {"first", "second"}


def oracle(q):
    f, d, (r1, m1, s1), (r2, m2, s2), sign = statement(q.stem)
    v = int(re.search(r"f\(k \+ 1\) = (\d+)f\(k\) \+ m × ", q.parts[1].prompt).group(1))
    t2 = int(re.search(r"= (\d+)\. Use your", q.parts[2].prompt).group(1))
    f2 = int(re.search(r"f\(2\) = (\d+)", q.parts[2].prompt).group(1))
    assert f2 == f(2) and t2 == r1 ** (m1 * 2 + s1)
    ms = {(f(kk + 1) - v * f(kk)) // r1 ** (m1 * kk + s1) for kk in range(1, 6)}
    assert len(ms) == 1
    (m,) = ms
    assert f(3) % d == 0
    return {"a": f(1), "b": m, "c": v * f(2) + m * t2, "d": f(3) // d}


def test_exam_clean_and_graded():
    check_exam_clean(SKILL_ID, EXAM, n=40)


def test_exam_key_rederived_from_stem():
    check_exam_oracle(EXAM, oracle, n=200)


def test_exam_answer_space():
    check_exam_shape(EXAM, floor_distinct=100, n=300)


def test_exam_parts_shape():
    for q in exam_questions(EXAM, 20):
        assert [p.marks for p in q.parts] == [1, 2, 2, 1]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_exam_follow_through_from_a_wrong_m():
    """A wrong m carried correctly into f(3) and the quotient earns the later parts as follow-through."""
    sk = skill(SKILL_ID)
    done = 0
    for q in exam_questions(EXAM, 60):
        f, d, (r1, m1, s1), (r2, m2, s2), sign = statement(q.stem)
        v = int(re.search(r"f\(k \+ 1\) = (\d+)f\(k\)", q.parts[1].prompt).group(1))
        wrong_m = int(q.parts[1].distractors[0])
        t2 = r1 ** (m1 * 2 + s1)
        f3 = v * f(2) + wrong_m * t2
        reply = {"a": q.parts[0].answer_text, "b": str(wrong_m), "c": str(f3), "d": str(Fraction(f3, d))}
        if Fraction(f3, d).denominator != 1:
            reply["d"] = str(float(Fraction(f3, d)))
        got = grade(sk, q, reply)
        assert got.parts["b"].marks == 0
        assert got.parts["c"].marks == got.parts["c"].max_marks and got.parts["c"].follow_through, q.text
        done += 1
    assert done >= 40


def test_exam_follow_through_from_a_wrong_f3():
    """A wrong but plausible f(3) (positive, so inside the part's sensible range) divided correctly is carried."""
    sk = skill(SKILL_ID)
    done = 0
    for q in exam_questions(EXAM, 80):
        f, d, *_ = statement(q.stem)
        for text in q.parts[2].distractors:
            wrong = int(text)
            if wrong <= 0 or wrong % d:
                continue
            got = grade(sk, q, canonical(q) | {"c": text, "d": str(wrong // d)})
            assert got.parts["c"].marks == 0
            assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text
            done += 1
            break
    assert done >= 10
