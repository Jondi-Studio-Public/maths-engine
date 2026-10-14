"""Exam tier of `prime-factorisation-exponent`: exponents of primes in a product of composite bases, then a factor count."""

from __future__ import annotations

import re
from math import prod

from sympy import factorint

from mathgen.engine import grade
from tests.exd_x_sc6_support import (
    assert_answer_space,
    assert_follow_through,
    assert_grading,
    assert_new_shapes,
    assert_no_text_defects,
    assert_source_wired,
    canonical,
    exam,
    skill_of,
)

SKILL = skill_of("prime-factorisation-exponent")
TOPIC = "exd_x_sc6_prime_powers"
_SUP = {c: str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}


def _factors(stem: str) -> list[tuple[int, int]]:
    body = stem.split("N = ", 1)[1].rstrip(".")
    out = []
    for term in body.split(" × "):
        m = re.fullmatch(r"(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]*)", term)
        assert m, term
        out.append((int(m.group(1)), int("".join(_SUP[c] for c in m.group(2))) if m.group(2) else 1))
    return out


def oracle(q) -> dict[str, str]:
    n = prod(b**e for b, e in _factors(q.stem))
    primes = factorint(n)
    asked = [int(m) for m in re.findall(r"exponent of (\d+)", " ".join(p.prompt for p in q.parts))]
    assert asked == sorted(primes)[:2]
    return {
        "a": str(primes[asked[0]]),
        "b": str(primes[asked[1]]),
        "c": str(prod(e + 1 for e in primes.values())),
    }


def test_chain_shape():
    for seed in range(60):
        q = exam(SKILL, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert len(factorint(prod(b**e for b, e in _factors(q.stem)))) == 3


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(SKILL, seed)
        assert oracle(q) == canonical(q), (seed, q.stem)


def test_grader_accepts_key_rejects_distractors():
    assert_grading(SKILL)


def test_follow_through_from_wrong_exponents():
    checked = 0
    for seed in range(120):
        q = exam(SKILL, seed)
        key = canonical(q)
        e3_plus_1 = int(key["c"]) // ((int(key["a"]) + 1) * (int(key["b"]) + 1))
        wrong_a = q.parts[0].distractors[0]
        c_after_a = (int(wrong_a) + 1) * (int(key["b"]) + 1) * e3_plus_1
        assert_follow_through(SKILL, q, {"a": wrong_a, "b": key["b"], "c": str(c_after_a)}, "a")
        wrong_b = q.parts[1].distractors[0]
        c_after_b = (int(key["a"]) + 1) * (int(wrong_b) + 1) * e3_plus_1
        assert_follow_through(SKILL, q, {"a": key["a"], "b": wrong_b, "c": str(c_after_b)}, "b")
        checked += 1
    assert checked == 120


def test_all_wrong_earns_nothing():
    for seed in range(30):
        result = grade(SKILL, exam(SKILL, seed), {"a": "99", "b": "98", "c": "97"})
        assert result.marks == 0


def test_deterministic():
    assert exam(SKILL, 7).to_dict() == exam(SKILL, 7).to_dict()
