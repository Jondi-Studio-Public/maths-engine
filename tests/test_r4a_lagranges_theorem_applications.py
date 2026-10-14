"""Tests `uc2-lagranges-theorem-applications`: every key is recomputed from the numbers in the question TEXT by brute
force over divisors, the typed grader on every key and distractor, and the four-part exam chain with follow-through."""

from __future__ import annotations

import re
from math import gcd

import mathgen.generators.r4a_lagranges_theorem_applications_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4a_support as sup

TOPIC = "r4a_lagranges_theorem_applications"
SKILL_ID = "uc2-lagranges-theorem-applications"


def _ints(text):
    return [int(v) for v in re.findall(r"\d+", text)]


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "number")


def test_easy_key():
    seen = set()
    for seed in range(120):
        question = sup.q(TOPIC, "easy", seed)
        nums = _ints(question.text)
        if "index" in question.text or "[G:H]" in question.text:
            a, b, c = nums[-3:] if "[G:H]" not in question.text else nums[:3]
            assert int(question.answer) == a * b * c, (seed, question.text)
            seen.add("tower")
        else:
            n, d = nums[:2]
            assert n % d == 0 and int(question.answer) == n // d, (seed, question.text)
            seen.add("element")
    assert seen == {"tower", "element"}


def test_medium_key():
    seen = set()
    for seed in range(120):
        question = sup.q(TOPIC, "medium", seed)
        nums = _ints(question.text)
        if "rule out" in question.text or "impossible" in question.text:
            n = max(nums)
            assert int(question.answer) == sum(1 for m in range(1, n + 1) if n % m), (seed, question.text)
            seen.add("ruled-out")
        else:
            n, a, b = nums[:3]
            assert gcd(a, b) == 1 and a * b <= n and int(question.answer) == a * b, (seed, question.text)
            seen.add("product")
    assert seen == {"ruled-out", "product"}


def test_hard_key():
    seen = set()
    for seed in range(160):
        question = sup.q(TOPIC, "hard", seed)
        nums = _ints(question.text)
        if "how many" in question.text.lower():
            n, a, b = nums[:3]
            possible = [d for d in _divisors(gcd(a, b)) if (a * b) % d == 0 and a * b // d <= n]
            assert int(question.answer) == len(possible), (seed, question.text)
            assert 2 <= len(possible) <= 5
            seen.add("count")
        else:
            m = nums[-1]
            assert int(question.answer) == max(d for d in _divisors(m) if d < m), (seed, question.text)
            seen.add("largest")
    assert seen == {"count", "largest"}


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "number")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 15, "medium": 12, "hard": 6})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (2, 2, 1, 3), ("number",) * 4, (0, 1, 1, 0))


def test_exam_keys_by_divisor_search():
    for seed in range(150):
        question = sup.exam(TOPIC, seed)
        n, h, t = _ints(question.stem)
        k = h * t
        a, b, c, d = question.parts
        assert int(a.answer) == n // h
        assert int(b.answer) == n // k
        assert int(c.answer) == k
        above = [m for m in _divisors(n) if m % k == 0 and k < m < n]
        assert int(d.answer) == max(above), (seed, question.stem)


def test_exam_follow_through_chain():
    s = sup.skill(SKILL_ID)
    checked = 0
    for seed in range(80):
        question = sup.exam(TOPIC, seed)
        n, h, t = _ints(question.stem)
        replies = {p.label: p.answer_text for p in question.parts}
        wrong_a = 2 * int(question.parts[0].answer)  # twice the true index; the quotients that follow are whole
        sent = replies | {"a": str(wrong_a)}
        b_ft = wrong_a // t
        sent["b"] = str(b_ft)
        if n % b_ft or n // b_ft in (b_ft, wrong_a):
            continue  # a repeated number is not follow-through
        sent["c"] = str(n // b_ft)
        result = grade(s, question, sent)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, sent)
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, sent)
        checked += 1
    assert checked >= 10, checked
