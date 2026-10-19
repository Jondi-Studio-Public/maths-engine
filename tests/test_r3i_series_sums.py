"""Tests the arithmetic-geometric-series-sums skill: keys re-derived from the stem, grading, and the exam chain."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen.engine import grade
from tests import r3i_support as sup

SKILL = "arithmetic-geometric-series-sums"
TOPIC = "r3i_series_sums"
EXAM = "r3i_exam_series_sums"


def _terms(text):
    return [int(v) for v in re.search(r"(-?\d+), (-?\d+), (-?\d+), \.\.\.", text).groups()]


def _running_sum(first, step, n, geometric):
    total, term = 0, first
    for _ in range(n):
        total += term
        term = term * step if geometric else term + step
    return total


def _derive(text):
    if "first three terms" in text or "sequence begins" in text:
        t1, t2, t3 = _terms(text)
        n = int(re.search(r"first (\d+) terms", text).group(1))
        if t2 - t1 == t3 - t2:
            return _running_sum(t1, t2 - t1, n, False)
        assert Fraction(t2, t1) == Fraction(t3, t2)
        return _running_sum(t1, t2 // t1, n, True)
    found = re.search(
        r"How many terms of the (arithmetic|geometric) series ([\d+ .]+) must be added to give a sum of (\d+)", text
    )
    if found:
        kind, series, target = found.group(1), found.group(2), int(found.group(3))
        t1, t2 = (int(v) for v in re.findall(r"\d+", series)[:2])
        step = t2 - t1 if kind == "arithmetic" else t2 // t1
        for n in range(1, 100):
            if _running_sum(t1, step, n, kind == "geometric") == target:
                return n
        raise AssertionError(text)
    found = re.search(r"S\(n\) = (\d)n² (?:([+-]) (\d+)n)?.* Find the (\d+)\w\w term", text)
    if found:
        p, sign, q, k = found.groups()
        p, q, k = int(p), (int(q) * (1 if sign == "+" else -1) if q else 0), int(k)
        return (p * k * k + q * k) - (p * (k - 1) ** 2 + q * (k - 1))
    found = re.search(
        r"first (\d+) terms of an arithmetic series is (-?\d+) and the sum of the first (\d+) terms is (-?\d+)\. Find the (\d+)",
        text,
    )
    if found:
        m, sm, n, sn, k = (int(v) for v in found.groups())
        a, d = sympy.symbols("a d")
        sol = sympy.solve([m * (2 * a + (m - 1) * d) / 2 - sm, n * (2 * a + (n - 1) * d) / 2 - sn], [a, d])
        return int(sol[a] + (k - 1) * sol[d])
    found = re.search(r"converges to a sum of (\d+)\. Its second term is (-?\d+)", text)
    if found:
        s, t2 = (int(v) for v in found.groups())
        a = sympy.symbols("a")
        roots = sympy.solve(a * a - s * a + s * t2, a)
        (good,) = [r for r in roots if abs(t2 / r) < 1]
        return int(good)
    raise AssertionError(text)


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "number",
        [
            "senior-use-arithmetic-sequences-in-contexts-involving-discrete-linear",
            "senior-use-geometric-sequences-in-contexts-involving-geometric-growth",
        ],
    )


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_derivation(tier):
    for seed in range(250):
        q = sup.tier_question(TOPIC, tier, seed)
        assert Fraction(q.answer_text) == _derive(q.text), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=20, top=0.1)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(80)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(80)]
    assert all("first three terms" in t or "sequence begins" in t for t in easy)
    assert any(t.startswith("How many terms") for t in medium) and any("S(n)" in t for t in medium)
    assert any("converges" in t for t in hard) and any("arithmetic series is" in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_easy_covers_both_series_kinds():
    texts = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    arithmetic = [t for t in texts if _terms(t)[1] - _terms(t)[0] == _terms(t)[2] - _terms(t)[1]]
    assert 10 < len(arithmetic) < len(texts) - 10


def test_hard_geometric_rejects_the_root_that_does_not_converge():
    seen = 0
    for seed in range(120):
        q = sup.tier_question(TOPIC, "hard", seed)
        if "converges" not in q.text:
            continue
        seen += 1
        s, t2 = (int(v) for v in re.search(r"sum of (\d+)\. Its second term is (-?\d+)", q.text).groups())
        a = int(q.answer_text)
        assert a / (1 - Fraction(t2, a)) == s and abs(Fraction(t2, a)) < 1
        assert str(s - a) in q.distractors  # the other root of the quadratic is offered, and it is wrong
    assert seen >= 10


def test_exam_shape_and_grading():
    for seed in range(40):
        q = sup.exam_question(EXAM, seed)
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, True] or [
            bool(p.follow_through) for p in q.parts
        ] == [False, True, True, False]
        assert all(p.sensible_range is not None for p in q.parts)
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.08)


def test_exam_keys_match_an_independent_derivation():
    kinds = set()
    for seed in range(150):
        q = sup.exam_question(EXAM, seed)
        keys = [Fraction(p.answer_text) for p in q.parts]
        if "savings plan" in q.text:
            kinds.add("ap")
            a, s10 = (int(v) for v in re.findall(r"\$(\d+)", q.text)[:2])
            d = Fraction(s10 - 10 * a, 45)
            k = int(re.search(r"in week (\d+)\.", q.parts[1].prompt).group(1))
            big = int(re.search(r"after (\d+) weeks", q.parts[2].prompt).group(1))
            target = int(re.search(r"exactly \$(\d+)", q.parts[3].prompt).group(1))
            n = next(n for n in range(1, 200) if n * (2 * a + (n - 1) * d) / 2 == target)
            assert keys == [d, a + (k - 1) * d, big * (2 * a + (big - 1) * d) / 2, n], (seed, q.text)
        else:
            kinds.add("gp")
            a, s = (int(v) for v in re.findall(r"(\d+)", q.text)[:2])
            r = 1 - Fraction(a, s)
            assert keys == [r, a * r**3, a * (1 + r + r * r), s - a * (1 + r + r * r)], (seed, q.text)
    assert kinds == {"ap", "gp"}


def test_exam_follow_through_from_a_wrong_first_part():
    sk = sup.skill_object(SKILL)
    found = 0
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        if "savings plan" not in q.text:
            continue
        canon = {p.label: p.answer_text for p in q.parts}
        a = int(re.findall(r"\$(\d+)", q.text)[0])
        wrong_d = int(canon["a"]) + 3
        k = int(re.search(r"in week (\d+)\.", q.parts[1].prompt).group(1))
        replies = {**canon, "a": str(wrong_d), "b": str(a + (k - 1) * wrong_d)}
        result = grade(sk, q, replies)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, replies)
        found += 1
    assert found >= 5
