"""Tests the definite-integral-properties skill: keys re-derived from the stem by the rules the skill teaches."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from tests import r3b_support as sup

SKILL = "definite-integral-properties"
TOPIC = "r3b_integral_properties"
EXAM = "r3b_exam_integral_properties"
_GIVEN = re.compile(r"∫ from (-?\d+) to (-?\d+) of f\(x\) dx = (-?\d+)")


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "number", ["definite-integral"])


def _path_value(edges, start, end):
    """The integral from start to end by adding stated integrals, reversing the sign of one read backwards."""
    graph = {}
    for (a, b), v in edges.items():
        graph.setdefault(a, []).append((b, v))
        graph.setdefault(b, []).append((a, -v))
    seen, frontier = {start: 0}, [start]
    for _ in range(50):
        if not frontier:
            break
        nxt = []
        for node in frontier:
            for other, v in graph.get(node, []):
                if other not in seen:
                    seen[other] = seen[node] + v
                    nxt.append(other)
        frontier = nxt
    return seen[end]


def _linear(body):
    """(coefficient of f(x), coefficient of g(x), constant) of '(2f(x) + 5)' or 'f(x)'."""
    body = body.strip()
    body = body[1:-1] if body.startswith("(") and body.endswith(")") else body
    total = {"f": 0, "g": 0, "c": 0}
    for sign, coef, name in re.findall(r"([+-]?)\s*(\d*)\s*([fg])\(x\)", body):
        total[name] += (-1 if sign == "-" else 1) * int(coef or 1)
    rest = re.sub(r"[+-]?\s*\d*\s*[fg]\(x\)", "", body)
    for sign, num in re.findall(r"([+-]?)\s*(\d+)", rest):
        total["c"] += (-1 if sign == "-" else 1) * int(num)
    return total["f"], total["g"], total["c"]


def _derive_easy(text):
    edges = {(int(a), int(b)): int(v) for a, b, v in _GIVEN.findall(text)}
    asked = re.search(r"\. \w+ ∫ from (-?\d+) to (-?\d+) of f\(x\) dx\.$", text)
    return _path_value(edges, int(asked.group(1)), int(asked.group(2)))


def _half(kind, v, lo, hi):
    """Integral of an even or odd function over [lo, hi] when its integral over [0, a] is v, for lo = -a or 0, hi = a or 0."""
    if lo == 0:
        return v
    if hi == 0:
        return v if kind == "even" else -v
    return 2 * v if kind == "even" else 0


def _derive_medium(text):
    kinds = re.findall(r"(even|odd)", text.split("with")[0])
    values = [int(v) for v in re.findall(r"from 0 to \d+ of [fg]\(x\) dx = (-?\d+)", text)]
    lo, hi, body = re.search(r"\. \w+ ∫ from (-?\d+) to (-?\d+) of (.*) dx\.$", text).groups()
    lo, hi = int(lo), int(hi)
    mf, mg, c = _linear(body)
    total = mf * _half(kinds[0], values[0], lo, hi) + c * (hi - lo)
    if len(kinds) == 2:
        total += mg * _half(kinds[1], values[1], lo, hi)
    return total


def _derive_hard(text):
    lo, hi, body = re.search(r"\. \w+ ∫ from (-?\d+) to (-?\d+) of (.*) dx\.$", text).groups()
    lo, hi = int(lo), int(hi)
    period = re.search(r"of period (\d+)", text)
    if period:
        return Fraction(int(_GIVEN.search(text).group(3)) * (hi - lo), int(period.group(1)))
    reflect = re.search(r"f\((\d+) - x\) = f\(x\) for all x", text)
    if reflect:
        a, b, v = (int(t) for t in _GIVEN.search(text).groups())
        mf, _, c = _linear(body)
        assert (a, b) == (0, int(reflect.group(1)) // 2) and (lo, hi) == (0, int(reflect.group(1)))
        return 2 * v + c * (hi - lo)
    a, b, v = (int(t) for t in _GIVEN.search(text).groups())
    mirrored = re.search(r"f\((\d+) - x\)", body)
    if mirrored:
        assert (lo, hi) == (a, b) and int(mirrored.group(1)) == a + b
        coefs = [int(("-" if s == "-" else "") + (n or "1")) for s, n in re.findall(r"([+-]?)\s*(\d*)f\(", body)]
        return sum(coefs) * v
    shifted = re.search(r"f\(x ([+-]) (\d+)\)", body)
    shift = int(shifted.group(2)) * (1 if shifted.group(1) == "+" else -1)
    assert (lo + shift, hi + shift) == (a, b), text
    return v


def _derive(text):
    if text.startswith("The function f satisfies"):
        return _derive_easy(text)
    if re.match(r"Let f be an? (even|odd) function", text):
        return _derive_medium(text)
    return _derive_hard(text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_the_rules_applied_to_the_stem(tier):
    for seed in range(200):
        q = sup.tier_question(TOPIC, tier, seed)
        assert Fraction(q.answer_text) == _derive(q.text), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=10 if tier != "easy" else 8)


def test_tiers_differ():
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(60)]
    assert all("even" in t or "odd" in t for t in medium)
    assert (
        any("period" in t for t in hard)
        and any("- x)" in t for t in hard)
        and any("f(x +" in t or "f(x -" in t for t in hard)
    )
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number"] * 4, [1, 1, 2, 3], [False, False, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.08)


def test_exam_keys_match_the_rules_applied_to_the_stem():
    for seed in range(150):
        q = sup.exam_question(EXAM, seed)
        (lo1, hi1, v1), (lo2, hi2, v2) = [(int(a), int(b), int(v)) for a, b, v in _GIVEN.findall(q.text)]
        assert (lo1, hi1) == (0, lo2)
        whole_half = v1 + v2
        keys = [2 * v1, whole_half, 2 * whole_half]
        assert [int(pt.answer_text) for pt in q.parts[:3]] == keys, (seed, q.text)
        found = re.search(r"of \((-?\d*)f\(x\) ([+-]) (\d*)x ([+-]) (\d+)\) dx", q.parts[3].prompt)
        assert found, q.parts[3].prompt
        k = {"": 1, "-": -1}.get(found.group(1)) or int(found.group(1))
        n = int(found.group(5)) * (1 if found.group(4) == "+" else -1)
        width = 2 * hi2
        # the x term is odd, so it integrates to 0 over the symmetric interval
        assert int(q.parts[3].answer_text) == k * keys[2] + n * width, (seed, q.parts[3].prompt)
