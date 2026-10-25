"""Tests for `power-function-graph-features`: `vm_power_graph_features` (easy, medium, hard) and the exam chain.

The oracle reads the rule back out of the rendered text and decides each answer from sampled values of the real
function (real odd roots of negative numbers, no values where an even root of a negative number is wanted), by brute
search over integers, or with SymPy limits, never from the generator's own parity and quadrant rules.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
import sympy

import r4e_support as t
from r4e_support import sup

SKILL = "power-function-graph-features"
CHAIN = "vm_power_graph_features_exam"
x = sympy.Symbol("x")


def rpow(v: float, p: int, q: int):
    """Real value of v^(p/q), or None where it is not a real number."""
    if v == 0:
        return 0.0 if p > 0 else None
    if v > 0:
        return v ** (p / q)
    if q % 2 == 0:
        return None
    return (-1) ** p * (-v) ** (p / q)


def index(text: str):
    m = re.fullmatch(r"(-?\d+)(?:/(\d+))?", text)
    return int(m.group(1)), int(m.group(2) or 1)


def parse_rule(rule: str):
    """(a, h, p, q, k) from `a(x - h)^(p/q) + k` as the generator writes it."""
    m = re.fullmatch(r"(-?\d*)\(?x(?: ([-+]) (\d+)\))?\^\(?(-?\d+(?:/\d+)?)\)?(?: ([-+]) (\d+))?", rule.strip())
    assert m, rule
    a = {"": 1, "-": -1}.get(m.group(1), None)
    a = a if a is not None else int(m.group(1))
    h = 0 if not m.group(2) else (-1 if m.group(2) == "+" else 1) * int(m.group(3))
    p, q = index(m.group(4))
    k = 0 if not m.group(5) else (-1 if m.group(5) == "-" else 1) * int(m.group(6))
    return a, h, p, q, k


def evaluate(rule, v):
    a, h, p, q, k = rule
    r = rpow(v - h, p, q)
    return None if r is None else a * r + k


def quadrants(fn):
    found = set()
    for sx in (1, -1):
        for xv in (0.37, 1.5, 4.2, 11.0):
            yv = fn(sx * xv)
            if yv is None or yv == 0:
                continue
            found.add({(1, 1): 1, (-1, 1): 2, (-1, -1): 3, (1, -1): 4}[(sx, 1 if yv > 0 else -1)])
    return sorted(found)


def show_list(vs):
    return ", ".join(str(v) for v in vs)


def oracle(q):
    text = q.text
    m = re.match(r"Let [fy](?:\(x\))? = (.+?)\. Is f odd", text)
    if m:
        rule = parse_rule(m.group(1))
        pairs = [(evaluate(rule, v), evaluate(rule, -v)) for v in (1.5, 2.5, 7.0)]
        if any(b is None for _, b in pairs):
            return "neither"
        if all(math.isclose(b, a_) for a_, b in pairs):
            return "even"
        return "odd" if all(math.isclose(b, -a_) for a_, b in pairs) else "neither"
    m = re.match(r"A function f\(x\) = a(x\^\S+) has a graph through the point \((\d+), (-?\d+)\)", text)
    if m:
        p, qq = index(m.group(1)[2:].strip("()"))
        big, y = int(m.group(2)), int(m.group(3))
        ratio = rpow(-big, p, qq) / rpow(big, p, qq)
        return str(round(y * ratio))
    m = re.match(r"Let y = (.+?)\. The graph has a (horizontal|vertical) asymptote", text)
    if m:
        a, h, p, qq, k = parse_rule(m.group(1))
        if m.group(2) == "vertical":
            return str(h)
        expr = a * sympy.Pow(x - h, sympy.Rational(p, qq)) + k
        return str(sympy.limit(expr, x, sympy.oo))
    m = re.match(r"Let y = (.+?)\. Give the x-value of the vertical", text)
    if m:
        a, h, p, qq, k = parse_rule(m.group(1))
        expr = a * sympy.Pow(x - h, sympy.Rational(p, qq)) + k
        return f"{h}, {sympy.limit(expr, x, sympy.oo)}"
    m = re.match(r"Let y = (.+?)\. State the quadrants", text)
    if m:
        rule = parse_rule(m.group(1))
        return show_list(quadrants(lambda v: evaluate(rule, v)))
    m = re.match(
        r"The graph of y = (.+?) is reflected in (the x-axis|the y-axis|both axes[^.]*)\. State the quadrants", text
    )
    if m:
        rule = parse_rule(m.group(1))
        kind = m.group(2)
        sx = -1 if kind.startswith(("the y-axis", "both")) else 1
        sy = -1 if kind.startswith(("the x-axis", "both")) else 1
        return show_list(quadrants(lambda v: (lambda r: None if r is None else sy * r)(evaluate(rule, sx * v))))
    m = re.match(r"Let f\(x\) = (.+?)\. The graph of y = f\(x\) meets", text)
    if m:
        a, _, p, qq, _ = parse_rule(m.group(1))
        hits = [
            v for v in range(-600, 601) if (r := rpow(v, p, qq)) is not None and math.isclose(a * r, v, abs_tol=1e-9)
        ]
        return str(hits[-1] - hits[0])
    m = re.match(
        r"(?:The graph of|Starting from) y = (\S+?),? (?:is|the graph is) .*?(?:factor of (\d+).*?)?reflected in the (x|y)-axis\. "
        r"Find the y-coordinate of the point on the image with x-coordinate (-?\d+)",
        text,
    )
    if m:
        p, qq = index(m.group(1)[2:].strip("()"))
        a = int(m.group(2) or 1)
        X = int(m.group(4))
        r = rpow(-X, p, qq) if m.group(3) == "y" else rpow(X, p, qq)
        return str(round(a * r * (-1 if m.group(3) == "x" else 1)))
    m = re.match(r"Let y = (.+?)\. Find the x-coordinate of the right-most", text)
    rule = parse_rule(m.group(1))
    hits = [
        v
        for v in range(-40, 41)
        if (r := evaluate(rule, v)) is not None and math.isclose(r, 0, abs_tol=1e-9) and v != rule[1]
    ]
    return str(hits[-1])


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_tier(tier):
    t.assert_tier(SKILL, tier, seeds=range(60), oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier,distinct,top", [("easy", 6, 0.4), ("medium", 6, 0.35), ("hard", 8, 0.3)])
def test_answer_space(tier, distinct, top):
    t.assert_answer_space(SKILL, tier, distinct=distinct, top=top)


def test_unknown_tier_is_rejected():
    from mathgen import make

    with pytest.raises(ValueError):
        make("vm_power_graph_features", seed=1, tier="nonsense")
    with pytest.raises(ValueError):
        make("vm_power_graph_features_exam", seed=1, tier="nonsense")


def test_easy_covers_each_symmetry_word():
    words = {t.question(SKILL, "easy", s).answer_text for s in range(80)}
    assert {"odd", "even", "neither"} <= words


def test_hard_covers_each_kind():
    texts = [t.question(SKILL, "hard", s).text for s in range(80)]
    assert any("meets the line y = x" in v for v in texts)
    assert any("dilated" in v or "stretched" in v for v in texts)
    assert any("right-most x-intercept" in v for v in texts)


def test_registered_and_chain_shape():
    sup.assert_registered(SKILL, CHAIN)
    sup.assert_chain_shape(CHAIN, ["fraction", "fraction", "ordered_list", "fraction"], [2, 2, 2, 3], [0, 1, 0, 1])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN, seeds=range(60))
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=100, top=0.05)


def test_chain_key_matches_independent_derivation():
    for seed in range(80):
        q = sup.exam_question(CHAIN, seed)
        n = int(re.search(r"x\^\(-(\d)\)", q.stem).group(1))
        u, v = (int(g) for g in re.search(r"\((\d+), (-?\d+)\)", q.stem).groups())
        a = sympy.Rational(v) * u**n
        got = {p.label: p.answer_text for p in q.parts}
        assert got["a"] == str(a)
        assert sympy.Rational(got["b"]) == a * sympy.Integer(-u) ** (-n)
        assert got["c"] == show_list(quadrants(lambda val: rpow(val, -n, 1) * float(a)))
        k = int(re.search(r"(up|down)", q.parts[3].prompt).group(1) == "up") * 2 - 1
        shift = int(re.search(r"translated (\d+) units?", q.parts[3].prompt).group(1)) * k
        roots = [r for r in sympy.solve(a * x ** (-n) + shift, x) if r.is_real and r > 0]
        assert sympy.Rational(got["d"]) == roots[0]


def _rank(key: str, options) -> int:
    """How many wrong options sort below the key: by value when they are numbers, else by text."""

    def sort_key(text):
        try:
            return (0, float(Fraction(text.replace(" ", ""))), "")
        except (ValueError, ZeroDivisionError):
            return (1, 0.0, text)

    return sum(1 for o in options if sort_key(o) < sort_key(key))


def test_key_rank_varies_power_graph():
    """The key must not sit in one place among the options: each tier (and each exam part) takes at least three ranks."""
    from collections import Counter

    for tier in ("easy", "medium", "hard"):
        ranks = Counter()
        for seed in range(240):
            q = t.question(SKILL, tier, seed)
            ranks[_rank(q.answer_text, q.distractors)] += 1
        assert len(ranks) >= 3, (tier, ranks)
    parts = {}
    for seed in range(240):
        q = sup.exam_question(CHAIN, seed)
        for p in q.parts:
            parts.setdefault(p.label, Counter())[_rank(p.answer_text, p.distractors)] += 1
    for label, ranks in parts.items():
        assert len(ranks) >= 3, (label, ranks)


def test_no_option_is_zero_for_a_value_of_a_power_function():
    """a x^(p/q) at a non-zero x is never 0, so no option of these value questions may be 0 (it gives the key away)."""
    for seed in range(300):
        q = t.question(SKILL, "easy", seed)
        if q.text.startswith("A function f(x) = a"):
            assert "0" not in q.distractors, (seed, q.text, q.distractors)
        q = t.question(SKILL, "hard", seed)
        if " image " in q.text or "image with" in q.text:
            assert "0" not in q.distractors, (seed, q.text, q.distractors)
    for seed in range(300):
        q = sup.exam_question(CHAIN, seed)
        for label in ("a", "b", "d"):
            part = next(p for p in q.parts if p.label == label)
            assert "0" not in part.distractors, (seed, label, part.distractors)


def test_key_position_in_the_option_list_varies():
    from mathgen.core import mcq_options

    for tier in ("easy", "medium", "hard"):
        spots = {mcq_options(q).index(q.answer_text) for q in (t.question(SKILL, tier, s) for s in range(60))}
        assert len(spots) >= 2, (tier, spots)
