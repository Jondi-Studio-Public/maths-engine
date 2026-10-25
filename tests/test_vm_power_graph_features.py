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


# -- what each stem rules out ---------------------------------------------------------------------------------------
#
# Every part of every tier, with what its stem implies about the answer and so about EVERY option (slips and fillers
# alike; the key is never moved to fit).
#
#   part                                  constraint on every option
#   ------------------------------------  --------------------------------------------------------------------------
#   easy symmetry                         one of the words odd, even, neither
#   easy value at -u (point given)        a non-zero whole number (a power function is never 0 at x != 0)
#   easy asymptote c                      a whole number
#   quadrants, reflection image, exam (c) a strictly increasing list of numbers from 1 to 4, as the stem asks
#   both asymptotes "h, k"                two whole numbers
#   hard distance between intersections   a whole number, at least 1 (a distance between two different points)
#   hard value on the image               a non-zero whole number
#   hard right-most x-intercept           a whole number other than h (x = h is the asymptote, not on the graph)
#   exam (a) a, (b) y at -u               a non-zero number; (b) at most 8 times the key in size (CROSS-PART: it
#                                         mirrors the point (u, v) given in the stem)
#   exam (d) positive x-intercept         a positive number
#   every part                            no option repeats the key


def _numbers(text: str) -> list[Fraction]:
    return [Fraction(v) for v in re.split(r",\s*", text.strip())]


def _check_single(q):
    text, key, opts = q.text, q.answer_text, list(q.distractors)
    assert key not in opts, text
    if text.startswith("Let f(x) =") and "odd, even or neither" in text:
        assert set(opts) <= {"odd", "even", "neither"}, (text, opts)
        return
    if "quadrant" in text:
        for o in [key, *opts]:
            vs = _numbers(o)
            assert all(v == int(v) and 1 <= v <= 4 for v in vs) and vs == sorted(set(vs)), (text, o)
        return
    nums = [_numbers(o) for o in opts]
    assert all(v == int(v) for vs in nums for v in vs), (text, opts)
    if "Give the x-value of the vertical asymptote" in text:
        assert all(len(vs) == 2 for vs in nums), (text, opts)
    elif text.startswith("A function f(x) = a") or "Find the y-coordinate of the point on the image" in text:
        assert all(vs[0] != 0 for vs in nums), (text, opts)
    elif "distance along the x-axis" in text:
        assert all(vs[0] >= 1 for vs in nums), (text, opts)
    elif "right-most x-intercept" in text:
        m = re.search(r"\(x ([-+]) (\d+)\)", text)
        h = 0 if not m else (int(m.group(2)) if m.group(1) == "-" else -int(m.group(2)))
        assert all(vs[0] != h for vs in nums), (text, opts, h)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_no_option_is_ruled_out_by_the_stem(tier):
    for seed in range(300):
        _check_single(t.question(SKILL, tier, seed))


def test_no_option_of_an_exam_part_is_ruled_out_by_the_stem():
    for seed in range(300):
        q = sup.exam_question(CHAIN, seed)
        parts = {p.label: p for p in q.parts}
        for p in parts.values():
            assert p.answer_text not in p.distractors, (seed, p.label)
        for label in ("a", "b", "d"):
            assert all(Fraction(o) != 0 for o in parts[label].distractors), (seed, label)
        assert all(Fraction(o) > 0 for o in parts["d"].distractors), (seed, parts["d"].distractors)
        # (b) is the mirror point of the (u, v) the stem gave: no option is wildly bigger than the key
        key_b = abs(Fraction(parts["b"].answer_text))
        assert all(abs(Fraction(o)) <= 8 * key_b for o in parts["b"].distractors), (seed, parts["b"].distractors)
        for o in [parts["c"].answer_text, *parts["c"].distractors]:
            vs = _numbers(o)
            assert all(v == int(v) and 1 <= v <= 4 for v in vs) and vs == sorted(set(vs)), (seed, o)


def test_text_has_no_double_signs():
    """A negative number is never written after a sign (`--3`, `+ -3`, `- -3`), nor as `(+ ...)` or `+ (-3)`."""
    pattern = re.compile(r"--|\+ -|- -|\+ \(-\d")
    for seed in range(300):
        texts = []
        for tier in ("easy", "medium", "hard"):
            q = t.question(SKILL, tier, seed)
            texts += [q.text, *q.working]
        ex = sup.exam_question(CHAIN, seed)
        texts += [ex.stem, *(w for p in ex.parts for w in [p.prompt, *p.working])]
        for text in texts:
            assert not pattern.search(text), (seed, text)


# -- the key is not told apart by its shape ---------------------------------------------------------------------------


def _kinds(text: str):
    """The sign, number of integer digits, decimal places and integer-ness of a numeric option (None if not a number)."""
    text = text.replace(" ", "")
    try:
        v = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return {
        "sign": v < 0,
        "digits": len(str(int(abs(v)))),
        "places": len(text.split(".")[1]) if "." in text else 0,
        "whole": v.denominator == 1,
    }


def _lone_kinds(items):
    """Over (key, options) pairs: how often the key is the ONLY option of its kind, per kind."""
    from collections import Counter

    lone, total = Counter(), 0
    for key, options in items:
        k, others = _kinds(key), [_kinds(o) for o in options]
        if k is None or None in others or len(others) < 3:
            continue
        total += 1
        for dim in k:
            if all(o[dim] != k[dim] for o in others):
                lone[dim] += 1
    return {dim: lone[dim] / max(total, 1) for dim in ("sign", "digits", "places", "whole")}, total


def test_the_key_is_never_told_apart_by_sign_size_or_format():
    """Over 1000 seeds per tier and per exam part, the key is the only option of its sign, integer-digit count, decimal
    places or whole-ness in at most a small share of questions (chance alone gives a few percent; a generator whose
    wrong answers share a shape that the key lacks gives a third or more)."""
    from collections import defaultdict

    groups = {}
    for tier in ("easy", "medium", "hard"):
        qs = [t.question(SKILL, tier, s) for s in range(1000)]
        groups[tier] = [(q.answer_text, q.distractors) for q in qs]
    parts = defaultdict(list)
    for s in range(1000):
        for p in sup.exam_question(CHAIN, s).parts:
            parts[f"exam {p.label}"].append((p.answer_text, p.distractors))
    groups.update(parts)
    for name, items in groups.items():
        shares, total = _lone_kinds(items)
        if total < 100:
            continue
        for dim, share in shares.items():
            assert share <= 0.12, (name, dim, round(share, 3), shares)


def _lone_sign(key: str, options) -> bool:
    """True when the key is a non-zero number and every wrong option is on the other side of zero."""
    try:
        k = Fraction(key.replace(" ", ""))
        vs = [Fraction(o.replace(" ", "")) for o in options]
    except (ValueError, ZeroDivisionError):
        return False  # a word or a list: no sign to give it away
    return k != 0 and all(v * k <= 0 for v in vs)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_the_key_is_never_the_only_option_of_its_sign(tier):
    bad = []
    for seed in range(1200):
        q = t.question(SKILL, tier, seed)
        if _lone_sign(q.answer_text, q.distractors):
            bad.append((seed, q.answer_text, list(q.distractors)))
    assert not bad, bad[:5]


def test_the_key_of_an_exam_part_is_never_the_only_option_of_its_sign():
    bad = []
    for seed in range(1200):
        for p in sup.exam_question(CHAIN, seed).parts:
            if _lone_sign(p.answer_text, p.distractors):
                bad.append((seed, p.label, p.answer_text, list(p.distractors)))
    assert not bad, bad[:5]
