"""Tests for `trapezium-rule-estimate`: the practice tiers (`sc_trapezium`) and the exam chain (`sc_trapezium_chain`).

The oracle reads the formula, the table, the stated estimate or the stated overestimate back out of the rendered text
and recomputes the key with SymPy (one trapezium at a time) rather than with the generator's own helpers.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy

import r4e_support as t
from mathgen.engine import grade
from mathgen.text_quality import prompt_shape
from r4e_support import sup

SKILL = "trapezium-rule-estimate"
CHAIN = "sc_trapezium_chain"
x = sympy.Symbol("x")
NUM = r"-?\d+(?:\.\d+)?"
LIST = rf"{NUM}(?:, {NUM})+"
CURVES = {
    "e^x": (sympy.exp(x), "convex"),
    "1/x": (1 / x, "convex"),
    "sqrt(x)": (sympy.sqrt(x), "concave"),
    "sin(x)": (sympy.sin(x), "concave"),
}


def trap(f_values, h):
    """Sum of the areas of the trapezia, one strip at a time."""
    h = sympy.sympify(h)
    return sum(h * (sympy.sympify(l) + sympy.sympify(r)) / 2 for l, r in zip(f_values, f_values[1:]))


def curve_of(text):
    for name, (f, kind) in CURVES.items():
        if re.search(rf"(?:integral of |y = |f\(x\) = ){re.escape(name)}(?![\w^(])", text):
            return name, f, kind
    return None, None, None


def n_of(text):
    return int(re.search(r"(\d) (?:equal )?strips", text).group(1))


def ends_of(text):
    a, b = re.search(rf"(?:x = |from |between )({NUM}) (?:to|and) (?:x = )?({NUM})", text).groups()
    return sympy.nsimplify(a), sympy.nsimplify(b)


def curve_estimate(f, a, b, n):
    h = sympy.sympify(b - a) / n
    return trap([f.subs(x, a + i * h) for i in range(n + 1)], h)


def oracle(q):
    text = q.text
    if re.search(r"\bk\b", text):
        n = n_of(text)
        vals = re.search(rf"((?:{NUM}|k)(?:, (?:{NUM}|k))+) at x = ", text).group(1).split(", ")
        xs = [sympy.nsimplify(c) for c in re.search(rf"at x = ({LIST})\.", text).group(1).split(", ")]
        h = xs[1] - xs[0]
        est = sympy.nsimplify(re.search(rf"(?:gives|estimated to be) ({NUM})(?: as|\.)", text).group(1))
        j = vals.index("k")
        known = sum((1 if i in (0, n) else 2) * sympy.nsimplify(v) for i, v in enumerate(vals) if i != j)
        return sup.dp((2 * est / h - known) / (1 if j in (0, n) else 2), 1)
    if "number of strips n" in text:
        f = sup.parse_poly(re.search(r"(?:y|f\(x\)) = (.+?)(?:,| from x| between)", text).group(1), x=x)
        a, b = ends_of(text)
        e = sympy.nsimplify((re.search(rf"is ({NUM}) more than", text) or re.search(rf"by ({NUM})\.", text)).group(1))
        for m in range(1, 40):
            h = (b - a) / m
            over = trap([f.subs(x, a + i * h) for i in range(m + 1)], h) - sympy.integrate(f, (x, a, b))
            if over == e:
                return str(m)
        raise AssertionError(text)
    n = n_of(text)
    name, f, kind = curve_of(text)
    if "exact value minus" in text:
        a, b = ends_of(text)
        exact4 = sympy.Rational(re.search(rf"({NUM}),? (?:correct to|to) 4 decimal", text).group(1))
        diff = exact4 - curve_estimate(f, a, b, n)
        assert (diff > 0) == (kind == "concave"), text
        return sup.dp(diff, 4)
    if name:
        a, b = ends_of(text)
        return sup.dp(curve_estimate(f, a, b, n), 3)
    if "every" in text:
        vals = [sympy.nsimplify(c) for c in re.search(rf"({LIST})\.", text).group(1).split(", ")]
        return sup.dp(trap(vals, sympy.Integer(re.search(r"every (\d+) ", text).group(1))), 2)
    if " at x = " in text:
        vals = [sympy.nsimplify(c) for c in re.search(rf"({LIST}) at x = ", text).group(1).split(", ")]
        xs = [sympy.nsimplify(c) for c in re.search(rf"at x = ({LIST})", text).group(1).split(", ")]
        return _exact(trap(vals, xs[1] - xs[0]))
    f = sup.parse_poly(re.search(r"(?:y|f\(x\)) = (.+?)(?:\.| from| between)", text).group(1), x=x)
    a, b = ends_of(text)
    return _exact(curve_estimate(f, a, b, n))


def _exact(value):
    value = sympy.Rational(value)
    places = 0
    while (value * 10**places).q != 1:
        places += 1
    return sup.dp(value, places)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_tier(tier):
    t.assert_tier(SKILL, tier, oracle=oracle, seeds=range(90))


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier,minimum", [("easy", 3), ("medium", 3), ("hard", 3)])
def test_three_prompt_shapes(tier, minimum):
    shapes = {prompt_shape(t.question(SKILL, tier, s)) for s in range(90)}
    assert len(shapes) >= minimum, shapes


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=30)


def test_stems_are_clean():
    for tier in ("easy", "medium", "hard"):
        for seed in range(80):
            text = t.question(SKILL, tier, seed).text
            assert "*" not in text and "**" not in text, text
            assert not re.search(r"\b1 (?:parts|strips)\b", text), text
            assert "  " not in text, text
    for seed in range(100):
        q = sup.exam_question(CHAIN, seed)
        assert "*" not in q.stem and not re.search(r"\b1 (?:parts|strips)\b", q.stem), q.stem


def test_easy_table_and_poly_both_appear():
    kinds = Counter("table" if "g" in t.question(SKILL, "easy", s).text.split() else "poly" for s in range(60))
    assert kinds["table"] and kinds["poly"]


def test_hard_covers_all_three_tasks():
    seen = Counter()
    for seed in range(120):
        text = t.question(SKILL, "hard", seed).text
        seen[
            "missing" if "Find the value of k" in text else "strips" if "number of strips n" in text else "difference"
        ] += 1
    assert len(seen) == 3 and min(seen.values()) >= 10, seen


def test_difference_sign_matches_concavity():
    signs = set()
    for seed in range(200):
        q = t.question(SKILL, "hard", seed)
        if "exact value minus" not in q.text:
            continue
        _, _, kind = curve_of(q.text)
        value = sympy.Rational(q.answer_text)
        assert (value > 0) == (kind == "concave"), (seed, q.text, q.answer_text)
        signs.add(value > 0)
    assert signs == {True, False}


def test_strip_counts_cover_two_to_five():
    medium = {n_of(t.question(SKILL, "medium", s).text) for s in range(100)}
    assert {2, 3, 4, 5} <= medium, medium
    exam = {len(re.findall(r"y\d", sup.exam_question(CHAIN, s).parts[1].prompt)) // 1 for s in range(60)}
    assert len(exam) >= 3


def test_registered():
    sources = {s["difficulty"]: s for s in sup.skill_yaml(SKILL)["sources"]}
    assert set(sources) == {"easy", "medium", "hard", "exam"}
    sup.assert_registered(SKILL, CHAIN)


def test_unknown_tier_raises():
    from mathgen import make

    with pytest.raises(ValueError):
        make("sc_trapezium", seed=1, tier="impossible")
    with pytest.raises(ValueError):
        make("sc_trapezium_chain", seed=1, tier="easy")


def test_chain_shape():
    sup.assert_chain_shape(
        CHAIN, ["fraction", "number", "number", "number", "number"], [1, 2, 2, 2, 1], [0, 0, 1, 1, 1]
    )


def test_chain_clean_and_tagged():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)


def test_chain_grading():
    sup.assert_grading(SKILL, CHAIN)


def test_chain_shapes_and_answer_space():
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=100, top=0.05)


def test_chain_key_matches_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(CHAIN, seed)
        name, f, kind = curve_of(q.stem)
        a, b = ends_of(q.stem)
        n = n_of(q.stem)
        h = (b - a) / n
        ys = [f.subs(x, a + i * h) for i in range(n + 1)]
        total = ys[0] + ys[-1] + 2 * sum(ys[1:-1])
        est = trap(ys, h)
        assert sympy.simplify(est - h * total / 2) == 0
        exact4 = sympy.Rational(re.search(rf"({NUM}),? (?:correct to|to) 4 decimal", q.stem).group(1))
        got = {p.label: p.answer_text for p in q.parts}
        assert sympy.nsimplify(got["a"]) == h
        assert got["b"] == sup.dp(total, 4), (seed, got["b"], sup.dp(total, 4))
        assert got["c"] == sup.dp(est, 4)
        assert got["d"] == sup.dp(exact4 - est, 4)
        assert got["e"] == sup.dp(abs(exact4 - est), 4)
        assert (sympy.Rational(got["d"]) > 0) == (kind == "concave")


def test_no_distractor_is_a_correct_answer_in_any_part():
    sk = sup.skill_object(SKILL)
    for tier in ("easy", "medium", "hard"):
        for seed in range(60):
            q = t.question(SKILL, tier, seed)
            for d in q.distractors:
                assert not grade(sk, q, d).correct, (tier, seed, d)


def test_difference_stems_say_to_use_the_exact_value_given():
    seen = 0
    for seed in range(200):
        text = t.question(SKILL, "hard", seed).text
        if "exact value minus" in text:
            seen += 1
            assert "using the exact value given" in text, text
    assert seen >= 30


def test_non_terminating_widths_ask_for_a_fraction():
    seen = 0
    for seed in range(100):
        q = sup.exam_question(CHAIN, seed)
        prompt = q.parts[0].prompt
        terminating = all(
            c in "0.123456789" for c in sup.dp(sympy.Rational(q.parts[0].answer_text), 8).rstrip("0")
        ) and (sympy.Rational(q.parts[0].answer_text).q in (1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100))
        if sympy.Rational(q.parts[0].answer_text).q % 3 == 0 or sympy.Rational(q.parts[0].answer_text).q % 7 == 0:
            seen += 1
            assert "fraction in simplest form" in prompt, (seed, prompt)
        elif terminating:
            assert "fraction" not in prompt, (seed, prompt)
    assert seen >= 10
