"""Tests `uc2-generating-function-coefficient` (topic `r4j_generating_function_coefficient`) and its exam chain.

The oracle reads the generating function (or the counting question) out of the TEXT and extracts the coefficient with
SymPy's series expansion, or by brute-force counting for the counting phrasings.
"""

from __future__ import annotations

import itertools
import re

import sympy

import mathgen.generators.r4j_generating_function_coefficient_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r4j_support import check_carried, check_parts, check_single

TOPIC = "r4j_generating_function_coefficient"
EXAM = "exd_x_r4j_generating_function_coefficient"
SKILL = next(s for s in load_skills()[1] if s.id == "uc2-generating-function-coefficient")
x = sympy.Symbol("x")
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _expr(text):
    s = text.replace("−", "-").replace("^", "**")
    s = re.sub(r"(\d)x", r"\1*x", s)
    s = s.replace(")(", ")*(")
    return sympy.sympify(s, locals={"x": x})


def _coefficient(gf, k):
    return sympy.series(gf, x, 0, k + 1).removeO().coeff(x, k)


def _oracle(text):
    m = re.search(
        r"How many ordered triples \(p, q, r\) of whole numbers, each at most (\d+), have p \+ q \+ r = (\d+)", text
    )
    if m:
        u, n = map(int, m.groups())
        return sum(1 for t in itertools.product(range(u + 1), repeat=3) if sum(t) == n)
    m = re.search(r"w \+ x \+ y \+ z = (\d+) have if w ≤ (\d+) and x ≤ (\d+)", text)
    if m:
        k, a, b = map(int, m.groups())
        return sum(1 for t in itertools.product(range(k + 1), repeat=4) if sum(t) == k and t[0] <= a and t[1] <= b)
    k = int(
        (
            re.search(r"coefficient of x\^?(\d*)", text)
            or re.search(r"term in x\^(\d+)", text)
            or re.search(r"g([₀-₉]+)", text)
        )
        .group(1)
        .translate(_SUB)
        or 1
    )
    if "g" in text and re.search(r"Find g([₀-₉]+)", text):
        k = int(re.search(r"Find g([₀-₉]+)", text).group(1).translate(_SUB))
    m = re.search(r"term in x\^(\d+)", text)
    if m:
        k = int(m.group(1))
    gf = re.search(
        r"((?:1/\(\(.*?\)\))|(?:\(\(1 − x\^\d+\)/\(1 − x\)\)\^3)|(?:x(?:\^\d+)?/\(1 − \d*x\)\^\d)|(?:\(1 − x\^\d+\)\^2/\(1 − x\)\^4))",
        text,
    ).group(1)
    return int(_coefficient(_expr(gf), k))


def test_skill_metadata():
    assert list(SKILL.year_level) == ["university"] and "university-year-2" in SKILL.tags


def test_every_tier_matches_series_expansion():
    for tier in ("easy", "medium", "hard"):
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert int(q.answer_text) == _oracle(q.text), (tier, seed, q.text, q.answer_text)
            check_single(q, "number")


def test_tiers_differ_in_shape():
    shapes = {
        t: {re.sub(r"\d+", "#", make(TOPIC, seed=s, difficulty=t).text) for s in range(40)}
        for t in ("easy", "medium", "hard")
    }
    assert shapes["easy"].isdisjoint(shapes["medium"]) and shapes["medium"].isdisjoint(shapes["hard"])
    hard_text = " ".join(make(TOPIC, seed=s, difficulty="hard").text for s in range(60))
    assert "ordered triples" in hard_text and "(1 − x^" in hard_text


def test_exam_chain_matches_series_expansion():
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        k = int(re.search(r"coefficient of x\^(\d+) in F", q.text).group(1))
        a, b = map(int, re.search(r"\(1 − x\^(\d+)\)\(1 − x\^(\d+)\)", q.text).groups())
        want = [
            _coefficient(1 / (1 - x) ** 3, k),
            _coefficient(x**a / (1 - x) ** 3, k),
            _coefficient(x**b / (1 - x) ** 3, k),
            _coefficient((1 - x**a) * (1 - x**b) / (1 - x) ** 3, k),
        ]
        assert [int(p.answer_text) for p in q.parts] == [int(v) for v in want], (seed, q.text)
        total, over_a, over_b = (int(p.answer_text) for p in q.parts[:3])
        wrong_b = over_b + 3
        carried = total - over_a - wrong_b + (int(q.parts[3].answer_text) - (total - over_a - over_b))
        check_carried(q, "c", str(wrong_b), str(carried))
