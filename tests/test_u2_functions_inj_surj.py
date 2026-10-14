"""Tests `ucv1-functions-injective-surjective` (topics `u2_functions_injective_surjective` and its exam chain).

Oracles read the question TEXT and recount: finite-set counts by listing every function, the quadratic questions by
evaluating the rule on integers, the modular question by listing the images of every multiplier.
"""

from __future__ import annotations

import itertools
import re
from math import gcd

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from tests import u2_support as su

TOPIC = "u2_functions_injective_surjective"
EXAM = "exd_x_u2_functions_injective_surjective"
SK = su.skill("ucv1-functions-injective-surjective")


def _size(token):
    token = token.strip()
    if "..." in token:
        return int(token.rstrip("}").split(",")[-1])
    return len(token.strip("{}").split(","))


_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _poly(text):
    """The rule 'f(x) = ax² - bx + c' read out of the text, as a function of an integer."""
    body = re.search(r"f\(x\) = (.*?)(?:,| is)", text).group(1).replace("²", "**2")
    expr = parse_expr(body, local_dict={"x": _X}, transformations=_TRANSFORMS)
    return lambda value: int(expr.subs(_X, value))


def test_wiring():
    su.check_wiring(SK, TOPIC, EXAM, "number")


@pytest.mark.parametrize("tier", su.TIERS)
def test_key_and_distractors(tier):
    su.check_tier(SK, TOPIC, tier)


def test_easy_counts_by_listing_functions():
    kinds = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        if "one-to-one" in q.text:
            m, n = (_size(t) for t in re.findall(r"= (\{[^}]*\})", q.text))
            if n <= 6:
                listed = sum(1 for f in itertools.product(range(n), repeat=m) if len(set(f)) == m)
                assert int(q.answer) == listed, (seed, q.text)
            kinds.add("inj")
        else:
            n = _size(re.search(r"X = (\{[^}]*\})", q.text).group(1))
            if n <= 6:
                g = [int(v) for v in re.findall(r"g\((\d+)\) = (\d+)", q.text)[0]]
                h = [int(v) for v in re.findall(r"g\((\d+)\) = (\d+)", q.text)[1]]
                listed = sum(
                    1 for p in itertools.permutations(range(1, n + 1)) if p[g[0] - 1] == g[1] and p[h[0] - 1] == h[1]
                )
                assert int(q.answer) == listed, (seed, q.text)
            kinds.add("bij")
    assert kinds == {"inj", "bij"}


def test_medium_counts_by_listing_functions():
    kinds = set()
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="medium")
        m, n = (_size(t) for t in re.findall(r"(\{[^}]*\})", q.text)[:2])
        functions = list(itertools.product(range(n), repeat=m))
        if "onto" in q.text:
            assert int(q.answer) == sum(1 for f in functions if len(set(f)) == n), (seed, q.text)
            kinds.add("onto")
        else:
            assert int(q.answer) == sum(1 for f in functions if len(set(f)) < m), (seed, q.text)
            kinds.add("not-injective")
    assert kinds == {"onto", "not-injective"}


def test_hard_keys_by_evaluating_the_rule():
    kinds = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "ℤ_" in q.text:
            n = int(re.search(r"ℤ_(\d+) →", q.text).group(1))
            b = int(re.search(r"ax \+ (\d+)", q.text).group(1))
            count = sum(1 for a in range(n) if len({(a * x + b) % n for x in range(n)}) == n)
            assert int(q.answer) == count == sum(1 for a in range(n) if gcd(a, n) == 1)
            kinds.add("mod")
            continue
        f = _poly(q.text)
        a = int(q.answer)
        if "smallest" in q.text:
            assert all(f(a + k) < f(a + k + 1) for k in range(10))
            assert f(a - 1) > f(a)  # the value just left of a is higher, so a smaller start would repeat a value
            kinds.add("least-a")
        elif "largest value of a" in q.text:
            assert all(f(a - k) < f(a - k - 1) for k in range(10))
            assert f(a + 1) > f(a)
            kinds.add("greatest-a")
        else:
            assert min(f(x) for x in range(-40, 41)) == a
            kinds.add("minimum")
    assert kinds == {"mod", "least-a", "greatest-a", "minimum"}


def test_exam_chain_shape_and_keys():
    su.check_exam(SK, EXAM, "abcd", (2, 2, 2, 3), (False, True, False, True))
    for seed in range(80):
        q = make(EXAM, seed=seed, difficulty="exam")
        f = _poly(q.stem)
        a, m, c, d = (int(p.answer) for p in q.parts)
        assert all(f(a + k) < f(a + k + 1) for k in range(10)) and f(a - 1) > f(a)
        assert m == f(a) == min(f(x) for x in range(a - 30, a + 31))
        y0 = int(re.search(r"f⁻¹\((-?\d+)\)", q.parts[2].prompt).group(1))
        assert [x for x in range(a, a + 40) if f(x) == y0] == [c]
        y1 = int(re.search(r"f\(x\) ≤ (-?\d+)", q.parts[3].prompt).group(1))
        assert d == len([x for x in range(a, a + 60) if f(x) <= y1])


def test_exam_follow_through_from_a_wrong_range():
    """A wrong m (zero marks) carries into (d): the count is worked out from the learner's m."""

    def alpha(q):
        f = _poly(q.stem)
        a = int(q.parts[0].answer)
        return f(a + 1) - f(a)

    def k_of(q):
        return int(q.parts[3].answer) - 1

    def count_from(q, m):
        y1 = int(re.search(r"f\(x\) ≤ (-?\d+)", q.parts[3].prompt).group(1))
        return int(((y1 - m) / alpha(q)) ** 0.5) + 1

    def wrong_m(q):
        # An m a little above the true minimum, chosen so the count it implies is not the m itself.
        for steps in (2 * k_of(q) - 1, 2 * k_of(q), 2 * k_of(q) + 1):
            m = int(q.parts[1].answer) + alpha(q) * steps
            if count_from(q, m) != m and m != int(q.parts[1].answer):
                return m
        raise AssertionError

    su.check_follow_through(SK, EXAM, "b", "d", wrong_m, count_from)


def test_stems_vary():
    for tier in su.TIERS:
        su.distinct_stems(TOPIC, tier, minimum=15)
    assert len({make(TOPIC, seed=s, difficulty="hard").answer_text for s in range(80)}) >= 8
