"""Tests `uc2-geometric-negative-binomial` (topic `r3k_geometric`) and its exam chain `exd_x_r3k_geometric`.

The oracle reads p, k, n, r back out of the question TEXT and re-derives each key from the probability mass function
with exact fractions, and the variances from SymPy's generating-function derivatives.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy

import mathgen.generators.r3k_geometric_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.r3k_support import check_no_follow_through_collision, second_route_on  # noqa: F401  (autouse fixture)
from tests.r3k_support import (
    all_fractions,
    as_fraction,
    check_exam_grading,
    check_follow_through,
    check_single_tier,
    check_wiring,
    first_fraction,
    skill,
    stem,
)

TOPIC = "r3k_geometric"
EXAM = "exd_x_r3k_geometric"
SK = skill("uc2-geometric-negative-binomial")
TIERS = ("easy", "medium", "hard")
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _binom(n, k):
    return Fraction(sympy.binomial(n, k))


def _pgf_var(p, r):
    s = sympy.Symbol("s")
    g = (
        sympy.Rational(p.numerator, p.denominator) * s / (1 - (1 - sympy.Rational(p.numerator, p.denominator)) * s)
    ) ** r
    d1, d2 = sympy.diff(g, s).subs(s, 1), sympy.diff(g, s, 2).subs(s, 1)
    v = sympy.nsimplify(d2 + d1 - d1**2)
    return Fraction(int(v.p), int(v.q))


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["binomial-point-probability", "discrete-expected-value"], "fraction")


def test_easy_key_is_the_geometric_mass_or_tail():
    shapes = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        p = first_fraction(q.text)
        k = int(re.search(r"(?:P\(X = |P\(X > |more than )(\d+)", q.text).group(1))
        if "P(X = " in q.text:
            expected = (1 - p) ** (k - 1) * p
            shapes["mass"] += 1
        else:
            expected = (1 - p) ** k
            shapes["tail"] += 1
        assert as_fraction(q.answer) == expected, (seed, q.text)
    assert shapes["mass"] > 20 and shapes["tail"] > 20, shapes


def test_medium_key_is_the_negative_binomial_mass():
    for seed in range(120):
        q = _q("medium", seed)
        p = first_fraction(q.text)
        n = int(re.search(r"(?:number |P\(Y = )(\d+)", q.text).group(1))
        r = next(v for w, v in _ORD.items() if f"the {w} " in q.text)
        assert as_fraction(q.answer) == _binom(n - 1, r - 1) * p**r * (1 - p) ** (n - r), (seed, q.text)


def test_hard_keys():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        t = q.text
        if "is even" in t:
            p = first_fraction(t)
            expected = sum(((1 - p) ** (j - 1) * p for j in range(2, 600, 2)), Fraction(0))
            assert abs(float(as_fraction(q.answer) - expected)) < 1e-12, (seed, t)
            seen["even"] += 1
        elif "takes turns" in t or "take turns" in t:
            pa, pb = all_fractions(t)[:2]
            expected = sum((((1 - pa) * (1 - pb)) ** j * pa for j in range(0, 300)), Fraction(0))
            assert abs(float(as_fraction(q.answer) - expected)) < 1e-12, (seed, t)
            seen["race"] += 1
        elif "Var(Y)" in t:
            p = first_fraction(t)
            r = _ORD[re.search(r"including the (\w+) ", t).group(1)]
            assert as_fraction(q.answer) == _pgf_var(p, r), (seed, t)
            seen["var"] += 1
        else:
            p = first_fraction(t)
            m, k = (int(x) for x in re.search(r"first (\d+) .*?next (\d+)", t).groups())
            # P(m < X <= m + k) / P(X > m), summed from the masses
            q_ = 1 - p
            num = sum((q_ ** (j - 1) * p for j in range(m + 1, m + k + 1)), Fraction(0))
            assert as_fraction(q.answer) == num / q_**m, (seed, t)
            seen["memoryless"] += 1
    assert all(seen[k] >= 20 for k in ("even", "race", "var", "memoryless")), seen


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=80)


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(100)]
        assert len(set(answers)) >= 35, (tier, Counter(answers).most_common(3))


def test_tiers_differ_in_shape():
    def kinds(tier):
        return {("k" if "Var(" in _q(tier, s).text else "p") for s in range(60)}

    easy_text = " ".join(_q("easy", s).text for s in range(40))
    assert "second" not in easy_text and "Var(" not in easy_text
    medium_text = " ".join(_q("medium", s).text for s in range(40))
    assert "second" in medium_text and "Var(" not in medium_text
    hard_text = " ".join(_q("hard", s).text for s in range(60))
    assert "Var(" in hard_text and "turns" in hard_text and "even" in hard_text
    assert kinds("easy") == {"p"}


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 2, 1]
        assert [p.answer_kind for p in q.parts] == ["fraction"] * 4
        assert q.parts[3].follow_through.endswith("*part_c")
        assert q.parts[0].sensible_range == (0, 1) and q.parts[2].sensible_range == (0, None)


def test_exam_keys_match_the_mass_function():
    for seed in range(120):
        q = _exam(seed)
        s = stem(q)
        p = first_fraction(s)
        r = _ORD[re.search(r"(?:Y be the number of \w+ up to and including the|produces the) (\w+) ", s).group(1)]
        k = int(re.search(r"P\(X = (\d+)\)", q.parts[0].prompt).group(1))
        n = int(re.search(r"P\(Y = (\d+)\)", q.parts[1].prompt).group(1))
        qq = 1 - p
        assert as_fraction(q.parts[0].answer) == qq ** (k - 1) * p
        assert as_fraction(q.parts[1].answer) == _binom(n - 1, r - 1) * p**r * qq ** (n - r)
        assert as_fraction(q.parts[2].answer) == _pgf_var(p, 1)
        assert as_fraction(q.parts[3].answer) == _pgf_var(p, r)


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=40)


def test_exam_follow_through_from_a_wrong_variance():
    assert check_follow_through(SK, _exam, "d", seeds=30) > 0


def test_exam_whole_chain_carries_a_wrong_c():
    q = _exam(5)
    canon = {p.label: p.answer_text for p in q.parts}
    r = int(q.parts[3].follow_through.split("*")[0])
    wrong_c = Fraction(7, 3)
    reply = {**canon, "c": str(wrong_c), "d": str(r * wrong_c)}
    result = grade(SK, q, reply)
    assert result.parts["d"].marks == q.parts[3].marks
    assert result.parts["c"].marks == 0
