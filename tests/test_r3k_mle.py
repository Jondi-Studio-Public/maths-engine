"""Tests `uc2-mle-method-of-moments` (topic `r3k_mle`) and its exam chain `exd_x_r3k_mle`.

The oracle reads the data back out of the question TEXT. For the medium estimators it parses the density, builds the
log-likelihood of a small numeric sample and checks that the stated estimator makes the derivative vanish at a maximum
(so it shares no algebra with the generator); the other tiers are recomputed from the data summary with exact fractions.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.r3k_mle_build  # noqa: F401  (registers the topics)
from mathgen import make
from tests.r3k_support import check_no_follow_through_collision, second_route_on  # noqa: F401  (autouse fixture)
from tests.r3k_support import (
    as_fraction,
    check_exam_grading,
    check_follow_through,
    check_single_tier,
    check_wiring,
    skill,
    stem,
)

TOPIC = "r3k_mle"
EXAM = "exd_x_r3k_mle"
SK = skill("uc2-mle-method-of-moments")
TIERS = ("easy", "medium", "hard")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
x_, a_, n_ = sympy.symbols("x a n", positive=True)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _frac(q):
    return as_fraction(q.answer)


def _expr(text, **names):
    text = text.replace("²", "^2").replace("√", "sqrt").replace("π", "pi").replace("^", "**")
    text = re.sub(r"\be\*\*\(", "exp(", text)
    return parse_expr(
        text,
        local_dict={"x": x_, "a": a_, "n": n_, "exp": sympy.exp, "sqrt": sympy.sqrt, "pi": sympy.pi, **names},
        transformations=_TRANSFORMS,
    )


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["bernoulli-mle", "ucv1-one-sample-t-test"], "expression")


def test_easy_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        text = q.text
        if "Poisson" in text:
            counts = [(int(k), int(c)) for k, c in re.findall(r"(\d+): (\d+)", text.split("were", 1)[1])]
            total = sum(k * c for k, c in counts)
            intervals = sum(c for _, c in counts)
            assert intervals == int(re.search(r"in each of (\d+)", text).group(1))
            assert _frac(q) == Fraction(total, intervals), (seed, text)
            seen["poisson"] += 1
        elif "success probability" in text:
            data = [int(v) for v in re.search(r"occasions was ([\d, ]+)\.", text).group(1).split(", ")]
            assert _frac(q) == Fraction(len(data), sum(data)), (seed, text)
            seen["geometric"] += 1
        else:
            n = int(re.search(r"sample of (\d+) independent", text).group(1))
            total = int(re.search(r"total (\d+)", text).group(1))
            assert _frac(q) == Fraction(n, total), (seed, text)
            seen["exponential"] += 1
    assert len(seen) == 3 and min(seen.values()) >= 20, seen


def test_medium_estimator_is_a_maximum_of_the_log_likelihood():
    seen = Counter()
    big_sample = [sympy.Rational(3, 2), sympy.Rational(11, 5), sympy.Rational(31, 10), sympy.Rational(7, 3)]
    unit_sample = [sympy.Rational(1, 3), sympy.Rational(1, 2), sympy.Rational(3, 4), sympy.Rational(2, 5)]
    for seed in range(150):
        q = _q("medium", seed)
        text = q.text
        if "the normal density" in text:
            dens = re.search(r"f\(x; a\) = (.+?), the normal", text).group(1)
        else:
            dens = re.search(r"density f\(x; a\) = (.+?) for ", text).group(1)
        density = _expr(dens)
        stat_name = re.search(r"Let (\w) = ", text).group(1)
        sample = unit_sample if "0 < x < 1" in text else big_sample
        n = len(sample)
        stat = {"T": sum(sample), "L": sum(sympy.log(v) for v in sample), "W": sum(v**2 for v in sample)}[stat_name]
        loglik = sum(sympy.log(density.subs(x_, v)) for v in sample)
        est = _expr(str(q.answer), **{stat_name: sympy.Symbol(stat_name)}).subs({n_: n, sympy.Symbol(stat_name): stat})
        est = sympy.N(est, 30)
        assert est > 0, (seed, text)
        score = sympy.N(sympy.diff(loglik, a_).subs(a_, est), 25)
        assert abs(score) < sympy.Float("1e-15"), (seed, text, q.answer, score)
        assert sympy.N(sympy.diff(loglik, a_, 2).subs(a_, est), 25) < 0, (seed, text)
        seen[dens] += 1
    assert len(seen) >= 6, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        text = q.text
        if "uniform distribution" in text:
            data = [int(v) for v in re.search(r"is ([\d, ]+)\.", text).group(1).split(", ")]
            if "maximum likelihood" in text:
                assert _frac(q) == max(data), (seed, text)
                seen["uniform-mle"] += 1
            else:
                assert _frac(q) == Fraction(2 * sum(data), len(data)), (seed, text)
                seen["uniform-mom"] += 1
        elif "normal distribution" in text:
            n, s1, s2 = (int(v) for v in re.search(r"size (\d+).*?Σx = (\d+) and Σx² = (\d+)", text).groups())
            assert _frac(q) == Fraction(s2, n) - Fraction(s1, n) ** 2, (seed, text)
            seen["normal"] += 1
        elif "(a + 1)x^a" in text:
            m = Fraction(
                re.search(r"mean is (\d+)/(\d+)", text).group(1) + "/" + re.search(r"mean is \d+/(\d+)", text).group(1)
            )
            assert _frac(q) == (2 * m - 1) / (1 - m), (seed, text)
            seen["mom-power"] += 1
        else:
            n = int(re.search(r"of (\d+) independent", text).group(1))
            total = int(re.search(r"total (\d+)", text).group(1))
            t = int(re.search(r"longer than (\d+)", text).group(1))
            expected = sympy.exp(-sympy.Rational(t * n, total))
            assert sympy.simplify(sympy.sympify(str(q.answer).replace("^", "**")) - expected) == 0, (seed, text)
            seen["tail"] += 1
    assert len(seen) == 5 and min(seen.values()) >= 8, seen


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=60)


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, seed).answer_text for seed in range(120)]
        assert len(set(answers)) >= 5, (tier, Counter(answers).most_common(3))


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 1, 2]
        assert q.parts[2].follow_through == "1/part_b"
        assert q.parts[3].follow_through.startswith("exp(-") and q.parts[3].follow_through.endswith("*part_b)")


def test_exam_keys_match_sympy():
    r = sympy.Symbol("r", positive=True)
    for seed in range(100):
        q = _exam(seed)
        text = stem(q)
        n = int(re.search(r"x_1, \.\.\., x_(\d+)", text).group(1))
        total = int(re.search(r"total (\d+) hours", text).group(1))
        loglik = sum(sympy.log(r) - r * total / n for _ in range(n))
        got = sympy.sympify(str(q.parts[0].answer).replace("^", "**"), locals={"r": r})
        assert sympy.simplify(got - loglik) == 0, (seed, text)
        (root,) = sympy.solve(sympy.diff(loglik, r), r)
        assert as_fraction(q.parts[1].answer) == Fraction(int(root.p), int(root.q))
        assert as_fraction(q.parts[2].answer) == Fraction(total, n)
        tt = int(re.search(r"longer than (\d+)", q.parts[3].prompt).group(1))
        assert as_fraction(q.parts[3].answer) == Fraction(f"{float(sympy.exp(-tt * root)):.3f}")


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=30)


def test_exam_follow_through_never_lands_on_a_distractor():
    check_no_follow_through_collision(_exam)


def test_exam_follow_through_carries_a_wrong_rate():
    check_follow_through(SK, _exam, "c", seeds=20)
