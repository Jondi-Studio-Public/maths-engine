"""Tests `f10-algebra-conjecture-algorithm` (topic `fr2e_conjecture`, exam chain `exd_x_fr2e_conjecture`).

The oracle re-derives every answer from the question TEXT: easy substitutes scaled inputs into the stated formula with
SymPy; medium searches for the first whole number (or prime) at which the stated conjecture fails with `isprime`;
hard and the exam parse the rule and the table and compare them entry by entry.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.fr2e_conjecture_build  # noqa: F401  (registers the topics)
from tests.fr2e_support import (
    SEEDS,
    TIERS,
    assert_chain_shape,
    assert_clean_and_deterministic,
    assert_exam_grading,
    assert_follow_through,
    assert_registered,
    assert_single_grading,
    assert_tiers_differ,
    exam,
    single,
)

SKILL = "f10-algebra-conjecture-algorithm"
TOPIC = "fr2e_conjecture"
EXAM = "exd_x_fr2e_conjecture"
X, N = sympy.symbols("x n")
_T = standard_transformations + (implicit_multiplication_application,)

_FORMULAS = {
    "The area of a circle": "pi*r**2",
    "The volume of a cylinder": "pi*r**2*h",
    "The kinetic energy": "m*v**2/2",
    "The volume of a cube": "s**3",
    "The power used": "I**2*R",
    "The time to fill": "V/R",
    "The speed of a runner": "d/t",
    "The gravitational force": "G*M*m/d**2",
    "The light from a lamp": "P/d**2",
}
_CHANGES = {"doubled": 2, "tripled": 3, "halved": Fraction(1, 2), "multiplied by 4": 4, "divided by 3": Fraction(1, 3)}


def _expr(text: str, symbol):
    body = text.replace("×", "*").replace("^", "**")
    return parse_expr(body, local_dict={symbol.name: symbol}, transformations=_T)


def _easy_factor(text: str) -> Fraction:
    formula = next(expr for prefix, expr in _FORMULAS.items() if text.startswith(prefix))
    f = sympy.sympify(
        formula, locals={name: sympy.Symbol(name) for name in "IRVdtGMmPvshr"}
    )  # I is the current, not the imaginary unit
    scaled = {}
    for symbol, change in re.findall(
        r"the (?:(?!is )[a-z]+ ){0,2}(\w) is (doubled|tripled|halved|multiplied by 4|divided by 3)", text, re.I
    ):
        scaled[sympy.Symbol(symbol)] = (
            sympy.Rational(_CHANGES[change].numerator, _CHANGES[change].denominator)
            if isinstance(_CHANGES[change], Fraction)
            else _CHANGES[change]
        )
    assert scaled, text
    ratio = sympy.simplify(f.subs({s: s * k for s, k in scaled.items()}, simultaneous=True) / f)
    return Fraction(int(ratio.p), int(ratio.q))


def test_registered_sources():
    assert_registered(SKILL, TOPIC, EXAM)


def test_easy_factor_matches_substitution():
    changes = set()
    for seed in SEEDS:
        q = single(TOPIC, "easy", seed)
        assert Fraction(q.answer_text) == _easy_factor(q.text), (seed, q.text, q.answer_text)
        assert Fraction(q.answer_text) != 1
        assert all(Fraction(d) != Fraction(q.answer_text) for d in q.distractors)
        changes.add(len(re.findall(r" is (?:doubled|tripled|halved|multiplied by 4|divided by 3)", q.text)))
    assert changes == {1, 2}


def test_medium_smallest_counterexample():
    kinds = set()
    for seed in SEEDS:
        q = single(TOPIC, "medium", seed)
        if "2^n - 1" in q.text:
            kinds.add("mersenne")
            first = next(n for n in range(2, 200) if sympy.isprime(n) and not sympy.isprime(2**n - 1))
        else:
            kinds.add("polynomial")
            body = re.search(r"the expression (.+?) gives a prime number", q.text).group(1)
            f = _expr(body, N)
            first = next(n for n in range(1, 200) if not sympy.isprime(int(f.subs(N, n))))
        assert int(q.answer_text) == first, (seed, q.text, q.answer_text)
        assert first >= 4
    assert "polynomial" in kinds


def _rule_and_table(text: str):
    xs, ys = re.search(r"x: ([\d, ]+); y: ([\d, ]+)", text).groups()
    segment = text[: text.index(" x: ")]
    body = re.split(r" gives every|\.$", segment.rsplit("y = ", 1)[1])[0]
    return _expr(body, X), [int(v) for v in xs.split(",")], [int(v) for v in ys.split(",")]


def test_hard_finds_the_one_broken_entry():
    kinds = set()
    for seed in SEEDS:
        q = single(TOPIC, "hard", seed)
        rule, xs, ys = _rule_and_table(q.text)
        broken = [a for a, b in zip(xs, ys) if rule.subs(X, a) != b]
        assert len(broken) == 1, (seed, q.text)
        assert int(q.answer_text) == broken[0]
        kinds.add("exp" if any(p.exp.has(X) for p in rule.atoms(sympy.Pow)) else "quad")
    assert kinds == {"exp", "quad"}


@pytest.mark.parametrize("tier", TIERS)
def test_tiers_differ_in_shape(tier):
    assert_tiers_differ(TOPIC)
    questions = [single(TOPIC, tier, seed) for seed in SEEDS]
    marker = {"easy": "By what factor", "medium": "smallest", "hard": "breaks the conjecture"}[tier]
    assert all(marker in q.text for q in questions)


def test_clean_and_deterministic():
    assert_clean_and_deterministic(TOPIC, TIERS)
    assert_clean_and_deterministic(EXAM, ("exam",))


def test_single_grading():
    assert_single_grading(SKILL, TOPIC)


def test_exam_matches_oracle():
    for seed in range(80):
        q = exam(EXAM, seed)
        rule, xs, ys = _rule_and_table(q.text)
        n = int(re.search(r"tests the rule at x = (\d+)", q.text).group(1))
        predicted = int(rule.subs(X, n))
        observed = ys[xs.index(n)]
        a, b, c = q.parts
        assert int(a.answer_text) == predicted
        assert int(b.answer_text) == observed - predicted
        breaks = [x for x, y in zip(xs, ys) if rule.subs(X, x) != y]
        assert breaks in ([], [n]), (seed, q.text)
        if observed != predicted:
            assert c.answer_text.startswith(f"x = {n} is a counterexample, so the conjecture is false")
        else:
            assert c.answer_text.startswith(f"The rule fits at x = {n}, but that does not prove it")


def test_exam_chain_shape():
    assert_chain_shape(EXAM, ["number", "number", "mcq"], [1, 1, 2], [False, True, False])


def test_exam_grading():
    assert_exam_grading(SKILL, EXAM)


def test_exam_covers_both_a_counterexample_and_a_fit():
    outcomes = {exam(EXAM, seed).parts[2].answer_text.split(",")[0].split(" ")[0] for seed in range(60)}
    assert outcomes == {"x", "The"}


def test_difference_follows_through_from_a_wrong_prediction():
    def difference(q, canon, wrong_prediction):
        rule, xs, ys = _rule_and_table(q.text)
        n = int(re.search(r"tests the rule at x = (\d+)", q.text).group(1))
        return str(ys[xs.index(n)] - int(wrong_prediction))

    assert_follow_through(SKILL, EXAM, "b", "a", difference)
