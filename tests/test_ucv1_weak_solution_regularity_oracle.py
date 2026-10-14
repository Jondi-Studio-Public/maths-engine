"""Independent oracle for `ucv1-weak-solution-regularity`.

Reads only `question.text`, the public option list and the public `mcq` grader. It identifies the equation, reads
the label of the data (given as "f is in Hk but not in Hk+1", or found from an explicit power with the stated
integrability criterion, cross-checked by computing the integral with SymPy), adds the order of the equation to
get the exact order of the solution, and decides which option is true. It does not import the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache

import pytest
import sympy

from mathgen import make
from mathgen.grading.typed import grade_mcq, mcq_options

TOPIC = "ucv1_weak_solution_regularity"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = 250

_SUPER = {"⁰": 0, "¹": 1, "²": 2, "³": 3, "⁴": 4, "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8, "⁹": 9}


def _order(token):
    """The Sobolev order of 'L²' (0) or 'H³' (3) or 'H¹⁰' (10)."""
    if token.startswith("L"):
        return 0
    return int("".join(str(_SUPER[c]) for c in token[1:]))


def _equation_order(text):
    return 4 if "u'''' =" in text else 2


@lru_cache(maxsize=None)
def _power_in_sobolev(alpha, k):
    """Whether |t|^alpha is in H^k near 0: the k-th derivative is a multiple of |t|^(alpha-k) (with a jump or delta
    when alpha is an integer not exceeding k - 1), so decide with the integral of its square near 0."""
    t = sympy.symbols("t", positive=True)
    if alpha.denominator == 1 and 0 <= alpha < k:
        return False  # the (alpha+1)-th derivative of |t|^alpha contains a Dirac delta
    exponent = 2 * (sympy.Rational(alpha.numerator, alpha.denominator) - k)
    return sympy.integrate(t**exponent, (t, 0, 1)).is_finite is True


def _label_of_data(text):
    m = re.search(r"The data satisfy \w ∈ (L²|H[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\(.*?\) but \w ∉ (L²|H[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\(", text)
    if m:
        k, k1 = _order(m.group(1)), _order(m.group(2))
        assert k1 == k + 1, text
        return k
    m = re.search(r"The data are \w\(x\) = \|x(?: [-+] [\d.]+)?\|(?:\^\(?(\d+(?:/\d+)?)\)?)?\.", text)
    assert m, text
    alpha = Fraction(m.group(1) or 1)
    k = max(j for j in range(0, 12) if _power_in_sobolev(alpha, j))
    assert _power_in_sobolev(alpha, k) and not _power_in_sobolev(alpha, k + 1)
    return k


def _statement_orders(option):
    m = re.fullmatch(
        r"u ∈ (L²|H[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\((-?\d+), (-?\d+)\), but u ∉ (L²|H[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\((-?\d+), (-?\d+)\)", option
    )
    assert m, option
    low, high = _order(m.group(1)), _order(m.group(4))
    assert (m.group(2), m.group(3)) == (m.group(5), m.group(6))
    return low, high


@pytest.mark.parametrize("tier", TIERS)
def test_exactly_one_option_is_true_and_it_is_the_key(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        expected = _label_of_data(q.text) + _equation_order(q.text)
        options = mcq_options(q)
        truths = []
        for option in options:
            low, high = _statement_orders(option)
            assert high == low + 1
            truths.append(low == expected)
        assert sum(truths) == 1, (q.text, options)
        assert options[truths.index(True)] == q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_the_option_domain_matches_the_question_domain(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        a, b = re.search(r"weak solution in H[¹²]₀\((-?\d+), (-?\d+)\)", q.text).groups()
        for option in mcq_options(q):
            assert f"({a}, {b})" in option


@pytest.mark.parametrize("tier", TIERS)
def test_the_grader_marks_the_true_option_correct_and_the_others_wrong(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        expected = _label_of_data(q.text) + _equation_order(q.text)
        for index, option in enumerate(mcq_options(q)):
            assert grade_mcq(q, index).correct == (_statement_orders(option)[0] == expected)


def test_the_stated_power_criterion_matches_direct_integration_for_every_exponent_used():
    # alpha > k - 1/2 is the criterion the question states; here it is checked against the integral for each
    # exponent the generator can draw (read from the questions themselves) and for the order range the options use.
    seen = set()
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = re.search(r"The data are \w\(x\) = \|x(?: [-+] [\d.]+)?\|(?:\^\(?(\d+(?:/\d+)?)\)?)?\.", q.text)
        seen.add(Fraction(m.group(1) or 1))
    assert seen == {Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(5, 2), Fraction(3), Fraction(7, 2)}
    for alpha in seen:
        for k in range(0, 10):
            assert _power_in_sobolev(alpha, k) == (alpha > k - Fraction(1, 2)), (alpha, k)


def _falling(g, i):
    out = Fraction(1)
    for j in range(i):
        out *= g - j
    return out


def _largest_order_of_even_singular_solution(g):
    """The largest m with u in H^m near 0 for u = c |t|^g, found from the derivatives themselves: u is in H^m when the
    derivatives below order m are continuous across 0 and the derivative of order m is square integrable."""

    def continuous(i):
        if g - i > 0:
            return True  # both one-sided limits are 0
        if g - i == 0:
            return i % 2 == 0  # the one-sided limits are c g! and (-1)^i c g!
        return _falling(g, i) == 0  # an exactly zero derivative; otherwise it blows up at 0

    def square_integrable(m):
        return _falling(g, m) == 0 or 2 * (g - m) > -1

    best = 0
    for m in range(0, 20):
        if all(continuous(i) for i in range(m)) and square_integrable(m):
            best = m
    return best


@pytest.mark.parametrize("gain", (2, 4))
def test_solving_the_equation_for_pure_power_data_gives_the_stated_gain(gain):
    # With data |t|^alpha, the even singular part of the solution of a second-order (gain 2) or fourth-order (gain 4)
    # equation is c |t|^(alpha + gain). Its exact Sobolev order, found from its own derivatives, is the order of the
    # data plus the gain.
    for alpha in (Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(5, 2), Fraction(3), Fraction(7, 2)):
        k = max(j for j in range(0, 12) if alpha > j - Fraction(1, 2))
        assert _largest_order_of_even_singular_solution(alpha + gain) == k + gain, (alpha, gain)
