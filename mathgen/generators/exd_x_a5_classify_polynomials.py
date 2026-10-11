"""Exam-tier source for polynomial classification: simplify first, then classify.

The practice tiers hand the learner a polynomial already written in standard form, so degree and term count can
be read straight off the page. Here the expression is NOT in standard form: it is a difference of expanded
powers or products, or an algebraic fraction that cancels. The learner has to choose to expand and collect (or
cancel), and only then read the degree and the number of terms. The naive reading of the unsimplified
expression is always a different, wrong answer.

Built backwards: each family fixes its parameters, builds the expression text, and the key is taken from the
simplified polynomial (``sympy.Poly`` total degree and term count). The unsimplified degree is recorded per
family and checked to differ from the true degree, so the "read it off the page" distractor can never equal
the key.

Families (random choice per seed)
    power_difference    (px + a)^3 - (px)^3 or (px + a)^3 - (px - a)^3 : the cubic terms cancel
    square_difference   (px + q)^2 - (px - q)^2 or (px + q)^2 - p^2x^2 : the quadratic terms cancel
    two_variable        (x^2 + ay)^2 - (x^2 - ay)^2, (xy + a)^2 - (xy - a)^2 or (x^2 + ay)(x^2 - ay) - x^4 : two variables
    cancelling_fraction (x^2 - a^2)/(x - a) or (x^3 - a^3)/(x - a) : divides out exactly
    product_minus_power (x + a)(x^2 + bx + c) - x^3 : the leading terms cancel

Distractors (each a named misconception)
    naive-degree         the degree of the expression as written, with the right term count
    naive-degree-adjacent  the same wrong degree with a neighbouring term-count label
    adjacent-type        the right degree with a neighbouring term-count label
    off-by-one           the right term count with the degree one too high or one too low
(three of these are drawn per seed)
    variable-denominator (fraction families only) "No - the variable is in the denominator"
"""

from __future__ import annotations

import random

from sympy import Poly, cancel, expand, symbols

from ..core import check, check_distinct, generator, join_terms, latex_answer, pick_distractors

x, y = symbols("x y")

_TYPE_NAMES = {1: "monomial", 2: "binomial", 3: "trinomial"}
_ADJACENT = {1: (2,), 2: (1, 3), 3: (2,)}
_NOT_POLY = "No - the variable is in the denominator"

_PROMPTS = (
    "Simplify fully, then classify the result by degree and number of terms: {expr}",
    "Write the expression in its simplest form, then state its degree and number of terms: {expr}",
)


def _lin(p: int, q: int, var: str = "x") -> str:
    """Linear form p*var + q as student text, e.g. '2x - 5'."""
    return join_terms([(p, var), (q, "")])


def _family_power_difference(rng: random.Random):
    p = rng.randint(1, 3)
    a = rng.randint(1, 5)
    if rng.random() < 0.5:
        expr = (p * x + a) ** 3 - (p * x) ** 3
        pt = "x" if p == 1 else f"{p}x"
        text = f"({_lin(p, a)})^3 - ({pt})^3"
    else:
        expr = (p * x + a) ** 3 - (p * x - a) ** 3
        text = f"({_lin(p, a)})^3 - ({_lin(p, -a)})^3"
    return expr, text, 3, False


def _family_square_difference(rng: random.Random):
    p = rng.randint(1, 4)
    q = rng.randint(1, 6)
    if rng.random() < 0.5:
        expr = (p * x + q) ** 2 - (p * x - q) ** 2
        text = f"({_lin(p, q)})^2 - ({_lin(p, -q)})^2"
    else:
        expr = (p * x + q) ** 2 - (p * x) ** 2
        pt = "x" if p == 1 else f"{p}x"
        text = f"({_lin(p, q)})^2 - ({pt})^2"
    return expr, text, 2, False


def _family_two_variable(rng: random.Random):
    a = rng.randint(1, 5)
    ay = "y" if a == 1 else f"{a}y"
    variant = rng.choice(["x2y", "xy", "conjugate"])
    if variant == "x2y":
        expr = (x**2 + a * y) ** 2 - (x**2 - a * y) ** 2
        text = f"(x^2 + {ay})^2 - (x^2 - {ay})^2"
    elif variant == "xy":
        expr = (x * y + a) ** 2 - (x * y - a) ** 2
        text = f"(xy + {a})^2 - (xy - {a})^2"
    else:
        expr = (x**2 + a * y) * (x**2 - a * y) - x**4
        text = f"(x^2 + {ay})(x^2 - {ay}) - x^4"
    return expr, text, 4, False


def _family_cancelling_fraction(rng: random.Random):
    a = rng.randint(2, 9)
    if rng.random() < 0.5:
        expr = x + a
        text = f"(x^2 - {a * a})/(x - {a})"
        naive = 2
    else:
        expr = x**2 + a * x + a * a
        text = f"(x^3 - {a**3})/(x - {a})"
        naive = 3
    return expr, text, naive, True


def _family_product_minus_power(rng: random.Random):
    a = rng.randint(1, 5)
    # a + b != 0 keeps the x^2 term, so the result is always degree 2 (never a constant).
    b = rng.choice([v for v in (-4, -3, -2, -1, 1, 2, 3, 4) if v != -a])
    c = rng.choice([-5, -3, -2, 1, 2, 4, 6])
    expr = (x + a) * (x**2 + b * x + c) - x**3
    text = f"({_lin(1, a)})({join_terms([(1, 'x^2'), (b, 'x'), (c, '')])}) - x^3"
    return expr, text, 3, False


_FAMILIES = (
    _family_power_difference,
    _family_square_difference,
    _family_two_variable,
    _family_cancelling_fraction,
    _family_product_minus_power,
)


def _label(degree: int, terms: int) -> str:
    return f"Degree {degree}, {_TYPE_NAMES[terms]}"


def _exam(rng: random.Random) -> dict:
    expr, expr_text, naive_degree, is_fraction = rng.choice(_FAMILIES)(rng)
    simplified = cancel(expand(expr)) if is_fraction else expand(expr)
    poly = Poly(simplified, x, y)
    degree = poly.total_degree()
    terms = len(poly.terms())
    check(degree >= 1, "a constant has no meaningful degree for this skill")
    check(terms in _TYPE_NAMES, "term count must be a monomial, binomial or trinomial")
    check_distinct(degree, naive_degree, msg="the unsimplified degree must differ from the true degree")

    answer_text = _label(degree, terms)
    adjacent = rng.choice(_ADJACENT[terms])
    pool = [
        _label(naive_degree, terms),
        _label(degree, adjacent),
        _label(naive_degree, adjacent),
        # off-by-one slips: the degree miscounted as one too low or one too high
        _label(degree + 1, terms),
    ]
    if degree > 1:
        pool.append(_label(degree - 1, terms))
    # Which wrong options appear is drawn per seed, so the key is not always the lowest-degree choice.
    pool = [o for o in dict.fromkeys(pool) if o != answer_text]
    candidates = rng.sample(pool, 3 if not is_fraction else 2)
    if is_fraction:
        candidates.insert(0, _NOT_POLY)

    text = rng.choice(_PROMPTS).format(expr=expr_text)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


@generator(
    "exd_x_a5_classify_polynomials",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a5_classify_polynomials(rng: random.Random, tier: str = "exam") -> dict:
    """Simplify an unsimplified expression, then classify it by degree and term count."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
