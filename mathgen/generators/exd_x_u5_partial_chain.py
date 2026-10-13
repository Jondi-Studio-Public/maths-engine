"""Exam-tier source for `partial-derivative-x` and `partial-derivative-y`: a partial derivative of a product.

The practice tiers differentiate a sum of monomials. Here the function is a product of three factors, one of them
a power of a bracket, ``f(x, y) = c x^m y^k (x + b y)^n``, so the learner must choose the product rule and the chain
rule, and the partial derivative is the first step of a three-part chain:

(a) the partial derivative of f with respect to the asked variable      expression, 2 marks
(b) its value at a given point P                                        fraction,   1 mark,  ``Subs(part_a, ...)``
(c) the linear estimate of the change in f for a small step in that
    variable, the other held fixed                                      fraction,   2 marks, ``part_b * step``

`var` selects the variable ("x" or "y"); both skills share the one function family and differ in which partial
derivative is asked. Every partial derivative is checked against the hand-built expression with SymPy.
"""

from __future__ import annotations

import functools
import random
from fractions import Fraction

import sympy
from sympy import Rational, diff, factor, symbols

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

x, y = symbols("x y")
_SUP = {1: "", 2: "²", 3: "³", 4: "⁴"}
_STEPS = (Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(1, 2))
_POINT = (-2, -1, 1, 2)


def _typed(expr) -> str:
    """The expression as a learner types it: ``*`` for products, ``^`` for powers."""
    return str(factor(expr)).replace("**", "^")


def _frac_option(tag, value) -> Distractor:
    return Distractor(rational_text(value), tag, exact_value(value))


def _fraction_part(prompt, value, marks, working, candidates, follow_through=None):
    """A `fraction` part whose answer is `value`; `candidates` are (misconception id or None, value) pairs."""
    value = Fraction(int(value.p), int(value.q)) if isinstance(value, sympy.Rational) else value
    text = rational_text(value)
    options = []
    for tag, v in candidates:
        v = Fraction(int(v.p), int(v.q)) if isinstance(v, sympy.Rational) else v
        if v != value:
            options.append(_frac_option(tag, v))
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, options, 3),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@functools.lru_cache(maxsize=None)
def _algebra(c0: int, m: int, k: int, n: int, b: int, var: str) -> dict:
    """The function, the asked partial derivative, the other one and the wrong candidates, with their typed forms.

    Depends only on the five integers and the variable (a few hundred combinations), so it is worked out once
    per combination: factoring each expression is the slow part of building a question.
    """
    s = x + b * y
    f = c0 * x**m * y**k * s**n
    fx, fy = diff(f, x), diff(f, y)
    # Independent hand-built product-rule forms.
    fx_hand = c0 * (m * x ** (m - 1) * y**k * s**n + n * x**m * y**k * s ** (n - 1))
    fy_hand = c0 * (k * x**m * y ** (k - 1) * s**n + n * b * x**m * y**k * s ** (n - 1))
    check(sympy.expand(fx - fx_hand) == 0, "f_x disagrees with the product rule")
    check(sympy.expand(fy - fy_hand) == 0, "f_y disagrees with the product rule")

    if var == "x":
        true, other = fx, fy
        wrong = [
            ("dropped-product-rule-term", c0 * n * x**m * y**k * s ** (n - 1)),
            ("dropped-product-rule-term", c0 * m * x ** (m - 1) * y**k * s**n),
            ("differentiated-wrong-variable", fy),
            ("summed-both-partials", fx + fy),
        ]
    else:
        true, other = fy, fx
        wrong = [
            ("dropped-product-rule-term", c0 * n * b * x**m * y**k * s ** (n - 1)),
            ("dropped-product-rule-term", c0 * k * x**m * y ** (k - 1) * s**n),
            ("differentiated-wrong-variable", fx),
            ("forgot-inner-derivative", c0 * (k * x**m * y ** (k - 1) * s**n + n * x**m * y**k * s ** (n - 1))),
        ]
    wrong = [(t, e) for t, e in wrong if sympy.expand(e - true) != 0]
    return dict(
        f=f,
        s=s,
        true=true,
        other=other,
        true_text=_typed(true),
        true_latex=latex_answer(true),
        wrong=[(t, e, _typed(e)) for t, e in wrong],
    )


def _draw(rng: random.Random, var: str) -> dict | None:
    c0 = rng.choice((1, 2, 3))
    m, k = rng.choice(((1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)))
    n = rng.choice((2, 3))
    b = rng.choice((-3, -2, -1, 1, 2, 3))
    p, q = rng.choice(_POINT), rng.choice(_POINT)
    step = rng.choice(_STEPS)

    algebra = _algebra(c0, m, k, n, b, var)
    f, s, true, other = algebra["f"], algebra["s"], algebra["true"], algebra["other"]
    at = {x: p, y: q}
    slope = Rational(true.subs(at))
    f_at = Rational(f.subs(at))
    if slope == 0 or f_at == 0 or s.subs(at) == 0:
        return None
    wrong = [(t, e) for t, e, _ in algebra["wrong"]]

    true_text = algebra["true_text"]
    a_options = [Distractor(text, t, str(e)) for t, e, text in algebra["wrong"]]
    part_a = dict(
        prompt=f"Find ∂f/∂{var}.",
        answer_kind="expression",
        answer=str(true),
        answer_text=true_text,
        answer_latex=algebra["true_latex"],
        marks=2,
        working=[
            f"Use the product rule on the factors that contain {var}, and the chain rule on the bracket.",
            f"∂f/∂{var} = {true_text}.",
        ],
        distractors=pick_distractors(true_text, a_options, 3),
    )

    delta = Rational(slope * Rational(step.numerator, step.denominator))
    b_cands = [("evaluates-function-not-derivative", f_at)]
    b_cands += [(t, Rational(e.subs(at))) for t, e in wrong]
    b_cands.append(("differentiated-wrong-variable", Rational(other.subs(at))))
    part_b = _fraction_part(
        f"Hence find the value of ∂f/∂{var} at P({p}, {q}).",
        slope,
        1,
        [f"At ({p}, {q}): ∂f/∂{var} = {rational_text(Fraction(int(slope.p), int(slope.q)))}."],
        b_cands,
        follow_through=f"Subs(part_a, (x, y), ({p}, {q}))",
    )
    step_text = f"{float(step):g}"
    part_c = _fraction_part(
        f"Use this to estimate the change in f when {var} increases by {step_text} from P and "
        f"{'y' if var == 'x' else 'x'} is held at {q if var == 'x' else p}.",
        delta,
        2,
        [f"Δf ≈ (∂f/∂{var}) × Δ{var} = {rational_text(Fraction(int(slope.p), int(slope.q)))} × {step_text}."],
        [
            ("divided-by-step", slope / Rational(step.numerator, step.denominator)),
            ("added-step-not-multiplied", slope + Rational(step.numerator, step.denominator)),
            ("evaluates-function-not-derivative", f_at * Rational(step.numerator, step.denominator)),
            ("differentiated-wrong-variable", Rational(other.subs(at)) * Rational(step.numerator, step.denominator)),
        ],
        follow_through=f"part_b*({step.numerator}/{step.denominator})",
    )
    if any(len(pt["distractors"]) < 3 for pt in (part_a, part_b, part_c)):
        return None

    inner = join_terms([(1, "x"), (b, "y")])
    f_text = f"{'' if c0 == 1 else c0}x{_SUP[m]}y{_SUP[k]}({inner}){_SUP[n]}"
    stem = phrase(
        rng,
        (
            "Let f(x, y) = {f}, and let P be the point ({p}, {q}).",
            "The function f is defined by f(x, y) = {f}. The point P has coordinates ({p}, {q}).",
            "Consider f(x, y) = {f} at the point P({p}, {q}).",
        ),
        f=f_text,
        p=p,
        q=q,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator("exd_x_u5_partial_chain", level=18, presets={"exam": dict(tier="exam")})
def exd_x_u5_partial_chain(rng: random.Random, tier: str = "exam", var: str = "x") -> dict:
    """Exam tier: a partial derivative of a product, its value at a point, and a linear estimate from it."""
    check(tier == "exam", f"unknown tier {tier!r}")
    check(var in ("x", "y"), f"unknown variable {var!r}")
    for _ in attempts(1000, "a function and point with three distinct distractors per part"):
        question = _draw(rng, var)
        if question is not None:
            return question
