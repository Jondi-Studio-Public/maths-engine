"""Generates `fy1ca_hyperbolic` and `fy1ca_hyperbolic_exam` questions.

One first-year skill, ``ucv1-hyperbolic-functions``: the hyperbolic functions sinh, cosh and tanh from their definitions
in terms of e^x, the identity cosh²x - sinh²x = 1, their derivatives and integrals, and solving equations such as
cosh x = a with logarithms. Every key is graded as an expression (an exact value, or an expression in x for a
derivative).

Tiers (the reasoning escalates):

- ``easy``: values from the definitions: sinh, cosh or tanh of ln k, or the other hyperbolic ratios when one of them is
  given and x > 0.
- ``medium``: derivatives, as an expression in x: the chain rule on sinh(ax) and cosh(ax), the product rule,
  ln(cosh x), cosh(x²), sinh x cosh x and tanh(ax).
- ``hard``: solving cosh x = a or sinh x = a for x as a logarithm, solving a cosh x + b sinh x = c through a quadratic in
  e^x, and definite integrals of hyperbolic functions over [0, ln k].
- ``exam`` (topic ``fy1ca_hyperbolic_exam``): a four-part chain from cosh x = c with x > 0: sinh x, tanh x, the value
  of x to 3 decimal places, and the exact value of cosh 2x.

Every key is re-derived by a second route, using floats: the definition with math.exp, central differences for a
derivative, Simpson's rule for an integral, and substitution of the solution back into the equation.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy
from sympy import Rational, cosh, log, sinh, sqrt, tanh

from ..core import check, generator, latex_answer, phrase
from ._fy1ca_kit import (
    Redraw,
    build_with_redraws,
    clear_of_rounding,
    close,
    expression_distractors,
    fraction_distractors,
    fraction_part,
    need_distractors,
    number_distractors,
    number_part,
    simpson,
    symbolic_distractors,
    typed,
)

x = sympy.Symbol("x", real=True)


def _show(value) -> str:
    text = typed(sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value)
    return text.replace("*", " × ")


def _nice(candidates):
    kept = []
    for value, tag in candidates:
        if value is None:
            continue
        value = sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value
        if value.is_Rational and value.q > 400:
            continue
        if len(typed(value)) > 22:
            continue
        kept.append((value, tag))
    return kept


def _result(text, expr, distractors, working) -> dict:
    need_distractors(distractors)
    return dict(
        text=text,
        answer=str(expr),
        answer_text=typed(expr),
        answer_kind="expression",
        answer_latex=latex_answer(expr),
        working=working,
        distractors=distractors,
    )


def _question(text, answer, wrong, working) -> dict:
    expr = sympy.nsimplify(answer) if not isinstance(answer, sympy.Basic) else answer
    return _result(text, expr, expression_distractors(expr, _nice(wrong)), working)


# -- easy ------------------------------------------------------------------------------------------------------------

# (cosh x, sinh x) pairs of rationals with cosh² - sinh² = 1; e^x = cosh + sinh
_TRIPLES = (
    (Rational(5, 4), Rational(3, 4)),
    (Rational(5, 3), Rational(4, 3)),
    (Rational(13, 12), Rational(5, 12)),
    (Rational(13, 5), Rational(12, 5)),
    (Rational(17, 8), Rational(15, 8)),
    (Rational(25, 24), Rational(7, 24)),
    (Rational(17, 15), Rational(8, 15)),
    (Rational(25, 7), Rational(24, 7)),
)


_EXAM_U = tuple(
    Rational(v) for v in ("2", "3", "4", "5", "6", "7", "3/2", "5/2", "7/2", "4/3", "5/3", "5/4", "7/3", "7/4")
)


def _value_at_log(rng: random.Random) -> dict:
    fn = rng.choice(["sinh", "cosh", "tanh"])
    k = rng.choice([2, 3, 4, 5, 6])
    kk = Rational(k)
    values = {"sinh": (kk - 1 / kk) / 2, "cosh": (kk + 1 / kk) / 2, "tanh": (kk**2 - 1) / (kk**2 + 1)}
    answer = values[fn]
    reference = {"sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh}[fn](math.log(k))
    check(close(reference, float(answer), 1e-9), "float evaluation disagrees")
    if fn == "sinh":
        wrong = [
            (kk - 1 / kk, "fy1ca-hyp-half-forgotten"),
            ((kk + 1 / kk) / 2, "fy1ca-hyp-sinh-cosh-swapped"),
            (kk, "fy1ca-hyp-negative-exponent"),
            ((kk - 1 / kk) / 4, "fy1ca-hyp-half-doubled"),
        ]
    elif fn == "cosh":
        wrong = [
            (kk + 1 / kk, "fy1ca-hyp-half-forgotten"),
            ((kk - 1 / kk) / 2, "fy1ca-hyp-sinh-cosh-swapped"),
            (kk, "fy1ca-hyp-negative-exponent"),
            ((kk + 1 / kk) / 4, "fy1ca-hyp-half-doubled"),
        ]
    else:
        wrong = [
            ((kk**2 + 1) / (kk**2 - 1), "fy1ca-hyp-tanh-inverted"),
            ((kk - 1) / (kk + 1), "fy1ca-hyp-squares-forgotten"),
            ((kk - 1 / kk) / 2, "fy1ca-hyp-sinh-for-tanh"),
            ((kk**2 - 1) / (2 * kk**2), "fy1ca-hyp-denominator-slip"),
        ]
    text = phrase(
        rng,
        (
            "Find the exact value of {fn}(ln {k}).",
            "Use the definition of {fn} x in terms of e^x to evaluate {fn}(ln {k}) exactly.",
        ),
        fn=fn,
        k=k,
    )
    definition = {
        "sinh": "sinh x = (e^x - e^(-x))/2",
        "cosh": "cosh x = (e^x + e^(-x))/2",
        "tanh": "tanh x = (e^x - e^(-x))/(e^x + e^(-x))",
    }[fn]
    working = [
        f"{definition}, with e^(ln {k}) = {k} and e^(-ln {k}) = 1/{k}.",
        {
            "sinh": f"sinh(ln {k}) = ({k} - 1/{k})/2.",
            "cosh": f"cosh(ln {k}) = ({k} + 1/{k})/2.",
            "tanh": f"tanh(ln {k}) = ({k} - 1/{k})/({k} + 1/{k}).",
        }[fn],
        f"{fn}(ln {k}) = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _from_identity(rng: random.Random) -> dict:
    c, s = rng.choice(_TRIPLES)
    ask = rng.choice(["sinh x", "tanh x"])
    answer = s if ask == "sinh x" else s / c
    check(
        close(
            math.sinh(math.acosh(float(c))) if ask == "sinh x" else math.tanh(math.acosh(float(c))), float(answer), 1e-9
        ),
        "float evaluation disagrees",
    )
    if ask == "sinh x":
        wrong = [
            (c**2 - 1, "fy1ca-hyp-root-forgotten"),
            (-s, "fy1ca-hyp-sign-of-sinh"),
            (c - 1, "fy1ca-hyp-linear-slip"),
            (1 / c, "fy1ca-hyp-reciprocal"),
            (c * s, "fy1ca-hyp-product-for-ratio"),
        ]
    else:
        wrong = [
            (c / s, "fy1ca-hyp-tanh-inverted"),
            (s * c, "fy1ca-hyp-product-for-ratio"),
            (s, "fy1ca-hyp-sinh-for-tanh"),
            (s / c**2, "fy1ca-hyp-denominator-slip"),
            (1 / c, "fy1ca-hyp-reciprocal"),
        ]
    text = phrase(
        rng,
        (
            "Given that cosh x = {c} and x > 0, find the exact value of {ask}.",
            "If cosh x = {c} and x is positive, use the identity cosh²x - sinh²x = 1 to find the exact value of {ask}.",
        ),
        c=_show(c),
        ask=ask,
    )
    working = [
        f"cosh²x - sinh²x = 1 gives sinh²x = ({_show(c)})² - 1 = {_show(s**2)}.",
        f"Since x > 0, sinh x is positive, so sinh x = {_show(s)}.",
    ] + ([] if ask == "sinh x" else [f"tanh x = sinh x ÷ cosh x = {_show(s)} ÷ {_show(c)} = {_show(answer)}."])
    return _question(text, answer, wrong, working)


def _easy(rng: random.Random) -> dict:
    return rng.choice([_value_at_log, _from_identity])(rng)


# -- medium --------------------------------------------------------------------------------------------------------------


def _derivative(rng: random.Random) -> dict:
    kind = rng.choice(["sinh_k", "cosh_k", "product", "log_cosh", "cosh_sq", "sinh_cosh", "tanh_k"])
    a = rng.choice([2, 3, 4, 5])
    if kind == "sinh_k":
        f, ftext, d = sinh(a * x), f"sinh({a}x)", a * cosh(a * x)
        wrong = [
            (cosh(a * x), "fy1ca-hyp-chain-rule-forgotten"),
            (-a * cosh(a * x), "fy1ca-hyp-derivative-sign"),
            (a * sinh(a * x), "fy1ca-hyp-derivative-unchanged"),
            (cosh(a * x) / a, "fy1ca-hyp-chain-divided"),
        ]
        steps = [f"d/dx sinh(u) = cosh(u) × du/dx with u = {a}x and du/dx = {a}."]
    elif kind == "cosh_k":
        f, ftext, d = cosh(a * x), f"cosh({a}x)", a * sinh(a * x)
        wrong = [
            (sinh(a * x), "fy1ca-hyp-chain-rule-forgotten"),
            (-a * sinh(a * x), "fy1ca-hyp-derivative-sign"),
            (a * cosh(a * x), "fy1ca-hyp-derivative-unchanged"),
            (sinh(a * x) / a, "fy1ca-hyp-chain-divided"),
        ]
        steps = [f"d/dx cosh(u) = sinh(u) × du/dx with u = {a}x and du/dx = {a}."]
    elif kind == "product":
        which = rng.choice(["sinh", "cosh"])
        g, gd = (sinh(x), cosh(x)) if which == "sinh" else (cosh(x), sinh(x))
        f, ftext, d = x * g, f"x {which} x", g + x * gd
        wrong = [
            (gd, "fy1ca-hyp-product-rule-forgotten"),
            (x * gd, "fy1ca-hyp-product-rule-forgotten"),
            (g - x * gd, "fy1ca-hyp-product-rule-sign"),
            (g * gd, "fy1ca-hyp-product-rule-forgotten"),
        ]
        steps = [f"Product rule: d/dx (x {which} x) = 1 × {which} x + x × {'cosh' if which == 'sinh' else 'sinh'} x."]
    elif kind == "log_cosh":
        f, ftext, d = log(cosh(x)), "ln(cosh x)", tanh(x)
        wrong = [
            (1 / cosh(x), "fy1ca-hyp-log-derivative-slip"),
            (sinh(x), "fy1ca-hyp-chain-rule-forgotten"),
            (-tanh(x), "fy1ca-hyp-derivative-sign"),
            (1 / tanh(x), "fy1ca-hyp-tanh-inverted"),
        ]
        steps = ["d/dx ln(u) = u'/u with u = cosh x and u' = sinh x, so the derivative is sinh x ÷ cosh x = tanh x."]
    elif kind == "cosh_sq":
        f, ftext, d = cosh(x**2), "cosh(x²)", 2 * x * sinh(x**2)
        wrong = [
            (sinh(x**2), "fy1ca-hyp-chain-rule-forgotten"),
            (2 * x * cosh(x**2), "fy1ca-hyp-derivative-unchanged"),
            (-2 * x * sinh(x**2), "fy1ca-hyp-derivative-sign"),
            (x * sinh(x**2), "fy1ca-hyp-chain-divided"),
        ]
        steps = ["d/dx cosh(u) = sinh(u) × du/dx with u = x² and du/dx = 2x."]
    elif kind == "sinh_cosh":
        f, ftext, d = sinh(x) * cosh(x), "sinh x cosh x", cosh(x) ** 2 + sinh(x) ** 2
        wrong = [
            (cosh(x) ** 2 - sinh(x) ** 2, "fy1ca-hyp-product-rule-sign"),
            (2 * sinh(x) * cosh(x), "fy1ca-hyp-product-rule-forgotten"),
            (cosh(x) * sinh(x), "fy1ca-hyp-product-rule-forgotten"),
            (-(cosh(x) ** 2) - sinh(x) ** 2, "fy1ca-hyp-derivative-sign"),
        ]
        steps = ["Product rule: d/dx (sinh x cosh x) = cosh x × cosh x + sinh x × sinh x."]
    else:
        f, ftext, d = tanh(a * x), f"tanh({a}x)", a / cosh(a * x) ** 2
        wrong = [
            (a / cosh(a * x), "fy1ca-hyp-sech-squared-slip"),
            (-a / cosh(a * x) ** 2, "fy1ca-hyp-derivative-sign"),
            (a / sinh(a * x) ** 2, "fy1ca-hyp-tanh-derivative-confused"),
            (1 / cosh(a * x) ** 2, "fy1ca-hyp-chain-rule-forgotten"),
        ]
        steps = [f"d/dx tanh(u) = (1/cosh²u) × du/dx with u = {a}x and du/dx = {a}."]
    # second route: central differences of the function itself
    fn = sympy.lambdify(x, f, "math")
    dn = sympy.lambdify(x, d, "math")
    h = 1e-6
    for point in (0.4, 0.9, 1.3):
        check(close((fn(point + h) - fn(point - h)) / (2 * h), dn(point), 1e-6), "numerical derivative disagrees")
    distractors = symbolic_distractors(d, wrong, x)
    text = phrase(
        rng,
        (
            "Differentiate f(x) = {f} with respect to x.",
            "Find f'(x) when f(x) = {f}.",
        ),
        f=ftext,
    )
    working = steps + [f"f'(x) = {_show(d)}."]
    return _result(text, d, distractors, working)


def _medium(rng: random.Random) -> dict:
    return _derivative(rng)


# -- hard ----------------------------------------------------------------------------------------------------------------


def _solve_single(rng: random.Random) -> dict:
    c, s = rng.choice(_TRIPLES)
    use_cosh = rng.random() < 0.5
    a = c if use_cosh else s
    e_x = c + s
    answer = log(e_x)
    xv = math.log(float(e_x))
    check(close(math.cosh(xv) if use_cosh else math.sinh(xv), float(a), 1e-9), "solution does not satisfy the equation")
    other = log(a + sqrt(a**2 + 1)) if use_cosh else log(a + sqrt(a**2 - 1)) if a > 1 else None
    wrong = [
        (e_x, "fy1ca-hyp-exponential-not-logged"),
        (log(a), "fy1ca-hyp-ln-of-rhs"),
        (log(c - s), "fy1ca-hyp-negative-root"),
        (other, "fy1ca-hyp-identity-sign"),
        (log(2 * a), "fy1ca-hyp-half-forgotten"),
    ]
    name = "cosh" if use_cosh else "sinh"
    text = phrase(
        rng,
        (
            "Solve {name} x = {a} for x > 0, giving your answer as an exact value.",
            "Find the positive value of x for which {name} x = {a}. Give the exact value.",
        ),
        name=name,
        a=_show(a),
    )
    working = [
        f"Write {name} x = (e^x {'+' if use_cosh else '-'} e^(-x))/2 = {_show(a)} and let u = e^x: u {'+' if use_cosh else '-'} 1/u = {_show(2 * a)}.",
        f"Multiplying by u: u² - {_show(2 * a)}u {'+' if use_cosh else '-'} 1 = 0, so u = {_show(a)} ± {_show(s if use_cosh else c)}, and the positive value giving x > 0 is u = {_show(e_x)}.",
        f"x = ln({_show(e_x)}) = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _solve_combination(rng: random.Random) -> dict:
    a, b = rng.choice([(5, 3), (5, 4), (4, 3), (5, -3), (3, 1), (5, 1)])
    u0 = rng.choice([2, 3, 4])
    u1 = Rational(a - b, (a + b) * u0)
    if u1 >= u0 or u1 <= 0:
        raise Redraw
    c = Rational(a + b) * (u0 + u1) / 2
    answer = log(u0)
    # both roots of (a + b)u^2 - 2cu + (a - b) = 0 satisfy the equation; u0 is the larger
    for u in (u0, float(u1)):
        xv = math.log(float(u))
        check(close(a * math.cosh(xv) + b * math.sinh(xv), float(c), 1e-9), "root does not satisfy the equation")
    wrong = [
        (log(u1), "fy1ca-hyp-other-root"),
        (u0, "fy1ca-hyp-exponential-not-logged"),
        (log(c), "fy1ca-hyp-ln-of-rhs"),
        (log(c / a), "fy1ca-hyp-coefficient-slip"),
        (log(2 * c / (a + b)), "fy1ca-hyp-sum-of-roots"),
    ]
    sign = "+" if b > 0 else "-"
    bb = abs(b)
    coeff = lambda v: "" if v == 1 else str(v)
    text = phrase(
        rng,
        (
            "Solve {a} cosh x {sign} {bb} sinh x = {c}. Give the larger solution as an exact value.",
            "The equation {a} cosh x {sign} {bb} sinh x = {c} has two solutions. Find the larger one exactly.",
        ),
        a=a if a != 1 else "",
        sign=sign,
        bb=coeff(bb),
        c=_show(c),
    )
    text = text.replace("  ", " ").strip()
    working = [
        f"Substitute cosh x = (u + 1/u)/2 and sinh x = (u - 1/u)/2 with u = e^x, then multiply through by 2u.",
        f"This gives {_show(a + b)}u² - {_show(2 * c)}u + {_show(a - b)} = 0, with roots u = {u0} and u = {_show(u1)}.",
        f"The larger solution is x = ln({u0}) = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _integral(rng: random.Random) -> dict:
    kind = rng.choice(["sech2", "cosh2x", "sinh_cosh", "cosh_sq"])
    k = rng.choice([2, 3, 4])
    kk = Rational(k)
    if kind == "sech2":
        f, ftext = 1 / cosh(x) ** 2, "1/cosh²x"
        answer = (kk**2 - 1) / (kk**2 + 1)
        wrong = [
            (log(cosh(log(kk))), "fy1ca-hyp-antiderivative-slip"),
            ((kk**2 - 1) / (2 * kk), "fy1ca-hyp-sinh-for-tanh"),
            ((kk**2 + 1) / (kk**2 - 1), "fy1ca-hyp-tanh-inverted"),
            (1 / cosh(log(kk)), "fy1ca-hyp-antiderivative-slip"),
        ]
        steps = ["The integral of 1/cosh²x is tanh x.", f"tanh(ln {k}) - tanh 0 = ({k}² - 1)/({k}² + 1)."]
    elif kind == "cosh2x":
        f, ftext = cosh(2 * x), "cosh(2x)"
        answer = (kk**2 - 1 / kk**2) / 4
        wrong = [
            ((kk**2 - 1 / kk**2) / 2, "fy1ca-hyp-chain-divided-forgotten"),
            ((kk**2 - 1 / kk**2) / 8, "fy1ca-hyp-chain-divided"),
            ((kk**2 + 1 / kk**2) / 4, "fy1ca-hyp-sinh-cosh-swapped"),
            ((kk - 1 / kk) / 2, "fy1ca-hyp-chain-rule-forgotten"),
        ]
        steps = ["The integral of cosh(2x) is (1/2)sinh(2x).", f"(1/2) sinh(2 ln {k}) = (1/4)({k}² - 1/{k}²)."]
    elif kind == "sinh_cosh":
        f, ftext = sinh(x) * cosh(x), "sinh x cosh x"
        answer = (kk - 1 / kk) ** 2 / 8
        wrong = [
            ((kk - 1 / kk) ** 2 / 4, "fy1ca-hyp-half-forgotten"),
            ((kk - 1 / kk) / 2 * (kk + 1 / kk) / 2, "fy1ca-hyp-value-for-integral"),
            ((kk + 1 / kk) ** 2 / 8, "fy1ca-hyp-sinh-cosh-swapped"),
            ((kk - 1 / kk) ** 2 / 2, "fy1ca-hyp-half-doubled"),
        ]
        steps = [
            "Since d/dx (sinh²x) = 2 sinh x cosh x, the integral is (1/2) sinh²x.",
            f"(1/2) sinh²(ln {k}) = (1/2)(({k} - 1/{k})/2)².",
        ]
    else:
        f, ftext = cosh(x) ** 2, "cosh²x"
        answer = log(kk) / 2 + (kk**2 - 1 / kk**2) / 8
        wrong = [
            (log(kk) / 2, "fy1ca-hyp-term-dropped"),
            ((kk**2 - 1 / kk**2) / 8, "fy1ca-hyp-term-dropped"),
            (log(kk) + (kk**2 - 1 / kk**2) / 4, "fy1ca-hyp-half-forgotten"),
            (log(kk) / 2 - (kk**2 - 1 / kk**2) / 8, "fy1ca-hyp-identity-sign"),
            ((kk**3 - 1 / kk**3) / 3, "fy1ca-hyp-power-rule-slip"),
        ]
        steps = [
            "Use cosh²x = (1 + cosh 2x)/2; the integral is x/2 + sinh(2x)/4.",
            f"At x = ln {k}: (ln {k})/2 + ({k}² - 1/{k}²)/8.",
        ]
    fn = sympy.lambdify(x, f, "math")
    check(close(simpson(fn, 0.0, math.log(k), 2400), float(sympy.N(answer)), 1e-8), "numerical integral disagrees")
    text = phrase(
        rng,
        (
            "Evaluate the integral of {f} dx from x = 0 to x = ln {k}, giving the exact value.",
            "Find the exact value of the definite integral of {f} with respect to x between x = 0 and x = ln {k}.",
        ),
        f=ftext,
        k=k,
    )
    working = steps + [f"The value is {_show(answer)}."]
    return _question(text, answer, wrong, working)


def _hard(rng: random.Random) -> dict:
    return rng.choice([_solve_single, _solve_combination, _integral])(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fy1ca_hyperbolic", level=19, presets={t: dict(tier=t) for t in _TIERS})
def fy1ca_hyperbolic(rng: random.Random, tier: str = "easy") -> dict:
    """Hyperbolic values (easy), derivatives (medium), equations and integrals (hard)."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build_with_redraws(rng, _TIERS[tier], f"a {tier} hyperbolic question")


# -- exam ----------------------------------------------------------------------------------------------------------------


def _exam_build(rng: random.Random) -> dict:
    u = rng.choice(_EXAM_U)
    c, s = (u + 1 / u) / 2, (u - 1 / u) / 2
    e_x = c + s
    x_value = math.log(float(e_x))
    check(close(math.cosh(x_value), float(c), 1e-9), "x does not satisfy cosh x = c")
    tanh_x = s / c
    cosh2 = 1 + 2 * s**2
    check(close(math.cosh(2 * x_value), float(cosh2), 1e-9), "cosh 2x disagrees")
    if not clear_of_rounding(x_value, 3):
        raise Redraw

    def frac(v):
        v = Rational(v)
        return Fraction(int(v.p), int(v.q))

    a_dist = fraction_distractors(
        frac(s),
        [
            (c**2 - 1, "fy1ca-hyp-root-forgotten"),
            (-s, "fy1ca-hyp-sign-of-sinh"),
            (c - 1, "fy1ca-hyp-linear-slip"),
            (1 / c, "fy1ca-hyp-reciprocal"),
        ],
    )
    b_dist = fraction_distractors(
        frac(tanh_x),
        [
            (c / s, "fy1ca-hyp-tanh-inverted"),
            (s * c, "fy1ca-hyp-product-for-ratio"),
            (s, "fy1ca-hyp-sinh-for-tanh"),
            (s / c**2, "fy1ca-hyp-denominator-slip"),
        ],
    )
    c_dist = number_distractors(
        x_value,
        [
            (float(log(c)), "fy1ca-hyp-ln-of-cosh"),
            (float(log(s)), "fy1ca-hyp-ln-of-sinh"),
            (float(e_x), "fy1ca-hyp-exponential-not-logged"),
            (float(log(c + sqrt(c**2 + 1))), "fy1ca-hyp-identity-sign"),
            (float(log(c - s)), "fy1ca-hyp-negative-root"),
        ],
        3,
    )
    d_dist = fraction_distractors(
        frac(cosh2),
        [
            (2 * s * c, "fy1ca-hyp-double-angle-confusion"),
            (c**2 - s**2, "fy1ca-hyp-identity-misapplied"),
            (2 * c, "fy1ca-hyp-double-angle-linear"),
            (2 * c**2, "fy1ca-hyp-double-angle-constant-dropped"),
            (1 + 2 * s, "fy1ca-hyp-double-angle-linear"),
        ],
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        need_distractors(dist)
    stem = phrase(
        rng,
        (
            "The positive number x satisfies cosh x = {c}.",
            "Let x be the positive number for which cosh x = {c}.",
            "A hanging cable has a catenary shape, and x is the positive number with cosh x = {c}.",
        ),
        c=_show(c),
    )
    parts = [
        fraction_part(
            "Find the exact value of sinh x.",
            frac(s),
            2,
            [
                f"sinh²x = cosh²x - 1 = ({_show(c)})² - 1 = {_show(s**2)}, and sinh x > 0 because x > 0, so sinh x = {_show(s)}."
            ],
            a_dist,
        ),
        fraction_part(
            "Hence find the exact value of tanh x.",
            frac(tanh_x),
            1,
            [f"tanh x = sinh x ÷ cosh x = {_show(s)} ÷ {_show(c)} = {_show(tanh_x)}."],
            b_dist,
            follow_through=f"part_a/({_show(c)})",
        ),
        number_part(
            "Find x, correct to 3 decimal places.",
            x_value,
            3,
            2,
            [
                f"e^x = cosh x + sinh x = {_show(c)} + {_show(s)} = {_show(e_x)}, so x = ln({_show(e_x)}) = {x_value:.3f}."
            ],
            c_dist,
            follow_through=f"log({_show(c)} + part_a)",
            sensible_range=(0, None),
        ),
        fraction_part(
            "Hence find the exact value of cosh 2x.",
            frac(cosh2),
            2,
            [f"cosh 2x = 1 + 2sinh²x = 1 + 2({_show(s)})² = {_show(cosh2)}."],
            d_dist,
            follow_through="1 + 2*part_a**2",
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("fy1ca_hyperbolic_exam", level=19, presets={"exam": {}})
def fy1ca_hyperbolic_exam(rng: random.Random) -> dict:
    """From cosh x = c with x > 0: sinh x, tanh x, x to 3 decimal places, and cosh 2x."""
    return build_with_redraws(rng, _exam_build, "a hyperbolic exam chain")
