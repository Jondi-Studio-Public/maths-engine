"""Two-point Gauss-Legendre quadrature: `r4b_gauss_legendre` (easy, medium, hard) and the exam chain
`exd_x_r4b_gauss_legendre`.

``uc2-gauss-legendre-2pt`` (``numerical-and-computational-mathematics``). The rule is
``∫ₐᵇ f ≈ ((b − a)/2)·[f(m − h/√3) + f(m + h/√3)]`` with ``m = (a + b)/2`` and ``h = (b − a)/2``.

* easy: a quartic on ``[−1, 1]``, where the nodes are ±1/√3 themselves; the answer is an exact fraction (the sum of a
  polynomial at two conjugate nodes is rational).
* medium: a quartic on ``[a, b]`` with ``a`` and ``b`` of the same parity, so the nodes must be mapped onto the interval
  first; an exact fraction.
* hard: ``∫ₐᵇ 1/(x + c) dx`` or ``∫ₐᵇ 1/(x² + c) dx``, whose function values at the irrational nodes are surds but
  whose two-point sum is rational; an exact fraction.
* exam: a quartic on ``[a, b]`` in four parts: the two-point value (fraction), the exact integral (fraction), the
  absolute error between them (follow-through) and the relative error as a percentage to 2 decimal places
  (follow-through).

The rule is checked a second way with the nodes as floats, and the exact integral against SymPy's ``integrate``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r4b_kit import (
    X,
    Redraw,
    dec,
    fraction_distractors,
    fraction_part,
    fr,
    margin_ok,
    number_distractors,
    number_part,
    poly_eval,
    poly_prompt,
    poly_sympy,
    with_redraw,
)
from ._ucv1_texsay import say_marked

M_NO_MAP = "num2-gauss-nodes-not-mapped-to-the-interval"
M_NO_SCALE = "num2-gauss-interval-scale-factor-dropped"
M_THIRD = "num2-gauss-nodes-at-plus-minus-one-third"
M_MID = "num2-gauss-one-point-rule-used"
M_EXACT = "num2-gauss-exact-integral-given"
M_HALF = "num2-gauss-weights-halved"
M_TRAP = "num2-gauss-endpoints-used"
M_THREE = "num2-gauss-three-point-nodes-used"
M_SIGNED = "num2-gauss-error-sign"
M_REL_DENOM = "num2-gauss-relative-error-against-the-rule"
M_FRACTION_NOT_PERCENT = "num2-gauss-relative-error-not-as-percentage"
M_ERR_OF_RULE = "num2-gauss-rule-value-for-error"


def compose_shift(coeffs, m):
    """Coefficients of g(u) = f(m + u)."""
    out = [Fraction(0)]
    power = [Fraction(1)]
    base = [fr(m), Fraction(1)]
    for c in coeffs:
        out = [
            (out[i] if i < len(out) else 0) + fr(c) * (power[i] if i < len(power) else 0)
            for i in range(max(len(out), len(power)))
        ]
        nxt = [Fraction(0)] * (len(power) + 1)
        for i, v in enumerate(power):
            nxt[i] += v * base[0]
            nxt[i + 1] += v * base[1]
        power = nxt
    return out


def pair_sum(coeffs, m, s_squared):
    """f(m + s) + f(m - s) for s with s² = s_squared: twice the even part of g(u) = f(m + u)."""
    g = compose_shift(coeffs, m)
    return 2 * sum(g[k] * s_squared ** (k // 2) for k in range(0, len(g), 2))


def rule(coeffs, a, b, s2_unit=Fraction(1, 3)):
    h = (fr(b) - fr(a)) / 2
    m = (fr(a) + fr(b)) / 2
    return h * pair_sum(coeffs, m, h * h * s2_unit)


def exact_integral(coeffs, a, b):
    anti = lambda x: sum(fr(c) * fr(x) ** (k + 1) / (k + 1) for k, c in enumerate(coeffs))  # noqa: E731
    value = anti(b) - anti(a)
    ref = sympy.Poly(poly_sympy(coeffs), X).integrate().eval(b) - sympy.Poly(poly_sympy(coeffs), X).integrate().eval(a)
    check(ref == sympy.Rational(value.numerator, value.denominator), "SymPy's integral disagrees")
    return value


def rule_by_floats(f, a, b) -> float:
    """The rule from its nodes as floats: m ± h/√3 with weights h. A route with no even-part shortcut."""
    h = (b - a) / 2
    m = (a + b) / 2
    s = h / math.sqrt(3)
    return h * (f(m - s) + f(m + s))


def poly_float(coeffs):
    cs = [float(c) for c in coeffs]
    return lambda x: sum(c * x**k for k, c in enumerate(cs))


def _quartic(rng, lo=-3, hi=3):
    for _ in attempts(500, "a quartic"):
        c = [rng.randint(lo, hi) for _ in range(4)] + [rng.choice([-2, -1, 1, 2, 3])]
        if c[2] == 0 or c[0] == 0:
            continue
        return [Fraction(v) for v in c]
    raise AssertionError("unreachable")


def _interval(rng, tier):
    if tier == "easy":
        return -1, 1
    for _ in attempts(200, "an interval"):
        a = rng.randint(-2, 3)
        b = a + 2 * rng.randint(1, 3)
        if (a, b) != (-1, 1) and a != b:
            return a, b
    raise AssertionError("unreachable")


def _poly_slips(coeffs, a, b):
    h = (fr(b) - fr(a)) / 2
    m = (fr(a) + fr(b)) / 2
    true = rule(coeffs, a, b)
    out = [
        (M_NO_MAP, h * pair_sum(coeffs, Fraction(0), Fraction(1, 3))),
        (M_NO_SCALE, true / h),
        (M_THIRD, h * pair_sum(coeffs, m, h * h * Fraction(1, 9))),
        (M_MID, 2 * h * poly_eval(coeffs, m)),
        (M_EXACT, exact_integral(coeffs, a, b)),
        (M_HALF, true / 2),
        (M_THREE, h * pair_sum(coeffs, m, h * h * Fraction(3, 5))),
        (M_TRAP, h * (poly_eval(coeffs, a) + poly_eval(coeffs, b))),
    ]
    return out


@generator(
    "r4b_gauss_legendre",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4b_gauss_legendre(rng, tier="easy"):
    return with_redraw(lambda: _single(rng, tier), "a Gauss-Legendre question with enough distractors")


def _single(rng, tier):
    if tier in ("easy", "medium"):
        coeffs = _quartic(rng)
        a, b = _interval(rng, tier)
        value = rule(coeffs, a, b)
        check(
            abs(float(value) - rule_by_floats(poly_float(coeffs), a, b)) < 1e-9 * max(1, abs(float(value))),
            "the float route disagrees",
        )
        if value == exact_integral(coeffs, a, b) or value.denominator == 1:
            raise Redraw
        cands = _poly_slips(coeffs, a, b)
        text = say_marked(
            f"Approximate «∫ f(x) dx» from «x = {_n(a)}» to «x = {_n(b)}», where «f(x) = {poly_prompt(coeffs)}», "
            "using the two-point Gauss-Legendre rule. Give the exact value as a fraction."
        )
        dist = fraction_distractors(value, cands)
        if len(dist) < 3:
            raise Redraw
        part = fraction_part(
            "",
            value,
            cands,
            1,
            working=[
                f"The nodes are x = {(a + b) // 2 if (a + b) % 2 == 0 else (a + b) / 2} ± {(b - a) // 2}/√3 and the weights are both {(b - a) // 2}.",
                f"The rule gives {value}.",
            ],
        )
        return dict(
            text=text[0],
            latex=text[1],
            answer=part["answer"],
            answer_text=part["answer_text"],
            answer_kind="fraction",
            distractors=dist,
            working=part["working"],
        )
    # hard: a rational integrand, so the irrational nodes still give a rational rule value
    for _ in attempts(500, "a rational integrand"):
        kind = rng.choice(["recip", "lorentz"])
        a = rng.randint(0, 3)
        b = a + 2 * rng.randint(1, 2)
        c = rng.randint(1, 4)
        f, pair, text_f = _rational_integrand(kind, c)
        if kind == "recip" and a + c < 1:
            continue
        h = Fraction(b - a, 2)
        m = Fraction(a + b, 2)
        value = h * pair(m, h * h / 3)
        if value.denominator == 1 or value.denominator > 5000:
            continue
        break
    check(
        abs(float(value) - rule_by_floats(lambda x: float(f(Fraction(x).limit_denominator(10**12))), a, b)) < 1e-8,
        "the float route disagrees",
    )
    cands = [
        (M_NO_MAP, h * pair(Fraction(0), Fraction(1, 3))),
        (M_NO_SCALE, pair(m, h * h / 3)),
        (M_THIRD, h * pair(m, h * h / 9)),
        (M_MID, 2 * h * f(m)),
        (M_TRAP, h * (f(Fraction(a)) + f(Fraction(b)))),
        (M_THREE, h * pair(m, h * h * Fraction(3, 5))),
        (M_HALF, value / 2),
    ]
    text = say_marked(
        f"Approximate «∫ {text_f} dx» from «x = {a}» to «x = {b}» using the two-point Gauss-Legendre rule. "
        "Give the exact value as a fraction."
    )
    dist = fraction_distractors(value, cands)
    if len(dist) < 3:
        raise Redraw
    part = fraction_part(
        "",
        value,
        cands,
        1,
        working=[
            f"With m = {m} and h = {h}, the nodes are m ± h/√3 and the weights are both h.",
            f"The sum of the integrand at the two nodes simplifies to a fraction, giving {value}.",
        ],
    )
    return dict(
        text=text[0],
        latex=text[1],
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind="fraction",
        distractors=dist,
        working=part["working"],
    )


def _rational_integrand(kind, c):
    """(f, pair, text): f on Fractions, pair(m, s²) = f(m + s) + f(m − s) for s² given, and the integrand's text."""
    if kind == "recip":
        return (
            lambda x: 1 / (x + c),
            lambda m, s2: 2 * (m + c) / ((m + c) ** 2 - s2),
            f"1/(x + {c})",
        )
    return (
        lambda x: 1 / (x * x + c),
        lambda m, s2: 2 * (m * m + s2 + c) / ((m * m + s2 + c) ** 2 - 4 * m * m * s2),
        f"1/(x² + {c})",
    )


def _n(v) -> str:
    return str(v).replace("-", "−")


def _p(v) -> str:
    return f"({_n(v)})" if v < 0 else str(v)


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4b_gauss_legendre", level=22, presets={"exam": dict(tier="exam")})
def exd_x_r4b_gauss_legendre(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam Gauss-Legendre question")


def _exam(rng):
    for _ in attempts(1000, "an exam Gauss-Legendre draw"):
        coeffs = _quartic(rng)
        a, b = _interval(rng, "medium")
        value = rule(coeffs, a, b)
        exact = exact_integral(coeffs, a, b)
        if exact == 0 or value == exact or value.denominator == 1:
            continue
        err = abs(exact - value)
        rel = 100 * err / abs(exact)
        if margin_ok(rel, 2, 0.12) and 0.01 < rel < 1000:
            break
    check(
        abs(float(value) - rule_by_floats(poly_float(coeffs), a, b)) < 1e-9 * max(1, abs(float(value))),
        "the float route disagrees",
    )
    stem, latex = say_marked(
        f"The integral «∫ f(x) dx» from «x = {_n(a)}» to «x = {_n(b)}», where «f(x) = {poly_prompt(coeffs)}», is "
        "approximated by the two-point Gauss-Legendre rule."
    )
    cands = _poly_slips(coeffs, a, b)
    p_a = fraction_part(
        "Find the value of the two-point rule as an exact fraction.",
        value,
        cands,
        2,
        working=[f"The rule gives {_n(value)}."],
    )
    p_b = fraction_part(
        "Find the exact value of the integral as a fraction.",
        exact,
        [
            (M_NO_SCALE, exact / ((fr(b) - fr(a)) / 2)),
            (M_MID, 2 * ((fr(b) - fr(a)) / 2) * poly_eval(coeffs, (fr(a) + fr(b)) / 2)),
            (M_TRAP, ((fr(b) - fr(a)) / 2) * (poly_eval(coeffs, a) + poly_eval(coeffs, b))),
            (M_ERR_OF_RULE, value),
        ],
        2,
        working=[f"The antiderivative evaluated between the limits gives {_n(exact)}."],
    )
    p_c = fraction_part(
        "Find the absolute error of the rule, the size of the difference between your two values, as an exact fraction.",
        err,
        [
            (M_SIGNED, value - exact if value < exact else exact - value),
            (M_ERR_OF_RULE, value),
            ("num2-gauss-error-as-sum", exact + value),
        ],
        2,
        ft="Abs(part_b - part_a)",
        working=[f"|{_p(exact)} − {_p(value)}| = {err}."],
        value_range=(0, None),
    )
    rel_floor = Fraction(rel).limit_denominator(10**9)
    p_d = number_part(
        "Find the relative error of the rule as a percentage of the exact value, to 2 decimal places.",
        rel,
        2,
        [
            (M_REL_DENOM, 100 * err / abs(value)),
            (M_FRACTION_NOT_PERCENT, err / abs(exact)),
            ("num2-gauss-relative-error-against-one", 100 * err),
        ],
        2,
        ft="100*part_c/Abs(part_b)",
        working=[f"100 × {err}/|{_n(exact)}| ≈ {dec(rel, 2)}%."],
        value_range=(0, None),
    )
    for part in (p_a, p_b, p_c, p_d):
        part["prompt"], part["prompt_latex"] = say_marked(part["prompt"])
    return dict(text=stem, latex=latex, parts=[p_a, p_b, p_c, p_d])
