"""Exam tier for estimating lim (a^h - 1)/h from a table: a three-part chain on y = a^x.

The practice tiers read one limit off a table. Here the estimate is the first step of a derivative question:

(a) estimate L = lim (a^h - 1)/h from a table of values at h = 0.1, 0.01, 0.001, 0.0001 (2 decimal places);
(b) use L to estimate the gradient of y = a^x at x = c, which is L * a^c;
(c) use L again for where that tangent meets the x-axis, x = c - 1/L.

Parts (b) and (c) follow through from the learner's own estimate in (a). The tangent's x-intercept is sensitive
to the rounding of L, so (b) and (c) are stated "using your estimate from (a)" and keyed from the rounded L, and
the draw is kept clear of every rounding boundary so a learner's calculator route agrees.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, generator, round_half_up
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3h_kit import number_part

_H_ROWS = (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000), Fraction(1, 10000))
_A_TENTHS = [t for t in range(3, 100) if t != 10]
_STEMS = (
    "Let f(x) = {a}^x. Using technology, a student evaluates (a^h - 1)/h for a = {a} at decreasing values of h and "
    "records: {rows}.",
    "A student investigates the exponential function f(x) = {a}^x. For a = {a}, a calculator gives (a^h - 1)/h as: "
    "{rows}.",
    "Consider y = {a}^x. A spreadsheet is used to tabulate (a^h - 1)/h for a = {a} as h gets smaller: {rows}.",
)
_GRADIENT_PROMPTS = (
    "Using your estimate from (a), estimate f'({c}), the gradient of y = {a}^x at x = {c}, correct to 2 decimal places.",
    "Use your answer to (a) to estimate the gradient of the curve y = {a}^x at the point where x = {c}, correct to "
    "2 decimal places.",
    "Estimate the slope of y = {a}^x at x = {c} from your answer to (a), correct to 2 decimal places.",
)
_INTERCEPT_PROMPTS = (
    "Using your estimate from (a), find the x-intercept of the tangent to y = {a}^x at x = {c}, correct to 1 decimal "
    "place.",
    "At x = {c}, the tangent to y = {a}^x meets the x-axis at x = t. Use your answer to (a) to find t, correct to "
    "1 decimal place.",
    "Using your answer to (a), find the point where the x-axis is crossed by the tangent to the curve y = {a}^x "
    "drawn at x = {c}. Give its x-coordinate correct to 1 decimal place.",
)
_MARGIN = Fraction(1, 50)  # keep a displayed value this far (in units of its last place) from a rounding boundary


def _a_text(tenths: int) -> str:
    return str(tenths // 10) if tenths % 10 == 0 else f"{tenths // 10}.{tenths % 10}"


def _clear(value: Fraction, places: int) -> bool:
    offset = (value * 10**places) % 1
    return abs(offset - Fraction(1, 2)) > _MARGIN


def _row(a: float, h: Fraction) -> str:
    hf = float(h)
    return f"h = {h.numerator / h.denominator:g}: {(a**hf - 1) / hf:.4f}"


@generator("exd_x_r3h_limit_ah", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3h_limit_ah(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a base and point clear of every rounding boundary"):
        tenths = rng.choice(_A_TENTHS)
        c = rng.choice((1, 2, 3))
        a = Fraction(tenths, 10)
        ln_a = Fraction(math.log(float(a)))
        if not _clear(ln_a, 2):
            continue
        limit = round_half_up(ln_a, 2)  # the learner's estimate from the table
        gradient = limit * a**c
        intercept = c - 1 / limit
        if limit == 0 or abs(intercept) < Fraction(1, 2) or not (_clear(gradient, 2) and _clear(intercept, 1)):
            continue
        break
    af = float(a)
    a_text = _a_text(tenths)
    rows = [_row(af, h) for h in _H_ROWS]
    check_last = round_half_up(Fraction((af ** float(_H_ROWS[-1]) - 1) / float(_H_ROWS[-1])), 2)
    if check_last != limit:
        raise ValueError("the last table row does not round to the limit")
    first_row = round_half_up(Fraction((af**0.1 - 1) / 0.1), 2)
    a_frac = f"({a.numerator}/{a.denominator})"
    limit_text = f"{float(limit):.2f}"  # the estimate as stated to its 2 d.p., never a fraction
    limit_bracketed = f"({limit_text})" if limit < 0 else limit_text

    stem = rng.choice(_STEMS).format(a=a_text, rows=", ".join(rows))
    gradient_prompt = rng.choice(_GRADIENT_PROMPTS).format(a=a_text, c=c)
    intercept_prompt = rng.choice(_INTERCEPT_PROMPTS).format(a=a_text, c=c)
    return dict(
        text=stem,
        parts=[
            dict(
                **number_part(
                    "Estimate the limit of (a^h - 1)/h as h → 0, correct to 2 decimal places.",
                    limit,
                    2,
                    2,
                    ["The values settle towards one number as h decreases.", f"The limit is about {limit_text}."],
                    [
                        (first_row, "reads-coarsest-table-row-as-limit"),
                        (a - 1, "difference-quotient-taken-at-h-equals-1"),
                        (-limit, "sign-of-result-reversed"),
                        (Fraction(math.log10(af)), "common-log-for-natural-log"),
                    ],
                )
            ),
            dict(
                **number_part(
                    gradient_prompt,
                    gradient,
                    2,
                    2,
                    [
                        f"f'(x) = L × {a_text}^x, where L is the limit from (a), so f'({c}) = {limit_bracketed} × {a_text if c == 1 else f'{a_text}^{c}'}."
                    ],
                    [
                        (a**c, "exponential-derivative-missing-ln-factor"),
                        (limit * c * a ** (c - 1), "power-rule-applied-to-exponential"),
                        (limit, "gradient-taken-at-x-equals-zero"),
                        (-gradient, "sign-of-result-reversed"),
                    ],
                    follow_through=f"part_a*{a_frac}**{c}",
                )
            ),
            dict(
                **number_part(
                    intercept_prompt,
                    intercept,
                    1,
                    3,
                    [
                        f"The tangent at x = {c} has gradient L × {a_text if c == 1 else f'{a_text}^{c}'} and passes through ({c}, {a_text if c == 1 else f'{a_text}^{c}'}).",
                        f"Setting y = 0 gives x = {c} - 1/L = {c} - 1/{limit_bracketed}.",
                    ],
                    [
                        (c + 1 / limit, "sign-slip-in-formula"),
                        (c - limit, "tangent-intercept-ratio-inverted"),
                        (-1 / limit, "tangent-intercept-ignores-base-point"),
                        (c - 1 / gradient, "tangent-intercept-uses-gradient-for-subtangent"),
                    ],
                    follow_through=f"{c} - 1/part_a",
                )
            ),
        ],
    )
