"""Exam source of `limit-degree-comparison`: the growth rate of a factored quotient, then the sign effect at -infinity.

The practice tiers ask which of 0, infinity or minus infinity a quotient of powers of x tends to. An exam question
sets a quotient of factored products and asks for the growth law behind that answer:

(a) the integer n for which f(x) / x^n has a finite, non-zero limit as x -> infinity (the difference of the degrees,
    found by totalling the degrees of the factors without expanding);
(b) that limit L (the ratio of the leading coefficients, each raised to the power of its factor);
(c) the limit of f(x) / |x|^n as x -> -infinity, which is L times (-1)^n: the reason an odd difference of degrees
    reverses the sign of an infinite limit at minus infinity, and the reason a limit of 0 keeps its sign information.

Parts (b) and (c) follow through from (a) and (b). L is a whole number by construction. An independent check evaluates
the exact quotient with Fractions at x = +/-10^9 and compares f(x) / x^n and f(x) / |x|^n with L and with the key.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms, latex_answer, numeric_answer, phrase

from .exd_x_r3d_common import tagged
from ._exd_r3_distinct import distinct_part_answers

_BIG = 10**9

_STEMS = (
    "Let f(x) = {num} / {den}.",
    "The function f is defined by f(x) = {num} / {den}.",
    "Consider the function f(x) = {num} / {den}.",
    "A rational function is given by f(x) = {num} / {den}.",
)
_PROMPT_A = (
    "Find the integer n for which f(x) / x^n approaches a finite, non-zero limit as x → ∞.",
    "Determine the integer n such that f(x) / x^n tends to a finite limit that is not zero as x → ∞.",
)
_PROMPT_B = (
    "Using your value of n, find the limit of f(x) / x^n as x → ∞.",
    "With n as found in part (a), evaluate the limit of f(x) / x^n as x → ∞.",
)
_PROMPT_C = (
    "Using your value of n, find the limit of f(x) / |x|^n as x → -∞.",
    "With n as found in part (a), evaluate the limit of f(x) / |x|^n as x → -∞.",
)


def _factor(rng: random.Random):
    """A factor raised to a power: `(text, value(x), degree, leading coefficient, bare leading coefficient)`."""
    quadratic = rng.random() < 0.3
    power = rng.choice([1, 1, 2, 3])
    c = rng.randint(1, 4) if quadratic else rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    d = rng.choice([v for v in range(-6, 7) if v != 0])
    base = join_terms([(c, "x^2" if quadratic else "x"), (d, "")])
    text = f"({base})" if power == 1 else f"({base})^{power}"
    degree = (2 if quadratic else 1) * power

    def value(x, c=c, d=d, power=power, quadratic=quadratic):
        return (c * x * x + d if quadratic else c * x + d) ** power

    return text, value, degree, c**power, c, power


def _product(rng: random.Random, count: int):
    factors = [_factor(rng) for _ in range(count)]
    text = "".join(f[0] for f in factors)
    if count > 1:
        text = f"({text})"
    degree = sum(f[2] for f in factors)
    lead, bare, powers = 1, 1, 0
    for f in factors:
        lead *= f[3]
        bare *= f[4]
        powers += f[5]

    def value(x, fs=factors):
        out = Fraction(1)
        for f in fs:
            out *= f[1](x)
        return out

    return text, value, degree, lead, bare, powers


@generator("exd_x_r3d_limit_growth", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_limit_growth(rng: random.Random, tier: str = "exam") -> dict:
    """The growth exponent and constant of a factored quotient, and its behaviour as x -> -infinity."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a factored quotient with a whole-number limit"):
        num = _product(rng, rng.choice([1, 2]))
        den = _product(rng, rng.choice([1, 2]))
        n = num[2] - den[2]
        # An even n would make (c) the same number as (b): only odd differences of degree keep every part distinct.
        if n % 2 == 0 or abs(n) > 5 or num[2] > 8 or den[2] > 8:
            continue
        if num[3] % den[3] != 0 or abs(num[3]) > 150:
            continue
        limit = num[3] // den[3]
        key_c = limit * (-1 if n % 2 else 1)
        # Independent route: the exact quotient far out on each side.
        f_up = num[1](_BIG) / den[1](_BIG)
        f_down = num[1](-_BIG) / den[1](-_BIG)
        scale = Fraction(_BIG) ** abs(n)
        up = f_up / scale if n > 0 else f_up * scale
        down = f_down / scale if n > 0 else f_down * scale
        if not (abs(up - limit) * 1000 < abs(limit) and abs(down - key_c) * 1000 < abs(limit)):
            continue
        text_n, text_d = num[0], den[0]

        a_cands = [
            (-n, "inverted-degree-comparison"),
            (num[2] + den[2], "degrees-added-for-quotient"),
            (num[5] - den[5], "factor-degree-undercounted") if num[5] - den[5] not in (0, n) else None,
            (num[2], "denominator-degree-ignored"),
        ]
        base_ratio = Fraction(num[4], den[4])
        b_cands = [
            (
                base_ratio if base_ratio.denominator == 1 and base_ratio != limit else None,
                "leading-coefficient-power-ignored",
            ),
            (-limit, "limit-constant-sign-flipped"),
            (num[3], "denominator-coefficient-ignored"),
            (num[3] * den[3], "coefficients-multiplied-not-divided"),
        ]
        c_cands = [
            (-key_c, "ignored-side" if n % 2 else "limit-constant-sign-flipped"),
            (0, "denominator-grows-so-zero") if n < 0 else None,
        ]
        a_d = tagged(str(n), [c for c in a_cands if c])
        b_d = tagged(str(limit), [c for c in b_cands if c])
        c_d = tagged(str(key_c), [c for c in c_cands if c])
        if not (len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 1):
            continue
        break

    stem = phrase(rng, _STEMS, num=text_n, den=text_d)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=rng.choice(_PROMPT_A),
                answer_kind="number",
                answer=n,
                answer_text=str(n),
                answer_latex=latex_answer(n),
                marks=2,
                working=[
                    f"The numerator has degree {num[2]} and the denominator has degree {den[2]}, "
                    f"so f(x) behaves like a constant times x^{n}."
                ],
                distractors=a_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_B),
                answer_kind="number",
                answer=numeric_answer(Fraction(limit)),
                answer_text=str(limit),
                answer_latex=latex_answer(limit),
                marks=2,
                working=[f"The ratio of the leading coefficients is {num[3]}/{den[3]} = {limit}."],
                distractors=b_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_C),
                answer_kind="number",
                answer=key_c,
                answer_text=str(key_c),
                answer_latex=latex_answer(key_c),
                marks=1,
                follow_through="part_b*cos(pi*part_a)",
                working=[
                    f"For x < 0, f(x) ≈ {limit}·x^{n} and |x|^n = (-x)^n, so the limit is {limit}·(-1)^{n} = {key_c}."
                ],
                distractors=c_d,
            ),
        ],
    )
