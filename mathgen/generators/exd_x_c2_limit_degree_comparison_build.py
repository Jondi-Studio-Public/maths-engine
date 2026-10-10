"""Exam-tier source for ``limit-degree-comparison``: factored products, and limits towards negative infinity.

At the lower tiers the numerator and denominator are single terms ``a x^m`` and the degrees can be read
off.  Here each is a product of powers of linear and quadratic factors, such as ``(2x - 3)^2 (x + 1)`` over
``(x^2 + 5)^2``, so the learner has to total the degrees (3 against 4 here) without expanding.  Half the
questions also approach ``-infinity``, where an odd difference in degree reverses the sign of an infinite limit.

The degrees always differ, so the answer is one of ``0``, ``∞`` or ``-∞``:

- numerator degree below denominator degree: ``0``;
- otherwise the sign is the sign of the ratio of leading coefficients, times ``(-1)^(dN - dD)`` when
  ``x -> -infinity``.

Built backwards from the factor lists.  The independent check evaluates the exact rational function at
``x = +/- 10^9`` with ``Fraction`` and tests the size and sign of the value.

Distractors:

- reading the sign as if ``x`` were positive (when approaching ``-infinity``);
- the opposite infinity;
- the ratio of leading coefficients (applying the equal-degree rule);
- answering 0 for a limit that is infinite, or an infinity for a limit that is 0.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, join_terms, phrase, pick_distractors, rational_text

_PATTERNS = (
    "Find the limit as x → {target} of {expr}.",
    "Evaluate lim x→{target} of {expr}.",
    "Determine the limit of {expr} as x → {target}.",
)

_BIG = 10**9


def _factor(rng: random.Random):
    """One factor ``(text, evaluate(x), degree, leading_coefficient)`` raised to a power."""
    kind = rng.choice(["linear", "linear", "quadratic"])
    power = rng.choice([1, 1, 2, 3])
    if kind == "linear":
        c = rng.choice([v for v in range(-4, 5) if v != 0])
        d = rng.choice([v for v in range(-6, 7) if v != 0])
        base = join_terms([(c, "x"), (d, "")])
        return _power_text(base, power), (lambda x, c=c, d=d, n=power: (c * x + d) ** n), power, c**power
    c = rng.randint(1, 4)
    d = rng.choice([v for v in range(-6, 7) if v != 0])
    base = join_terms([(c, "x^2"), (d, "")])
    return _power_text(base, power), (lambda x, c=c, d=d, n=power: (c * x * x + d) ** n), 2 * power, c**power


def _power_text(base: str, power: int) -> str:
    return f"({base})" if power == 1 else f"({base})^{power}"


def _product(rng: random.Random, count: int):
    factors = [_factor(rng) for _ in range(count)]
    text = "".join(t for t, _, _, _ in factors)
    degree = sum(d for _, _, d, _ in factors)
    leading = 1
    for _, _, _, lc in factors:
        leading *= lc

    def evaluate(x, fs=factors):
        value = Fraction(1)
        for _, fn, _, _ in fs:
            value *= fn(x)
        return value

    if count > 1:
        text = f"({text})"
    return text, evaluate, degree, leading


def _exam(rng: random.Random) -> dict:
    infinite = rng.random() < 0.6
    while True:
        num_text, num_eval, d_num, lc_num = _product(rng, rng.choice([1, 2]))
        den_text, den_eval, d_den, lc_den = _product(rng, rng.choice([1, 2]))
        if d_num != d_den and abs(d_num - d_den) <= 3 and d_num <= 7 and d_den <= 7 and (d_num > d_den) == infinite:
            break
    toward_minus = rng.random() < 0.5
    sign_ratio = 1 if lc_num * lc_den > 0 else -1
    if d_num < d_den:
        key = "0"
    else:
        sign = sign_ratio * ((-1) ** (d_num - d_den) if toward_minus else 1)
        key = "∞" if sign > 0 else "-∞"

    # Independent route: the exact rational function far out on the chosen side.
    x_far = -_BIG if toward_minus else _BIG
    value = num_eval(x_far) / den_eval(x_far)
    if key == "0":
        check(abs(value) < Fraction(1, 1000), "expected the value to be near zero far out")
    else:
        check(abs(value) > 1000 and (value > 0) == (key == "∞"), "expected a large value of the stated sign far out")

    ratio = rational_text(Fraction(lc_num, lc_den))
    if key == "0":
        wrong = [ratio, "∞", "-∞"]
    else:
        ignoring_direction = "∞" if sign_ratio > 0 else "-∞"
        opposite = "-∞" if key == "∞" else "∞"
        wrong = [ignoring_direction, opposite, ratio, "0"]
    expr = f"{num_text} / {den_text}"
    text = phrase(rng, _PATTERNS, expr=expr, target="-∞" if toward_minus else "∞")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=key,
        answer_latex=key,
        distractors=pick_distractors(key, wrong),
    )


@generator("exd_x_c2_limit_degree_comparison", level=16, presets={"exam": dict(tier="exam")})
def exd_x_c2_limit_degree_comparison(rng: random.Random, tier: str = "exam") -> dict:
    """Degree comparison for factored products, towards +/- infinity."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
