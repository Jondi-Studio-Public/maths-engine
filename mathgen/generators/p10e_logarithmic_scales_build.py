"""Secondary-number cluster: one new
generator for one genuinely new graded operation in the ``secondary-number``
domain, framework ``acara-f10-v9``.

``p10e_logarithmic_scales`` (AC9M10M02, prerequisites ``logarithm-laws``,
``scientific-notation``) -- given a logarithmic scale with two labelled
reference values and a stated position on the scale expressed as a fraction
of the scale-distance between them, calculate the numeric value at that
position, using the fact that equal scale-distances correspond to equal
multiplicative ratios. Scope: "Given a
logarithmic scale with two labelled reference values and a stated position
on the scale expressed as a fraction of the scale-distance between them,
calculate the numeric value at that position, using the fact that equal
scale-distances correspond to equal multiplicative ratios."

**Design so every answer is exact (never irrational)**: reference values
``A`` (a small integer) and ``B = A * 10**k`` for an integer ``k`` (the
number of decades between them). The position ``t`` is a stated fraction
``j/k`` for an integer ``j`` -- ``value(t) = A * 10**(k*t) = A * 10**j`` is
then always an exact integer power-of-ten multiple of ``A``. ``A`` is always
drawn as a MULTIPLE of ``k`` (not merely a small integer 1-20): this is
stricter than the descriptor requires, and is what keeps the
``linear-interpolation`` misconception's own value
(``A + t*(B-A) = A + j*(A/k)*(10**k - 1)``) an exact integer at every seed
too, so every option in a forward-direction question shares the same
integer shape -- an unconstrained ``A`` produces a decimal linear-
interpolation distractor next to three integer options at roughly half of
all draws, exactly the "only decimal in the room" shape defect the
``midpoint_coordinate_scalar`` precedent warns against.

**Independent check** (``_verify_log_position``): a genuinely different
route from however construction computed ``value`` (integer exponent
arithmetic) -- recomputes via actual base-10 logarithms (``math.log10``),
asserting the log-ratio equals ``t`` to within ``1e-9``. Used both ways: the
forward direction checks the constructed ``value`` against the stated ``t``;
the inverse direction checks the stated ``value`` against the constructed
``t`` -- same formula, same function, since the underlying relationship is
symmetric.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, generator, numeric_answer, pick_distractors, stated_precision_tolerance

_DECADE_POOL = (2, 3, 4)
_MAX_A = 20


def _draw_A(rng, k: int) -> int:
    """A small integer 1-20 that is a MULTIPLE of k -- see module docstring
    for why this is required, not merely convenient."""
    pool = list(range(k, _MAX_A + 1, k))
    return rng.choice(pool)


def _verify_log_position(A: int, B: int, value, t: Fraction) -> None:
    """Independent check, genuinely different route: actual base-10
    logarithms, never the integer-exponent arithmetic construction uses."""
    log_ratio = (math.log10(float(value)) - math.log10(A)) / (math.log10(B) - math.log10(A))
    check(
        abs(log_ratio - float(t)) < 1e-9,
        f"log-ratio check failed: computed t={log_ratio}, expected {float(t)}",
    )


def _linear_interp(A: int, B: int, t: Fraction) -> Fraction:
    return A + t * (B - A)


def _off_by_one_values(A: int, j: int) -> tuple[int, int]:
    return A * 10 ** (j + 1), A * 10 ** max(0, j - 1)


def _build_forward(rng, k: int, j: int, *, swap_applicable: bool, easy_ratio_halved: bool, extrapolation_note: bool):
    A = _draw_A(rng, k)
    B = A * 10**k
    t = Fraction(j, k)
    value = A * 10**j
    _verify_log_position(A, B, value, t)

    linear_frac = _linear_interp(A, B, t)
    check(linear_frac.denominator == 1, f"linear-interpolation distractor {linear_frac} is not an exact integer")
    linear = linear_frac.numerator
    over, under = _off_by_one_values(A, j)

    candidates = [linear, over, under]
    if swap_applicable:
        swapped_j = k - j
        candidates.append(A * 10**swapped_j)
    elif easy_ratio_halved:
        # Fixed easy-tier t=1/2 substitute for swap-reference-direction (see
        # module docstring): halves the whole multiplicative RATIO A->B
        # arithmetically (B // 2, scaled by A) instead of taking the correct
        # geometric half (the square root of the ratio, i.e. halving the
        # EXPONENT). Always an exact integer since 10**k is even for k >= 1.
        candidates.append((A * 10**k) // 2)

    check_distinct(value, *candidates, msg="a logarithmic-scale distractor equals the correct value")

    frac_text = f"{j}/{k}"
    extra = " (continuing past B along the same equal-ratio pattern)" if extrapolation_note else ""
    text = (
        f"On a logarithmic scale, A is marked at {A} and B is marked at {B}, {k} equal "
        f"scale-steps apart (each step means multiplying by 10). A point lies {frac_text} "
        f"of the way from A towards B{extra}. What exact value sits at that point?"
    )
    answer_text = str(value)
    working = [
        f"Equal scale-distances mean equal ratios, so the value is A x 10^({frac_text} x {k}) = {A} x 10^{j} = {value}."
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=numeric_answer(Fraction(value)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [str(c) for c in candidates]),
    )


def _fmt2(value: Fraction) -> str:
    return f"{float(value):.2f}"


def _build_inverse(rng):
    k = 4
    j = rng.choice([1, 3])
    A = _draw_A(rng, k)
    B = A * 10**k
    t = Fraction(j, k)
    value = A * 10**j
    _verify_log_position(A, B, value, t)

    linear_t = Fraction(value - A, B - A)
    over_t, under_t = Fraction(j + 1, k), Fraction(max(0, j - 1), k)
    swap_t = 1 - t

    candidates = [linear_t, over_t, under_t, swap_t]
    check_distinct(t, *candidates, msg="a logarithmic-scale (inverse) distractor equals the correct position")
    for candidate in candidates:
        check(abs(candidate - t) > stated_precision_tolerance(2), f"distractor {candidate} too close to {t}")

    text = (
        f"On a logarithmic scale, A is marked at {A} and B is marked at {B}, {k} equal "
        f"scale-steps apart (each step means multiplying by 10). The value {value} sits at "
        f"some position between A and B. Expressed as a fraction of the scale-distance "
        f"from A to B, what position is that? Give your answer as a decimal to 2 decimal "
        f"places."
    )
    answer_text = _fmt2(t)
    working = [
        f"Equal scale-distances mean equal ratios: {value} = {A} x 10^{j}, and B is {k} "
        f"steps above A, so the position is {j}/{k} = {answer_text} of the way from A to B."
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=numeric_answer(t),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [_fmt2(c) for c in candidates]),
    )


def _easy(rng):
    return _build_forward(rng, k=2, j=1, swap_applicable=False, easy_ratio_halved=True, extrapolation_note=False)


def _medium(rng):
    k = rng.choice([3, 4])
    pool = [j for j in range(1, k) if 2 * j != k]
    j = rng.choice(pool)
    return _build_forward(rng, k, j, swap_applicable=True, easy_ratio_halved=False, extrapolation_note=False)


def _hard(rng):
    if rng.random() < 0.5:
        k = rng.choice(_DECADE_POOL)
        j = k + 1
        return _build_forward(rng, k, j, swap_applicable=False, easy_ratio_halved=False, extrapolation_note=True)
    return _build_inverse(rng)


_VARIANT_FNS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "p10e_logarithmic_scales",
    level=9,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def p10e_logarithmic_scales(rng, variant="medium"):
    """Interpret a logarithmic scale: read/interpolate a value at a stated
    fractional position between two labelled reference values, or (hard
    tier's inverse sub-variant) recover the fractional position from a
    stated value. See the module docstring for the exact-power-of-ten
    design and the misconception derivations."""
    if variant not in _VARIANT_FNS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_VARIANT_FNS)}")
    return _VARIANT_FNS[variant](rng)
