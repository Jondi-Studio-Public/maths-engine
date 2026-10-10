"""Generates a chained scientific-notation-to-engineering-notation conversion question as a worked example of
`mathgen.core.compose()`.

Registered under a brand-new topic, `exd_compose_scientific_to_engineering`.

The mantissa/exponent arithmetic below is written independently of the other notation generators, using only `mathgen.core` public
helpers (`exact_decimal_string`, `check`, `check_equivalent`, `pick_distractors`, `latex_answer`, `phrase`,
`compose`). A reusable `to_engineering_notation(value) -> (mantissa, exponent)` helper would serve at least
this module, `p8a_notation_build.py` and one other exam-style source; not built here.

Deliberately drops the SI-prefix name (`kilo`/`milli`/...) `p8a`'s
`engineering_notation` answer carries -- the prefix table only covers
+-{3,6,9,12}, this generator's exponent can legitimately land on 0 (a valid
multiple of 3), and a test oracle only needs to
re-derive `<mantissa> x 10^<exponent>` from the rendered text, not a name
lookup -- fewer moving parts on both sides of the independence boundary.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    compose,
    exact_decimal_string,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

# Exponents drawn for each of the two given scientific-notation values.
# Excludes 0 (a scientific-notation exponent of 0 -- "3.2 x 10^0" -- is
# technically valid but not how this shape is normally posed); |e1| and |e2|
# stay <= 6 so the combined exponent (multiply: e1+e2, divide: e1-e2) never
# exceeds the +-12 band engineering notation conventionally works within.
_EXPONENTS = tuple(e for e in range(-6, 7) if e != 0)

# The divisor coefficient for a QUOTIENT question is drawn from this small,
# curated set rather than freely from [1.0, 9.9] like the dividend -- every
# value here has only 2 and 5 as prime factors, so `dividend_coefficient /
# divisor_coefficient` is GUARANTEED to terminate in decimal for every seed
# (proved by construction below, not by sampling): dividing a tenths-
# precision Fraction by 1, 2, 4, 5 or 5/2 can only ever multiply its
# denominator by further powers of 2 and 5. A freely-drawn divisor
# (e.g. 3.0) can produce a non-terminating quotient (1/3), which
# `exact_decimal_string` would then raise on for SOME seed -- a live-seed
# failure this module cannot afford, since `mathgen.make` must succeed for
# every seed in 1..2**31, not only the ones a test sweep happens to draw.
_TERMINATING_DIVISORS = (Fraction(1, 1), Fraction(2, 1), Fraction(4, 1), Fraction(5, 1), Fraction(5, 2))

_TEMPLATES = (
    "Calculate ({c1_text} × 10^{e1}) {op} ({c2_text} × 10^{e2}), then write "
    "the result in engineering notation (coefficient between 1 and 1000, exponent a "
    "multiple of 3).",
    "Evaluate ({c1_text} × 10^{e1}) {op} ({c2_text} × 10^{e2}) and express your answer in engineering notation.",
)


def _normalise_scientific(coefficient: Fraction, exponent: int) -> tuple[Fraction, int]:
    """Renormalise `coefficient * 10**exponent` so the coefficient lands in
    [1, 10), the way an ordinary un-normalised product/quotient of two
    already-normalised scientific-notation values must be re-normalised
    before it is itself valid scientific notation."""
    while coefficient >= 10:
        coefficient /= 10
        exponent += 1
    while coefficient < 1:
        coefficient *= 10
        exponent -= 1
    return coefficient, exponent


def _part_product(rng, values):
    """Chain part 1: the scientific-notation product/quotient of the two
    given values, re-normalised into [1, 10) x 10^exponent. `values` is
    unused (empty) -- this is the first part."""
    op = rng.choice(("×", "÷"))
    e1 = rng.choice(_EXPONENTS)
    e2 = rng.choice(_EXPONENTS)
    c1 = Fraction(rng.randint(10, 99), 10)  # a tenths-precision value in [1.0, 9.9]
    if op == "×":
        c2 = Fraction(rng.randint(10, 99), 10)
        raw_coefficient = c1 * c2
        raw_exponent = e1 + e2
    else:
        c2 = rng.choice(_TERMINATING_DIVISORS)
        raw_coefficient = c1 / c2
        raw_exponent = e1 - e2
    raw_coefficient, raw_exponent = _normalise_scientific(raw_coefficient, raw_exponent)

    c1_text = exact_decimal_string(c1, 6)
    c2_text = exact_decimal_string(c2, 6)
    result_text = exact_decimal_string(raw_coefficient, 6)
    working = f"({c1_text} × 10^{e1}) {op} ({c2_text} × 10^{e2}) = {result_text} × 10^{raw_exponent}"
    # Stash the pieces `_part_engineering`/`_final` need for the prompt text
    # and the independent in-generator check, alongside compose()'s own
    # threaded `value`. `value` itself is the (coefficient, exponent) pair
    # only -- never this working string.
    return (raw_coefficient, raw_exponent, op, e1, e2, c1, c2, c1_text, c2_text), working


def _part_engineering(rng, values):
    """Chain part 2: convert chain part 1's normalised scientific-notation
    value into engineering notation (mantissa in [1, 1000), exponent a
    multiple of 3). `values[0]` is `_part_product`'s own tuple."""
    raw_coefficient, raw_exponent, *_ = values[0]
    shift = raw_exponent % 3  # always in {0, 1, 2}; Python's floor-mod
    eng_exponent = raw_exponent - shift
    eng_mantissa = raw_coefficient * (Fraction(10) ** shift)
    mantissa_text = exact_decimal_string(eng_mantissa, 6)
    if shift == 0:
        working = (
            f"{raw_exponent} is already a multiple of 3, so the engineering form "
            f"keeps the same coefficient: {mantissa_text} × 10^{eng_exponent}."
        )
    else:
        working = (
            f"The nearest lower multiple of 3 below {raw_exponent} is {eng_exponent}, "
            f"so shift the decimal point {shift} place{'s' if shift != 1 else ''}: "
            f"{mantissa_text} × 10^{eng_exponent}."
        )
    return (eng_mantissa, eng_exponent), working


def _final(values, rng):
    (raw_coefficient, raw_exponent, op, e1, e2, c1, c2, c1_text, c2_text) = values[0]
    eng_mantissa, eng_exponent = values[1]

    # Independent in-generator verification: recompute the engineering
    # mantissa a second way, straight from the two ORIGINAL given values --
    # not from the already-normalised `raw_coefficient`/`raw_exponent`
    # `_part_engineering` consumed -- so a slip in either chain part is
    # caught rather than confirmed by reusing the same intermediate.
    value1 = c1 * (Fraction(10) ** e1)
    value2 = c2 * (Fraction(10) ** e2)
    raw_value = value1 * value2 if op == "×" else value1 / value2
    recomputed = eng_mantissa * (Fraction(10) ** eng_exponent)
    check_equivalent(
        raw_value,
        recomputed,
        "exd_compose_scientific_to_engineering: engineering value disagrees with the original product/quotient",
    )
    check(
        Fraction(1) <= eng_mantissa < Fraction(1000),
        "exd_compose_scientific_to_engineering: mantissa outside [1, 1000)",
    )
    check(eng_exponent % 3 == 0, "exd_compose_scientific_to_engineering: exponent not a multiple of 3")

    mantissa_text = exact_decimal_string(eng_mantissa, 6)
    answer_text = f"{mantissa_text} × 10^{eng_exponent}"

    text = phrase(
        rng,
        _TEMPLATES,
        c1_text=c1_text,
        e1=e1,
        c2_text=c2_text,
        e2=e2,
        op=op,
    )

    # Misconceptions: (a) stopped after chain part 1 -- offered the
    # scientific-notation product/quotient unconverted; (b) shifted one
    # multiple of 3 too far in either direction.
    result_text = exact_decimal_string(raw_coefficient, 6)
    candidates = [
        f"{result_text} × 10^{raw_exponent}",
        f"{mantissa_text} × 10^{eng_exponent + 3}",
        f"{mantissa_text} × 10^{eng_exponent - 3}",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates),
    )


@generator("exd_compose_scientific_to_engineering", level=9)
def exd_compose_scientific_to_engineering(rng):
    """Chain composition: a scientific-notation product/quotient of two given values (chain part 1), converted
    into engineering notation (chain part 2), one final answer. See the module docstring for the contract
    this proves."""
    return compose(rng, [_part_product, _part_engineering], final=_final)
