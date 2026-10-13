"""Exam-tier source for `scientific-notation`: a measurement problem told in scientific notation.

Two quantities are given in scientific notation (a count and an amount per item). The three connected parts are:

* (a) multiply them and write the product as an ordinary number,
* (b) give the power of ten n when that product is written as a x 10^n with 1 <= a < 10 (when the coefficients
  multiply to 10 or more, the exponent is one larger than the sum of the exponents),
* (c) convert the product to a more convenient unit.

(b) and (c) follow through from (a). The learner has to multiply with the index laws, renormalise, and decide
which way a unit conversion goes, instead of reading one value off a single scientific-notation number.

The settings split into large totals (water, energy, ticket revenue) and tiny ones (cell widths, ink droplets,
paper thickness), so the exponent of the answer is sometimes negative.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import floor, log10

from mathgen.answers import Distractor
from mathgen.core import (
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    pick_distractors,
    stated_precision_string,
)

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sci(coefficient: Fraction, exponent: int) -> str:
    return f"{stated_precision_string(coefficient, 1)} × 10{str(exponent).translate(_SUPERSCRIPT)}"


# Each setting: stem with {A} (count or size) and {B}; the prompt for (a); the conversion sentence for (c) and the
# factor the product is multiplied by to get it; coefficient ranges in tenths and exponent ranges.
_SETTINGS = [
    dict(
        stem="A regional water authority supplies {A} households. On average, each household uses {B} litres of "
        "water per day.",
        ask_a="Calculate the total volume of water used by all of these households in one day, in litres. "
        "Write your answer as an ordinary number.",
        ask_c="One megalitre (ML) is 1 000 000 litres. Convert your answer to (a) into megalitres.",
        unit_c="megalitres",
        factor=Fraction(1, 10**6),
        a_tenths=(12, 98),
        a_exp=(4, 5),
        b_tenths=(30, 98),
        b_exp=(2, 2),
    ),
    dict(
        stem="A solar farm generates {A} kilowatt-hours (kWh) of electricity each day. It operates for {B} days.",
        ask_a="Calculate the total electricity generated, in kilowatt-hours. Write your answer as an ordinary number.",
        ask_c="One megawatt-hour (MWh) is 1000 kWh. Convert your answer to (a) into megawatt-hours.",
        unit_c="megawatt-hours",
        factor=Fraction(1, 1000),
        a_tenths=(12, 98),
        a_exp=(3, 4),
        b_tenths=(11, 36),
        b_exp=(2, 2),
    ),
    dict(
        stem="A stadium has {A} seats. Over a season, each seat earns an average of ${B}.",
        ask_a="Calculate the total revenue from the seats, in dollars. Write your answer as an ordinary number.",
        ask_c="Convert your answer to (a) into millions of dollars.",
        unit_c="millions of dollars",
        factor=Fraction(1, 10**6),
        a_tenths=(20, 98),
        a_exp=(4, 4),
        b_tenths=(12, 95),
        b_exp=(2, 2),
    ),
    dict(
        stem="A red blood cell is {A} metres wide. A scientist lines up {B} cells in a single row.",
        ask_a="Calculate the length of the row, in metres. Write your answer as an ordinary number.",
        ask_c="Convert your answer to (a) into millimetres (1 metre is 1000 millimetres).",
        unit_c="millimetres",
        factor=Fraction(1000),
        a_tenths=(60, 80),
        a_exp=(-6, -6),
        b_tenths=(20, 98),
        b_exp=(2, 3),
    ),
    dict(
        stem="An inkjet printer fires droplets that each have a volume of {A} litres. One page needs {B} droplets.",
        ask_a="Calculate the volume of ink used for one page, in litres. Write your answer as an ordinary number.",
        ask_c="Convert your answer to (a) into millilitres (1 litre is 1000 millilitres).",
        unit_c="millilitres",
        factor=Fraction(1000),
        a_tenths=(12, 98),
        a_exp=(-12, -12),
        b_tenths=(12, 98),
        b_exp=(8, 8),
    ),
    dict(
        stem="A sheet of copy paper is {A} metres thick. A stack contains {B} sheets.",
        ask_a="Calculate the height of the stack, in metres. Write your answer as an ordinary number.",
        ask_c="Convert your answer to (a) into centimetres (1 metre is 100 centimetres).",
        unit_c="centimetres",
        factor=Fraction(100),
        a_tenths=(60, 98),
        a_exp=(-5, -5),
        b_tenths=(20, 98),
        b_exp=(2, 3),
    ),
]


def _ten(power: int) -> Fraction:
    return Fraction(10) ** power


def _exponent_of(value: Fraction) -> int:
    """floor(log10(value)) computed exactly for a positive Fraction."""
    n = floor(log10(float(value)))
    while _ten(n) > value:
        n -= 1
    while _ten(n + 1) <= value:
        n += 1
    return n


def _plain(value: Fraction) -> str:
    return exact_decimal_string(value, 24)


def _numeric(value: Fraction):
    return int(value) if value.denominator == 1 else value


def _draw(rng: random.Random):
    needs_renormalising = rng.random() < 0.5  # about half the questions have coefficients that multiply to 10 or more
    for _ in range(1000):
        s = rng.choice(_SETTINGS)
        a = Fraction(rng.randint(*s["a_tenths"]), 10)
        b = Fraction(rng.randint(*s["b_tenths"]), 10)
        m = rng.randint(*s["a_exp"])
        k = rng.randint(*s["b_exp"])
        product = a * b
        if product in (10, 100) or product.denominator > 100 or (product >= 10) != needs_renormalising:
            continue
        return s, a, b, m, k
    raise AssertionError("no scientific-notation draw in 1000 tries")


@generator("exd_x_sc6_sci_notation", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc6_sci_notation(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    s, a, b, m, k = _draw(rng)
    total = a * b * _ten(m + k)
    n = _exponent_of(total)
    converted = total * s["factor"]
    renormalised = a * b >= 10
    check(n == m + k + (1 if renormalised else 0), "exponent disagrees with the coefficient product")
    check(1 <= total / _ten(n) < 10, "coefficient is not between 1 and 10")

    total_text = _plain(total)
    conv_text = _plain(converted)
    stem = s["stem"].format(A=_sci(a, m), B=_sci(b, k))

    # (a) candidates: decimal point slips, the wrong index law.
    a_cands = [
        Distractor(_plain(total * 10), "decimal-point-misplaced", total * 10),
        Distractor(_plain(total / 10), "decimal-point-misplaced", total / 10),
        Distractor(_plain((a + b) * _ten(m + k)), "adds-coefficients-when-multiplying", (a + b) * _ten(m + k)),
        Distractor(_plain(a * b * _ten(m - k)), "subtracts-exponents-of-ten", a * b * _ten(m - k)),
    ]
    if m * k != m + k and abs(m * k) <= 14:
        a_cands.append(Distractor(_plain(a * b * _ten(m * k)), "multiplies-exponents-of-ten", a * b * _ten(m * k)))

    # (b) candidates: not renormalised, over-adjusted, sign, counting digits.
    b_cands = []
    if renormalised:
        b_cands.append(Distractor(str(m + k), "exponent-not-renormalised", m + k))
    else:
        b_cands.append(Distractor(str(n + 1), "off-by-one-exponent", n + 1))
    b_cands.append(Distractor(str(-n), "exponent-sign-reversed", -n))
    b_cands.append(Distractor(str(n - 1), "off-by-one-exponent", n - 1))
    b_cands.append(Distractor(str(n + 1), "off-by-one-exponent", n + 1))

    # (c) candidates: converted the wrong way, forgot to convert, slipped a power of ten.
    c_cands = [
        Distractor(_plain(total / s["factor"]), "converted-wrong-direction", total / s["factor"]),
        Distractor(total_text, "forgot-to-convert-units", total),
        Distractor(_plain(converted * 10), "conversion-power-of-ten-slip", converted * 10),
        Distractor(_plain(converted / 10), "conversion-power-of-ten-slip", converted / 10),
    ]

    def tagged(cands, true):
        picked = pick_distractors(str(true), cands, 3)
        check(len(picked) >= 2, "too few distractors")
        check(all(Fraction(d.value) != Fraction(true) for d in picked), "distractor equals key")
        return picked

    def part(prompt, value, text, marks, **extra):
        return dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=text,
            answer_latex=latex_answer(text),
            marks=marks,
            **extra,
        )

    flipped = "multiply" if s["factor"] > 1 else "divide"
    factor_text = _plain(s["factor"]) if s["factor"] > 1 else _plain(1 / s["factor"])
    return dict(
        text=stem,
        parts=[
            part(
                s["ask_a"],
                _numeric(total),
                total_text,
                2,
                sensible_range=(0, None),
                working=[f"{_sci(a, m)} × {_sci(b, k)} = ({a} × {b}) × 10^{m + k} = {total_text}."],
                distractors=tagged(a_cands, total),
            ),
            part(
                "Your answer to (a) can be written in scientific notation as a number a, with 1 ≤ a < 10, "
                "multiplied by 10 to the power n. Find n.",
                n,
                str(n),
                1,
                follow_through="floor(log(part_a, 10))",
                working=[f"{total_text} = {_plain(total / _ten(n))} × 10^{n}, so n = {n}."],
                distractors=tagged(b_cands, n),
            ),
            part(
                s["ask_c"],
                _numeric(converted),
                conv_text,
                2,
                sensible_range=(0, None),
                follow_through=f"part_a * {_plain(s['factor'])}",
                working=[f"To convert, {flipped} by {factor_text}: {total_text} → {conv_text} {s['unit_c']}."],
                distractors=tagged(c_cands, converted),
            ),
        ],
    )
