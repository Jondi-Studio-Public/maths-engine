"""Primary-number place-value generators (group A): two more generators for two genuinely new graded operations in the
``primary-number`` domain, both framework ``acara-f10-v9``.

The discriminator between the two skills (both scale a value by a
power/multiple of ten, so a clear line is required or they blur together):

- ``powers_ten_multiply_divide``: the operand and the power/multiple of ten
  are both given directly in the question (e.g. "What is 4.6 x 100?") -- a
  direct place-value-shift computation, no prior fact is referenced.
- ``mental_mult_div_extension``: the question states a known basic fact as
  input (e.g. "Given that 6 x 7 = 42, what is 6 x 700?") and asks for the
  derived scaled result -- the stated fact is the discriminator; the graded
  operation is extending it, not performing a fresh place-value shift from
  scratch.

- ``powers_ten_multiply_divide`` (AC9M4N05 and AC9M6N06 -- one skill, two
  target areas -- prerequisites ``decimal-arithmetic``, ``place-value-to-100``).
  The graded scope is exactly multiplying or dividing a natural number or
  decimal by a multiple or power of ten using place-value relationships;
  explaining digit movement (a rubric-graded clause) is out of scope. AC9M4N05 (Year 4) names only natural numbers; AC9M6N06
  (Year 6) is its own decimal extension -- so the whole-number tiers
  (``easy``, ``medium``) evidence AC9M4N05 and the decimal tier (``hard``)
  evidences AC9M6N06's own extension; neither area's own text is claimed to
  be fully evidenced by every tier.
    - ``easy`` (``variant="whole_power"``): multiply or divide a natural
      number by 10, 100 or 1000 -- a direct digit shift, no decimal.
    - ``medium`` (``variant="whole_multiple"``): multiply or divide a natural
      number by a non-power multiple of ten (e.g. x30, x400, /20) --
      genuinely two steps (shift by the power-of-ten component, then scale
      by the remaining single digit), a real procedural escalation over
      easy's single shift.
    - ``hard`` (``variant="decimal_power"``): multiply or divide a
      **decimal** by a power of ten (e.g. 4.6 x 100 = 460, 3.2 / 10 = 0.32)
      -- the decimal-point shift is a distinct, genuinely harder place-value
      operation than an integer digit shift (a digit can cross the decimal
      point, and trailing/leading zeros behave differently), matching
      AC9M6N06's own decimal extension. Scoped to pure powers of ten (not
      "multiple of ten" decimal factors) to match the two
      illustrative examples and to keep the decimal-point-shift reasoning
      itself, not a compounded two-step multiple, the hard tier's own added
      complexity (the medium tier already covers the two-step-multiple
      reasoning, for whole numbers).
  Construction always shifts the operand's own digit string (or decimal
  point) directly -- see ``_shift_decimal_point``. The independent check
  never re-runs that same string/index shift: it recomputes via exact
  integer arithmetic (``_verify_direct_scale``, easy/medium) or exact
  ``Decimal`` arithmetic (``_verify_decimal_shift``, hard) against the
  literal power/factor value, a genuinely different route.
  Misconceptions: wrong-direction (applies the opposite operation, or -- at
  the decimal tier -- shifts the decimal point the wrong way);
  off-by-one-place (shifts one place too few or too many, e.g. x100 treated
  as x10); drops-the-decimal-point (hard-tier only -- treats the decimal
  operand as if it were the bare digit string, e.g. 4.6 treated as 46,
  before applying the shift). Hard-tier distractors are additionally
  resampled (``max_attempts`` loop) until every candidate differs from the
  correct answer by strictly more than 0.051 -- ``grade_number``'s numeric
  tolerance (``mathgen/grading/spoken.py::_TOLERANCE``) -- since a
  decimal-valued ``answer: number`` skill cannot rely on "distinct" alone.

- ``mental_mult_div_extension`` (AC9M4A02 -- prerequisites
  ``multiplication-facts``, ``whole-number-division-facts``). The
  graded scope is exactly using a known multiplication or division fact to
  compute the related larger calculation; explaining the extension (which
  would need a ``communication`` capability gate) is out of scope.
    - ``easy`` (``variant="scale_a_by_ten"``): a known single-digit x
      single-digit multiplication fact is stated; scale one factor by 10
      (e.g. 6x7=42 -> 6x70).
    - ``medium`` (``variant="scale_dividend_by_ten"``): a known single-digit
      division fact is stated instead (e.g. 42/7=6); scale the dividend by
      10 (42/7=6 -> 420/7=60). The escalation from easy is an *operation*
      change (multiplication -> division), not a bigger multiplier --
      AC9M4A02's own text names "multiplication facts ... and related
      division facts" as the two things a learner must extend.
    - ``hard`` (``variant="scale_both_factors"``): a known multiplication
      fact is stated; scale *both* factors, each independently by 10 or 100
      (e.g. 6x7=42 -> 60x700), requiring the learner to combine two
      independent scalings -- a genuinely different derivation path from
      either easier tier's single-quantity scaling.
  Construction always scales the known stated result directly (the product
  ``p`` for the multiplication tiers, the quotient ``a`` for the division
  tier) by the stated scale factor(s). The independent check never reuses
  that route: it re-derives the scaled operand(s) independently and
  multiplies/divides them fresh (``_verify_scaled_multiplication``,
  ``_verify_scaled_division``).
  Misconceptions: forgets-to-scale (states the original fact's answer
  unchanged); wrong-power-scale (scales by the wrong power -- one zero too
  many/few); scales-one-factor-only (hard-tier only -- scales only one of
  the two factors when both needed scaling).
"""

from __future__ import annotations

from decimal import Decimal

from ..core import check, generator, pick_distractors

# -- powers_ten_multiply_divide -----------------------------------------------

_POWERS = (10, 100, 1000)
_EXPONENT_OF_POWER = {10: 1, 100: 2, 1000: 3}
_POWER_OF_EXPONENT = {1: 10, 2: 100, 3: 1000}
# One place too few/many, clamped within the 10/100/1000 family -- matches
# the misconception's own illustrative example (x100 treated as x10).
_ADJACENT_EXPONENT = {1: 2, 2: 1, 3: 2}


def _shift_decimal_point(value_str: str, places: int) -> str:
    """Construction primitive shared by every ``powers_ten_multiply_divide``
    tier: shifts the decimal point in a (non-negative) decimal digit string
    ``places`` positions to the right (positive) or left (negative), padding
    with zeros as needed and normalising leading/trailing zeros. This is a
    literal string-level place-value shift -- mirrors how a learner would
    physically move the decimal point -- never numeric multiplication or
    division. A plain integer string (no ``.``) shifted right is exactly the
    "append zeros" digit shift used by the whole-number tiers; the same
    primitive shifted left strips/pads zeros for whole-number division.
    """
    if "." in value_str:
        int_part, frac_part = value_str.split(".")
    else:
        int_part, frac_part = value_str, ""
    digits = int_part + frac_part
    point = len(int_part) + places
    if point <= 0:
        digits = "0" * (-point) + digits
        point = 0
    if point >= len(digits):
        digits = digits + "0" * (point - len(digits))
    new_int = digits[:point] or "0"
    new_frac = digits[point:]
    new_int = new_int.lstrip("0") or "0"
    if new_frac:
        new_frac = new_frac.rstrip("0")
    return f"{new_int}.{new_frac}" if new_frac else new_int


def _verify_direct_scale(operand: int, scale: int, operation: str, answer: int) -> None:
    """Independent check for the ``whole_power``/``whole_multiple`` tiers:
    construction shifts ``operand``'s own digit string. This check never
    re-runs that shift -- it recomputes via exact integer multiplication or
    floor division by the literal ``scale`` value (the power or the
    non-power multiple), a genuinely different arithmetic route."""
    if operation == "multiply":
        expected = operand * scale
    else:
        quotient, remainder = divmod(operand, scale)
        check(remainder == 0, f"{operand} does not divide exactly by {scale}")
        expected = quotient
    check(expected == answer, f"{operation} of {operand} and {scale} gives {expected}, expected {answer}")


def _verify_decimal_shift(operand_str: str, power: int, operation: str, answer_str: str) -> None:
    """Independent check for the ``decimal_power`` tier: construction shifts
    ``operand_str``'s own decimal-point string. This check never re-runs
    that shift -- it recomputes via exact ``Decimal`` multiplication or
    division by the literal power-of-ten value, a genuinely different
    arithmetic route."""
    operand = Decimal(operand_str)
    if operation == "multiply":
        expected = operand * Decimal(power)
    else:
        expected = operand / Decimal(power)
    check(
        expected == Decimal(answer_str),
        f"{operation} of {operand_str} and {power} gives {expected}, expected {answer_str}",
    )


def _whole_power(rng):
    operation = rng.choice(("multiply", "divide"))
    power = rng.choice(_POWERS)
    exponent = _EXPONENT_OF_POWER[power]
    if operation == "multiply":
        operand = rng.randint(2, 99)
    else:
        quotient = rng.randint(2, 99)
        operand = quotient * power  # guarantees exact division

    places = exponent if operation == "multiply" else -exponent
    answer_str = _shift_decimal_point(str(operand), places)
    check("." not in answer_str, f"whole-number tier produced a non-integer result {answer_str!r}")
    answer = int(answer_str)
    _verify_direct_scale(operand, power, operation, answer)

    # Misconceptions:
    #  - wrong-direction: applies the opposite operation (divides when
    #    multiplying was asked, or vice versa).
    #  - off-by-one-place: applies the same operation but shifts by an
    #    adjacent power of ten (e.g. x100 treated as x10).
    if operation == "multiply":
        wrong_direction = operand // power
    else:
        wrong_direction = operand * power
    adjacent_exponent = _ADJACENT_EXPONENT[exponent]
    adjacent_places = adjacent_exponent if operation == "multiply" else -adjacent_exponent
    off_by_one_str = _shift_decimal_point(str(operand), adjacent_places)

    candidates = [str(wrong_direction)]
    if "." not in off_by_one_str:
        candidates.append(off_by_one_str)
    distractors = pick_distractors(str(answer), candidates)

    symbol = "×" if operation == "multiply" else "÷"
    text = f"What is {operand} {symbol} {power}?"
    return dict(
        text=text,
        latex=rf"{operand} {'\\times' if operation == 'multiply' else '\\div'} {power} = \square",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"{operand} {symbol} {power} = {answer}."],
        distractors=distractors,
    )


def _whole_multiple(rng):
    operation = rng.choice(("multiply", "divide"))
    single_digit = rng.randint(2, 9)
    power_component = rng.choice((10, 100))
    exponent = _EXPONENT_OF_POWER[power_component]
    factor = single_digit * power_component

    if operation == "multiply":
        operand = rng.randint(2, 99)
    else:
        quotient = rng.randint(2, 99)
        operand = quotient * factor  # guarantees exact division

    # Two-step construction: shift by the power-of-ten component, then
    # scale by the remaining single digit -- a genuine procedural
    # escalation over the whole_power tier's single shift.
    if operation == "multiply":
        shifted = _shift_decimal_point(str(operand), exponent)
        check("." not in shifted, f"whole-number tier produced a non-integer result {shifted!r}")
        answer = int(shifted) * single_digit
    else:
        check(operand % single_digit == 0, f"{operand} does not divide exactly by the single digit {single_digit}")
        step1 = operand // single_digit
        shifted = _shift_decimal_point(str(step1), -exponent)
        check("." not in shifted, f"whole-number tier produced a non-integer result {shifted!r}")
        answer = int(shifted)
    _verify_direct_scale(operand, factor, operation, answer)

    # Misconceptions:
    #  - wrong-direction: applies the opposite operation.
    #  - off-by-one-place: uses the adjacent power-of-ten component (same
    #    single digit), e.g. x30 treated as x300.
    if operation == "multiply":
        wrong_direction = operand // factor
    else:
        wrong_direction = operand * factor
    adjacent_exponent = _ADJACENT_EXPONENT[exponent]
    adjacent_power_component = _POWER_OF_EXPONENT[adjacent_exponent]
    adjacent_factor = single_digit * adjacent_power_component
    if operation == "multiply":
        off_by_one = operand * adjacent_factor
    else:
        off_by_one = operand // adjacent_factor

    candidates = [str(wrong_direction), str(off_by_one)]
    distractors = pick_distractors(str(answer), candidates)

    symbol = "×" if operation == "multiply" else "÷"
    text = f"What is {operand} {symbol} {factor}?"
    return dict(
        text=text,
        latex=rf"{operand} {'\\times' if operation == 'multiply' else '\\div'} {factor} = \square",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"{operand} {symbol} {factor} = {answer}."],
        distractors=distractors,
    )


def _decimal_power(rng):
    max_attempts = 2000
    for _ in range(max_attempts):
        operation = rng.choice(("multiply", "divide"))
        power = rng.choice(_POWERS)
        exponent = _EXPONENT_OF_POWER[power]
        intpart = rng.randint(1, 99)
        decimal_digit = rng.randint(1, 9)  # nonzero: a genuine one-decimal-place value
        operand_str = f"{intpart}.{decimal_digit}"

        places = exponent if operation == "multiply" else -exponent
        answer_str = _shift_decimal_point(operand_str, places)
        answer_val = float(answer_str)

        # Misconceptions:
        #  - wrong-direction: shifts the decimal point the wrong way.
        #  - off-by-one-place: shifts by an adjacent power of ten.
        #  - drops-the-decimal-point: treats the operand as if it were the
        #    bare digit string (4.6 -> 46) before shifting.
        wrong_direction_str = _shift_decimal_point(operand_str, -places)
        adjacent_exponent = _ADJACENT_EXPONENT[exponent]
        adjacent_places = adjacent_exponent if operation == "multiply" else -adjacent_exponent
        off_by_one_str = _shift_decimal_point(operand_str, adjacent_places)
        digits_as_int_str = f"{intpart}{decimal_digit}"
        drops_decimal_str = _shift_decimal_point(digits_as_int_str, places)

        candidate_strs = [wrong_direction_str, off_by_one_str, drops_decimal_str]
        # grade_number's ~0.05 numeric tolerance (mathgen/grading/spoken.py
        # _TOLERANCE = 0.051) means a decimal distractor within that gap of
        # the correct answer would be silently accepted as correct -- every
        # candidate must clear it, checked directly, not assumed.
        if any(abs(answer_val - float(c)) <= 0.051 for c in candidate_strs if c != answer_str):
            continue
        break
    else:
        raise RuntimeError(
            f"could not sample a decimal_power question with every distractor > 0.051 from the answer "
            f"in {max_attempts} attempts"
        )

    _verify_decimal_shift(operand_str, power, operation, answer_str)

    distractors = pick_distractors(answer_str, candidate_strs)

    symbol = "×" if operation == "multiply" else "÷"
    text = f"What is {operand_str} {symbol} {power}?"
    return dict(
        text=text,
        latex=rf"{operand_str} {'\\times' if operation == 'multiply' else '\\div'} {power} = \square",
        answer=answer_val,
        answer_text=answer_str,
        answer_latex=answer_str,
        working=[f"{operand_str} {symbol} {power} = {answer_str}."],
        distractors=distractors,
    )


def _decimal_power_missing_factor(rng):
    """hard tier (new): gives the operand and the RESULT and asks for the
    missing power of ten -- the inverse direction of every other tier
    (which give the operand and the power, and ask for the result). Text
    shape "OPERAND SYMBOL ___ = RESULT. What is the missing power of ten?"
    is structurally different from every other tier's "What is X op Y?" --
    the blank sits in the middle of a stated equation, not at the end of a
    question -- and the reasoning direction (read the place-value shift
    backwards, off the two decimal strings) is inverted relative to every
    other tier's forward compute."""
    operation = rng.choice(("multiply", "divide"))
    power = rng.choice(_POWERS)
    exponent = _EXPONENT_OF_POWER[power]
    intpart = rng.randint(1, 99)
    decimal_digit = rng.randint(1, 9)
    operand_str = f"{intpart}.{decimal_digit}"

    places = exponent if operation == "multiply" else -exponent
    answer_str = _shift_decimal_point(operand_str, places)

    # Independent check: reuses `_verify_decimal_shift` (already an
    # independent Decimal-arithmetic route relative to `_shift_decimal_point`'s
    # string-level construction) -- confirms operand_str op power == answer_str.
    _verify_decimal_shift(operand_str, power, operation, answer_str)

    # Misconception: off-by-one-place -- the adjacent power of ten in the
    # 10/100/1000 family (matches this skill's own "off-by-one-place"
    # misconception id).
    adjacent_exponent = _ADJACENT_EXPONENT[exponent]
    adjacent_power = _POWER_OF_EXPONENT[adjacent_exponent]
    candidate_strs = [str(adjacent_power)]

    symbol = "×" if operation == "multiply" else "÷"
    text = f"{operand_str} {symbol} ___ = {answer_str}. What is the missing power of ten?"
    return dict(
        text=text,
        latex=rf"{operand_str} {'\\times' if operation == 'multiply' else '\\div'} \square = {answer_str}",
        answer=power,
        answer_text=str(power),
        answer_latex=str(power),
        working=[f"{operand_str} {symbol} {power} = {answer_str}."],
        distractors=pick_distractors(str(power), candidate_strs),
    )


_VARIANTS = {
    "whole_power": _whole_power,
    "whole_multiple": _whole_multiple,
    "decimal_power": _decimal_power,
    "decimal_power_missing_factor": _decimal_power_missing_factor,
}


@generator(
    "powers_ten_multiply_divide",
    level=4,
    presets={
        "easy": dict(variant="whole_power"),
        "medium": dict(variant="whole_multiple"),
        "hard": dict(variant="decimal_power"),
    },
)
def powers_ten_multiply_divide(rng, variant="whole_power"):
    """Multiply or divide a natural number or decimal by a multiple or power
    of ten using place-value relationships. See the module docstring for the
    difficulty rationale and the AC9M4N05/AC9M6N06 tier mapping."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_VARIANTS)}")
    return _VARIANTS[variant](rng)


# -- mental_mult_div_extension ------------------------------------------------

_ADJACENT_SCALE = {10: 100, 100: 10}


def _verify_scaled_multiplication(a: int, b: int, sa: int, sb: int, answer: int) -> None:
    """Independent check: construction scales the known product ``p = a*b``
    directly (``p * sa * sb``). This check never reuses ``p`` -- it
    re-derives both scaled operands independently (``a*sa``, ``b*sb``) and
    multiplies them fresh, a genuinely different route. At the easy tier
    ``sb`` is 1 (only one factor scaled); the check still performs a fresh
    multiplication of the two independently-scaled operands rather than
    reusing the stated product, so it remains a distinct route even there."""
    scaled_a = a * sa
    scaled_b = b * sb
    fresh = scaled_a * scaled_b
    check(fresh == answer, f"{scaled_a} × {scaled_b} = {fresh}, expected {answer}")


def _verify_scaled_division(p: int, b: int, a: int, s: int, answer: int) -> None:
    """Independent check for the division-fact tier: construction scales the
    known quotient ``a`` directly (``a * s``). This check never reuses
    ``a`` -- it re-derives the scaled dividend (``p * s``) independently and
    divides it fresh by the stated divisor ``b``, a genuinely different
    route."""
    scaled_p = p * s
    quotient, remainder = divmod(scaled_p, b)
    check(remainder == 0, f"{scaled_p} does not divide exactly by {b}")
    check(quotient == answer, f"{scaled_p} ÷ {b} = {quotient}, expected {answer}")


def _scale_a_by_ten(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    p = a * b
    sa = 10
    answer = p * sa
    _verify_scaled_multiplication(a, b, sa, 1, answer)

    # Misconceptions:
    #  - forgets-to-scale: states the original fact's answer unchanged.
    #  - wrong-power-scale: scales by the wrong power (100 instead of 10).
    forgets_to_scale = p
    wrong_power = p * _ADJACENT_SCALE[sa]
    candidates = [str(forgets_to_scale), str(wrong_power)]

    text = f"Given that {a} × {b} = {p}, what is {a * sa} × {b}?"
    return dict(
        text=text,
        latex=rf"{a} \times {b} = {p} \Rightarrow {a * sa} \times {b} = \square",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"{a} × {b} = {p}.", f"{a} × {sa} = {a * sa}, so {a * sa} × {b} = {answer}."],
        distractors=pick_distractors(str(answer), candidates),
    )


def _scale_dividend_by_ten(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    p = a * b  # exact by construction: p / b = a
    s = 10
    answer = a * s
    _verify_scaled_division(p, b, a, s, answer)

    # Misconceptions:
    #  - forgets-to-scale: states the original fact's answer unchanged.
    #  - wrong-power-scale: scales by the wrong power (100 instead of 10).
    forgets_to_scale = a
    wrong_power_dividend = p * _ADJACENT_SCALE[s]
    wrong_power, remainder = divmod(wrong_power_dividend, b)
    check(remainder == 0, f"{wrong_power_dividend} does not divide exactly by {b}")
    candidates = [str(forgets_to_scale), str(wrong_power)]

    text = f"Given that {p} ÷ {b} = {a}, what is {p * s} ÷ {b}?"
    return dict(
        text=text,
        latex=rf"{p} \div {b} = {a} \Rightarrow {p * s} \div {b} = \square",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"{p} ÷ {b} = {a}.", f"{p} × {s} = {p * s}, so {p * s} ÷ {b} = {answer}."],
        distractors=pick_distractors(str(answer), candidates),
    )


def _scale_both_factors(rng):
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    p = a * b
    sa = rng.choice((10, 100))
    sb = rng.choice((10, 100))
    answer = p * sa * sb
    _verify_scaled_multiplication(a, b, sa, sb, answer)

    # Misconceptions:
    #  - forgets-to-scale: states the original fact's answer unchanged.
    #  - wrong-power-scale: uses an adjacent power on one factor's scale.
    #  - scales-one-factor-only: scales only a (forgets to scale b).
    forgets_to_scale = p
    wrong_power = p * _ADJACENT_SCALE[sa] * sb
    scales_one_factor_only = p * sa
    candidates = [str(forgets_to_scale), str(wrong_power), str(scales_one_factor_only)]

    text = f"Given that {a} × {b} = {p}, what is {a * sa} × {b * sb}?"
    return dict(
        text=text,
        latex=rf"{a} \times {b} = {p} \Rightarrow {a * sa} \times {b * sb} = \square",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"{a} × {b} = {p}.",
            f"{a} × {sa} = {a * sa} and {b} × {sb} = {b * sb}.",
            f"{a * sa} × {b * sb} = {answer}.",
        ],
        distractors=pick_distractors(str(answer), candidates),
    )


def _scale_product_divide_back(rng):
    """hard tier (new): states a MULTIPLICATION fact, asks a DIVISION
    question built from the scaled product -- text shape "Given that A x B
    = P, what is C / D?" exists in neither easy (x/x) nor medium (/,/), and
    the reasoning (recognise which scaled quantity cancels when dividing a
    scaled product by a scaled factor) is a genuine escalation over scaling
    both factors and multiplying."""
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    p = a * b
    sa = rng.choice((10, 100))
    sb = rng.choice((10, 100))
    scaled_b = b * sb
    scaled_product = p * sa * sb
    answer = a * sa

    # Independent check: construction computed `answer` directly as `a * sa`.
    # This check never reuses that -- it recomputes `scaled_product` fresh
    # from `p`, `sa`, `sb` and divides it by `scaled_b` via divmod, a
    # genuinely different route (division, not the direct multiplication
    # construction used).
    fresh_scaled_product = p * sa * sb
    quotient, remainder = divmod(fresh_scaled_product, scaled_b)
    check(remainder == 0, f"{fresh_scaled_product} does not divide exactly by {scaled_b}")
    check(quotient == answer, f"{fresh_scaled_product} / {scaled_b} = {quotient}, expected {answer}")

    # Misconceptions (reuse the skill's existing three ids):
    #  - forgets-to-scale: answers the original a, unscaled.
    #  - wrong-power-scale: scales a by the WRONG power (the adjacent one to sa).
    #  - scales-one-factor-only: divides the scaled product by the UNSCALED b,
    #    as if only the numerator had been scaled (forgets b was scaled too).
    forgets_to_scale = a
    wrong_power_scale = a * _ADJACENT_SCALE[sa]
    scales_one_factor_only = scaled_product // b
    candidates = [str(forgets_to_scale), str(wrong_power_scale), str(scales_one_factor_only)]

    text = f"Given that {a} × {b} = {p}, what is {scaled_product} ÷ {scaled_b}?"
    return dict(
        text=text,
        latex=rf"{a} \times {b} = {p} \Rightarrow {scaled_product} \div {scaled_b} = \square",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"{a} × {b} = {p}.",
            f"{p} × {sa} × {sb} = {scaled_product} and {b} × {sb} = {scaled_b}.",
            f"{scaled_product} ÷ {scaled_b} = {answer}.",
        ],
        distractors=pick_distractors(str(answer), candidates),
    )


_MENTAL_VARIANTS = {
    "scale_a_by_ten": _scale_a_by_ten,
    "scale_dividend_by_ten": _scale_dividend_by_ten,
    "scale_both_factors": _scale_both_factors,
    "scale_product_divide_back": _scale_product_divide_back,
}


@generator(
    "mental_mult_div_extension",
    level=4,
    presets={
        "easy": dict(variant="scale_a_by_ten"),
        "medium": dict(variant="scale_dividend_by_ten"),
        "hard": dict(variant="scale_both_factors"),
    },
)
def mental_mult_div_extension(rng, variant="scale_a_by_ten"):
    """Use a known multiplication or division fact to compute the related
    larger calculation. See the module docstring for the difficulty
    rationale."""
    if variant not in _MENTAL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_MENTAL_VARIANTS)}")
    return _MENTAL_VARIANTS[variant](rng)
