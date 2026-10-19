"""Primary-number decimal/fraction generators (group B): two more generators for two
genuinely new graded operations in the ``primary-number`` domain, both
framework ``acara-f10-v9``.

- ``decimal_estimate_add_subtract`` (AC9M6N04, prerequisites
  ``decimal-arithmetic``, ``round-decimals``) -- estimate the result of a
  decimal addition or subtraction by rounding the decimals to a stated
  place. Every question always states the rounding place explicitly (never
  a bare "estimate" with no stated rule) and always asks for the *estimate*
  (rounded-then-combined), never the exact computation (that is the
  prerequisite ``decimal-arithmetic``'s job).

  **This is a special case of what
  ``f10-estimate-counts-and-check-reasonableness`` covers** (estimating a
  collection or a numerical result by rounding to a stated place),
  distinguished from it only by target area (AC9M6N04 is not among that
  skill's own four target areas). The round-to-a-stated-place
  convention implemented here (see ``_round_by_digit_inspection`` below) is
  the one that other skill should match, rather than reinventing it.

  **Design constraint driven directly by ``grade_number``'s 0.051 numeric
  tolerance** (``mathgen/grading/spoken.py`` ``_TOLERANCE = 0.051``):
  every sampled question is wrapped in a construction-time resample loop
  (``max_attempts``) that rejects any candidate whose *estimate* (the
  correct answer) lands within 0.051 of the *exact* result or of either
  other misconception distractor, resampling fresh operands until every
  candidate clears the gap. This is a genuine, necessary guard, not a
  theoretical worry: because the estimate's own rounding errors on each
  operand can partially or wholly cancel when the operands are combined
  (e.g. one operand rounds down while the other rounds up by the same
  amount), naive construction *can* produce an exact seed where the exact
  result and the stated-rule estimate coincide within tolerance -- proven
  directly by the algebra below, not merely observed.

  Three tiers, escalating genuine reasoning steps (never operand magnitude
  alone):

  - ``easy``: two decimals, each with exactly one decimal place (a single
    non-zero tenths digit), rounded to the **nearest whole number** before
    adding or subtracting -- a single rounding step per operand, the
    coarsest stated place this generator uses.
  - ``medium``: two decimals, each with exactly two decimal places, rounded
    to the **nearest tenth** -- a genuinely *finer* stated place. This is
    exactly why the >0.051 gap
    guard matters *more* here: a finer rounding place sits genuinely closer
    to the exact value, shrinking the natural gap between "exact" and
    "estimate" before the resample guard is even applied.
  - ``hard``: **three** decimals (each one decimal place, matching easy's
    own coarser stated place deliberately, to isolate "more terms" as the
    *only* escalation variable from easy -- not a compounded change of both
    term-count and rounding fineness at once), combined via a chain of two
    operations (either two additions, or an addition then a subtraction),
    rounded to the **nearest whole number**. Three operands each needing
    their own rounding step, then two combining operations, is a genuine
    increase in reasoning steps over easy's two operands/one operation.

  Independent check, genuinely different route from construction:
  construction rounds each operand via
  ``_round_by_digit_inspection`` -- literal string/digit inspection of the
  decimal's own digit at the place immediately after the stated rounding
  place, with no arithmetic scaling at all. The check,
  ``_verify_round_half_up_arithmetic``, never touches the decimal string --
  it scales the *value* by ``10 ** place`` (Decimal arithmetic), adds
  ``0.5``, floors to an integer, and scales back down, mirroring
  ``round_decimal``'s own ``(raw + 5) // 10`` integer-arithmetic route in
  ``mathgen/generators/primary_math.py`` -- a genuinely different code path
  (scaled floor-arithmetic vs. string-slice digit inspection), not the same
  rounding call repeated.

  Misconceptions, every value re-verified against the 0.051 grader
  tolerance by the construction-time resample loop above, never merely
  assumed distinct:

  - ``rounds-only-one-operand``: rounds the *first* sampled operand only,
    leaving every other operand exact before combining (e.g. at
    two-operand tiers, ``round(a) + b`` instead of ``round(a) + round(b)``;
    at the three-operand hard tier, ``round(a) + b + c`` -- rounds exactly
    one of the three, leaves the other two exact).
  - ``rounds-to-a-coarser-place``: rounds every operand to a place **one
    step coarser** than the question actually states (e.g. nearest *ten*
    instead of the stated nearest whole number; nearest *whole number*
    instead of the stated nearest tenth). **Deliberately implemented as
    "coarser" only, not "finer/coarser" --
    disclosed explicitly, not silently narrowed**: at this generator's two
    whole-number-place tiers (easy/hard, stated place = nearest whole
    number), every sampled operand has *exactly* one decimal digit, so
    rounding it "one place finer" (nearest tenth) is a no-op identity --
    the operand already *is* its own nearest-tenth value, so a "finer"
    version of this misconception would always coincide exactly with the
    ``computes-exact-result`` misconception below, at every single seed,
    defeating the point of having two separate distractor labels. Rounding
    one place *coarser* (nearest ten) is a real, different value at every
    tier and never structurally collides with the exact-result distractor.
  - ``computes-exact-result-instead-of-estimating`` (the single most
    important distractor): the true,
    unrounded sum/difference/chain. Proven to always differ from the
    correct estimate by strictly more than 0.051 by the construction-time
    resample guard above -- this is the literal edge case the guard exists
    to prevent, not something accepted as an unavoidable collision.

- ``percentage_to_fraction`` (AC9M5N04, prerequisites
  ``fraction-decimal-percent``, ``percentage-of-quantity``) -- express a
  familiar percentage as an equivalent fraction (or decimal), treating 100%
  as the whole. **Scoping decision**: the target output form is fixed to **fraction**
  (``answer: fraction``) -- canonical, exact, and the only form that
  handles a non-terminating-decimal percentage (e.g. 33 1/3%) without
  inventing an undocumented rounding rule the raw contract never states.
  The decimal output form is never graded here.

  **Three-way differentiator against the existing prerequisite skill
  ``fraction_decimal_percent`` in ``mathgen/generators/primary_math.py``**
  (read directly before writing this module) -- that generator already
  does percent<->decimal<->fraction conversion as an ``mcq``, picking a
  random conversion *direction* per question, over a "familiar" percentage
  set only. This skill is genuinely additional, not a re-skin, on three
  independent axes:

  1. **Direction**: always percent -> fraction, one direction only, never
     the reverse and never decimal-involving (``fraction_decimal_percent``
     randomises among three directions including two that never appear
     here at all).
  2. **Response kind**: ``answer: fraction``, **typed**, not ``mcq`` -- the
     learner produces the value directly rather than selecting one of four
     pre-built options.
  3. **Range extension**: the "using 100% as the whole" framing is
     extended, at the hard tier, to percentages **>100% and <1%** (e.g.
     150% -> 3/2, 0.5% -> 1/200) -- a genuine hard-tier reasoning
     extension past the "familiar" 1-100% band that
     ``fraction_decimal_percent`` never leaves.

  Three tiers, escalating genuine reasoning steps:

  - ``easy`` (``variant="benchmark"``): familiar benchmark percentages
    (10%, 20%, 25%, 50%, 75%) -- convert to a fraction the learner is
    expected to already know by heart (already in simplest form, no
    computed reduction required as a *reasoning step*, only recall).
  - ``medium`` (``variant="two_digit_simplify"``): less-familiar two-digit
    percentages (12%, 15%, 24%, 35%, 36%, 45%, 48%, 55%, 65%, 84%, 85%,
    95%) that all genuinely share a common factor with 100 -- the learner
    must find and divide out that common factor as a real computed step
    easy's benchmark percentages skip (e.g. 35% -> 7/20, 15% -> 3/20, both
    standard examples).
  - ``hard`` (``variant="beyond_familiar_band"``): percentages **outside**
    the familiar 1%-100% band -- randomised each seed between an
    **above-100%** sub-case (110%, 120%, 125%, 150%, 175%, 200%, 225%,
    250%) and a **below-1%** sub-case (0.5%, 0.25%, 0.1%, 0.75%), both
    sub-cases confirmed to actually occur across a seed sweep (not merely
    possible in theory) -- genuinely extends "100% as the whole" reasoning
    past the boundary a learner might otherwise assume is fixed.

  Independent check, genuinely different route from construction:
  construction builds the answer as ``percent_value / 100`` directly (a
  single ``Fraction`` division, relying on ``Fraction``'s own automatic
  reduction on construction). ``_verify_percent_fraction_via_explicit_gcd``
  never performs that division -- it builds the *raw*, unreduced
  numerator/denominator pair by hand
  (``percent_value.numerator``, ``percent_value.denominator * 100``) and
  reduces it explicitly via ``math.gcd``, then compares the resulting
  ``(numerator, denominator)`` pair component-wise against the already-
  reduced answer's own ``.numerator``/``.denominator`` -- a genuinely
  different code path (manual GCD reduction of the raw pair vs.
  ``Fraction``'s own automatic reduction on division), never the same
  ``Fraction`` call repeated.

  Misconceptions, each proven a genuinely **different Fraction value**
  (verified by value, not display form, every seed;
  "does not simplify" is never offered as a distractor here, since
  ``grade_fraction`` compares by exact value and an unreduced typed
  fraction is silently ACCEPTED, not graded wrong):

  - ``drops-the-percent-divisor``: states the percentage number itself as
    if it were already the numerator over a denominator of 1 (e.g. 25% ->
    25/1), i.e. ``percent_value`` used directly with no ``/100`` division
    at all. Proven algebraically always distinct from the correct answer
    ``percent_value / 100``: the two values are equal only when
    ``percent_value == percent_value / 100``, i.e. only when
    ``percent_value == 0``, which never occurs (every sampled percent
    value in this generator is strictly positive).
  - ``wrong-power-of-ten-denominator``: an off-by-one-decimal-place error,
    implemented as ``percent_value / 1000`` (one zero too many in the
    denominator -- e.g. "25% -> 25/1000"). Proven
    algebraically always distinct from ``percent_value / 100``: equal only
    when ``percent_value == 0``, never true here.
  - ``treats-100-percent-as-the-ceiling`` (**hard-tier, above-100%
    sub-case only** -- not applicable to easy/medium or the below-1%
    sub-case, stated honestly rather than forced): for a percentage over
    100%, answers as if the whole were capped at 100% and only the
    excess-over-100 part counted (e.g. 150% answered as if it were 50%),
    implemented as ``(percent_value - 100) / 100``. Proven algebraically
    always distinct from ``percent_value / 100``: equal only when
    ``-100 == 0``, never true, for any ``percent_value`` at all.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_FLOOR
from fractions import Fraction

from ..core import check, generator, pick_distractors

# -- decimal_estimate_add_subtract ----------------------------------------

_PLACE_LABELS = {-1: "nearest ten", 0: "nearest whole number", 1: "nearest tenth", 2: "nearest hundredth"}
_TOLERANCE = Decimal("0.051")


def _sample_decimal(rng, places: int, whole_hi: int = 40) -> Decimal:
    """A positive decimal with exactly `places` decimal digits, the last of
    which is always non-zero (so no operand is ever already an exact
    multiple of a coarser place by construction alone -- the resample loop
    below still guards the remaining edge cases directly)."""
    whole = rng.randint(1, whole_hi)
    frac_digits = [rng.randint(0, 9) for _ in range(places - 1)] + [rng.randint(1, 9)]
    frac_str = "".join(str(d) for d in frac_digits)
    return Decimal(f"{whole}.{frac_str}")


def _round_by_digit_inspection(value: Decimal, place: int) -> Decimal:
    """Construction route: round `value` half-up to `place` decimal digits
    (0 = nearest whole, 1 = nearest tenth, -1 = nearest ten, ...) by literal
    digit-string inspection -- looks at the single digit immediately after
    the target place and rounds the truncated value up by one unit there if
    that digit is 5 or more, never via Decimal.quantize or Python's round()."""
    whole_str, _, frac_str = str(value).partition(".")
    if place >= 0:
        frac_str = frac_str.ljust(place + 1, "0")
        kept_frac = frac_str[:place]
        next_digit = int(frac_str[place])
        base = Decimal(f"{whole_str}.{kept_frac}") if kept_frac else Decimal(whole_str)
        unit = Decimal(1).scaleb(-place)
    else:
        drop = -place
        whole_str = whole_str.rjust(drop + 1, "0")
        kept_whole = whole_str[:-drop]
        next_digit = int(whole_str[-drop])
        base = Decimal(kept_whole + "0" * drop)
        unit = Decimal(10**drop)
    if next_digit >= 5:
        base += unit
    return base


def _verify_round_half_up_arithmetic(value: Decimal, place: int, expected: Decimal) -> None:
    """Independent check for `_round_by_digit_inspection`: recomputes the
    round-half-up value via scaled Decimal arithmetic (never touches the
    decimal string) -- shift `value` by `10 ** place` so the target place
    lines up with the ones digit, add 0.5, floor to an integer, then shift
    back. Mirrors round_decimal's own (raw + 5) // 10 integer-arithmetic
    route in mathgen/generators/primary_math.py, a genuinely different code
    path from the string-slice digit inspection construction used."""
    scale = Decimal(10) ** place
    shifted = value * scale
    rounded_units = (shifted + Decimal("0.5")).to_integral_value(rounding=ROUND_FLOOR)
    recomputed = rounded_units / scale
    check(
        recomputed == expected,
        f"rounding {value} to place {place} half-up via scaled arithmetic gives {recomputed}, "
        f"but digit-inspection construction gave {expected}",
    )


def _decimal_text(value: Decimal) -> str:
    return str(value)


def _rounds_only_one_is_viable(target_place: int) -> bool:
    """`rounds-only-one-operand` leaves exactly one operand's own rounding residual uncorrected -- the resulting
    gap from the true estimate is exactly that one operand's own rounding error, which is mathematically
    bounded by half of one unit at `target_place` (`0.5 * 10 ** -target_place`). That bound only exceeds
    grade_number's 0.051 tolerance when rounding to the nearest whole number or coarser (target_place <= 0);
    at the nearest tenth (target_place == 1) the bound is exactly 0.05 < 0.051, so this distractor can *never*
    clear the grader tolerance there, for any operand at all -- not a sampling artefact, a structural
    impossibility, confirmed directly. Disabled at that tier rather than silently offered as a distractor that
    would be accepted as correct."""
    return Decimal("0.5") * (Decimal(10) ** -target_place) > _TOLERANCE


def _two_operand_estimate(rng, places: int, target_place: int) -> dict:
    max_attempts = 3000
    coarser_place = target_place - 1
    only_one_viable = _rounds_only_one_is_viable(target_place)
    for _ in range(max_attempts):
        a = _sample_decimal(rng, places)
        b = _sample_decimal(rng, places)
        op = rng.choice(("add", "subtract"))
        if op == "subtract" and a <= b:
            a, b = b, a

        round_a = _round_by_digit_inspection(a, target_place)
        round_b = _round_by_digit_inspection(b, target_place)
        _verify_round_half_up_arithmetic(a, target_place, round_a)
        _verify_round_half_up_arithmetic(b, target_place, round_b)

        if op == "add":
            exact = a + b
            estimate = round_a + round_b
            rounds_only_one = round_a + b
        else:
            exact = a - b
            estimate = round_a - round_b
            rounds_only_one = round_a - b

        coarse_a = _round_by_digit_inspection(a, coarser_place)
        coarse_b = _round_by_digit_inspection(b, coarser_place)
        rounds_coarser = coarse_a + coarse_b if op == "add" else coarse_a - coarse_b

        if estimate < 0:
            continue
        candidates = [exact, rounds_coarser] + ([rounds_only_one] if only_one_viable else [])
        if any(abs(estimate - c) <= _TOLERANCE for c in candidates):
            continue
        break
    else:
        raise RuntimeError(
            f"could not sample a two-operand decimal estimate question with every distractor "
            f"> {_TOLERANCE} from the answer in {max_attempts} attempts"
        )

    misconceptions = [rounds_coarser] + ([rounds_only_one] if only_one_viable else [])
    return _estimate_result(a, b, None, op, target_place, exact, estimate, misconceptions)


def _three_operand_estimate(rng, places: int, target_place: int) -> dict:
    """`rounds_only_one` here leaves *two* operands' own rounding residuals
    uncorrected (round only `a`, leave `b` and `c` exact), so its bound from
    the true estimate is up to `2 * 0.5 * 10 ** -target_place` -- twice the
    two-operand bound in `_rounds_only_one_is_viable` -- so it remains
    resample-viable at target_place == 0 (this generator's only hard-tier
    preset) even though the two-operand tier at target_place == 1 cannot
    use the same misconception at all (see that function's own docstring)."""
    max_attempts = 4000
    coarser_place = target_place - 1
    for _ in range(max_attempts):
        a = _sample_decimal(rng, places)
        b = _sample_decimal(rng, places)
        c = _sample_decimal(rng, places)
        chain = rng.choice(("add_add", "add_subtract"))
        if chain == "add_subtract" and (a + b) <= c:
            continue

        round_a = _round_by_digit_inspection(a, target_place)
        round_b = _round_by_digit_inspection(b, target_place)
        round_c = _round_by_digit_inspection(c, target_place)
        _verify_round_half_up_arithmetic(a, target_place, round_a)
        _verify_round_half_up_arithmetic(b, target_place, round_b)
        _verify_round_half_up_arithmetic(c, target_place, round_c)

        if chain == "add_add":
            exact = a + b + c
            estimate = round_a + round_b + round_c
            rounds_only_one = round_a + b + c
        else:
            exact = a + b - c
            estimate = round_a + round_b - round_c
            rounds_only_one = round_a + b - c

        coarse_a = _round_by_digit_inspection(a, coarser_place)
        coarse_b = _round_by_digit_inspection(b, coarser_place)
        coarse_c = _round_by_digit_inspection(c, coarser_place)
        rounds_coarser = coarse_a + coarse_b + coarse_c if chain == "add_add" else coarse_a + coarse_b - coarse_c

        if estimate < 0:
            continue
        candidates = [exact, rounds_only_one, rounds_coarser]
        if any(abs(estimate - cand) <= _TOLERANCE for cand in candidates):
            continue
        break
    else:
        raise RuntimeError(
            f"could not sample a three-operand decimal estimate question with every distractor "
            f"> {_TOLERANCE} from the answer in {max_attempts} attempts"
        )

    return _estimate_result(
        a,
        b,
        c,
        chain,
        target_place,
        exact,
        estimate,
        [rounds_only_one, rounds_coarser],
    )


def _estimate_result(a, b, c, op_or_chain, target_place, exact, estimate, misconceptions) -> dict:
    place_label = _PLACE_LABELS[target_place]
    if c is None:
        symbol = "+" if op_or_chain == "add" else "-"
        verb = "adding" if op_or_chain == "add" else "subtracting"
        text = (
            f"Estimate {_decimal_text(a)} {symbol} {_decimal_text(b)} by rounding each decimal "
            f"to the {place_label} before {verb}."
        )
    else:
        symbol = "+" if op_or_chain == "add_add" else "-"
        text = (
            f"Estimate {_decimal_text(a)} + {_decimal_text(b)} {symbol} {_decimal_text(c)} by rounding "
            f"each decimal to the {place_label} before combining."
        )

    answer_val = float(estimate)
    answer_str = str(estimate)
    candidates = [str(exact)] + [str(m) for m in misconceptions]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_val,
        answer_text=answer_str,
        answer_latex=answer_str,
        working=[f"Round each decimal to the {place_label}, then combine: {answer_str}."],
        distractors=pick_distractors(answer_str, candidates),
    )


_ESTIMATE_VARIANT_FNS = {
    "two_place_round_whole": lambda rng: _two_operand_estimate(rng, places=1, target_place=0),
    "two_place_round_tenth": lambda rng: _two_operand_estimate(rng, places=2, target_place=1),
    "three_place_round_whole": lambda rng: _three_operand_estimate(rng, places=1, target_place=0),
}


@generator(
    "decimal_estimate_add_subtract",
    level=6,
    presets={
        "easy": dict(variant="two_place_round_whole"),
        "medium": dict(variant="two_place_round_tenth"),
        "hard": dict(variant="three_place_round_whole"),
    },
)
def decimal_estimate_add_subtract(rng, variant="two_place_round_whole"):
    """Estimate a decimal addition or subtraction by rounding to a stated
    place. See the module docstring for the full tier/misconception design
    and the 0.051-gap construction-time resample guard."""
    if variant not in _ESTIMATE_VARIANT_FNS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_ESTIMATE_VARIANT_FNS)}")
    return _ESTIMATE_VARIANT_FNS[variant](rng)


# -- percentage_to_fraction -------------------------------------------------

_EASY_PERCENTS = (10, 20, 25, 50, 75)
_EASY_PERCENTS_WIDE = (5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90)
_ASKS = (
    "What fraction is equivalent to {d}, using 100% as the whole?",
    "Write {d} as a fraction in simplest form, using 100% as the whole.",
    "{d} of a whole is what fraction of that whole?",
)
_MEDIUM_PERCENTS = (12, 15, 24, 35, 36, 45, 48, 55, 65, 84, 85, 95)
_HARD_ABOVE_PERCENTS = (110, 120, 125, 150, 175, 200, 225, 250)
_HARD_BELOW = (
    (Fraction(1, 2), "0.5%"),
    (Fraction(1, 4), "0.25%"),
    (Fraction(1, 10), "0.1%"),
    (Fraction(3, 4), "0.75%"),
)


def _fraction_text(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _verify_percent_fraction_via_explicit_gcd(percent_value: Fraction, answer: Fraction) -> None:
    """Independent check: construction builds the answer as
    `percent_value / 100` directly, relying on Fraction's own automatic
    reduction on division. This check never performs that division -- it
    builds the raw, unreduced numerator/denominator pair by hand and
    reduces it explicitly via math.gcd, then compares the resulting pair
    component-wise to the already-reduced answer."""
    raw_num = percent_value.numerator
    raw_den = percent_value.denominator * 100
    g = math.gcd(raw_num, raw_den)
    reduced_num, reduced_den = raw_num // g, raw_den // g
    check(
        (reduced_num, reduced_den) == (answer.numerator, answer.denominator),
        f"{percent_value}% of the whole reduces via explicit gcd to {reduced_num}/{reduced_den}, "
        f"but the answer is {answer.numerator}/{answer.denominator}",
    )


def _build_percentage_question(percent_value: Fraction, display: str, above_100: bool, ask: str | None = None) -> dict:
    answer = percent_value / 100
    _verify_percent_fraction_via_explicit_gcd(percent_value, answer)

    drops_percent_divisor = percent_value
    wrong_power_of_ten = percent_value / 1000
    candidates = [drops_percent_divisor, wrong_power_of_ten]
    if above_100:
        candidates.append((percent_value - 100) / 100)

    answer_text = _fraction_text(answer)
    text = (ask or _ASKS[0]).format(d=display)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{display} = {answer_text} of the whole (100%), in simplest form."],
        distractors=pick_distractors(answer_text, [_fraction_text(c) for c in candidates]),
    )


def _benchmark(rng, wide: bool = False) -> dict:
    if wide:  # more familiar percentages and three wordings; drawn after the original draws are skipped
        percent = rng.choice(_EASY_PERCENTS_WIDE)
        return _build_percentage_question(Fraction(percent), f"{percent}%", above_100=False, ask=rng.choice(_ASKS))
    percent = rng.choice(_EASY_PERCENTS)
    return _build_percentage_question(Fraction(percent), f"{percent}%", above_100=False)


def _two_digit_simplify(rng) -> dict:
    percent = rng.choice(_MEDIUM_PERCENTS)
    return _build_percentage_question(Fraction(percent), f"{percent}%", above_100=False)


def _beyond_familiar_band(rng) -> dict:
    sub_case = rng.choice(("above", "below"))
    if sub_case == "above":
        percent = rng.choice(_HARD_ABOVE_PERCENTS)
        return _build_percentage_question(Fraction(percent), f"{percent}%", above_100=True)
    percent_value, display = rng.choice(_HARD_BELOW)
    return _build_percentage_question(percent_value, display, above_100=False)


_PERCENTAGE_VARIANT_FNS = {
    "benchmark": _benchmark,
    "two_digit_simplify": _two_digit_simplify,
    "beyond_familiar_band": _beyond_familiar_band,
}


@generator(
    "percentage_to_fraction",
    level=5,
    presets={
        "easy": dict(variant="benchmark"),
        "medium": dict(variant="two_digit_simplify"),
        "hard": dict(variant="beyond_familiar_band"),
    },
)
def percentage_to_fraction(rng, variant="benchmark", wide=False):
    """Convert a percentage to its equivalent fraction, using 100% as the
    whole. See the module docstring for the full tier design, the
    three-way differentiator against fraction_decimal_percent, and the
    misconception derivations."""
    if variant not in _PERCENTAGE_VARIANT_FNS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_PERCENTAGE_VARIANT_FNS)}")
    if wide and variant == "benchmark":
        return _benchmark(rng, wide=True)
    return _PERCENTAGE_VARIANT_FNS[variant](rng)
