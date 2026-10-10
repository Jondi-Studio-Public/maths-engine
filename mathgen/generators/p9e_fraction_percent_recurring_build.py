"""Generates `p9e_fraction_percent_quantity_strategy` and `p9e_terminating_recurring_decimal` questions:
fractions or percentages of a quantity, and terminating versus recurring decimals.

Two F-10 number skills.

- ``p9e_fraction_percent_quantity_strategy`` (`f10-fraction-percent-quantity-
  strategy`, AC9M6N07) -- find a familiar fraction/decimal/percentage of a
  whole-number quantity, with the REGISTER (fraction, decimal or percentage)
  drawn at random each seed. AC9M6N07 already carries three shipping rows:
  ``fraction-of-quantity`` (fraction operand only), ``percentage-of-
  quantity``/``percentage-discount`` (percentage operand only, from the
  fixed set {5, 10, 20, 25, 50, 75}). None of the three ever shows a bare
  DECIMAL operand, and none ever uses an eighths-family value (1/8, 3/8,
  5/8, 7/8 <-> 12.5%, 37.5%, 62.5%, 87.5% <-> 0.125, 0.375, 0.625, 0.875).
  This skill's un-owned slice: the SAME underlying operation (find a
  familiar fraction/decimal/percentage of a whole-number quantity, result
  always exact), but the register is drawn randomly from all three each
  seed, and the eighths family plus the bare decimal register -- neither of
  which any sibling ever reaches -- are always in scope.

- ``p9e_terminating_recurring_decimal`` (`f10-terminating-recurring-
  decimals`, AC9M8N03) -- classify a decimal (an ellipsis marks a repeating
  block; its absence means the shown finite decimal IS the exact value) and
  convert it to its equivalent fraction in lowest terms. The un-owned slice
  against the two prerequisites: ``recurring-decimal-to-fraction``
  (``mathgen/generators/curriculum_expansion.py``) only ever handles a
  SINGLE repeating digit; ``fraction-decimal-percent`` only ever handles a
  TERMINATING decimal from a small fixed denominator set. This skill mixes
  both terminating and recurring cases in the same draw space -- neither
  prerequisite ever mixes the two -- and adds the two-digit repeating block
  and the mixed (non-repeating lead + repeating tail) case that neither
  prerequisite ever reaches. The TYPE (terminating vs recurring) is legible
  directly from the shown notation rather than independently inferred from
  an abstract fraction; this is disclosed honestly here,
  not claimed as an independent-judgement classifier.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    pick_distractors,
)

_RESAMPLE_ATTEMPTS = 100

# =============================================================================
# Skill 1: p9e_fraction_percent_quantity_strategy
# =============================================================================

# Every entry: key -> (Fraction value, fraction text, decimal text, percent
# text). All three registers of the same value are pre-computed together
# here, so a question never derives one register from another at render time
# by a route different from construction -- this table IS the single source
# of truth for all three registers.
_VALUE_TABLE: dict[str, tuple[Fraction, str, str, str]] = {
    "1/4": (Fraction(1, 4), "1/4", "0.25", "25%"),
    "1/2": (Fraction(1, 2), "1/2", "0.5", "50%"),
    "3/4": (Fraction(3, 4), "3/4", "0.75", "75%"),
    "1/5": (Fraction(1, 5), "1/5", "0.2", "20%"),
    "2/5": (Fraction(2, 5), "2/5", "0.4", "40%"),
    "1/8": (Fraction(1, 8), "1/8", "0.125", "12.5%"),
    "3/8": (Fraction(3, 8), "3/8", "0.375", "37.5%"),
    "5/8": (Fraction(5, 8), "5/8", "0.625", "62.5%"),
    "7/8": (Fraction(7, 8), "7/8", "0.875", "87.5%"),
}
_EASY_KEYS = ("1/4", "1/2", "3/4", "1/5", "2/5")
_EIGHTHS_KEYS = ("1/8", "3/8", "5/8", "7/8")
_ALL_KEYS = _EASY_KEYS + _EIGHTHS_KEYS
_REGISTERS = ("fraction", "decimal", "percentage")

# hard tier: the register forced onto each value is the one furthest from its
# own usual mental route. Quarters/halves are read most fluently as a
# fraction or a percent (a quarter, 25%), so the decimal register (0.25) is
# forced. Fifths are read most fluently as a decimal or a percent (0.2,
# 20%), so the fraction register (1/5) is forced. Eighths are read most
# fluently as a fraction (an eighth), so the percentage register is forced
# -- 37.5% = 3/8 is genuinely the harder direction, matching the standard
# worked example.
_LEAST_NATURAL_REGISTER = {
    "1/4": "decimal",
    "1/2": "decimal",
    "3/4": "decimal",
    "1/5": "fraction",
    "2/5": "fraction",
    "1/8": "percentage",
    "3/8": "percentage",
    "5/8": "percentage",
    "7/8": "percentage",
}
# wrong-benchmark (medium/hard, eighths only): the nearest quarter/half a
# learner might substitute for the true eighths-family value.
_NEAREST_BENCHMARK = {
    "1/8": Fraction(1, 4),
    "3/8": Fraction(1, 2),
    "5/8": Fraction(1, 2),
    "7/8": Fraction(3, 4),
}


def _draw_quantity(rng, denominator: int, lo: int, hi: int) -> int:
    """A quantity in [lo, hi] divisible by `denominator`, so value*quantity
    is exactly a whole number -- reject-resample, never force-rounded."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        quantity = rng.randint(lo, hi)
        if quantity % denominator == 0:
            return quantity
    raise ValueError("p9e_fraction_percent_quantity_strategy: no divisible quantity found")


def _shown_value_as_fraction(register: str, frac_text: str, dec_text: str, pct_text: str) -> Fraction:
    """Re-derive the value from the PRESENTED register's own text -- a route
    independent of the `_VALUE_TABLE` Fraction object construction already
    holds, used by the independent check below."""
    if register == "fraction":
        num_str, den_str = frac_text.split("/")
        return Fraction(int(num_str), int(den_str))
    if register == "decimal":
        return Fraction(dec_text)
    return Fraction(pct_text.rstrip("%")) / 100


def _fraction_percent_question(key: str, register: str, quantity: int) -> dict:
    value, frac_text, dec_text, pct_text = _VALUE_TABLE[key]
    numerator, denominator = value.numerator, value.denominator
    check(quantity % denominator == 0, "quantity must be exactly divisible by the value's denominator")
    groups = quantity // denominator
    answer = numerator * groups

    shown_value = _shown_value_as_fraction(register, frac_text, dec_text, pct_text)
    check_equivalent(
        shown_value * quantity,
        answer,
        "the value re-parsed from the presented register's own text must match construction",
    )

    if register == "fraction":
        text = f"What is {frac_text} of {quantity}?"
    elif register == "decimal":
        text = f"What is {dec_text} of {quantity}?"
    else:
        text = f"What is {pct_text} of {quantity}?"

    candidates: list[int] = []
    if register == "percentage":
        percent_number = Fraction(pct_text.rstrip("%"))
        candidates.append(int(quantity * percent_number))  # percent-as-decimal-digits
    elif register == "decimal":
        candidates.append(answer * 10)  # decimal-place-shift
    else:
        candidates.append(groups)  # ignores-numerator

    if key in _EIGHTHS_KEYS:
        benchmark = _NEAREST_BENCHMARK[key]
        check(benchmark != value, "the wrong-benchmark value must differ from the true value")
        benchmark_answer = int(quantity * benchmark)
        check_distinct(answer, benchmark_answer, msg="wrong-benchmark distractor must differ from the answer")
        candidates.append(benchmark_answer)  # wrong-benchmark

    # Both are ALWAYS distinct from the answer by construction (numerator >= 1 every seed:
    # extra_group - answer == numerator, missing_group - answer == -numerator), which the named
    # misconceptions above are not (e.g. "ignores-numerator" collides with the answer whenever
    # numerator == 1) -- so these guarantee >=2 surviving distractors every seed, never left to the
    # named misconceptions' own collision rate.
    extra_group = numerator * (groups + 1)
    missing_group = numerator * (groups - 1)
    check_distinct(
        answer, extra_group, missing_group, msg="the universal fallback distractors must differ from the answer"
    )
    candidates.extend([extra_group, missing_group])

    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(c) for c in candidates], n=4)
    check(len(distractors) >= 2, "expected at least 2 surviving distractors")

    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[f"{frac_text} = {dec_text} = {pct_text} of {quantity} is {answer}."],
        distractors=distractors,
    )


@generator(
    "p9e_fraction_percent_quantity_strategy",
    level=6,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9e_fraction_percent_quantity_strategy(rng, tier: str = "easy"):
    """Find a familiar fraction/decimal/percentage of a whole-number
    quantity, the value's REGISTER drawn at random each seed (or, at hard,
    forced to the least-natural reading for that value). easy:
    quarters/halves/fifths, any register. medium: the eighths family, any
    register. hard: the full table, register forced to the least-natural
    reading."""
    if tier == "easy":
        key = rng.choice(_EASY_KEYS)
        register = rng.choice(_REGISTERS)
    elif tier == "medium":
        key = rng.choice(_EIGHTHS_KEYS)
        register = rng.choice(_REGISTERS)
    elif tier == "hard":
        key = rng.choice(_ALL_KEYS)
        register = _LEAST_NATURAL_REGISTER[key]
    else:
        raise ValueError(f"unknown tier: {tier!r}")

    denominator = _VALUE_TABLE[key][0].denominator
    quantity = _draw_quantity(rng, denominator, denominator * 2, denominator * 30)
    return _fraction_percent_question(key, register, quantity)


# =============================================================================
# Skill 2: p9e_terminating_recurring_decimal
# =============================================================================

_TRUNCATION_REPEATS = 20  # a "sufficient truncation" for the independent
# check's decimal-string route below.

# A terminating decimal's reduced denominator is ALWAYS of the form
# 2**a * 5**b (the necessary-and-sufficient condition), and it needs exactly
# n = max(a, b) decimal places to show without a trailing zero. Drawing the
# REDUCED fraction first from this pool -- rather than drawing an arbitrary
# n-digit numerator over a flat 10**n denominator -- is what makes "reduce to
# lowest terms" genuine work: the standard worked examples (0.125 -> 1/8,
# 0.0625 -> 1/16, 0.9375 -> 15/16) are exactly this shape, and a flat-10**n
# draw would almost never land on one.
_TERMINATING_DENOMINATORS: dict[int, tuple[int, ...]] = {
    1: (2, 5, 10),
    2: (4, 20, 25, 50, 100),
    3: (8, 40, 125, 200, 250, 500, 1000),
    4: (16, 80, 400, 625, 1250, 2000, 2500, 5000),
}


def _terminating_digits(rng, n: int) -> tuple[int, int, int]:
    """A reduced fraction numerator/D drawn from `_TERMINATING_DENOMINATORS[n]`
    (numerator coprime to D, so `Fraction(numerator, D)` is already in lowest
    terms), together with the n-digit integer `digits_int` such that
    `0.{digits_int:0n}` is its exact decimal expansion -- whose last digit is
    nonzero (so the decimal genuinely needs all n places -- never a display
    with a trailing zero a learner would just drop)."""
    from math import gcd

    for _ in range(_RESAMPLE_ATTEMPTS):
        D = rng.choice(_TERMINATING_DENOMINATORS[n])
        numerator = rng.randint(1, D - 1)
        if gcd(numerator, D) != 1:
            continue
        digits_int = numerator * (10**n // D)
        if digits_int % 10 == 0:
            continue
        return numerator, D, digits_int
    raise ValueError("p9e_terminating_recurring_decimal: no valid terminating digit string found")


def _universal_fallbacks(value: Fraction) -> tuple[Fraction, Fraction]:
    """Two universal fallback misconceptions, available for every branch
    below, ALWAYS a genuine proper fraction (never a bare integer that would
    make the true answer identifiable by shape alone -- the defect the
    earlier reciprocal/doubled design had whenever `value` was a unit
    fraction, e.g. 1/2's reciprocal is the integer 2 and its double is the
    integer 1) and ALWAYS distinct from `value` by construction:
    - **misplaces-decimal-point**: `value / 10`. Every value here lies
      strictly in (0, 1) (a decimal fractional part), so `value / 10` lies
      strictly in (0, 0.1) -- never an integer (no integer lies strictly
      between 0 and 1), never equal to `value` (which would need value = 0).
    - **wrong-denominator-by-one**: `p / (q + 1)` for the reduced value
      p/q. Since p < q < q + 1, this is always in (0, 1) -- never an
      integer, never equal to p/q (q + 1 != q).
    Both proven algebraically, not sampled -- see `check` below."""
    check(0 < value < 1, "every decimal value here must lie strictly between 0 and 1")
    p, q = value.numerator, value.denominator
    check(q >= 2, "a value strictly between 0 and 1 must have denominator >= 2")
    misplaces_decimal_point = value / 10
    wrong_denominator_by_one = Fraction(p, q + 1)
    check_distinct(
        value,
        misplaces_decimal_point,
        wrong_denominator_by_one,
        msg="the universal fallback distractors must differ from the answer",
    )
    return misplaces_decimal_point, wrong_denominator_by_one


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _fraction_result(shown: str, latex_notation: str, value: Fraction, candidates: list[Fraction]) -> dict:
    text = f"Write {shown} as a fraction in lowest terms."
    latex_text = rf"\text{{Write }} {latex_notation} \text{{ as a fraction in lowest terms.}}"
    answer_text = _fraction_text(value)
    structured = {"p": value.numerator, "q": value.denominator}
    candidate_texts = [_fraction_text(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidate_texts, n=4)
    check(len(distractors) >= 2, "expected at least 2 surviving distractors")
    return dict(
        text=text,
        latex=latex_text,
        answer=structured,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[f"{shown} = {answer_text} in lowest terms."],
        distractors=distractors,
    )


def _terminating_question(rng, n: int) -> dict:
    numerator, D, digits_int = _terminating_digits(rng, n)
    shown = f"0.{digits_int:0{n}d}"
    value = Fraction(numerator, D)  # already reduced -- lowest terms is genuine work, not a no-op

    # Independent check: parse the shown finite digit string directly (a
    # separate call through Fraction's own decimal-string parser), never
    # reusing the numerator/D construction used above.
    check_equivalent(Fraction(shown), value, "the shown terminating digit string must parse to the constructed value")

    misclassifies_as_recurring = Fraction(digits_int, 10**n - 1)  # misclassifies-terminating-as-recurring
    if misclassifies_as_recurring.denominator == 1:
        # Only possible when digits_int == 10**n - 1 (e.g. shown "0.9",
        # "0.99" -- an all-9s digit string), which makes this formula land
        # exactly on 1, a bare integer that would make the true (always
        # proper-fraction) answer identifiable by shape alone. Shift the
        # numerator by one: still a genuinely different value from the
        # (numerically distinct, always < 1) true answer, and -- since
        # 10**n - 1 and 10**n are always coprime -- never an integer itself.
        misclassifies_as_recurring = Fraction(digits_int + 1, 10**n - 1)
    misplaces_decimal_point, wrong_denominator_by_one = _universal_fallbacks(value)
    check_distinct(value, misclassifies_as_recurring, msg="misclassifies-as-recurring must differ from the answer")
    candidates = [misclassifies_as_recurring, misplaces_decimal_point, wrong_denominator_by_one]
    return _fraction_result(shown, shown, value, candidates)


def _recurring_value(L: int, p: int, R: int, q: int) -> Fraction:
    """The standard algebraic closed form for 0.[lead][repeat repeat...]:
    x = 0.LRRRR...; 10**(p+q)*x - 10**p*x = LR - L (as integers, LR the
    concatenation of lead and one repeat block); x = (LR - L) /
    (10**p * (10**q - 1))."""
    concat = L * 10**q + R
    return Fraction(concat - L, 10**p * (10**q - 1))


def _recurring_check_value(L: int, p: int, R: int, q: int) -> Fraction:
    """Independent check: parse a SUFFICIENT truncation of the actual
    decimal digit string (never the L/R closed-form construction above) via
    `Fraction`'s own exact decimal-string parsing, then add the
    geometric-series closed form for the infinite repeating tail beyond
    that truncation -- a genuinely different code path from
    `_recurring_value`'s integer arithmetic."""
    lead_str = f"{L:0{p}d}" if p else ""
    repeat_str = f"{R:0{q}d}"
    truncated_str = lead_str + repeat_str * _TRUNCATION_REPEATS
    truncated = Fraction(f"0.{truncated_str}")
    shift = 10 ** (p + q * _TRUNCATION_REPEATS)
    tail = Fraction(R, 10**q - 1) / shift
    return truncated + tail


def _recurring_shown(L: int, p: int, R: int, q: int) -> tuple[str, str]:
    """(plain-text, latex) display -- text uses an ellipsis after three full
    repeats of the block (matching the standard worked examples,
    "0.777...", "0.454545...", "0.41666..."); latex uses a proper overline
    over just the repeating block."""
    lead_str = f"{L:0{p}d}" if p else ""
    repeat_str = f"{R:0{q}d}"
    plain = f"0.{lead_str}{repeat_str * 3}..."
    latex_notation = rf"0.{lead_str}\overline{{{repeat_str}}}"
    return plain, latex_notation


def _recurring_question(L: int, p: int, R: int, q: int, extra_candidates: list[Fraction]) -> dict:
    value = _recurring_value(L, p, R, q)
    check_value = _recurring_check_value(L, p, R, q)
    check_equivalent(check_value, value, "the decimal-string-truncation + geometric-tail check must match construction")

    shown, latex_notation = _recurring_shown(L, p, R, q)
    misplaces_decimal_point, wrong_denominator_by_one = _universal_fallbacks(value)
    candidates = [*extra_candidates, misplaces_decimal_point, wrong_denominator_by_one]
    return _fraction_result(shown, latex_notation, value, candidates)


def _single_digit_recurring(rng) -> dict:
    """easy tier's recurring case: a single repeating digit (denominator a
    factor of 9), e.g. 0.777... -> 7/9."""
    R = rng.randint(1, 8)
    shown_digits = f"{R}" * 3
    treats_as_terminating = Fraction(int(shown_digits), 10 ** len(shown_digits))  # treats-recurring-as-terminating
    return _recurring_question(0, 0, R, 1, [treats_as_terminating])


def _two_digit_block_recurring(rng) -> dict:
    """medium tier's recurring case: a two-digit repeating BLOCK (denominator
    a factor of 99), e.g. 0.454545... -> 5/11 -- the un-owned slice against
    `recurring-decimal-to-fraction`'s single-digit-only scope."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        R = rng.randint(10, 98)  # excludes 99 (the all-9s block, which
        # collapses to a terminating-equivalent value)
        a, b = divmod(R, 10)
        if a != b:  # a genuine 2-digit block, not a disguised single-digit repeat
            break
    else:
        raise ValueError("p9e_terminating_recurring_decimal: no valid 2-digit block found")
    shown_digits = f"{R:02d}" * 3
    treats_as_terminating = Fraction(int(shown_digits), 10 ** len(shown_digits))  # treats-recurring-as-terminating
    extra_candidates = [treats_as_terminating]
    wrong_repeat_length = Fraction(R, 9)  # wrong-repeat-length: denominator 9 instead of 99
    if wrong_repeat_length.denominator != 1:  # R a multiple of 9 (e.g. 72/9 = 8) would be a bare
        extra_candidates.append(wrong_repeat_length)  # integer -- shape-identifiable, so skip it that seed
    return _recurring_question(0, 0, R, 2, extra_candidates)


def _mixed_recurring(rng) -> dict:
    """hard tier's recurring case: one or two non-repeating lead digits
    before a repeating block, e.g. 0.41666... -> 5/12, 0.1666... -> 1/6."""
    p = rng.choice((1, 2))
    q = rng.choice((1, 2))
    L = rng.randint(1, 9) if p == 1 else rng.randint(10, 99)
    R = rng.randint(1, 8) if q == 1 else rng.randint(10, 98)  # excludes the all-9s block, see above
    lead_str = f"{L:0{p}d}"
    repeat_str = f"{R:0{q}d}"
    shown_digits = lead_str + repeat_str * 3
    treats_as_terminating = Fraction(int(shown_digits), 10 ** len(shown_digits))  # treats-recurring-as-terminating
    ignores_lead = Fraction(R, 10**q - 1)  # ignores-non-repeating-lead
    return _recurring_question(L, p, R, q, [treats_as_terminating, ignores_lead])


@generator(
    "p9e_terminating_recurring_decimal",
    level=8,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9e_terminating_recurring_decimal(rng, tier: str = "easy"):
    """Classify a decimal (explicit notation discloses terminating vs
    recurring) and convert it to its equivalent fraction in lowest terms.
    easy: a single repeating digit OR a terminating decimal whose reduced
    denominator is a power of 2 and/or 5 needing 1-3 places (e.g. 0.125 ->
    1/8), drawn 50/50. medium: a two-digit repeating block OR a terminating
    decimal needing 3-4 places (e.g. 0.0625 -> 1/16), drawn 50/50. hard: a
    mixed (lead + repeating tail) decimal OR a terminating decimal needing
    exactly 4 places (e.g. 0.9375 -> 15/16), drawn 50/50 -- the terminating
    branch's digit-count range overlaps medium and hard at 4 places
    the tier's genuine escalation
    is carried entirely by the recurring branch (a two-digit block at medium
    vs a mixed lead-plus-tail decimal at hard). Every terminating draw is a
    genuinely reduced fraction from the outset (numerator coprime to its
    denominator), so "in lowest terms" is real reduction work, never a
    no-op."""
    if tier == "easy":
        return _single_digit_recurring(rng) if rng.random() < 0.5 else _terminating_question(rng, rng.choice((1, 2, 3)))
    if tier == "medium":
        return _two_digit_block_recurring(rng) if rng.random() < 0.5 else _terminating_question(rng, rng.choice((3, 4)))
    if tier == "hard":
        return _mixed_recurring(rng) if rng.random() < 0.5 else _terminating_question(rng, 4)
    raise ValueError(f"unknown tier: {tier!r}")
