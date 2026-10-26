"""Primary-number place-value generators (group B): two more generators for three genuinely new graded operations in
the ``primary-number`` domain, both framework ``acara-f10-v9``.

- ``division_remainder_context`` (AC9M5N07, prerequisites
  ``whole-number-division-facts``, ``f10-multiply-divide-one-digit-arrays``)
  -- solve a division problem and express or interpret the remainder as a
  whole number or a decimal, appropriate to the stated context. Reading a
  remainder in context resolves to a single deterministic value once the
  question's own wording states which reading is wanted. **Scoping decision, recorded as a
  deliberate exclusion, not an oversight**: a single skill declares one
  grader kind (``answer: number`` here), so the fraction-remainder reading is
  out of scope for this build. Three context variants, each producing a genuinely
  different computed quantity from the same ``total``/``divisor`` shape (not
  a bigger number on the same operation):

  - ``leftover`` (easy): "how many are left over" -- state the total and the
    group size, find the remainder itself (``total % divisor``) as a plain
    whole number. The remainder is always forced non-zero (a "how many are
    left over" question whose answer is exactly 0 is a degenerate, untested
    edge of the ``grade_number``/``check_answer`` zero-handling path, and
    reads as pedagogically pointless -- the exact-division case is already
    the existing ``multiply_divide_one_digit_arrays`` skill's own territory).
  - ``groups_needed`` (medium): "how many groups/containers are needed to
    fit everything, with none left uncontained" -- state the total and the
    group size, find ``ceil(total / divisor)``. This is a genuinely
    different operation from the raw remainder itself: it requires adding 1
    to the floor quotient specifically when a remainder exists, and *not*
    adding 1 when the division is exact. Both branches are left to natural
    sampling (never forced either way) and asserted directly, over many
    seeds, to actually occur.
  - ``decimal_share`` (hard): "share the total equally, expressed as a
    decimal amount each" -- state the total and the number of shares, find
    ``total / num_shares`` as a decimal. The divisor is drawn only from
    ``{2, 4, 5, 8, 20, 25}`` -- every one of these is a pure product of 2s
    and 5s, so the division always terminates in a bounded number of decimal
    places (never a repeating decimal with no stated rounding rule), and the
    remainder is forced non-zero so the answer always has a genuine
    fractional part (an "express as a decimal" reading, not
    a whole number that merely happens to be representable as one). ``10``
    is deliberately excluded from this set even though it also terminates
    cleanly: with divisor 10, the ``remainder-as-decimal-digits``
    misconception (below) is *algebraically identical* to the correct
    answer (``quotient + remainder/10`` is exactly what writing the digits
    "quotient.remainder" produces when the remainder is a single digit),
    which would silently delete that distractor on every single seed
    (``pick_distractors`` filters candidates equal to the answer) rather
    than merely make it occasionally exercisable. Every distractor and
    canonical answer is resampled at construction time (not just asserted in
    a test) until it clears ``grade_number``'s own ~0.05 numeric tolerance
    (``mathgen/grading/spoken.py`` ``_TOLERANCE = 0.051``) -- the same
    necessary finding for any decimal-valued, wide-magnitude-range
    ``answer: number`` skill (a test-side assertion alone is not sufficient
    to guarantee this property; a naive construction can produce a
    distractor within a few hundredths of the true answer, which the real
    grader would then silently accept as correct).

  Independent check discipline, one distinct route per variant, none of
  which re-runs the operator used to construct the answer: ``leftover``
  verifies via genuine repeated subtraction (mirrors the manual long-division
  algorithm, distinct from Python's ``%`` operator); ``groups_needed``
  verifies via a "just enough, and not one more than needed" cover-and-
  minimality check (``answer * divisor >= total`` and
  ``(answer - 1) * divisor < total``), distinct from the floor+1 branch logic
  used to construct it; ``decimal_share`` verifies by multiplying the found
  decimal answer back against the number of shares and asserting it exactly
  reconstructs the integer total (via ``decimal.Decimal``, exact for a
  terminating decimal), distinct from the ``Fraction``-based reduction used
  to construct the decimal string.

- ``integer_numberline_order`` (AC9M6N01 and AC9M7N07 -- one skill, two
  target areas -- prerequisites ``integer-arithmetic``,
  ``coordinate-plane-readiness``) -- order a supplied set of distinct
  integers from smallest to largest. Genuine difficulty axis is **sign mix**, not
  span:

  - ``easy`` (``sign_mix="all_positive"``): all-positive integers -- a
    warm-up restating the existing ``order_to_1000``-style reasoning with
    negative numbers simply absent; this tier alone is not new content, the
    genuine new reasoning is in the medium/hard tiers below.
  - ``medium`` (``sign_mix="mixed_with_zero"``): a mix of positive integers
    and zero, with at least one negative integer forced to appear (never
    left to chance).
  - ``hard`` (``sign_mix="mixed_trap"``): a mix of positive and negative
    integers, deliberately forcing the specific trap pair ``-7`` and ``3``
    to appear on *every* seed -- the real, documented misconception "-7 > -3
    because 7 > 3" / "treats the sign as irrelevant and orders by magnitude
    alone" (named here as ``magnitude-ignores-sign``).

  Independent check: same all-pairs discipline as ``order_to_1000``'s own
  ``_verify_order_to_1000``, adapted for integers -- every
  candidate value is shifted by a large positive constant (so every shifted
  value is positive and the shift preserves true order exactly, since
  ``a < b`` iff ``a + K < b + K`` for any constant ``K``), then compared by
  shifted-digit-count and shifted-digit-string, exactly mirroring
  ``order_to_1000``'s own route. This is a genuinely different route from
  the ``sorted()`` call used to construct the ordered list, and is
  deliberately *not* an absolute-value comparison -- the shift constant is
  added to the true signed integer, never to its magnitude, so it cannot
  accidentally reproduce the very "magnitude-ignores-sign" misconception the
  hard tier is designed to trap.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

from ..core import check, generator, pick_distractors, reversed_and_adjacent_swap

# -- division_remainder_context ----------------------------------------------

_VARIANTS = ("leftover", "groups_needed", "decimal_share")

# Deliberately excludes 10 -- see the module docstring's decimal_share
# section for why (the remainder-as-decimal-digits misconception would be
# algebraically identical to the correct answer whenever the divisor is 10).
_DECIMAL_SHARE_DIVISORS = (2, 4, 5, 8, 20, 25)

_SHARE_ITEMS = (
    ("litres of juice", "bottles"),
    ("kilograms of flour", "bags"),
    ("litres of paint", "tins"),
    ("kilograms of rice", "bags"),
)


def _verify_remainder_by_subtraction(total: int, divisor: int, remainder: int) -> int:
    """Independent check for the ``leftover`` variant: construction computes
    the remainder via Python's ``%`` operator. This check never reuses that
    operator -- it repeatedly subtracts ``divisor`` from ``total`` (the
    manual long-division algorithm a learner would actually perform) until
    what remains is less than ``divisor``, and asserts that final remainder
    matches. Returns the number of subtractions performed (the quotient), so
    callers can cross-check it too."""
    remaining = total
    count = 0
    while remaining >= divisor:
        remaining -= divisor
        count += 1
    check(
        remaining == remainder,
        f"repeated subtraction of {divisor} from {total} leaves {remaining}, expected remainder {remainder}",
    )
    return count


def _verify_groups_needed(total: int, divisor: int, answer: int) -> None:
    """Independent check for the ``groups_needed`` variant: construction
    computes the answer via a floor-quotient-plus-conditional-1 branch. This
    check never re-runs that branch logic -- it asserts the two properties
    that together *define* "the minimum number of groups needed to fit
    everything": the answer's groups are enough to cover the total, and one
    fewer group would not have been enough."""
    check(answer * divisor >= total, f"{answer} groups of {divisor} do not cover total {total}")
    check(
        (answer - 1) * divisor < total,
        f"{answer - 1} groups of {divisor} already cover total {total}; {answer} is not the minimum needed",
    )


def _decimal_places_needed(denominator: int) -> int:
    """How many decimal places a fraction with this (already-reduced)
    denominator needs to terminate exactly -- only valid when the
    denominator's prime factors are exclusively 2 and/or 5."""
    n = denominator
    twos = 0
    while n % 2 == 0:
        n //= 2
        twos += 1
    fives = 0
    while n % 5 == 0:
        n //= 5
        fives += 1
    if n != 1:
        raise ValueError(f"denominator {denominator} has prime factors other than 2 and 5; not a terminating decimal")
    return max(twos, fives)


def _exact_decimal_string(total: int, divisor: int) -> str:
    """Format ``total / divisor`` as its exact, minimal terminating decimal
    string (no trailing zeros beyond what is needed), using exact integer
    arithmetic throughout -- never a float division."""
    frac = Fraction(total, divisor)
    places = _decimal_places_needed(frac.denominator)
    if places == 0:
        return str(frac.numerator)
    scaled_numerator = frac.numerator * (10**places)
    check(
        scaled_numerator % frac.denominator == 0, "scaled numerator does not divide exactly by the reduced denominator"
    )
    scaled = scaled_numerator // frac.denominator
    int_part = scaled // (10**places)
    frac_part = scaled % (10**places)
    return f"{int_part}.{str(frac_part).zfill(places)}"


def _verify_decimal_share(total: int, num_shares: int, answer_str: str) -> None:
    """Independent check for the ``decimal_share`` variant: construction
    reduces a ``Fraction`` and formats its exact terminating decimal digits.
    This check never touches that reduction -- it multiplies the found
    decimal answer back against the number of shares (via ``decimal.Decimal``,
    exact for a terminating decimal) and asserts it exactly reconstructs the
    integer total, the reverse arithmetic direction from construction."""
    reconstructed = Decimal(answer_str) * Decimal(num_shares)
    check(reconstructed == Decimal(total), f"{answer_str} × {num_shares} = {reconstructed}, expected {total}")


def _leftover(rng, total_range=(20, 99), divisor_range=(2, 9)):
    lo_t, hi_t = total_range
    lo_d, hi_d = divisor_range
    max_attempts = 2000
    for _ in range(max_attempts):
        total = rng.randint(lo_t, hi_t)
        divisor = rng.randint(lo_d, hi_d)
        remainder = total % divisor
        if remainder != 0:  # forced non-zero -- see module docstring
            break
    else:
        raise RuntimeError(
            f"could not sample a leftover question with a non-zero remainder in {max_attempts} attempts "
            f"(total_range={total_range}, divisor_range={divisor_range})"
        )

    quotient = _verify_remainder_by_subtraction(total, divisor, remainder)
    check(quotient * divisor + remainder == total, f"{quotient} × {divisor} + {remainder} does not reconstruct {total}")
    answer = remainder

    # Misconceptions:
    #  - reports-quotient: reports the number of full groups instead of what
    #    is left over.
    #  - off-by-one-remainder: one more than the true remainder.
    #  - reports-divisor: reports the group size itself, confusing "how many
    #    left over" with "how many in each group".
    candidates = [str(quotient), str(remainder + 1), str(divisor)]

    text = f"{total} counters are shared into groups of {divisor}, with any leftover counters set aside. How many counters are left over?"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"{quotient} full groups of {divisor} use {quotient * divisor} counters.",
            f"{total} - {quotient * divisor} = {remainder} left over.",
        ],
        distractors=pick_distractors(str(answer), candidates),
    )


def _groups_needed(rng, total_range=(20, 99), divisor_range=(2, 9)):
    lo_t, hi_t = total_range
    lo_d, hi_d = divisor_range
    total = rng.randint(lo_t, hi_t)
    divisor = rng.randint(lo_d, hi_d)
    quotient, remainder = divmod(total, divisor)
    answer = quotient + 1 if remainder else quotient
    _verify_groups_needed(total, divisor, answer)

    # Misconceptions:
    #  - forgets-to-round-up: always reports the floor quotient, even when a
    #    remainder means one more group is genuinely needed.
    #  - always-rounds-up: always adds 1, even when the division is exact
    #    and no extra group is needed.
    #  - reports-remainder-instead (only when there is a genuine remainder to
    #    report -- a "0" distractor when the division is exact is
    #    pedagogically nonsensical and not a real error a learner would make).
    candidates = [str(quotient), str(quotient + 1)]
    if remainder:
        candidates.append(str(remainder))

    if remainder:
        working = [
            f"{total} ÷ {divisor} = {quotient} remainder {remainder}.",
            f"{quotient} groups are not enough ({quotient * divisor} < {total}), so one more group is needed: {answer}.",
        ]
    else:
        working = [
            f"{total} ÷ {divisor} = {quotient} exactly, with nothing left over.",
            f"{quotient} groups exactly fit everything, so no extra group is needed: {answer}.",
        ]

    text = f"{total} eggs need to be packed into cartons that each hold {divisor} eggs. How many cartons are needed to fit all the eggs, with no eggs left out?"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(str(answer), candidates),
    )


def _decimal_share(rng, total_range=(30, 199), divisors=_DECIMAL_SHARE_DIVISORS):
    max_attempts = 2000
    lo_t, hi_t = total_range
    for _ in range(max_attempts):
        num_shares = rng.choice(divisors)
        total = rng.randint(lo_t, hi_t)
        quotient, remainder = divmod(total, num_shares)
        if remainder == 0:
            continue  # forced non-zero -- see module docstring

        answer_str = _exact_decimal_string(total, num_shares)
        answer_val = float(answer_str)

        # Misconceptions:
        #  - reports-quotient-only: drops the fractional/decimal part
        #    entirely, reporting only the floor quotient.
        #  - remainder-as-decimal-digits: treats the whole-number remainder
        #    as if its digits *were* the decimal part, without actually
        #    dividing it by the number of shares (e.g. "quotient=2,
        #    remainder=6" written as "2.6" instead of the true 2 + 6/8).
        # Each distractor is shown to the answer's own number of decimal
        # places (zero-padded), so the key cannot be spotted by its shape.
        places = len(answer_str.partition(".")[2])
        quotient_only_str = f"{quotient}." + "0" * places
        remainder_digits_str = f"{quotient}." + str(remainder).ljust(places, "0")
        rounded_up_str = f"{quotient + 1}." + "0" * places

        candidate_strs = [quotient_only_str, rounded_up_str]
        if len(str(remainder)) <= places:  # else it could not match the answer's shape
            candidate_strs.insert(1, remainder_digits_str)
        # grade_number's ~0.05 numeric tolerance (mathgen/grading/spoken.py
        # _TOLERANCE = 0.051) means a decimal distractor within that gap of
        # the correct answer would be silently accepted as correct by the
        # real grader -- checked directly at construction time, not just
        # assumed or asserted only in a test.
        candidate_strs = [c for c in candidate_strs if c == answer_str or abs(answer_val - float(c)) > 0.051]

        distractors = pick_distractors(answer_str, candidate_strs)
        if not distractors:
            continue
        break
    else:
        raise RuntimeError(
            f"could not sample a decimal_share question with every distractor > 0.051 from the answer "
            f"in {max_attempts} attempts (total_range={total_range}, divisors={divisors})"
        )

    _verify_decimal_share(total, num_shares, answer_str)

    noun, receptacle = _SHARE_ITEMS[num_shares % len(_SHARE_ITEMS)]
    text = f"{total} {noun} are shared equally among {num_shares} {receptacle}. How much is in each {receptacle[:-1]}, as a decimal?"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_val,
        answer_text=answer_str,
        answer_latex=answer_str,
        working=[
            f"{total} ÷ {num_shares} = {answer_str} exactly (a terminating decimal).",
        ],
        distractors=distractors,
    )


_DIVISION_VARIANT_FNS = {
    "leftover": _leftover,
    "groups_needed": _groups_needed,
    "decimal_share": _decimal_share,
}


@generator(
    "division_remainder_context",
    level=5,
    presets={
        "easy": dict(variant="leftover"),
        "medium": dict(variant="groups_needed"),
        "hard": dict(variant="decimal_share"),
    },
)
def division_remainder_context(rng, variant="leftover", **kwargs):
    """Solve a division problem and express or interpret the remainder as a
    whole number or decimal, appropriate to the stated context. See the
    module docstring for the three-variant difficulty rationale and the
    deliberate fraction-remainder exclusion."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {_VARIANTS}")
    return _DIVISION_VARIANT_FNS[variant](rng, **kwargs)


# -- integer_numberline_order -------------------------------------------

_NEG_POOL = tuple(range(-20, 0))
_POS_POOL = tuple(range(1, 21))
_TRAP_PAIR = (-7, 3)

# Large enough that every shifted value in this generator's population
# (-20..20) is positive, with plenty of headroom.
_ORDER_SHIFT = 1000


def _verify_integer_order(sampled: list[int], ordered: list[int]) -> None:
    """Independent check, same all-pairs discipline as ``order_to_1000``'s
    own ``_verify_order_to_1000``: construction sorts via
    ``sorted(numbers)``; this check never re-runs that sort. Every value is
    first shifted by a large positive constant (``a < b`` iff
    ``a + K < b + K`` for any constant K, so this preserves true signed
    order exactly), then every pair (i, j) with i < j -- not just adjacent
    pairs -- is compared by shifted-digit-count first, then shifted-digit
    string for equal-length values. This is deliberately *not* an
    absolute-value comparison: the shift is applied to the true signed
    integer, never to its magnitude, so it cannot accidentally reproduce the
    magnitude-ignores-sign misconception this check exists to catch."""
    check(sorted(sampled) == sorted(ordered), "ordered list is not a permutation of the sampled integers")
    check(len(ordered) == len(set(ordered)), "sampled integers must be distinct")
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            shifted_a, shifted_b = a + _ORDER_SHIFT, b + _ORDER_SHIFT
            check(shifted_a > 0 and shifted_b > 0, f"shift constant {_ORDER_SHIFT} too small for values {a}, {b}")
            text_a, text_b = str(shifted_a), str(shifted_b)
            if len(text_a) != len(text_b):
                check(len(text_a) < len(text_b), f"{a} does not precede {b} (shifted-digit-count comparison)")
            else:
                check(text_a < text_b, f"{a} does not precede {b} (shifted-digit-string comparison)")


def _sample_all_positive(rng, count: int) -> list[int]:
    if count > len(_POS_POOL):
        raise ValueError(f"cannot sample {count} distinct positive integers from a pool of only {len(_POS_POOL)}")
    return list(rng.sample(_POS_POOL, count))


def _sample_mixed_with_zero(rng, count: int) -> list[int]:
    # At least one negative integer, plus zero, plus the rest positive --
    # forced every seed, never left to chance.
    if count < 3:
        raise ValueError("mixed_with_zero needs at least 3 items (a negative, zero, and at least one positive)")
    negative = rng.sample(_NEG_POOL, 1)
    remaining = count - 2  # minus the forced negative and the forced zero
    positives = rng.sample(_POS_POOL, remaining)
    numbers = negative + positives + [0]
    rng.shuffle(numbers)
    return numbers


def _sample_mixed_trap(rng, count: int) -> list[int]:
    # The trap pair (-7, 3) is forced to appear on every seed -- never left
    # to chance.
    if count < 2:
        raise ValueError("mixed_trap needs at least 2 items to include the forced trap pair")
    remaining_count = count - 2
    pool = [n for n in (_NEG_POOL + _POS_POOL) if n not in _TRAP_PAIR]
    rest = rng.sample(pool, remaining_count)
    numbers = list(_TRAP_PAIR) + rest
    rng.shuffle(numbers)
    return numbers


_SIGN_MIX_SAMPLERS = {
    "all_positive": _sample_all_positive,
    "mixed_with_zero": _sample_mixed_with_zero,
    "mixed_trap": _sample_mixed_trap,
}


@generator(
    "integer_numberline_order",
    level=6,
    presets={
        "easy": dict(count=4, sign_mix="all_positive"),
        "medium": dict(count=4, sign_mix="mixed_with_zero"),
        "hard": dict(count=5, sign_mix="mixed_trap"),
    },
)
def integer_numberline_order(rng, count=4, sign_mix="all_positive"):
    """Order a supplied set of distinct integers from smallest to largest.
    See the module docstring for the sign-mix difficulty rationale."""
    if count < 2:
        raise ValueError("count must be at least 2")
    if sign_mix not in _SIGN_MIX_SAMPLERS:
        raise ValueError(f"unknown sign_mix {sign_mix!r}; expected one of {sorted(_SIGN_MIX_SAMPLERS)}")

    numbers = _SIGN_MIX_SAMPLERS[sign_mix](rng, count)
    check(len(numbers) == len(set(numbers)), "sampled integers must be distinct")
    ordered = sorted(numbers)
    _verify_integer_order(numbers, ordered)

    display_order = list(numbers)
    rng.shuffle(display_order)
    display_text = ", ".join(str(n) for n in display_order)
    answer_text = ", ".join(str(n) for n in ordered)

    # Misconceptions:
    #  - reversed-order: largest to smallest instead of smallest to largest.
    #  - adjacent-swap: correctly ordered overall, but the two middle values
    #    are swapped, as if two neighbouring values did not need comparing.
    #  - magnitude-ignores-sign (not all_positive only, where it would always
    #    coincide with the true order and never be a usable distractor):
    #    orders as if comparing absolute values, so e.g. -7 is placed after
    #    3 (the real, documented "-7 > -3 because 7 > 3" misconception).
    reversed_order, adjacent_swap = reversed_and_adjacent_swap(ordered)
    candidates = [
        ", ".join(str(n) for n in reversed_order),
        ", ".join(str(n) for n in adjacent_swap),
    ]
    if sign_mix != "all_positive":
        magnitude_ignores_sign = sorted(ordered, key=abs)
        candidates.append(", ".join(str(n) for n in magnitude_ignores_sign))

    return dict(
        text=f"Order these integers from smallest to largest: {display_text}.",
        latex=rf"\text{{Order: }} {display_text}",
        answer=list(ordered),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            "Compare the integers on the number line: further left is always smaller, no matter the sign.",
            f"In order: {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )
