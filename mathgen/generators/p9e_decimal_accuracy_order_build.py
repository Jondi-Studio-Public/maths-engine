"""Decimal accuracy + decimal ordering, both ``acara-f10-v9``, domain ``primary-number``.

Two topics:

- ``p9e_round_decimal_accuracy`` (``f10-round-decimals-accuracy``, AC9M7N05,
  ``answer: number``) -- round a decimal value to the number of decimal
  places a described real-world CONTEXT implies (never a STATED place --
  that is ``round-decimals``' own territory). Every question states its
  context (money, a fine length measurement, a headcount average) and also
  states the recognised rounding phrase (``mathgen.text_quality
  .requested_rounding``'s own six words) so the question stays gradable.
- ``p9e_decimal_order_numberline`` (``f10-decimal-order-numberline``,
  AC9M5N01, ``answer: ordered_list``) -- order 4 distinct decimals with 3-4
  decimal places, ascending, including values >= 1.

Every rounded value is held and compared as an exact integer count of its own smallest displayed unit (never a
binary float, never ``Decimal`` -- plain integers round-half-up cleanly with ``(n + scale // 2) // scale``,
the same idiom ``primary_math.round_decimal`` already uses), and every list-ordering value is held as an exact
``Fraction`` parsed from its own decimal string. Distractor safety margins use ``stated_precision_tolerance``
(never the loose spoken-transcript ``GRADER_TOLERANCE``).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, latex_answer, pick_distractors, stated_precision_tolerance

# ---------------------------------------------------------------------------
# p9e_round_decimal_accuracy
# ---------------------------------------------------------------------------

_RESAMPLE_ATTEMPTS = 400


def _round_half_up(units: int, scale: int) -> int:
    """Round a non-negative integer ``units`` (a value held in some small
    unit) to the nearest multiple of ``scale``, half up -- ``units=2995,
    scale=10`` (2.995 held in thousandths, rounding to hundredths) gives
    300 (hundredths), i.e. 3.00, never 2.99."""
    return (units + scale // 2) // scale


def _format_units(units: int, dp: int) -> str:
    """A non-negative integer count of ``10**-dp`` units as a plain decimal
    string with exactly ``dp`` decimal places -- ``dp=0`` yields a bare
    integer with no decimal point at all (so the real grader's exact-match
    rule at zero stated places applies), matching
    ``primary_math.round_decimal``'s own ``at_target`` formatter."""
    if dp == 0:
        return str(units)
    scale = 10**dp
    whole, part = divmod(units, scale)
    return f"{whole}.{part:0{dp}d}"


# Each context fixes exactly one context -> decimal-place mapping (never left to chance): money
# implies the nearest hundredth of a dollar, a fine length measurement implies the nearest tenth of
# a centimetre, and an average headcount implies the nearest whole number.
#
# `raw_dp` decimal digits are drawn for the unrounded value: `target_dp` KEPT digits, then exactly
# one DROPPED digit (tier-controlled: never a bare 5 at easy/medium, forced to exactly 5 at hard).
# money/length stop there (raw_dp = target_dp + 1). headcount also draws one further, unconstrained
# EXTRA digit (raw_dp = target_dp + 2) -- with only the dropped digit, a "wrong accuracy" distractor
# rounded to `alt_dp` (finer than target_dp) would be literally the unrounded value shown in the
# question, which is a structurally-invalid distractor (within the grader's own tolerance of the
# correct rounding -- that is what rounding means).
_CONTEXTS = {
    # whole-part ranges are chosen so the rendered figure is a plausible
    # real-world quantity, not just a number the digit-construction needed:
    # a $2-60 per-person share of a group order, a beetle length up to 12 cm
    # (covers even record-sized species with their horns), a daily absence
    # headcount of 8-88 at a school with several hundred students enrolled.
    "money": dict(target_dp=2, raw_dp=3, alt_dp=1, whole_range=(2, 60)),
    "length": dict(target_dp=1, raw_dp=2, alt_dp=0, whole_range=(1, 12)),
    "headcount": dict(target_dp=0, raw_dp=2, alt_dp=1, whole_range=(8, 88)),
}


def _template(context: str, raw_text: str) -> tuple[str, str]:
    """(text, latex) for one context; the recognised rounding phrase (one of
    the six words `requested_rounding` accepts) appears verbatim in both."""
    if context == "money":
        body = (
            f"Splitting the cost of a shared takeaway order evenly works out to "
            f"${raw_text} per person. Round this amount to the nearest hundredth "
            f"of a dollar."
        )
    elif context == "length":
        body = (
            f"A digital calliper reads the length of a beetle as {raw_text} cm. "
            f"Round this length to the nearest tenth of a centimetre."
        )
    else:
        body = (
            f"Across several school terms, the average number of students absent "
            f"per day worked out to {raw_text}. Round this average to the nearest "
            f"whole number."
        )
    latex = r"\text{" + body.replace("$", r"\$") + "}"
    return body, latex


def _verify_round_half_up(raw_text: str, target_dp: int, correct_text: str) -> None:
    """Independent check: derives the rounded value from the *rendered*
    ``raw_text`` and ``target_dp`` via exact ``Fraction`` arithmetic -- a
    route completely different from the integer-digit construction above
    (which built the answer directly from ``whole``/``kept``/``dropped``
    without ever re-parsing a string): ``Fraction(round(Fraction(value) *
    10**dp), 10**dp)``, as the independent check."""
    exact = Fraction(raw_text)
    scale = 10**target_dp
    scaled = exact * scale
    expected_units = math.floor(scaled + Fraction(1, 2))  # half-up, positives only
    expected = Fraction(expected_units, scale)
    check(Fraction(correct_text) == expected, "independent Fraction recompute disagrees with the constructed rounding")


def _build_round_decimal_accuracy(rng, context: str, *, force_five: bool, force_carry: bool):
    cfg = _CONTEXTS[context]
    target_dp, raw_dp, alt_dp = cfg["target_dp"], cfg["raw_dp"], cfg["alt_dp"]
    lo, hi = cfg["whole_range"]
    drop_scale = 10 ** (raw_dp - target_dp)  # collapses every dropped/extra digit
    has_extra = raw_dp - target_dp - 1 > 0  # only headcount: one free extra digit

    for _ in range(_RESAMPLE_ATTEMPTS):
        if force_carry and target_dp == 0:
            # headcount has no kept decimal digits at all (target_dp=0), so
            # a carry can only come from the WHOLE part's own last digit
            # being 9 (e.g. 49 -> 50); drawn directly from the values ending
            # in 9 WITHIN the context's own whole-part range (never bumped
            # past `hi`, which would ship an implausible out-of-range
            # figure).
            nines = [v for v in range(lo, hi + 1) if v % 10 == 9] or [hi]
            whole = rng.choice(nines)
            kept = 0
        elif force_carry:
            # money/length: forcing every KEPT digit to 9 (99 cents, or the
            # single tenth-digit 9) is what cascades the carry -- e.g. money
            # "X.995" -> "(X+1).00" -- regardless of the whole part's own
            # last digit, so `whole` stays an ordinary draw from the range.
            whole = rng.randint(lo, hi)
            kept = 10**target_dp - 1
        else:
            whole = rng.randint(lo, hi)
            kept = rng.randint(0, 10**target_dp - 1)
        dropped = 5 if force_five else rng.choice([d for d in range(1, 10) if d != 5])
        extra = None
        if has_extra:
            extra = 0 if force_five else rng.randint(1, 9)

        raw_units = whole * 10**raw_dp + kept * drop_scale + dropped * (drop_scale // 10)
        if extra is not None:
            raw_units += extra
        correct_units = _round_half_up(raw_units, drop_scale)
        correct_whole = correct_units // 10**target_dp

        # easy's own no-carry constraint (only ever exercised by the
        # money-only easy tier: medium never uses money and hard always
        # forces the dropped digit to 5).
        if context == "money" and not force_five and not force_carry and correct_whole != whole:
            continue

        check(dropped != 0, "the dropped digit must be non-zero so truncation is provably a different value")

        raw_text = _format_units(raw_units, raw_dp)
        correct_text = _format_units(correct_units, target_dp)
        correct_value = Fraction(correct_units, 10**target_dp)

        # -- distractors, checked unconditionally, every candidate ----------
        truncated_text = _format_units(raw_units // drop_scale, target_dp)
        over_text = _format_units(correct_units + 1, target_dp)
        filler_text = _format_units(correct_units + 2, target_dp)

        candidates = []
        # A finer-place "wrong accuracy" rounding would show a decimal point
        # beside a whole-number key (headcount), giving the key away by shape;
        # the coarser-place ones are shown padded to the key's own places.
        wrong_accuracy_available = alt_dp <= target_dp
        if wrong_accuracy_available:
            alt_power = raw_dp - alt_dp
            alt_units = _round_half_up(raw_units, 10**alt_power) if alt_power >= 0 else raw_units * 10 ** (-alt_power)
            alt_value = Fraction(alt_units, 10**alt_dp)
            # Never "forgot to round": the alt-place rounding must differ
            # from both the correct rounding and the raw (unrounded) value
            # by more than the real grader's own stated-precision tolerance
            # -- an unrounded/over-precise value within that tolerance is
            # ACCEPTED by grade_number, not rejected.
            if target_dp == 0:
                distinct_from_correct = alt_value != correct_value
            else:
                distinct_from_correct = abs(alt_value - correct_value) > stated_precision_tolerance(target_dp)
            distinct_from_raw = alt_value != Fraction(raw_units, 10**raw_dp)
            if distinct_from_correct and distinct_from_raw:
                candidates.append(_format_units(alt_units * 10 ** (target_dp - alt_dp), target_dp))

        candidates.extend([truncated_text, over_text, filler_text])

        distractors = pick_distractors(
            correct_text,
            candidates,
            n=3,
            tolerance=(Fraction(0) if target_dp == 0 else stated_precision_tolerance(target_dp)),
        )
        if len(distractors) < 2:
            continue  # a rare collision cascade; resample

        _verify_round_half_up(raw_text, target_dp, correct_text)
        text, latex = _template(context, raw_text)
        place_phrase = {
            "money": "the nearest hundredth of a dollar",
            "length": "the nearest tenth of a centimetre",
            "headcount": "the nearest whole number",
        }[context]
        working = [
            f"This context implies rounding to {place_phrase}.",
            f"Look at the digit after the rounding place and apply the half-up rule: "
            f"{raw_text} rounds to {correct_text}.",
        ]
        return dict(
            text=text,
            latex=latex,
            answer=float(correct_value),
            answer_text=correct_text,
            answer_latex=latex_answer(correct_text),
            working=working,
            distractors=distractors,
        )

    raise RuntimeError(
        f"could not build a valid round_decimal_accuracy question for context={context!r} "
        f"force_five={force_five} force_carry={force_carry} in {_RESAMPLE_ATTEMPTS} attempts"
    )


@generator(
    "p9e_round_decimal_accuracy",
    level=6,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9e_round_decimal_accuracy(rng, tier="easy"):
    """Round a decimal to the accuracy its stated real-world context implies.

    easy: money context only, dropped digit never 5, no carry into the whole
    part. medium: length or headcount context, dropped digit never 5. hard:
    any context, dropped digit forced to exactly 5 (a genuine rounding-
    boundary reasoning step), with some seeds additionally forcing a 9-carry
    into the next place.
    """
    if tier == "easy":
        context = "money"
        force_five = False
        force_carry = False
    elif tier == "medium":
        context = rng.choice(["length", "headcount"])
        force_five = False
        force_carry = False
    else:
        context = rng.choice(["money", "length", "headcount"])
        force_five = True
        force_carry = rng.random() < 0.4
    return _build_round_decimal_accuracy(rng, context, force_five=force_five, force_carry=force_carry)


# ---------------------------------------------------------------------------
# p9e_decimal_order_numberline
# ---------------------------------------------------------------------------

_ORDER_RESAMPLE_ATTEMPTS = 400


def _decimal_text(whole: int, frac: str) -> str:
    return f"{whole}.{frac}"


def _sample_decimal_set(rng, tier: str):
    """4 distinct decimal values (as ``(whole, frac_digits)`` pairs), each
    with 3 or 4 decimal places, per the tier's own shape:

    easy: all four in [0, 1). medium: mixes values < 1 and >= 1. hard: all
    four share a 1-2 digit leading fractional prefix AND spans < 1 / >= 1.
    """
    for _ in range(_ORDER_RESAMPLE_ATTEMPTS):
        if tier == "easy":
            wholes = [0, 0, 0, 0]
        else:
            wholes = [0, 0, rng.randint(1, 9), rng.randint(1, 9)]
            rng.shuffle(wholes)

        shared_prefix = ""
        if tier == "hard":
            prefix_len = rng.choice([1, 2])
            shared_prefix = "".join(str(rng.randint(0, 9)) for _ in range(prefix_len))

        items = []
        for whole in wholes:
            dp = rng.choice([3, 4])
            if tier == "hard":
                dp = max(dp, len(shared_prefix) + 1)
                tail = "".join(str(rng.randint(0, 9)) for _ in range(dp - len(shared_prefix)))
                frac = shared_prefix + tail
            else:
                frac = "".join(str(rng.randint(0, 9)) for _ in range(dp))
            items.append((whole, frac))

        values = [Fraction(f"{w}.{f}") for w, f in items]
        if len(set(values)) != 4:
            continue  # a value collision (possibly via a trailing-zero display)
        if tier == "hard" and len(set(v >= 1 for v in values)) != 2:
            continue  # must genuinely span < 1 and >= 1
        return items, values

    raise RuntimeError(f"could not sample a valid decimal set for tier={tier!r} in {_ORDER_RESAMPLE_ATTEMPTS} attempts")


def _verify_ascending_permutation(values: list[Fraction], ordered: list[Fraction]) -> None:
    """Independent check: never reuses `sorted()` for the actual assertion --
    walks every pair (i, j) with i < j and asserts strictly ascending
    `Fraction` order directly, and confirms `ordered` is a permutation of the
    sampled multiset (compared as an exact `Fraction` multiset)."""
    check(sorted(values) == sorted(ordered), "ordered list is not a permutation of the sampled values")
    check(len(ordered) == len(set(ordered)), "sampled values must be distinct")
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            check(ordered[i] < ordered[j], f"{ordered[i]} does not precede {ordered[j]}")


@generator(
    "p9e_decimal_order_numberline",
    level=3,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9e_decimal_order_numberline(rng, tier="easy"):
    """Order 4 distinct decimals (3-4 d.p. each), ascending.

    easy: all four values in [0, 1). medium: mixes values < 1 and >= 1.
    hard: all four share a 1-2 digit leading fractional prefix AND spans
    < 1 / >= 1, forcing digit-by-digit comparison past the shared prefix.
    """
    for _ in range(_ORDER_RESAMPLE_ATTEMPTS):
        items, values = _sample_decimal_set(rng, tier)

        order_index = sorted(range(4), key=lambda i: values[i])
        ordered_items = [items[i] for i in order_index]
        ordered_values = [values[i] for i in order_index]
        _verify_ascending_permutation(values, ordered_values)

        # Distractor orderings: full re-orderings of the same 4 values.
        reversed_items = list(reversed(ordered_items))

        # longer-is-bigger: reads each item's own fractional digit string
        # directly as an unpadded integer ("0.45" -> 45, "0.5" -> 5) and
        # sorts by THAT, ignoring place value entirely -- this coincides
        # with the true order whenever every fractional string happens to
        # be the same length (same digit count means the naive integer read
        # and the true place-value read agree), so a collision here is
        # resampled rather than shipped.
        longer_is_bigger_items = sorted(items, key=lambda p: int(p[1]))

        # shorter-is-bigger: the mirror belief -- a shorter fractional digit
        # string is bigger, so ranks fewer-digit items last in ascending
        # order; same same-length collision risk as above.
        shorter_is_bigger_items = sorted(items, key=lambda p: (-len(p[1]), Fraction(f"{p[0]}.{p[1]}")))

        swap_at = rng.randrange(3)
        adjacent_swap_items = list(ordered_items)
        adjacent_swap_items[swap_at], adjacent_swap_items[swap_at + 1] = (
            adjacent_swap_items[swap_at + 1],
            adjacent_swap_items[swap_at],
        )

        # Distinctness is checked directly, every candidate, every seed
        # (exact tuple comparison over Fraction values -- these are lists,
        # not scalars, so no tolerance applies). `reversed`/`adjacent_swap`
        # can never collide with the correct order by construction (4
        # distinct values can never equal their own reversal, and swapping
        # two distinct adjacent values always changes the tuple); only
        # `longer_is_bigger`/`shorter_is_bigger` can coincide by chance
        # (same-length fractional strings), so a collision there resamples
        # the whole item set rather than shipping a question with two
        # "different" options that are secretly the same order.
        correct_tuple = tuple(ordered_values)
        collided = any(
            tuple(Fraction(f"{w}.{f}") for w, f in candidate_items) == correct_tuple
            for candidate_items in (longer_is_bigger_items, shorter_is_bigger_items)
        )
        if collided:
            continue
        break
    else:
        raise RuntimeError(f"could not build distinct distractor orderings for tier={tier!r}")

    display_order = list(range(4))
    rng.shuffle(display_order)
    display_text = ", ".join(_decimal_text(*items[i]) for i in display_order)
    answer_text = ", ".join(_decimal_text(*p) for p in ordered_items)
    reversed_text = ", ".join(_decimal_text(*p) for p in reversed_items)
    longer_text = ", ".join(_decimal_text(*p) for p in longer_is_bigger_items)
    shorter_text = ", ".join(_decimal_text(*p) for p in shorter_is_bigger_items)
    swap_text = ", ".join(_decimal_text(*p) for p in adjacent_swap_items)

    candidates = [reversed_text, longer_text, shorter_text, swap_text]

    working = [
        "If two numbers have a different whole-number part, compare that first.",
        "Otherwise compare the decimal digits one place at a time, from the left "
        "(tenths, then hundredths, then thousandths, and so on).",
        f"In order: {answer_text}.",
    ]
    return dict(
        text=f"Arrange these decimals from smallest to largest: {display_text}.",
        latex=r"\text{Arrange from smallest to largest: }" + display_text,
        answer=[float(v) for v in ordered_values],
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=pick_distractors(answer_text, candidates, n=4),
    )
