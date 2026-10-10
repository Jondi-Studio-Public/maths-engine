"""Generates `senior-review-rates-and-percentages` questions (ACMGM001): a review of rates and percentages,
`answer: number`.

``senior-review-rates-and-percentages``, ``acara-senior-v8-4``, domain ``arithmetic``, ``answer: number``,
target area ACMGM001.

ACMGM001's own official ACARA name is "review rates AND percentages" -- both
concepts reviewed together, not either one alone. Every question here is a
genuine TWO-step calculation: (1) a per-unit rate x a quantity gives a
subtotal, then (2) a percentage adjustment (a GST/loading increase, or a
discount decrease) is applied to that subtotal. The GRADED value is always
the final amount after both steps -- never the subtotal alone, never the
bare rate alone, never the raw percentage amount alone.

Boundary against ``senior-use-rates-to-make-comparisons``
(``mathgen/generators/p7d_rates_build.py``, ACMEM074) and its sibling
``senior-complete-calculations-with-rates-including-solving-problems-
involving`` (ACMEM073): those skills own "compute/compare a bare per-unit
rate" territory -- a single rate x quantity total, or a comparison between
two rates. Neither of those generators ever applies a percentage adjustment
on top of a computed subtotal, and this generator never stops at a bare rate
or a bare rate comparison; the percentage step is mandatory on every
question.

Boundary against ``senior-review-calculating-a-percentage-of-a-given-amount``
(a sibling skill built separately and not
read by this module): that skill's own territory is a bare percentage-of-
amount calculation with no rate step at all (a stated amount x a percentage,
full stop). This generator never grades a bare percentage-of-amount either --
the amount being adjusted is never given directly, it is always the product
of a rate and a quantity computed first.

Boundary against ``curriculum_expansion.py:earnings(kind="overtime")``: that generator multiplies a whole
BLOCK of hours by a fixed rate MULTIPLIER (1.5x/2x overtime loading), never a stated PERCENTAGE adjustment
applied to a subtotal, and it is not wired to any senior skill. Every question here names an explicit
percentage (GST/loading/discount), never an hours-multiplier framing.

Exact ``Fraction``/``int`` arithmetic throughout, worked entirely in integer cents -- every canonical answer
lands on an exact cent by construction. A resample loop (``_RESAMPLE_ATTEMPTS``, modelled on
``p7d_rates_build.py``'s ``_compare_hard``) keeps drawing new rate/hours values and searching the percent
range until ``subtotal_cents * (100 +/- percent) / 100`` divides evenly; it raises if no valid draw is found
within the bound. The easy tier's draw always divides evenly on the first attempt (whole-dollar rate x whole
hours x a round percent); medium/hard use a two-level search -- an outer redraw of (rate, hours) and an inner
search of the whole 5-45 percent range for a value that divides evenly -- rather than blindly redrawing all
three together, because the blind version measured a 188/200 worst case over a 2000-seed sweep.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    currency_from_cents,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
)

_RESAMPLE_ATTEMPTS = 200

# Diagnostic only: the attempt number (1-based) on which each successful draw
# landed, appended by every `_draw_*` call. Read by the build test to report
# the measured resample-loop hit-rate; never read by the generator itself.
_ATTEMPT_LOG: list[int] = []

_PROFESSIONS = ("tradesperson", "plumber", "mechanic", "technician", "gardener", "painter", "carpenter", "builder")

_EASY_PERCENTS = (10, 15, 20, 25, 30)


def _hours_text(hours: Fraction) -> str:
    if hours.denominator == 1:
        return str(hours.numerator)
    if hours.denominator == 2:
        return f"{hours.numerator // 2}.5"
    raise ValueError(f"unsupported hours fraction: {hours}")


# --- In-generator independent check: re-parse the rendered text ------------
# Deliberately re-derives the subtotal and final amount from the DISPLAYED
# rate/hours/percent tokens in `text`, via Fraction arithmetic on dollar
# strings -- not by reusing the cents integers construction already holds.
# This is a *different* implementation from the oracle test file's own parse
# (which must not import this module at all); the two are independent by
# design, not merely by file location.
_RATE_HOURS_PCT_RE = re.compile(r"\$([\d.]+)/hour\. A job takes ([\d.]+) hours.*?(\d+)%")


def _reparse_final_dollars(text: str, direction: int) -> Fraction:
    m = _RATE_HOURS_PCT_RE.search(text)
    check(bool(m), "could not parse rate/hours/percent back out of the rendered question text")
    rate = Fraction(m.group(1))
    hours = Fraction(m.group(2))
    p = int(m.group(3))
    subtotal = rate * hours
    factor = Fraction(100 + direction * p, 100)
    return subtotal * factor


# =============================================================================
# Draw logic, one function per tier. Each returns a dict of the raw
# ingredients; the caller builds the question text/working/distractors.
# =============================================================================


def _draw_easy(rng):
    for attempt in range(1, _RESAMPLE_ATTEMPTS + 1):
        rate_cents = rng.randint(15, 45) * 100
        hours = Fraction(rng.randint(2, 10))
        p = rng.choice(_EASY_PERCENTS)
        direction = 1

        subtotal_raw = Fraction(rate_cents) * hours
        if subtotal_raw.denominator != 1:
            continue
        subtotal_cents = int(subtotal_raw)

        final_raw = subtotal_raw * Fraction(100 + direction * p, 100)
        if final_raw.denominator != 1:
            continue
        final_cents = int(final_raw)
        if final_cents >= 100000:
            continue

        ignore_rate_raw = Fraction(rate_cents) * Fraction(100 + direction * p, 100)
        check(ignore_rate_raw >= 0, "a cents amount must never be negative")
        ignore_rate_cents = int(round_half_up(ignore_rate_raw, 0))
        try:
            check_pairwise_distinct(final_cents, ignore_rate_cents, subtotal_cents)
        except VerificationError:
            continue

        _ATTEMPT_LOG.append(attempt)
        return dict(
            rate_cents=rate_cents,
            hours=hours,
            p=p,
            direction=direction,
            subtotal_cents=subtotal_cents,
            final_cents=final_cents,
            ignore_rate_cents=ignore_rate_cents,
        )
    raise ValueError(f"p8e_rates_percentages: no collision-free easy draw found within {_RESAMPLE_ATTEMPTS} attempts")


_PERCENT_RANGE = tuple(range(5, 46))


def _draw_medium(rng):
    """Two-level search: an outer draw of (rate, hours) fixes the subtotal; an inner search over the full 5-45
    percent range picks whichever values actually divide evenly, rather than redrawing all three independently
    and hoping. A blind three-way redraw (rate, hours AND percent every attempt) measured a worst case of
    188/200 attempts over a 2000-seed sweep -- close enough to the bound to be a real risk over the site's
    full lifetime of draws; this structure keeps the outer loop (the only part that can fail, since the inner
    search either finds a valid percent among 41 candidates or proves none exist for that subtotal) needing on
    average under 2 attempts."""
    attempts_used = 0
    for outer in range(1, _RESAMPLE_ATTEMPTS + 1):
        rate_cents = rng.randint(1500, 4500)
        if rate_cents % 100 == 0:
            continue  # medium tier must genuinely state cents, never a round dollar
        hours = Fraction(rng.randint(2, 10))

        subtotal_raw = Fraction(rate_cents) * hours
        if subtotal_raw.denominator != 1:
            attempts_used = outer
            continue
        subtotal_cents = int(subtotal_raw)

        candidate_ps = [p for p in _PERCENT_RANGE if (subtotal_cents * (100 + p)) % 100 == 0]
        rng.shuffle(candidate_ps)
        for p in candidate_ps:
            direction = 1
            final_raw = subtotal_raw * Fraction(100 + direction * p, 100)
            final_cents = int(final_raw)
            if final_cents >= 100000:
                continue
            ignore_rate_raw = Fraction(rate_cents) * Fraction(100 + direction * p, 100)
            check(ignore_rate_raw >= 0, "a cents amount must never be negative")
            ignore_rate_cents = int(round_half_up(ignore_rate_raw, 0))
            try:
                check_pairwise_distinct(final_cents, ignore_rate_cents, subtotal_cents)
            except VerificationError:
                continue

            _ATTEMPT_LOG.append(outer)
            return dict(
                rate_cents=rate_cents,
                hours=hours,
                p=p,
                direction=direction,
                subtotal_cents=subtotal_cents,
                final_cents=final_cents,
                ignore_rate_cents=ignore_rate_cents,
            )
        attempts_used = outer
    raise ValueError(
        "p8e_rates_percentages: no collision-free medium draw found "
        f"within {_RESAMPLE_ATTEMPTS} outer attempts (last={attempts_used})"
    )


_HOURS_HALF_STEPS = (Fraction(5, 2), Fraction(7, 2), Fraction(9, 2), Fraction(11, 2), Fraction(13, 2), Fraction(15, 2))


def _draw_hard(rng):
    """Same two-level search as `_draw_medium` (see its docstring): the
    outer draw fixes rate/hours, the inner search picks a discount
    percentage from the full 5-45 range that actually divides evenly."""
    attempts_used = 0
    for outer in range(1, _RESAMPLE_ATTEMPTS + 1):
        rate_cents = rng.randint(1500, 4500)
        if rate_cents % 100 == 0:
            continue  # hard tier must genuinely state cents, never a round dollar
        if rate_cents % 2 != 0:
            continue  # must be even so a half-hour draw lands on an exact cent
        hours = rng.choice(_HOURS_HALF_STEPS)

        subtotal_raw = Fraction(rate_cents) * hours
        if subtotal_raw.denominator != 1:
            attempts_used = outer
            continue
        subtotal_cents = int(subtotal_raw)

        candidate_ps = [p for p in _PERCENT_RANGE if (subtotal_cents * (100 - p)) % 100 == 0]
        rng.shuffle(candidate_ps)
        for p in candidate_ps:
            direction = -1  # a discount: decrease
            final_raw = subtotal_raw * Fraction(100 + direction * p, 100)
            final_cents = int(final_raw)
            if final_cents >= 100000:
                continue
            ignore_rate_raw = Fraction(rate_cents) * Fraction(100 + direction * p, 100)
            check(ignore_rate_raw >= 0, "a cents amount must never be negative")
            ignore_rate_cents = int(round_half_up(ignore_rate_raw, 0))
            wrong_direction_raw = subtotal_raw * Fraction(100 - direction * p, 100)
            check(wrong_direction_raw >= 0, "a cents amount must never be negative")
            wrong_direction_cents = int(round_half_up(wrong_direction_raw, 0))
            try:
                check_pairwise_distinct(final_cents, ignore_rate_cents, subtotal_cents, wrong_direction_cents)
            except VerificationError:
                continue

            _ATTEMPT_LOG.append(outer)
            return dict(
                rate_cents=rate_cents,
                hours=hours,
                p=p,
                direction=direction,
                subtotal_cents=subtotal_cents,
                final_cents=final_cents,
                ignore_rate_cents=ignore_rate_cents,
                wrong_direction_cents=wrong_direction_cents,
            )
        attempts_used = outer
    raise ValueError(
        "p8e_rates_percentages: no collision-free hard draw found "
        f"within {_RESAMPLE_ATTEMPTS} outer attempts (last={attempts_used})"
    )


def _build(rng, mode):
    prof = rng.choice(_PROFESSIONS)
    if mode == "easy":
        draw = _draw_easy(rng)
    elif mode == "medium":
        draw = _draw_medium(rng)
    elif mode == "hard":
        draw = _draw_hard(rng)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    rate_cents = draw["rate_cents"]
    hours = draw["hours"]
    p = draw["p"]
    direction = draw["direction"]
    subtotal_cents = draw["subtotal_cents"]
    final_cents = draw["final_cents"]

    check(final_cents < 100000, "final amount must stay under $1000 by construction")

    rate_text = currency_from_cents(rate_cents)
    hours_text = _hours_text(hours)

    if mode == "easy":
        text = (
            f"A {prof} charges {rate_text}/hour. A job takes {hours_text} hours, plus a "
            f"{p}% call-out loading. What is the total cost?"
        )
        working = [
            f"Subtotal = {rate_text} x {hours_text} = {currency_from_cents(subtotal_cents)}.",
            f"Total cost = {currency_from_cents(subtotal_cents)} x (100 + {p})/100 = "
            f"{currency_from_cents(final_cents)}.",
        ]
    elif mode == "medium":
        text = (
            f"A {prof} charges {rate_text}/hour. A job takes {hours_text} hours, plus a "
            f"{p}% GST charge on the labour cost. What is the total cost?"
        )
        working = [
            f"Subtotal = {rate_text} x {hours_text} = {currency_from_cents(subtotal_cents)}.",
            f"Total cost = {currency_from_cents(subtotal_cents)} x (100 + {p})/100 = "
            f"{currency_from_cents(final_cents)}.",
        ]
    else:  # hard
        text = (
            f"A {prof} charges {rate_text}/hour. A job takes {hours_text} hours, but a "
            f"{p}% loyalty discount applies to the labour cost. What is the total cost "
            f"after the discount?"
        )
        working = [
            f"Subtotal = {rate_text} x {hours_text} = {currency_from_cents(subtotal_cents)}.",
            f"Total cost after discount = {currency_from_cents(subtotal_cents)} x (100 - {p})/100 = "
            f"{currency_from_cents(final_cents)}.",
        ]

    # In-generator independent check: re-derive the final amount by parsing
    # rate/hours/percent back out of the rendered text (Fraction dollar
    # arithmetic), not by reusing construction's own cents integers.
    reparsed = _reparse_final_dollars(text, direction)
    check_equivalent(
        reparsed, Fraction(final_cents, 100), "text-reparsed final amount must equal the constructed answer"
    )

    answer_text = currency_from_cents(final_cents)
    candidate_texts = [currency_from_cents(draw["ignore_rate_cents"]), currency_from_cents(draw["subtotal_cents"])]
    if mode == "hard":
        candidate_texts.append(currency_from_cents(draw["wrong_direction_cents"]))

    distractors = pick_distractors(answer_text, candidate_texts, n=len(candidate_texts))
    check(len(distractors) == len(candidate_texts), f"{mode} tier expects {len(candidate_texts)} distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(Fraction(final_cents, 100)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


@generator(
    "rate_percentage_combined",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def rate_percentage_combined(rng, mode="easy"):
    """A per-unit rate x hours gives a subtotal, then a percentage
    adjustment is applied to that subtotal -- the graded value is always the
    final amount after both steps. easy: whole-dollar rate, whole hours, a
    round percentage INCREASE (GST/loading framing). medium: rate stated
    with cents, same hours range, percentage increase from a wider range --
    the two-decimal-place rate is the genuine extra reasoning step. hard:
    rate with cents, fractional (half-step) hours, and the percentage is a
    DISCOUNT (decrease) instead of an increase -- the direction reversal is
    the genuine escalation, not merely more decimals."""
    return _build(rng, mode)
