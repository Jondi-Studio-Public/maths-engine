"""Rate calculations and rate comparisons, both senior ``acara-senior-v8-4``, domain ``arithmetic``, ``answer:
number``.

Two generators:

- ``p7d_complete_rate_calculations`` (``senior-complete-calculations-with-
  rates-including-solving-problems-involving``, ACMEM073) -- rate x
  quantity arithmetic across four contexts (speed, flow, pay-as-total-cost,
  production). Three tiers: easy (rate x quantity/time -> total, one step),
  medium (the *unknown* moves -- total and one factor given, solve for the
  other), hard (a combined-rate word problem: two taps/workers at different
  rates working together on one fixed, shared total).

- ``p7d_use_rates_to_compare`` (``senior-use-rates-to-make-comparisons``,
  ACMEM074) -- always ends in a *comparison* between two rates (a positive
  difference), never a bare rate. Three tiers: easy (rates given directly,
  compare), medium (totals and quantities given, compute both rates first,
  then compare), hard (two pack prices in different units -- mL and L --
  requiring a unit conversion before the two per-litre prices can be
  compared, to the nearest cent).

Boundary against ``unit-pricing`` (``financial-discrete``): that skill's
only graded shape is "divide one pack's price by its quantity to find the
price per item". Neither generator here ever reduces to that single
division as its whole answer -- ``p7d_complete_rate_calculations`` ranges
across non-price rate contexts and its money context is a *time*-rate total
cost ("$X per hour of hire" x hours), never a "$X for Y items" pack price;
``p7d_use_rates_to_compare`` always grades a comparison between two
computed rates, an operation ``unit-pricing`` never performs at all.

Boundary between the two skills built here: ``p7d_complete_rate_
calculations`` always ends in a single computed rate/quantity/time;
``p7d_use_rates_to_compare`` always ends in a value that only makes sense
as a comparison between two rates (a difference), never a bare rate.

Both use exact ``Fraction``/``int`` arithmetic throughout -- every quantity
by construction lands on an exact answer (an integer, or an exact multiple
of one cent for the hard-tier currency comparison); no generator here rounds
a division forward to build its own canonical answer.
"""

from __future__ import annotations

import math
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
)

_RESAMPLE_ATTEMPTS = 100


# --- Local currency formatting -----------------------------------------------
# `_currency` is a dollar-Fraction-to-string convenience wrapper around the
# shared `mathgen.core.currency_from_cents`, used throughout this module.


def _currency(value) -> str:
    """Format an exact-cents amount as "$X.YY" -- always two decimal places
    ("$9.50", never "$9.5"; "$9.00", never "$9"). Raises if `value` is not
    an exact multiple of one cent."""
    cents = Fraction(value) * 100
    if cents.denominator != 1:
        raise ValueError(f"{value} is not an exact number of cents")
    return currency_from_cents(int(cents))


def _plain(value) -> str:
    return str(int(value))


def _count_noun(n, singular: str, plural: str) -> str:
    """ "1 litre" vs "2 litres" -- singular/plural noun agreement for a
    computed count that can legitimately land on exactly one (the working
    text's rate *difference*, unlike the rates/quantities drawn directly
    from a range that excludes 1)."""
    return f"{n} {singular if n == 1 else plural}"


# =============================================================================
# Skill 1: p7d_complete_rate_calculations (ACMEM073)
# =============================================================================

_CONTEXTS_CALC = {
    "speed": dict(rate_range=(40, 120), qty_range=(2, 9)),
    "flow": dict(rate_range=(2, 20), qty_range=(3, 15)),
    "pay": dict(rate_range=(12, 45), qty_range=(2, 10)),
    "production": dict(rate_range=(5, 40), qty_range=(2, 12)),
}


def _verify_easy_calc(r: int, n: int, total: int):
    """Independent check: rebuild `total` by repeated addition of the rate,
    a genuinely different route from the multiplication that built it."""
    check_equivalent(sum(r for _ in range(n)), total, "easy total must equal r added n times")


def _verify_medium_calc(known: int, unknown: int, total: int):
    """Independent check: re-derive the unknown via Fraction division from
    the *stated* total and known factor -- the reverse operation from the
    multiplication that built `total`."""
    check_equivalent(Fraction(total, known), unknown, "medium unknown must equal total / known exactly")


def _verify_hard_calc(a: int, b: int, total: int, time: int):
    """Independent check: re-derive `time` as Fraction(total, a + b) from the
    *stated* total/a/b -- a different route from the construction, which
    built `total` from `time` (total = (a + b) * time)."""
    check_equivalent(Fraction(total, a + b), time, "hard time must equal total / (a + b) exactly")


def _calc_easy(rng):
    context = rng.choice(sorted(_CONTEXTS_CALC))
    cfg = _CONTEXTS_CALC[context]
    for _ in range(_RESAMPLE_ATTEMPTS):
        r = rng.randint(*cfg["rate_range"])
        n = rng.randint(*cfg["qty_range"])
        total = r * n
        cand_sum, cand_extra, cand_rate_only = r + n, r * (n + 1), r
        try:
            check_pairwise_distinct(total, cand_sum, cand_extra, cand_rate_only)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p7d_complete_rate_calculations: no collision-free easy draw found")

    _verify_easy_calc(r, n, total)

    fmt = _currency if context == "pay" else _plain
    if context == "speed":
        text = f"A car travels at {r} km/h. How far does it travel in {n} hours?"
        working = [f"Total distance = rate x time = {r} x {n} = {total} km."]
    elif context == "flow":
        text = f"Water flows from a tap at {r} litres per minute. How many litres flow in {n} minutes?"
        working = [f"Total volume = rate x time = {r} x {n} = {total} litres."]
    elif context == "pay":
        text = f"A tool-hire company charges {_currency(r)} per hour. How much does {n} hours of hire cost?"
        working = [f"Total cost = rate x time = {_currency(r)} x {n} = {_currency(total)}."]
    else:  # production
        text = f"A machine produces {r} items per hour. How many items does it produce in {n} hours?"
        working = [f"Total produced = rate x time = {r} x {n} = {total} items."]

    answer_text = fmt(total)
    candidates = [fmt(cand_sum), fmt(cand_extra), fmt(cand_rate_only)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "easy tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=total,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _calc_medium(rng):
    context = rng.choice(sorted(_CONTEXTS_CALC))
    cfg = _CONTEXTS_CALC[context]
    unknown_slot = rng.choice(("rate", "time"))
    for _ in range(_RESAMPLE_ATTEMPTS):
        r = rng.randint(*cfg["rate_range"])
        n = rng.randint(*cfg["qty_range"])
        if r == n:
            continue  # the "known value" distractor would equal the answer
        total = r * n
        if unknown_slot == "rate":
            known, unknown = n, r
        else:
            known, unknown = r, n
        cand_total, cand_known, cand_mult = total, known, total * known
        try:
            check_pairwise_distinct(unknown, cand_total, cand_known, cand_mult)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p7d_complete_rate_calculations: no collision-free medium draw found")

    _verify_medium_calc(known, unknown, total)

    use_currency = context == "pay" and unknown_slot == "rate"
    fmt = _currency if use_currency else _plain

    if context == "speed":
        if unknown_slot == "rate":
            text = f"A car travels {total} km in {n} hours at a constant speed. What is its speed, in km/h?"
            working = [f"Speed = distance / time = {total} / {n} = {r} km/h."]
        else:
            text = f"A car travels at {r} km/h and covers {total} km. How many hours does the journey take?"
            working = [f"Time = distance / speed = {total} / {r} = {n} hours."]
    elif context == "flow":
        if unknown_slot == "rate":
            text = (
                f"A tap fills a tank with {total} litres in {n} minutes at a constant rate. "
                f"What is the flow rate, in litres per minute?"
            )
            working = [f"Rate = volume / time = {total} / {n} = {r} litres per minute."]
        else:
            text = (
                f"A tap flows at {r} litres per minute and fills a container with {total} litres. "
                f"How many minutes does this take?"
            )
            working = [f"Time = volume / rate = {total} / {r} = {n} minutes."]
    elif context == "pay":
        if unknown_slot == "rate":
            text = (
                f"A worker earns {_currency(total)} for {n} hours of work at a constant hourly rate. "
                f"What is their hourly pay rate?"
            )
            working = [f"Rate = total pay / hours = {_currency(total)} / {n} = {_currency(r)} per hour."]
        else:
            text = (
                f"A worker is paid {_currency(r)} per hour and earns {_currency(total)} in total. "
                f"How many hours did they work?"
            )
            working = [f"Time = total pay / rate = {_currency(total)} / {_currency(r)} = {n} hours."]
    else:  # production
        if unknown_slot == "rate":
            text = (
                f"A machine produces {total} items in {n} hours at a constant rate. "
                f"What is its production rate, in items per hour?"
            )
            working = [f"Rate = items / time = {total} / {n} = {r} items per hour."]
        else:
            text = (
                f"A machine produces {r} items per hour and produces {total} items in total. "
                f"How many hours does this take?"
            )
            working = [f"Time = items / rate = {total} / {r} = {n} hours."]

    answer_text = fmt(unknown)
    candidates = [fmt(cand_total), fmt(cand_known), fmt(cand_mult)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "medium tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=unknown,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


_CONTEXTS_HARD_CALC = ("tap", "worker")


def _calc_hard(rng):
    context = rng.choice(_CONTEXTS_HARD_CALC)
    a = rng.randint(2, 7)
    b = rng.randint(2, 7)
    while b == a:
        b = rng.randint(2, 7)

    time = a * b
    total = (a + b) * a * b
    check(
        total % a == 0 and total % b == 0 and total % (a + b) == 0,
        "hard-tier construction must be exact by construction",
    )
    ignored_b = total // a  # time if only tap/worker A's rate were used
    ignored_a = total // b  # time if only tap/worker B's rate were used
    mult_instead = total // (a * b)  # = a + b -- multiplied the rates instead of adding them
    check_pairwise_distinct(time, ignored_b, ignored_a, mult_instead)

    _verify_hard_calc(a, b, total, time)

    if context == "tap":
        text = (
            f"Tap A fills a tank at {a} litres per minute. Tap B fills the same tank at {b} litres "
            f"per minute. Working together, starting at the same time, they fill a {total} litre "
            f"tank from empty. How many minutes does it take to fill the tank?"
        )
    else:
        text = (
            f"Worker A completes a job at {a} items per hour. Worker B completes the same kind of "
            f"job at {b} items per hour. Working together, starting at the same time, they must "
            f"complete a job requiring {total} items in total. How many hours does it take them to "
            f"finish?"
        )

    answer_text = _plain(time)
    candidates = [_plain(ignored_b), _plain(ignored_a), _plain(mult_instead)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "hard tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=time,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[f"Combined rate = {a} + {b} = {a + b}; time = total / combined rate = {total} / {a + b} = {time}."],
        distractors=distractors,
    )


@generator(
    "p7d_complete_rate_calculations",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def p7d_complete_rate_calculations(rng, mode="easy"):
    """Rate x quantity/time arithmetic across speed/flow/pay/production
    contexts. easy: one-step, rate x quantity -> total. medium: the unknown
    moves -- total and one factor given, solve for the other. hard:
    combined-rate word problem, two agents at different rates on one shared
    total."""
    if mode == "easy":
        return _calc_easy(rng)
    if mode == "medium":
        return _calc_medium(rng)
    if mode == "hard":
        return _calc_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# =============================================================================
# Skill 2: p7d_use_rates_to_compare (ACMEM074)
# =============================================================================

_CONTEXTS_COMPARE = {
    "speed": dict(rate_range=(40, 120), qty_range=(2, 10), currency=False),
    "flow": dict(rate_range=(2, 20), qty_range=(3, 15), currency=False),
    "pay": dict(rate_range=(15, 45), qty_range=(2, 10), currency=True),
    "production": dict(rate_range=(5, 40), qty_range=(2, 12), currency=False),
}


def _verify_easy_compare(rA: int, rB: int, answer: int):
    """Independent check: recompute the difference via max() - min(),
    rather than reusing whichever branch decided which rate was larger."""
    check_equivalent(max(rA, rB) - min(rA, rB), answer, "easy comparison must equal max - min")


def _verify_medium_compare(rA: int, rB: int, hoursA: int, hoursB: int, totalA: int, totalB: int, answer: int):
    """Independent check: confirm the totals were built from the stated
    rates by multiplication (the reverse of the division a learner performs
    to recover the rate), then re-derive both rates by Fraction division and
    re-subtract -- the actual route a learner is expected to take, kept
    separate from the construction above."""
    check(hoursA * rA == totalA and hoursB * rB == totalB, "medium totals must equal rate x hours exactly")
    check_equivalent(
        abs(Fraction(totalA, hoursA) - Fraction(totalB, hoursB)),
        answer,
        "medium comparison must equal the difference of the re-derived rates",
    )


def _verify_hard_compare(
    volA_mL: int,
    volB_L: int,
    packA_price: Fraction,
    packB_price: Fraction,
    priceA_perL: Fraction,
    priceB_perL: Fraction,
    answer_cents: int,
):
    """Independent check: re-derive both unit prices from the *stated* pack
    price and volume alone (mL -> L via division by 1000, not by reusing
    whatever multiplication built the pack price), then re-subtract."""
    rederived_A = packA_price / (Fraction(volA_mL, 1) / 1000)
    rederived_B = packB_price / volB_L
    check_equivalent(rederived_A, priceA_perL, "hard re-derived price A must match construction")
    check_equivalent(rederived_B, priceB_perL, "hard re-derived price B must match construction")
    check_equivalent(
        abs(rederived_A - rederived_B),
        Fraction(answer_cents, 100),
        "hard comparison must equal the re-derived price difference",
    )


def _compare_easy(rng):
    context = rng.choice(sorted(_CONTEXTS_COMPARE))
    cfg = _CONTEXTS_COMPARE[context]
    fmt = _currency if cfg["currency"] else _plain
    for _ in range(_RESAMPLE_ATTEMPTS):
        rA = rng.randint(*cfg["rate_range"])
        rB = rng.randint(*cfg["rate_range"])
        if rA == rB:
            continue
        hi, lo = max(rA, rB), min(rA, rB)
        answer = hi - lo
        cand_sum, cand_max, cand_min = rA + rB, hi, lo
        try:
            check_pairwise_distinct(answer, cand_sum, cand_max, cand_min)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p7d_use_rates_to_compare: no collision-free easy draw found")

    _verify_easy_compare(rA, rB, answer)
    faster = "A" if rA > rB else "B"

    if context == "speed":
        text = f"Car A travels at {rA} km/h and Car B at {rB} km/h. How many km/h faster is Car {faster}?"
        working = [f"Difference = {hi} - {lo} = {answer} km/h."]
    elif context == "flow":
        text = (
            f"Tap A flows at {rA} litres per minute and Tap B at {rB} litres per minute. "
            f"How many litres per minute faster is Tap {faster}?"
        )
        working = [f"Difference = {hi} - {lo} = {_count_noun(answer, 'litre', 'litres')} per minute."]
    elif context == "pay":
        text = (
            f"Worker A is paid {_currency(rA)} per hour and Worker B is paid {_currency(rB)} per hour. "
            f"How much more per hour does Worker {faster} earn?"
        )
        working = [f"Difference = {_currency(hi)} - {_currency(lo)} = {_currency(answer)}."]
    else:  # production
        text = (
            f"Machine A produces {rA} items per hour and Machine B produces {rB} items per hour. "
            f"How many more items per hour does Machine {faster} produce?"
        )
        working = [f"Difference = {hi} - {lo} = {_count_noun(answer, 'item', 'items')} per hour."]

    answer_text = fmt(answer)
    candidates = [fmt(cand_sum), fmt(cand_max), fmt(cand_min)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "easy comparison tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _compare_medium(rng):
    context = rng.choice(sorted(_CONTEXTS_COMPARE))
    cfg = _CONTEXTS_COMPARE[context]
    fmt = _currency if cfg["currency"] else _plain
    for _ in range(_RESAMPLE_ATTEMPTS):
        rA = rng.randint(*cfg["rate_range"])
        rB = rng.randint(*cfg["rate_range"])
        if rA == rB:
            continue
        hoursA = rng.randint(2, 10)
        hoursB = rng.randint(2, 10)
        if hoursA == hoursB:
            continue
        totalA, totalB = rA * hoursA, rB * hoursB
        answer = abs(rA - rB)
        cand_raw_totals = abs(totalA - totalB)
        cand_added_rates = rA + rB
        cand_raw_qty = abs(hoursA - hoursB)
        try:
            check_pairwise_distinct(answer, cand_raw_totals, cand_added_rates, cand_raw_qty)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p7d_use_rates_to_compare: no collision-free medium draw found")

    _verify_medium_compare(rA, rB, hoursA, hoursB, totalA, totalB, answer)
    faster = "A" if rA > rB else "B"

    if context == "speed":
        text = (
            f"Car A travels {totalA} km in {hoursA} hours. Car B travels {totalB} km in {hoursB} hours. "
            f"Based on their average speeds, how many km/h faster is Car {faster}?"
        )
        working = [f"Rates: {totalA}/{hoursA} = {rA} km/h, {totalB}/{hoursB} = {rB} km/h; difference = {answer} km/h."]
    elif context == "flow":
        text = (
            f"Tap A releases {totalA} litres in {hoursA} minutes. Tap B releases {totalB} litres in "
            f"{hoursB} minutes. How many litres per minute faster is Tap {faster}?"
        )
        working = [
            f"Rates: {totalA}/{hoursA} = {rA} L/min, {totalB}/{hoursB} = {rB} L/min; "
            f"difference = {_count_noun(answer, 'litre', 'litres')} per minute."
        ]
    elif context == "pay":
        text = (
            f"Worker A earns {_currency(totalA)} for {hoursA} hours of work. Worker B earns "
            f"{_currency(totalB)} for {hoursB} hours of work. How much more per hour does Worker "
            f"{faster} earn?"
        )
        working = [
            f"Rates: {_currency(totalA)}/{hoursA} = {_currency(rA)}/h, "
            f"{_currency(totalB)}/{hoursB} = {_currency(rB)}/h; difference = {_currency(answer)}."
        ]
    else:  # production
        text = (
            f"Machine A produces {totalA} items in {hoursA} hours. Machine B produces {totalB} items "
            f"in {hoursB} hours. How many more items per hour does Machine {faster} produce?"
        )
        working = [
            f"Rates: {totalA}/{hoursA} = {rA} items/h, {totalB}/{hoursB} = {rB} items/h; "
            f"difference = {_count_noun(answer, 'item', 'items')} per hour."
        ]

    answer_text = fmt(answer)
    candidates = [fmt(cand_raw_totals), fmt(cand_added_rates), fmt(cand_raw_qty)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "medium comparison tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _compare_hard(rng):
    for _ in range(_RESAMPLE_ATTEMPTS * 3):
        xA = rng.randint(20, 600)  # price A, whole cents per litre
        k = rng.randint(1, 20)  # volA_mL = 100 * k
        if (xA * k) % 10 != 0:
            continue  # cheap check first: packA would not land on an exact cent
        volA_mL = 100 * k
        xB = rng.randint(20, 600)
        if xB == xA:
            continue
        volB_L = rng.randint(1, 5)

        priceA_perL = Fraction(xA, 100)
        priceB_perL = Fraction(xB, 100)
        packA_cents = (xA * k) // 10
        packB_cents = xB * volB_L
        answer_cents = abs(xA - xB)

        wrong_A_per_L = priceA_perL / 1000
        d1_value = abs(wrong_A_per_L - priceB_perL)
        d1_cents = math.floor(d1_value * 100 + Fraction(1, 2))
        d2_cents = xA + xB  # priceA_perL + priceB_perL, in cents

        if d1_cents == answer_cents or d2_cents == answer_cents:
            continue  # cheap check: rounded-to-cent distractor would clear the grader
        try:
            check_pairwise_distinct(Fraction(answer_cents, 100), Fraction(d1_cents, 100), Fraction(d2_cents, 100))
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p7d_use_rates_to_compare: no collision-free hard draw found")

    packA_price = Fraction(packA_cents, 100)
    packB_price = Fraction(packB_cents, 100)
    _verify_hard_compare(volA_mL, volB_L, packA_price, packB_price, priceA_perL, priceB_perL, answer_cents)

    text = (
        f"Brand A sells a {volA_mL} mL bottle for {_currency(packA_price)}. "
        f"Brand B sells a {volB_L} L bottle for {_currency(packB_price)}. (1 L = 1000 mL.) "
        f"Find the difference between the two brands' prices per litre, to the nearest cent."
    )
    answer_text = currency_from_cents(answer_cents)
    candidates = [currency_from_cents(d1_cents), currency_from_cents(d2_cents)]
    distractors = pick_distractors(answer_text, candidates, n=2, tolerance=Fraction(1, 200))
    check(
        len(distractors) == 2,
        "hard comparison tier's construction genuinely supports only 2 distractors "
        "(a raw-mL-as-litres error and an added-instead-of-subtracted error)",
    )

    return dict(
        text=text,
        latex=text,
        answer=float(Fraction(answer_cents, 100)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"Price A per litre = {_currency(priceA_perL)}; price B per litre = "
            f"{_currency(priceB_perL)}; difference = {answer_text}."
        ],
        distractors=distractors,
    )


@generator(
    "p7d_use_rates_to_compare",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def p7d_use_rates_to_compare(rng, mode="easy"):
    """Comparisons between two rates -- always a positive difference, never
    a bare rate. easy: rates given directly, no computation needed to reach
    a rate. medium: totals and quantities given, compute both rates first.
    hard: two pack prices in different units (mL, L) requiring conversion
    before the per-litre prices can be compared, to the nearest cent."""
    if mode == "easy":
        return _compare_easy(rng)
    if mode == "medium":
        return _compare_medium(rng)
    if mode == "hard":
        return _compare_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")
