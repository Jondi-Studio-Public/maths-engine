"""Builds the `f10-direct-proportion-scale-model` skill (AC9M9M05/AC9M10M05): direct proportion applied to scale
models.

Topic: ``p13a_direct_proportion_scale``. The core of the skill is the
CLASSIFICATION decision itself: given a stated ratio/rate/scale context,
decide whether it is directly proportional (a pure per-unit rate, no fixed
component) or NOT (a per-unit rate plus a fixed component), then compute a
requested total. Unlike
``senior-examine-examples-of-direct-proportion-and-linearly-related`` (whose
own contract excludes "classifying an unlabelled table of values as
direct, inverse or quadratic from the data alone; the relation type and its
formula are always stated"), this skill NEVER states the relation type or
formula up front -- the learner infers it from the scenario.

This module now asks the classification EXPLICITLY in every tier's question text ("Is this relationship
directly proportional, and...?") and grades it as part of a single ``mcq`` answer: each option BUNDLES a
classification label with a computed value (e.g. "Yes, directly proportional -- the total cost is $35.00."),
so picking the right option requires getting BOTH the classification and the value right.

Two scenario kinds, never named to the learner:

- ``pure_rate``: total = rate * quantity, exactly directly proportional.
- ``rate_plus_fixed``: total = fixed + rate * quantity, NOT proportional
  (doubling the quantity does not double the total).

Four tiers, genuine reasoning escalation (never magnitude alone):

- easy: ``pure_rate`` only (a deliberate, disclosed constant baseline tier --
  there is nothing yet to compare against), rate stated explicitly as a
  number, forward direction (given a quantity, find the total). The
  classification is still explicitly asked and graded here (one consistent
  mcq shape across the whole skill), but every option shares the same "Yes,
  directly proportional" label -- only the computed total varies -- because
  there is only ever one kind at this tier.
- medium: BOTH ``pure_rate`` and ``rate_plus_fixed`` are now drawn (a
  fix -- previously ``pure_rate`` only), roughly 50/50, on EACH of the
  two existing reasoning-step variants: EITHER the rate must first be
  EXTRACTED from a stated group cost ("n units cost $x") before multiplying
  by a NEW quantity, OR the direction is INVERSE (given the total, find the
  quantity). The ``rate_plus_fixed`` draw adds a one-off delivery fee to the
  new order/total (never to the stated group-cost sentence itself, so the
  extraction step over easy's already-isolated rate remains genuine and
  uncontaminated by the fee).
- hard: BOTH ``pure_rate`` and ``rate_plus_fixed`` are now drawn, roughly 50/50, using the same
  scenario families (plumber call-out, phone plan, car rental) and the same,
  larger rate pool (``_HARD_RATE_CENTS``) as before -- hard's escalation over
  medium remains the bigger numbers and the call-out/monthly-plan/rental-fee
  contexts, not merely "sometimes has a fee now".
- exam: two stated data points (quantity, total), no formula named -- the
  learner must test whether the two implied unit rates agree (``pure_rate``)
  or disagree (``rate_plus_fixed``, solve the 2x2 system for rate and
  fixed) before finding a third total, and now states the classification
  explicitly in the same combined mcq option as before. `difficulty: exam,
  stages: [exam]`.

Every value is exact integer-cents ``Fraction`` arithmetic; every displayed
money value uses ``mathgen.core.stated_precision_string`` to 2dp
(``stated_precision_tolerance(2) == 0.005``), and the medium tier's inverse
variant answers with a plain whole-number quantity baked into its mcq option
text -- so no rounding-boundary risk exists anywhere in this generator.
Misconceptions are checked pairwise-distinct as raw ``Fraction`` cent values
(`check_pairwise_distinct`, exactly as before) AND, separately, as the four
fully rendered mcq option strings via plain set-length equality --
`check_pairwise_distinct`/`check_distinct` call `symbolically_equal`, which
`sympify`s its arguments; a natural-language mcq option sentence such as
"Yes, directly proportional -- the total cost is $35.00." is not valid
Python/sympy syntax and `sympify` raises `SympifyError` on it (verified
directly against `mathgen.core.symbolically_equal` while building this fix,
including sympifying a string against itself) -- so `check_pairwise_distinct`
cannot be used on the option strings themselves, only on the underlying
numeric candidates. `mathgen/generators/p10d_similarity_scale_factor_build.py`
follows the same discipline: its own natural-language ``OPTION_*`` mcq
literals are compared with plain equality/membership, and `check_distinct` is
only ever applied there to its numeric candidates. This module does the same:
plain set-length equality on the four formatted option strings, inside the
existing resample-on-collision retry loop.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    currency_from_cents,
    generator,
    latex_answer,
    money_cents_rounded_2dp,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_RESAMPLE_ATTEMPTS = 200
_TOL2 = stated_precision_tolerance(2)

# The two classification labels bundled into every mcq option.
# Public contract: quoted verbatim by the independent oracle test, never imported from
# there or anywhere else.
_LABEL_YES = "Yes, directly proportional"
_LABEL_NO = "No, not directly proportional (a fixed fee applies)"


def _money(cents) -> str:
    """A canonical money answer/given value, forced to exactly 2dp -- the
    caller must already have landed on an exact integer cents value.
    Delegates to `mathgen.core.currency_from_cents` (never hand-rolled)."""
    cents = Fraction(cents)
    check(cents.denominator == 1, f"{cents} is not an exact integer cents value")
    return currency_from_cents(int(cents), symbol="")


def _money_rounded(cents) -> str:
    """A DISTRACTOR display value that may not land on an exact cent (a
    misconception's own arithmetic, e.g. an inverted rate) -- rounds
    half-up to the cent. Never used for a canonical answer_text.

    Delegates to `mathgen.core.money_cents_rounded_2dp`, added to replace the
    local helper this function used to carry (Fraction cents, possibly not
    integer, -> half-up-rounded 2dp dollar string): this exact three-line
    body was duplicated verbatim as `_money_rounded` here and in
    `mathgen/generators/p12a_financial_model_build.py`."""
    return money_cents_rounded_2dp(cents)


def _classification_question(stem: str, tail: str) -> str:
    """Bolt the explicit classification question onto a scenario stem, so
    every tier ASKS "is this directly proportional" rather than only asking
    for the computed value. `tail` is the lowercase
    value-question clause, e.g. "what is the total cost for 6 litres?"."""
    return f"{stem} Is this relationship directly proportional, and {tail}"


def _money_opt(label: str, money_text: str) -> str:
    """One mcq option bundling a classification label with a computed money
    value -- selecting it requires getting BOTH right."""
    return f"{label} -- the total cost is ${money_text}."


def _qty_opt(label: str, qty_text: str, unit: str, unit_plural: str) -> str:
    """One mcq option bundling a classification label with a computed
    whole-number quantity (medium tier's inverse-direction variant).
    Pluralises the unit noun from the value itself -- "1 metre" but
    "2 metres" -- never a fixed plural regardless of the number."""
    noun = unit if qty_text in ("1", "-1") else unit_plural
    return f"{label} -- you can buy {qty_text} {noun}."


# ---------------------------------------------------------------------------
# Shared item/context pools -- every item below is a mass noun (never a
# countable plural), so "costs" is always the correct singular verb and no
# article/plural agreement logic is needed anywhere in this module.
# ---------------------------------------------------------------------------

_RATE_ITEMS = (
    ("petrol", "litre", "litres"),
    ("diesel", "litre", "litres"),
    ("flour", "kg", "kg"),
    ("rice", "kg", "kg"),
    ("fabric", "metre", "metres"),
    ("ribbon", "metre", "metres"),
)

#: Per-unit rates, every value a multiple of 25c -- chosen so a rate times
#: any integer quantity always lands on an exact cent (never a rounding
#: question), by construction.
_UNIT_RATES_CENTS = (50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450)

#: "n units cost $x" group sizes for the medium tier's extraction variant.
_GROUP_SIZES = (2, 3, 4, 5, 6, 8, 10)

#: Fixed-fee pool shared by every ``rate_plus_fixed`` draw at every tier
#: (medium's delivery fee, hard's call-out/monthly/booking fee, exam's
#: implied fixed component). Deliberately kept a single shared pool (never
#: narrowed per-tier) so the medium-inverse tier's construction-based
#: no-collision guarantee holds everywhere: every value here is >= $5.00
#: while every `_UNIT_RATES_CENTS` value is <= $4.50, so fixed/rate > 1.11
#: always -- see `_medium_inverse_question`'s docstring for why this rules
#: out the "omits-fixed-fee" distractor ever coinciding with the answer.
_FIXED_FEES_CENTS = (500, 750, 1000, 1500, 2000, 2500, 3000)
_HARD_RATE_CENTS = (500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000)

# Hard tier's rate_plus_fixed scenario stems and pure_rate counterparts introduced, same larger rate
# pool, with the fixed-fee clause simply absent.
_HARD_TEMPLATES = (
    (
        lambda f, r: f"A plumber charges a ${f} call-out fee plus ${r} per hour.",
        lambda q: f"what is the total cost for {q} hours of work?",
        "hour",
        "hours",
    ),
    (
        lambda f, r: f"A phone plan costs ${f} a month plus ${r} per GB used.",
        lambda q: f"what is the total cost for {q} GB used this month?",
        "GB",
        "GB",
    ),
    (
        lambda f, r: f"A car rental costs a ${f} flat booking fee plus ${r} per day of hire.",
        lambda q: f"what is the total cost for {q} days of hire?",
        "day",
        "days",
    ),
)

_HARD_TEMPLATES_PURE = (
    (
        lambda r: f"A plumber charges ${r} per hour.",
        lambda q: f"what is the total cost for {q} hours of work?",
        "hour",
        "hours",
    ),
    (
        lambda r: f"A phone plan costs ${r} per GB used.",
        lambda q: f"what is the total cost for {q} GB used this month?",
        "GB",
        "GB",
    ),
    (
        lambda r: f"A car rental costs ${r} per day of hire.",
        lambda q: f"what is the total cost for {q} days of hire?",
        "day",
        "days",
    ),
)

_EXAM_TEMPLATES = (
    ("hour", "hours", "A tradesperson's total charge depends on the number of hours worked"),
    ("GB", "GB", "A subscription service's monthly bill depends on the GB used"),
    ("day", "days", "A rental company's total charge depends on the number of days hired"),
)


# ---------------------------------------------------------------------------
# Easy: pure_rate only (disclosed baseline), rate stated explicitly, forward
# direction. The classification is asked but held constant (always "Yes").
# ---------------------------------------------------------------------------


def _easy_question(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        item, unit, unit_plural = rng.choice(_RATE_ITEMS)
        rate_cents = rng.choice(_UNIT_RATES_CENTS)
        quantity = rng.randint(2, 20)
        total_cents = Fraction(rate_cents * quantity)

        # inverts-the-rate: divides instead of multiplies (both kinds).
        inverts_rate_cents = Fraction(quantity * 10000, rate_cents)
        # decimal-shift: a factor of ten too large.
        decimal_shift_cents = total_cents * 10
        # ignores-quantity: reports the per-unit rate itself as the total.
        # easy is the deliberate, disclosed pure_rate-only baseline tier, so every option keeps the "Yes" label; only
        # the computed value varies.
        rate_as_total_cents = Fraction(rate_cents)

        answer_opt = _money_opt(_LABEL_YES, _money(total_cents))
        distractor_opts = [
            _money_opt(_LABEL_YES, _money_rounded(inverts_rate_cents)),
            _money_opt(_LABEL_YES, _money(decimal_shift_cents)),
            _money_opt(_LABEL_YES, _money(rate_as_total_cents)),
        ]

        try:
            check_pairwise_distinct(
                total_cents,
                inverts_rate_cents,
                decimal_shift_cents,
                rate_as_total_cents,
                msg="easy-tier candidates must be pairwise distinct",
            )
            check(len({answer_opt, *distractor_opts}) == 4, "easy-tier mcq options must be pairwise distinct")
        except Exception:
            continue

        rate_text = _money(rate_cents)
        item_cap = item[0].upper() + item[1:]
        stem = rng.choice(
            (
                f"{item_cap} costs ${rate_text} per {unit}.",
                f"{item_cap} is priced at ${rate_text} per {unit}.",
            )
        )
        text = _classification_question(stem, f"what is the total cost for {quantity} {unit_plural}?")

        # Independent check: repeated-addition route, never the
        # multiplication construction used.
        repeated_sum_cents = Fraction(sum(rate_cents for _ in range(quantity)))
        check_equivalent(repeated_sum_cents, total_cents, "easy pure-rate total mismatch")

        distractors = pick_distractors(answer_opt, distractor_opts, n=3)
        check(len(distractors) == 3, "expected exactly 3 surviving easy-tier distractors")

        return dict(
            text=text,
            answer=answer_opt,
            answer_text=answer_opt,
            answer_latex=latex_answer(answer_opt, plain=True),
            working=[
                f"{quantity} {unit_plural} at ${rate_text} per {unit}: ${_money(total_cents)}. "
                f"Directly proportional (a pure per-unit rate, no fixed component)."
            ],
            distractors=distractors,
        )
    raise ValueError("p13a_direct_proportion_scale/easy: no valid draw found")


# ---------------------------------------------------------------------------
# Medium: BOTH pure_rate and rate_plus_fixed now drawn on
# each of the two reasoning-step variants (extract-from-a-ratio, inverse).
# ---------------------------------------------------------------------------


def _medium_extract_question(rng) -> dict:
    """The rate is not given directly: "n units cost $x" must be divided
    down to a per-unit rate before multiplying by a NEW quantity -- a
    genuine added step over easy's already-isolated rate. On a
    ``rate_plus_fixed`` draw a one-off delivery fee is added to the NEW
    order/total only (never to the stated group-cost sentence), so the
    extraction step itself stays pure and uncontaminated by the fee."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        item, unit, unit_plural = rng.choice(_RATE_ITEMS)
        n = rng.choice(_GROUP_SIZES)
        rate_cents = rng.choice(_UNIT_RATES_CENTS)
        x_cents = rate_cents * n  # the stated group cost, exact by construction
        quantity2 = rng.randint(2, 20)
        if quantity2 == n:
            continue
        kind = rng.choice(("pure_rate", "rate_plus_fixed"))
        fixed_cents = 0 if kind == "pure_rate" else rng.choice(_FIXED_FEES_CENTS)

        variable_cents = Fraction(rate_cents * quantity2)
        total_cents = Fraction(fixed_cents) + variable_cents

        # skips-the-division-step: uses the stated group cost x directly as if it were itself the
        # per-unit rate, ignoring the group size n. Paired with "No" as the most defensible reading.
        skips_division_cents = Fraction(fixed_cents) + Fraction(x_cents * quantity2)
        # inverts-the-rate: divides instead of multiplies (both kinds),
        # still correctly adding any fixed component once.
        inverts_rate_cents = Fraction(fixed_cents) + Fraction(quantity2 * 10000 * n, x_cents)
        decimal_shift_cents = total_cents * 10
        # omits-the-fixed-fee -- only meaningfully offered on a
        # rate_plus_fixed draw. Paired with "Yes": a learner who omits the
        # fee usually also mis-identifies the situation as proportional.
        omits_fixed_cents = variable_cents
        # applies-the-fixed-fee-per-unit -- only meaningfully offered on a
        # rate_plus_fixed draw. Paired with "No" (aware of the fee, wrong
        # model).
        applies_fixed_per_unit_cents = Fraction(fixed_cents * quantity2) + variable_cents

        correct_label = _LABEL_NO if kind == "rate_plus_fixed" else _LABEL_YES
        answer_opt = _money_opt(correct_label, _money(total_cents))

        if kind == "pure_rate":
            distractor_opts = [
                _money_opt(_LABEL_NO, _money(skips_division_cents)),
                _money_opt(_LABEL_YES, _money_rounded(inverts_rate_cents)),
                _money_opt(_LABEL_YES, _money(decimal_shift_cents)),
            ]
            candidates_cents = (skips_division_cents, inverts_rate_cents, decimal_shift_cents)
        else:
            distractor_opts = [
                _money_opt(_LABEL_YES, _money(omits_fixed_cents)),
                _money_opt(_LABEL_NO, _money(applies_fixed_per_unit_cents)),
                _money_opt(_LABEL_NO, _money_rounded(inverts_rate_cents)),
            ]
            candidates_cents = (omits_fixed_cents, applies_fixed_per_unit_cents, inverts_rate_cents)

        try:
            check_pairwise_distinct(
                total_cents,
                *candidates_cents,
                msg="medium-extract candidates must be pairwise distinct",
            )
            check(len({answer_opt, *distractor_opts}) == 4, "medium-extract mcq options must be pairwise distinct")
        except Exception:
            continue

        x_text = _money(x_cents)
        item_cap = item[0].upper() + item[1:]
        stem = rng.choice(
            (
                f"{n} {unit_plural} of {item} cost ${x_text} in total.",
                f"{n} {unit_plural} of {item} together cost ${x_text}.",
                f"Buying {n} {unit_plural} of {item} costs ${x_text} altogether.",
            )
        )
        if kind == "rate_plus_fixed":
            fixed_text = _money(fixed_cents)
            stem = f"{stem} There is also a one-off ${fixed_text} delivery fee for the whole order."
            tail = f"what is the total cost for {quantity2} {unit_plural}, including delivery?"
        else:
            tail = f"what is the total cost for {quantity2} {unit_plural}?"
        text = _classification_question(stem, tail)

        # Independent check: cross-multiplication route, never computing the
        # per-unit rate as its own intermediate value first.
        cross_mult_variable = Fraction(x_cents * quantity2, n)
        check(cross_mult_variable.denominator == 1, "medium extract total must land on an exact cent")
        check_equivalent(Fraction(fixed_cents) + cross_mult_variable, total_cents, "medium extract-rate total mismatch")

        distractors = pick_distractors(answer_opt, distractor_opts, n=3)
        check(len(distractors) == 3, "expected exactly 3 surviving medium-extract distractors")

        rate_text = _money(rate_cents)
        if kind == "rate_plus_fixed":
            working = [
                f"Rate: ${x_text} / {n} {unit_plural} = ${rate_text} per {unit}; x {quantity2} = "
                f"${_money(variable_cents)}; + delivery ${_money(fixed_cents)} = ${_money(total_cents)}. "
                f"Not directly proportional (a fixed delivery fee applies)."
            ]
        else:
            working = [
                f"Rate: ${x_text} / {n} {unit_plural} = ${rate_text} per {unit}; x {quantity2} = "
                f"${_money(total_cents)}. Directly proportional (a pure per-unit rate)."
            ]

        return dict(
            text=text,
            answer=answer_opt,
            answer_text=answer_opt,
            answer_latex=latex_answer(answer_opt, plain=True),
            working=working,
            distractors=distractors,
        )
    raise ValueError("p13a_direct_proportion_scale/medium(extract): no valid draw found")


def _medium_inverse_question(rng) -> dict:
    """The rate is given explicitly, but the question runs BACKWARDS: given
    a total, find the quantity -- the inverse of easy's own forward
    direction. On a ``rate_plus_fixed`` draw a one-off delivery fee is also
    stated, and must be subtracted before dividing by the rate. The answer
    is a plain whole-number quantity, baked into the mcq option text.

    Construction-based no-collision guarantee for the "omits-fixed-fee"
    distractor (rounds `total / rate` without first removing the fee):
    every `_FIXED_FEES_CENTS` value is >= $5.00 and every `_UNIT_RATES_CENTS`
    value is <= $4.50, so `fixed_cents / rate_cents > 1.11` always -- the
    omitted-fee distractor is always at least ~1.11 units away from the
    correct quantity, well clear of the 0.5 half-up rounding boundary, so it
    can never round to the same integer as the correct quantity."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        item, unit, unit_plural = rng.choice(_RATE_ITEMS)
        rate_cents = rng.choice(_UNIT_RATES_CENTS)
        quantity_answer = rng.randint(2, 20)
        kind = rng.choice(("pure_rate", "rate_plus_fixed"))
        fixed_cents = 0 if kind == "pure_rate" else rng.choice(_FIXED_FEES_CENTS)
        total_cents = Fraction(fixed_cents + rate_cents * quantity_answer)

        # decimal-shift, applied to the quantity instead of a money value --
        # an arithmetic-only slip, so it keeps the SAME classification label
        # as the correct answer at every draw.
        shift_int = int(quantity_answer * 10)
        correct_label = _LABEL_NO if kind == "rate_plus_fixed" else _LABEL_YES

        if kind == "pure_rate":
            # inverts-the-rate: multiplies instead of divides.
            inverts_rate_raw = Fraction(total_cents * rate_cents, 10000)
            inverts_rate_int = int(round_half_up(inverts_rate_raw))
            # reports the per-unit rate itself, ignoring the total entirely
            # -- paired with "No" (a flat-charge reading), the same
            # reasoning as easy/hard's "ignores-quantity" pairing.
            rate_as_answer_int = int(round_half_up(Fraction(rate_cents, 100)))

            answer_opt = _qty_opt(correct_label, str(quantity_answer), unit, unit_plural)
            distractor_opts = [
                _qty_opt(_LABEL_YES, str(inverts_rate_int), unit, unit_plural),
                _qty_opt(_LABEL_YES, str(shift_int), unit, unit_plural),
                _qty_opt(_LABEL_NO, str(rate_as_answer_int), unit, unit_plural),
            ]
            candidates_int = (inverts_rate_int, shift_int, rate_as_answer_int)
        else:
            # omits-the-fixed-fee: divides the WHOLE total by the rate,
            # never subtracting the fee first -- paired with "Yes" (the
            # textbook direct-proportion-vs-not error).
            omits_fixed_int = int(round_half_up(Fraction(total_cents, rate_cents)))
            # applies-the-fixed-fee-per-unit: divides by (rate + fixed)
            # instead of removing the fee once and dividing by rate alone
            # -- paired with "No" (aware of the fee, wrong model).
            applies_fixed_int = int(round_half_up(Fraction(total_cents, rate_cents + fixed_cents)))

            answer_opt = _qty_opt(correct_label, str(quantity_answer), unit, unit_plural)
            distractor_opts = [
                _qty_opt(_LABEL_YES, str(omits_fixed_int), unit, unit_plural),
                _qty_opt(_LABEL_NO, str(applies_fixed_int), unit, unit_plural),
                _qty_opt(_LABEL_NO, str(shift_int), unit, unit_plural),
            ]
            candidates_int = (omits_fixed_int, applies_fixed_int, shift_int)

        try:
            check_pairwise_distinct(
                quantity_answer,
                *candidates_int,
                msg="medium-inverse candidates must be pairwise distinct",
            )
            check(len({answer_opt, *distractor_opts}) == 4, "medium-inverse mcq options must be pairwise distinct")
        except Exception:
            continue

        rate_text = _money(rate_cents)
        total_text = _money(total_cents)
        item_cap = item[0].upper() + item[1:]
        if kind == "rate_plus_fixed":
            fixed_text = _money(fixed_cents)
            stem = rng.choice(
                (
                    f"{item_cap} costs ${rate_text} per {unit}, plus a one-off ${fixed_text} delivery fee "
                    f"for the whole order.",
                    f"{item_cap} is priced at ${rate_text} per {unit}, with a one-off ${fixed_text} delivery "
                    f"fee added to any order.",
                )
            )
            tail = f"how many {unit_plural} can you buy if the total spent, including delivery, is ${total_text}?"
        else:
            stem = rng.choice(
                (
                    f"{item_cap} costs ${rate_text} per {unit}.",
                    f"{item_cap} is priced at ${rate_text} per {unit}.",
                )
            )
            tail = f"how many {unit_plural} can you buy for ${total_text}?"
        text = _classification_question(stem, tail)

        # Independent check: repeated subtraction route -- how many times
        # does the rate come out of the total ONCE THE FEE IS REMOVED --
        # never a division of the same two Fraction objects the
        # construction used. Reduces to the original pure-rate check when
        # fixed_cents == 0.
        remaining = total_cents - fixed_cents
        count = 0
        while remaining > 0:
            remaining -= rate_cents
            count += 1
        check(
            remaining == 0, "medium inverse: total (after removing any fixed fee) is not an exact multiple of the rate"
        )
        check_equivalent(count, quantity_answer, "medium inverse repeated-subtraction mismatch")

        distractors = pick_distractors(answer_opt, distractor_opts, n=3, tolerance=0)
        check(len(distractors) == 3, "expected exactly 3 surviving medium-inverse distractors")

        if kind == "rate_plus_fixed":
            working = [
                f"(${total_text} - ${fixed_text} delivery) / ${rate_text} per {unit} = {quantity_answer} "
                f"{unit_plural}. Not directly proportional (a fixed delivery fee applies)."
            ]
        else:
            working = [
                f"${total_text} / ${rate_text} per {unit} = {quantity_answer} {unit_plural}. "
                f"Directly proportional (a pure per-unit rate)."
            ]

        return dict(
            text=text,
            answer=answer_opt,
            answer_text=answer_opt,
            answer_latex=latex_answer(answer_opt, plain=True),
            working=working,
            distractors=distractors,
        )
    raise ValueError("p13a_direct_proportion_scale/medium(inverse): no valid draw found")


def _medium_question(rng) -> dict:
    variant = rng.choice(("extract_forward", "inverse"))
    if variant == "extract_forward":
        return _medium_extract_question(rng)
    return _medium_inverse_question(rng)


# ---------------------------------------------------------------------------
# Hard: BOTH pure_rate and rate_plus_fixed now drawn, using
# the same larger-rate scenario families both ways.
# ---------------------------------------------------------------------------


def _hard_question(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        rate_cents = rng.choice(_HARD_RATE_CENTS)
        quantity = rng.randint(2, 15)
        kind = rng.choice(("pure_rate", "rate_plus_fixed"))

        if kind == "rate_plus_fixed":
            fixed_cents = rng.choice(_FIXED_FEES_CENTS)
            stem_fn, tail_fn, unit, unit_plural = rng.choice(_HARD_TEMPLATES)
            rate_text = _money(rate_cents)
            fixed_text = _money(fixed_cents)
            stem = stem_fn(fixed_text, rate_text)
            total_cents = Fraction(fixed_cents + rate_cents * quantity)

            # omits-the-fixed-fee: the textbook direct-proportion-vs-not
            # error this tier exists to exercise. Paired with "Yes".
            omits_fixed_cents = Fraction(rate_cents * quantity)
            # applies-the-fixed-fee-per-unit: treats the fixed component as
            # if it recurred every unit instead of being a one-off. Paired
            # with "No" (aware of the fee, wrong model).
            applies_fixed_per_unit_cents = Fraction((fixed_cents + rate_cents) * quantity)
            # inverts-the-rate: divides instead of multiplies (both kinds),
            # still correctly adding the fixed component once. Paired with
            # "No" (arithmetic-only slip).
            inverts_rate_cents = Fraction(fixed_cents) + Fraction(quantity * 10000, rate_cents)

            answer_opt = _money_opt(_LABEL_NO, _money(total_cents))
            distractor_opts = [
                _money_opt(_LABEL_YES, _money(omits_fixed_cents)),
                _money_opt(_LABEL_NO, _money(applies_fixed_per_unit_cents)),
                _money_opt(_LABEL_NO, _money_rounded(inverts_rate_cents)),
            ]
            candidates_cents = (omits_fixed_cents, applies_fixed_per_unit_cents, inverts_rate_cents)
            expected_total = Fraction(fixed_cents) + Fraction(sum(rate_cents for _ in range(quantity)))
        else:
            fixed_cents = 0
            rate_text = _money(rate_cents)
            stem_fn, tail_fn, unit, unit_plural = rng.choice(_HARD_TEMPLATES_PURE)
            stem = stem_fn(rate_text)
            total_cents = Fraction(rate_cents * quantity)

            # inverts-the-rate: divides instead of multiplies. Arithmetic
            # slip, keeps the correct "Yes" label.
            inverts_rate_cents = Fraction(quantity * 10000, rate_cents)
            # decimal-shift: a factor of ten too large. Arithmetic slip,
            # keeps the correct "Yes" label.
            decimal_shift_cents = total_cents * 10
            # ignores-quantity: reports the per-unit rate itself as the
            # total -- paired with "No" (a flat-charge reading).
            rate_as_total_cents = Fraction(rate_cents)

            answer_opt = _money_opt(_LABEL_YES, _money(total_cents))
            distractor_opts = [
                _money_opt(_LABEL_YES, _money_rounded(inverts_rate_cents)),
                _money_opt(_LABEL_YES, _money(decimal_shift_cents)),
                _money_opt(_LABEL_NO, _money(rate_as_total_cents)),
            ]
            candidates_cents = (inverts_rate_cents, decimal_shift_cents, rate_as_total_cents)
            expected_total = Fraction(sum(rate_cents for _ in range(quantity)))

        try:
            check_pairwise_distinct(
                total_cents, *candidates_cents, msg="hard-tier candidates must be pairwise distinct"
            )
            check(len({answer_opt, *distractor_opts}) == 4, "hard-tier mcq options must be pairwise distinct")
        except Exception:
            continue

        # Independent check: repeated-addition route for the variable
        # component, never the multiplication construction used.
        check_equivalent(expected_total, total_cents, "hard total mismatch")

        text = _classification_question(stem, tail_fn(quantity))
        distractors = pick_distractors(answer_opt, distractor_opts, n=3)
        check(len(distractors) == 3, "expected exactly 3 surviving hard-tier distractors")

        if kind == "rate_plus_fixed":
            working = [
                f"Fixed ${fixed_text} + {quantity} × ${rate_text} = ${_money(total_cents)}. "
                f"Not directly proportional (a fixed fee applies)."
            ]
        else:
            working = [
                f"{quantity} × ${rate_text} = ${_money(total_cents)}. Directly proportional (a pure per-unit rate)."
            ]

        return dict(
            text=text,
            answer=answer_opt,
            answer_text=answer_opt,
            answer_latex=latex_answer(answer_opt, plain=True),
            working=working,
            distractors=distractors,
        )
    raise ValueError("p13a_direct_proportion_scale/hard: no valid draw found")


# ---------------------------------------------------------------------------
# Exam: two stated data points, no formula named -- test proportionality
# first, then solve.
# ---------------------------------------------------------------------------


def _solve_two_point(q1: int, t1: Fraction, q2: int, t2: Fraction) -> tuple[Fraction, Fraction]:
    """Solve the 2x2 linear system total(q) = fixed + rate*q for (rate,
    fixed) from two (quantity, total) data points -- the independent route
    required for independence: recovers the model's own parameters from the
    data, rather than replaying a `fixed + rate*quantity` formula forward."""
    check(q2 != q1, "exam data points must have distinct quantities")
    rate = Fraction(t2 - t1, q2 - q1)
    fixed = t1 - rate * q1
    return rate, fixed


def _exam_question(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        unit, unit_plural, subject = rng.choice(_EXAM_TEMPLATES)
        rate_cents = rng.choice(_HARD_RATE_CENTS)
        kind = rng.choice(("pure_rate", "rate_plus_fixed"))
        fixed_cents = 0 if kind == "pure_rate" else rng.choice(_FIXED_FEES_CENTS)

        q1, q2, q3 = rng.sample(range(2, 16), 3)
        q1, q2 = sorted((q1, q2))

        t1_cents = Fraction(fixed_cents + rate_cents * q1)
        t2_cents = Fraction(fixed_cents + rate_cents * q2)
        t3_cents = Fraction(fixed_cents + rate_cents * q3)

        # Independent check: solve the 2x2 system from the two DATA POINTS
        # and confirm it recovers construction's own (rate, fixed), then
        # forward-compute the third total from THAT solved pair -- never
        # simply replaying construction's own fixed + rate*quantity formula
        # on the same variables.
        solved_rate, solved_fixed = _solve_two_point(q1, t1_cents, q2, t2_cents)
        check_equivalent(solved_rate, Fraction(rate_cents), "exam rate re-derivation mismatch")
        check_equivalent(solved_fixed, Fraction(fixed_cents), "exam fixed re-derivation mismatch")
        t3_check = solved_fixed + solved_rate * q3
        check_equivalent(t3_check, t3_cents, "exam third-total re-derivation mismatch")

        # inverts-the-rate: divides instead of multiplies for the rate
        # component (both kinds), still correctly adding the fixed part.
        inverts_rate_cents = Fraction(fixed_cents) + Fraction(q3 * 10000, rate_cents)
        decimal_shift_cents = t3_cents * 10

        if kind == "rate_plus_fixed":
            # wrong-data-point-ratio: uses total1/quantity1 as "the" rate
            # even though the two stated ratios disagree -- paired with
            # "Yes" (a learner blindly extrapolating one ratio is
            # implicitly assuming the situation is proportional).
            wrong_data_point_cents = Fraction(t1_cents, q1) * q3
            answer_opt = _money_opt(_LABEL_NO, _money(t3_cents))
            distractor_opts = [
                _money_opt(_LABEL_YES, _money_rounded(wrong_data_point_cents)),
                _money_opt(_LABEL_NO, _money_rounded(inverts_rate_cents)),
                _money_opt(_LABEL_NO, _money(decimal_shift_cents)),
            ]
            candidates_cents = (wrong_data_point_cents, inverts_rate_cents, decimal_shift_cents)
        else:
            # reports-a-given-total: reports one of the two stated totals
            # directly, ignoring that a NEW quantity was asked for --
            # paired with "No" (a flat-charge/flat-price reading).
            reports_given_total_cents = t2_cents
            answer_opt = _money_opt(_LABEL_YES, _money(t3_cents))
            distractor_opts = [
                _money_opt(_LABEL_NO, _money_rounded(reports_given_total_cents)),
                _money_opt(_LABEL_YES, _money_rounded(inverts_rate_cents)),
                _money_opt(_LABEL_YES, _money(decimal_shift_cents)),
            ]
            candidates_cents = (reports_given_total_cents, inverts_rate_cents, decimal_shift_cents)

        try:
            check_pairwise_distinct(t3_cents, *candidates_cents, msg="exam-tier candidates must be pairwise distinct")
            check(len({answer_opt, *distractor_opts}) == 4, "exam-tier mcq options must be pairwise distinct")
        except Exception:
            continue

        t1_text, t2_text = _money(t1_cents), _money(t2_cents)
        stem = (
            f"{subject}: for {q1} {unit_plural} the total is ${t1_text}; for {q2} {unit_plural} the total is "
            f"${t2_text}."
        )
        tail = f"what is the total for {q3} {unit_plural}?"
        text = _classification_question(stem, tail)

        distractors = pick_distractors(answer_opt, distractor_opts, n=3)
        check(len(distractors) == 3, "expected exactly 3 surviving exam-tier distractors")

        label_word = "Not directly proportional" if kind == "rate_plus_fixed" else "Directly proportional"
        working = [
            f"Rate ${_money(rate_cents)} per {unit}, fixed ${_money(fixed_cents)}: total for {q3} {unit_plural} "
            f"is ${_money(t3_cents)}. {label_word}."
        ]

        return dict(
            text=text,
            answer=answer_opt,
            answer_text=answer_opt,
            answer_latex=latex_answer(answer_opt, plain=True),
            working=working,
            distractors=distractors,
        )
    raise ValueError("p13a_direct_proportion_scale/exam: no valid draw found")


@generator(
    "p13a_direct_proportion_scale",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13a_direct_proportion_scale(rng, tier: str = "easy") -> dict:
    """Determine whether a stated ratio/rate/scale context is directly
    proportional (a pure per-unit rate) or NOT (a per-unit rate plus a
    fixed component) -- asked and graded explicitly as a single mcq option
    bundling the classification with the computed value -- and
    compute the requested total/quantity. easy: pure-rate only (disclosed
    baseline), rate stated explicitly, forward direction. medium: both
    kinds now drawn, on either the extract-from-a-ratio or the inverse-
    direction reasoning step. hard: both kinds now drawn, using the larger
    rate pool and call-out/monthly-plan/rental-fee scenario families. exam:
    two stated data points, no formula named -- test whether the two
    implied rates agree (pure-rate) or disagree (rate-plus-fixed, solve for
    rate and fixed) before finding a third total."""
    if tier == "easy":
        return _easy_question(rng)
    if tier == "medium":
        return _medium_question(rng)
    if tier == "hard":
        return _hard_question(rng)
    if tier == "exam":
        return _exam_question(rng)
    raise ValueError(f"unknown tier: {tier!r}")
