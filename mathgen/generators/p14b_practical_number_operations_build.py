"""Generates `senior-solve-practical-problems-requiring-basic-number-operations` questions (ACMEM001): practical
problems that need basic arithmetic.

**The core (the overlap with neighbouring skills).** ACMEM001's descriptor is the widest in the library
("solve practical problems requiring basic number operations"). The two nearest
neighbours are ``f10-additive-multiplicative-money-model`` and ``percentage-discount`` (a single
discount step, or two CHAINED discount steps at its hard tier, WHOLE-DOLLAR prices, never combined with a
non-percentage step). The kernel that separates this skill from both: a CENTS-PRECISION money problem (never
whole dollars) that chains EXACTLY ONE percentage-based step (a discount or a GST/service-charge addition)
together with a genuinely non-percentage operation (a flat add/subtract, a per-item multiply, or an
equal-split divide) to reach a final cents-precision total.

Deliberately never a two-percentage-only chain (that shape reads as
``percentage-discount``'s own chained-discount hard
tier): every tier here carries exactly one percentage step, chained with one
or two non-percentage steps instead of a second percentage step. The hard
tier's worked example ("GST, then a flat delivery fee
added, then divided by quantity") is a single-percentage three-step chain,
and that is the shape implemented here.

**Exactness discipline.** Every intermediate value stays an exact integer
number of cents -- never rounded -- so the final cents total is never a
mid-step rounding call. This is achieved by construction, not by retrying: a
price is drawn as ``k * d`` where ``d = 100 // gcd(100, pct)``, which makes
``price * pct`` divisible by 100 for ANY integer ``k`` (`_draw_price_cents`
below); a downstream equal-split divisor is folded into the same draw via an
extra step multiplier so the post-percentage total is ALSO an exact multiple
of the number of people (`_draw_price_cents` again, via `extra_divisor`); a
flat amount added/subtracted after an exact-split point is drawn as a
multiple of that same divisor so the running total stays divisible by it.
Multiplying an exact cents value by an integer quantity, or adding/
subtracting another exact cents value, never introduces a fraction, so no
other step needs special treatment. The in-generator independent check
(``check_equivalent``) recomputes the whole chain via the algebraically
equivalent "final = base * (100 (+/-) pct) / 100 [chained op]" one-line
Fraction expression, never reusing the step-by-step integer variables
construction accumulated.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    article_for_number,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
    stated_precision_tolerance,
)

_PCT_SET = [5, 10, 15, 20, 25, 30, 40, 50]

# GST is a fixed Australian rate -- never drawn from a set. A service charge
# or surcharge (café/restaurant/import) is drawn from a realistic Australian
# suggested cap; kept at the full suggested width, not narrowed further).
_GST_PCT = 10
_SERVICE_PCT_LO, _SERVICE_PCT_HI = 5, 15


def _draw_service_pct(rng) -> int:
    return rng.randint(_SERVICE_PCT_LO, _SERVICE_PCT_HI)


_DISCOUNT_ITEMS = ["jacket", "pair of shoes", "backpack", "board game", "bicycle helmet", "tennis racquet"]
_GST_CONTEXTS = ["café bill", "restaurant bill", "hardware order", "plumbing invoice", "catering order"]
_DELIVERY_ITEMS = ["furniture order", "online order", "hardware delivery", "appliance order"]
_VOUCHER_ITEMS = ["loyalty voucher", "gift card", "membership discount", "coupon"]

_VOWEL_SOUND_STARTS = ("a", "e", "i", "o", "u")


def _article(word: str) -> str:
    """ "A" or "An" for `word` -- only ever called on this module's own fixed
    item/context word lists, none of which starts with a silent-h or a
    consonant-sounding vowel letter (no "hour", no "university"), so a plain
    first-letter check is exact for every value actually reachable here."""
    return "An" if word[0].lower() in _VOWEL_SOUND_STARTS else "A"


def _pct_factor(pct: int, op: str) -> tuple[int, int]:
    """``(g, factor)`` where ``g = gcd(100, pct)`` and ``factor = 100 +/- pct``
    (``op`` is ``"add"`` for a GST/service-charge increase, ``"sub"`` for a
    discount). ``factor`` is always divisible by ``g`` (both ``100`` and
    ``pct`` are), which is what makes `_draw_price_cents` exact below."""
    g = math.gcd(100, pct)
    factor = 100 + pct if op == "add" else 100 - pct
    return g, factor


def _draw_price_cents(rng, pct: int, op: str, lo_cents: int, hi_cents: int, extra_divisor: int = 1):
    """Draw ``(price_cents, after_pct_cents)`` in ``[lo_cents, hi_cents]``
    such that ``after_pct_cents = price_cents * factor / 100`` is an EXACT
    integer, and additionally divisible by ``extra_divisor`` (pass the number
    of people an equal split will divide by, so the post-percentage running
    total is already exact for that split; ``1`` means no extra constraint).

    ``price_cents = k * d`` where ``d = 100 // g`` makes ``price_cents * pct``
    divisible by 100 for every integer ``k`` (``g = gcd(100, pct)``, so ``d *
    pct`` is a multiple of 100). ``after_pct_cents = k * m`` where ``m =
    factor // g`` (an integer -- both ``100`` and ``pct`` are divisible by
    ``g``, so their sum/difference ``factor`` is too). Requiring
    ``k * m`` divisible by ``extra_divisor`` is then just requiring ``k`` be a
    multiple of ``extra_divisor // gcd(m, extra_divisor)``.
    """
    g, factor = _pct_factor(pct, op)
    d = 100 // g
    m = factor // g
    step = extra_divisor // math.gcd(m, extra_divisor)
    lo_k = -(-lo_cents // d)
    hi_k = hi_cents // d
    lo_t = -(-lo_k // step)
    hi_t = hi_k // step
    check(hi_t >= lo_t, "price draw range too narrow for exact divisibility")
    t = rng.randint(lo_t, hi_t)
    k = t * step
    price_cents = k * d
    after_pct_cents = k * m
    return price_cents, after_pct_cents


def _wrong_direction_value(cents: int, pct: int, op: str) -> int:
    """The value applying the percentage step's OPPOSITE direction to
    ``cents`` would have produced -- e.g. adding a discount instead of
    subtracting it. Exact (see `_draw_price_cents`'s docstring: `factor` is
    always divisible by `g`, and `cents * factor` is therefore always an
    exact multiple of 100 whenever `cents` itself is a multiple of
    `100 // g`, which every value passed here always is by construction)."""
    other_op = "add" if op == "sub" else "sub"
    g, other_factor = _pct_factor(pct, other_op)
    return cents * other_factor // g // (100 // g)


def _round_up_divide(value: int, n: int) -> int:
    """`value // n`, rounded UP when it does not divide exactly -- used only
    to turn a distractor candidate into a plausible per-person dollar figure,
    never the graded answer (which is always drawn to divide exactly)."""
    return value // n if value % n == 0 else value // n + 1


def _deconflict_cents(answer: int, *candidates: int) -> list[int]:
    """Nudge each `candidates` value by whole cents (deterministically, no
    extra randomness) until it differs from `answer` and from every earlier
    (already-deconflicted) candidate. Several of this module's misconception
    formulas are ALGEBRAICALLY IDENTICAL to another option for a particular
    (pct, n) combination -- e.g. at pct=50, n=3, "add 50% then split 3 ways"
    (`price*150/300`) collapses to exactly `price*0.5`, the same value as
    "subtract 50% and never divide" -- a real coincidence, not a construction
    bug, and one `check_pairwise_distinct` would otherwise reject on those
    seeds. A same-value misconception is not a usable distractor either way,
    so nudging it a cent off keeps it a plausible-but-wrong nearby figure
    rather than reworking the whole misconception design around finitely many
    coincidental (pct, n) pairs."""
    seen = {answer}
    out = []
    for c in candidates:
        while c in seen:
            c += 1
        seen.add(c)
        out.append(c)
    return out


def _money(cents: int) -> str:
    """Bare decimal string ("64.50") for `answer_text`/a distractor -- no
    currency symbol, matching every other `answer: number` money skill in
    this repo (`f10-additive-multiplicative-money-model`,
    `percentage-discount`)."""
    return stated_precision_string(Fraction(cents, 100), 2)


def _dollar(cents: int) -> str:
    """`$`-prefixed display string ("$64.50") for the learner-facing prompt
    text and `working` -- never for `answer_text`/a distractor (see `_money`
    above)."""
    return f"${_money(cents)}"


def _independent_recompute(price_cents: int, pct: int, op: str) -> Fraction:
    """The ``after_pct`` value as ONE Fraction expression -- ``price *
    (100 +/- pct) / 100`` -- computed independently of however
    `_draw_price_cents` built its own integer ``k``/``m`` bookkeeping."""
    _, factor = _pct_factor(pct, op)
    return Fraction(price_cents) * factor / 100


# ---------------------------------------------------------------------------
# easy: one percentage step (discount or GST/service-charge addition) + one
# flat add/subtract step, cents-precision throughout.
# ---------------------------------------------------------------------------


def _build_easy(rng) -> dict:
    pct_op = rng.choice(["sub", "add"])  # sub = discount, add = service charge
    # add here always renders as "service charge" (never the literal word
    # "GST"), so it draws from the realistic service-charge range, not the
    # discount set.
    pct = rng.choice(_PCT_SET) if pct_op == "sub" else _draw_service_pct(rng)
    price_cents, after_pct = _draw_price_cents(rng, pct, pct_op, 1500, 9999)
    flat_op = rng.choice(["add", "sub"])
    if flat_op == "add":
        flat_cents = rng.randint(100, 2000)
        final_cents = after_pct + flat_cents
    else:
        flat_cents = rng.randint(100, min(2000, after_pct - 100))
        final_cents = after_pct - flat_cents

    item = rng.choice(_DISCOUNT_ITEMS)
    voucher = rng.choice(_VOUCHER_ITEMS)
    if pct_op == "sub":
        pct_clause = f"A {item} costs {_dollar(price_cents)}. A {pct}% discount is applied"
    else:
        context = rng.choice(_GST_CONTEXTS)
        pct_clause = f"A {context} of {_dollar(price_cents)} has {article_for_number(pct)} {pct}% service charge added"
    if flat_op == "add":
        flat_clause = f", then {article_for_number(_dollar(flat_cents))} {_dollar(flat_cents)} delivery fee is added"
    else:
        flat_clause = f", then {article_for_number(_dollar(flat_cents))} {_dollar(flat_cents)} {voucher} is subtracted"
    text = f"{pct_clause}{flat_clause}. What is the final amount to pay?"

    # Independent check: one Fraction expression from the STATED inputs,
    # never reusing the integer bookkeeping `_draw_price_cents` built.
    recomputed_after_pct = _independent_recompute(price_cents, pct, pct_op)
    check_equivalent(after_pct, recomputed_after_pct, "easy: percentage-step recompute mismatch")
    recomputed_final = recomputed_after_pct + flat_cents if flat_op == "add" else recomputed_after_pct - flat_cents
    check_equivalent(final_cents, recomputed_final, "easy: final-total recompute mismatch")

    # Misconceptions:
    #  - wrong direction: applies the percentage adjustment the OTHER way
    #    (adds a discount instead of subtracting it, or vice versa);
    #  - drops the flat step: reports only the percentage step's own result;
    #  - drops the percentage step: applies only the flat step to the base price.
    wrong_direction_after_pct = _wrong_direction_value(price_cents, pct, pct_op)
    wrong_direction = (
        wrong_direction_after_pct + flat_cents if flat_op == "add" else wrong_direction_after_pct - flat_cents
    )
    dropped_flat_step = after_pct
    dropped_pct_step = price_cents + flat_cents if flat_op == "add" else price_cents - flat_cents
    wrong_direction, dropped_flat_step, dropped_pct_step = _deconflict_cents(
        final_cents, wrong_direction, dropped_flat_step, dropped_pct_step
    )

    check_pairwise_distinct(
        final_cents,
        wrong_direction,
        dropped_flat_step,
        dropped_pct_step,
        msg="easy misconception collides with the answer or another option",
    )

    answer_text = _money(final_cents)
    working = [
        f"Percentage step: {_dollar(price_cents)} {'-' if pct_op == 'sub' else '+'} {pct}% = {_dollar(after_pct)}.",
        f"Flat step: {_dollar(after_pct)} {'+' if flat_op == 'add' else '-'} {_dollar(flat_cents)} = {_dollar(final_cents)}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(Fraction(final_cents, 100)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(
            answer_text,
            [_money(wrong_direction), _money(dropped_flat_step), _money(dropped_pct_step)],
            tolerance=stated_precision_tolerance(2),
        ),
    )


# ---------------------------------------------------------------------------
# medium: one percentage step + a multiplication or division step.
# ---------------------------------------------------------------------------


def _build_medium_split(rng) -> dict:
    """Split a discounted-or-GST-inclusive bill equally among n people."""
    n = rng.randint(2, 6)
    pct_op = rng.choice(["sub", "add"])
    # add here renders as "service charge" (never the literal word "GST"),
    # so it draws from the realistic service-charge range.
    pct = rng.choice(_PCT_SET) if pct_op == "sub" else _draw_service_pct(rng)
    price_cents, after_pct = _draw_price_cents(rng, pct, pct_op, 3000, 19999, extra_divisor=n)
    per_person = after_pct // n
    context = rng.choice(_GST_CONTEXTS)
    if pct_op == "sub":
        text = (
            f"A {context} of {_dollar(price_cents)} has {article_for_number(pct)} {pct}% loyalty discount applied, then is "
            f"split equally among {n} people. How much does each person pay?"
        )
    else:
        text = (
            f"A {context} of {_dollar(price_cents)} has {article_for_number(pct)} {pct}% service charge added, then is "
            f"split equally among {n} people. How much does each person pay?"
        )

    recomputed_after_pct = _independent_recompute(price_cents, pct, pct_op)
    check_equivalent(after_pct, recomputed_after_pct, "medium split: percentage-step recompute mismatch")
    check(recomputed_after_pct % n == 0, "medium split: post-percentage total is not an exact multiple of n")
    check_equivalent(per_person, recomputed_after_pct / n, "medium split: split recompute mismatch")

    wrong_direction_after_pct = _wrong_direction_value(price_cents, pct, pct_op)
    misc_wrong_direction = _round_up_divide(wrong_direction_after_pct, n)
    misc_dropped_split = after_pct  # forgot to divide at all
    misc_used_base_price = _round_up_divide(price_cents, n)  # ignored the pct step
    misc_wrong_direction, misc_dropped_split, misc_used_base_price = _deconflict_cents(
        per_person, misc_wrong_direction, misc_dropped_split, misc_used_base_price
    )

    check_pairwise_distinct(
        per_person,
        misc_wrong_direction,
        misc_dropped_split,
        misc_used_base_price,
        msg="medium split misconception collides with the answer or another option",
    )

    answer_text = _money(per_person)
    working = [
        f"Percentage step: {_dollar(price_cents)} {'-' if pct_op == 'sub' else '+'} {pct}% = {_dollar(after_pct)}.",
        f"Split step: {_dollar(after_pct)} / {n} = {_dollar(per_person)} each.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(Fraction(per_person, 100)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(
            answer_text,
            [_money(misc_wrong_direction), _money(misc_dropped_split), _money(misc_used_base_price)],
            tolerance=stated_precision_tolerance(2),
        ),
    )


def _build_medium_multiply(rng) -> dict:
    """Multiply a per-item discounted/GST-inclusive price by a quantity bought."""
    qty = rng.randint(2, 8)
    pct_op = rng.choice(["sub", "add"])
    # add here renders as "import surcharge" (never the literal word "GST"),
    # so it draws from the realistic service-charge/surcharge range.
    pct = rng.choice(_PCT_SET) if pct_op == "sub" else _draw_service_pct(rng)
    unit_price_cents, unit_after_pct = _draw_price_cents(rng, pct, pct_op, 300, 2500)
    total_cents = unit_after_pct * qty
    item = rng.choice(_DISCOUNT_ITEMS)
    if pct_op == "sub":
        text = (
            f"A {item} normally costs {_dollar(unit_price_cents)} but is discounted by {pct}%. "
            f"What is the total cost of buying {qty} of them at the discounted price?"
        )
    else:
        text = (
            f"A {item} costs {_dollar(unit_price_cents)} before {article_for_number(pct)} {pct}% import surcharge is added. "
            f"What is the total cost of buying {qty} of them, surcharge included?"
        )

    recomputed_unit_after_pct = _independent_recompute(unit_price_cents, pct, pct_op)
    check_equivalent(unit_after_pct, recomputed_unit_after_pct, "medium multiply: percentage-step recompute mismatch")
    check_equivalent(total_cents, recomputed_unit_after_pct * qty, "medium multiply: total recompute mismatch")

    wrong_direction_unit = _wrong_direction_value(unit_price_cents, pct, pct_op)
    misc_wrong_direction = wrong_direction_unit * qty
    misc_dropped_multiply = unit_after_pct  # forgot to multiply by the quantity
    misc_used_base_price = unit_price_cents * qty  # ignored the pct step entirely
    misc_wrong_direction, misc_dropped_multiply, misc_used_base_price = _deconflict_cents(
        total_cents, misc_wrong_direction, misc_dropped_multiply, misc_used_base_price
    )

    check_pairwise_distinct(
        total_cents,
        misc_wrong_direction,
        misc_dropped_multiply,
        misc_used_base_price,
        msg="medium multiply misconception collides with the answer or another option",
    )

    answer_text = _money(total_cents)
    working = [
        f"Percentage step: {_dollar(unit_price_cents)} {'-' if pct_op == 'sub' else '+'} {pct}% = {_dollar(unit_after_pct)} per item.",
        f"Multiply step: {_dollar(unit_after_pct)} x {qty} = {_dollar(total_cents)}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(Fraction(total_cents, 100)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(
            answer_text,
            [_money(misc_wrong_direction), _money(misc_dropped_multiply), _money(misc_used_base_price)],
            tolerance=stated_precision_tolerance(2),
        ),
    )


# ---------------------------------------------------------------------------
# hard: three-step chain -- percentage step + flat add/subtract + equal split.
# The alternate worked example ("GST, then a flat delivery
# fee added, then divided by quantity") is exactly this shape.
# ---------------------------------------------------------------------------


def _build_hard(rng) -> dict:
    n = rng.randint(2, 5)
    pct_op = rng.choice(["sub", "add"])
    # add here renders as the literal word "GST" -- Australian GST is always
    # exactly 10%, a hard constant, never drawn from a set.
    pct = rng.choice(_PCT_SET) if pct_op == "sub" else _GST_PCT
    price_cents, after_pct = _draw_price_cents(rng, pct, pct_op, 4000, 29999, extra_divisor=n)
    flat_op = rng.choice(["add", "sub"])
    flat_k = rng.randint(20, 300)  # cents-per-share contribution to the flat fee/credit
    flat_cents = flat_k * n  # kept a multiple of n so the running total stays exactly splittable
    combined = after_pct + flat_cents if flat_op == "add" else after_pct - flat_cents
    check(combined > 0, "hard: combined total must stay positive before splitting")
    per_person = combined // n

    item = rng.choice(_DELIVERY_ITEMS)
    if pct_op == "sub":
        pct_clause = f"{_article(item)} {item} totalling {_dollar(price_cents)} has {article_for_number(pct)} {pct}% loyalty discount applied"
    else:
        pct_clause = f"{_article(item)} {item} totalling {_dollar(price_cents)} has {pct}% GST added"
    if flat_op == "add":
        flat_clause = f", then a flat {_dollar(flat_cents)} delivery fee is added"
    else:
        flat_clause = (
            f", then {article_for_number(_dollar(flat_cents))} {_dollar(flat_cents)} loyalty credit is subtracted"
        )
    text = (
        f"{pct_clause}{flat_clause}. The total is then split equally among {n} people. How much does each person pay?"
    )

    recomputed_after_pct = _independent_recompute(price_cents, pct, pct_op)
    check_equivalent(after_pct, recomputed_after_pct, "hard: percentage-step recompute mismatch")
    recomputed_combined = recomputed_after_pct + flat_cents if flat_op == "add" else recomputed_after_pct - flat_cents
    check_equivalent(combined, recomputed_combined, "hard: combined-total recompute mismatch")
    check(recomputed_combined % n == 0, "hard: combined total is not an exact multiple of n")
    check_equivalent(per_person, recomputed_combined / n, "hard: split recompute mismatch")

    wrong_direction_after_pct = _wrong_direction_value(price_cents, pct, pct_op)
    wrong_direction_combined = (
        wrong_direction_after_pct + flat_cents if flat_op == "add" else wrong_direction_after_pct - flat_cents
    )
    misc_wrong_direction = _round_up_divide(wrong_direction_combined, n)
    misc_dropped_flat_step = after_pct // n  # ignored the flat step (after_pct always exact / n by construction)
    misc_dropped_split = combined  # forgot to divide at all
    misc_wrong_direction, misc_dropped_flat_step, misc_dropped_split = _deconflict_cents(
        per_person, misc_wrong_direction, misc_dropped_flat_step, misc_dropped_split
    )

    check_pairwise_distinct(
        per_person,
        misc_wrong_direction,
        misc_dropped_flat_step,
        misc_dropped_split,
        msg="hard misconception collides with the answer or another option",
    )

    answer_text = _money(per_person)
    working = [
        f"Percentage step: {_dollar(price_cents)} {'-' if pct_op == 'sub' else '+'} {pct}% = {_dollar(after_pct)}.",
        f"Flat step: {_dollar(after_pct)} {'+' if flat_op == 'add' else '-'} {_dollar(flat_cents)} = {_dollar(combined)}.",
        f"Split step: {_dollar(combined)} / {n} = {_dollar(per_person)} each.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(Fraction(per_person, 100)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(
            answer_text,
            [_money(misc_wrong_direction), _money(misc_dropped_flat_step), _money(misc_dropped_split)],
            tolerance=stated_precision_tolerance(2),
        ),
    )


# ---------------------------------------------------------------------------
# exam: the hard three-step chain, plus one stated-but-irrelevant quantity
# (a red herring) the learner must recognise plays no role in the
# calculation -- the "which numbers and which operations apply" modelling
# ---------------------------------------------------------------------------

_RED_HERRINGS = [
    "the order was placed on a Tuesday",
    "the group had booked the table for {n2} people the previous week",
    "the delivery driver made {n2} other stops that day",
    "the loyalty program has {n2} tiers",
    "the shop had {n2} branches nationally",
]


def _build_exam(rng) -> dict:
    n = rng.randint(2, 5)
    pct_op = rng.choice(["sub", "add"])
    # add here renders as the literal word "GST" -- Australian GST is always
    # exactly 10%, a hard constant, never drawn from a set.
    pct = rng.choice(_PCT_SET) if pct_op == "sub" else _GST_PCT
    price_cents, after_pct = _draw_price_cents(rng, pct, pct_op, 4000, 29999, extra_divisor=n)
    flat_op = rng.choice(["add", "sub"])
    flat_k = rng.randint(20, 300)  # cents-per-share contribution to the flat fee/credit
    flat_cents = flat_k * n
    combined = after_pct + flat_cents if flat_op == "add" else after_pct - flat_cents
    check(combined > 0, "exam: combined total must stay positive before splitting")
    per_person = combined // n

    item = rng.choice(_DELIVERY_ITEMS)
    herring_template = rng.choice(_RED_HERRINGS)
    n2 = rng.randint(2, 40)
    while n2 == n:
        n2 = rng.randint(2, 40)
    herring = herring_template.format(n2=n2)

    if pct_op == "sub":
        pct_clause = f"{_article(item)} {item} totalling {_dollar(price_cents)} has {article_for_number(pct)} {pct}% loyalty discount applied"
    else:
        pct_clause = f"{_article(item)} {item} totalling {_dollar(price_cents)} has {pct}% GST added"
    if flat_op == "add":
        flat_clause = f", then a flat {_dollar(flat_cents)} delivery fee is added"
    else:
        flat_clause = (
            f", then {article_for_number(_dollar(flat_cents))} {_dollar(flat_cents)} loyalty credit is subtracted"
        )
    text = (
        f"{pct_clause}{flat_clause} ({herring}). The total is then split equally among {n} people. "
        f"How much does each person pay?"
    )

    recomputed_after_pct = _independent_recompute(price_cents, pct, pct_op)
    check_equivalent(after_pct, recomputed_after_pct, "exam: percentage-step recompute mismatch")
    recomputed_combined = recomputed_after_pct + flat_cents if flat_op == "add" else recomputed_after_pct - flat_cents
    check_equivalent(combined, recomputed_combined, "exam: combined-total recompute mismatch")
    check(recomputed_combined % n == 0, "exam: combined total is not an exact multiple of n")
    check_equivalent(per_person, recomputed_combined / n, "exam: split recompute mismatch")

    wrong_direction_after_pct = _wrong_direction_value(price_cents, pct, pct_op)
    wrong_direction_combined = (
        wrong_direction_after_pct + flat_cents if flat_op == "add" else wrong_direction_after_pct - flat_cents
    )
    misc_wrong_direction = _round_up_divide(wrong_direction_combined, n)
    misc_dropped_flat_step = after_pct // n
    # exam-specific misconception: treats the red-herring quantity as though
    # it were a genuine split divisor instead of `n`.
    misc_used_red_herring_as_divisor = _round_up_divide(combined, n2)
    misc_wrong_direction, misc_dropped_flat_step, misc_used_red_herring_as_divisor = _deconflict_cents(
        per_person, misc_wrong_direction, misc_dropped_flat_step, misc_used_red_herring_as_divisor
    )

    check_pairwise_distinct(
        per_person,
        misc_wrong_direction,
        misc_dropped_flat_step,
        misc_used_red_herring_as_divisor,
        msg="exam misconception collides with the answer or another option",
    )

    answer_text = _money(per_person)
    working = [
        f"The irrelevant quantity ({herring}) plays no role in the calculation.",
        f"Percentage step: {_dollar(price_cents)} {'-' if pct_op == 'sub' else '+'} {pct}% = {_dollar(after_pct)}.",
        f"Flat step: {_dollar(after_pct)} {'+' if flat_op == 'add' else '-'} {_dollar(flat_cents)} = {_dollar(combined)}.",
        f"Split step: {_dollar(combined)} / {n} = {_dollar(per_person)} each.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(Fraction(per_person, 100)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(
            answer_text,
            [_money(misc_wrong_direction), _money(misc_dropped_flat_step), _money(misc_used_red_herring_as_divisor)],
            tolerance=stated_precision_tolerance(2),
        ),
    )


_MEDIUM_BUILDERS = [_build_medium_split, _build_medium_multiply]


@generator(
    "p14b_practical_number_operations",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p14b_practical_number_operations(rng, tier="easy"):
    """Solve a practical cents-precision money problem chaining exactly one
    percentage-based step (a discount or a GST/service-charge addition) with
    a genuinely non-percentage operation -- the core of
    ``senior-solve-practical-problems-requiring-basic-number-operations``
    (ACMEM001). See the module docstring for the separation from
    ``f10-additive-multiplicative-money-model`` and ``percentage-discount``.
    """
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        builder = rng.choice(_MEDIUM_BUILDERS)
        return builder(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise KeyError(f"unknown tier {tier!r}")
