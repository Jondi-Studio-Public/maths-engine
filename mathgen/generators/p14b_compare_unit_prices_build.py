"""Generates `p14b_compare_unit_prices` questions: decide which of two packs is cheaper per unit, or whether they
are tied.

**The kernel.** The graded answer is a DECISION
-- which pack is cheaper by unit cost, or whether the two are tied -- never
a numeric unit price (that lands on `unit-pricing`) and never the numeric
GAP between two unit prices (that lands on `senior-use-rates-to-make-
comparisons`'s hard tier). Every price is supplied in cents; every unit
price comparison is an exact `fractions.Fraction` comparison, never a float
or a rounded value. A "tie" answer is only ever produced when the two exact
`Fraction` unit prices are equal by construction.

Tiers (genuine reasoning escalation, not range):

- **easy**: two packs, same unit of measure (grams, millilitres or items),
  quantities and prices given directly -- one division each, then compare.
- **medium**: two packs in MISMATCHED units (mL vs L, g vs kg, item vs
  dozen) -- the learner must convert one pack onto the other's unit before
  the two unit prices are comparable.
- **hard**: THREE packs, same unit -- the learner must compute and compare
  three unit prices to find the single cheapest, or discover that two of
  the three tie for cheapest (a different decision: "identify the best of
  several", not "compare a pair"). Answer space widens to {A, B, C, tie}.
- **exam**: medium's mismatched-unit shape layered with a real tie
  possibility, PLUS a deliberately constructed near-miss: some seeds make
  the two prices agree to the nearest cent while remaining exactly unequal
  as `Fraction`s, so a learner who rounds the unit price too early before
  comparing wrongly concludes "tie". `mathgen.core.rounds_unambiguously`
  keeps that rounding itself unambiguous, so the trap is the early rounding
  choice, not an accidental ".5" coin flip.

Every tier's mcq option set is exhaustive and fixed size: {"Pack A",
"Pack B", "tie label"} for easy/medium/exam (2 packs), {"Pack A", "Pack B",
"Pack C", "tie label"} for hard (3 packs) -- so every draw offers every
applicable option; there is never a filtered-down 2-option mcq.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    check,
    check_equivalent,
    currency_from_cents,
    generator,
    latex_answer,
    round_half_up,
    stated_precision_string,
    rounds_unambiguously,
)

_RESAMPLE_ATTEMPTS = 300


def _decide(prices: dict) -> str:
    """The one place the decision logic lives: the label of the pack with
    the strictly smallest exact `Fraction` unit price, or ``"tie"`` if two or
    more share the smallest value. Called on FINAL, already-constructed
    values only -- this is the independent re-derivation:
    it never sees or reuses whichever branch of construction built the
    scenario, only the finished (price_cents, quantity) numbers."""
    smallest = min(prices.values())
    winners = sorted(label for label, price in prices.items() if price == smallest)
    if len(winners) > 1:
        return "tie"
    return winners[0]


def _verify_decision(prices: dict, expected: str, msg: str = "decision mismatch"):
    """Re-derive the decision from `prices` (a fresh dict of exact
    `Fraction` unit prices) via `_decide` and fail unless it equals
    `expected`. Every tier below calls this exactly once, on values
    recomputed from the FINAL (price_cents, quantity) numbers only -- this
    is the one seam a corrupted answer must pass through, so it is also the
    one unit-tested directly for firing on a corrupted case."""
    check(_decide(prices) == expected, msg)


# ---------------------------------------------------------------------------
# easy: two packs, same unit
# ---------------------------------------------------------------------------

_UNITS_EASY = {
    "g": dict(qty_lo=50, qty_hi=1000, price_lo=100, price_hi=3000, plural="grams", per="per gram"),
    "mL": dict(qty_lo=100, qty_hi=2000, price_lo=100, price_hi=3500, plural="millilitres", per="per millilitre"),
    "item": dict(qty_lo=2, qty_hi=24, price_lo=200, price_hi=3000, plural="items", per="per item"),
}


def _draw_two_pack(rng, cfg: dict, target: str):
    """Draw (qtyA, priceA, qtyB, priceB) whose DECISION is `target`
    ("A"/"B"/"tie"), exact Fraction unit prices throughout. Shared by the
    easy (same unit) and medium/exam (converted-unit) callers -- they each
    pass `cfg`'s ranges and interpret the returned quantities in their own
    unit; only the tie branch (a literal same-unit scale-up) is specific to
    a shared unit, so medium/exam build their own tie branch instead of
    calling this for target == "tie"."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        qtyA = rng.randint(cfg["qty_lo"], cfg["qty_hi"])
        priceA = rng.randint(cfg["price_lo"], cfg["price_hi"])
        uA = Fraction(priceA, qtyA)
        if target == "tie":
            m = rng.randint(2, 4)
            qtyB, priceB = qtyA * m, priceA * m
        else:
            qtyB = rng.randint(cfg["qty_lo"], cfg["qty_hi"])
            priceB = rng.randint(cfg["price_lo"], cfg["price_hi"])
        uB = Fraction(priceB, qtyB)
        decision = _decide({"A": uA, "B": uB})
        if decision != target:
            continue
        return qtyA, priceA, qtyB, priceB, uA, uB
    raise ValueError(f"p14b_compare_unit_prices: no collision-free draw for target={target!r}")


def _option_texts(kind: str) -> dict:
    if kind == "two_pack":
        return {"A": "Pack A", "B": "Pack B", "tie": "Tie, the same price per unit"}
    return {"A": "Pack A", "B": "Pack B", "C": "Pack C", "tie": "Tie, two packs share the cheapest price"}


def _mcq_dict(answer_key: str, kind: str, text: str, working: list):
    options = _option_texts(kind)
    answer_text = options[answer_key]
    distractors = [v for k, v in options.items() if k != answer_key]
    return dict(
        text=text,
        latex=text,
        answer=answer_key,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


def _easy(rng):
    ctx_name = rng.choice(sorted(_UNITS_EASY))
    cfg = _UNITS_EASY[ctx_name]
    target = rng.choice(("A", "B", "tie"))
    qtyA, priceA, qtyB, priceB, uA, uB = _draw_two_pack(rng, cfg, target)

    # Independent re-derivation, from the final numbers only.
    recomputed = {"A": Fraction(priceA, qtyA), "B": Fraction(priceB, qtyB)}
    _verify_decision(recomputed, target, "easy: re-derived decision must match the constructed target")
    decision = target
    check_equivalent(uA, recomputed["A"], "easy: unit price A must be exactly price/quantity")
    check_equivalent(uB, recomputed["B"], "easy: unit price B must be exactly price/quantity")

    text = (
        f"Pack A: {qtyA} {cfg['plural']} for {currency_from_cents(priceA)}. "
        f"Pack B: {qtyB} {cfg['plural']} for {currency_from_cents(priceB)}. "
        f"Which is the cheaper buy {cfg['per']}, or are the two the same price {cfg['per']}?"
    )
    working = [
        f"Unit price A = {currency_from_cents(priceA)} / {qtyA} = {stated_precision_string(round_half_up(uA, 4), 4)} cents {cfg['per']}.",
        f"Unit price B = {currency_from_cents(priceB)} / {qtyB} = {stated_precision_string(round_half_up(uB, 4), 4)} cents {cfg['per']}.",
    ]
    return _mcq_dict(decision, "two_pack", text, working)


# ---------------------------------------------------------------------------
# medium / exam: two packs, mismatched units
# ---------------------------------------------------------------------------

_UNITS_MISMATCH = {
    "volume": dict(
        factor=1000,
        small="mL",
        big="L",
        small_word="millilitre",
        big_word="litre",
        small_qty=(200, 1000),
        big_qty=(1, 4),
        price_lo=100,
        price_hi=1200,
        big_rate=(50, 6000),
    ),
    "mass": dict(
        factor=1000,
        small="g",
        big="kg",
        small_word="gram",
        big_word="kilogram",
        small_qty=(200, 1000),
        big_qty=(1, 4),
        price_lo=100,
        price_hi=1500,
        big_rate=(80, 7500),
    ),
    "count": dict(
        factor=12,
        small="item",
        big="dozen",
        small_word="item",
        big_word="dozen",
        small_qty=(2, 20),
        big_qty=(1, 4),
        price_lo=100,
        price_hi=1500,
        big_rate=(60, 9000),
    ),
}


def _plural_unit(unit: str, qty: int) -> str:
    """ "item" pluralises with the count ("1 item", "6 items"); the other
    unit tokens here are already abbreviations/collective nouns that do not
    ("mL", "L", "g", "kg", "dozen")."""
    return "items" if unit == "item" and qty != 1 else unit


def _draw_mismatch(rng, cfg: dict, target: str, *, avoid_ambiguous_round: bool = False):
    """Draw a small-unit pack and a big-unit pack whose DECISION -- compared
    on the small unit's basis -- is `target` (``"small"``/``"big"``/``"tie"``,
    naming which SIDE is cheaper, before either side is assigned to Pack A or
    Pack B). Every price stays an integer count of cents; the small-unit
    unit price is an exact `Fraction` in cents-per-small-unit, and the
    big-unit pack's price is chosen so its own small-unit-basis unit price
    is EXACTLY equal (tie) or strictly different (small/big), never by
    rounding."""
    small, big, factor = cfg["small"], cfg["big"], cfg["factor"]
    for _ in range(_RESAMPLE_ATTEMPTS):
        big_qty = rng.randint(*cfg["big_qty"])
        if target == "tie":
            # Draw a WHOLE-cent per-small-unit rate directly, so both packs'
            # prices are exact integer cents by construction -- never a
            # divisibility gamble on independently-drawn quantities.
            rate = rng.randint(
                max(1, cfg["price_lo"] // cfg["small_qty"][1]), max(2, cfg["price_hi"] // cfg["small_qty"][0])
            )
            small_qty = rng.randint(*cfg["small_qty"])
            small_price = rate * small_qty
            small_unit_price = Fraction(small_price, small_qty)
            big_price = rate * factor * big_qty
        else:
            small_qty = rng.randint(*cfg["small_qty"])
            small_price = rng.randint(cfg["price_lo"], cfg["price_hi"])
            small_unit_price = Fraction(small_price, small_qty)  # cents per small unit
            big_price = big_qty * rng.randint(*cfg["big_rate"])
        big_unit_price_on_small_basis = Fraction(big_price, big_qty * factor)
        if avoid_ambiguous_round and not rounds_unambiguously(float(small_unit_price), 2):
            continue
        decision = _decide({"small": small_unit_price, "big": big_unit_price_on_small_basis})
        if decision != target:
            continue
        return small_qty, small_price, big_qty, big_price, small_unit_price, big_unit_price_on_small_basis
    raise ValueError(f"p14b_compare_unit_prices: no collision-free mismatch draw for target={target!r}")


def _build_mismatch_question(
    rng, cfg, draw, target: str, small_is_a: bool, working_extra=None, exact_note: bool = False
):
    """Assign the small-unit pack to A or B per `small_is_a`, then
    independently RE-DERIVE the A/B decision from the final numbers alone
    (never trusting whichever "small"/"big" target built the draw) and check
    it against `target` (translated into A/B terms via `small_is_a`)."""
    (small_qty, small_price, big_qty, big_price, small_unit_price, big_unit_price_on_small_basis) = draw
    small, big = cfg["small"], cfg["big"]
    small_label = "millilitre" if small == "mL" else ("gram" if small == "g" else "item")

    if small_is_a:
        qtyA, unitA, priceA, priceA_rate = small_qty, small, small_price, small_unit_price
        qtyB, unitB, priceB, priceB_rate = big_qty, big, big_price, big_unit_price_on_small_basis
    else:
        qtyA, unitA, priceA, priceA_rate = big_qty, big, big_price, big_unit_price_on_small_basis
        qtyB, unitB, priceB, priceB_rate = small_qty, small, small_price, small_unit_price

    # Independent re-derivation: recompute each pack's per-small-unit rate
    # from its OWN final (price, quantity, unit) alone.
    factor = cfg["factor"]
    rateA_recomputed = Fraction(priceA, qtyA if unitA == small else qtyA * factor)
    rateB_recomputed = Fraction(priceB, qtyB if unitB == small else qtyB * factor)
    check_equivalent(rateA_recomputed, priceA_rate, "mismatch: re-derived rate A must match construction")
    check_equivalent(rateB_recomputed, priceB_rate, "mismatch: re-derived rate B must match construction")
    if target == "tie":
        expected_ab = "tie"
    else:
        label_of = {"small": "A", "big": "B"} if small_is_a else {"small": "B", "big": "A"}
        expected_ab = label_of[target]
    recomputed_ab = {"A": rateA_recomputed, "B": rateB_recomputed}
    _verify_decision(recomputed_ab, expected_ab, "mismatch: re-derived decision must match the constructed target")
    decision_key = expected_ab

    text = (
        f"Pack A: {qtyA} {_plural_unit(unitA, qtyA)} for {currency_from_cents(priceA)}. "
        f"Pack B: {qtyB} {_plural_unit(unitB, qtyB)} for {currency_from_cents(priceB)}. "
        f"(1 {big} = {cfg['factor']} {_plural_unit(small, cfg['factor'])}.) "
        f"Which is the cheaper buy per {small_label}, or are the two the same price per {small_label}?"
    )
    if exact_note:
        # A distinct sentence, not present at any lower tier -- carries the
        # exam tier's own reasoning demand (compare EXACTLY, never by an
        # early rounding) and gives this tier a `prompt_shape()` no lower
        # tier can produce, on top of the near-miss/true-tie construction
        # itself.
        text += f" Compare the EXACT price per {small_label} -- do not round until after you have decided."
    working = [
        f"Pack A price per {small_label} = {stated_precision_string(round_half_up(rateA_recomputed, 2), 2)} cents.",
        f"Pack B price per {small_label} = {stated_precision_string(round_half_up(rateB_recomputed, 2), 2)} cents.",
    ]
    if working_extra:
        working.extend(working_extra)
    return _mcq_dict(decision_key, "two_pack", text, working)


def _medium(rng):
    ctx_name = rng.choice(sorted(_UNITS_MISMATCH))
    cfg = _UNITS_MISMATCH[ctx_name]
    target = rng.choice(("small", "big", "tie"))
    draw = _draw_mismatch(rng, cfg, target)
    small_is_a = rng.random() < 0.5
    return _build_mismatch_question(rng, cfg, draw, target, small_is_a)


def _exam(rng):
    """medium's mismatched-unit shape, plus a genuine near-miss trap on a
    third of draws: the two prices agree to the nearest cent while remaining
    exactly UNEQUAL as Fractions, so an early-rounding learner wrongly calls
    a tie. The other two thirds are an ordinary true tie, or an ordinary
    clean A/B decision, both still in mismatched units."""
    ctx_name = rng.choice(sorted(_UNITS_MISMATCH))
    cfg = _UNITS_MISMATCH[ctx_name]
    branch = rng.choice(("true_tie", "near_miss", "clean"))

    if branch == "true_tie":
        draw = _draw_mismatch(rng, cfg, "tie")
        small_is_a = rng.random() < 0.5
        return _build_mismatch_question(
            rng,
            cfg,
            draw,
            "tie",
            small_is_a,
            exact_note=True,
            working_extra=["Exactly equal despite the different units -- a genuine tie, not a rounding coincidence."],
        )

    if branch == "clean":
        target = rng.choice(("small", "big"))
        draw = _draw_mismatch(rng, cfg, target)
        small_is_a = rng.random() < 0.5
        return _build_mismatch_question(rng, cfg, draw, target, small_is_a, exact_note=True)

    # near_miss: force the small-unit price to round to the SAME cent as the
    # big-unit price's exact (integer-cent) rate, while remaining exactly
    # unequal. `rounds_unambiguously` keeps the rounding itself unambiguous,
    # so the trap is purely "rounded too early", never a ".5" coin flip.
    factor = cfg["factor"]
    for _ in range(_RESAMPLE_ATTEMPTS):
        small_qty = rng.randint(*cfg["small_qty"])
        small_price = rng.randint(cfg["price_lo"], cfg["price_hi"])
        small_unit_price = Fraction(small_price, small_qty)  # cents per small unit, may not be a whole cent
        if small_unit_price.denominator == 1:
            continue  # need a genuine fractional-cent value to round away from
        if not rounds_unambiguously(float(small_unit_price), 0):
            continue
        rounded_cents = int(round_half_up(small_unit_price, 0))
        if rounded_cents <= 0:
            continue
        big_qty = rng.randint(*cfg["big_qty"])
        big_price = rounded_cents * big_qty * factor  # exact integer cents
        big_unit_price_on_small_basis = Fraction(big_price, big_qty * factor)
        check(
            big_unit_price_on_small_basis == rounded_cents,
            "near-miss: big pack's rate must equal the rounded small-pack rate exactly",
        )
        if small_unit_price == big_unit_price_on_small_basis:
            continue  # not actually a near-miss
        draw = (small_qty, small_price, big_qty, big_price, small_unit_price, big_unit_price_on_small_basis)
        near_miss_target = _decide({"small": small_unit_price, "big": big_unit_price_on_small_basis})
        small_is_a = rng.random() < 0.5
        return _build_mismatch_question(
            rng,
            cfg,
            draw,
            near_miss_target,
            small_is_a,
            exact_note=True,
            working_extra=[
                "Rounding the small-pack rate to the nearest cent first makes the two prices "
                "LOOK the same -- the exact fractions are not equal, so this is not a tie.",
            ],
        )
    raise ValueError("p14b_compare_unit_prices: no near-miss exam draw found")


# ---------------------------------------------------------------------------
# hard: three packs, same unit
# ---------------------------------------------------------------------------

_UNITS_HARD = {
    "g": dict(qty_lo=50, qty_hi=1000, price_lo=100, price_hi=3000, plural="grams", per="per gram"),
    "mL": dict(qty_lo=100, qty_hi=2000, price_lo=100, price_hi=3500, plural="millilitres", per="per millilitre"),
    "item": dict(qty_lo=2, qty_hi=24, price_lo=200, price_hi=3000, plural="items", per="per item"),
}
_HARD_LABELS = ("A", "B", "C")


def _hard(rng):
    ctx_name = rng.choice(sorted(_UNITS_HARD))
    cfg = _UNITS_HARD[ctx_name]
    target = rng.choice(("A", "B", "C", "tie"))

    for _ in range(_RESAMPLE_ATTEMPTS):
        qtys = {lab: rng.randint(cfg["qty_lo"], cfg["qty_hi"]) for lab in _HARD_LABELS}
        prices = {lab: rng.randint(cfg["price_lo"], cfg["price_hi"]) for lab in _HARD_LABELS}
        if target == "tie":
            tied_pair = rng.sample(_HARD_LABELS, 2)
            # Scale pack `tied_pair[1]` up from `tied_pair[0]` by an integer
            # factor `k` -- guarantees an EXACT tie (same construction idiom
            # as the easy/medium tiers' own tie branch), never a divisibility
            # gamble on independently-drawn quantities.
            k = rng.randint(2, 4)
            qtys[tied_pair[1]] = qtys[tied_pair[0]] * k
            prices[tied_pair[1]] = prices[tied_pair[0]] * k
        unit_prices = {lab: Fraction(prices[lab], qtys[lab]) for lab in _HARD_LABELS}
        decision = _decide(unit_prices)
        if decision != target:
            continue
        break
    else:
        raise ValueError(f"p14b_compare_unit_prices: no collision-free hard draw for target={target!r}")

    # Independent re-derivation from the final numbers only.
    unit_prices_recomputed = {lab: Fraction(prices[lab], qtys[lab]) for lab in _HARD_LABELS}
    for lab in _HARD_LABELS:
        check_equivalent(
            unit_prices[lab], unit_prices_recomputed[lab], f"hard: unit price {lab} must be exactly price/quantity"
        )
    _verify_decision(unit_prices_recomputed, target, "hard: re-derived decision must match the constructed target")
    decision_final = target

    text = (
        f"Pack A: {qtys['A']} {cfg['plural']} for {currency_from_cents(prices['A'])}. "
        f"Pack B: {qtys['B']} {cfg['plural']} for {currency_from_cents(prices['B'])}. "
        f"Pack C: {qtys['C']} {cfg['plural']} for {currency_from_cents(prices['C'])}. "
        f"Which pack is the cheapest {cfg['per']}, or do two of them tie for cheapest?"
    )
    working = [
        f"Unit price {lab} = {currency_from_cents(prices[lab])} / {qtys[lab]} "
        f"= {stated_precision_string(round_half_up(unit_prices_recomputed[lab], 4), 4)} cents {cfg['per']}."
        for lab in _HARD_LABELS
    ]
    return _mcq_dict(decision_final, "three_pack", text, working)


@generator(
    "p14b_compare_unit_prices",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
        "exam": dict(mode="exam"),
    },
)
def p14b_compare_unit_prices(rng, mode="easy"):
    """Compare two (or, at hard, three) supplied pack prices by exact unit
    cost and choose the cheaper option, or a tie -- never a numeric unit
    price or numeric gap. easy: same unit, two packs. medium: mismatched
    units, two packs. hard: three packs, same unit, single cheapest or a
    two-way tie. exam: medium's mismatched-unit shape with a real tie
    possibility and a constructed rounds-to-the-same-cent-but-not-equal
    trap."""
    if mode == "easy":
        return _easy(rng)
    if mode == "medium":
        return _medium(rng)
    if mode == "hard":
        return _hard(rng)
    if mode == "exam":
        return _exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")
