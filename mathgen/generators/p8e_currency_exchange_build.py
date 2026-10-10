"""Currency exchange conversion, senior ``acara-senior-v8-4``, domain ``arithmetic``, ``answer: number``.

One generator: ``currency_exchange_conversion``
(``senior-use-currency-exchange-rates-to-determine-the-cost``, ACMGM007) --
every question converts a stated foreign-currency amount (``US$``, ``€`` or
``£``) into Australian dollars (``A$``), never the reverse direction, per the
skill's own two named examples (``US$1500``, ``€2050``).

The genuine reasoning content is which OPERATION applies, which depends on
which currency's "1 unit" the stated exchange rate is quoted against:

- **AUD-based** rate ("A$1 = US$0.65"): converting a foreign amount ``F`` to
  AUD is ``aud = F / rate``.
- **Foreign-based** rate ("US$1 = A$1.35"): converting a foreign amount ``F``
  to AUD is ``aud = F * rate``.

Both directions are real-world-authentic; the question always states, in
words, which currency's "1 unit" the rate is quoted against, so the correct
operation is never ambiguous from the text.

**Why the exact-cent condition needs no resample loop at all, at any tier**: the foreign-based direction
always lands on an exact cent by construction -- ``aud_cents = aud * 100 = F * rate * 100 = F * p`` where
``rate = p / 100`` and ``F``/``p`` are both integers, so the product is always an integer. Only the AUD-based
(division) direction can fail to land on an exact cent, and exactly when it will is computable in closed form:
``aud_cents = F * 10000 / p``, an integer iff ``p``'s part coprime to 10 (call it ``d = p / gcd(p, 10000)``)
divides ``F``. Choosing ``F`` as a multiple of ``d`` up front is therefore a *deterministic* construction, not
a trial-and-error resample, and it is what every tier does below. A loop is still used, but only to avoid a
distractor landing on the same displayed cents value as another option or the answer -- the same purpose
``p7d_rates_build.py``'s resample loops serve.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    currency_from_cents,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
)

_RESAMPLE_ATTEMPTS = 100

FOREIGN_LO, FOREIGN_HI = 50, 400
RATE_LO_CENTS, RATE_HI_CENTS = 50, 190  # 0.50 .. 1.90; 1.00 is excluded below
NICE_RATES_CENTS = (50, 60, 75, 80, 125, 150)  # easy tier's "nice" rate set
SYMBOLS = ("US$", "€", "£")


def _rate_str(p_cents: int) -> str:
    """A rate quoted to exactly 2 decimal places, no currency symbol: "0.65", "1.35"."""
    check(p_cents >= 0, "a rate must never be negative")
    return f"{p_cents // 100}.{p_cents % 100:02d}"


def _divisor_for_aud_based(p_cents: int) -> int:
    """The smallest step a foreign amount ``F`` must be a multiple of so that
    ``F / (p_cents / 100)`` lands on an exact cent: ``aud_cents = F * 10000 /
    p_cents`` is an integer iff ``d = p_cents / gcd(p_cents, 10000)`` divides
    ``F``, since ``10000 = 2**4 * 5**4`` absorbs every factor of 2 and 5 in
    ``p_cents`` and ``d`` is what is left over (the part of the rate coprime
    to 10)."""
    return p_cents // math.gcd(p_cents, 10000)


def _pick_amount_for_aud_based(rng, p_cents: int) -> int:
    """A foreign amount in ``[FOREIGN_LO, FOREIGN_HI]`` that is a multiple of
    the divisor `p_cents` requires -- deterministic, not trial and error."""
    d = _divisor_for_aud_based(p_cents)
    lo = -(-FOREIGN_LO // d)  # ceil division
    hi = FOREIGN_HI // d
    check(hi >= lo, f"no valid foreign amount multiple of {d} for rate {p_cents}c in range")
    return d * rng.randint(lo, hi)


def _random_rate_cents(rng) -> int:
    """A general 2-decimal-place rate in [0.50, 1.90], excluding exactly
    1.00 -- kept out of the sampled range so `treats-as-one-to-one` (which
    reports the answer unchanged, as if the rate were exactly 1) is
    UNCONDITIONALLY distinct from the true answer at every seed, never merely
    "usually" distinct."""
    p = rng.randint(RATE_LO_CENTS, RATE_HI_CENTS)
    while p == 100:
        p = rng.randint(RATE_LO_CENTS, RATE_HI_CENTS)
    return p


# --- Independent in-generator check: parse the RENDERED TEXT, not the local
# construction variables, and recompute via the same direction-dependent
# formula -----------------------------------------------------------------

_TEXT_RE = re.compile(
    r"The exchange rate is (A\$|US\$|€|£)1 = (A\$|US\$|€|£)(\d+\.\d{2})\. "
    r"What is the (?:cost, in Australian dollars, of buying|value, in Australian dollars, of) "
    r"(US\$|€|£)(\d+)\?"
)


def _verify_from_text(text: str, answer_cents: int):
    """Parse the rate, its stated direction and the foreign amount back out
    of the rendered question text by regex (never the construction's own
    `direction`/`rate`/`F` local variables), and recompute the AUD value via
    the same direction-dependent formula, as an exact `Fraction` -- a
    genuinely different route from however construction built `answer_cents`."""
    m = _TEXT_RE.search(text)
    check(m is not None, f"currency_exchange_conversion: unparseable question text: {text!r}")
    lhs_sym, rhs_sym, rate_str, amount_sym, amount_str = m.groups()
    rate = Fraction(rate_str)
    foreign_amount = Fraction(int(amount_str))
    if lhs_sym == "A$":
        # AUD-based: "A$1 = <foreign>0.XX" -- 1 Australian dollar buys `rate`
        # units of the foreign currency, so converting a foreign amount to
        # AUD is a division.
        check(rhs_sym in SYMBOLS, "an AUD-based rate sentence must quote a foreign currency on the right")
        check(amount_sym == rhs_sym, "the amount's currency must match the rate's foreign currency")
        recomputed = foreign_amount / rate
    else:
        # Foreign-based: "<foreign>1 = A$X.XX" -- 1 foreign-currency unit is
        # worth `rate` Australian dollars, so converting a foreign amount to
        # AUD is a multiplication.
        check(rhs_sym == "A$", "a foreign-based rate sentence must quote AUD on the right")
        check(amount_sym == lhs_sym, "the amount's currency must match the rate's foreign currency")
        recomputed = foreign_amount * rate
    check_equivalent(
        recomputed * 100,
        answer_cents,
        "text-parsed independent recompute must match the constructed AUD value",
    )


def _try_build(direction: str, p_cents: int, foreign_amount: int):
    """Compute the canonical AUD cents and every misconception's displayed
    cents value for one (direction, rate, foreign amount) draw. Returns None
    if the canonical value does not land on an exact cent (never happens for
    `foreign_based`; only possible for `aud_based` when the caller did not
    already pick `foreign_amount` as a multiple of `_divisor_for_aud_based`)."""
    rate = Fraction(p_cents, 100)
    foreign_amount = Fraction(foreign_amount)
    if direction == "aud_based":
        aud = foreign_amount / rate
        wrong_direction = foreign_amount * rate  # the OTHER operation
    else:
        aud = foreign_amount * rate
        wrong_direction = foreign_amount / rate  # the OTHER operation

    aud_cents_frac = aud * 100
    if aud_cents_frac.denominator != 1:
        return None
    answer_cents = int(aud_cents_frac)

    check(wrong_direction >= 0, "an AUD amount must never be negative")
    wrong_direction_cents = int(round_half_up(wrong_direction, 2) * 100)
    one_to_one_cents = int(foreign_amount * 100)  # reports F unchanged as the AUD value
    adds_rate = foreign_amount + rate
    check(adds_rate >= 0, "an AUD amount must never be negative")
    adds_rate_cents = int(round_half_up(adds_rate, 2) * 100)

    cents_values = (answer_cents, wrong_direction_cents, one_to_one_cents, adds_rate_cents)
    if len(set(cents_values)) != len(cents_values):
        return None  # a displayed collision; caller resamples

    return dict(
        rate=rate,
        answer_cents=answer_cents,
        wrong_direction_cents=wrong_direction_cents,
        one_to_one_cents=one_to_one_cents,
        adds_rate_cents=adds_rate_cents,
    )


def _finalize(direction: str, p_cents: int, foreign_amount: int, sym: str, framing: str, built: dict):
    rate_str = _rate_str(p_cents)
    if direction == "aud_based":
        rate_sentence = f"The exchange rate is A$1 = {sym}{rate_str}."
    else:
        rate_sentence = f"The exchange rate is {sym}1 = A${rate_str}."
    if framing == "cost":
        ask = f"What is the cost, in Australian dollars, of buying {sym}{foreign_amount}?"
    else:
        ask = f"What is the value, in Australian dollars, of {sym}{foreign_amount}?"
    text = f"{rate_sentence} {ask}"

    answer_text = currency_from_cents(built["answer_cents"], symbol="A$", allow_negative=False)
    _verify_from_text(text, built["answer_cents"])
    check_pairwise_distinct(
        built["answer_cents"],
        built["wrong_direction_cents"],
        built["one_to_one_cents"],
        built["adds_rate_cents"],
    )

    if direction == "aud_based":
        working = [
            f"A$1 = {sym}{rate_str}, so 1 Australian dollar buys {rate_str} of the foreign currency: "
            f"divide by the rate. {sym}{foreign_amount} / {rate_str} = {answer_text}."
        ]
    else:
        working = [
            f"{sym}1 = A${rate_str}, so each foreign-currency unit is worth {rate_str} Australian "
            f"dollars: multiply by the rate. {sym}{foreign_amount} x {rate_str} = {answer_text}."
        ]

    candidates = [
        currency_from_cents(built["wrong_direction_cents"], symbol="A$", allow_negative=False),
        currency_from_cents(built["one_to_one_cents"], symbol="A$", allow_negative=False),
        currency_from_cents(built["adds_rate_cents"], symbol="A$", allow_negative=False),
    ]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) == 3, "currency_exchange_conversion: expects 3 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(Fraction(built["answer_cents"], 100)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


def _easy(rng):
    """AUD-based rate only, drawn from a small "nice" set chosen so the
    division lands on an exact cent for a WIDE range of foreign amounts with
    no resample loop -- `_divisor_for_aud_based` is 1 for {0.50, 0.80, 1.25}
    (any foreign amount in [50, 400] works, 351 values) and 3 for {0.60,
    0.75, 1.50} (foreign amount must be a multiple of 3, ~117 values in
    range) -- both "wide"."""
    p_cents = rng.choice(NICE_RATES_CENTS)
    foreign_amount = _pick_amount_for_aud_based(rng, p_cents)
    sym = rng.choice(SYMBOLS)
    framing = rng.choice(("cost", "value"))
    built = _try_build("aud_based", p_cents, foreign_amount)
    check(built is not None, "easy tier must land on an exact cent by construction (nice rate set)")
    return _finalize("aud_based", p_cents, foreign_amount, sym, framing, built)


def _medium(rng):
    """AUD-based rate only, but a GENERAL 2-decimal rate -- the foreign
    amount is still chosen deterministically as a multiple of
    `_divisor_for_aud_based(p_cents)` (see module docstring for why this
    needs no blind resampling), so the loop below only guards against a
    distractor collision, exactly as `p7d_rates_build.py`'s resample loops
    do."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        p_cents = _random_rate_cents(rng)
        foreign_amount = _pick_amount_for_aud_based(rng, p_cents)
        built = _try_build("aud_based", p_cents, foreign_amount)
        if built is None:
            continue
        break
    else:
        raise ValueError("currency_exchange_conversion: no collision-free medium draw found")
    sym = rng.choice(SYMBOLS)
    framing = rng.choice(("cost", "value"))
    return _finalize("aud_based", p_cents, foreign_amount, sym, framing, built)


def _hard(rng):
    """The rate's direction is drawn randomly each seed, roughly 50/50. A
    `foreign_based` draw always lands on an exact cent for ANY foreign
    amount (see module docstring); an `aud_based` draw uses the same
    deterministic multiple-of-the-divisor construction as `_medium`. The
    loop only guards against a distractor collision."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        direction = rng.choice(("aud_based", "foreign_based"))
        p_cents = _random_rate_cents(rng)
        if direction == "aud_based":
            foreign_amount = _pick_amount_for_aud_based(rng, p_cents)
        else:
            foreign_amount = rng.randint(FOREIGN_LO, FOREIGN_HI)
        built = _try_build(direction, p_cents, foreign_amount)
        if built is None:
            continue
        break
    else:
        raise ValueError("currency_exchange_conversion: no collision-free hard draw found")
    sym = rng.choice(SYMBOLS)
    framing = rng.choice(("cost", "value"))
    return _finalize(direction, p_cents, foreign_amount, sym, framing, built)


@generator(
    "currency_exchange_conversion",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def currency_exchange_conversion(rng, mode="easy"):
    """Convert a stated foreign-currency amount (US$, EUR or GBP) to
    Australian dollars using a given exchange rate. easy: the rate is always
    AUD-based ("A$1 = ...") from a small nice-rate set. medium: the rate is
    always AUD-based but a general 2-decimal value. hard: the rate's
    direction (AUD-based or foreign-based) is drawn randomly each seed, so
    the learner must read which currency's "1 unit" the rate is quoted
    against before choosing whether to multiply or divide."""
    if mode == "easy":
        return _easy(rng)
    if mode == "medium":
        return _medium(rng)
    if mode == "hard":
        return _hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")
