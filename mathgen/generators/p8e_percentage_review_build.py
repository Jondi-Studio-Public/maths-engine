"""Two senior ("adult numeracy revision") percentage skills, both ``acara-senior-v8-4``, domain ``arithmetic``,
``answer: number``.

Two generators:

- ``percentage_of_amount_review`` (``senior-review-calculating-a-percentage-
  of-a-given-amount``, ACMEM061) -- "What is p% of $X?" with p drawn from the
  WIDE 1-99 (easy/medium) or 101-150 (hard) range, every amount stated as
  money. This is the collision boundary against
  ``skills/primary-number/percentage-of-quantity.yaml`` (topic
  ``percentage_quantity``): that skill draws its percent ONLY from
  ``{5, 10, 20, 25, 50, 75}`` and states its quantity as a bare number, never
  a dollar amount. Every question here draws uniformly from the wide range
  and every amount is formatted ``$X.YY`` (always two decimal places), so a
  rendered question from either skill could never be mistaken for the other.

- ``amount_as_percentage_review`` (``senior-review-one-amount-expressed-as-
  a-percentage-of``, ACMEM062) -- "What percentage of B is A?" for two
  independently-stated whole-number amounts. This is the collision boundary
  against ``skills/secondary-number/percentage-change.yaml`` (topic
  ``percentage_change``): that skill computes a CHANGE from an "original"
  value to a "final" one (a rise). This skill never uses before/after,
  rises/increase/original/final framing -- every question states two
  amounts that exist independently at the same time (two surveys, two
  teams, two routes), never a value moving from one state to another.

Both generators use exact ``Fraction``/``int`` arithmetic throughout.
``percentage_of_amount_review`` constructs every draw so the percentage
amount lands on an exact cent (never a genuine float rounding); the
divisibility route is: for percent ``p``, ``d = 100 // gcd(p, 100)`` is the
smallest amount (in cents) that ``p`` divides evenly, so any multiple of
``d`` is a valid amount -- drawn directly, never by rejection sampling.
``amount_as_percentage_review`` states its answer to the nearest tenth of a
percent from medium tier on, guarding against a draw whose true value sits
exactly on the ``.X5`` rounding boundary (see ``_TIE`` below).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_distinct,
    check_equivalent,
    currency_from_cents,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
)

_RESAMPLE_ATTEMPTS = 100

# --- Local currency helpers ---------------------------------------------------
# Currency-string formatting uses the shared `mathgen.core.currency_from_cents`.
# `_parse_currency` is its inverse, used only by this module's independent
# checks (parsing the RENDERED "$X.YY" string back to a Fraction, a different
# route from the cents-integer arithmetic that built it).


def _parse_currency(s: str) -> Fraction:
    """ "$X.YY" -> Fraction(X) + Fraction(YY, 100), independent of whatever
    cents-integer arithmetic produced the string."""
    body = s.lstrip("$")
    dollars_str, cents_str = body.split(".")
    return Fraction(int(dollars_str)) + Fraction(int(cents_str), 100)


# =============================================================================
# Skill 1: percentage_of_amount_review (ACMEM061)
# =============================================================================

_MAX_CENTS = 100_000  # $1000.00, exclusive bound every answer must clear


def _cents_multiple_k_range(d: int, lo_cents: int, hi_cents: int) -> tuple[int, int]:
    """(k_lo, k_hi) so that d*k lands in [lo_cents, hi_cents]."""
    k_lo = -(-lo_cents // d)  # ceil division
    k_hi = hi_cents // d
    return k_lo, k_hi


def _draw_percent_excluding_fifty(rng, lo: int, hi: int) -> int:
    while True:
        p = rng.randint(lo, hi)
        if p != 50:
            return p


def _verify_single_amount(amount_str: str, p: int, portion_cents: int):
    """Independent check (easy/hard tiers): reparse the DISPLAYED dollar
    string -- not the cents integers construction used -- and recompute via
    Fraction(p, 100)."""
    check_equivalent(
        _parse_currency(amount_str) * Fraction(p, 100),
        Fraction(portion_cents, 100),
        "re-derived percentage of the parsed dollar amount must match construction",
    )


def _verify_summed_amounts(A_str: str, B_str: str, p: int, portion_cents: int):
    """Independent check (medium tier): reparse both displayed dollar
    strings, sum them and apply Fraction(p, 100) -- a different route from
    the cents-integer sum construction used."""
    check_equivalent(
        (_parse_currency(A_str) + _parse_currency(B_str)) * Fraction(p, 100),
        Fraction(portion_cents, 100),
        "re-derived percentage of the parsed dollar sum must match construction",
    )


def _amount_easy(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        p = _draw_percent_excluding_fifty(rng, 1, 99)
        d = 100 // math.gcd(p, 100)
        k_lo, k_hi = _cents_multiple_k_range(d, 1000, 60000)
        if k_lo > k_hi:
            continue
        amount_cents = d * rng.randint(k_lo, k_hi)
        check(amount_cents * p % 100 == 0, "easy portion must land on an exact cent")
        portion_cents = amount_cents * p // 100
        check(portion_cents < _MAX_CENTS, "easy answer must stay under $1000")

        divides_by_ten_cents = 10 * portion_cents
        subtracts_cents = amount_cents - portion_cents
        try:
            check_distinct(portion_cents, divides_by_ten_cents, subtracts_cents)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("percentage_of_amount_review: no collision-free easy draw found")

    amount_str = currency_from_cents(amount_cents)
    answer_text = currency_from_cents(portion_cents)
    _verify_single_amount(amount_str, p, portion_cents)

    answer_dollars = Fraction(portion_cents, 100)
    for cand_cents, label in (
        (divides_by_ten_cents, "divides-by-ten-not-hundred"),
        (subtracts_cents, "subtracts-the-percentage-amount"),
    ):
        check(
            abs(answer_dollars - Fraction(cand_cents, 100)) > Fraction(1, 200),
            f"easy: {label} distractor must clear the 0.005 grading tolerance",
        )

    text = f"What is {p}% of {amount_str}?"
    working = [f"{p}% of {amount_str} = {amount_str} x {p} / 100 = {answer_text}."]
    candidates = [currency_from_cents(divides_by_ten_cents), currency_from_cents(subtracts_cents)]
    distractors = pick_distractors(answer_text, candidates, n=2, tolerance=Fraction(1, 200))
    check(len(distractors) == 2, "easy tier expects 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(Fraction(portion_cents, 100)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


_ITEM_PAIRS = [
    ("jacket", "shirt"),
    ("bicycle", "helmet"),
    ("sofa", "coffee table"),
    ("phone", "phone case"),
]


def _amount_medium(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        p = _draw_percent_excluding_fifty(rng, 1, 99)
        d = 100 // math.gcd(p, 100)
        k_lo, k_hi = _cents_multiple_k_range(d, 1000, 30000)
        if k_lo > k_hi:
            continue
        A_cents = d * rng.randint(k_lo, k_hi)
        B_cents = d * rng.randint(k_lo, k_hi)
        sum_cents = A_cents + B_cents
        check(sum_cents * p % 100 == 0, "medium portion must land on an exact cent")
        portion_cents = sum_cents * p // 100
        check(portion_cents < _MAX_CENTS, "medium answer must stay under $1000")

        divides_by_ten_cents = 10 * portion_cents
        subtracts_cents = sum_cents - portion_cents
        forgets_sum_cents = sum_cents
        try:
            check_distinct(portion_cents, divides_by_ten_cents, subtracts_cents, forgets_sum_cents)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("percentage_of_amount_review: no collision-free medium draw found")

    item_a, item_b = rng.choice(_ITEM_PAIRS)
    A_str = currency_from_cents(A_cents)
    B_str = currency_from_cents(B_cents)
    sum_str = currency_from_cents(sum_cents)
    answer_text = currency_from_cents(portion_cents)
    _verify_summed_amounts(A_str, B_str, p, portion_cents)

    answer_dollars = Fraction(portion_cents, 100)
    for cand_cents, label in (
        (divides_by_ten_cents, "divides-by-ten-not-hundred"),
        (subtracts_cents, "subtracts-the-percentage-amount"),
        (forgets_sum_cents, "forgets-to-apply-percent-to-sum"),
    ):
        check(
            abs(answer_dollars - Fraction(cand_cents, 100)) > Fraction(1, 200),
            f"medium: {label} distractor must clear the 0.005 grading tolerance",
        )

    text = f"A {item_a} costs {A_str} and a {item_b} costs {B_str}. What is {p}% of the total cost of both items?"
    working = [f"Total cost = {A_str} + {B_str} = {sum_str}; {p}% of {sum_str} = {answer_text}."]
    candidates = [
        currency_from_cents(divides_by_ten_cents),
        currency_from_cents(subtracts_cents),
        currency_from_cents(forgets_sum_cents),
    ]
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=Fraction(1, 200))
    check(len(distractors) >= 2, "medium tier expects at least 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(Fraction(portion_cents, 100)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


def _amount_hard(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        p = rng.randint(101, 150)
        d = 100 // math.gcd(p, 100)
        k_lo, k_hi = _cents_multiple_k_range(d, 1000, 60000)
        if k_lo > k_hi:
            continue
        amount_cents = d * rng.randint(k_lo, k_hi)
        check(amount_cents * p % 100 == 0, "hard portion must land on an exact cent")
        portion_cents = amount_cents * p // 100
        check(portion_cents < _MAX_CENTS, "hard answer must stay under $1000")

        divides_by_ten_cents = 10 * portion_cents
        caps_cents = amount_cents
        try:
            check_distinct(portion_cents, divides_by_ten_cents, caps_cents)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("percentage_of_amount_review: no collision-free hard draw found")

    amount_str = currency_from_cents(amount_cents)
    answer_text = currency_from_cents(portion_cents)
    _verify_single_amount(amount_str, p, portion_cents)

    answer_dollars = Fraction(portion_cents, 100)
    for cand_cents, label in (
        (divides_by_ten_cents, "divides-by-ten-not-hundred"),
        (caps_cents, "caps-at-the-whole-amount"),
    ):
        check(
            abs(answer_dollars - Fraction(cand_cents, 100)) > Fraction(1, 200),
            f"hard: {label} distractor must clear the 0.005 grading tolerance",
        )

    text = f"What is {p}% of {amount_str}?"
    working = [
        f"{p}% of {amount_str} = {amount_str} x {p} / 100 = {answer_text} "
        f"(more than {amount_str} because {p}% is more than 100%)."
    ]
    candidates = [currency_from_cents(divides_by_ten_cents), currency_from_cents(caps_cents)]
    distractors = pick_distractors(answer_text, candidates, n=2, tolerance=Fraction(1, 200))
    check(len(distractors) == 2, "hard tier expects 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(Fraction(portion_cents, 100)),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )


@generator(
    "percentage_of_amount_review",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def percentage_of_amount_review(rng, mode="easy"):
    """ "What is p% of $X?" with p drawn from the wide 1-99 (easy/medium) or
    101-150 (hard) range, every amount stated as money. easy: one amount.
    medium: two amounts summed before the percent is applied. hard: percent
    exceeds 100, so the percentage amount exceeds the amount itself."""
    if mode == "easy":
        return _amount_easy(rng)
    if mode == "medium":
        return _amount_medium(rng)
    if mode == "hard":
        return _amount_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# =============================================================================
# Skill 2: amount_as_percentage_review (ACMEM062)
# =============================================================================
# Every question states two independent whole-number amounts A and B (never
# before/after, never rises/increase/original/final framing -- that is
# percentage-change's shape) and asks "what percentage of B is A?"


def _fmt_tenths(tenths: int) -> str:
    sign = "-" if tenths < 0 else ""
    tenths = abs(tenths)
    whole, tenth = divmod(tenths, 10)
    return f"{sign}{whole}.{tenth}"


# (stem template, question template with a `{rounding}` slot) -- B before A
# in every stem (B is always the larger, "base" quantity here).
_CONTEXTS_LE = [
    ("In a survey of {B} people, {A} own a car.", "What percentage{rounding} of the people surveyed own a car?"),
    (
        "A school has {B} students enrolled. {A} of them play a sport.",
        "What percentage{rounding} of the students play a sport?",
    ),
    (
        "A shop receives {B} orders in a week. {A} of the orders are returned.",
        "What percentage{rounding} of the orders are returned?",
    ),
    ("Out of {B} matches played, a team won {A} of them.", "What percentage{rounding} of the matches were won?"),
]

# A before B in every stem (A is always the larger quantity, since A > B here).
_CONTEXTS_GT = [
    (
        "Route A is {A} km long. Route B is {B} km long.",
        "What percentage{rounding} of Route B's length is Route A's length?",
    ),
    (
        "Team A scored {A} points. Team B scored {B} points in the same competition.",
        "What percentage{rounding} of Team B's points is Team A's points?",
    ),
    (
        "Shop A sold {A} tickets. Shop B sold {B} tickets for the same event.",
        "What percentage{rounding} of Shop B's ticket sales is Shop A's ticket sales?",
    ),
]

_LE_PATTERNS = [
    re.compile(r"In a survey of (?P<B>\d+) people, (?P<A>\d+) own a car\."),
    re.compile(r"A school has (?P<B>\d+) students enrolled\. (?P<A>\d+) of them play a sport\."),
    re.compile(r"A shop receives (?P<B>\d+) orders in a week\. (?P<A>\d+) of the orders are returned\."),
    re.compile(r"Out of (?P<B>\d+) matches played, a team won (?P<A>\d+) of them\."),
]

_GT_PATTERNS = [
    re.compile(r"Route A is (?P<A>\d+) km long\. Route B is (?P<B>\d+) km long\."),
    re.compile(r"Team A scored (?P<A>\d+) points\. Team B scored (?P<B>\d+) points in the same competition\."),
    re.compile(r"Shop A sold (?P<A>\d+) tickets\. Shop B sold (?P<B>\d+) tickets for the same event\."),
]


def _question_text(stem_tpl: str, question_tpl: str, A: int, B: int, rounding_phrase: str) -> str:
    stem = stem_tpl.format(A=A, B=B)
    rounding = f", {rounding_phrase}," if rounding_phrase else ""
    question = question_tpl.format(rounding=rounding)
    return f"{stem} {question}"


def _extract_A_B(text: str, patterns: list) -> tuple[int, int]:
    """Independent re-derivation route: pull A and B back out of the
    RENDERED text via regex, never by reusing construction's own local
    variables."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            return int(m.group("A")), int(m.group("B"))
    raise ValueError(f"unparseable amount_as_percentage_review question text: {text!r}")


_ROUNDING_PHRASE = "to the nearest tenth of a percent"


def _verify_exact_percentage(text: str, patterns: list, P: int):
    """Independent check (easy tier): pull A, B back out of the RENDERED
    text via regex and recompute A/B*100 exactly, never reusing
    construction's own A/B local variables."""
    A2, B2 = _extract_A_B(text, patterns)
    check_equivalent(
        Fraction(A2, B2) * 100,
        Fraction(P),
        "re-parsed A/B*100 (from the rendered text) must equal the constructed exact percent",
    )


def _verify_rounded_percentage(text: str, patterns: list, rounded_tenths: int):
    """Independent check (medium/hard tiers): pull A, B back out of the
    RENDERED text via regex and re-round A/B*100 to the nearest tenth,
    never reusing construction's own A/B local variables."""
    A2, B2 = _extract_A_B(text, patterns)
    reparsed_tenths = int(round_half_up(Fraction(A2, B2) * 100, 1) * 10)
    check_equivalent(
        Fraction(reparsed_tenths, 10),
        Fraction(rounded_tenths, 10),
        "re-parsed A/B rounded to the nearest tenth (from the rendered text) must match construction",
    )


def _pct_easy(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        P = rng.randint(1, 99)
        frac = Fraction(P, 100)
        num, d = frac.numerator, frac.denominator
        k_lo = max(-(-20 // d), -(-2 // num))  # B >= 20 (realistic scale), A >= 2 (plural agreement)
        k_hi = 400 // d
        if k_lo > k_hi:
            continue
        k = rng.randint(k_lo, k_hi)
        B = d * k
        A = num * k
        check(A * 100 == P * B, "easy: A/B*100 must equal P exactly by construction")
        check(A >= 2, "easy: A must be at least 2 for plural agreement")

        swap_val = int(round_half_up(Fraction(B, A) * 100, 0))
        subtract_val = B - A
        forgets_val = int(round_half_up(Fraction(A, B), 0))
        try:
            check_distinct(P, swap_val, subtract_val, forgets_val)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("amount_as_percentage_review: no collision-free easy draw found")

    stem_tpl, question_tpl = rng.choice(_CONTEXTS_LE)
    text = _question_text(stem_tpl, question_tpl, A, B, "")
    _verify_exact_percentage(text, _LE_PATTERNS, P)

    answer_text = str(P)
    for cand_val, label in (
        (swap_val, "swaps-the-base"),
        (subtract_val, "subtracts-instead-of-dividing"),
        (forgets_val, "forgets-times-100"),
    ):
        check(P != cand_val, f"easy: {label} distractor must be a genuinely different integer")

    working = [f"{A} / {B} x 100 = {P}%."]
    candidates = [str(swap_val), str(subtract_val), str(forgets_val)]
    distractors = pick_distractors(answer_text, candidates, n=3)
    check(len(distractors) >= 2, "easy tier expects at least 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=P,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _pct_medium(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        B = rng.randint(20, 300)
        A = rng.randint(2, B - 1)
        tenths_scaled = Fraction(A, B) * 1000
        if tenths_scaled % 1 == Fraction(1, 2):
            continue  # exact .X5 rounding-boundary tie -- resample
        rounded_tenths = round(tenths_scaled)

        swap_tenths = int(round_half_up(Fraction(B, A) * 100, 1) * 10)
        subtract_tenths = (B - A) * 10
        forgets_tenths = int(round_half_up(Fraction(A, B), 1) * 10)
        try:
            check_distinct(rounded_tenths, swap_tenths, subtract_tenths, forgets_tenths)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("amount_as_percentage_review: no collision-free medium draw found")

    stem_tpl, question_tpl = rng.choice(_CONTEXTS_LE)
    text = _question_text(stem_tpl, question_tpl, A, B, _ROUNDING_PHRASE)
    _verify_rounded_percentage(text, _LE_PATTERNS, rounded_tenths)

    answer_text = _fmt_tenths(rounded_tenths)
    for cand_tenths, label in (
        (swap_tenths, "swaps-the-base"),
        (subtract_tenths, "subtracts-instead-of-dividing"),
        (forgets_tenths, "forgets-times-100"),
    ):
        check(
            abs(Fraction(rounded_tenths - cand_tenths, 10)) > Fraction(1, 20),
            f"medium: {label} distractor must clear the 0.05 grading tolerance",
        )

    working = [f"{A} / {B} x 100 rounds to {answer_text}%, to the nearest tenth of a percent."]
    candidates = [_fmt_tenths(swap_tenths), _fmt_tenths(subtract_tenths), _fmt_tenths(forgets_tenths)]
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=Fraction(1, 20))
    check(len(distractors) >= 2, "medium tier expects at least 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(Fraction(rounded_tenths, 10)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _pct_hard(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        B = rng.randint(50, 250)
        A = B + rng.randint(1, B)  # A > B: percentage strictly exceeds 100%
        tenths_scaled = Fraction(A, B) * 1000
        if tenths_scaled % 1 == Fraction(1, 2):
            continue  # exact .X5 rounding-boundary tie -- resample
        rounded_tenths = round(tenths_scaled)

        swap_tenths = int(round_half_up(Fraction(B, A) * 100, 1) * 10)
        subtract_tenths = (A - B) * 10
        forgets_tenths = int(round_half_up(Fraction(A, B), 1) * 10)
        caps_tenths = 1000  # "100.0" unconditionally
        try:
            check_distinct(rounded_tenths, swap_tenths, subtract_tenths, forgets_tenths, caps_tenths)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("amount_as_percentage_review: no collision-free hard draw found")

    stem_tpl, question_tpl = rng.choice(_CONTEXTS_GT)
    text = _question_text(stem_tpl, question_tpl, A, B, _ROUNDING_PHRASE)
    _verify_rounded_percentage(text, _GT_PATTERNS, rounded_tenths)

    answer_text = _fmt_tenths(rounded_tenths)
    for cand_tenths, label in (
        (swap_tenths, "swaps-the-base"),
        (subtract_tenths, "subtracts-instead-of-dividing"),
        (forgets_tenths, "forgets-times-100"),
        (caps_tenths, "caps-at-100-percent"),
    ):
        check(
            abs(Fraction(rounded_tenths - cand_tenths, 10)) > Fraction(1, 20),
            f"hard: {label} distractor must clear the 0.05 grading tolerance",
        )

    working = [
        f"{A} / {B} x 100 rounds to {answer_text}%, to the nearest tenth of a percent "
        f"(more than 100% because {A} is greater than {B})."
    ]
    candidates = [
        _fmt_tenths(swap_tenths),
        _fmt_tenths(subtract_tenths),
        _fmt_tenths(forgets_tenths),
        _fmt_tenths(caps_tenths),
    ]
    distractors = pick_distractors(answer_text, candidates, n=4, tolerance=Fraction(1, 20))
    check(len(distractors) >= 2, "hard tier expects at least 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=float(Fraction(rounded_tenths, 10)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "amount_as_percentage_review",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def amount_as_percentage_review(rng, mode="easy"):
    """ "What percentage of B is A?" for two independently-stated
    whole-number amounts -- never before/after framing. easy: B is
    constructed so A/B*100 is an exact whole percent. medium: general A, B
    (A < B), always stated and rounded to the nearest tenth of a percent.
    hard: A > B, so the percentage exceeds 100%."""
    if mode == "easy":
        return _pct_easy(rng)
    if mode == "medium":
        return _pct_medium(rng)
    if mode == "hard":
        return _pct_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")
