"""Generates `fr2b_written_division` questions for `f10-written-division` (AC9M4N06, AC9M5N07). `answer: number`.

Scope: dividing a three- or four-digit number by a one-digit divisor with a written method, and giving what is left
over as a fraction or a decimal. Sharing with a remainder as a whole number is the neighbouring skill
`f10-division-remainder-context`; this skill never asks for "the remainder" as a count.

Ladder (reasoning escalation, not size):
  easy:   no remainder, and the quotient has a 0 digit, so a step of the written method gives 0 and the 0 has to be
          written (3 612 / 4 = 903, not 93). Answer kind `number`.
  medium: a remainder that is written as a fraction of the divisor, as a mixed numeral. Answer kind `fraction`.
  hard:   the divisor is 2, 4, 5 or 8, so the remainder is written as a terminating decimal. Every distractor is
          shown to the same number of decimal places as the key. Answer kind `number`.

Every key is `divmod` of the dividend; it is re-derived by multiplying back (quotient times divisor plus the
remainder) and, for the mixed numeral and the decimal, by an exact Fraction.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, phrase, round_half_up, stated_precision_string
from ._fr2b_kit import build, grouped, int_dist, mixed_text, tagged

_TIERS = ("easy", "medium", "hard")

#: (setting, question, unit, lowest total, highest total) for a quantity that can be shared into parts of any size;
#: the range keeps each share a plausible size for the thing being shared.
_MEASURES = (
    ("A roll of ribbon is {n} cm long. It is cut into {d} equal pieces.", "How long is each piece?", "cm", 200, 1600),
    (
        "A water tank holds {n} L. The water is shared equally among {d} paddocks.",
        "How much water does each paddock get?",
        "L",
        400,
        4000,
    ),
    (
        "A truck carries {n} kg of flour. It is packed equally into {d} bulk bins.",
        "How much flour is in each bin?",
        "kg",
        500,
        4000,
    ),
    ("A cycle path is {n} m long. It is split into {d} equal sections.", "How long is each section?", "m", 200, 3000),
)


def _easy(rng):
    for _ in attempts(500, "a quotient with a zero digit"):
        d = rng.randint(2, 9)
        q = (
            rng.randint(1, 9) * 100 + rng.randint(1, 9)
            if rng.random() < 0.7
            else rng.randint(1, 9) * 1000 + rng.randint(1, 9) * 10
        )
        n = q * d
        if 100 <= n <= 9999 and "0" in str(q)[1:] and not str(q).endswith("00"):
            break
    check(divmod(n, d) == (q, 0), "the dividend does not divide exactly")
    text = phrase(
        rng,
        [
            "A school orders {n} exercise books to share equally among {d} classes. How many books does each class get?",
            "{n} stickers are packed equally into {d} boxes. How many stickers are in each box?",
            "A farm packs {n} eggs into {d} equal crates. How many eggs are in each crate?",
        ],
        n=grouped(n),
        d=d,
    )
    skipped = int(str(q).replace("0", "", 1))
    cands = [
        int_dist(skipped, "fr2b-division-zero-in-quotient-left-out"),
        int_dist(q * 10, "fr2b-division-quotient-place-value-wrong"),
        int_dist(q + 10, "fr2b-division-regrouping-slip"),
        int_dist(q - 10, "fr2b-division-regrouping-slip"),
    ]
    working = [
        f"Divide {grouped(n)} by {d} one place at a time. A step that gives 0 still writes a 0 in the answer.",
        f"{grouped(q)} x {d} = {grouped(n)}, so each share is {grouped(q)}.",
    ]
    return dict(
        text=text,
        latex="",
        answer_kind="number",
        answer=q,
        answer_text=str(q),
        answer_latex=str(q),
        working=working,
        distractors=tagged(str(q), cands),
    )


def _draw_with_remainder(rng, divisors, low, high):
    for _ in attempts(500, "a dividend with a remainder"):
        d = rng.choice(divisors)
        n = rng.randint(low, high)
        q, r = divmod(n, d)
        if r and q > 9:
            return n, d, q, r
    raise AssertionError  # pragma: no cover


def _measure_text(measure, n, d, tail):
    setting, question, unit, _low, _high = measure
    return setting.format(n=grouped(n), d=d) + " " + question + " " + tail, unit


def _medium(rng):
    measure = rng.choice(_MEASURES)
    n, d, q, r = _draw_with_remainder(rng, (3, 4, 6, 7, 8, 9), measure[3], measure[4])
    value = Fraction(n, d)
    check(value == q + Fraction(r, d) and q * d + r == n, "quotient and remainder disagree")
    answer_text = mixed_text(value)
    text, unit = _measure_text(measure, n, d, "Write the answer as a mixed numeral.")
    cands = [
        Distractor(f"{q} {r}/{q}", "fr2b-remainder-written-over-the-quotient", exact_value(q + Fraction(r, q))),
        Distractor(f"{q} {r}/{n}", "fr2b-remainder-written-over-the-dividend", exact_value(q + Fraction(r, n))),
        Distractor(f"{q} {d}/{r}", "fr2b-remainder-and-divisor-swapped", exact_value(q + Fraction(d, r))),
        Distractor(str(q + 1), "fr2b-remainder-dropped-or-rounded-up", q + 1),
        Distractor(str(q), "fr2b-remainder-dropped-or-rounded-up", q),
    ]
    cands = [c for c in cands if _as_fraction(c.value) != value]
    working = [
        f"{grouped(n)} / {d} = {q} remainder {r}, because {q} x {d} + {r} = {grouped(n)}.",
        f"The {r} left over is shared into {d} parts, so it is {r}/{d} of a share.",
        f"Each share is {answer_text} {unit}.",
    ]
    return dict(
        text=text,
        latex="",
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=tagged(answer_text, cands),
    )


def _as_fraction(value) -> Fraction:
    return Fraction(value["p"], value["q"]) if isinstance(value, dict) else Fraction(value)


def _decimal(value: Fraction, places: int) -> str:
    return stated_precision_string(round_half_up(value, places), places)


def _hard(rng):
    measure = rng.choice(_MEASURES)
    n, d, q, r = _draw_with_remainder(rng, (2, 4, 5, 8), measure[3], measure[4])
    value = Fraction(n, d)
    places = 0
    while (value * 10**places).denominator != 1:
        places += 1
        check(places <= 3, "the quotient does not terminate within three places")
    check(places >= 1, "the answer must have a decimal part")
    answer_text = _decimal(value, places)
    check(Fraction(answer_text) == value and Fraction(q) + Fraction(r, d) == value, "decimal key is not the quotient")
    text, unit = _measure_text(measure, n, d, "Write the answer as a decimal.")
    digits_after = Fraction(f"{q}.{r}")
    cands = [
        Distractor(
            _decimal(digits_after, places),
            "fr2b-remainder-written-as-decimal-digits",
            float(_decimal(digits_after, places)),
        ),
        Distractor(_decimal(value * 10, places), "fr2b-decimal-point-misplaced", float(_decimal(value * 10, places))),
        Distractor(
            _decimal(Fraction(q), places), "fr2b-remainder-dropped-or-rounded-up", float(_decimal(Fraction(q), places))
        ),
        Distractor(
            _decimal(Fraction(q + 1), places),
            "fr2b-remainder-dropped-or-rounded-up",
            float(_decimal(Fraction(q + 1), places)),
        ),
    ]
    working = [
        f"{grouped(n)} / {d} = {q} remainder {r}.",
        f"The remainder is {r}/{d} = {_decimal(Fraction(r, d), places)} of a share.",
        f"Each share is {answer_text} {unit}.",
    ]
    return dict(
        text=text,
        latex="",
        answer_kind="number",
        answer=float(answer_text),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=tagged(answer_text, cands),
    )


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fr2b_written_division", level=5, presets={tier: dict(tier=tier) for tier in _TIERS})
def fr2b_written_division(rng, tier="easy"):
    """Short and long division by a one-digit divisor, with the remainder as a fraction or a decimal."""
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return build(_BUILDERS[tier], rng, tier)
