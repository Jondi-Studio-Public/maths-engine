"""Generates `fr2b_fraction_decimal_links` questions for `f10-fraction-decimal-links` (AC9M4N03). `answer: number`.

Scope: tenths and hundredths written as fractions and as decimals, moved between the two and combined in one chain.
Composing a decimal from a stated number of tenths and hundredths is the neighbouring skill
`f10-decimal-notation-tenths-hundredths`; this skill never asks for that.

Ladder (reasoning escalation, not size):
  easy:   a decimal is written as a fraction (answer kind `fraction`, a decimal with tenths and/or hundredths). The
          place of the last digit decides the denominator, so 0.7 and 0.07 are different fractions.
  medium: tenths and hundredths written as fractions with different denominators are added or subtracted and the
          result is written as a decimal (answer kind `number`, always two decimal places).
  hard:   three amounts in three forms (a tenths fraction, a decimal, a hundredths fraction) are combined and the
          total is counted in hundredths (answer kind `number`, a whole number).

Every key is computed with exact `Fraction`s, then re-derived by integer arithmetic in hundredths. Every distractor
is the answer of one named error and is checked against the key by value.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, phrase, round_half_up, stated_precision_string
from ._fr2b_kit import build, int_dist, tagged

_TIERS = ("easy", "medium", "hard")


def _decimal_text(hundredths: int) -> str:
    """Hundredths (1..99) as a decimal below one: 7 -> "0.07", 40 -> "0.4", 43 -> "0.43"."""
    whole, rest = divmod(hundredths, 100)
    digits = f"{rest:02d}".rstrip("0")
    return f"{whole}.{digits}" if digits else str(whole)


def _frac_dist(num: int, den: int, misconception: str) -> Distractor:
    return Distractor(f"{num}/{den}", misconception, exact_value(Fraction(num, den)))


def _easy(rng):
    kind = rng.choice(("tenths", "hundredths-small", "hundredths", "hundredths"))
    if kind == "tenths":
        hundredths = rng.randint(1, 9) * 10
    elif kind == "hundredths-small":
        hundredths = rng.randint(1, 9)
    else:
        for _ in attempts(100, "two non-zero digits"):
            hundredths = rng.randint(11, 99)
            if hundredths % 10:
                break
    den = 10 if hundredths % 10 == 0 else 100
    num = hundredths // 10 if den == 10 else hundredths
    decimal = _decimal_text(hundredths)
    value = Fraction(hundredths, 100)
    check(
        Fraction(int(decimal.split(".")[1]), 10 ** len(decimal.split(".")[1])) == value, "decimal and fraction differ"
    )
    answer = Fraction(num, den)
    check(answer == value, "the fraction does not equal the decimal")
    text = phrase(
        rng,
        [
            "Write {d} as a fraction.",
            "Write the decimal {d} as a fraction with a denominator of 10 or 100.",
            "Which fraction is the same as the decimal {d}? Write it with a denominator of 10 or 100.",
        ],
        d=decimal,
    )
    other_den = 100 if den == 10 else 10
    cands = [_frac_dist(num, other_den, "fr2b-decimal-place-gives-wrong-denominator")]
    if den == 100 and hundredths >= 10:
        cands.append(_frac_dist(hundredths // 10, hundredths % 10, "fr2b-decimal-digits-split-into-a-fraction"))
        cands.append(_frac_dist(hundredths, 10, "fr2b-decimal-place-gives-wrong-denominator"))
        cands.append(_frac_dist(hundredths, 1000, "fr2b-decimal-place-gives-wrong-denominator"))
    elif den == 100:
        cands.append(_frac_dist(1, hundredths, "fr2b-decimal-digits-split-into-a-fraction"))
        cands.append(_frac_dist(hundredths, 1000, "fr2b-decimal-place-gives-wrong-denominator"))
        cands.append(_frac_dist(hundredths, 1, "fr2b-decimal-point-ignored"))
    else:
        cands.append(_frac_dist(num, 1000, "fr2b-decimal-place-gives-wrong-denominator"))
        cands.append(_frac_dist(num, 1, "fr2b-decimal-point-ignored"))
        if num > 1:
            cands.append(_frac_dist(1, num, "fr2b-decimal-digits-split-into-a-fraction"))
    cands = [c for c in cands if Fraction(c.text) != answer]
    answer_text = f"{num}/{den}"
    working = [
        f"The last digit of {decimal} is in the {'tenths' if den == 10 else 'hundredths'} place, so the denominator is {den}.",
        f"{decimal} is {num} out of {den}, which is {answer_text}.",
    ]
    return dict(
        text=text,
        latex="",
        answer_kind="fraction",
        answer=exact_value(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=tagged(answer_text, cands),
    )


def _two_dp(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 2), 2)


def _num_dist(value: Fraction, misconception: str) -> Distractor:
    text = _two_dp(value)
    return Distractor(text, misconception, float(text))


def _medium(rng):
    for _ in attempts(500, "tenths and hundredths with a two-place result"):
        tenths = rng.randint(1, 8)
        hundredths = rng.randint(11, 79)
        if hundredths % 10 == 0:
            continue
        subtract = rng.random() < 0.4
        total = Fraction(tenths, 10) + (-1 if subtract else 1) * Fraction(hundredths, 100)
        if 0 < total < 1 and (total * 100) % 10 != 0:
            break
    check((total * 100).denominator == 1, "the result must be a whole number of hundredths")
    # Second route: everything in hundredths.
    in_hundredths = tenths * 10 + (-hundredths if subtract else hundredths)
    check(Fraction(in_hundredths, 100) == total, "hundredths route disagrees")
    answer_text = _two_dp(total)
    op = "-" if subtract else "+"
    text = phrase(
        rng,
        [
            "Work out {t}/10 {op} {h}/100. Write the answer as a decimal.",
            "Write {t}/10 {op} {h}/100 as a decimal.",
        ],
        t=tenths,
        h=hundredths,
        op=op,
    )
    signed = Fraction(hundredths, 100) * (-1 if subtract else 1)
    cands = [
        _num_dist(
            Fraction(abs(tenths + (-hundredths if subtract else hundredths)), 100), "fr2b-tenths-added-as-hundredths"
        ),
        _num_dist(Fraction(tenths, 10), "fr2b-ignores-the-hundredths-part"),
        _num_dist(Fraction(hundredths, 100), "fr2b-ignores-the-tenths-part"),
        _num_dist(Fraction(tenths, 10) + signed * 10, "fr2b-hundredths-read-as-tenths"),
    ]
    cands = [c for c in cands if not c.text.startswith("-")]
    working = [
        f"{tenths}/10 is {tenths * 10}/100.",
        f"{tenths * 10}/100 {op} {hundredths}/100 = {abs(in_hundredths)}/100 = {answer_text}.",
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


def _hard(rng):
    for _ in attempts(500, "three amounts that total under two"):
        tenths = rng.randint(1, 9)
        decimal_h = rng.randint(11, 89)
        if decimal_h % 10 == 0:
            continue
        hundredths = rng.randint(1, 49)
        total = tenths * 10 + decimal_h + hundredths
        if total < 200:
            break
    check(total == int(Fraction(tenths, 10) * 100 + Fraction(decimal_h, 100) * 100 + hundredths), "total differs")
    decimal = _decimal_text(decimal_h)
    text = phrase(
        rng,
        [
            "How many hundredths are there altogether in {t}/10 + {d} + {h}/100?",
            "Add {t}/10, the decimal {d} and {h}/100. Give the total as a number of hundredths.",
        ],
        t=tenths,
        d=decimal,
        h=hundredths,
    )
    cands = [
        int_dist(tenths + decimal_h + hundredths, "fr2b-tenths-counted-as-hundredths"),
        int_dist(tenths * 100 + decimal_h + hundredths, "fr2b-tenths-counted-as-hundreds-of-hundredths"),
        int_dist(tenths * 10 + decimal_h // 10 + hundredths, "fr2b-decimal-counted-in-tenths"),
        int_dist(total * 10, "fr2b-counts-thousandths-not-hundredths"),
    ]
    working = [
        f"{tenths}/10 is {tenths * 10} hundredths, and {decimal} is {decimal_h} hundredths.",
        f"{tenths * 10} + {decimal_h} + {hundredths} = {total} hundredths.",
    ]
    return dict(
        text=text,
        latex="",
        answer_kind="number",
        answer=total,
        answer_text=str(total),
        answer_latex=str(total),
        working=working,
        distractors=tagged(str(total), cands),
    )


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fr2b_fraction_decimal_links", level=4, presets={tier: dict(tier=tier) for tier in _TIERS})
def fr2b_fraction_decimal_links(rng, tier="easy"):
    """Decimals and fractions of tenths and hundredths, moved between and combined; tiers in the module docstring."""
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return build(_BUILDERS[tier], rng, tier)
