"""Exam chain for flat-rate loans: find the term from the repayment (four parts).

A sum ``B`` is borrowed at a flat annual rate ``r`` (of the amount borrowed, every year) and repaid in equal monthly
repayments ``R``. The stem does not give the length of the loan.

(a) the interest charged in one year, ``B r``;
(b) how much of a year's repayments reduces the amount owing, ``12 R - part_a`` (follow-through);
(c) the length of the loan in whole years, ``B / part_b`` (follow-through);
(d) the total interest over the loan, ``part_a * part_c`` (follow-through).

The loan is built forwards from a whole number of years ``t`` and the repayment is ``B (1 + r t) / (12 t)``. A draw is
kept only when the repayment is a whole number of cents, so the stated figure is exact.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from mathgen.slots import NAMES

_ITEMS = (
    "a used car",
    "a caravan",
    "a second-hand boat",
    "a motorbike",
    "a rooftop solar system",
    "a kitchen renovation",
    "a ute",
)
_STEM = (
    "{name} borrows ${b} to pay for {item}. The loan charges flat interest of {r}% per year on the amount borrowed, "
    "and is repaid in equal monthly repayments of ${rep}.",
    "{name} takes out a flat-interest loan of ${b} for {item}, charged at {r}% per year of the amount borrowed. "
    "The loan is repaid with equal monthly repayments of ${rep}.",
)


def _fmt(v: Fraction, dp: int) -> str:
    return stated_precision_string(round_half_up(v, dp), dp)


def _fmtc(v: Fraction, dp: int) -> str:
    """`_fmt` with thousands commas, for dollar amounts in the working (the stem writes ``$20,000``)."""
    whole, _, frac = _fmt(v, dp).partition(".")
    return f"{int(whole):,}" + (f".{frac}" if frac else "")


def _dists(key: Fraction, pairs, dp: int, minimum: int = 2):
    """Each wrong value written to the key's own places, so the format never gives the key away."""
    cands = [Distractor(_fmt(v, dp), tag, exact_value(round_half_up(v, dp))) for tag, v in pairs]
    out = pick_distractors(_fmt(key, dp), cands, 3, tolerance=stated_precision_tolerance(max(dp, 1)))
    check(len(out) >= minimum, "too few distractors")
    return out


def _part(prompt, key: Fraction, dp: int, marks: int, working, dists, ft=None):
    text = _fmt(key, dp)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(round_half_up(key, dp)),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=dists,
    )
    if ft:
        part["follow_through"] = ft
    return part


@generator("exd_x_r2d_flat_loan_term_chain", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r2d_flat_loan_term_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Annual interest, principal repaid per year, the term in years and the total interest on a flat loan."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(4000, "a flat loan with a whole-cent repayment"):
        b = Fraction(rng.randint(6, 40) * 1000)
        r = rng.randint(3, 9)
        t = rng.randint(2, 7)
        rep = b * (100 + r * t) / (1200 * t)
        if (rep * 100).denominator != 1:
            continue
        interest_year = b * r / 100
        principal_year = 12 * rep - interest_year
        if principal_year <= 0 or b / principal_year != t:
            continue
        break
    total_interest = interest_year * t
    # Independent route: the total repaid, month by month, equals the amount borrowed plus the interest.
    check(12 * t * rep == b + total_interest, "total repaid disagrees with principal plus interest")
    check(principal_year * t == b, "principal repayments over the term do not clear the loan")

    a_pairs = [
        ("used-monthly-figure-as-annual", interest_year / 12),
        ("gave-total-not-interest", b + interest_year),
    ]
    b_pairs = [
        ("ignored-the-interest", 12 * rep),
        ("used-monthly-figure-as-annual", rep - interest_year / 12),
        ("substitution-sign-error", 12 * rep + interest_year),
        ("used-monthly-figure-as-annual", rep),
    ]
    c_pairs = [
        ("ignored-the-interest", b / (12 * rep)),
        ("substitution-sign-error", b / (12 * rep + interest_year)),
        ("used-monthly-figure-as-annual", b / rep),
        ("ignored-the-interest", b / (rep * 12 - interest_year / 12)),
        ("wrong-number-of-periods", Fraction(12 * t)),
        ("off-by-one-count", Fraction(t + 1)),
        ("off-by-one-count", Fraction(t - 1)),
    ]
    d_pairs = [
        ("gave-total-not-interest", 12 * rep * t),
        ("wrong-number-of-periods", interest_year),
        ("ignored-the-interest", interest_year * (b / (12 * rep))),
        ("gave-total-not-interest", b + total_interest),
    ]
    stem = rng.choice(_STEM).format(
        name=rng.choice(NAMES), b=f"{int(b):,}", item=rng.choice(_ITEMS), r=r, rep=_fmtc(rep, 2)
    )
    return dict(
        text=stem,
        parts=[
            _part(
                "Find the interest charged on the loan each year, in dollars.",
                interest_year,
                2,
                1,
                [f"Interest per year = {r}% of {int(b)} = ${_fmtc(interest_year, 2)}."],
                _dists(interest_year, a_pairs, 2),
            ),
            _part(
                "Hence find how much of the repayments made in one year goes towards reducing the amount owing "
                "(the part that is not interest), in dollars.",
                principal_year,
                2,
                2,
                [
                    f"Repaid per year = 12 × {_fmt(rep, 2)} = ${_fmtc(12 * rep, 2)}.",
                    f"Less interest: ${_fmtc(principal_year, 2)}.",
                ],
                _dists(principal_year, b_pairs, 2),
                ft=f"12*{_fmt(rep, 2)} - part_a",
            ),
            _part(
                "Hence find how many years it takes to repay the loan.",
                Fraction(t),
                0,
                2,
                [f"Years = {int(b)} ÷ {_fmt(principal_year, 2)} = {t}."],
                _dists(Fraction(t), c_pairs, 0),
                ft=f"{int(b)}/part_b",
            ),
            _part(
                "Hence find the total interest paid over the whole loan, in dollars.",
                total_interest,
                2,
                1,
                [f"Total interest = {_fmt(interest_year, 2)} × {t} = ${_fmtc(total_interest, 2)}."],
                _dists(total_interest, d_pairs, 2),
                ft="part_a*part_c",
            ),
        ],
    )
