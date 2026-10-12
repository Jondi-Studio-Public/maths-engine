"""Exam source for `senior-use-technology-to-compare-numerically-and-graphically-the`: a four-part chain that
compares a simple-interest account with a compound-interest account whose rate has to be recovered first.

The stem describes two investments. The simple-interest account is described by its principal and the dollars it
earns each year. The compound account is described by its principal and its balance after the first year, so its
rate is not stated.

- (a) the annual rate of the compound account, as a percentage (2 marks);
- (b) its balance after ``n`` years, to the nearest dollar (2 marks, ``follow_through`` from (a));
- (c) the balance of the simple-interest account after ``n`` years (1 mark);
- (d) the first whole year at the end of which the compound balance exceeds the simple balance of (c)
  (3 marks, ``follow_through`` from (a) and (c), through a logarithm).

## Construction

The principals are whole thousands and the rate has one non-zero decimal place, so the first-year balance is a
whole number of dollars and the rate can be read back exactly. Every balance is computed in exact rational
arithmetic and rounded once. The compound balance is drawn to be below the simple balance at year ``n``, so (d)
is a genuine catch-up question with an answer later than ``n``. The crossing year is found by search over whole
years, then confirmed against the logarithm that the follow-through expression uses, which must sit clear of a
whole number so that float and exact routes cannot disagree.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    attempts,
    check,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_PEOPLE = [("Wei", "his"), ("Owen", "his"), ("Deng", "his"), ("Tom", "his")]
_PEOPLE_B = [("Zara", "her"), ("Priya", "her"), ("Mia", "her"), ("Yara", "her")]
_LAST_YEAR = 45


def _money(x: int) -> str:
    return f"${x:,}"


def _first_year_above(principal: int, growth: Fraction, target: int) -> int:
    """Smallest whole year t with principal * growth**t > target, by search."""
    balance = Fraction(principal)
    for year in range(1, 200):
        balance *= growth
        if balance > target:
            return year
    raise AssertionError("the balance never exceeds the target")


def _distractors(items, canonical, tolerance=0):
    """Tagged candidates with the canonical answer's text removed (by `pick_distractors`) and values kept."""
    return pick_distractors(canonical, items, 3, tolerance=tolerance)


@generator("exd_x_r2e_interest_crossover", level=12, presets={"exam": dict(tier="exam")})
def interest_crossover(rng: random.Random, tier: str = "exam") -> dict:
    """Recover a compound rate, then use it for a balance, and for the year it catches up with a simple account."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(4000, "an interest chain with a clear catch-up year"):
        n = rng.randint(6, 12)
        p_simple = 1000 * rng.randint(20, 60)
        rate_simple = rng.randint(3, 8)
        p_comp = 1000 * rng.randint(10, int(p_simple * 0.8 / 1000))
        r10 = rng.randint(41, 95)
        if r10 % 10 == 0:
            continue
        yearly = p_simple * rate_simple // 100
        growth = 1 + Fraction(r10, 1000)
        first_year = p_comp * growth
        if first_year.denominator != 1:
            continue
        comp_n = p_comp * growth**n
        simple_n = p_simple + yearly * n
        if not comp_n < simple_n:
            continue
        year = _first_year_above(p_comp, growth, simple_n)
        if not n + 2 <= year <= _LAST_YEAR:
            continue
        # The two routes to the catch-up year (search, and the logarithm that follow-through uses) must agree
        # with room to spare.
        x = math.log(simple_n / p_comp) / math.log(float(growth))
        if abs(x - round(x)) < 1e-4 or math.floor(x) + 1 != year:
            continue
        frac = (comp_n * 2) % 2  # distance from a half-dollar rounding boundary
        if abs(frac - 1) < Fraction(1, 1000):
            continue
        break

    rate = Fraction(r10, 10)
    comp_n_dollars = int(round_half_up(comp_n))
    check(abs(float(comp_n) - comp_n_dollars) <= 0.5, "balance rounding disagrees")
    interest_1 = int(first_year) - p_comp

    (wei, _), (zara, zpron) = rng.choice(_PEOPLE), rng.choice(_PEOPLE_B)
    stem = (
        f"{wei} invests {_money(p_simple)} in an account that pays simple interest and earns {_money(yearly)} in "
        f"interest each year. {zara} invests {_money(p_comp)} in an account that pays compound interest, "
        f"compounded annually. At the end of the first year {zpron} balance is {_money(int(first_year))}. "
        f"Neither makes any deposits or withdrawals."
    )

    rate_text = stated_precision_string(rate, 1)
    ratio = Fraction(int(first_year), p_comp)  # the growth factor, e.g. 1.064
    share_of_final = round_half_up(Fraction(interest_1 * 100, int(first_year)), 1)
    cand_a = [
        Distractor(str(interest_1), "gave-change-not-percent", interest_1),
        Distractor(stated_precision_string(share_of_final, 1), "divides-by-final", share_of_final),
        Distractor(f"{float(ratio):.3f}", "factor-given-as-rate", ratio),
        Distractor(str(rate_simple), "mixes-up-the-two-accounts", rate_simple),
    ]
    d_a = _distractors(cand_a, rate_text, stated_precision_tolerance(1))

    cand_b = [
        Distractor(
            str(int(round_half_up(p_comp * (1 + n * Fraction(r10, 1000))))),
            "uses-simple-interest",
            int(round_half_up(p_comp * (1 + n * Fraction(r10, 1000)))),
        ),
        Distractor(
            str(int(round_half_up(p_comp * growth ** (n - 1)))),
            "off-by-one-count",
            int(round_half_up(p_comp * growth ** (n - 1))),
        ),
        Distractor(
            str(int(round_half_up(p_comp * growth ** (n + 1)))),
            "off-by-one-count",
            int(round_half_up(p_comp * growth ** (n + 1))),
        ),
        Distractor(
            str(int(round_half_up(comp_n - p_comp))),
            "interest-instead-of-balance",
            int(round_half_up(comp_n - p_comp)),
        ),
    ]
    d_b = _distractors(cand_b, str(comp_n_dollars))

    wg = 1 + Fraction(rate_simple, 100)
    cand_c = [
        Distractor(
            str(int(round_half_up(p_simple * wg**n))), "treats-simple-as-compound", int(round_half_up(p_simple * wg**n))
        ),
        Distractor(str(yearly * n), "interest-instead-of-balance", yearly * n),
        Distractor(str(p_simple + yearly * (n - 1)), "off-by-one-count", p_simple + yearly * (n - 1)),
        Distractor(str(p_simple + yearly * (n + 1)), "off-by-one-count", p_simple + yearly * (n + 1)),
    ]
    d_c = _distractors(cand_c, str(simple_n))

    # (d) distractors: truncating the logarithm, one year late, stopping at year n, and a simple-interest model.
    simple_t = math.ceil((simple_n / p_comp - 1) / (r10 / 1000))
    cand_d = [
        Distractor(str(year - 1), "rounds-down-when-must-exceed", year - 1),
        Distractor(str(year + 1), "adds-a-year-after-rounding-up", year + 1),
        Distractor(str(simple_t), "uses-simple-interest", simple_t),
        Distractor(str(n), "stops-at-the-comparison-year", n),
    ]
    d_d = _distractors(cand_d, str(year))
    for ds, key in ((d_a, rate), (d_b, comp_n_dollars), (d_c, simple_n), (d_d, year)):
        check(ds and all(str(d) != str(key) for d in ds), "a distractor equals its key")

    # Follow-through, written with this question's numbers: Zara's balance from her rate, and the catch-up year
    # from her rate and Wei's balance.
    g = "(1 + part_a/100)"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the annual interest rate of {zara}'s account, as a percentage.",
                answer_kind="number",
                answer=float(rate),
                answer_text=rate_text,
                answer_latex=latex_answer(rate_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Interest in year 1 = {_money(int(first_year))} - {_money(p_comp)} = {_money(interest_1)}.",
                    f"Rate = {interest_1} / {p_comp} x 100 = {rate_text}%.",
                ],
                distractors=d_a,
            ),
            dict(
                prompt=f"Find the balance of {zara}'s account at the end of year {n}, to the nearest dollar.",
                answer_kind="number",
                answer=comp_n_dollars,
                answer_text=str(comp_n_dollars),
                answer_latex=latex_answer(str(comp_n_dollars)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{p_comp}*{g}**{n}",
                working=[
                    f"Balance = {p_comp} x (1 + {rate_text}/100)^{n} = {_money(comp_n_dollars)} to the nearest dollar."
                ],
                distractors=d_b,
            ),
            dict(
                prompt=f"Find the balance of {wei}'s account at the end of year {n}.",
                answer_kind="number",
                answer=simple_n,
                answer_text=str(simple_n),
                answer_latex=latex_answer(str(simple_n)),
                marks=1,
                sensible_range=(0, None),
                working=[f"Balance = {p_simple} + {n} x {yearly} = {_money(simple_n)}."],
                distractors=d_c,
            ),
            dict(
                prompt=f"{zara} leaves {zpron} money invested for longer. Find the first whole year at the end of which "
                f"{zpron} balance is greater than {wei}'s balance at the end of year {n}.",
                answer_kind="number",
                answer=year,
                answer_text=str(year),
                answer_latex=latex_answer(str(year)),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"floor(log(part_c/{p_comp})/log({g})) + 1",
                working=[
                    f"Need {p_comp} x (1 + {rate_text}/100)^t > {simple_n}.",
                    f"t > ln({simple_n}/{p_comp}) / ln({growth}) = {x:.3f}, so t = {year}.",
                ],
                distractors=d_d,
            ),
        ],
    )
