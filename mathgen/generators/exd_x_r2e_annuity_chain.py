"""Exam source for `senior-with-the-aid-of-a-financial-calculator-or-acmgm100`: a four-part annuity chain, saving
and then living off the savings.

One person pays a fixed amount into an account at the end of every month for some years (the interest rate is
compounded monthly), then withdraws equal monthly amounts, at the same rate, until the balance is exactly used up.

- (a) the balance when the saving stops, to the nearest dollar (2 marks);
- (b) the monthly withdrawal that uses that balance up over the stated number of years, to the nearest dollar
  (3 marks, ``follow_through`` from (a));
- (c) the total withdrawn (1 mark, ``follow_through`` from (b));
- (d) the interest the account earns while it is being drawn down (1 mark, ``follow_through`` from (a) and (c)).

## Construction

The annual rate is a whole number of tenths of a percent that is a multiple of three, so the monthly rate is a
terminating decimal and the follow-through expressions can be written with it exactly. All amounts are computed
in exact rational arithmetic. (b) is computed from the (a) a learner has in hand, the balance rounded to a
dollar, and the draw is rejected unless the exact balance gives the same rounded withdrawal and neither value
is close to a rounding boundary.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, pick_distractors, round_half_up

_NAMES = ["Priya", "Ben", "Aisha", "Tom", "Mei", "Lachlan", "Sofia", "Kofi"]
_MARGIN = Fraction(1, 200)


def _money(x: int) -> str:
    return f"${x:,}"


def _near_half(x: Fraction) -> bool:
    return abs(x % 1 - Fraction(1, 2)) < _MARGIN


def _dollars(x: Fraction) -> int:
    return int(round_half_up(x))


def _tag(value: int, tag: str) -> Distractor:
    return Distractor(str(value), tag, value)


@generator("exd_x_r2e_annuity_chain", level=12, presets={"exam": dict(tier="exam")})
def annuity_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Balance of a savings annuity, the withdrawal it supports, the total drawn and the interest earned."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(4000, "an annuity chain clear of rounding boundaries"):
        deposit = 50 * rng.randint(2, 18)
        years_in = rng.randint(15, 35)
        years_out = rng.randint(15, 25)
        r10 = 3 * rng.randint(11, 25)  # 3.3% .. 7.5%
        i = Fraction(r10, 12000)
        n_in, n_out = 12 * years_in, 12 * years_out
        balance = deposit * ((1 + i) ** n_in - 1) / i
        pay_exact = balance * i / (1 - (1 + i) ** (-n_out))
        a = _dollars(balance)
        pay_from_a = a * i / (1 - (1 + i) ** (-n_out))
        b = _dollars(pay_from_a)
        if _near_half(balance) or _near_half(pay_from_a) or _dollars(pay_exact) != b:
            continue
        c = b * n_out
        d = c - a
        if d <= 0:
            continue
        break

    name = rng.choice(_NAMES)
    rate_text = f"{r10 / 10:g}"
    i_text = f"{float(i):.10g}"
    stem = (
        f"{name} pays {_money(deposit)} at the end of every month into a retirement account for {years_in} years. "
        f"The account earns {rate_text}% p.a., compounded monthly. When the payments stop, {name} starts "
        f"withdrawing an equal amount at the end of every month for {years_out} years, so that the balance is "
        f"exactly used up. The interest rate stays the same throughout."
    )

    ordinary = deposit * n_in
    a_cands = [
        _tag(ordinary, "ignores-compounding"),
        _tag(_dollars(deposit * ((1 + i) ** years_in - 1) / i), "months-and-years-confused"),
        _tag(_dollars(deposit * (1 - (1 + i) ** (-n_in)) / i), "present-value-for-future-value"),
        _tag(_dollars(deposit * (1 + i) ** n_in), "annuity-as-single-deposit"),
    ]
    b_cands = [
        _tag(_dollars(Fraction(a, n_out)), "ignored-the-interest"),
        _tag(_dollars(a * i), "perpetuity-for-annuity"),
        _tag(_dollars(a * i / (1 - (1 + i) ** (-n_in))), "wrong-number-of-periods"),
        _tag(_dollars(Fraction(a, years_out)), "months-and-years-confused"),
    ]
    c_cands = [
        _tag(b * years_out, "months-and-years-confused"),
        _tag(b * (n_out - 1), "off-by-one-count"),
        _tag(a, "total-as-starting-balance"),
        _tag(b * n_in, "wrong-number-of-periods"),
    ]
    d_cands = [
        _tag(c + a, "adds-instead-of-subtracts"),
        _tag(c - ordinary, "compares-with-deposits-not-balance"),
        _tag(c, "gave-total-not-interest"),
        _tag(a - c, "sign-reversed-result"),
    ]
    parts_d = [
        pick_distractors(str(key), cands, 3) for key, cands in ((a, a_cands), (b, b_cands), (c, c_cands), (d, d_cands))
    ]
    check(all(parts_d), "a part has no distractors")
    pay = f"(1 - (1 + {i_text})**(-{n_out}))"

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the balance of the account when the payments stop, to the nearest dollar.",
                answer_kind="number",
                answer=a,
                answer_text=str(a),
                answer_latex=latex_answer(str(a)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Monthly rate i = {rate_text}/12 % = {i_text}; n = {n_in}.",
                    f"Balance = {deposit} x ((1 + i)^{n_in} - 1) / i = {_money(a)}.",
                ],
                distractors=parts_d[0],
            ),
            dict(
                prompt="Hence find the amount of each monthly withdrawal, to the nearest dollar.",
                answer_kind="number",
                answer=b,
                answer_text=str(b),
                answer_latex=latex_answer(str(b)),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"part_a*{i_text}/{pay}",
                working=[
                    f"The balance {_money(a)} is the present value of {n_out} withdrawals.",
                    f"Withdrawal = {a} x i / (1 - (1 + i)^-{n_out}) = {_money(b)}.",
                ],
                distractors=parts_d[1],
            ),
            dict(
                prompt=f"Using your answer to (b), find the total amount withdrawn over the {years_out} years.",
                answer_kind="number",
                answer=c,
                answer_text=str(c),
                answer_latex=latex_answer(str(c)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{n_out}*part_b",
                working=[f"Total = {n_out} x {b} = {_money(c)}."],
                distractors=parts_d[2],
            ),
            dict(
                prompt=f"Using your answers to (a) and (c), find the interest the account earns while {name} is withdrawing.",
                answer_kind="number",
                answer=d,
                answer_text=str(d),
                answer_latex=latex_answer(str(d)),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_c - part_a",
                working=[f"Interest = total withdrawn - starting balance = {c} - {a} = {_money(d)}."],
                distractors=parts_d[3],
            ),
        ],
    )
