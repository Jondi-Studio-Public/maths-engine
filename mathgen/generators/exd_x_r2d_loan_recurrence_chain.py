"""Exam chain for a loan recurrence ``B(n+1) = B(n)(1 + r) - R`` with a change of rate (four parts).

A reducing-balance home loan is repaid monthly. After ``n1`` months the interest rate changes and the repayment is
unchanged.

(a) the balance after ``n1`` months, to the nearest cent;
(b) the balance ``m`` months after the change, from the learner's own (a) (follow-through, closed form);
(c) the monthly repayment that would clear that balance in exactly ``k`` further months at the new rate (follow-through
    from ``part_b`` through the annuity formula);
(d) the total interest charged over those ``k`` months, ``k * part_c - part_b`` (follow-through).

Every key after (a) is computed from the displayed value of the part before it, because the prompts tell the learner to
use their earlier answer; the first balance is the exact recurrence rounded once, at the cent. All arithmetic is exact
``Fraction``, so the closed forms and the month-by-month recurrence must agree.
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

_STEM = (
    "{name} takes out a ${p} home loan. Each month, interest of {r1}% is added to the balance and a repayment of "
    "${rep} is made, following B(n + 1) = B(n)(1 + r) − R, where r is the monthly interest rate as a decimal. "
    "After {n1} months the interest rate {dir} to {r2}% per month, and the repayment stays the same.",
    "A ${p} home loan taken out by {name} follows B(n + 1) = B(n)(1 + r) − R, with r the monthly rate as a decimal. "
    "The rate is {r1}% per month and the repayment is ${rep} each month. After {n1} months the rate {dir} to "
    "{r2}% per month; the repayment does not change.",
)


def _fmt(v: Fraction, dp: int = 2) -> str:
    return stated_precision_string(round_half_up(v, dp), dp)


def _dists(key: Fraction, pairs):
    cands = [Distractor(_fmt(v), tag, exact_value(round_half_up(v, 2))) for tag, v in pairs if v > 0]
    out = pick_distractors(_fmt(key), cands, 3, tolerance=stated_precision_tolerance(2))
    check(len(out) >= 2, "too few distractors")
    return out


def _part(prompt, key: Fraction, marks: int, working, dists, ft=None):
    text = _fmt(key)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(round_half_up(key, 2)),
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


def _step(b: Fraction, r: Fraction, rep: Fraction, n: int) -> Fraction:
    """n applications of the recurrence, month by month (the independent route)."""
    for _ in range(n):
        b = b * (1 + r) - rep
    return b


def _closed(b: Fraction, r: Fraction, rep: Fraction, n: int) -> Fraction:
    g = (1 + r) ** n
    return b * g - rep * (g - 1) / r


def _annuity(b: Fraction, r: Fraction, k: int) -> Fraction:
    return b * r / (1 - (1 + r) ** (-k))


def _frac_text(x: Fraction) -> str:
    return f"({x.numerator}/{x.denominator})"


@generator("exd_x_r2d_loan_recurrence_chain", level=14, presets={"exam": dict(tier="exam")})
def exd_x_r2d_loan_recurrence_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Balance after a first phase, after a rate change, the repayment to clear it, and the interest."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a loan chain with distinct answers"):
        p = Fraction(rng.randint(12, 48) * 10000)
        r1 = Fraction(rng.randint(8, 12), 2000)  # 0.40% .. 0.60% per month
        shift = Fraction(rng.choice((-3, -2, 2, 3, 4)), 2000)
        r2 = r1 + shift
        term = rng.choice((240, 300, 360))
        float_r = float(r1)
        rep = Fraction(round(float(p) * float_r / (1 - (1 + float_r) ** (-term))))
        n1, m, k = rng.choice((12, 18, 24, 36)), rng.choice((12, 18, 24)), rng.choice((60, 84, 120, 180))
        b1 = round_half_up(_closed(p, r1, rep, n1), 2)
        b2 = round_half_up(_closed(b1, r2, rep, m), 2)
        r_new = round_half_up(_annuity(b2, r2, k), 2)
        interest = k * r_new - b2
        if min(b1, b2, r_new, interest) <= 0:
            continue
        break
    # Independent routes: the recurrence month by month.
    check(round_half_up(_step(p, r1, rep, n1), 2) == b1, "first balance disagrees with the recurrence")
    check(round_half_up(_step(b1, r2, rep, m), 2) == b2, "second balance disagrees with the recurrence")
    cleared = _step(b2, r2, r_new, k)
    # The repayment is rounded to the cent, so the balance left can differ by half a cent compounded over the term.
    slack = Fraction(1, 200) * ((1 + r2) ** k - 1) / r2 + Fraction(1, 100)
    check(abs(cleared) <= slack, "the new repayment does not clear the balance")

    a_pairs = [
        ("repayment-before-interest", _alt_order(p, r1, rep, n1)),
        ("forgot-to-subtract-repayment", p * (1 + r1) ** n1),
        ("uses-simple-interest", p * (1 + r1 * n1) - rep * n1),
    ]
    b_pairs = [
        ("rate-change-not-applied", _closed(b1, r1, rep, m)),
        ("forgot-to-subtract-repayment", b1 * (1 + r2) ** m),
        ("restarted-from-the-original-principal", _closed(p, r2, rep, n1 + m)),
        ("repayment-before-interest", _alt_order(b1, r2, rep, m)),
    ]
    c_pairs = [
        ("ignored-the-interest", b2 / k),
        ("rate-change-not-applied", _annuity(b2, r1, k)),
        ("wrong-number-of-periods", _annuity(b2, r2, k + 12)),
        ("wrong-number-of-periods", _annuity(b2, r2, k - 12)),
    ]
    d_pairs = [
        ("gave-total-not-interest", k * r_new),
        ("uses-simple-interest", b2 * r2 * k),
        ("interest-from-wrong-base", k * r_new - p),
    ]
    direction = "rises" if r2 > r1 else "falls"

    def pct(x: Fraction) -> str:
        return str(float(x * 100)).rstrip("0").rstrip(".")

    stem = rng.choice(_STEM).format(
        name=rng.choice(NAMES), p=f"{int(p):,}", rep=f"{int(rep):,}", r1=pct(r1), r2=pct(r2), n1=n1, dir=direction
    )
    ft_b = f"part_a*(1 + {_frac_text(r2)})**{m} - {int(rep)}*((1 + {_frac_text(r2)})**{m} - 1)/{_frac_text(r2)}"
    ft_c = f"part_b*{_frac_text(r2)}/(1 - (1 + {_frac_text(r2)})**(-{k}))"
    return dict(
        text=stem,
        parts=[
            _part(
                f"Find the balance owing after {n1} months, to the nearest cent.",
                b1,
                2,
                [f"Apply B(n + 1) = B(n)(1 + {pct(r1)}%) − {int(rep)} for {n1} months.", f"B({n1}) = ${_fmt(b1)}."],
                _dists(b1, a_pairs),
            ),
            _part(
                f"Using your answer to (a), find the balance owing {m} months after the rate changes, "
                "to the nearest cent.",
                b2,
                3,
                [
                    f"Continue the recurrence from ${_fmt(b1)} with r = {pct(r2)}% for {m} months.",
                    f"Balance = ${_fmt(b2)}.",
                ],
                _dists(b2, b_pairs),
                ft=ft_b,
            ),
            _part(
                f"Using your answer to (b), find the monthly repayment that would clear this balance in exactly "
                f"{k} further months at the new rate, to the nearest cent.",
                r_new,
                3,
                [f"R = B r ÷ (1 − (1 + r)^−{k}) with B = {_fmt(b2)} and r = {pct(r2)}%.", f"R = ${_fmt(r_new)}."],
                _dists(r_new, c_pairs),
                ft=ft_c,
            ),
            _part(
                f"Using your answers to (b) and (c), find the total interest charged over those {k} months, "
                "to the nearest cent.",
                interest,
                2,
                [
                    f"Total repaid = {k} × {_fmt(r_new)}; interest = {k} × {_fmt(r_new)} − {_fmt(b2)} = ${_fmt(interest)}."
                ],
                _dists(interest, d_pairs),
                ft=f"{k}*part_c - part_b",
            ),
        ],
    )


def _alt_order(b: Fraction, r: Fraction, rep: Fraction, n: int) -> Fraction:
    """The repayment taken off before the interest is added: B(n+1) = (B(n) - R)(1 + r)."""
    for _ in range(n):
        b = (b - rep) * (1 + r)
    return b
