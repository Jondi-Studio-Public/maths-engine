"""Exam chain for compound interest: recover the rate from a balance, then project it forward (four parts).

An amount ``P`` earns a fixed annual rate, compounded annually. The stem states the balance ``B`` after ``t`` years and
does not name the rate.

(a) the annual growth factor, to 3 decimal places (the ``t``-th root of ``B / P``);
(b) the annual rate as a percentage to 1 decimal place (follow-through ``(part_a - 1) * 100``);
(c) the balance a further ``n`` years later, to the nearest cent (follow-through ``B * part_a ** n``);
(d) the total interest earned over all ``t + n`` years (follow-through ``part_c - P``).

The rate is a multiple of 0.5%, so the growth factor is exactly a three-decimal number; ``B`` is the exact balance
rounded to the cent, and a draw is kept only when the root of the rounded ``B / P`` still rounds to the exact factor.
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

_ACCOUNTS = ("savings account", "term deposit", "investment account", "fixed-term deposit")
_STEM = (
    "{name} invests ${p} in {art} that pays a fixed annual rate of interest, compounded annually. "
    "No money is added or taken out, and after {t} years the balance is ${b}.",
    "{name} puts ${p} into {art} at a fixed annual rate, with interest compounded annually and no deposits or "
    "withdrawals. The balance after {t} years is ${b}.",
)
_ARTICLE = {"savings account": "a", "term deposit": "a", "investment account": "an", "fixed-term deposit": "a"}


def _fmt(v: Fraction, dp: int) -> str:
    return stated_precision_string(round_half_up(v, dp), dp)


def _dists(key: Fraction, pairs, dp: int, n: int = 3):
    cands = [Distractor(_fmt(v, dp), tag, exact_value(round_half_up(v, dp))) for tag, v in pairs]
    out = pick_distractors(_fmt(key, dp), cands, n, tolerance=stated_precision_tolerance(dp))
    check(len(out) >= 2, "too few distractors")
    return out


def _part(prompt, key: Fraction, dp: int, marks: int, working, dists, ft=None, sensible=(0, None)):
    text = _fmt(key, dp)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(round_half_up(key, dp)),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=dists,
    )
    if sensible is not None:
        part["sensible_range"] = sensible
    if ft:
        part["follow_through"] = ft
    return part


def _safe(v: Fraction, dp: int) -> bool:
    """True when `v` is not within 1e-9 of a rounding boundary at `dp` places."""
    x = v * 10**dp
    frac = x - (x.numerator // x.denominator)
    return abs(frac - Fraction(1, 2)) > Fraction(1, 10**9)


def _root(x: Fraction, t: int) -> Fraction:
    """The t-th root of x, as a Fraction accurate far beyond 3 decimal places (bisection)."""
    lo, hi = Fraction(1), Fraction(2)
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid**t > x:
            hi = mid
        else:
            lo = mid
    return lo


@generator("exd_x_r2d_compound_growth_chain", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r2d_compound_growth_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Growth factor and rate from a balance, the balance n years on, and the total interest."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a compound-interest chain clear of rounding boundaries"):
        p = Fraction(rng.randint(4, 40) * 500)
        rate = Fraction(rng.randint(5, 15), 200)  # 2.5% .. 7.5% in steps of 0.5%
        t, n = rng.randint(3, 8), rng.randint(2, 6)
        f = 1 + rate
        b = round_half_up(p * f**t, 2)
        root = _root(b / p, t)
        if abs(root - f) > Fraction(1, 20000):
            continue
        c = b * f**n
        if not (_safe(c, 2) and _safe(c - p, 2)):
            continue
        # every distractor must be clear of its key at the stated precision
        wrong_c = [b * (1 + rate * n), b * f ** (t + n), p * f**n]
        if any(abs(round_half_up(w, 2) - round_half_up(c, 2)) < Fraction(2, 100) for w in wrong_c):
            continue
        break
    check(round_half_up(root, 3) == f, "the recovered growth factor is not the drawn one")
    # Independent route: build the balance period by period.
    bal = p
    for _ in range(t):
        bal += bal * rate
    check(round_half_up(bal, 2) == b, "period-by-period balance disagrees")
    later = b
    for _ in range(n):
        later += later * rate
    check(round_half_up(later, 2) == round_half_up(c, 2), "period-by-period later balance disagrees")

    ratio = b / p
    f_wrong_root = ratio  # forgot to take the root
    f_simple = 1 + (ratio - 1) / t  # simple-interest reading
    f_periods = _root(ratio, t + 1)  # one period too many
    r_pct = rate * 100
    a_pairs = [
        ("forgot-to-take-the-root", f_wrong_root),
        ("uses-simple-interest", f_simple),
        ("wrong-number-of-periods", f_periods),
    ]
    b_pairs = [
        ("factor-given-as-rate", f * 100),
        ("forgot-to-take-the-root", (f_wrong_root - 1) * 100),
        ("uses-simple-interest", (f_simple - 1) * 100),
        ("wrong-number-of-periods", (f_periods - 1) * 100),
    ]
    c_pairs = [
        ("uses-simple-interest", b * (1 + rate * n)),
        ("wrong-number-of-periods", b * f ** (t + n)),
        ("restarted-from-the-original-principal", p * f**n),
    ]
    total_interest = c - p
    d_pairs = [
        ("interest-from-wrong-base", c - b),
        ("uses-simple-interest", p * rate * (t + n)),
        ("interest-from-wrong-base", b - p),
    ]
    stem = rng.choice(_STEM).format(
        name=rng.choice(NAMES),
        p=f"{int(p):,}",
        art=f"{_ARTICLE[(acct := rng.choice(_ACCOUNTS))]} {acct}",
        t=t,
        b=f"{round_half_up(b, 2):,.2f}",
    )
    return dict(
        text=stem,
        parts=[
            _part(
                "Find the annual growth factor, correct to 3 decimal places.",
                f,
                3,
                2,
                [
                    f"B = P × (growth factor)^{t}, so the factor = ({_fmt(b, 2)} ÷ {int(p)})^(1/{t}).",
                    f"The factor is {_fmt(f, 3)}.",
                ],
                _dists(f, a_pairs, 3),
            ),
            _part(
                "Hence find the annual rate of interest, as a percentage correct to 1 decimal place.",
                r_pct,
                1,
                1,
                [f"Rate = ({_fmt(f, 3)} − 1) × 100 = {_fmt(r_pct, 1)}%."],
                _dists(r_pct, b_pairs, 1),
                ft="(part_a - 1)*100",
            ),
            _part(
                f"Using your growth factor, find the balance a further {n} years later, to the nearest cent.",
                c,
                2,
                2,
                [f"Balance = {_fmt(b, 2)} × {_fmt(f, 3)}^{n} = ${_fmt(c, 2)}."],
                _dists(c, c_pairs, 2),
                ft=f"{_fmt(b, 2)}*part_a**{n}",
            ),
            _part(
                f"Hence find the total interest earned over the whole {t + n} years, to the nearest cent.",
                total_interest,
                2,
                1,
                [f"Interest = {_fmt(c, 2)} − {int(p)} = ${_fmt(total_interest, 2)}."],
                _dists(total_interest, d_pairs, 2),
                ft=f"part_c - {int(p)}",
            ),
        ],
    )
