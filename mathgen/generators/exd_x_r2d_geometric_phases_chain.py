"""Exam chain for geometric sequences in context: two phases with different rates (four parts).

A quantity changes by ``r1`` percent per period for ``n1`` periods, then by ``r2`` percent per period. The context is an
investment, a bacterial culture or a machine losing value.

(a) the value at the end of the first phase, to the nearest whole number;
(b) the value ``m`` periods into the second phase (follow-through ``part_a * g2 ** m``);
(c) the number of further whole periods for the value to pass a stated threshold (follow-through through a logarithm of
    ``part_a``);
(d) the total number of periods from the start (follow-through ``n1 + part_c``).

The second phase starts from the rounded first-phase value, as the prompts say, so every later key is computed from the
displayed ``part_a``. A draw is kept only when the threshold is crossed well inside a period (so the count does not
change if the first value is not rounded) and the answers to (b) and (c) are clear of their neighbours.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
)
from mathgen.slots import NAMES

# kind -> (growing?, stem, unit word for a period, how a value is named, money?)
_CONTEXTS = {
    "invest": dict(
        grow=True,
        money=True,
        period="year",
        stems=(
            "{name} invests ${p} in an account that earns {r1}% p.a. interest, compounded annually, for the first "
            "{n1} years. The rate then changes to {r2}% p.a., still compounded annually.",
            "{name} puts ${p} into an investment paying {r1}% p.a. compounded annually for {n1} years, after which "
            "the rate becomes {r2}% p.a., also compounded annually.",
        ),
        a="Find the value of the investment at the end of the first {n1} years, to the nearest dollar.",
        b="Using your answer to (a), find the value of the investment {m} years after the rate changes, to the "
        "nearest dollar.",
        c="Using your answer to (a), find the number of whole years after the rate changes until the investment "
        "first exceeds ${t}.",
        d="Hence find the total number of years from the start until the investment first exceeds ${t}.",
    ),
    "culture": dict(
        grow=True,
        money=False,
        period="hour",
        stems=(
            "A bacterial culture starts with {p} cells. For the first {n1} hours its population grows by {r1}% each "
            "hour. A change in conditions then makes it grow by {r2}% each hour.",
            "A laboratory culture contains {p} cells at the start. Over each of the first {n1} hours it grows by "
            "{r1}%, and after that it grows by {r2}% each hour.",
        ),
        a="Find the population at the end of the first {n1} hours, to the nearest whole number.",
        b="Using your answer to (a), find the population {m} hours after the change in conditions, to the nearest "
        "whole number.",
        c="Using your answer to (a), find the number of whole hours after the change in conditions until the "
        "population first exceeds {t}.",
        d="Hence find the total number of hours from the start until the population first exceeds {t}.",
    ),
    "machine": dict(
        grow=False,
        money=True,
        period="year",
        stems=(
            "A machine costs ${p} new. Its value falls by {r1}% each year for the first {n1} years, and then by "
            "{r2}% each year after that.",
            "{name}'s business buys a machine for ${p}. It loses {r1}% of its value each year for {n1} years, and "
            "then {r2}% of its value each year.",
        ),
        a="Find the value of the machine at the end of the first {n1} years, to the nearest dollar.",
        b="Using your answer to (a), find the value of the machine {m} years after the rate of loss changes, to the "
        "nearest dollar.",
        c="Using your answer to (a), find the number of whole years after the rate of loss changes until the value "
        "first falls below ${t}.",
        d="Hence find the total number of years from purchase until the value first falls below ${t}.",
    ),
}


def _dists(key: int, pairs, n: int = 3):
    cands = [Distractor(str(v), tag, exact_value(v)) for tag, v in pairs if v != key and v > 0]
    out = pick_distractors(str(key), cands, n)
    check(len(out) >= 2, "too few distractors")
    return out


def _part(prompt, key: int, marks, working, dists, ft=None):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(key),
        answer_text=str(int(key)),
        answer_latex=latex_answer(str(int(key))),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=dists,
    )
    if ft:
        part["follow_through"] = ft
    return part


def _first_crossing(v0: Fraction, g: Fraction, target: int, grow: bool) -> int:
    """The least k >= 1 with v0 g^k past the target, by direct iteration (the independent route)."""
    v = v0
    for k in range(1, 400):
        v = v * g
        if (v > target) if grow else (v < target):
            return k
    raise ValueError("threshold never crossed")


@generator("exd_x_r2d_geometric_phases_chain", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r2d_geometric_phases_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Value after phase one, value into phase two, periods to a threshold, and the total periods."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    kind = rng.choice(tuple(_CONTEXTS))
    ctx = _CONTEXTS[kind]
    grow = ctx["grow"]
    for _ in attempts(4000, "a two-phase chain clear of rounding"):
        if kind == "invest":
            p, r1, r2 = rng.randint(4, 80) * 500, rng.randint(3, 9), rng.randint(2, 10)
        elif kind == "culture":
            p, r1, r2 = rng.randint(5, 90) * 100, rng.randint(6, 18), rng.randint(5, 22)
        else:
            p, r1, r2 = rng.randint(20, 120) * 1000, rng.randint(10, 22), rng.randint(5, 16)
        if r1 == r2:
            continue
        n1, m, k_target = rng.randint(2, 8), rng.randint(2, 6), rng.randint(4, 14)
        sgn = 1 if grow else -1
        g1, g2 = 1 + sgn * Fraction(r1, 100), 1 + sgn * Fraction(r2, 100)
        v1 = round_half_up(p * g1**n1, 0)
        v2 = round_half_up(v1 * g2**m, 0)
        u = Fraction(rng.randint(25, 75), 100)  # cross well inside a period
        # threshold between the (k-1)th and kth values of the second phase
        lo, hi = sorted((v1 * g2 ** (k_target - 1), v1 * g2**k_target))
        t = int(round_half_up(lo + (hi - lo) * u, 0))
        if not (lo < t < hi) or abs(t - lo) < max(2, lo / 200) or abs(hi - t) < max(2, hi / 200):
            continue
        k = _first_crossing(Fraction(v1), g2, t, grow)
        if k != k_target:
            continue
        # the count must not change if the learner carries the unrounded first value forward
        if _first_crossing(p * g1**n1, g2, t, grow) != k:
            continue
        if v2 <= 1 or v1 <= 1 or t < 10:
            continue
        break
    check(k == k_target, "crossing count disagrees with the construction")
    # Independent route for (a) and (b): period by period.
    chk = Fraction(p)
    for _ in range(n1):
        chk *= g1
    check(round_half_up(chk, 0) == v1, "first-phase value disagrees")
    chk2 = Fraction(v1)
    for _ in range(m):
        chk2 *= g2
    check(round_half_up(chk2, 0) == v2, "second-phase value disagrees")

    def r_g1(n: int) -> int:
        return int(round_half_up(p * g1**n, 0))

    a_pairs = [
        ("wrong-number-of-periods", r_g1(n1 + 1)),
        ("wrong-number-of-periods", r_g1(n1 - 1)),
        ("uses-simple-interest", int(round_half_up(p * (1 + sgn * Fraction(r1 * n1, 100)), 0))),
    ]
    b_pairs = [
        ("restarted-from-the-original-principal", int(round_half_up(p * g2**m, 0))),
        ("rate-change-not-applied", int(round_half_up(v1 * g1**m, 0))),
        ("wrong-number-of-periods", int(round_half_up(v1 * g2 ** (n1 + m), 0))),
    ]
    k_from_p = None
    if (p < t) if grow else (p > t):
        k_from_p = _first_crossing(Fraction(p), g2, t, grow)
    k_g1 = _first_crossing(Fraction(v1), g1, t, grow)
    c_pairs = [
        ("off-by-one-count", k - 1),
        ("off-by-one-count", k + 1),
        ("rate-change-not-applied", k_g1),
        ("restarted-from-the-original-principal", k_from_p if k_from_p is not None else 0),
    ]
    d_pairs = [
        ("forgot-to-add-first-phase-years", k),
        ("off-by-one-count", n1 + k - 1),
        ("wrong-number-of-periods", n1 + m + k),
    ]
    total = n1 + k
    slots = dict(
        name=rng.choice(NAMES),
        p=f"{p:,}" if ctx["money"] else p,
        r1=r1,
        r2=r2,
        n1=n1,
        m=m,
        t=f"{t:,}" if ctx["money"] else t,
    )
    stem = rng.choice(ctx["stems"]).format(**slots)
    ft_c = f"floor(log({t}/part_a)/log({g2.numerator}/{g2.denominator})) + 1"
    unit = ctx["period"]
    return dict(
        text=stem,
        parts=[
            _part(
                ctx["a"].format(**slots),
                v1,
                2,
                [f"Value = {slots['p']} × {float(g1)}^{n1} = {v1}."],
                _dists(v1, a_pairs),
            ),
            _part(
                ctx["b"].format(**slots),
                v2,
                2,
                [f"Value = {v1} × {float(g2)}^{m} = {v2}."],
                _dists(v2, b_pairs),
                ft=f"part_a*({g2.numerator}/{g2.denominator})**{m}",
            ),
            _part(
                ctx["c"].format(**slots),
                k,
                3,
                [
                    f"Solve {v1} × {float(g2)}^k {'>' if grow else '<'} {t}, giving k > "
                    f"{math.log(t / v1) / math.log(float(g2)):.2f}.",
                    f"So k = {k} {unit}s.",
                ],
                _dists(k, c_pairs),
                ft=ft_c,
            ),
            _part(
                ctx["d"].format(**slots),
                total,
                1,
                [f"Total = {n1} + {k} = {total} {unit}s."],
                _dists(total, d_pairs),
                ft=f"{n1} + part_c",
            ),
        ],
    )
