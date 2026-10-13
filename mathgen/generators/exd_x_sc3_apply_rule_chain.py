"""Exam tier for `apply-a-rule-to-inputs`: substitute into a real formula twice, then compare.

The hard tier evaluates one quadratic at one whole-number input. Here a formula from a real setting (kinetic energy,
the games in a round-robin, the height of a thrown ball, the total of a savings plan) is applied at two inputs, and
the third part asks for the change between the two results, so a later part uses the earlier answers. Each formula has
a built-in order-of-operations trap: a square taken before the multiplication, a bracket that holds a difference, a
division that applies to the whole product.

Built backwards from whole-number inputs; every result is a whole number. Part (c) carries `follow_through` from
parts (a) and (b), so a slip in one evaluation does not lose the marks for the change.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator
from .exd_x_sc3_support import number_part, redraw, tagged

F = Fraction


def _kinetic(rng: random.Random) -> dict:
    m = rng.choice(range(40, 101, 2))
    v1 = rng.randint(3, 9)
    v2 = v1 + rng.randint(2, 6)

    def rule(v: int) -> Fraction:
        return F(1, 2) * m * v * v

    def slips(v: int) -> list[tuple[Fraction, str]]:
        return [
            (F(1, 2) * (m * v) ** 2, "squares-more-than-the-letter"),
            (F(m * v * v), "drops-the-division-by-two"),
            (F(1, 2) * m * 2 * v, "squared-as-doubled"),
        ]

    stem = (
        f"A cyclist and bike have a combined mass of {m} kg. The kinetic energy, in joules, of an object of mass "
        f"m kilograms moving at v metres per second is given by E = ½mv²."
    )
    return dict(
        stem=stem,
        inputs=(v1, v2),
        rule=rule,
        slips=slips,
        asks=[f"Find the kinetic energy when the speed is {v} m/s, in joules." for v in (v1, v2)],
        change_ask=f"Find the increase in kinetic energy when the speed rises from {v1} m/s to {v2} m/s, in joules.",
        change_slips=lambda a, b: [
            (rule(v2 - v1), "applies-the-formula-to-the-difference"),
            (a - b, "subtracts-in-the-wrong-order"),
            (F(1, 2) * m * (v2 - v1), "squared-as-doubled"),
        ],
        working=lambda v, r: f"E = ½ × {m} × {v}² = ½ × {m} × {v * v} = {r} J.",
    )


def _round_robin(rng: random.Random) -> dict:
    n1 = rng.randint(5, 12)
    n2 = n1 + rng.randint(2, 6)

    def rule(n: int) -> Fraction:
        return F(n * (n - 1), 2)

    def slips(n: int) -> list[tuple[Fraction, str]]:
        return [
            (F(n * (n - 1)), "drops-the-division-by-two"),
            (F(n * n, 2), "reads-n-minus-one-as-n"),
            (F(n * (n + 1), 2), "sign-error-in-a-bracket"),
        ]

    stem = (
        "In a round-robin netball competition every team plays every other team once. The number of games with n "
        "teams is given by G = n(n - 1) ÷ 2."
    )
    return dict(
        stem=stem,
        inputs=(n1, n2),
        rule=rule,
        slips=slips,
        asks=[f"Find the number of games when there are {n} teams." for n in (n1, n2)],
        change_ask=f"How many more games are played when the number of teams grows from {n1} to {n2}?",
        change_slips=lambda a, b: [
            (rule(n2 - n1), "applies-the-formula-to-the-difference"),
            (F(n2 - n1), "counts-only-the-new-teams"),
            (a - b, "subtracts-in-the-wrong-order"),
        ],
        working=lambda n, r: f"G = {n} × ({n} - 1) ÷ 2 = {n} × {n - 1} ÷ 2 = {r} games.",
    )


def _thrown_ball(rng: random.Random) -> dict:
    u = rng.choice([20, 25, 30, 35, 40])
    t1 = rng.randint(1, u // 5 - 2)
    t2 = t1 + rng.randint(1, min(3, u // 5 - 1 - t1))

    def rule(t: int) -> Fraction:
        return F(u * t - 5 * t * t)

    def slips(t: int) -> list[tuple[Fraction, str]]:
        return [
            (F(u * t - (5 * t) ** 2), "squares-the-coefficient"),
            (F(u * t - 10 * t), "squared-as-doubled"),
            (F((u - 5) * t * t), "operate-before-substituting"),
        ]

    stem = (
        f"A cricket ball is hit straight up from the ground at {u} metres per second. Its height in metres after "
        f"t seconds is given by h = {u}t - 5t², until it comes back down."
    )
    return dict(
        stem=stem,
        inputs=(t1, t2),
        rule=rule,
        slips=slips,
        asks=[f"Find the height of the ball after {t} seconds, in metres." for t in (t1, t2)],
        change_ask=f"Find the change in height of the ball between t = {t1} and t = {t2}, in metres "
        "(a negative answer means it has dropped).",
        change_slips=lambda a, b: [
            (rule(t2 - t1), "applies-the-formula-to-the-difference"),
            (a - b, "subtracts-in-the-wrong-order"),
            (F(abs(int(b - a))), "drops-the-sign-of-the-change"),
        ],
        working=lambda t, r: f"h = {u} × {t} - 5 × {t}² = {u * t} - {5 * t * t} = {r} m.",
    )


def _savings(rng: random.Random) -> dict:
    a = rng.choice([5, 10, 15, 20, 25])
    d = rng.randint(2, 6)
    n1 = rng.randint(4, 10)
    n2 = n1 + rng.randint(2, 8)

    def rule(n: int) -> Fraction:
        return F(n, 2) * (2 * a + (n - 1) * d)

    def slips(n: int) -> list[tuple[Fraction, str]]:
        return [
            (F(n) * (2 * a + (n - 1) * d), "drops-the-division-by-two"),
            (F(n, 2) * 2 * a + (n - 1) * d, "multiplies-only-the-first-term-in-a-bracket"),
            (F(n, 2) * (2 * a + n * d), "reads-n-minus-one-as-n"),
        ]

    stem = (
        f"A student saves ${a} in the first week and ${d} more each week than in the week before. The total saved, "
        "in dollars, after n weeks is given by S = (n ÷ 2)(2a + (n - 1)d), where a is the amount saved in the first "
        f"week and d is the weekly increase. Here a = {a} and d = {d}."
    )
    return dict(
        stem=stem,
        inputs=(n1, n2),
        rule=rule,
        slips=slips,
        asks=[f"Find the total saved after {n} weeks, in dollars." for n in (n1, n2)],
        change_ask=f"How much more is saved in total after {n2} weeks than after {n1} weeks, in dollars?",
        change_slips=lambda x, y: [
            (rule(n2 - n1), "applies-the-formula-to-the-difference"),
            (x - y, "subtracts-in-the-wrong-order"),
            (F((n2 - n1) * a), "ignores-the-weekly-increase"),
        ],
        working=lambda n, r: f"S = ({n} ÷ 2)(2 × {a} + ({n} - 1) × {d}) = {F(n, 2)} × {2 * a + (n - 1) * d} = {r}.",
    )


_BUILDERS = [_kinetic, _round_robin, _thrown_ball, _savings]


@generator("exd_x_sc3_apply_rule_chain", level=10, presets={"exam": dict(tier="exam")})
def exd_x_sc3_apply_rule_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Apply a real formula at two inputs, then find the change between the two results."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return redraw(_build, rng, "a formula draw with three usable wrong answers per part")


def _build(rng: random.Random) -> dict:
    spec = rng.choice(_BUILDERS)(rng)
    i1, i2 = spec["inputs"]
    r1, r2 = spec["rule"](i1), spec["rule"](i2)
    check(r1.denominator == 1 and r2.denominator == 1, "a result is not a whole number")
    change = r2 - r1
    sensible = (0, None) if r1 > 0 and r2 > 0 else None
    parts = [
        number_part(
            spec["asks"][0],
            r1,
            0,
            2,
            [spec["working"](i1, r1)],
            tagged(r1, spec["slips"](i1), 0, lo=0),
            sensible_range=sensible,
        ),
        number_part(
            spec["asks"][1],
            r2,
            0,
            2,
            [spec["working"](i2, r2)],
            tagged(r2, spec["slips"](i2), 0, lo=0),
            sensible_range=sensible,
        ),
        number_part(
            spec["change_ask"],
            change,
            0,
            1,
            [f"{r2} - {r1} = {change}."],
            tagged(change, spec["change_slips"](r1, r2), 0),
            follow_through="part_b - part_a",
        ),
    ]
    return dict(text=spec["stem"], parts=parts)
