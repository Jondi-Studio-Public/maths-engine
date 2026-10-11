"""Exam source for `solve-equation-word-problems`: the learner must build the equation from two competing models.

The practice tiers describe one number that is transformed and ask for it. Here two quantities each depend on the
unknown and the situation says they match, so the equation has the unknown on both sides and the learner has to
decide what to set equal to what. Three situations:

  * ``plans``  -- two price plans (fixed fee plus a rate per unit); after how many units do they cost the same?
  * ``ages``   -- one person is m times as old as another now, and k times as old n years later; how old is the
                  younger person now?
  * ``chase``  -- one traveller leaves h hours before a faster one; how long after the second leaves is the first
                  caught?

Every question is built BACKWARDS from an integer answer, with a context-valid, positive, unique solution; the
equation is re-solved with SymPy before the question is returned.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Eq, Rational, Symbol, solve

from ..core import GRADER_TOLERANCE, check, generator, phrase, pick_distractors, rational_text

_N = Symbol("n")

_PLANS = [
    ("a gym", "Membership", "visit", "visits", "joining fee"),
    ("a printing shop", "Plan", "print job", "print jobs", "set-up fee"),
    ("a taxi company", "Company", "kilometre", "kilometres", "booking fee"),
]


def _txt(v: Fraction) -> str:
    return rational_text(Rational(v.numerator, v.denominator))


def _finish(text: str, answer: int, cands: list[Fraction], working: list[str]) -> dict | None:
    d = pick_distractors(str(answer), [_txt(c) for c in cands if c != answer and c > 0], 3, tolerance=GRADER_TOLERANCE)
    if len(d) < 3:
        return None
    return dict(
        text=text, latex="plain:" + text, answer=answer, answer_text=str(answer), distractors=d, working=working
    )


def _plans(rng: random.Random) -> dict | None:
    who, label, unit, units, fee_word = rng.choice(_PLANS)
    n0 = rng.randint(4, 24)
    b, d = rng.sample(range(1, 9), 2)
    if b < d:
        b, d = d, b
    a = rng.randint(2, 40)
    c = a + (b - d) * n0  # A: a + b*n, B: c + d*n  with b > d  =>  A cheaper to start, B dearer; equal at n0
    # A has the larger rate but the smaller fee
    fee_a, rate_a, fee_b, rate_b = a, b, c, d
    sol = solve(Eq(fee_a + rate_a * _N, fee_b + rate_b * _N), _N)
    check(sol == [n0] and fee_b > fee_a)
    text = phrase(
        rng,
        [
            "At {who}, Option A costs ${fa} {fee} plus ${ra} per {unit}. Option B costs ${fb} {fee} plus ${rb} per "
            "{unit}. For how many {units} do the two options cost the same?",
            "{who_cap} offers two options. Option A has a ${fa} {fee} and ${ra} per {unit}; Option B has a ${fb} {fee} "
            "and ${rb} per {unit}. After how many {units} are the total costs equal?",
        ],
        who=who,
        who_cap=who.capitalize(),
        fa=fee_a,
        ra=rate_a,
        fb=fee_b,
        rb=rate_b,
        fee=fee_word,
        unit=unit,
        units=units,
    )
    f = Fraction
    cands = [
        f(fee_b + fee_a, rate_a - rate_b),  # adds the fees
        f(fee_b - fee_a, rate_a + rate_b),  # adds the rates
        f(fee_a + rate_a * n0),  # reports the common cost, not the number of units
        f(fee_a, rate_a - rate_b),  # forgets option B's fee
    ]
    return _finish(text, n0, cands, [f"{fee_a} + {rate_a}n = {fee_b} + {rate_b}n", f"n = {n0}"])


def _ages(rng: random.Random) -> dict | None:
    k = rng.choice([2, 3])
    m = rng.choice([v for v in range(k + 1, 7)])
    y = rng.randint(4, 16)
    if (y * (m - k)) % (k - 1):
        return None
    n = y * (m - k) // (k - 1)
    if not 2 <= n <= 20 or m * y > 60:
        return None
    sol = solve(Eq(m * _N + n, k * (_N + n)), _N)
    check(sol == [y])
    text = phrase(
        rng,
        [
            "A parent is {m} times as old as their child. In {n} years the parent will be {k} times as old as the "
            "child. How old is the child now?",
            "Right now an adult is {m} times as old as a young relative. In {n} years the adult will be {k} times as "
            "old as the relative. How old is the relative now?",
        ],
        m=m,
        n=n,
        k=k,
    )
    f = Fraction
    cands = [
        f(m * y),  # the older person's age, not the younger one's
        f(y + n),  # the younger person's age in n years
        f(k * n, m - k),  # forgets the older person also gets n years older
        f(m * y + n),  # the older person's age in n years
    ]
    return _finish(text, y, cands, [f"{m}x + {n} = {k}(x + {n})", f"x = {y}"])


def _chase(rng: random.Random) -> dict | None:
    t = rng.randint(2, 8)
    w = rng.randint(5, 25)  # speed difference
    h = rng.choice([v for v in (1, 2, 3, 4) if (t * w) % v == 0])
    u = t * w // h
    if not 8 <= u <= 70:
        return None
    v = u + w
    vehicle = rng.choice([("cyclist", "km/h"), ("runner", "km/h"), ("boat", "km/h")])
    sol = solve(Eq(u * (_N + h), v * _N), _N)
    check(sol == [t])
    text = phrase(
        rng,
        [
            "A {veh} leaves a port travelling at {u} {unit}. {h} {hrs} later a second {veh} leaves the same port on "
            "the same route at {v} {unit}. How many hours after the second {veh} leaves does it catch the first?",
            "One {veh} sets out at {u} {unit}. {h} {hrs} afterwards another {veh}, travelling at {v} {unit}, sets "
            "out from the same place along the same route. How many hours after the faster {veh} sets out does it "
            "catch up?",
        ],
        veh=vehicle[0],
        unit=vehicle[1],
        u=u,
        v=v,
        h=h,
        hrs="hour" if h == 1 else "hours",
    )
    f = Fraction
    cands = [
        f(u * h, v + u),  # treats the closing speed as the sum of the speeds
        f(t + h),  # time since the first one left
        f(u * h),  # the head-start distance
        f(u * h, v),  # divides the head start by the faster speed only
    ]
    return _finish(text, t, cands, [f"{u}(t + {h}) = {v}t", f"t = {t}"])


_BUILDERS = {"plans": _plans, "ages": _ages, "chase": _chase}


@generator(
    "exd_x_a3_word_two_expressions",
    level=7,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_a3_word_two_expressions(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        out = _BUILDERS[rng.choice(sorted(_BUILDERS))](rng)
        if out is not None:
            return out
