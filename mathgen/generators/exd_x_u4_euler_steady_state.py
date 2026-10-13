"""Exam-tier source for `explicit-euler-linear-ode`: Euler steps for a model that must be read, then compared with
its steady state.

The practice tiers state `y' = (lambda)y + (c)` and ask for one or two Euler steps. The exam question describes a
quantity that is topped up at a steady rate and drains in proportion to what is present, `dC/dt = r - kC`, with a
starting amount and a step size, and asks for

1. the estimate after one step,
2. the estimate after a second step (the learner must re-evaluate the slope at the new estimate), and
3. how far that estimate still is below the steady state, the amount the model settles at, which the learner finds by
   setting the rate to zero.

Parts (b) and (c) carry follow-through. The key is built from the closed form of the Euler recurrence
`C(n+1) - r/k = (1 - hk)(C(n) - r/k)`, an identity independent of the step-by-step route the answer uses.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, generator, rational_text

from ._u4_kit import fraction_distractors, fraction_payload, survives_nsimplify

_SETTINGS = [
    "The concentration C (in mg per litre) of a nutrient in a rainwater tank is topped up at a steady rate and drains "
    "away in proportion to the amount present, so that dC/dt = {r} − {k}C, with t in days. Initially C = {c0}.",
    "The amount C (in kilograms) of fertiliser in a paddock's soil is topped up by irrigation and lost to runoff in "
    "proportion to the amount present, so that dC/dt = {r} − {k}C, with t in weeks. Initially C = {c0}.",
    "The chlorine level C (in mg per litre) in a backyard pool is topped up by a dosing pump and lost to sunlight in "
    "proportion to the level, so that dC/dt = {r} − {k}C, with t in hours. Initially C = {c0}.",
]
_UNITS = ["day", "week", "hour"]


@generator("exd_x_u4_euler_steady_state", level=16, presets={"exam": dict(tier="exam")})
def exd_x_u4_euler_steady_state(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: two explicit Euler steps for dC/dt = r - kC, then the distance below the steady state."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        k = rng.randint(2, 6)
        r = rng.randint(3, 30)
        n = rng.randint(k + 1, k + 4)  # h = 1/n keeps hk below 1, so the estimates climb towards the steady state
        if r % k == 0:
            continue
        steady = Fraction(r, k)
        c0 = rng.randint(0, int(steady))
        h = Fraction(1, n)
        c1 = c0 + h * (r - k * c0)
        c2 = c1 + h * (r - k * c1)
        gap = steady - c2
        if {r, k, n, c0} & {c1, c2, gap, steady} or 1 in (c1.denominator, c2.denominator, gap.denominator):
            continue
        if not survives_nsimplify(c1, c2, gap):
            continue
        break
    else:
        check(False, "no usable rate, drain, step and start in 1000 tries")

    # Independent route: the closed form of the Euler recurrence for dC/dt = r - kC.
    factor = 1 - h * k
    check(c1 == steady + factor * (c0 - steady), "closed form disagrees with the first step")
    check(c2 == steady + factor**2 * (c0 - steady), "closed form disagrees with the second step")
    check(gap == factor**2 * (steady - c0) and gap > 0, "gap to the steady state disagrees with the closed form")

    slope0 = r - k * c0
    slope1 = r - k * c1
    setting = rng.randrange(len(_SETTINGS))
    stem = (
        f"{_SETTINGS[setting].format(r=r, k=k, c0=c0)} "
        f"Euler's method is used with a step size of h = 1/{n} {_UNITS[setting]}."
    )
    a_text, b_text, c_text = (rational_text(v) for v in (c1, c2, gap))
    a_candidates = [
        (c0 + slope0, "euler-omits-step-size"),
        (c0 - h * slope0, "euler-sign-of-increment"),
        (c0 + h * (r + k * c0), "substitution-sign-error"),
        (c0 + h * r, None),
    ]
    b_candidates = [
        (c0 + 2 * h * slope0, "euler-reuses-first-slope"),
        (c1 + slope1, "euler-omits-step-size"),
        (c1 - h * slope1, "euler-sign-of-increment"),
        (c1, None),
    ]
    c_candidates = [
        (c2 - steady, "gap-sign-reversed"),
        (Fraction(k, r) - c2, "equilibrium-inverted"),
        (steady - c1, None),
        (steady, None),
    ]
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the estimate C₁ of C after one step (an exact fraction).",
                answer_kind="fraction",
                answer=fraction_payload(c1)[0],
                answer_text=a_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The rate at the start is {r} − {k}({c0}) = {slope0}.",
                    f"C₁ = {c0} + (1/{n})({slope0}) = {a_text}.",
                ],
                distractors=fraction_distractors(c1, a_candidates),
            ),
            dict(
                prompt="Using your C₁, find the estimate C₂ after a second step (an exact fraction).",
                answer_kind="fraction",
                answer=fraction_payload(c2)[0],
                answer_text=b_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a + ({r} - {k}*part_a)/{n}",
                working=[
                    f"The rate at C₁ is {r} − {k}({a_text}) = {rational_text(slope1)}.",
                    f"C₂ = {a_text} + (1/{n})({rational_text(slope1)}) = {b_text}.",
                ],
                distractors=fraction_distractors(c2, b_candidates),
            ),
            dict(
                prompt="The model settles at the value of C where dC/dt = 0. Using your C₂, find how far C₂ is "
                "below that steady-state value (an exact fraction).",
                answer_kind="fraction",
                answer=fraction_payload(gap)[0],
                answer_text=c_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{steady.numerator}/{steady.denominator} - part_b",
                working=[
                    f"dC/dt = 0 when {r} − {k}C = 0, so the steady state is C = {rational_text(steady)}.",
                    f"{rational_text(steady)} − {b_text} = {c_text}.",
                ],
                distractors=fraction_distractors(gap, c_candidates),
            ),
        ],
    )
