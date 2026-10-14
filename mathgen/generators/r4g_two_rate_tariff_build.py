"""Two-rate tariffs as piecewise linear models: `r4g_two_rate_tariff` (easy, medium, hard).

``piecewise-linear-tariff-model`` (``financial-discrete``, answer ``number``). A tariff is a fixed charge plus a
rate per unit up to a threshold and a different rate per unit beyond it:

    cost(u) = F + r1 * u            for u <= T
    cost(u) = F + r1 * T + r2 * (u - T)   for u > T

Rates are whole cents, so every cost is exact to the cent. Contexts avoid household energy and water bills (the
consumer-finance bill skills own those): taxi fares, equipment hire, phone data and courier parcels.

* easy: the cost of a stated usage beyond the threshold.
* medium: the usage that produces a stated cost (the cost tells which section it lies in).
* hard: the usage at which the tariff costs the same as a flat-rate plan (break-even, found in the second
  section), or the second rate recovered from the costs at two usages beyond the threshold.

Keys are checked by evaluating the piecewise function and searching whole units for the break-even.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4g_kit import number_text, single_number, retry

CONTEXTS = [
    dict(
        provider="Zippy Cabs",
        fixed="a flagfall of {F}",
        unit="kilometre",
        units="kilometres",
        first="{r1} per kilometre for the first {T} kilometres",
        second="{r2} per kilometre after that",
        use="a trip of {u} kilometres",
        rates=((120, 260), (80, 200)),
        threshold=(8, 20),
        fixed_range=(300, 600),
    ),
    dict(
        provider="Kestrel Hire",
        fixed="a delivery fee of {F}",
        unit="day",
        units="days",
        first="{r1} per day for the first {T} days",
        second="{r2} per day after that",
        use="a hire of {u} days",
        rates=((4500, 7500), (3000, 4400)),
        threshold=(3, 7),
        fixed_range=(2500, 6000),
    ),
    dict(
        provider="NetRoo Mobile",
        fixed="a monthly fee of {F}",
        unit="gigabyte",
        units="gigabytes",
        first="{r1} per gigabyte for the first {T} gigabytes",
        second="{r2} per gigabyte after that",
        use="{u} gigabytes of data in a month",
        rates=((100, 250), (300, 600)),
        threshold=(4, 12),
        fixed_range=(1500, 3500),
    ),
    dict(
        provider="Swiftpost",
        fixed="a handling fee of {F}",
        unit="kilogram",
        units="kilograms",
        first="{r1} per kilogram for the first {T} kilograms",
        second="{r2} per kilogram after that",
        use="a parcel of {u} kilograms",
        rates=((400, 800), (250, 390)),
        threshold=(5, 15),
        fixed_range=(500, 1200),
    ),
]


def cents(c) -> str:
    c = int(c)
    return f"${c // 100}.{c % 100:02d}"


class Tariff:
    """fixed fee `f`, then rates `r1` up to `t` units and `r2` beyond (all in cents)."""

    def __init__(self, f, r1, t, r2):
        self.f, self.r1, self.t, self.r2 = f, r1, t, r2

    def cost(self, u):
        u = Fraction(u)
        return self.f + (self.r1 * u if u <= self.t else self.r1 * self.t + self.r2 * (u - self.t))

    def text(self, ctx):
        fixed = ctx["fixed"].format(F=cents(self.f))
        second = ctx["second"].format(r2=cents(self.r2))
        if self.r1 == 0:
            return f"{ctx['provider']} charges {fixed}, which includes the first {self.t} {ctx['units']}, and {second}."
        return (
            f"{ctx['provider']} charges {fixed} plus {ctx['first'].format(r1=cents(self.r1), T=self.t)} and {second}."
        )


def draw_tariff(rng, *, ctx=None):
    ctx = ctx or rng.choice(CONTEXTS)
    lo1, hi1 = ctx["rates"][0]
    lo2, hi2 = ctx["rates"][1]
    step = 5 if hi1 < 1000 else 50
    for _ in attempts(500, "a tariff whose second rate is lower than its first"):
        r1 = step * rng.randint(lo1 // step, hi1 // step)
        r2 = step * rng.randint(lo2 // step, hi2 // step)
        if ctx["provider"] == "NetRoo Mobile":
            r1, r2 = 0, r2  # the first section is included in the fee
        elif r2 >= r1:
            continue
        t = rng.randint(*ctx["threshold"])
        f = 50 * rng.randint(ctx["fixed_range"][0] // 50, ctx["fixed_range"][1] // 50)
        return ctx, Tariff(f, r1, t, r2)
    raise AssertionError("unreachable")


def money(value) -> str:
    return f"${number_text(Fraction(value) / 100, 2)}"


def _plain(cost_cents):
    return number_text(Fraction(cost_cents) / 100, 2)


def _usage_text(ctx, u):
    return ctx["use"].format(u=u)


def _easy(rng):
    ctx, tf = draw_tariff(rng)
    u = tf.t + rng.randint(3, 15)
    cost = tf.cost(u)
    first = tf.f + tf.r1 * tf.t
    text = f"{tf.text(ctx)} Find the cost of {_usage_text(ctx, u)}, in dollars."
    slips = [
        (tf.f + tf.r1 * u, "r4g-first-rate-used-for-all-units"),
        (tf.f + tf.r2 * u, "r4g-second-rate-used-for-all-units"),
        (tf.r1 * tf.t + tf.r2 * (u - tf.t), "r4g-fixed-charge-left-out"),
        (tf.f + tf.r1 * tf.t + tf.r2 * u, "r4g-threshold-not-subtracted"),
        (tf.f + tf.r2 * (u - tf.t), "r4g-first-section-left-out"),
    ]
    slips = [(Fraction(v) / 100, tag) for v, tag in slips]
    working = [
        f"The first {tf.t} {ctx['units']} cost {cents(tf.f)} + {tf.t} x {cents(tf.r1)} = {cents(first)}.",
        f"The remaining {u - tf.t} {ctx['units']} cost {u - tf.t} x {cents(tf.r2)} = {cents(tf.r2 * (u - tf.t))}.",
        f"Total = {money(cost)}.",
    ]
    check(cost == first + tf.r2 * (u - tf.t), "cost re-derivation differs")
    return single_number(text, Fraction(cost) / 100, 2, working, slips)


def _medium(rng):
    ctx, tf = draw_tariff(rng)
    beyond = rng.random() < 0.6
    u = tf.t + rng.randint(2, 18) if beyond else rng.randint(2, tf.t - 1) if tf.t > 3 else 2
    cost = tf.cost(u)
    if tf.r1 == 0 and not beyond:
        beyond, u = True, tf.t + rng.randint(2, 18)
        cost = tf.cost(u)
    text = f"{tf.text(ctx)} A customer's total was {money(cost)}. How many {ctx['units']} were used?"
    at_t = tf.cost(tf.t)
    # independent check: search whole units
    found = [k for k in range(0, 80) if tf.cost(k) == cost]
    check(found == [u], "usage is not unique")
    slips = [
        (Fraction(cost - tf.f, tf.r1) if tf.r1 else 0, "r4g-first-rate-used-for-all-units"),
        (Fraction(cost - tf.f, tf.r2), "r4g-second-rate-used-for-all-units"),
        (Fraction(cost, tf.r2), "r4g-fixed-charge-left-out"),
        (Fraction(cost - at_t, tf.r2), "r4g-threshold-not-added-back") if beyond else (u + 1, "r4g-usage-off-by-one"),
        (Fraction(cost - tf.f - tf.r1 * tf.t, tf.r2), "r4g-threshold-not-added-back"),
        (u + 1, "r4g-usage-off-by-one"),
    ]
    slips = [(v, tag) for v, tag in slips if Fraction(v) > 0]
    working = [
        f"At the threshold the cost is {money(at_t)}.",
        (
            f"{money(cost)} is more than {money(at_t)}, so the usage is beyond {tf.t} {ctx['units']}: "
            f"{tf.t} + ({money(cost)} - {money(at_t)}) / {cents(tf.r2)} = {u}."
        )
        if beyond
        else f"{money(cost)} is less than {money(at_t)}, so the usage is within the first {tf.t} {ctx['units']}: ({money(cost)} - {cents(tf.f)}) / {cents(tf.r1)} = {u}.",
    ]
    return single_number(text, u, 0, working, slips)


def _break_even_draw(rng):
    for _ in attempts(3000, "a break-even inside the second section with exactly one crossing"):
        ctx, a = draw_tariff(rng)
        u_star = a.t + rng.randint(4, 20)
        r_b = a.r2 + rng.choice((30, 40, 50, 60, 75, 100)) if a.r2 < 500 else a.r2 + 100
        r_b = max(r_b, a.r1 + 10) if a.r1 else r_b
        f_b = a.cost(u_star) - r_b * u_star
        if f_b < 200 or f_b >= a.f or (f_b % 5):
            continue
        diff = lambda u: a.cost(u) - (f_b + r_b * u)  # noqa: E731
        zeros = [k for k in range(0, 4 * u_star) if diff(k) == 0]
        positive_before = all(diff(k) > 0 for k in range(0, u_star))
        if zeros != [u_star] or not positive_before:
            continue
        return ctx, a, f_b, r_b, u_star
    raise AssertionError("unreachable")


def _plan_b_text(ctx, f_b, r_b):
    return f"Plan B is a flat plan: a fixed charge of {cents(f_b)} plus {cents(r_b)} per {ctx['unit']} for all use."


def _break_even(rng):
    ctx, a, f_b, r_b, u_star = _break_even_draw(rng)
    text = (
        f"Plan A: {a.text(ctx)} {_plan_b_text(ctx, f_b, r_b)} "
        f"For how many {ctx['units']} of use do the two plans cost exactly the same?"
    )
    slips = [
        (Fraction(a.f - f_b, r_b - a.r1) if r_b != a.r1 else u_star + 3, "r4g-break-even-in-first-section"),
        (Fraction(a.f - f_b, r_b - a.r2), "r4g-threshold-not-subtracted"),
        (Fraction(a.cost(u_star), r_b), "r4g-fixed-charges-left-out"),
        (Fraction(a.cost(a.t) - f_b, r_b - a.r2) if a.t else u_star, "r4g-break-even-from-threshold-cost"),
        (u_star + a.t, "r4g-threshold-added-to-break-even"),
    ]
    slips = [(v, tag) for v, tag in slips if Fraction(v) > 0]
    working = [
        f"Beyond {a.t} {ctx['units']}, cost A = {cents(a.f)} + {a.t} x {cents(a.r1)} + (u - {a.t}) x {cents(a.r2)}.",
        f"Set cost A = cost B = {cents(f_b)} + {cents(r_b)} u and solve: u = {u_star}.",
        f"Check: both plans cost {money(a.cost(u_star))} for {u_star} {ctx['units']}.",
    ]
    check(a.cost(u_star) == f_b + r_b * u_star, "plans do not agree at the break-even")
    return single_number(text, u_star, 0, working, slips)


def _second_rate(rng):
    ctx, a = draw_tariff(rng)
    u1 = a.t + rng.randint(2, 6)
    u2 = u1 + rng.randint(3, 9)
    c1, c2 = a.cost(u1), a.cost(u2)
    rate = Fraction(c2 - c1, u2 - u1)
    check(rate == Fraction(a.r2), "second rate re-derivation differs")
    head = (
        f"A tariff has a fixed charge that includes the first {a.t} {ctx['units']}, then a rate per {ctx['unit']} for all use beyond that."
        if a.r1 == 0
        else f"A tariff has a fixed charge, a first rate per {ctx['unit']} up to {a.t} {ctx['units']} and a lower second rate for all use beyond {a.t} {ctx['units']}."
    )
    text = (
        f"{head} For {_usage_text(ctx, u1)} the cost is {money(c1)}, and for "
        f"{_usage_text(ctx, u2)} the cost is {money(c2)}. What is the second rate, in dollars per {ctx['unit']}?"
    )
    slips = [
        (Fraction(c2 - c1), "r4g-cost-difference-not-divided"),
        (Fraction(c2, u2), "r4g-average-cost-per-unit"),
        (Fraction(c2 - c1, u2), "r4g-difference-over-larger-usage"),
        (Fraction(c1, u1), "r4g-average-cost-per-unit"),
        (Fraction(c2 - c1, u2 - a.t), "r4g-difference-over-wrong-units"),
    ]
    slips = [(Fraction(v) / 100, tag) for v, tag in slips]
    working = [
        f"Both usages are beyond {a.t} {ctx['units']}, so the cost rises at the second rate.",
        f"Rate = ({money(c2)} - {money(c1)}) / ({u2} - {u1}) = {number_text(Fraction(c2 - c1, 100), 2)} / {u2 - u1} = ${number_text(rate / 100, 2)}.",
    ]
    return single_number(text, rate / 100, 2, working, slips)


def _hard(rng):
    return _break_even(rng) if rng.random() < 0.65 else _second_rate(rng)


_TIER = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "r4g_two_rate_tariff",
    level=11,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def r4g_two_rate_tariff(rng, tier="easy"):
    return retry(_TIER[tier], rng)
