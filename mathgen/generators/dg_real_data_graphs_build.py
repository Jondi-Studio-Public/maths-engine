"""Generates `dg_real_data_graphs` questions (easy, medium, hard) and the exam chain `dg_real_data_graphs_chain`.

``read-values-and-rates-from-real-data-graphs`` (``linear-relations``, answer ``number``): a graph of real-world data
(a cost graph, a conversion graph, a distance-time graph, a fuel graph) is drawn on gridlines with the axes scaled in
steps that are not 1, and the learner reads a value, a rate or a prediction from it. Every vertex of every graph sits on a
gridline crossing, so a reading is exact, and a registered reader recomputes the answer from the drawn segments and the
numbers in the question alone.

* easy: a straight-line graph through the origin (cost by mass, a currency conversion, fuel used by distance, distance by
  time at a steady speed). Read the value for a stated amount, or the amount for a stated value.
* medium: a rate from a through-origin graph, or a straight line that does not start at zero (a taxi fare with a
  flag-fall, a bike hire with a deposit, a fuel tank that empties): the starting value, a value read between gridlines'
  labels, the rate per unit, or the amount that gives a stated value.
* hard: use the graph beyond its edge (the line extended at its rate, with its starting value), or a distance-time journey
  with a stop: the speed of a stage, the time stopped, or the average speed over the whole journey including the stop.
* exam (``dg_real_data_graphs_chain``): a fare or hire graph: (a) the starting charge, (b) the rate per unit, (c) the cost
  of a trip beyond the graph, following through from (a) and (b), (d) the largest trip a budget allows, following through
  from (a) and (b).

Answers are whole numbers or half steps; distractors share the answer's decimal places.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .. import figures
from ..core import attempts, check, generator
from ._dg_kit import numd, numtext, p_number

AREA = "read-values-and-rates-from-real-data-graphs"

MISCONCEPTIONS = [
    (
        "dg-rd-gridlines-not-scaled",
        "Counts the gridlines up to the line and gives that count, forgetting that each gridline stands for more than 1.",
        AREA,
    ),
    ("dg-rd-off-one-gridline", "Reads the value one gridline too high or too low.", AREA),
    (
        "dg-rd-axes-swapped",
        "Gives the amount that was stated in the question, reading the wrong axis instead of the one asked for.",
        AREA,
    ),
    (
        "dg-rd-intercept-at-first-unit",
        "Reads the starting value at the first unit of the horizontal axis instead of at zero.",
        AREA,
    ),
    (
        "dg-rd-intercept-zero",
        "Says the starting value is 0, as if the line began at the origin, when it starts higher up the vertical axis.",
        AREA,
    ),
    (
        "dg-rd-rate-ignores-intercept",
        "Divides a reading by its horizontal amount to find the rate, which only works for a line through the origin and includes the starting value.",
        AREA,
    ),
    (
        "dg-rd-rate-as-rise",
        "Gives the whole rise between two readings as the rate instead of dividing by the run.",
        AREA,
    ),
    (
        "dg-rd-rate-inverted",
        "Divides the run by the rise, finding the units of one thing per unit of the other, upside down.",
        AREA,
    ),
    (
        "dg-rd-rate-as-value",
        "Gives a reading on the line (including the starting value) where the rate of change was asked.",
        AREA,
    ),
    (
        "dg-rd-rate-gridlines",
        "Finds the rise per gridline across, forgetting that a step on each axis stands for more than 1.",
        AREA,
    ),
    (
        "dg-rd-rate-sign",
        "Gives the rate of a falling line as a negative amount (or a rising one as a loss) when the amount used or added each unit was asked.",
        AREA,
    ),
    (
        "dg-rd-extrapolate-ignores-intercept",
        "Multiplies the rate by the new amount and leaves out the starting value.",
        AREA,
    ),
    (
        "dg-rd-extrapolate-adds-units",
        "Carries on the line by adding 1 for each extra unit instead of adding the rate.",
        AREA,
    ),
    (
        "dg-rd-extrapolate-ratio",
        "Scales the last reading up in proportion to the amount, which is wrong when the line does not start at zero.",
        AREA,
    ),
    (
        "dg-rd-extrapolate-intercept-twice",
        "Adds the starting value on twice when carrying the line past the edge of the graph.",
        AREA,
    ),
    (
        "dg-rd-far-ignores-intercept",
        "Divides the whole budget by the rate without taking away the starting charge first.",
        AREA,
    ),
    (
        "dg-rd-far-adds-intercept",
        "Adds the starting charge to the budget instead of taking it away before dividing by the rate.",
        AREA,
    ),
    ("dg-rd-stop-as-distance", "Gives the distance at the stop instead of the time it lasted.", AREA),
    ("dg-rd-stop-as-end-time", "Gives the time the stop ended instead of how long it lasted.", AREA),
    (
        "dg-rd-average-of-speeds",
        "Averages the speeds of the stages (with 0 for the stop) instead of dividing the total distance by the total time.",
        AREA,
    ),
    (
        "dg-rd-average-excludes-stop",
        "Divides the total distance by the time spent moving only, leaving out the stop.",
        AREA,
    ),
    ("dg-rd-average-as-distance", "Gives the total distance where the average speed was asked.", AREA),
    (
        "dg-rd-stage-reads-end",
        "Gives the distance at the end of the stage as its speed, forgetting to subtract the start and divide by the time.",
        AREA,
    ),
    (
        "dg-rd-stage-distance-not-speed",
        "Gives the distance travelled in the stage without dividing by the hours it took.",
        AREA,
    ),
]


# ---------------------------------------------------------------------------------------------- contexts

# Through the origin: y = p * gy * (x / gx), the line passes through a gridline crossing at every x gridline.
THROUGH = (
    dict(
        key="apples",
        alt="A straight-line graph of the cost of apples against their mass.",
        x_label="Mass (kg)",
        y_label="Cost ($)",
        gx=1,
        gy=2,
        ps=(1, 2, 3),
        jx=(5, 8),
        xs=1,
        read_y="What is the cost of {x} kg of apples, in dollars?",
        read_x="What mass of apples, in kilograms, can be bought for ${y}?",
        rate="What is the price of apples per kilogram, in dollars?",
        extra="The graph stops at {edge} kg. At the same price, what would {X} kg of apples cost, in dollars?",
    ),
    dict(
        key="euro",
        alt="A straight-line graph that converts euros to Australian dollars.",
        x_label="Euros (EUR)",
        y_label="Australian dollars (AUD)",
        gx=2,
        gy=3,
        ps=(1,),
        jx=(5, 8),
        xs=1,
        read_y="How many Australian dollars is {x} euros worth?",
        read_x="How many euros are worth {y} Australian dollars?",
        rate="How many Australian dollars is 1 euro worth?",
        extra="The graph stops at {edge} euros. At the same exchange rate, how many Australian dollars is {X} euros worth?",
    ),
    dict(
        key="fuel",
        alt="A straight-line graph of the fuel a car uses against the distance driven.",
        x_label="Distance (km)",
        y_label="Fuel used (L)",
        gx=50,
        gy=4,
        ps=(1, 2),
        jx=(5, 8),
        xs=100,
        read_y="How many litres of fuel does the car use to drive {x} km?",
        read_x="How far, in kilometres, can the car drive on {y} litres of fuel?",
        rate="How many litres does the car use for every 100 km?",
        extra="The graph stops at {edge} km. At the same rate, how many litres would the car use to drive {X} km?",
    ),
    dict(
        key="cyclist",
        alt="A straight-line graph of the distance a cyclist rides against the time at a steady speed.",
        x_label="Time (hours)",
        y_label="Distance (km)",
        gx=1,
        gy=10,
        ps=(1, 2, 3),
        jx=(5, 8),
        xs=1,
        read_y="How far does the cyclist ride in {x} hours, in kilometres?",
        read_x="For how many hours has the cyclist ridden when {y} km have been covered?",
        rate="What is the cyclist's speed, in kilometres per hour?",
        extra="The graph stops at {edge} hours. Riding at the same speed, how far would the cyclist go in {X} hours?",
    ),
)

# Starting value c (a multiple of gy) then a straight line of slope s * p * gy per gx; s = +1 rises, -1 falls.
INTERCEPT = (
    dict(
        key="taxi",
        alt="A straight-line graph of a taxi fare against the distance travelled.",
        x_label="Distance (km)",
        y_label="Fare ($)",
        gx=1,
        gy=2,
        ps=(1,),
        cs=(2, 3, 4),
        jx=(5, 8),
        sign=1,
        xs=1,
        fixed="What is the fixed charge (flag-fall) for a taxi trip, in dollars?",
        read_y="What is the fare for a {x} km trip, in dollars?",
        read_x="How far can you travel for a fare of ${y}, in kilometres?",
        rate="By how many dollars does the fare go up for each extra kilometre?",
        extra="The graph stops at {edge} km. At the same rates, what would the fare be for a {X} km trip, in dollars?",
        far="Mia has ${B} for a taxi. At the same rates, what is the longest trip she can take, in kilometres?",
    ),
    dict(
        key="bike",
        alt="A straight-line graph of the cost of hiring a bike against the number of hours.",
        x_label="Time (hours)",
        y_label="Cost ($)",
        gx=1,
        gy=5,
        ps=(1, 2),
        cs=(1, 2),
        jx=(4, 7),
        sign=1,
        xs=1,
        fixed="What is the deposit charged before the first hour of bike hire, in dollars?",
        read_y="What is the cost of hiring the bike for {x} hours, in dollars?",
        read_x="For how many hours can the bike be hired for ${y}?",
        rate="By how many dollars does the cost go up for each extra hour?",
        extra="The graph stops at {edge} hours. At the same rates, what would it cost to hire the bike for {X} hours, in dollars?",
        far="Sam has ${B} to spend on bike hire. At the same rates, for how many hours can he hire the bike?",
    ),
    dict(
        key="plumber",
        alt="A straight-line graph of a plumber's charge against the hours worked.",
        x_label="Time (hours)",
        y_label="Charge ($)",
        gx=1,
        gy=10,
        ps=(1, 2),
        cs=(1, 2),
        jx=(4, 6),
        sign=1,
        xs=1,
        fixed="What is the call-out fee the plumber charges before starting work, in dollars?",
        read_y="What does the plumber charge for a job of {x} hours, in dollars?",
        read_x="For how many hours of work does the plumber charge ${y}?",
        rate="By how many dollars does the charge go up for each extra hour of work?",
        extra="The graph stops at {edge} hours. At the same rates, what would the plumber charge for {X} hours of work, in dollars?",
        far="Ana has ${B} to pay a plumber. At the same rates, for how many hours of work can she book the plumber?",
    ),
    dict(
        key="tank",
        alt="A straight-line graph of the fuel left in a car's tank against the distance driven.",
        x_label="Distance (km)",
        y_label="Fuel in tank (L)",
        gx=100,
        gy=4,
        ps=(2, 3),
        cs=(10, 11, 12),
        jx=(3, 5),
        sign=-1,
        xs=100,
        fixed="How many litres are in the tank at the start of the trip?",
        read_y="How many litres are left in the tank after {x} km?",
        read_x="How far has the car driven when {y} litres are left in the tank?",
        rate="How many litres does the car use for every 100 km?",
    ),
)

JOURNEY_KEYS = ("speed", "stopped", "average")
_JOURNEY_Q = dict(
    speed="What was the cyclist's speed, in kilometres per hour, between {t1} and {t2} hours?",
    stopped="For how many hours did the cyclist stop?",
    average="What was the cyclist's average speed over the whole {T} hours, in kilometres per hour?",
)
_JOURNEY_ALT = "A distance-time graph of a bike ride made of straight segments, with one flat part."


def _all_contexts():
    return {c["key"]: c for c in THROUGH + INTERCEPT}


# ---------------------------------------------------------------------------------------------- line graphs


def _line(rng, ctx, intercept):
    """The numbers of a straight-line graph: c0 (starting value), slope per gridline step, columns, edge value."""
    p = rng.choice(ctx["ps"])
    j = rng.randint(*ctx["jx"])
    c = rng.choice(ctx["cs"]) * ctx["gy"] if intercept else 0
    s = ctx.get("sign", 1)
    step = p * ctx["gy"]  # y changes by this at each x gridline
    ys = [c + s * step * i for i in range(j + 1)]
    if min(ys) < ctx["gy"] and s < 0:
        return None
    return dict(p=p, j=j, c=c, s=s, step=step, ys=ys)


def _fig(ctx, g):
    gx, gy = ctx["gx"], ctx["gy"]
    j = g["j"]
    top = max(g["ys"])
    ymax = ((top + gy) // gy) * gy
    ymax = max(ymax, gy * 4)
    pts = [[gx * i, y] for i, y in enumerate(g["ys"])]
    return figures.cartesian(
        x_range=(0, gx * j),
        y_range=(0, ymax),
        alt=ctx["alt"],
        x_grid=gx,
        y_grid=gy,
        equal_scale=False,
        segments=[{"from": pts[0], "to": pts[-1]}],
        points=[{"at": pts[0]}, {"at": pts[-1]}],
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
    )


def _rate(ctx, g):
    """The rate in the units the question asks (per `xs` of the x unit), as a positive Fraction."""
    return Fraction(g["step"] * ctx["xs"], ctx["gx"])


def _text(ctx, key, **kw):
    # "a 8 km trip" reads "an 8 km trip": fix the article after the numbers are filled in.
    return re.sub(r"\ba (?=(?:8\d*|11|18)\b)", "an ", ctx[key].format(**kw))


def _num_dist(answer, cands, lo=0):
    return numd(answer, [c for c in cands if c[0] is not None], lo=lo)


def _wrap(text, fig, ans, working, ds):
    if len(ds) < 3:
        return None
    ans = Fraction(ans)
    return dict(
        text=text,
        figure=fig,
        answer_kind="number",
        answer=int(ans) if ans.denominator == 1 else float(ans),
        answer_text=numtext(ans),
        answer_latex=numtext(ans),
        working=working,
        distractors=ds,
    )


def _read_y_q(rng, ctx, g, fig):
    i = rng.randint(2, g["j"])
    x = ctx["gx"] * i
    y = g["ys"][i]
    text = _text(ctx, "read_y", x=x)
    cands = [
        (Fraction(x), "dg-rd-axes-swapped"),
        (Fraction(abs(y - g["c"]) // ctx["gy"] if g["c"] == 0 else y // ctx["gy"]), "dg-rd-gridlines-not-scaled"),
        (Fraction(y + ctx["gy"]), "dg-rd-off-one-gridline"),
        (Fraction(y - ctx["gy"]), "dg-rd-off-one-gridline"),
    ]
    work = [f"Go along to {x} on the horizontal axis, then up to the line.", f"Read across to the vertical axis: {y}."]
    return _wrap(text, fig, y, work, _num_dist(Fraction(y), cands))


def _read_x_q(rng, ctx, g, fig):
    i = rng.randint(2, g["j"])
    x = ctx["gx"] * i
    y = g["ys"][i]
    text = _text(ctx, "read_x", y=y)
    cands = [
        (Fraction(y), "dg-rd-axes-swapped"),
        (Fraction(i), "dg-rd-gridlines-not-scaled"),
        (Fraction(x + ctx["gx"]), "dg-rd-off-one-gridline"),
        (Fraction(x - ctx["gx"]), "dg-rd-off-one-gridline"),
    ]
    work = [
        f"Find {y} on the vertical axis, go across to the line, then down to the horizontal axis.",
        f"The reading is {x}.",
    ]
    return _wrap(text, fig, x, work, _num_dist(Fraction(x), cands))


def _rate_q(rng, ctx, g, fig, intercept):
    r = _rate(ctx, g)
    x1 = ctx["gx"] * g["j"]
    y_end = g["ys"][-1]
    text = _text(ctx, "rate")
    cands = [
        (Fraction(abs(y_end - g["c"])), "dg-rd-rate-as-rise"),
        (
            Fraction(y_end, x1) * ctx["xs"] if intercept and Fraction(y_end, x1) * ctx["xs"] != r else None,
            "dg-rd-rate-ignores-intercept",
        ),
        (Fraction(ctx["gy"] * g["p"]) if Fraction(ctx["gy"] * g["p"]) != r else None, "dg-rd-rate-gridlines"),
        (1 / r if (1 / r).denominator in (1, 2) else None, "dg-rd-rate-inverted"),
        (Fraction(g["ys"][1]), "dg-rd-rate-as-value"),
        (-r if g["s"] < 0 else None, "dg-rd-rate-sign"),
    ]
    first = f"From the first gridline to the last, the line goes from {g['ys'][0]} to {y_end} over {x1}."
    work = [
        first,
        f"{abs(y_end - g['ys'][0])} / {x1} = {numtext(r / ctx['xs'])} per unit"
        if ctx["xs"] == 1
        else f"The line changes by {g['step']} for each {ctx['gx']}, so by {numtext(r)} for every {ctx['xs']}.",
    ]
    return _wrap(text, fig, r, work, _num_dist(r, cands, lo=None if g["s"] < 0 else 0))


def _fixed_q(rng, ctx, g, fig):
    c = Fraction(g["c"])
    text = _text(ctx, "fixed")
    cands = [
        (Fraction(g["ys"][1]), "dg-rd-intercept-at-first-unit"),
        (Fraction(0), "dg-rd-intercept-zero"),
        (Fraction(g["c"] // ctx["gy"]), "dg-rd-gridlines-not-scaled"),
        (c + ctx["gy"], "dg-rd-off-one-gridline"),
        (c - ctx["gy"], "dg-rd-off-one-gridline"),
    ]
    work = [
        "The fixed charge is the value when the amount is 0, where the line meets the vertical axis.",
        f"It is {g['c']}.",
    ]
    return _wrap(text, fig, c, work, _num_dist(c, cands))


def _easy(rng):
    ctx = rng.choice(THROUGH)
    g = _line(rng, ctx, False)
    if g is None:
        return None
    fig = _fig(ctx, g)
    return (_read_y_q if rng.random() < 0.6 else _read_x_q)(rng, ctx, g, fig)


def _medium(rng):
    if rng.random() < 0.3:
        ctx = rng.choice(THROUGH)
        g = _line(rng, ctx, False)
        if g is None:
            return None
        return _rate_q(rng, ctx, g, _fig(ctx, g), False)
    ctx = rng.choice(INTERCEPT)
    g = _line(rng, ctx, True)
    if g is None:
        return None
    fig = _fig(ctx, g)
    shape = rng.choice(("fixed", "rate", "read_y", "read_x"))
    if shape == "fixed":
        return _fixed_q(rng, ctx, g, fig)
    if shape == "rate":
        return _rate_q(rng, ctx, g, fig, True)
    return (_read_y_q if shape == "read_y" else _read_x_q)(rng, ctx, g, fig)


# ---------------------------------------------------------------------------------------------- hard


def _extra_q(rng, ctx, g, fig, intercept):
    r = _rate(ctx, g) / ctx["xs"]  # per unit of x
    edge = ctx["gx"] * g["j"]
    X = edge + ctx["gx"] * rng.randint(1, 3)
    ans = Fraction(g["c"]) + r * X
    if ans.denominator not in (1, 2):
        return None
    text = _text(ctx, "extra", edge=edge, X=X)
    y_end = g["ys"][-1]
    cands = [
        (r * X if intercept else None, "dg-rd-extrapolate-ignores-intercept"),
        (Fraction(y_end, edge) * X if intercept else None, "dg-rd-extrapolate-ratio"),
        (Fraction(y_end) + (X - edge), "dg-rd-extrapolate-adds-units"),
        (g["c"] * 2 + r * X if intercept else None, "dg-rd-extrapolate-intercept-twice"),
        (ans + ctx["gy"], "dg-rd-off-one-gridline"),
    ]
    rate_line = f"The line rises by {g['step']} for every {ctx['gx']} across, which is {numtext(r * ctx['gx'])} for every {ctx['gx']} unit{'' if ctx['gx'] == 1 else 's'}."
    work = [
        rate_line,
        (
            f"The line starts at {g['c']}, so the value at {X} is {g['c']} + {numtext(r * X)} = {numtext(ans)}."
            if intercept
            else f"The value at {X} is {numtext(r * X)}."
        ),
    ]
    return _wrap(text, fig, ans, work, _num_dist(ans, cands))


def _journey(rng):
    for _ in attempts(300, "a journey graph"):
        gy = rng.choice((5, 10))
        a, b = rng.sample((1, 2, 3), 2)
        d1, w, d2 = rng.randint(2, 3), rng.randint(2, 5), rng.randint(1, 3)
        s1, s2 = a * gy, b * gy
        pts = [(0, 0), (d1, s1 * d1), (d1 + w, s1 * d1), (d1 + w + d2, s1 * d1 + s2 * d2)]
        T, D = pts[-1]
        if T > 10 or D > 12 * gy * 2 or (Fraction(D, T)).denominator not in (1, 2):
            continue
        return dict(gy=gy, s1=s1, s2=s2, d1=d1, w=w, d2=d2, pts=pts, T=T, D=D)
    raise AssertionError  # pragma: no cover


def _journey_fig(j):
    gy = j["gy"]
    ymax = ((j["D"] + gy) // gy) * gy
    ymax = max(ymax, gy * 4)
    segs = [{"from": list(a), "to": list(b)} for a, b in zip(j["pts"], j["pts"][1:])]
    return figures.cartesian(
        x_range=(0, j["T"]),
        y_range=(0, ymax),
        alt=_JOURNEY_ALT,
        x_grid=1,
        y_grid=gy,
        equal_scale=False,
        segments=segs,
        points=[{"at": list(p)} for p in j["pts"]],
        x_label="Time (hours)",
        y_label="Distance (km)",
    )


def _journey_q(rng, shape=None):
    j = _journey(rng)
    shape = shape or rng.choice(JOURNEY_KEYS)
    fig = _journey_fig(j)
    (t0, y0), (t1, y1), (t2, y2), (t3, y3) = j["pts"]
    if shape == "speed":
        stage = rng.choice((1, 2))
        ta, tb = (t0, t1) if stage == 1 else (t2, t3)
        ya, yb = (y0, y1) if stage == 1 else (y2, y3)
        ans = Fraction(yb - ya, tb - ta)
        text = _JOURNEY_Q["speed"].format(t1=ta, t2=tb)
        cands = [
            (Fraction(yb), "dg-rd-stage-reads-end"),
            (Fraction(yb - ya), "dg-rd-stage-distance-not-speed"),
            (Fraction(yb, tb) if ya else None, "dg-rd-rate-ignores-intercept"),
            (ans + j["gy"], "dg-rd-off-one-gridline"),
            (ans - j["gy"], "dg-rd-off-one-gridline"),
        ]
        work = [
            f"Between {ta} and {tb} hours the distance goes from {ya} km to {yb} km.",
            f"{yb} - {ya} = {yb - ya} km in {tb - ta} hour{'' if tb - ta == 1 else 's'}, so the speed is {numtext(ans)} km/h.",
        ]
    elif shape == "stopped":
        ans = Fraction(t2 - t1)
        text = _JOURNEY_Q["stopped"]
        # Three wrong values, how many of them fall below the key drawn at random, so the key has no fixed rank.
        wrong = {}
        for v, tag in (
            (ans - 1, "dg-rd-off-one-gridline"),
            (ans - 2, "dg-rd-off-one-gridline"),
            (ans + 1, "dg-rd-off-one-gridline"),
            (ans + 2, "dg-rd-off-one-gridline"),
            (Fraction(y1), "dg-rd-stop-as-distance"),
            (Fraction(t2), "dg-rd-stop-as-end-time"),
            (Fraction(t1), "dg-rd-stop-as-end-time"),
        ):
            if v > 0 and v != ans:
                wrong.setdefault(v, tag)
        below = [(v, t) for v, t in wrong.items() if v < ans]
        above = [(v, t) for v, t in wrong.items() if v > ans]
        k = rng.randint(0, min(3, len(below)))
        cands = rng.sample(below, k) + rng.sample(above, 3 - k)
        rng.shuffle(cands)
        work = [
            f"The flat part of the graph runs from {t1} to {t2} hours.",
            f"It lasted {t2} - {t1} = {t2 - t1} hour{'' if t2 - t1 == 1 else 's'}.",
        ]
    else:
        ans = Fraction(y3, t3)
        text = _JOURNEY_Q["average"].format(T=t3)
        v1, v2 = Fraction(y1 - y0, t1 - t0), Fraction(y3 - y2, t3 - t2)
        cands = [
            (Fraction(y3), "dg-rd-average-as-distance"),
            ((v1 + v2) / 3, "dg-rd-average-of-speeds"),
            (Fraction(y3, t3 - (t2 - t1)), "dg-rd-average-excludes-stop"),
            ((v1 + v2) / 2, "dg-rd-average-of-speeds"),
            (ans + 1, "dg-rd-off-one-gridline"),
        ]
        work = [
            f"The total distance is {y3} km and the total time, including the stop, is {t3} hour{'' if t3 == 1 else 's'}.",
            f"{y3} / {t3} = {numtext(ans)} km/h.",
        ]
    return _wrap(text, fig, ans, work, _num_dist(ans, cands))


def _hard(rng):
    if rng.random() < 0.5:
        return _journey_q(rng)
    ctx = rng.choice([c for c in THROUGH + INTERCEPT if "extra" in c])
    intercept = ctx in INTERCEPT
    g = _line(rng, ctx, intercept)
    if g is None:
        return None
    return _extra_q(rng, ctx, g, _fig(ctx, g), intercept)


@generator("dg_real_data_graphs", level=12, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
def dg_real_data_graphs(rng, tier="easy"):
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(600, f"a {tier} real-data graph question"):
        d = build(rng)
        if d is not None:
            return d
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- exam chain


@generator("dg_real_data_graphs_chain", level=12, presets={"exam": {"tier": "exam"}})
def dg_real_data_graphs_chain(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(600, "a real-data graph chain"):
        ctx = rng.choice([c for c in INTERCEPT if c["sign"] > 0])
        g = _line(rng, ctx, True)
        if g is None:
            continue
        fig = _fig(ctx, g)
        c = Fraction(g["c"])
        r = Fraction(g["step"], ctx["gx"])
        edge = ctx["gx"] * g["j"]
        X = edge + ctx["gx"] * rng.randint(1, 3)
        if ctx["key"] == "taxi" and re.match(r"(?:8|11$|18$)", str(X)):
            continue  # "an 8 km trip": the chain keeps to "a N km trip", the form its follow-through test reads
        cost = c + r * X
        far_x = edge + ctx["gx"] * rng.randint(1, 3)
        budget = c + r * far_x
        if cost.denominator != 1 or budget.denominator != 1 or r.denominator != 1:
            continue
        noun = "km" if ctx["key"] == "taxi" else "hours"
        cur = lambda v: f"${numtext(v)}"  # noqa: E731
        y_end = g["ys"][-1]
        da = _num_dist(
            c,
            [
                (Fraction(g["ys"][1]), "dg-rd-intercept-at-first-unit"),
                (Fraction(0), "dg-rd-intercept-zero"),
                (c + ctx["gy"], "dg-rd-off-one-gridline"),
                (c - ctx["gy"], "dg-rd-off-one-gridline"),
            ],
        )
        db = _num_dist(
            r,
            [
                (Fraction(y_end, edge), "dg-rd-rate-ignores-intercept"),
                (Fraction(y_end - g["c"]), "dg-rd-rate-as-rise"),
                (Fraction(g["ys"][1]), "dg-rd-rate-as-value"),
                (Fraction(ctx["gy"]), "dg-rd-rate-gridlines"),
                (r + 1, "dg-rd-off-one-gridline"),
            ],
        )
        dc = _num_dist(
            cost,
            [
                (r * X, "dg-rd-extrapolate-ignores-intercept"),
                (Fraction(y_end, edge) * X, "dg-rd-extrapolate-ratio"),
                (c * 2 + r * X, "dg-rd-extrapolate-intercept-twice"),
                (Fraction(y_end) + (X - edge), "dg-rd-extrapolate-adds-units"),
            ],
        )
        dd = _num_dist(
            Fraction(far_x),
            [
                (budget / r, "dg-rd-far-ignores-intercept"),
                ((budget + c) / r, "dg-rd-far-adds-intercept"),
                (Fraction(far_x + 1), "dg-rd-off-one-gridline"),
                (Fraction(far_x - 1), "dg-rd-off-one-gridline"),
            ],
        )
        if min(len(x) for x in (da, db, dc, dd)) < 2:
            continue
        stem = (
            "The graph shows the fare for a taxi trip against the distance travelled."
            if ctx["key"] == "taxi"
            else "The graph shows the cost of hiring a bike against the number of hours."
            if ctx["key"] == "bike"
            else "The graph shows a plumber's charge against the number of hours worked."
        )
        return dict(
            text=stem,
            figure=fig,
            parts=[
                p_number(
                    ctx["fixed"], c, 1, [f"The line meets the vertical axis at {g['c']}."], da, sensible_range=(-1, 100)
                ),
                p_number(
                    ctx["rate"],
                    r,
                    2,
                    [
                        f"From 0 to {edge} the value goes from {g['c']} to {y_end}: ({y_end} - {g['c']}) / {edge} = {numtext(r)}."
                    ],
                    db,
                    sensible_range=(0, 100),
                ),
                p_number(
                    _text(ctx, "extra", edge=edge, X=X),
                    cost,
                    2,
                    [f"{cur(c)} + {cur(r)} x {X} = {cur(cost)}."],
                    dc,
                    follow_through=f"part_a + part_b * {X}",
                    sensible_range=(0, 1000),
                ),
                p_number(
                    _text(ctx, "far", B=int(budget)),
                    far_x,
                    2,
                    [
                        f"Take away the starting charge: {int(budget)} - {numtext(c)} = {numtext(budget - c)}.",
                        f"{numtext(budget - c)} / {numtext(r)} = {far_x} {noun}.",
                    ],
                    dd,
                    follow_through=f"({int(budget)} - part_a) / part_b",
                    sensible_range=(0, 200),
                ),
            ],
        )
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- the reader


def _pattern(template):
    esc = re.escape(template).replace(r"\ a\ \{", r"\ an?\ \{")
    for name in ("x", "y", "t1", "t2", "T", "edge", "X", "B"):
        esc = esc.replace(re.escape("{" + name + "}"), rf"(?P<{name}>\d+(?:\.\d+)?)")
    return re.compile(esc)


def _value_at(spec, x):
    pts = []
    for s in spec["segments"]:
        pts += [tuple(s["from"]), tuple(s["to"])]
    pts = sorted(set(pts))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1 and x1 > x0:
            return Fraction(y0) + Fraction(y1 - y0, 1) * Fraction(x - x0) / Fraction(x1 - x0)
    raise AssertionError("x off the graph")


def _x_at(spec, y):
    pts = []
    for s in spec["segments"]:
        pts += [tuple(s["from"]), tuple(s["to"])]
    pts = sorted(set(pts))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        lo, hi = sorted((y0, y1))
        if lo <= y <= hi and y1 != y0:
            return Fraction(x0) + Fraction(x1 - x0) * Fraction(y - y0) / Fraction(y1 - y0)
    raise AssertionError("y off the graph")


def _slope(spec, per=1):
    (x0, y0), (x1, y1) = spec["segments"][0]["from"], spec["segments"][-1]["to"]
    return abs(Fraction(y1 - y0) / Fraction(x1 - x0)) * per


def _read_real(question) -> str:
    """The answer recomputed from the drawn segments and the numbers in the question alone."""
    spec, text = question.figure, question.text
    pts = [tuple(p) for s in spec["segments"] for p in (s["from"], s["to"])]
    if len(spec["segments"]) == 3:  # the journey
        t1_, t2_ = pts[0], pts[-1]
        stop_start, stop_end = pts[2][0], pts[3][0]
        for key, template in _JOURNEY_Q.items():
            m = _pattern(template).search(text)
            if not m:
                continue
            if key == "stopped":
                return numtext(stop_end - stop_start)
            if key == "average":
                return numtext(Fraction(t2_[1], int(m["T"])))
            ta, tb = Fraction(m["t1"]), Fraction(m["t2"])
            return numtext((_value_at(spec, tb) - _value_at(spec, ta)) / (tb - ta))
        raise AssertionError(f"unrecognised journey question: {text}")
    for ctx in _all_contexts().values():
        for key in ("fixed", "read_y", "read_x", "rate", "extra", "far"):
            if key not in ctx:
                continue
            m = _pattern(ctx[key]).search(text)
            if not m:
                continue
            xs = ctx["xs"]
            if key == "fixed":
                return numtext(_value_at(spec, 0))
            if key == "read_y":
                return numtext(_value_at(spec, Fraction(m["x"])))
            if key == "read_x":
                return numtext(_x_at(spec, Fraction(m["y"])))
            if key == "rate":
                return numtext(_slope(spec, xs))
            slope = _slope(spec)
            if key == "extra":
                return numtext(_value_at(spec, 0) + slope * Fraction(m["X"]))
            return numtext((Fraction(m["B"]) - _value_at(spec, 0)) / slope)
    raise AssertionError(f"unrecognised question: {text}")


figures.register_reader("dg_real_data_graphs", _read_real)
