"""Transforming data to linearity: `r4g_transform_linearity` (easy, medium, hard).

``data-transformation-to-linearity`` (``statistics-probability``, answer ``number``). Three one-step transformations
straighten a curved scatterplot, and each is recognised by the shape of the curve:

    x squared  (z = x^2):      y rises and the curve gets steeper                (y = a + b x^2)
    log x      (z = log10 x):  y rises quickly at first then flattens            (y = a + b log x)
    1 / x      (z = 1/x):      y falls quickly at first then flattens to a floor (y = a + b / x)

with b positive in every case. The x values are chosen so every key is exact (powers of ten for log x, divisors of b
for 1/x) and shown to at most 1 decimal place.

* easy: the shape is described and three fitted lines are given, one for each transformation; predict y at a stated
  x using the line that matches the shape.
* medium: the shape is described and two points on the curve are given; build the line on the transformed variable
  and predict y at a stated x.
* hard: no description, only a table of four points; recognise which transformation makes the data linear and then
  predict y at a new x, or find the x that gives a stated y.

Each key is checked by evaluating the model from its coefficients, and a table is checked to be exactly linear in
its own transformed variable and in neither of the others.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4g_kit import number_text, single_number, retry

SHAPES = {
    "sq": "The scatterplot of y against x is increasing and curves upward, getting steeper as x increases.",
    "log": "The scatterplot of y against x is increasing, rising quickly at first and then flattening out.",
    "rec": "The scatterplot of y against x is decreasing, falling quickly at first and then flattening out towards a floor.",
}
CONTEXTS = {
    "sq": [
        "Let x be a car's speed in tens of kilometres per hour and y its stopping distance in metres.",
        "Let x be the side length of a square lawn in metres and y the cost in dollars of laying turf on it.",
    ],
    "log": [
        "Let x be the dose of a nutrient in milligrams and y the growth of a culture in millimetres.",
        "Let x be the money spent on advertising in dollars and y the number of new customers.",
    ],
    "rec": [
        "Let x be the number of painters on a job and y the time taken in hours.",
        "Let x be the number of pickers in a crew and y the time to harvest a field in hours.",
    ],
}
NAMES = {"sq": "x squared", "log": "log x", "rec": "1 / x"}
LOG_NOTE = "Here log x means the logarithm of x to base 10."


def z_exact(kind: str, x: int) -> Fraction:
    if kind == "sq":
        return Fraction(x * x)
    if kind == "rec":
        return Fraction(1, x)
    k = len(str(x)) - 1
    check(10**k == x, "log x is only exact for powers of ten")
    return Fraction(k)


def z_float(kind: str, x) -> float:
    return {"sq": x * x, "log": math.log10(x), "rec": 1 / x}[kind]


def value(kind: str, a, b, x) -> Fraction:
    return Fraction(a) + Fraction(b) * z_exact(kind, x)


def line_text(kind: str, a, b) -> str:
    a, b = (
        number_text(a, 0 if Fraction(a).denominator == 1 else 1),
        number_text(b, 0 if Fraction(b).denominator == 1 else 1),
    )
    return {"sq": f"y = {a} + {b}x²", "log": f"y = {a} + {b} log x", "rec": f"y = {a} + {b}/x"}[kind]


def draw_model(rng, kind):
    """(a, b, usable x values) with b positive and a at least 1."""
    a = Fraction(rng.randint(1, 15))
    if kind == "sq":
        return a, Fraction(rng.choice((1, 2, 3, 4, 5)), rng.choice((1, 2))), [2, 3, 4, 5, 6, 7, 8, 9]
    if kind == "log":
        return a, Fraction(rng.choice((2, 3, 4, 5, 6, 8, 10))), [1, 10, 100, 1000, 10000]
    b = Fraction(rng.choice((12, 20, 24, 30, 40, 60)))
    return a, b, [x for x in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20) if (b / x * 10).denominator == 1]


def places_for(*vals):
    return 0 if all(Fraction(v).denominator == 1 for v in vals) else 1


def clean(v) -> bool:
    return (Fraction(v) * 10).denominator == 1


def fit_through(kind, p1, p2):
    """(a, b) of the line y = a + b z through two points, using floats when z is not exact."""
    (x1, y1), (x2, y2) = p1, p2
    z1, z2 = z_float(kind, x1), z_float(kind, x2)
    b = (y2 - y1) / (z2 - z1)
    return y1 - b * z1, b


def wrong_models(kind, p1, p2, x0):
    """Predictions at x0 from lines through p1 and p2 fitted to each of the OTHER transformations, with the tag."""
    out = []
    for other in ("sq", "log", "rec"):
        if other == kind:
            continue
        a, b = fit_through(other, p1, p2)
        out.append((Fraction(a + b * z_float(other, x0)), "r4g-wrong-transformation-chosen"))
    return out


def _shape_for(kind):
    return SHAPES[kind] + (f" {LOG_NOTE}" if kind == "log" else "")


def _easy(rng):
    for _ in attempts(500, "a prediction with one decimal place"):
        kind = rng.choice(("sq", "log", "rec"))
        a, b, xs = draw_model(rng, kind)
        x0 = rng.choice({"sq": [6, 7, 8, 9, 10, 12], "log": [10, 100, 1000], "rec": xs}[kind])
        if kind == "rec" and x0 not in xs:
            continue
        ans = value(kind, a, b, x0)
        if clean(ans):
            break
    others = [k for k in ("sq", "log", "rec") if k != kind]
    fits = {kind: (a, b)}
    for k in others:
        _, ob, _ = draw_model(rng, k)
        fits[k] = (Fraction(rng.randint(1, 15)), ob)
    order = list(fits)
    rng.shuffle(order)
    labels = dict(zip(order, "PQR"))
    models = "; ".join(f"Model {labels[k]}: {line_text(k, *fits[k])}" for k in sorted(order, key=lambda k: labels[k]))
    text = (
        f"{rng.choice(CONTEXTS[kind])} {_shape_for(kind)} Three lines were fitted after transforming x in different ways. "
        f"{models}. Use the model that suits the shape of the scatterplot to predict y when x = {x0}."
    )
    slips = []
    for k in others:
        ak, bk = fits[k]
        slips.append(
            (
                Fraction(ak) + Fraction(bk) * Fraction(z_float(k, x0)).limit_denominator(10**6),
                "r4g-wrong-transformation-chosen",
            )
        )
    slips += [
        (a + b * x0, "r4g-transformed-variable-not-computed"),
        (a * b, "r4g-intercept-times-slope"),
    ]
    places = places_for(ans)
    working = [
        f"The shape matches the {NAMES[kind]} transformation, which is Model {labels[kind]}.",
        f"y = {number_text(a, places_for(a))} + {number_text(b, places_for(b))} x {number_text(z_exact(kind, x0), places_for(z_exact(kind, x0)) if kind != 'rec' else 3)} = {number_text(ans, places)}.",
    ]
    return single_number(text, ans, places, working, slips)


def _medium(rng):
    for _ in attempts(500, "a prediction with one decimal place"):
        kind = rng.choice(("sq", "log", "rec"))
        a, b, xs = draw_model(rng, kind)
        if kind == "sq":
            x1, x2, x0 = rng.sample([2, 3, 4, 5, 6], 2) + [rng.choice((8, 9, 10))]
        elif kind == "log":
            x1, x2, x0 = 10, 1000, rng.choice((100, 10000))
        else:
            if len(xs) < 4:
                continue
            x1, x2, x0 = sorted(rng.sample(xs, 3))
        x1, x2 = sorted((x1, x2))
        if x1 == x2:
            continue
        y1, y2 = value(kind, a, b, x1), value(kind, a, b, x2)
        ans = value(kind, a, b, x0)
        if clean(ans) and clean(y1) and clean(y2):
            break
    text = (
        f"{rng.choice(CONTEXTS[kind])} {_shape_for(kind)} The points (x, y) = ({x1}, {number_text(y1, places_for(y1))}) and "
        f"({x2}, {number_text(y2, places_for(y2))}) lie on the curve. Transform x so that the points lie on a straight line, "
        f"then use that line to predict y when x = {x0}."
    )
    p1, p2 = (x1, float(y1)), (x2, float(y2))
    slips = wrong_models(kind, p1, p2, x0)
    slope_raw = (y2 - y1) / (x2 - x1)
    slips += [
        (y1 + slope_raw * (x0 - x1), "r4g-straight-line-through-raw-points"),
        (value(kind, a, b, x0) - a, "r4g-intercept-left-out"),
    ]
    z1, z2, z0 = z_exact(kind, x1), z_exact(kind, x2), z_exact(kind, x0)
    check(y1 + (y2 - y1) / (z2 - z1) * (z0 - z1) == ans, "line through the transformed points disagrees with the model")
    places = places_for(ans)
    working = [
        f"The shape matches the {NAMES[kind]} transformation, so use z = {NAMES[kind]}: the points become ({number_text(z1, 3 if kind == 'rec' else 0)}, {number_text(y1, places_for(y1))}) and ({number_text(z2, 3 if kind == 'rec' else 0)}, {number_text(y2, places_for(y2))}).",
        f"The line has gradient {number_text((y2 - y1) / (z2 - z1), 2)} and gives y = {number_text(ans, places)} when x = {x0}.",
    ]
    return single_number(text, ans, places, working, slips)


def _hard(rng):
    for _ in attempts(1000, "a table that is linear in exactly one transformation"):
        kind = rng.choice(("sq", "log", "rec"))
        a, b, xs = draw_model(rng, kind)
        if kind == "sq":
            table = [1, 2, 3, 4]
            pool = [6, 7, 8]
        elif kind == "log":
            table = [1, 10, 100, 1000]
            pool = [10000]
        else:
            if len(xs) < 6:
                continue
            table = sorted(xs)[:4]
            pool = [x for x in sorted(xs)[4:]]
        ys = [value(kind, a, b, x) for x in table]
        if not all(clean(y) for y in ys) or not pool:
            continue
        inverse = rng.random() < 0.45
        x0 = rng.choice(pool)
        ans = value(kind, a, b, x0)
        if not clean(ans):
            continue

        # the table must be exactly linear in its own transformation and in neither of the others
        def linear(k):
            zs = [Fraction(z_float(k, x)).limit_denominator(10**9) for x in table]
            g = [(ys[i + 1] - ys[i]) / (zs[i + 1] - zs[i]) for i in range(3)]
            return abs(g[0] - g[1]) < Fraction(1, 10**6) and abs(g[1] - g[2]) < Fraction(1, 10**6)

        if not linear(kind) or any(linear(k) for k in ("sq", "log", "rec") if k != kind):
            continue
        break
    xs_text = ", ".join(str(x) for x in table)
    ys_text = ", ".join(number_text(y, places_for(*ys)) for y in ys)
    ctx = rng.choice(CONTEXTS[kind])
    note = f" {LOG_NOTE}" if kind == "log" else ""
    base = f"{ctx} The scatterplot of y against x is curved. For x = {xs_text}, the values of y are {ys_text}.{note}"
    if inverse:
        yt = ans
        text = f"{base} Transform x to make the data linear, then find the value of x for which the line gives y = {number_text(yt, places_for(yt))}."
        goal, label = x0, "x"
        slips = []
        for other in ("sq", "log", "rec"):
            if other == kind:
                continue
            a_o, b_o = fit_through(other, (table[0], float(ys[0])), (table[-1], float(ys[-1])))
            z_target = (float(yt) - a_o) / b_o
            try:
                guess = {"sq": math.sqrt(z_target), "log": 10**z_target, "rec": 1 / z_target}[other]
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            slips.append((Fraction(guess).limit_denominator(1000), "r4g-wrong-transformation-chosen"))
        raw_slope = (ys[-1] - ys[0]) / (table[-1] - table[0])
        slips.append((table[0] + (yt - ys[0]) / raw_slope, "r4g-straight-line-through-raw-points"))
        z_goal = (yt - a) / b
        slips.append((z_goal, "r4g-transformed-value-given-as-x"))
        working = [
            f"The data are linear against {NAMES[kind]}: y = {number_text(a, places_for(a))} + {number_text(b, places_for(b))} z.",
            f"For y = {number_text(yt, places_for(yt))}, z = {number_text(z_goal, 3)}, so x = {x0}.",
        ]
        return single_number(text, goal, 0, working, slips)
    text = f"{base} Transform x to make the data linear, then use the line to predict y when x = {x0}."
    p1, p2 = (table[0], float(ys[0])), (table[-1], float(ys[-1]))
    slips = wrong_models(kind, p1, p2, x0)
    raw_slope = (ys[-1] - ys[0]) / (table[-1] - table[0])
    slips.append((ys[-1] + raw_slope * (x0 - table[-1]), "r4g-straight-line-through-raw-points"))
    places = places_for(ans)
    working = [
        f"The y values are linear against {NAMES[kind]}: y = {number_text(a, places_for(a))} + {number_text(b, places_for(b))} z.",
        f"At x = {x0}, y = {number_text(ans, places)}.",
    ]
    return single_number(text, ans, places, working, slips)


_TIER = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "r4g_transform_linearity",
    level=11,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def r4g_transform_linearity(rng, tier="easy"):
    return retry(_TIER[tier], rng)
