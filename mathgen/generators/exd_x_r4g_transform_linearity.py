"""Exam chain for a straight line fitted after a transformation (x squared, log x or 1 / x).

A table of four exact points is given with the transformation to use, z = f(x).

(a) the gradient of y against z, from the first and last points                 number, 2 marks
(b) the intercept of the line (follow-through (a))                              number, 1 mark
(c) the predicted y at a stated x (follow-through (a), (b))                     number, 2 marks
(d) the x that gives a stated y (follow-through (a), (b))                       number, 2 marks

The table is checked to be exactly linear in the stated z and in neither of the other two transformations.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4g_kit import number_part, number_text, retry
from .r4g_transform_linearity_build import (
    CONTEXTS,
    LOG_NOTE,
    NAMES,
    SHAPES,
    clean,
    draw_model,
    places_for,
    value,
    z_exact,
    z_float,
)


def _z_text(z: Fraction) -> str:
    return str(z.numerator) if z.denominator == 1 else f"{z.numerator}/{z.denominator}"


@distinct_part_answers
def _build(rng):
    for _ in attempts(2000, "a table that is linear in exactly one transformation"):
        kind = rng.choice(("sq", "log", "rec"))
        a, b, xs = draw_model(rng, kind)
        if kind == "sq":
            table, pool = [2, 3, 4, 5], [6, 7, 8, 9]
        elif kind == "log":
            table, pool = [10, 100, 1000, 10000], [100000]
        else:
            if len(xs) < 7:
                continue
            table, pool = sorted(xs)[:4], sorted(xs)[4:]
        ys = [value(kind, a, b, x) for x in table]
        if not all(clean(y) for y in ys):
            continue
        if kind == "log":
            x0, xt = 100000, 1000000  # two powers of ten beyond the table
        else:
            if len(pool) < 2:
                continue
            x0, xt = rng.sample(pool, 2)
        y0, yt = value(kind, a, b, x0), value(kind, a, b, xt)
        if clean(y0) and clean(yt):
            break
    z1, z4 = z_exact(kind, table[0]), z_exact(kind, table[-1])
    gradient = (ys[-1] - ys[0]) / (z4 - z1)
    check(gradient == b and ys[0] - gradient * z1 == a, "gradient or intercept disagrees with the model")
    xs_text = ", ".join(str(x) for x in table)
    ys_text = ", ".join(number_text(y, places_for(*ys)) for y in ys)
    note = f" {LOG_NOTE}" if kind == "log" else ""
    text = (
        f"{rng.choice(CONTEXTS[kind])} {SHAPES[kind]} The values of y at x = {xs_text} are {ys_text}.{note} "
        f"To straighten the scatterplot, let z = {NAMES[kind]} and fit a line y = a + bz through the first and last points."
    )
    first_raw = (ys[-1] - ys[0]) / (table[-1] - table[0])
    z0, zt = z_exact(kind, x0), z_exact(kind, xt)
    other_z = {k: [z_float(k, x) for x in (table[0], table[-1], x0)] for k in ("sq", "log", "rec")}

    def wrong_pred(k):
        zz1, zz4, zz0 = other_z[k]
        bb = (float(ys[-1]) - float(ys[0])) / (zz4 - zz1)
        return Fraction(float(ys[0]) - bb * zz1 + bb * zz0).limit_denominator(1000)

    others = [k for k in ("sq", "log", "rec") if k != kind]
    ft_inverse = {
        "sq": f"sqrt((({yt}) - part_b)/part_a)",
        "log": f"10**((({yt}) - part_b)/part_a)",
        "rec": f"part_a/(({yt}) - part_b)",
    }[kind]
    parts = [
        number_part(
            f"Find the gradient b of the line, from the first and last points.",
            gradient,
            places_for(gradient),
            2,
            [
                f"b = ({number_text(ys[-1], places_for(ys[-1]))} - {number_text(ys[0], places_for(ys[0]))}) / ({_z_text(z4)} - {_z_text(z1)}) = {number_text(gradient, places_for(gradient))}."
            ],
            [
                (first_raw, "r4g-gradient-on-raw-x"),
                (ys[-1] - ys[0], "r4g-rise-without-the-run"),
                (1 / gradient, "r4g-gradient-inverted"),
                (z4 - z1, "r4g-run-for-gradient"),
            ],
        ),
        number_part(
            "Use your gradient to find the intercept a of the line.",
            a,
            places_for(a),
            1,
            [
                f"a = {number_text(ys[0], places_for(ys[0]))} - {number_text(gradient, places_for(gradient))} x {_z_text(z1)} = {number_text(a, places_for(a))}."
            ],
            [
                (ys[0], "r4g-first-y-as-intercept"),
                (ys[0] - gradient * table[0], "r4g-transformed-variable-not-computed"),
                (ys[0] + gradient * z1, "r4g-intercept-sign-slip"),
                (gradient, "r4g-gradient-given-as-intercept"),
            ],
            follow_through=f"{ys[0]} - part_a*{_z_text(z1)}",
            sensible_range=(None, None),
        ),
        number_part(
            f"Use your line to predict y when x = {x0}.",
            y0,
            places_for(y0),
            2,
            [
                f"z = {_z_text(z0)}, so y = {number_text(a, places_for(a))} + {number_text(gradient, places_for(gradient))} x {_z_text(z0)} = {number_text(y0, places_for(y0))}."
            ],
            [
                (a + gradient * x0, "r4g-transformed-variable-not-computed"),
                (Fraction(wrong_pred(others[0])), "r4g-wrong-transformation-chosen"),
                (Fraction(wrong_pred(others[1])), "r4g-wrong-transformation-chosen"),
                (ys[-1] + first_raw * (x0 - table[-1]), "r4g-straight-line-through-raw-points"),
            ],
            follow_through=f"part_b + part_a*{_z_text(z0)}",
        ),
        number_part(
            f"Use your line to find the value of x for which y = {number_text(yt, places_for(yt))}.",
            xt,
            0,
            2,
            [
                f"z = ({number_text(yt, places_for(yt))} - {number_text(a, places_for(a))}) / {number_text(gradient, places_for(gradient))} = {_z_text(zt)}, so x = {xt}."
            ],
            [
                (zt, "r4g-transformed-value-given-as-x"),
                (
                    (yt - a) / gradient * 1 if kind != "log" else (yt - a) / gradient * 10,
                    "r4g-transformed-value-given-as-x",
                ),
                (Fraction(table[-1]) + (yt - ys[-1]) / first_raw, "r4g-straight-line-through-raw-points"),
                (xt * 10 if kind != "rec" else xt + 1, "r4g-undoing-the-transformation-slip"),
            ],
            follow_through=ft_inverse,
        ),
    ]
    return dict(text=text, parts=parts)


@generator("exd_x_r4g_transform_linearity", level=11, presets={"exam": {}})
def exd_x_r4g_transform_linearity(rng):
    return retry(_build, rng)
