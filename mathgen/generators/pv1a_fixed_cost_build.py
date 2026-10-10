"""The fixed cost read off a contextual line.

One skill: ``senior-interpret-in-context-the-slope-and-intercept-of``
(ACMGM042), ``domain: linear-relations``, ``answer: number`` (an integer number
of dollars, graded exactly), ``prerequisites: [rate-of-change]``.

Scope: a straight-line
graph modelling a practical situation (total cost against a count) is drawn on
a labelled grid; the learner states the fixed cost (the y-intercept) in
dollars. The rate is never stated and the learner draws nothing.

Four tiers, escalating by the DECISION:

- ``read`` (easy): the y-axis is drawn from zero and the line meets it at a
  lattice point; the fixed cost is read at x = 0.
- ``offset`` (medium): the y scale starts above zero (the bottom-left corner
  is not the origin) but x = 0 is still drawn; the fixed cost is read off the
  y-axis against a scale that does not start at 0.
- ``extrapolate`` (hard): the x scale starts above 0 so the y-axis is not
  drawn; the learner finds the slope from two lattice points and extrapolates
  back to x = 0.
- ``named_fee`` (exam): extrapolation again, on stepped axes (x and y both
  step by more than 1 real unit per grid line) with a fractional rate per real
  unit, asked by the fee's own name (hire fee, joining fee, call-out fee,
  set-up cost, ...).

Geometry lives in GRID UNITS ``(u, v)`` (see ``pv1a_line_figure_build``); a
line ``v = s*u + c`` has an integer grid intercept ``c`` so the fixed cost
``c * ys`` is an integer number of dollars, and both through points are
lattice points, so nothing the key needs is interpolated.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen import figures
from mathgen.core import check, check_pairwise_distinct, generator
from mathgen.generators.pv1a_line_figure_build import (
    Window,
    lattice_points,
    line_figure,
    pick_two_points,
    read_line,
)

_ATTEMPTS = 400
_ALT = "a straight line on a coordinate grid"

# Each context: the activity (gerund phrase), the noun for one unit of x, the
# fee's own name (exam asks for it), the measure phrase, the x axis label, the
# y steps (dollars per grid line) and the x steps (real units per grid line)
# for the unstepped tiers and for the stepped exam tier.
_CONTEXTS = (
    dict(
        activity="hiring a kayak",
        who="a kayak hire shop",
        fee="hire fee",
        x_sg="hour",
        x_pl="hours",
        measure="the time hired, in hours",
        x_label="Time hired (hours)",
        ys=(5, 10, 15, 20, 25),
        xs=(1,),
        xs_exam=(2, 4),
    ),
    dict(
        activity="joining a gym",
        who="a gym",
        fee="joining fee",
        x_sg="class",
        x_pl="classes",
        measure="the number of classes attended",
        x_label="Classes attended",
        ys=(10, 20, 25, 30, 40),
        xs=(1,),
        xs_exam=(2, 4),
    ),
    dict(
        activity="calling out a plumber",
        who="a plumber",
        fee="call-out fee",
        x_sg="hour",
        x_pl="hours",
        measure="the time worked, in hours",
        x_label="Time worked (hours)",
        ys=(20, 25, 40, 50, 60),
        xs=(1,),
        xs_exam=(2, 4),
    ),
    dict(
        activity="a print run",
        who="a printing company",
        fee="set-up cost",
        x_sg="poster",
        x_pl="posters",
        measure="the number of posters printed",
        x_label="Posters printed",
        ys=(10, 20, 30, 40, 50),
        xs=(5, 10),
        xs_exam=(20, 50),
    ),
    dict(
        activity="hiring a bouncy castle",
        who="a party hire company",
        fee="hire fee",
        x_sg="hour",
        x_pl="hours",
        measure="the time hired, in hours",
        x_label="Time hired (hours)",
        ys=(20, 25, 30, 40, 50),
        xs=(1,),
        xs_exam=(2, 4),
    ),
    dict(
        activity="a mobile phone plan",
        who="a phone company",
        fee="connection fee",
        x_sg="gigabyte",
        x_pl="gigabytes",
        measure="the data used, in gigabytes",
        x_label="Data used (gigabytes)",
        ys=(10, 15, 20, 25, 30),
        xs=(1,),
        xs_exam=(2, 4, 5),
    ),
    dict(
        activity="a taxi trip",
        who="a taxi company",
        fee="flagfall",
        x_sg="kilometre",
        x_pl="kilometres",
        measure="the distance travelled, in kilometres",
        x_label="Distance travelled (kilometres)",
        ys=(5, 10, 15, 20, 25),
        xs=(1, 2),
        xs_exam=(4, 5),
    ),
    dict(
        activity="a furniture delivery",
        who="a furniture store",
        fee="delivery fee",
        x_sg="kilometre",
        x_pl="kilometres",
        measure="the distance delivered, in kilometres",
        x_label="Distance delivered (kilometres)",
        ys=(10, 20, 25, 30, 40),
        xs=(1, 2),
        xs_exam=(4, 5),
    ),
    dict(
        activity="a gardening service",
        who="a gardener",
        fee="call-out fee",
        x_sg="hour",
        x_pl="hours",
        measure="the time worked, in hours",
        x_label="Time worked (hours)",
        ys=(20, 25, 40, 50, 60),
        xs=(1,),
        xs_exam=(2, 4),
    ),
    dict(
        activity="a guided tour",
        who="a tour company",
        fee="booking fee",
        x_sg="hour",
        x_pl="hours",
        measure="the length of the tour, in hours",
        x_label="Tour length (hours)",
        ys=(10, 20, 25, 30, 40),
        xs=(1,),
        xs_exam=(2, 4),
    ),
)

_EASY_TEMPLATES = (
    "The graph shows the total cost, in dollars, of {activity} against {measure}. "
    "The total cost is a fixed cost plus a cost for each {x_sg}. What is the fixed cost, in dollars?",
    "{Activity} costs a fixed amount plus an amount for each {x_sg}. Use the graph of total cost "
    "(in dollars) against {measure}, to find the fixed cost, in dollars.",
    "Look at the graph of the total cost, in dollars, of {activity}. What is the fixed cost, in dollars: "
    "the part of the total cost that does not depend on {measure}?",
)
_HARD_TEMPLATES = (
    "The graph shows part of the total cost, in dollars, of {activity} against {measure}. "
    "The same straight-line rule holds all the way back to zero {x_pl}. What is the fixed cost, in dollars?",
    "Part of the graph of total cost, in dollars, for {activity} is shown, and it starts partway along the "
    "horizontal axis. Extend the line back to zero {x_pl}. What is the fixed cost, in dollars?",
    "{Activity} costs a fixed amount plus an amount for each {x_sg}. Only part of the graph is shown. "
    "What is the fixed cost, in dollars, that would be charged for zero {x_pl}?",
)
_EXAM_TEMPLATES = (
    "{Who} charges a {fee} plus an amount for each {x_sg}. Part of the graph of total cost, in dollars, "
    "against {measure} is shown. What is the {fee}, in dollars?",
    "The graph shows part of the total cost, in dollars, of {activity}. The cost is the {fee} plus a charge "
    "for each {x_sg}, and the same rule holds from zero {x_pl}. Find the {fee}, in dollars.",
    "A {fee} is charged for {activity} before any {x_pl} are counted. Use the graph of total cost, in "
    "dollars, against {measure}, to work out the {fee}. Give your answer in dollars.",
)
_TEMPLATES = {
    "read": _EASY_TEMPLATES,
    "offset": _EASY_TEMPLATES,
    "extrapolate": _HARD_TEMPLATES,
    "named_fee": _EXAM_TEMPLATES,
}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _draw_window_line(rng, mode: str, xs: int, ys: int):
    """One candidate ``(Window, slope, intercept)`` for the tier, or None."""
    if mode == "read":
        u0, v0 = 0, 0
        u1, v1 = rng.randint(6, 9), rng.randint(6, 9)
        slope = Fraction(rng.choice((1, 2)))
        c = rng.randint(1, 4)
    elif mode == "offset":
        u0, v0 = 0, rng.randint(1, 4)
        u1, v1 = rng.randint(6, 9), v0 + rng.randint(5, 9)
        slope = rng.choice((Fraction(1), Fraction(2), Fraction(3), Fraction(1, 2)))
        c = rng.randint(v0, v0 + 4)
    else:  # extrapolate, named_fee: x starts above 0 so the y-axis is not drawn
        u0 = rng.randint(1, 3)
        u1 = u0 + rng.randint(5, 9)
        v0 = rng.randint(0, 3)
        v1 = v0 + rng.randint(5, 9)
        slope = rng.choice((Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)))
        c = rng.randint(1, 6)
    win = Window(u0, u1, v0, v1, xs=xs, ys=ys)
    return win, slope, c


def _answer_text(value: Fraction) -> str:
    check(value.denominator == 1, "fixed cost must be a whole number of dollars")
    return str(int(value))


def _build(rng, mode: str) -> dict:
    for _ in range(_ATTEMPTS):
        ctx = rng.choice(_CONTEXTS)
        ys = rng.choice(ctx["ys"])
        xs = rng.choice(ctx["xs_exam"] if mode == "named_fee" else ctx["xs"])
        win, slope, c = _draw_window_line(rng, mode, xs, ys)
        real_rate = slope * ys / xs
        if mode == "named_fee" and real_rate.denominator == 1:
            continue  # the exam tier's rate per real unit is fractional
        pts = lattice_points(win, slope, Fraction(c))
        through_pool = [p for p in pts if p[0] >= max(1, win.u0)]
        try:
            p, q = pick_two_points(rng, through_pool, min_du=2)
        except ValueError:
            continue
        p, q = sorted((p, q))
        fig = line_figure(
            win, [(p, q)], _ALT, x_label=ctx["x_label"], y_label="Total cost (dollars)", equal_scale=False
        )

        # The key, re-derived from the drawn figure (two through points).
        m_real, c_real = read_line(fig)
        key = _answer_text(c_real)
        check(c_real == Fraction(c * ys), "figure intercept disagrees with the construction")
        check(m_real == real_rate, "figure slope disagrees with the construction")

        first_y = p[1] * ys  # cost at the first through point
        far = max(pts)  # the largest-x lattice point on the line in the window
        far_y = far[1] * ys  # cost at the largest x
        candidates = [(first_y, "first-point-cost"), (far_y, "cost-at-largest-x")]
        if real_rate.denominator == 1:
            candidates.append((int(real_rate), "rate-as-fixed-cost"))
        if xs > 1:
            # the rise per grid step used as if it were the rate per real unit of x
            confused = Fraction(first_y) - slope * ys * (p[0] * xs)
            if confused.denominator == 1:
                candidates.append((int(confused), "grid-step-as-unit"))
        seen = {int(key)}
        distractors = []
        for val, _name in candidates:
            val = int(val)
            if val <= 0 or val in seen:
                continue
            seen.add(val)
            distractors.append(str(val))
        if len(distractors) < 2:
            continue
        distractors = distractors[:3]
        check_pairwise_distinct(
            int(key),
            *[int(d) for d in distractors],
            msg="fixed-cost distractor collides with the key or another option",
        )
        try:
            figures.assert_no_leak(fig, [key, *distractors])
        except AssertionError:
            continue

        tpl = rng.choice(_TEMPLATES[mode])
        text = tpl.format(Activity=_cap(ctx["activity"]), Who=_cap(ctx["who"]), **ctx)
        working = [
            f"The line passes through ({p[0] * xs}, {p[1] * ys}) and ({q[0] * xs}, {q[1] * ys}).",
            f"The rate is ({q[1] * ys} - {p[1] * ys}) / ({q[0] * xs} - {p[0] * xs}) = {m_real} (dollars per {ctx['x_sg']}).",
            f"At x = 0 the cost is {p[1] * ys} - {m_real} × {p[0] * xs} = {key} (dollars).",
        ]
        return dict(
            text=text,
            latex=text,
            answer=int(key),
            answer_text=key,
            answer_latex=key,
            working=working,
            distractors=distractors,
            figure=fig,
        )
    raise ValueError(f"pv1a_fixed_cost: no valid {mode} draw")


@generator(
    "pv1a_fixed_cost",
    level=12,
    presets={
        "easy": dict(mode="read"),
        "medium": dict(mode="offset"),
        "hard": dict(mode="extrapolate"),
        "exam": dict(mode="named_fee"),
    },
)
def pv1a_fixed_cost(rng, mode="read"):
    """The fixed cost (y-intercept, dollars) of a straight-line cost graph.
    read: y-axis drawn from 0, read at x = 0. offset: y scale starts above 0.
    extrapolate: x starts above 0, extend back with the slope. named_fee:
    stepped axes, fractional rate, the fee asked by name."""
    if mode not in _TEMPLATES:
        raise ValueError(f"unknown mode {mode!r}")
    return _build(rng, mode)


def read_pv1a_fixed_cost(question) -> str:
    """The fixed cost from the drawn line alone: its y-value at x = 0."""
    _m, c = read_line(question.figure)
    return _answer_text(c)


figures.register_reader("pv1a_fixed_cost", read_pv1a_fixed_cost)
