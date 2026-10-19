"""The meaning of the slope of a straight-line cost graph: `r3j_slope_context` (easy, medium, hard) and the exam
chain `exd_x_r3j_slope_two_plans`.

``interpret-slope-in-context`` (``functions-sequences``, answer ``number``, ``needs_figure``). A straight-line graph of
total cost against a count is drawn on a labelled grid. The rate is never printed.

* easy: the cost of each extra unit, from a line on unit steps whose rate is a whole number of dollars.
* medium: the same, on stepped axes (a grid step is more than one unit of x), so the rate per unit is not the rise
  per grid line.
* hard: the rate put to use: the extra cost of a stated number of extra units, how many extra units a stated amount
  buys, or the total cost for a count beyond the drawn graph.
* exam (``exd_x_r3j_slope_two_plans``): two plans on one grid; the rate of each, how much more one charges per unit,
  and the count at which the two cost the same.

Geometry is in grid units, using the line-figure helpers of the neighbouring graph skills; every drawn line passes
through two lattice points, and the key is rebuilt from the drawn figure alone (`read_line`, `read_intersection`).
"""

from __future__ import annotations

import re
from fractions import Fraction

from .. import figures
from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import number_part, number_text, plain, sane, single_number
from .pv1a_line_figure_build import Window, lattice_points, line_figure, pick_two_points, read_intersection, read_line

_ALT = "a straight line on a coordinate grid"
_ALT_TWO = "two straight lines on a coordinate grid"

_CONTEXTS = [
    dict(
        activity="hiring a kayak",
        rmin=Fraction("5"),
        rmax=Fraction("30"),
        x_sg="hour",
        x_pl="hours",
        measure="the time hired, in hours",
        x_label="Time hired (hours)",
        ys=(5, 10, 15, 20),
        xs=(2, 4),
        xs1=1,
        pair="two kayak hire shops",
    ),
    dict(
        activity="a gym membership",
        rmin=Fraction("5"),
        rmax=Fraction("25"),
        x_sg="class",
        x_pl="classes",
        measure="the number of classes attended",
        x_label="Classes attended",
        ys=(10, 20, 25, 30),
        xs=(2, 4),
        xs1=1,
        pair="two gyms",
    ),
    dict(
        activity="a plumber's visit",
        rmin=Fraction("40"),
        rmax=Fraction("110"),
        x_sg="hour",
        x_pl="hours",
        measure="the time worked, in hours",
        x_label="Time worked (hours)",
        ys=(20, 25, 40, 50),
        xs=(2, 4),
        xs1=1,
        pair="two plumbers",
    ),
    dict(
        activity="a print run",
        rmin=Fraction("0.2"),
        rmax=Fraction("3"),
        x_sg="poster",
        x_pl="posters",
        measure="the number of posters printed",
        x_label="Posters printed",
        ys=(10, 20, 30, 40),
        xs=(5, 10),
        xs1=5,
        pair="two print shops",
    ),
    dict(
        activity="a mobile phone plan",
        rmin=Fraction("2"),
        rmax=Fraction("12"),
        x_sg="gigabyte",
        x_pl="gigabytes",
        measure="the data used, in gigabytes",
        x_label="Data used (gigabytes)",
        ys=(10, 15, 20, 25),
        xs=(2, 4, 5),
        xs1=1,
        pair="two phone companies",
    ),
    dict(
        activity="a taxi trip",
        rmin=Fraction("1.5"),
        rmax=Fraction("5"),
        x_sg="kilometre",
        x_pl="kilometres",
        measure="the distance travelled, in kilometres",
        x_label="Distance travelled (kilometres)",
        ys=(5, 10, 15, 20),
        xs=(2, 4, 5),
        xs1=1,
        pair="two taxi companies",
    ),
    dict(
        activity="a furniture delivery",
        rmin=Fraction("1.5"),
        rmax=Fraction("8"),
        x_sg="kilometre",
        x_pl="kilometres",
        measure="the distance delivered, in kilometres",
        x_label="Distance delivered (kilometres)",
        ys=(10, 20, 25, 30),
        xs=(4, 5),
        xs1=2,
        pair="two delivery firms",
    ),
]
_Y_LABEL = "Total cost (dollars)"
_NAMES = ["Mia", "Ravi", "Chloe", "Tom", "Aisha", "Liam", "Sofia", "Kofi", "Priya", "Jack"]
_EASY_FRAMES = [
    "{who} plots the total cost, in dollars, of {activity} against {measure}. How much does each extra {x_sg} cost, in dollars?",
    "The graph shows the total cost, in dollars, of {activity} against {measure}. What does one more {x_sg} add to the cost, in dollars?",
    "{who} is checking a quote for {activity}. The graph shows total cost in dollars against {measure}. By how many dollars does the cost go up for each extra {x_sg}?",
    "This graph of total cost, in dollars, against {measure} describes {activity}. What is the cost of each extra {x_sg}, in dollars?",
]
_MEDIUM_FRAMES = [
    "{who} reads the graph of the total cost, in dollars, of {activity} against {measure}. Find the rate of change of the cost, in dollars per {x_sg}.",
    "The line shows how the total cost of {activity} depends on {measure}. What is the gradient of the line in dollars per {x_sg}, as a decimal?",
    "A graph of the total cost, in dollars, of {activity} is drawn against {measure}. Work out the extra cost, in dollars, of one more {x_sg}.",
    "{who} is comparing prices for {activity}. Using the graph of cost against {measure}, find the change in cost, in dollars, per {x_sg}.",
]


def _cap(text):
    return text[0].upper() + text[1:]


def _two_decimals(value) -> bool:
    return (Fraction(value) * 100).denominator == 1


def _draw_line(rng, *, stepped):
    """(ctx, win, slope, c, p, q): one line through two lattice points of the window, y-axis drawn."""
    for _ in attempts(2000, "a line with two lattice points and a rate with at most two decimals"):
        ctx = rng.choice(_CONTEXTS)
        ys = rng.choice(ctx["ys"])
        xs = rng.choice(ctx["xs"]) if stepped else ctx["xs1"]
        slope = rng.choice(
            [Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)]
            if stepped
            else [Fraction(1), Fraction(2), Fraction(3)]
        )
        c = rng.randint(1, 3)
        win = Window(0, 8, 0, 9, xs=xs, ys=ys)
        rate = slope * ys / xs
        if not _two_decimals(rate) or (stepped and rate.denominator == 1) or (not stepped and rate.denominator != 1):
            continue
        if not ctx["rmin"] <= rate <= ctx["rmax"]:
            continue
        pts = [p for p in lattice_points(win, slope, Fraction(c)) if p[0] >= 1]
        try:
            p, q = sorted(pick_two_points(rng, pts, min_du=3))
        except ValueError:
            continue
        return ctx, win, slope, c, p, q
    raise AssertionError  # pragma: no cover


def _figure(win, lines, ctx, labels=None, alt=_ALT):
    return line_figure(win, lines, alt, labels=labels, x_label=ctx["x_label"], y_label=_Y_LABEL, equal_scale=False)


def _rate_slips(win, slope, c, p, q, rate):
    ys, xs = win.ys, win.xs
    return [
        (Fraction(c * ys), "r3j-slope-fixed-cost-as-rate"),
        (Fraction((q[1] - p[1]) * ys), "r3j-slope-rise-as-rate"),
        (
            slope * ys if xs > 1 else Fraction(q[1] * ys),
            "r3j-slope-grid-step-as-unit" if xs > 1 else "r3j-slope-total-as-rate",
        ),
        (Fraction(q[1] * ys), "r3j-slope-total-as-rate"),
        (Fraction(xs) / rate, "r3j-slope-run-over-rise"),
        (rate + c * ys, "r3j-slope-first-unit-total"),
    ]


def _easy_or_medium(rng, stepped):
    ctx, win, slope, c, p, q = _draw_line(rng, stepped=stepped)
    fig = _figure(win, [(p, q)], ctx)
    m, intercept = read_line(fig)
    rate = slope * win.ys / win.xs
    check(m == rate and intercept == c * win.ys, "figure disagrees with the construction")
    places = 0 if rate.denominator == 1 else 2
    who = rng.choice(_NAMES)
    frames = _MEDIUM_FRAMES if stepped else _EASY_FRAMES
    text = rng.choice(frames).format(who=who, activity=ctx["activity"], measure=ctx["measure"], x_sg=ctx["x_sg"])
    working = [
        f"The line passes through ({p[0] * win.xs}, {p[1] * win.ys}) and ({q[0] * win.xs}, {q[1] * win.ys}).",
        f"Rate = ({q[1] * win.ys} - {p[1] * win.ys}) / ({q[0] * win.xs} - {p[0] * win.xs}) = {number_text(rate, places)} dollars per {ctx['x_sg']}.",
    ]
    slips = _rate_slips(win, slope, c, p, q, rate)
    question = single_number(text, rate, places, working, slips, spread=100)
    return _guard_leak(question, fig)


def _guard_leak(question, fig):
    """Attach the figure; refuse a draw whose key or options are printed on the figure's own labels."""
    texts = [question["answer_text"]] + [d.text for d in question["distractors"]]
    figures.assert_no_leak(fig, texts)
    question["figure"] = fig
    return question


def _hard(rng):
    for _ in attempts(500, "a hard slope draw that does not leak"):
        ctx, win, slope, c, p, q = _draw_line(rng, stepped=True)
        fig = _figure(win, [(p, q)], ctx)
        m, intercept = read_line(fig)
        rate = slope * win.ys / win.xs
        variant = rng.choice(["extra", "count", "predict"])
        if variant == "extra":
            k = rng.choice([20, 25, 30, 40, 50, 60, 100])
            key, places = k * rate, 2
            text = (
                f"The graph shows the total cost, in dollars, of {ctx['activity']} against {ctx['measure']}. "
                f"How much extra does it cost for {k} more {ctx['x_pl']}?"
            )
            slips = [
                (k * intercept, "r3j-slope-fixed-cost-for-extra"),
                (k * rate + intercept, "r3j-slope-fixed-cost-added"),
                (k * (q[1] - p[1]) * win.ys, "r3j-slope-rise-as-rate"),
                (k * slope * win.ys, "r3j-slope-grid-step-as-unit"),
                (k / rate, "r3j-slope-divided-for-multiplied"),
            ]
            working = [
                f"Rate = {number_text(rate, 2)} dollars per {ctx['x_sg']}.",
                f"Extra cost = {k} x {number_text(rate, 2)} = {number_text(key, 2)}.",
            ]
        elif variant == "count":
            j = rng.randint(6, 30)
            budget = j * rate
            if not _two_decimals(budget):
                continue
            key, places = Fraction(j), 0
            text = (
                f"The graph shows the total cost, in dollars, of {ctx['activity']} against {ctx['measure']}. "
                f"How many more {ctx['x_pl']} can be bought for an extra {_dollars(budget)}?"
            )
            slips = [
                (budget * rate, "r3j-slope-multiplied-for-divided"),
                (budget / (rate + intercept), "r3j-slope-fixed-cost-added"),
                (budget / (slope * win.ys), "r3j-slope-grid-step-as-unit"),
                (budget / (q[1] - p[1]) / win.ys, "r3j-slope-rise-as-rate"),
                (budget / intercept, "r3j-slope-fixed-cost-for-extra"),
            ]
            working = [
                f"Rate = {number_text(rate, 2)} dollars per {ctx['x_sg']}.",
                f"{_dollars(budget)} / {number_text(rate, 2)} = {j}.",
            ]
        else:
            n = rng.choice([20, 25, 30, 40, 50])
            if n <= win.u1 * win.xs:
                n += win.u1 * win.xs
            key, places = n * rate + intercept, 2
            text = (
                f"The graph shows the total cost, in dollars, of {ctx['activity']} against {ctx['measure']}. "
                f"Assuming the same straight-line rule continues, what would the total cost be for {n} {ctx['x_pl']}?"
            )
            slips = [
                (n * rate, "r3j-slope-fixed-cost-left-out"),
                (n * rate + intercept * 2, "r3j-slope-fixed-cost-counted-twice"),
                (n * (q[1] * win.ys) / (q[0] * win.xs), "r3j-slope-average-cost-as-rate"),
                (n * slope * win.ys + intercept, "r3j-slope-grid-step-as-unit"),
                (n + intercept, "r3j-slope-count-added"),
            ]
            working = [
                f"Rate = {number_text(rate, 2)}, fixed cost = {plain(intercept)}.",
                f"Cost = {number_text(rate, 2)} x {n} + {plain(intercept)} = {number_text(key, 2)}.",
            ]
        try:
            question = single_number(text, key, places, working, slips, spread=100)
            return _guard_leak(question, fig)
        except AssertionError:
            continue
    raise AssertionError  # pragma: no cover


def _dollars(value):
    from ._r3j_kit import dollars

    return dollars(value)


@generator(
    "r3j_slope_context",
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3j_slope_context(rng, tier="easy"):
    if tier == "easy":
        return _easy_or_medium(rng, stepped=False)
    if tier == "medium":
        return _easy_or_medium(rng, stepped=True)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


def read_r3j_slope_context(question) -> str:
    """The answer re-derived from the drawn line and the numbers printed in the stem."""
    m, c = read_line(question.figure)
    text = question.text
    if match := re.search(r"for (\d+) more", text):
        value = int(match.group(1)) * m
    elif match := re.search(r"extra \$([\d,]+(?:\.\d+)?)", text):
        value = Fraction(match.group(1).replace(",", "")) / m
    elif match := re.search(r"cost be for (\d+)", text):
        value = int(match.group(1)) * m + c
    else:
        value = m
    return number_text(value, 0 if value.denominator == 1 else 2)


figures.register_reader("r3j_slope_context", read_r3j_slope_context)


# -- exam ---------------------------------------------------------------------------------------------------------


def _two_lines(rng):
    for _ in attempts(3000, "two plans that cross on a grid point with decimal-friendly rates"):
        ctx = rng.choice(_CONTEXTS)
        ys = rng.choice(ctx["ys"])
        xs = rng.choice(ctx["xs"])
        win = Window(0, 8, 0, 9, xs=xs, ys=ys)
        s_a = rng.choice([Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)])
        s_b = rng.choice([Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)])
        if s_a - s_b < Fraction(1, 2):
            continue
        pu, pv = rng.randint(2, 6), rng.randint(3, 8)
        c_a, c_b = pv - s_a * pu, pv - s_b * pu
        if c_a.denominator != 1 or c_b.denominator != 1 or not (0 <= c_a < c_b <= 4):
            continue
        rate_a, rate_b = s_a * ys / xs, s_b * ys / xs
        if not (_two_decimals(rate_a) and _two_decimals(rate_b)) or rate_a.denominator == 1 and rate_b.denominator == 1:
            continue
        if not (ctx["rmin"] <= rate_b < rate_a <= ctx["rmax"]):
            continue
        lines = []
        for s, c in ((s_a, c_a), (s_b, c_b)):
            pts = [pt for pt in lattice_points(win, s, c) if pt != (pu, pv) and pt[0] >= 1]
            try:
                lines.append(sorted(pick_two_points(rng, pts, min_du=2)))
            except ValueError:
                break
        else:
            return ctx, win, (s_a, c_a), (s_b, c_b), (pu, pv), lines
    raise AssertionError  # pragma: no cover


@generator("exd_x_r3j_slope_two_plans", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_slope_two_plans(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "a figure that does not print an answer"):
        ctx, win, (s_a, c_a), (s_b, c_b), (pu, pv), lines = _two_lines(rng)
        fig = _figure(win, lines, ctx, labels=["A", "B"], alt=_ALT_TWO)
        m_a, i_a = read_line(fig, 0)
        m_b, i_b = read_line(fig, 1)
        x_cross, y_cross = read_intersection(fig, 0, 1)
        check(
            (m_a, m_b) == (s_a * win.ys / win.xs, s_b * win.ys / win.xs), "figure rates disagree with the construction"
        )
        check((x_cross, y_cross) == (pu * win.xs, pv * win.ys), "figure crossing disagrees with the construction")
        diff = m_a - m_b
        answers = [number_text(m_a, 2), number_text(m_b, 2), number_text(diff, 2), str(int(x_cross))]
        try:
            figures.assert_no_leak(fig, answers)
        except AssertionError:
            continue
        break
    xs, ys = win.xs, win.ys
    text = (
        f"The graph shows the total cost, in dollars, of {ctx['activity']} from {ctx['pair']}, Plan A and Plan B, "
        f"against {ctx['measure']}."
    )
    parts = [
        number_part(
            f"How much does each extra {ctx['x_sg']} cost under Plan A, in dollars?",
            m_a,
            2,
            1,
            [
                f"Plan A rises ${plain(s_a * ys)} for each grid step of {xs} {ctx['x_pl']}: {number_text(m_a, 2)} per {ctx['x_sg']}."
            ],
            sane(
                [
                    (s_a * ys, "r3j-slope-grid-step-as-unit"),
                    (c_a * ys, "r3j-slope-fixed-cost-as-rate"),
                    (Fraction(xs) / m_a, "r3j-slope-run-over-rise"),
                    (m_b, "r3j-slope-other-line-read"),
                ],
                m_a,
                100,
            ),
            spread=100,
        ),
        number_part(
            f"How much does each extra {ctx['x_sg']} cost under Plan B, in dollars?",
            m_b,
            2,
            1,
            [
                f"Plan B rises ${plain(s_b * ys)} for each grid step of {xs} {ctx['x_pl']}: {number_text(m_b, 2)} per {ctx['x_sg']}."
            ],
            sane(
                [
                    (s_b * ys, "r3j-slope-grid-step-as-unit"),
                    (c_b * ys, "r3j-slope-fixed-cost-as-rate"),
                    (Fraction(xs) / m_b, "r3j-slope-run-over-rise"),
                    (m_a, "r3j-slope-other-line-read"),
                ],
                m_b,
                100,
            ),
            spread=100,
        ),
        number_part(
            f"How much more per {ctx['x_sg']} does Plan A charge than Plan B, in dollars?",
            diff,
            2,
            2,
            [f"{number_text(m_a, 2)} - {number_text(m_b, 2)} = {number_text(diff, 2)} dollars per {ctx['x_sg']}."],
            [
                (m_a + m_b, "r3j-slope-rates-added"),
                (m_b - m_a, "r3j-slope-difference-wrong-way"),
                ((c_b - c_a) * ys, "r3j-slope-fixed-cost-difference-given"),
                (diff * 2, "r3j-slope-difference-doubled"),
            ],
            follow_through="part_a - part_b",
        ),
        number_part(
            f"For how many {ctx['x_pl']} do the two plans cost the same?",
            x_cross,
            0,
            2,
            [
                f"The lines cross at ({int(x_cross)}, {int(y_cross)}), so the plans cost the same for {int(x_cross)} {ctx['x_pl']}."
            ],
            [
                (y_cross, "r3j-slope-cost-at-crossing-given"),
                (pu, "r3j-slope-grid-column-as-units"),
                ((c_b - c_a) * ys, "r3j-slope-fixed-cost-difference-given"),
                (x_cross + xs, "r3j-slope-neighbouring-gridline"),
            ],
        ),
    ]
    return dict(text=text, parts=parts, figure=fig)
