"""Generates the four draw-a-graph skills (easy, medium, hard) and their exam chains.

The learner draws the graph: places the points (``points``) or the joined points of a line graph (``polyline``) on a
blank grid, or types them as ``(3, 5)`` pairs. Both are graded exactly by the existing graders; the figure is a blank
grid whose canvas (`mathgen.figure_kinds.canvas`) enables the tool, snapped to whole units.

* ``dr1_line_graph`` (``draw-a-line-graph-from-continuous-data``, ACMEM042, answer ``polyline``): a quantity that
  changes continuously with time. Easy: five readings at equal time steps, plotted and joined. Medium: six or seven
  readings, with a rise and a fall. Hard: the readings are not listed; a rate for each stage is given, and the learner
  plots the points where the graph changes direction. Exam: draw the tank graph, read a volume, find when a volume was
  reached, and total the litres that left.
* ``dr1_graph_from_data`` (``draw-a-graph-from-practical-data``, ACMEM125, answer ``points``): plot a practical
  relationship. Easy: a table of values. Medium: a rule in words (a fixed charge plus a rate) to evaluate, then plot.
  Hard: a tiered rate, a quantity that runs down, or a rule found from two readings. Exam: plot three readings, then
  find the fixed charge, the rate and a prediction.
* ``dr1_scatterplot`` (``construct-a-scatterplot``, ACMGM052, answer ``points``): plot paired data. Easy: six pairs
  listed. Medium: the two variables listed separately, so the learner pairs them by position. Hard: ten pairs with an
  outlier, or only the pairs that meet a condition. Exam: plot, describe the association, predict from a stated line.
* ``dr1_time_series`` (``construct-a-time-series-plot``, ACMGM087, answer ``polyline``): every value is plotted and the
  points are joined in time order. Easy: six consecutive values. Medium: two cycles of a seasonal pattern. Hard: the
  series is given as starting value and changes. Exam: plot twelve values, find the length of the cycle, the change of
  level per cycle and a forecast.

Every key is built first as numbers and checked a second way (a line evaluated at two points, a series rebuilt from its
changes, a correlation recomputed). Every wrong reply is checked against the real grader before it is offered.
"""

from __future__ import annotations

import statistics
from math import ceil

from .. import figures
from ..core import VerificationError, attempts, check, generator, phrase
from ..figure_kinds.canvas import canvas as _canvas
from ..grading.typed import polyline_answer_text
from ._sr2a_kit import corr, mcq_part
from ._w6a_kit import num_dist, num_fields
from ._w9_graph_kit import TooFewDistractors, dist, enough, part, retry_draw

LINE = "dr1_line_graph"
DATA = "dr1_graph_from_data"
SCATTER = "dr1_scatterplot"
SERIES = "dr1_time_series"

# ---------------------------------------------------------------------------------------------- shared pieces


def _text(points) -> str:
    return " ".join(f"({x}, {y})" for x, y in points)


def _nd(correct, candidates):
    """Number distractors for a quantity that cannot be negative: a negative candidate would give the key away."""
    return num_dist(correct, [(v, tag) for v, tag in candidates if v is None or v >= 0])


def _minutes(n: int) -> str:
    return "minute" if n == 1 else "minutes"


def _unknown_tier(tier: str):
    raise ValueError(f"unknown tier {tier!r}")


def _grid_step(top: int) -> int:
    return 1 if top <= 30 else 2 if top <= 50 else 5


def _blank(alt, x_max, y_max, *, reply, tool, x_label, y_label, count):
    """A blank grid from 0 to ``x_max`` and ``y_max`` with the canvas for the reply the learner sends."""
    return figures.cartesian(
        x_range=(0, x_max),
        y_range=(0, y_max),
        alt=alt,
        grid=1,
        x_grid=1,
        y_grid=_grid_step(y_max),
        equal_scale=False,
        x_label=x_label,
        y_label=y_label,
        canvas=_canvas(reply, [tool], 1, max_points=min(count + 2, 60)),
    )


def _points_fields(points) -> dict:
    text = _text(points)
    return dict(
        answer=[[x, y] for x, y in points], answer_text=text, answer_latex="plain:" + text, answer_kind="points"
    )


def _poly_answer(vertices, keep: bool) -> dict:
    answer: dict = {"vertices": [[x, y] for x, y in vertices]}
    if keep:
        answer["keep_vertices"] = True
    return answer


def _poly_fields(vertices, keep: bool) -> dict:
    answer = _poly_answer(vertices, keep)
    text = polyline_answer_text(answer)
    return dict(answer=answer, answer_text=text, answer_latex="plain:" + text, answer_kind="polyline")


def _in_range(points, x_max, y_max) -> bool:
    return all(0 <= x <= x_max and 0 <= y <= y_max for x, y in points)


def _points_dist(fields, candidates, x_max, y_max):
    """Typed wrong replies for a points answer: ``candidates`` is ``[(points, tag)]``; off-grid sets are dropped."""
    cands = [
        (_text(pts), [[x, y] for x, y in pts], tag)
        for pts, tag in candidates
        if pts and _in_range(pts, x_max, y_max) and sorted(pts) != sorted(map(tuple, fields["answer"]))
    ]
    return dist("points", fields["answer"], fields["answer_text"], cands)


def _poly_dist(fields, candidates, keep, x_max, y_max):
    """Typed wrong replies for a polyline answer: ``candidates`` is ``[(vertices, tag)]``."""
    cands = []
    for vertices, tag in candidates:
        if len(vertices) < 2 or not _in_range(vertices, x_max, y_max):
            continue
        answer = _poly_answer(vertices, keep)
        cands.append((polyline_answer_text(answer), answer, tag))
    return dist("polyline", fields["answer"], fields["answer_text"], cands)


def _point_slips(points):
    """Wrong point sets for plotting ``points`` (a list of (x, y)) one at a time, each with its misconception."""
    pts = list(points)
    out = [([(y, x) for x, y in pts], "w9-swapped-coordinates")]
    mid = len(pts) // 2
    for step in (1, -1):
        slipped = list(pts)
        slipped[mid] = (pts[mid][0], pts[mid][1] + step)
        out.append((slipped, "w9-counts-from-the-wrong-line"))
    out.append((pts[:mid] + pts[mid + 1 :], "dr1-data-point-omitted"))
    return out


def _series_slips(vertices):
    """Wrong vertex lists for a joined line through ``vertices`` (time order)."""
    pts = list(vertices)
    out = _point_slips(pts)
    out.append(([(x + 1, y) for x, y in pts], "dr1-time-axis-offset"))
    by_height = sorted(pts, key=lambda p: (p[1], p[0]))
    if by_height != pts and by_height != pts[::-1]:
        out.append((by_height, "dr1-joined-by-height"))
    return out


def _poly_candidates(vertices):
    return _series_slips(vertices)


# ---------------------------------------------------------------------------------------------- A: line graph

_LINE_CONTEXTS = (
    dict(
        lead="The temperature in a classroom was recorded during the school day.",
        what="temperature",
        x_label="Time (hours after 8 am)",
        y_label="Temperature (degrees Celsius)",
        lo=14,
        hi=30,
    ),
    dict(
        lead="The height of the water in a rainwater tank was measured over several days.",
        what="height of the water",
        x_label="Day",
        y_label="Height (cm)",
        lo=8,
        hi=26,
    ),
    dict(
        lead="The speed of a cyclist was recorded during a training ride.",
        what="speed",
        x_label="Time (minutes)",
        y_label="Speed (km/h)",
        lo=10,
        hi=30,
    ),
    dict(
        lead="The height of the tide was measured at a jetty.",
        what="height of the tide",
        x_label="Time (hours)",
        y_label="Tide height (dm)",
        lo=4,
        hi=22,
    ),
)


def _walk(rng, n, lo, hi, turns=1):
    """``n`` whole-number readings in [lo, hi] that rise and fall at least ``turns`` times (no flat steps)."""
    for _ in attempts(500, "readings with a rise and a fall"):
        ys = [rng.randint(lo, hi)]
        for _ in range(n - 1):
            step = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
            ys.append(ys[-1] + step)
        if not all(lo <= y <= hi for y in ys):
            continue
        signs = [(b > a) - (b < a) for a, b in zip(ys, ys[1:])]
        if sum(1 for a, b in zip(signs, signs[1:]) if a != b) >= turns:
            return ys
    raise AssertionError  # pragma: no cover


def _line_easy(rng, medium: bool):
    ctx = rng.choice(_LINE_CONTEXTS)
    n = rng.choice((6, 7)) if medium else 5
    gap = rng.choice((1, 2, 5)) if medium else 1
    xs = [gap * i for i in range(n)]
    ys = _walk(rng, n, ctx["lo"], ctx["hi"], turns=2 if medium else 1)
    pts = list(zip(xs, ys))
    table = f"{ctx['x_label']}: {', '.join(map(str, xs))}. {ctx['y_label']}: {', '.join(map(str, ys))}."
    text = phrase(
        rng,
        (
            "{lead} The readings are in the table. {table} Draw a line graph of these readings.",
            "{lead} {table} Plot each reading and join the points in order.",
            "Draw a line graph for the readings below. {table} {lead}",
        ),
        lead=ctx["lead"],
        table=table,
    )
    x_max, y_max = xs[-1] + 1, ctx["hi"] + 3
    fields = _poly_fields(pts, keep=True)
    d = _poly_dist(fields, _poly_candidates(pts), True, x_max, y_max)
    enough(d, 3)
    fig = _blank(
        "A blank grid for a line graph of the readings.",
        x_max,
        y_max,
        reply="polyline",
        tool="polyline",
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        count=n,
    )
    working = [f"Plot {_text(pts)} and join the points in order from left to right."]
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


# a tank is emptied, held and refilled: each stage is (minutes, litres per minute)
_STAGE_SHAPES = (
    ("drained", "held", "refilled"),
    ("filled", "held", "drained"),
    ("filled", "drained", "held"),
    ("held", "drained", "filled"),
)
_RATE_WORD = {"drained": "drains at", "filled": "fills at", "held": "stays the same"}


def _stages(rng):
    """Three stages with a start volume. Returns (start, [(kind, minutes, rate_per_minute)])."""
    for _ in attempts(1000, "a tank scenario that fits the grid"):
        kinds = rng.choice(_STAGE_SHAPES)
        start = rng.randint(12, 30)
        stages = []
        vol = start
        ok = True
        for kind in kinds:
            minutes = rng.randint(2, 5)
            rate = 0 if kind == "held" else rng.randint(2, 5)
            vol += (rate if kind == "filled" else -rate if kind == "drained" else 0) * minutes
            stages.append((kind, minutes, rate))
            ok = ok and 3 <= vol <= 40
        if not ok:
            continue
        # no two neighbouring stages share a rate, so every breakpoint is a corner of the graph
        signed = [(rate if k == "filled" else -rate if k == "drained" else 0) for k, _, rate in stages]
        if len(set(signed)) == 3 and sum(m for _, m, _ in stages) <= 13:
            return start, stages
    raise AssertionError  # pragma: no cover


def _breakpoints(start, stages):
    pts = [(0, start)]
    t, v = 0, start
    for kind, minutes, rate in stages:
        t += minutes
        v += (rate if kind == "filled" else -rate if kind == "drained" else 0) * minutes
        pts.append((t, v))
    return pts


def _tank_text(start, stages):
    parts = []
    for kind, minutes, rate in stages:
        if kind == "held":
            parts.append(f"for {minutes} minutes the volume stays the same")
        else:
            parts.append(f"for {minutes} minutes the tank {_RATE_WORD[kind]} {rate} litres per minute")
    return f"A tank holds {start} litres at time 0. First {parts[0]}, then {parts[1]}, and finally {parts[2]}."


def _tank_slips(start, stages, pts):
    out = []
    # the rises and falls swapped: the volume goes the other way from the start
    flipped = [("filled" if k == "drained" else "drained" if k == "filled" else k, m, r) for k, m, r in stages]
    out.append((_breakpoints(start, flipped), "dr1-direction-of-change-reversed"))
    # the flat stage left out: its end point is not plotted
    held = [i for i, (k, _, _) in enumerate(stages) if k == "held"][0] + 1
    out.append((pts[:held] + pts[held + 1 :], "dr1-flat-stage-missed"))
    # the graph started from zero, not from the stated volume
    out.append(([(x, y - start) for x, y in pts], "dr1-start-value-ignored"))
    out.append(([(y, x) for x, y in pts], "w9-swapped-coordinates"))
    mid = pts[1:-1][0]
    out.append(([pts[0], (mid[0], mid[1] + 1), *pts[2:]], "w9-counts-from-the-wrong-line"))
    return out


def _line_hard(rng):
    start, stages = _stages(rng)
    pts = _breakpoints(start, stages)
    check(all(y >= 0 for _, y in pts), "a volume went negative")
    x_max, y_max = pts[-1][0] + 1, max(y for _, y in pts) + 3
    fields = _poly_fields(pts, keep=False)
    d = _poly_dist(fields, _tank_slips(start, stages, pts), False, x_max, y_max)
    enough(d, 3)
    story = _tank_text(start, stages)
    text = phrase(
        rng,
        (
            "{story} Draw the line graph of the volume against time.",
            "{story} Plot the points where the graph changes direction and join them to show the volume over time.",
            "Draw a line graph of the volume of water in a tank against time. {story}",
        ),
        story=story,
    )
    fig = _blank(
        "A blank grid for a line graph of the volume of water against time.",
        x_max,
        y_max,
        reply="polyline",
        tool="polyline",
        x_label="Time (minutes)",
        y_label="Volume (litres)",
        count=len(pts),
    )
    working = [f"The graph changes direction at {_text(pts)}; join them in order."]
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


@generator(
    LINE,
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def dr1_line_graph(rng, tier="easy"):
    """Draw a line graph of continuous change; the tiers are described in the module docstring."""
    build = {
        "easy": lambda r: _line_easy(r, False),
        "medium": lambda r: _line_easy(r, True),
        "hard": _line_hard,
    }.get(tier)
    if build is None:
        _unknown_tier(tier)
    return retry_draw(rng, build)


@generator("exd_x_" + LINE, level=11, presets={"exam": dict(tier="exam")})
def exd_x_dr1_line_graph(rng, tier="exam"):
    """Exam chain: draw the tank graph; read a volume; find when a volume was reached; total the litres drained."""
    if tier != "exam":
        _unknown_tier(tier)
    return retry_draw(rng, _line_exam)


def _volume_at(pts, t):
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    raise ValueError(t)


def _line_exam(rng):
    for _ in attempts(500, "a tank scenario with a whole-minute reading and target"):
        start, stages = _stages(rng)
        pts = _breakpoints(start, stages)
        moving = [(i, s) for i, s in enumerate(stages) if s[0] != "held"]
        i, (kind, minutes, rate) = rng.choice(moving)
        t0, v0 = pts[i]
        if minutes < 3:
            continue
        t = t0 + rng.randint(1, minutes - 1)
        v = _volume_at(pts, t)
        reached = rng.choice([s for s in range(1, minutes) if s != t - t0] or [1])
        target = _volume_at(pts, t0 + reached)
        if v != int(v) or target != int(target) or v == target:
            continue
        # the target volume must be met at one time only
        times = [x for x in range(pts[-1][0] + 1) if _volume_at(pts, x) == target]
        if len(times) != 1:
            continue
        break
    v, target, when = int(v), int(target), times[0]
    drained = sum(rate * minutes for kind, minutes, rate in stages if kind == "drained")
    check(drained == sum(max(0, a - b) for (_, a), (_, b) in zip(pts, pts[1:])), "drained litres differ")
    x_max, y_max = pts[-1][0] + 1, max(y for _, y in pts) + 3
    fa = _poly_fields(pts, keep=False)
    da = _poly_dist(fa, _tank_slips(start, stages, pts), False, x_max, y_max)
    enough(da, 3)
    sign = 1 if kind == "filled" else -1
    db = _nd(
        v,
        [
            (start + sign * -1 * rate * (t - t0), "dr1-direction-of-change-reversed"),
            (v0 + sign * rate * (t - t0) + sign * rate, "dr1-off-by-one"),
            (v0, "dr1-stage-start-read"),
            (t, "dr1-wrong-axis-read"),
            (v + 2, "dr1-off-by-one"),
        ],
    )
    dc = _nd(
        when,
        [
            (target, "dr1-wrong-axis-read"),
            (when + 1, "dr1-off-by-one"),
            (max(when - 1, 0), "dr1-off-by-one"),
            (abs(target - v0), "dr1-wrong-axis-read"),
        ],
    )
    dd = _nd(
        drained,
        [
            (drained + sum(rate * m for k, m, rate in stages if k == "filled"), "dr1-all-changes-added"),
            (max(rate for k, _, rate in stages if k == "drained") if drained else 1, "dr1-rate-for-total"),
            (drained + 2, "dr1-off-by-one"),
            (abs(pts[-1][1] - start) + 1, "dr1-net-change-for-total"),
        ],
    )
    for dl in (db, dc, dd):
        enough(dl)
    story = _tank_text(start, stages)
    fig = _blank(
        "A blank grid for a line graph of the volume of water against time.",
        x_max,
        y_max,
        reply="polyline",
        tool="polyline",
        x_label="Time (minutes)",
        y_label="Volume (litres)",
        count=len(pts),
    )
    return dict(
        text=story,
        figure=fig,
        parts=[
            part(
                "Draw the line graph of the volume against time. Plot the points where the graph changes direction.",
                "polyline",
                {k: fa[k] for k in ("answer", "answer_text", "answer_latex")},
                3,
                [f"The graph changes direction at {_text(pts)}."],
                da,
            ),
            part(
                f"Use your graph to find the volume, in litres, at {t} {_minutes(t)}.",
                "number",
                num_fields(v),
                1,
                [
                    f"Between {t0} and {t0 + minutes} minutes the volume changes by {sign * rate} litres each minute: "
                    f"{v0} {'+' if sign > 0 else '-'} {rate} x {t - t0} = {v}."
                ],
                db,
                sensible_range=(0, None),
            ),
            part(
                f"At what time, in minutes, was the volume first {target} litres?",
                "number",
                num_fields(when),
                2,
                [f"The volume is {target} litres at {when} {_minutes(when)}."],
                dc,
                sensible_range=(0, None),
            ),
            part(
                "How many litres in total flowed out of the tank?",
                "number",
                num_fields(drained),
                2,
                [
                    f"Only the draining stages count: {' + '.join(f'{rate} x {m}' for k, m, rate in stages if k == 'drained')}"
                    f" = {drained} litres."
                ],
                dd,
                sensible_range=(0, None),
            ),
        ],
    )


# ---------------------------------------------------------------------------------------------- B: practical data

_RULES = (
    dict(
        item="a taxi trip",
        fixed_word="flagfall",
        fixed_unit="$",
        x_name="distance",
        x_unit="kilometres",
        x_one="kilometre",
        y_label="Fare ($)",
        x_label="Distance (km)",
        fixed=(2, 3, 4, 5, 6, 7, 8),
        rate=(2, 3, 4, 5),
        story="A taxi charges a flagfall of ${fixed} plus ${rate} for each kilometre.",
    ),
    dict(
        item="hiring a kayak",
        fixed_word="booking fee",
        fixed_unit="$",
        x_name="time",
        x_unit="hours",
        x_one="hour",
        y_label="Cost ($)",
        x_label="Time (hours)",
        fixed=(3, 4, 5, 6, 7, 8, 9),
        rate=(2, 3, 4, 5),
        story="A kayak hire shop charges a booking fee of ${fixed} plus ${rate} for each hour.",
    ),
    dict(
        item="a phone plan",
        fixed_word="monthly fee",
        fixed_unit="$",
        x_name="data",
        x_unit="gigabytes",
        x_one="gigabyte",
        y_label="Bill ($)",
        x_label="Data used (GB)",
        fixed=(5, 6, 7, 8, 9, 10),
        rate=(2, 3, 4, 5),
        story="A phone plan has a monthly fee of ${fixed} plus ${rate} for each gigabyte of data used.",
    ),
    dict(
        item="a plumber's visit",
        fixed_word="call-out fee",
        fixed_unit="$",
        x_name="time",
        x_unit="hours",
        x_one="hour",
        y_label="Charge ($)",
        x_label="Time (hours)",
        fixed=(5, 6, 8, 9, 10),
        rate=(2, 3, 4, 5),
        story="A plumber charges a call-out fee of ${fixed} plus ${rate} for each hour of work.",
    ),
    dict(
        item="printing posters",
        fixed_word="set-up fee",
        fixed_unit="$",
        x_name="posters",
        x_unit="posters",
        x_one="poster",
        y_label="Cost ($)",
        x_label="Posters printed",
        fixed=(3, 4, 6, 7, 9),
        rate=(2, 3, 4, 5),
        story="A print shop charges a set-up fee of ${fixed} plus ${rate} for each poster printed.",
    ),
    dict(
        item="a gym visit pass",
        fixed_word="entry fee",
        fixed_unit="$",
        x_name="visits",
        x_unit="visits",
        x_one="visit",
        y_label="Total cost ($)",
        x_label="Visits",
        fixed=(4, 5, 7, 8, 10),
        rate=(2, 3, 4, 5),
        story="A gym charges a joining fee of ${fixed} plus ${rate} for each visit.",
    ),
)


def _cnt(n, rule) -> str:
    """``1 hour``, ``3 hours``."""
    return f"{n} {rule['x_one'] if n == 1 else rule['x_unit']}"


def _rule_draw(rng, count_lo=4, count_hi=6):
    for _ in attempts(200, "a rule whose largest cost fits the grid"):
        rule = rng.choice(_RULES)
        fixed, rate = rng.choice(rule["fixed"]), rng.choice(rule["rate"])
        n = rng.randint(count_lo, count_hi)
        if fixed + rate * n <= 30:
            return rule, fixed, rate, n
    raise AssertionError  # pragma: no cover


def _data_slips(pts, fixed, rate, xs):
    out = _point_slips(pts)
    out.append(([(x, rate * x) for x in xs], "dr1-fixed-charge-omitted"))
    out.append(([(x, fixed * x + rate) for x in xs], "dr1-rate-and-fixed-swapped"))
    mid = len(pts) // 2
    swapped = list(pts)
    swapped[0], swapped[mid] = (swapped[0][0], swapped[mid][1]), (swapped[mid][0], swapped[0][1])
    out.append((swapped, "dr1-pairs-mismatched"))
    return out


def _data_question(rng, text, pts, fixed, rate, xs, alt, rule, working):
    x_max, y_max = max(xs) + 1, max(y for _, y in pts) + 3
    y_max = max(y_max, 12)
    fields = _points_fields(pts)
    d = _points_dist(fields, _data_slips(pts, fixed, rate, xs), x_max, y_max)
    enough(d, 3)
    fig = _blank(
        alt,
        x_max,
        y_max,
        reply="points",
        tool="plot_points",
        x_label=rule["x_label"],
        y_label=rule["y_label"],
        count=len(pts),
    )
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


def _data_easy(rng):
    rule, fixed, rate, n = _rule_draw(rng)
    gap = rng.choice((1, 1, 2))
    xs = [gap * i for i in range(n + 1)] if fixed + rate * gap * n <= 30 else list(range(n + 1))
    pts = [(x, fixed + rate * x) for x in xs]
    check(len({y2 - y1 for (_, y1), (_, y2) in zip(pts, pts[1:])}) == 1, "the table is not linear")
    x_row = ", ".join(str(x) for x in xs)
    y_row = ", ".join(str(y) for _, y in pts)
    text = phrase(
        rng,
        (
            "The table gives the cost of {item}. {x_label}: {x_row}. {y_label}: {y_row}. Draw a graph of the cost.",
            "A graph is needed for {item}. {x_label}: {x_row}. {y_label}: {y_row}. Plot the points.",
            "Plot the points of this table for {item} on a graph. {x_label}: {x_row}. {y_label}: {y_row}.",
        ),
        item=rule["item"],
        x_label=rule["x_label"],
        y_label=rule["y_label"],
        x_row=x_row,
        y_row=y_row,
    )
    return _data_question(
        rng,
        text,
        pts,
        fixed,
        rate,
        xs,
        f"A blank grid for plotting the cost of {rule['item']}.",
        rule,
        [f"Plot {_text(pts)}."],
    )


def _data_medium(rng):
    rule, fixed, rate, n = _rule_draw(rng, 3, 5)
    xs = list(range(n + 1))
    pts = [(x, fixed + rate * x) for x in xs]
    story = rule["story"].format(fixed=fixed, rate=rate)
    xs_text = ", ".join(str(x) for x in xs[:-1]) + f" and {xs[-1]}"
    text = phrase(
        rng,
        (
            "{story} Plot the cost for {xs_text} {x_unit} on a graph.",
            "{story} Work out the cost for {xs_text} {x_unit}, and plot these points.",
            "Draw a graph of the cost of {item}. {story} Use {xs_text} {x_unit}.",
        ),
        story=story,
        item=rule["item"],
        x_unit=rule["x_unit"],
        xs_text=xs_text,
    )
    working = [f"Cost = {fixed} + {rate} x {rule['x_name']}: " + ", ".join(f"{x} gives {y}" for x, y in pts) + "."]
    return _data_question(
        rng, text, pts, fixed, rate, xs, f"A blank grid for plotting the cost of {rule['item']}.", rule, working
    )


def _data_hard_tiered(rng):
    for _ in attempts(200, "a tiered rule that fits the grid"):
        rule = rng.choice(_RULES)
        fixed = rng.choice((2, 3, 4))
        cap, r1, r2 = rng.randint(2, 3), rng.choice((2, 3)), rng.choice((4, 5, 6))
        n = rng.randint(5, 6)
        xs = list(range(n + 1))
        pts = [(x, fixed + r1 * min(x, cap) + r2 * max(x - cap, 0)) for x in xs]
        if max(y for _, y in pts) <= 30:
            break
    story = (
        f"The cost of {rule['item']} is a fixed ${fixed}, plus ${r1} for each {rule['x_one']} up to {_cnt(cap, rule)}, "
        f"and ${r2} for each {rule['x_one']} after that."
    )
    text = f"{story} Plot the cost for 0 to {n} {rule['x_unit']}."
    working = [
        f"Up to {cap}: {fixed} + {r1}x. After {cap}: {fixed + r1 * cap} + {r2}(x - {cap}). Points: {_text(pts)}."
    ]
    return _data_question(
        rng, text, pts, fixed, r1, xs, f"A blank grid for plotting the cost of {rule['item']}.", rule, working
    )


def _data_hard_down(rng):
    value = rng.choice((24, 28, 30, 32, 36))
    rate = rng.choice((3, 4, 6, 4, 2))
    days = value // rate
    n = min(days, 7)
    xs = list(range(n + 1))
    pts = [(x, value - rate * x) for x in xs]
    rule = dict(x_label="Day", y_label="Balance ($)", x_name="day", x_unit="days", item="a lunch card")
    text = (
        f"A lunch card starts with ${value} and ${rate} is spent each school day. "
        f"Plot the balance at the start of day 0 to day {n}."
    )
    working = [f"Balance = {value} - {rate} x day. Points: {_text(pts)}."]
    return _data_question(
        rng, text, pts, value, -rate, xs, "A blank grid for plotting the balance of a lunch card.", rule, working
    )


def _data_hard_two(rng):
    rule, fixed, rate, n = _rule_draw(rng, 5, 6)
    a, b = sorted(rng.sample(range(1, 6), 2))
    pts_given = [(a, fixed + rate * a), (b, fixed + rate * b)]
    xs = list(range(n + 1))
    pts = [(x, fixed + rate * x) for x in xs]
    text = (
        f"The cost of {rule['item']} changes by the same amount for each {rule['x_one']}. It costs ${pts_given[0][1]} "
        f"for {_cnt(a, rule)} and ${pts_given[1][1]} for {_cnt(b, rule)}. "
        f"Plot the cost for 0 to {n} {rule['x_unit']}."
    )
    working = [
        f"The cost rises ({pts_given[1][1]} - {pts_given[0][1]}) / ({b} - {a}) = {rate} for each {rule['x_one']}, "
        f"so the cost at 0 is {fixed}. Points: {_text(pts)}."
    ]
    return _data_question(
        rng, text, pts, fixed, rate, xs, f"A blank grid for plotting the cost of {rule['item']}.", rule, working
    )


@generator(
    DATA,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def dr1_graph_from_data(rng, tier="easy"):
    """Plot a practical relationship; the tiers are described in the module docstring."""
    pool = {
        "easy": (_data_easy,),
        "medium": (_data_medium,),
        "hard": (_data_hard_tiered, _data_hard_down, _data_hard_two),
    }.get(tier)
    if pool is None:
        _unknown_tier(tier)
    return retry_draw(rng, rng.choice(pool))


@generator("exd_x_" + DATA, level=12, presets={"exam": dict(tier="exam")})
def exd_x_dr1_graph_from_data(rng, tier="exam"):
    """Exam chain: plot three readings; find the fixed charge, the rate and a prediction."""
    if tier != "exam":
        _unknown_tier(tier)
    return retry_draw(rng, _data_exam)


def _data_exam(rng):
    rule, fixed, rate, _ = _rule_draw(rng, 5, 5)
    xs = sorted(rng.sample(range(1, 6), 3))
    pts = [(x, fixed + rate * x) for x in xs]
    target = rng.choice([x for x in range(7, 11) if fixed + rate * x <= 40] or [7])
    check(
        len({y2 - y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:]) if x2 - x1 == 1}) <= 1, "the readings are not linear"
    )
    answer = fixed + rate * target
    x_max, y_max = max(xs) + 1, max(y for _, y in pts) + 3
    fa = _points_fields(pts)
    da = _points_dist(fa, _data_slips(pts, fixed, rate, [p[0] for p in pts]), x_max, y_max)
    enough(da, 3)
    gap = xs[1] - xs[0]
    rise = pts[1][1] - pts[0][1]
    dfix = _nd(
        fixed,
        [
            (rate, "dr1-rate-and-fixed-swapped"),
            (pts[0][1], "dr1-first-reading-for-intercept"),
            (fixed + rate, "dr1-first-reading-for-intercept"),
            (max(fixed - 1, 0), "dr1-off-by-one"),
            (fixed + 2, "dr1-off-by-one"),
        ],
    )
    drate = _nd(
        rate,
        [
            (fixed, "dr1-rate-and-fixed-swapped"),
            (rise, "dr1-total-change-for-rate") if gap != 1 else (rate + 1, "dr1-off-by-one"),
            (pts[0][1], "dr1-first-reading-for-intercept"),
            (rate + 1, "dr1-off-by-one"),
            (rate * 2, "dr1-total-change-for-rate"),
        ],
    )
    dpred = _nd(
        answer,
        [
            (rate * target, "dr1-fixed-charge-omitted"),
            (answer + rate, "dr1-off-by-one"),
            (answer - rate, "dr1-off-by-one"),
            (pts[-1][1] + rate * target, "dr1-last-reading-restarted"),
            (fixed * target + rate, "dr1-rate-and-fixed-swapped"),
        ],
    )
    for dl in (dfix, drate, dpred):
        enough(dl)
    readings = ", ".join(f"{_cnt(x, rule)} cost ${y}" for x, y in pts)
    text = (
        f"The cost of {rule['item']} is a fixed charge plus the same amount for each {rule['x_one']}. "
        f"Three readings: {readings}."
    )
    fig = _blank(
        f"A blank grid for plotting the cost of {rule['item']}.",
        x_max,
        y_max,
        reply="points",
        tool="plot_points",
        x_label=rule["x_label"],
        y_label=rule["y_label"],
        count=len(pts),
    )
    return dict(
        text=text,
        figure=fig,
        parts=[
            part(
                "Plot the three readings on the graph.",
                "points",
                {k: fa[k] for k in ("answer", "answer_text", "answer_latex")},
                2,
                [f"Plot {_text(pts)}."],
                da,
            ),
            part(
                "Find the fixed charge, in dollars (the cost at 0).",
                "number",
                num_fields(fixed),
                2,
                [
                    f"The cost rises by {rate} for each {rule['x_one']}, so the cost at 0 is {pts[0][1]} - {rate} x {xs[0]} = {fixed}."
                ],
                dfix,
                sensible_range=(0, None),
            ),
            part(
                f"Find the cost, in dollars, for each extra {rule['x_one']}.",
                "number",
                num_fields(rate),
                1,
                [f"({pts[1][1]} - {pts[0][1]}) / ({xs[1]} - {xs[0]}) = {rate}."],
                drate,
                sensible_range=(0, None),
            ),
            part(
                f"Use the pattern to find the cost, in dollars, of {_cnt(target, rule)}.",
                "number",
                num_fields(answer),
                2,
                [f"{fixed} + {rate} x {target} = {answer}."],
                dpred,
                sensible_range=(0, None),
            ),
        ],
    )


# ---------------------------------------------------------------------------------------------- C: scatterplot

_PAIRS = (
    dict(
        x="hours of study",
        y="test mark",
        x_label="Study time (hours)",
        y_label="Test mark",
        unit_x="hours",
        pop="students",
        hi=20,
    ),
    dict(
        x="hours of sunshine",
        y="solar power",
        x_label="Sunshine (hours)",
        y_label="Power (kWh)",
        unit_x="hours",
        pop="days",
        hi=20,
    ),
    dict(
        x="staff on shift",
        y="customers served per hour",
        x_label="Staff on shift",
        y_label="Customers per hour",
        unit_x="staff",
        pop="shifts",
        hi=20,
    ),
    dict(
        x="age of a car",
        y="price",
        x_label="Age (years)",
        y_label="Price ($ thousand)",
        unit_x="years",
        pop="cars",
        hi=20,
        falling=True,
    ),
    dict(
        x="weeks of training",
        y="time for a 3 km run",
        x_label="Training (weeks)",
        y_label="Run time (minutes)",
        unit_x="weeks",
        pop="runners",
        hi=20,
        falling=True,
    ),
)


def _scatter_data(rng, n, *, strong=True, falling=None, ctx=None, outlier=False, band=None):
    """``n`` (x, y) pairs with distinct x. Returns (ctx, pairs, slope, intercept, r)."""
    ctx = ctx or rng.choice(_PAIRS)
    band = band or ("strong" if strong else "weak")
    if falling is None:
        falling = bool(ctx.get("falling"))
    for _ in attempts(2000, "paired data with a clear association"):
        xs = sorted(rng.sample(range(1, 13), n))
        slope = rng.choice((-1, -1, -2) if falling else (1, 2))
        top = ctx["hi"] if strong else ctx["hi"]
        intercept = rng.randint(1, 4) if slope > 0 else rng.randint(top - 4, top - 1)
        noise = {"strong": 1, "moderate": 4, "weak": 6}[band]
        ys = [intercept + slope * x + rng.randint(-noise, noise) for x in xs]
        if outlier:
            k = rng.randrange(1, n - 1)
            ys[k] = ys[k] + rng.choice((-1, 1)) * rng.randint(5, 7)
        if not all(1 <= y <= top + (9 if outlier else 4) for y in ys) or len(set(ys)) < max(4, n - 3):
            continue
        r = corr(xs, ys)
        if band == "strong" and outlier and abs(r) >= 0.5:
            return ctx, list(zip(xs, ys)), slope, intercept, r
        lo, hi = {"strong": (0.82, 1), "moderate": (0.55, 0.7), "weak": (0.15, 0.4)}[band]
        if not outlier and lo <= abs(r) <= hi:
            ls = corr(xs, ys) * statistics.pstdev(ys) / statistics.pstdev(xs)
            if abs(ls - slope) <= 0.5:  # the line the stem quotes is a fair best fit
                return ctx, list(zip(xs, ys)), slope, intercept, r
    raise AssertionError  # pragma: no cover


def _scatter_slips(pts):
    out = _point_slips(pts)
    ys = [p[1] for p in pts]
    shifted = [(x, y) for (x, _), y in zip(pts, ys[1:] + ys[:1])]
    out.append((shifted, "dr1-pairs-mismatched"))
    return out


def _scatter_question(rng, ctx, pts, text, working, alt):
    x_max, y_max = max(x for x, _ in pts) + 2, max(y for _, y in pts) + 3
    y_max = max(y_max, 10)
    fields = _points_fields(pts)
    d = _points_dist(fields, _scatter_slips(pts), x_max, y_max)
    enough(d, 3)
    fig = _blank(
        alt,
        x_max,
        y_max,
        reply="points",
        tool="plot_points",
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        count=len(pts),
    )
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


def _pair_list(pts):
    return ", ".join(f"({x}, {y})" for x, y in pts)


def _scatter_easy(rng):
    ctx, pts, *_ = _scatter_data(rng, 6)
    text = phrase(
        rng,
        (
            "Each pair gives the {xv} and the {yv} of one of {n} {pop}: {pairs}. Draw a scatterplot.",
            "Draw a scatterplot of {yv} against {xv} for these {n} {pop}: {pairs}.",
            "A scatterplot is to be drawn from the pairs ({xv}, {yv}) below. Plot {pairs}.",
        ),
        pop=ctx["pop"],
        n=len(pts),
        xv=ctx["x"],
        yv=ctx["y"],
        pairs=_pair_list(pts),
    )
    return _scatter_question(
        rng,
        ctx,
        pts,
        text,
        [f"Plot each pair as a point: {_text(pts)}."],
        f"A blank grid for a scatterplot of {ctx['y']} against {ctx['x']}.",
    )


def _scatter_medium(rng):
    ctx, pts, *_ = _scatter_data(rng, 8)
    xs = ", ".join(str(x) for x, _ in pts)
    ys = ", ".join(str(y) for _, y in pts)
    text = phrase(
        rng,
        (
            "Eight {pop} were measured. {xl}: {xs}. {yl}: {ys}. Each position in the two lists is one of the {pop}. Draw a scatterplot of {yv} against {xv}.",
            "{xl}: {xs}. {yl}: {ys}. Each position gives one of the eight {pop}. Plot the eight points on a scatterplot.",
            "Draw a scatterplot of {yv} (vertical) against {xv} (horizontal) for eight {pop}, using the lists in the same order. {xl}: {xs}. {yl}: {ys}.",
        ),
        pop=ctx["pop"],
        xv=ctx["x"],
        yv=ctx["y"],
        xl=ctx["x_label"],
        yl=ctx["y_label"],
        xs=xs,
        ys=ys,
    )
    return _scatter_question(
        rng,
        ctx,
        pts,
        text,
        [f"Pair the lists by position: {_text(pts)}."],
        f"A blank grid for a scatterplot of {ctx['y']} against {ctx['x']}.",
    )


def _scatter_hard_outlier(rng):
    ctx, pts, *_ = _scatter_data(rng, 10, outlier=True)
    text = (
        f"The pairs ({ctx['x']}, {ctx['y']}) for ten {ctx['pop']} are {_pair_list(pts)}. "
        "Draw a scatterplot including every point."
    )
    return _scatter_question(
        rng,
        ctx,
        pts,
        text,
        [f"Plot all ten pairs: {_text(pts)}."],
        f"A blank grid for a scatterplot of {ctx['y']} against {ctx['x']}.",
    )


def _scatter_hard_subset(rng):
    ctx, pts, *_ = _scatter_data(rng, 10)
    cut = pts[3][0]
    keep = [p for p in pts if p[0] > cut]
    text = (
        f"The pairs ({ctx['x']}, {ctx['y']}) for ten {ctx['pop']} are {_pair_list(pts)}. "
        f"Draw a scatterplot of only the pairs with x greater than {cut}, where x is the {ctx['x']}."
    )
    working = [f"Keep the pairs with x greater than {cut}: {_text(keep)}."]
    q = _scatter_question(
        rng, ctx, keep, text, working, f"A blank grid for a scatterplot of {ctx['y']} against {ctx['x']}."
    )
    # the slip of plotting every pair is a likely reply too
    return q


@generator(
    SCATTER,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def dr1_scatterplot(rng, tier="easy"):
    """Construct a scatterplot; the tiers are described in the module docstring."""
    pool = {
        "easy": (_scatter_easy,),
        "medium": (_scatter_medium,),
        "hard": (_scatter_hard_outlier, _scatter_hard_subset),
    }.get(tier)
    if pool is None:
        _unknown_tier(tier)
    return retry_draw(rng, rng.choice(pool))


@generator("exd_x_" + SCATTER, level=12, presets={"exam": dict(tier="exam")})
def exd_x_dr1_scatterplot(rng, tier="exam"):
    """Exam chain: plot eight pairs; describe the association; predict from a stated line; compare with a reading."""
    if tier != "exam":
        _unknown_tier(tier)
    return retry_draw(rng, _scatter_exam)


def _scatter_exam(rng):
    band = rng.choice(("strong", "moderate", "weak"))
    try:
        ctx, pts, slope, intercept, r = _scatter_data(rng, 8, band=band)
    except VerificationError:  # some context and band pairs have no data that fits the stated line: draw again
        raise TooFewDistractors("no data for this context and band") from None
    direction = "positive" if slope > 0 else "negative"
    other = "negative" if slope > 0 else "positive"
    x0 = rng.choice([x for x in range(2, 12) if x not in {p[0] for p in pts}] or [13])
    predicted = intercept + slope * x0
    if not 1 <= predicted <= ctx["hi"] + 4:
        raise TooFewDistractors("the prediction is off the grid")
    key = f"{band} {direction}"
    band2 = rng.choice([b for b in ("strong", "moderate", "weak") if b != band])
    mcq = mcq_part(
        "Describe the association shown by the scatterplot.",
        key,
        1,
        [f"The points {'rise' if slope > 0 else 'fall'} and r = {r:.2f}, which is {band}."],
        [
            (f"{band} {other}", "dr1-association-direction-reversed"),
            (f"{band2} {direction}", "dr1-strength-one-band-off"),
            (f"{band2} {other}", "dr1-strength-one-band-off"),
        ],
    )
    if mcq is None:
        raise TooFewDistractors("too few options")
    x_max, y_max = max(x for x, _ in pts) + 2, max(y for _, y in pts) + 3
    fa = _points_fields(pts)
    da = _points_dist(fa, _scatter_slips(pts), x_max, y_max)
    enough(da, 3)
    dp = _nd(
        predicted,
        [
            (intercept + slope * (x0 + 1), "dr1-off-by-one"),
            (slope * x0, "dr1-fixed-charge-omitted"),
            (intercept + x0, "dr1-rate-and-fixed-swapped"),
            (intercept - slope * x0, "dr1-direction-of-change-reversed"),
            (predicted + 2, "dr1-off-by-one"),
        ],
    )
    enough(dp)
    xt = "x" if abs(slope) == 1 else f"{abs(slope)}x"
    line = f"y = {xt} + {intercept}" if slope > 0 else f"y = {intercept} - {xt}"
    text = f"Eight {ctx['pop']} were measured. The pairs ({ctx['x']}, {ctx['y']}) are {_pair_list(pts)}."
    fig = _blank(
        f"A blank grid for a scatterplot of {ctx['y']} against {ctx['x']}.",
        x_max,
        y_max,
        reply="points",
        tool="plot_points",
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        count=len(pts),
    )
    return dict(
        text=text,
        figure=fig,
        parts=[
            part(
                "Draw a scatterplot of the eight pairs.",
                "points",
                {k: fa[k] for k in ("answer", "answer_text", "answer_latex")},
                3,
                [f"Plot {_text(pts)}."],
                da,
            ),
            mcq,
            part(
                f"A line of best fit for the data is {line}. Use it to predict {ctx['y']} when x = {x0}, where x is the {ctx['x']}.",
                "number",
                num_fields(predicted),
                2,
                [f"{line} at x = {x0} gives y = {predicted}."],
                dp,
                sensible_range=(0, None),
            ),
        ],
    )


# ---------------------------------------------------------------------------------------------- D: time series

_SERIES = (
    dict(
        intro="The number of pies sold at a school canteen was recorded each day.",
        x_word="Day",
        x_label="Day",
        y_label="Pies sold",
        p=7,
        cycle="week",
    ),
    dict(
        intro="The number of library books borrowed by a class was recorded each school day.",
        x_word="Day",
        x_label="Day",
        y_label="Books borrowed",
        p=5,
        cycle="week",
    ),
    dict(
        intro="The sales of a surf shop, in thousands of dollars, were recorded each quarter.",
        x_word="Quarter",
        x_label="Quarter",
        y_label="Sales ($ thousand)",
        p=4,
        cycle="year",
    ),
    dict(
        intro="The number of visitors to a museum, in hundreds, was recorded each month.",
        x_word="Month",
        x_label="Month",
        y_label="Visitors (hundreds)",
        p=6,
        cycle="half year",
    ),
)


def _seasonal(rng, p, cycles, delta_pool=(-3, -2, 2, 3, 4)):
    """(values, offsets, delta): value = base + delta per cycle + offsets[season], whole numbers in 2..22."""
    for _ in attempts(2000, "a seasonal series that fits the grid"):
        offsets = rng.sample(range(0, p + 4), p)
        if max(offsets) - min(offsets) < 4:
            continue
        delta = rng.choice(delta_pool)
        base = rng.randint(2, 6) + (abs(delta) * (cycles - 1) if delta < 0 else 0)
        ys = [base + delta * (t // p) + offsets[t % p] for t in range(p * cycles)]
        if min(ys) >= 2 and max(ys) <= 22:
            return ys, offsets, delta
    raise AssertionError  # pragma: no cover


def _series_question(rng, ctx, ys, text, working, alt):
    n = len(ys)
    pts = list(zip(range(1, n + 1), ys))
    x_max, y_max = n + 1, max(ys) + 3
    fields = _poly_fields(pts, keep=True)
    d = _poly_dist(fields, _series_slips(pts), True, x_max, y_max)
    enough(d, 3)
    fig = _blank(
        alt, x_max, y_max, reply="polyline", tool="polyline", x_label=ctx["x_label"], y_label=ctx["y_label"], count=n
    )
    return dict(text=text, figure=fig, working=working, distractors=d, **fields)


def _series_easy(rng):
    ctx = rng.choice(_SERIES)
    ys = _walk(rng, 6, 3, 20, turns=1)
    listed = ", ".join(f"{ctx['x_word'].lower()} {i}: {y}" for i, y in enumerate(ys, 1))
    text = phrase(
        rng,
        (
            "{intro} The values were {listed}. Draw a time series plot of these values.",
            "Draw a time series plot of the values below. {intro} {listed}.",
            "{intro} Plot these values in time order and join the points: {listed}.",
        ),
        intro=ctx["intro"],
        listed=listed,
    )
    pts = list(zip(range(1, 7), ys))
    return _series_question(
        rng, ctx, ys, text, [f"Plot {_text(pts)} and join them in time order."], "A blank grid for a time series plot."
    )


def _series_medium(rng):
    ctx = rng.choice([c for c in _SERIES if c["p"] in (4, 6)])
    p = ctx["p"]
    ys, offsets, delta = _seasonal(rng, p, 2)
    listed = ", ".join(str(y) for y in ys)
    text = phrase(
        rng,
        (
            "{intro} The {n} values were {listed}. Draw a time series plot, with time on the horizontal axis starting at 1.",
            "Draw a time series plot of these {n} values: {listed}. {intro}",
            "{intro} Plot the series in order, starting at time 1, and join the points: {listed}.",
        ),
        intro=ctx["intro"],
        n=2 * p,
        listed=listed,
    )
    pts = list(zip(range(1, 2 * p + 1), ys))
    return _series_question(
        rng, ctx, ys, text, [f"Plot {_text(pts)} and join them in time order."], "A blank grid for a time series plot."
    )


def _series_hard(rng):
    ctx = rng.choice(_SERIES)
    n = rng.choice((7, 8))
    for _ in attempts(500, "changes that stay on the grid"):
        start = rng.randint(6, 16)
        changes = [rng.choice((-4, -3, -2, -1, 1, 2, 3, 4)) for _ in range(n - 1)]
        ys = [start]
        for c in changes:
            ys.append(ys[-1] + c)
        if min(ys) >= 2 and max(ys) <= 22 and len(set(ys)) >= 5:
            break
    check(
        ys[0] == start and all(b - a == c for a, b, c in zip(ys, ys[1:], changes)),
        "the series does not match its changes",
    )
    ch_text = ", ".join(f"{c:+d}" for c in changes)
    text = phrase(
        rng,
        (
            "{intro} The first value was {start}. The change from one value to the next was {ch_text}. Draw a time series plot of the {n} values, starting at time 1.",
            "{intro} A series starts at {start} at time 1. The changes from one value to the next are {ch_text}. Plot all {n} values and join the points in time order.",
            "Draw a time series plot. The first value, at time 1, is {start}; after that the changes are {ch_text}. {intro}",
        ),
        intro=ctx["intro"],
        start=start,
        ch_text=ch_text,
        n=n,
    )
    pts = list(zip(range(1, n + 1), ys))
    return _series_question(
        rng,
        ctx,
        ys,
        text,
        [f"Add each change to the previous value: {', '.join(map(str, ys))}. Plot {_text(pts)}."],
        "A blank grid for a time series plot.",
    )


@generator(
    SERIES,
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def dr1_time_series(rng, tier="easy"):
    """Construct a time series plot; the tiers are described in the module docstring."""
    build = {"easy": _series_easy, "medium": _series_medium, "hard": _series_hard}.get(tier)
    if build is None:
        _unknown_tier(tier)
    return retry_draw(rng, build)


@generator("exd_x_" + SERIES, level=12, presets={"exam": dict(tier="exam")})
def exd_x_dr1_time_series(rng, tier="exam"):
    """Exam chain: plot twelve values; find the cycle length, the change of level per cycle and a forecast."""
    if tier != "exam":
        _unknown_tier(tier)
    return retry_draw(rng, _series_exam)


def _series_exam(rng):
    ctx = rng.choice([c for c in _SERIES if c["p"] in (4, 5, 6, 7)])
    p = ctx["p"]
    cycles = 3 if p <= 5 else 2
    ys, offsets, delta = _seasonal(rng, p, cycles, (-4, -3, -2, 2, 3, 4, 5))
    n = len(ys)
    pts = list(zip(range(1, n + 1), ys))
    forecast = ys[n - p] + delta
    if forecast < 2:
        raise TooFewDistractors("the forecast is not a plausible amount")
    peak_season = max(range(p), key=lambda s: offsets[s]) + 1
    x_max, y_max = n + 1, max(ys) + 3
    fa = _poly_fields(pts, keep=True)
    da = _poly_dist(fa, _series_slips(pts), True, x_max, y_max)
    enough(da, 3)
    db = _nd(
        p,
        [
            (p + 1, "dr1-off-by-one"),
            (p - 1, "dr1-off-by-one"),
            (n, "dr1-whole-series-for-cycle"),
            (2 * p, "dr1-two-cycles-for-cycle"),
            (cycles, "dr1-count-for-length"),
        ],
    )
    dc = _nd(
        abs(delta),
        [
            (abs(delta) + 1, "dr1-off-by-one"),
            (max(abs(delta) - 1, 1), "dr1-off-by-one"),
            (abs(delta) * 2, "dr1-two-cycles-for-cycle"),
            (abs(ys[-1] - ys[0]), "dr1-first-to-last-for-change"),
        ],
    )
    dd = _nd(
        forecast,
        [
            (ys[n - p], "dr1-no-trend-added"),
            (ys[-1] + delta, "dr1-last-value-for-season"),
            (forecast + delta, "dr1-trend-added-twice"),
            (forecast - 1, "dr1-off-by-one"),
            (forecast + 1, "dr1-off-by-one"),
        ],
    )
    for dl in (db, dc, dd):
        enough(dl)
    listed = ", ".join(str(y) for y in ys)
    unit = ctx["x_word"].lower()
    text = phrase(
        rng,
        (
            "{intro} The {n} values were {listed}. The pattern repeats regularly and the "
            "level changes by the same amount from one cycle to the next.",
            "{intro} A series of {n} values is {listed}. It follows a pattern that repeats, "
            "and each cycle is higher or lower than the last by the same amount.",
            "Look at this series: {listed}. {intro} The pattern starts again at regular intervals, "
            "and the level moves by a fixed amount from cycle to cycle.",
        ),
        intro=ctx["intro"],
        n=n,
        unit=unit,
        listed=listed,
        p=p,
    )
    fig = _blank(
        "A blank grid for a time series plot.",
        x_max,
        y_max,
        reply="polyline",
        tool="polyline",
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        count=n,
    )
    return dict(
        text=text,
        figure=fig,
        parts=[
            part(
                "Draw a time series plot of all the values, starting at time 1.",
                "polyline",
                {k: fa[k] for k in ("answer", "answer_text", "answer_latex")},
                3,
                [f"Plot {_text(pts)} and join them in time order."],
                da,
            ),
            part(
                f"How many {ctx['x_word'].lower()}s are in one cycle of the pattern?",
                "number",
                num_fields(p),
                1,
                [f"The pattern of the plot repeats every {p} {ctx['x_word'].lower()}s."],
                db,
                sensible_range=(1, None),
            ),
            part(
                f"By how much does the level change from one cycle to the next? Give the size of the change.",
                "number",
                num_fields(abs(delta)),
                2,
                [f"{ctx['x_word']} {p + 1} is {ys[p]} and {ctx['x_word'].lower()} 1 is {ys[0]}: a change of {delta}."],
                dc,
                sensible_range=(0, None),
            ),
            part(
                f"Forecast the value at {ctx['x_word'].lower()} {n + 1}, keeping the same pattern and the same change of level.",
                "number",
                num_fields(forecast),
                2,
                [f"The same point in the last cycle is {ys[n - p]}; add the change of {delta}: {forecast}."],
                dd,
                sensible_range=(0, None),
            ),
        ],
    )
