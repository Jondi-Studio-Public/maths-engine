"""Generates `f3-read-a-column-graph` questions (AC9M3ST02, AC9M4ST02, Years 3-4): read a column graph or a bar
graph drawn as a ``bar_chart`` figure, `answer: number`.

The prompt names the categories asked about and never a value. Orientation (columns standing up, or bars lying
across) is drawn at random for every tier.

Ladder (reasoning escalation, not size):
  easy:   read one bar against the value axis. The scale step is 1, 2 or 5 and every bar ends on a gridline.
  medium: combine two reads: how many more one bar is than another, or the two bars added. The step is 2, 5 or 10.
  hard:   one of four things the earlier tiers never ask for. (a) a grouped graph with two series: how many more
          one series than the other chose a category; (b) the same graph: both series added for one category;
          (c) a grouped graph: every bar of one series added; (d) a single series whose bars end on the half
          lines between gridlines (the step is 2, 10 or 20): the difference between the highest and lowest bar,
          found by the learner.

The figure reader registered for the topic re-derives every answer from ``question.figure`` and the names in
the prompt, never from the keyed answer.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase
from mathgen.figure_kinds.data import bar_chart
from mathgen.generators._stat_display_common import (
    leaks,
    mentioned_in_order,
    number_fields,
    pick_number_distractors,
    verify_wrong_everywhere,
)

TOPIC = "stat2_column_graph"
TIERS = ("easy", "medium", "hard")

_SINGLE = [
    dict(
        cats=("Ava", "Ben", "Chloe", "Dan", "Ella", "Finn", "Gus"),
        ordered=False,
        title="Books read in a term",
        x="Student",
        y="Number of books",
        read="How many books did {c} read?",
        more="How many more books did {a} read than {b}?",
        fewer="How many fewer books did {b} read than {a}?",
        total="How many books did {a} and {b} read altogether?",
        extreme="How many more books did the student who read the most read than the student who read the least?",
    ),
    dict(
        cats=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"),
        ordered=True,
        title="Apples sold at a stall",
        x="Day",
        y="Number of apples sold",
        read="How many apples were sold on {c}?",
        more="How many more apples were sold on {a} than on {b}?",
        fewer="How many fewer apples were sold on {b} than on {a}?",
        total="How many apples were sold on {a} and {b} altogether?",
        extreme="How many more apples were sold on the best day than on the worst day?",
    ),
    dict(
        cats=("January", "February", "March", "April", "May", "June"),
        ordered=True,
        title="Monthly rainfall",
        x="Month",
        y="Rainfall (mm)",
        read="How many millimetres of rain fell in {c}?",
        more="How many more millimetres of rain fell in {a} than in {b}?",
        fewer="How many fewer millimetres of rain fell in {b} than in {a}?",
        total="How many millimetres of rain fell in {a} and {b} altogether?",
        extreme="How many more millimetres of rain fell in the wettest month than in the driest month?",
    ),
    dict(
        cats=("red", "blue", "white", "silver", "black", "green"),
        ordered=False,
        title="Cars in a car park",
        x="Colour",
        y="Number of cars",
        read="How many {c} cars were in the car park?",
        more="How many more {a} cars than {b} cars were in the car park?",
        fewer="How many fewer {b} cars than {a} cars were in the car park?",
        total="How many {a} and {b} cars were in the car park altogether?",
        extreme="How many more cars were there of the most common colour than of the least common colour?",
    ),
    dict(
        cats=("Lions", "Eagles", "Sharks", "Tigers", "Hawks", "Wolves"),
        ordered=False,
        title="Goals scored in a season",
        x="Team",
        y="Number of goals",
        read="How many goals did the {c} score?",
        more="How many more goals did the {a} score than the {b}?",
        fewer="How many fewer goals did the {b} score than the {a}?",
        total="How many goals did the {a} and the {b} score altogether?",
        extreme="How many more goals did the top team score than the lowest team?",
    ),
]

_GROUPED = [
    dict(
        cats=("Soccer", "Netball", "Cricket", "Tennis", "Swimming", "Hockey"),
        title="Favourite sport",
        x="Sport",
        y="Number of students",
        verb="chose",
        pairs=(
            (("Girls", "girls"), ("Boys", "boys")),
            (("Year 5", "Year 5 students"), ("Year 6", "Year 6 students")),
        ),
    ),
    dict(
        cats=("Dog", "Cat", "Bird", "Fish", "Rabbit", "Horse"),
        title="Favourite pet",
        x="Pet",
        y="Number of students",
        verb="chose",
        pairs=(
            (("Girls", "girls"), ("Boys", "boys")),
            (("Class A", "Class A students"), ("Class B", "Class B students")),
        ),
    ),
    dict(
        cats=("Apple", "Banana", "Orange", "Grapes", "Pear", "Mango"),
        title="Favourite fruit",
        x="Fruit",
        y="Number of students",
        verb="chose",
        pairs=(
            (("Girls", "girls"), ("Boys", "boys")),
            (("Year 3", "Year 3 students"), ("Year 4", "Year 4 students")),
        ),
    ),
]

_GROUP_DIFF = (
    "How many more {hi} than {lo} {verb} {c}?",
    "Find how many more {hi} than {lo} {verb} {c}.",
    "How many more {hi} {verb} {c} than {lo}?",
)
_GROUP_TOTAL_CAT = (
    "How many students {verb} {c} altogether?",
    "Find the total number of students who {verb} {c}.",
    "How many students in all {verb} {c}?",
)
_GROUP_SERIES_TOTAL = (
    "How many {s} are shown on the graph altogether?",
    "Find the total number of {s} on the graph.",
    "How many {s} were asked altogether?",
)


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _fmt(v):
    return number_fields(v)[1]


def _axis(rng, steps, lo_m=5, hi_m=10):
    step = _pick(rng, steps)
    m = rng.randint(lo_m, hi_m)
    return step, step * m


def _cats(rng, ctx, n):
    pool = list(ctx["cats"])
    if ctx.get("ordered"):
        start = rng.randint(0, len(pool) - n)
        return pool[start : start + n]
    rng.shuffle(pool)
    return pool[:n]


def _chart(rng, ctx, cats, series, vmax, step, minor=False):
    horizontal = rng.random() < 0.5
    xl, yl = (ctx["y"], ctx["x"]) if horizontal else (ctx["x"], ctx["y"])
    return bar_chart(
        categories=cats,
        series=series,
        value_max=vmax,
        value_step=step,
        minor_lines=minor,
        orientation="horizontal" if horizontal else "vertical",
        alt="a bar graph" if horizontal else "a column graph",
        title=ctx["title"],
        x_label=xl,
        y_label=yl,
    )


def _values(rng, n, step, vmax, *, half=False, distinct=False, odd_halves=0):
    """``n`` values that are multiples of ``step`` (of ``step / 2`` with ``half``)."""
    unit = step // 2 if half else step
    top = vmax // unit
    for _ in attempts(500, "column-graph values"):
        vals = [unit * rng.randint(1, top) for _ in range(n)]
        if distinct and len(set(vals)) < n:
            continue
        if odd_halves and sum(1 for v in vals if v % step) < odd_halves:
            continue
        return vals


def _scale_candidates(step, *vals):
    out = []
    for v in vals:
        out += [v + step, v - step]
    return out


def _draw(rng, tier):
    if tier == "hard" and rng.random() < 0.6:
        return _draw_grouped(rng)
    ctx = _pick(rng, _SINGLE)
    if tier == "easy":
        step, vmax = _axis(rng, (1, 2, 5), 5, 10)
        n = rng.randint(4, 6)
        cats = _cats(rng, ctx, n)
        vals = _values(rng, n, step, vmax)
        fig = _chart(rng, ctx, cats, [{"name": "", "values": vals}], vmax, step)
        i = rng.randrange(n)
        key = vals[i]
        text = phrase(rng, _lead(ctx, "read"), c=cats[i])
        cand = [
            key + step,
            key - step,
            key // step if step > 1 else None,
            vals[(i + 1) % n],
            vals[i - 1],
            key + 1 if step > 1 else key + 2,
        ]
        working = [f"The {cats[i]} bar ends at {key} on the scale."]
        return fig, text, key, cand, working
    if tier == "medium":
        step, vmax = _axis(rng, (2, 5, 10), 6, 10)
        n = rng.randint(4, 6)
        cats = _cats(rng, ctx, n)
        vals = _values(rng, n, step, vmax, distinct=True)
        fig = _chart(rng, ctx, cats, [{"name": "", "values": vals}], vmax, step)
        a, b = rng.sample(range(n), 2)
        if rng.random() < 0.5:
            if vals[a] < vals[b]:
                a, b = b, a
            fewer = rng.random() < 0.35
            key = vals[a] - vals[b]
            if fewer:
                text = ctx["fewer"].format(a=cats[a], b=cats[b])
            else:
                text = ctx["more"].format(a=cats[a], b=cats[b])
            cand = [
                vals[a] + vals[b],
                vals[a] // step - vals[b] // step,
                vals[a],
                key + step,
                key - step,
                vals[b],
            ]
            working = [f"{cats[a]} is {vals[a]} and {cats[b]} is {vals[b]}.", f"{vals[a]} - {vals[b]} = {key}."]
        else:
            key = vals[a] + vals[b]
            text = ctx["total"].format(a=cats[a], b=cats[b])
            cand = [
                abs(vals[a] - vals[b]),
                vals[a] // step + vals[b] // step,
                key + step,
                key - step,
                vals[a],
                vals[b],
            ]
            working = [f"{cats[a]} is {vals[a]} and {cats[b]} is {vals[b]}.", f"{vals[a]} + {vals[b]} = {key}."]
        return fig, text, key, cand, working
    return _draw_single_halves(rng, ctx)


def _lead(ctx, field):
    # the leading wording stays the context's own; a second phrasing varies the instruction
    base = ctx[field]
    return (base, "Use the graph. " + base, "Look at the graph. " + base, "From the graph: " + base)


def _draw_single_halves(rng, ctx):
    step, vmax = _axis(rng, (2, 10, 20), 5, 8)
    n = rng.randint(4, 6)
    cats = _cats(rng, ctx, n)
    vals = _values(rng, n, step, vmax, half=True, distinct=True, odd_halves=2)
    fig = _chart(rng, ctx, cats, [{"name": "", "values": vals}], vmax, step, minor=True)
    hi, lo = max(vals), min(vals)
    check(vals.count(hi) == 1 and vals.count(lo) == 1, "extreme bars must be unique")
    key = hi - lo
    text = ctx["extreme"]
    srt = sorted(vals)
    cand = [hi + lo, hi, srt[-2] - lo, hi - srt[1], key + step // 2, key - step // 2, key + step]
    working = [f"Highest bar {hi}, lowest bar {lo}.", f"{hi} - {lo} = {key}."]
    return fig, text, key, cand, working


def _draw_grouped(rng):
    ctx = _pick(rng, _GROUPED)
    (s1, s2) = _pick(rng, ctx["pairs"])
    if rng.random() < 0.5:
        s1, s2 = s2, s1
    step, vmax = _axis(rng, (2, 5, 10), 5, 8)
    half = step in (2, 10)
    n = rng.randint(4, 5)
    cats = _cats(rng, dict(ctx, ordered=False), n)
    unit = step // 2 if half else step
    top = vmax // unit
    for _ in attempts(200, "grouped bars"):
        v1 = [unit * rng.randint(1, top) for _ in range(n)]
        v2 = [unit * rng.randint(1, top) for _ in range(n)]
        # on a half-line scale, several bars end on a half line so the finer reading is genuinely needed
        if not half or sum(1 for v in v1 + v2 if v % step) >= 3:
            break
    series = [{"name": s1[0], "values": v1}, {"name": s2[0], "values": v2}]
    fig = _chart(rng, ctx, cats, series, vmax, step, minor=half)
    variant = _pick(rng, ("diff", "total_cat", "series_total"))
    i = rng.randrange(n)
    verb = ctx["verb"]
    if variant == "diff":
        if v1[i] == v2[i]:
            v2[i] = v2[i] + unit if v2[i] + unit <= vmax else v2[i] - unit
            series[1]["values"] = v2
            fig = _chart(rng, ctx, cats, series, vmax, step, minor=half)
        hi_s, lo_s = (s1, s2) if v1[i] > v2[i] else (s2, s1)
        hi_v, lo_v = max(v1[i], v2[i]), min(v1[i], v2[i])
        key = hi_v - lo_v
        text = phrase(rng, _GROUP_DIFF, hi=hi_s[1], lo=lo_s[1], verb=verb, c=cats[i])
        cand = [hi_v + lo_v, hi_v, key + unit, key - unit, key + step, lo_v]
        working = [f"{hi_s[0]} is {hi_v} and {lo_s[0]} is {lo_v}.", f"{hi_v} - {lo_v} = {key}."]
    elif variant == "total_cat":
        key = v1[i] + v2[i]
        text = phrase(rng, _GROUP_TOTAL_CAT, verb=verb, c=cats[i])
        cand = [abs(v1[i] - v2[i]), max(v1[i], v2[i]), key + unit, key - unit, key + step, min(v1[i], v2[i])]
        working = [f"{s1[0]} is {v1[i]} and {s2[0]} is {v2[i]}.", f"{v1[i]} + {v2[i]} = {key}."]
    else:
        pick1 = rng.random() < 0.5
        s, vs, other = (s1, v1, v2) if pick1 else (s2, v2, v1)
        key = sum(vs)
        text = phrase(rng, _GROUP_SERIES_TOTAL, s=s[1])
        cand = [sum(other), sum(v1) + sum(v2), key + unit, key - unit, key - vs[0], key + step]
        working = [f"{s[0]} bars: {', '.join(str(v) for v in vs)}.", f"Total = {key}."]
    return fig, text, key, cand, working


@generator(TOPIC, level=5, presets={t: {"tier": t} for t in TIERS})
def stat2_column_graph(rng, tier="easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a column graph question"):
        fig, text, key, cand, working = _draw(rng, tier)
        answer, answer_text, answer_latex = number_fields(key)
        distractors = pick_number_distractors(TOPIC, key, cand)
        verify_wrong_everywhere(TOPIC, key, distractors)
        if leaks(fig, [answer_text, *distractors]):
            continue
        return dict(
            text=text,
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_latex,
            distractors=distractors,
            working=[line[:1].upper() + line[1:] for line in working],
            figure=fig,
        )


# ---------------------------------------------------------------- the figure reader


def _read(question) -> str:
    fig, text = question.figure, question.text
    cats = fig["categories"]
    series = fig["series"]
    low = text.lower()
    if len(series) == 1:
        vals = dict(zip(cats, series[0]["values"]))
        if re.search(r"the most|the best|wettest|most common|top team", low):
            return _fmt(max(vals.values()) - min(vals.values()))
        named = mentioned_in_order(cats, text)
        if len(named) == 1:
            return _fmt(vals[named[0]])
        check(len(named) == 2, "the prompt must name one or two bars")
        if "altogether" in low:
            return _fmt(vals[named[0]] + vals[named[1]])
        return _fmt(abs(vals[named[0]] - vals[named[1]]))
    named = mentioned_in_order(cats, text)
    by = {s["name"]: dict(zip(cats, s["values"])) for s in series}
    if len(named) == 1:
        c = named[0]
        if "more" in low:
            return _fmt(abs(by[series[0]["name"]][c] - by[series[1]["name"]][c]))
        return _fmt(sum(v[c] for v in by.values()))
    check(not named, "a series-total prompt names no category")
    chosen = [s for s in series if re.search(rf"\b{re.escape(s['name'].lower())}\b", low)]
    check(len(chosen) == 1, "the prompt must name exactly one series")
    return _fmt(sum(chosen[0]["values"]))


figures.register_reader(TOPIC, _read)
