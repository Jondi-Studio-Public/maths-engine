"""Generates figure questions on distance-versus-time graphs (`senior-interpret-distance-versus-time-graphs`).

Skill ``senior-interpret-distance-versus-time-graphs``. The learner sees a
connected polyline of straight segments on a labelled integer grid (time on
the x-axis, distance from home on the y-axis) and types the speed (or
velocity) during a stated time interval that lies inside ONE segment. The key
is that segment's gradient.

Construction is answer-first: the target segment's slope class is drawn
(stationary / away / return on the hard tier, an integer or a small fraction
elsewhere), then the polyline is built around it with every vertex on a grid
intersection, then the key is re-derived a second, different way
(:func:`_verify`: evaluate the polyline at the interval's two ends by linear
interpolation, then take rise over run), so a figure that does not carry the
key cannot ship.

Tiers escalate by the learner's DECISION, not by bigger numbers:

- easy: the stated interval is exactly one whole segment, every segment moves
  away from home at an integer speed -- identify the segment, rise over run.
- medium: three segments of different steepness (possibly stationary); the
  interval's ends are NOT vertices, so the learner must decide which segment
  covers the interval rather than reading between the interval's own points.
- hard: the graph has a stationary and a return segment; the prompt states
  "movement away from home is positive velocity" and asks for the velocity, so
  the key may be 0 or negative -- the sign comes from the direction of travel.
- exam: the time axis is in MINUTES but the speed is asked in km per hour --
  the raw gradient is km per minute, so the learner converts the unit.

Every distractor differs in VALUE from the key and from each other (the
`fraction` grader compares by value), checked on every candidate.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .. import figures
from ..core import (
    VerificationError,
    check,
    check_pairwise_distinct,
    generator,
    phrase,
    rational_text,
)

TOPIC = "pv1b_distance_time"
ALT = "a distance-time graph made of straight-line segments on a coordinate grid"

# context -> axis labels, time word, speed unit (prompt wording)
_CONTEXTS = {
    "hours": dict(x_label="Time (hours)", y_label="Distance from home (km)", t_word="hours", unit="km per hour"),
    "seconds": dict(
        x_label="Time (seconds)", y_label="Distance from home (metres)", t_word="seconds", unit="metres per second"
    ),
    "minutes": dict(x_label="Time (minutes)", y_label="Distance from home (km)", t_word="minutes", unit="km per hour"),
}

_ANSWER_FORM = " Give your answer as a whole number or a fraction."


# ---------------------------------------------------------------- geometry


def _slope(ts, ds, i) -> Fraction:
    return Fraction(ds[i + 1] - ds[i], ts[i + 1] - ts[i])


def _dist_at(ts, ds, t) -> Fraction:
    """Distance at time ``t`` by linear interpolation along the polyline."""
    for i in range(len(ts) - 1):
        if ts[i] <= t <= ts[i + 1]:
            return ds[i] + _slope(ts, ds, i) * (t - ts[i])
    raise VerificationError(f"time {t} is outside the graph")


def _verify(ts, ds, a, b, scale, key) -> None:
    """Independent route to the key: interpolate the polyline at both ends of
    the interval (never reading a segment's own endpoints), then rise / run,
    times ``scale`` (60 when the time axis is minutes and the speed is km/h)."""
    covering = [i for i in range(len(ts) - 1) if ts[i] <= a and b <= ts[i + 1]]
    check(len(covering) == 1, "the stated interval must lie inside exactly one segment")
    value = (_dist_at(ts, ds, b) - _dist_at(ts, ds, a)) / (b - a) * scale
    check(value == key, "distance-time: interpolated speed disagrees with the key")


def _figure(ts, ds, ctx, x_grid):
    segments = [{"from": [ts[i], ds[i]], "to": [ts[i + 1], ds[i + 1]]} for i in range(len(ts) - 1)]
    return figures.cartesian(
        x_range=(0, ts[-1]),
        y_range=(0, max(ds) + 1),
        alt=ALT,
        x_grid=x_grid,
        y_grid=1,
        equal_scale=False,
        segments=segments,
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        tick_labels=True,
    )


# ---------------------------------------------------------------- answer text


def _answer_fields(key: Fraction):
    if key.denominator == 1:
        answer = int(key)
        latex = str(int(key))
    else:
        answer = {"p": key.numerator, "q": key.denominator}
        sign = "-" if key < 0 else ""
        latex = f"{sign}\\frac{{{abs(key.numerator)}}}{{{key.denominator}}}"
    return answer, rational_text(key), latex


def _pick(key: Fraction, named: list[tuple[str, Fraction]], n: int = 3) -> list[str]:
    """First ``n`` candidates (fixed priority) whose VALUE differs from the key
    and from every candidate already kept; formatted like the key."""
    kept: list[Fraction] = []
    pool: list[Fraction] = []
    for _name, value in named:
        if value == key or value in kept or value in pool:
            continue
        if len(kept) < n:
            kept.append(value)
        else:
            pool.append(value)
    # keep at least one option of the key's shape (whole number vs fraction)
    same = lambda v: (v.denominator == 1) == (key.denominator == 1)
    if kept and not any(same(v) for v in kept):
        alt = next((v for v in pool if same(v)), None)
        if alt is not None:
            kept[-1] = alt
    check_pairwise_distinct(key, *kept)
    return [rational_text(v) for v in kept]


# ---------------------------------------------------------------- drawing paths


def _rise(rng, kind: str, dt: int, vmax: int, cur_d: int):
    """A segment's rise for ``kind`` ('stat' | 'away' | 'return'), or None when
    it cannot be drawn. Reduced speed denominators stay in {1, 2, 3}."""
    if kind == "stat":
        return 0
    cap = vmax * dt if kind == "away" else min(vmax * dt, cur_d)
    if cap < 1:
        return None
    for _ in range(30):
        r = rng.randint(1, cap)
        if Fraction(r, dt).denominator <= 3:
            return r if kind == "away" else -r
    return None


def _draw_path(rng, *, n, dt_target, dt_other, vmax, target_kinds, other_kinds, d0_max, dmax, tmax, need_all_kinds):
    """Answer-first path: the target segment's kind is drawn first, the others
    around it. Returns ``(ts, ds, idx)`` with every vertex an integer and
    adjacent segments of different slope."""
    for _attempt in range(800):
        idx = rng.randrange(n)
        kinds = [rng.choices(*other_kinds)[0] for _ in range(n)]
        kinds[idx] = rng.choices(*target_kinds)[0]
        if need_all_kinds and set(kinds) != {"away", "stat", "return"}:
            continue
        dts = [rng.choice(dt_target if i == idx else dt_other) for i in range(n)]
        ts = [0]
        for dt in dts:
            ts.append(ts[-1] + dt)
        if ts[-1] > tmax:
            continue
        ds = [rng.randint(0, d0_max)]
        ok = True
        for kind, dt in zip(kinds, dts):
            r = _rise(rng, kind, dt, vmax, ds[-1])
            if r is None:
                ok = False
                break
            ds.append(ds[-1] + r)
        if not ok or max(ds) > dmax or min(ds) < 0:
            continue
        if any(_slope(ts, ds, i) == _slope(ts, ds, i + 1) for i in range(n - 1)):
            continue
        return ts, ds, idx
    raise RuntimeError("distance_time: could not draw a path")


def _inside_interval(rng, lo: int, hi: int, step: int = 1):
    """Two distinct times strictly inside (lo, hi), multiples of ``step`` from lo."""
    inner = list(range(lo + step, hi, step))
    a, b = sorted(rng.sample(inner, 2))
    return a, b


# ---------------------------------------------------------------- shared finish

_SINGULAR = {"hours": "hour", "seconds": "second", "minutes": "minute", "metres": "metre"}


def _singular(s: str) -> str:
    """ "1 hours" -> "1 hour" etc. (a count of exactly one takes the singular)."""
    return re.sub(r"(?<![0-9/.])1 (hours|seconds|minutes|metres)\b", lambda m: "1 " + _SINGULAR[m.group(1)], s)


def _finish(rng, tier, ts, ds, idx, a, b, ctx, patterns, *, scale=1, hard=False):
    n = len(ts) - 1
    seg = _slope(ts, ds, idx)
    key = seg * scale
    _verify(ts, ds, a, b, scale, key)

    rise = ds[idx + 1] - ds[idx]
    run = ts[idx + 1] - ts[idx]
    prev_v = _slope(ts, ds, idx - 1) * scale if idx > 0 else None
    next_v = _slope(ts, ds, idx + 1) * scale if idx < n - 1 else None
    neighbours = [v for v in (next_v, prev_v) if v is not None]
    average = Fraction(ds[-1] - ds[0], ts[-1] - ts[0]) * scale
    rise_only = Fraction(rise)
    inverted = Fraction(run, rise) if rise != 0 else None

    named: list[tuple[str, Fraction]] = []
    if tier == "easy" or tier == "medium":
        if key == 0:
            named.append(("height-instead-of-speed", Fraction(ds[idx])))
            named.append(("segment-duration", Fraction(run)))
        named.append(("rise-only", rise_only))
        if inverted is not None:
            named.append(("inverted-slope", inverted))
        named += [("neighbouring-segment", v) for v in neighbours]
        named.append(("average-over-whole-graph", average))
    elif tier == "hard":
        if key != 0:
            named.append(("sign-flipped", -key))
        else:
            named.append(("height-instead-of-speed", Fraction(ds[idx])))
            named.append(("segment-duration", Fraction(run)))
        named.append(("rise-only", rise_only))
        if inverted is not None:
            named.append(("inverted-slope", inverted))
        named += [("neighbouring-segment", v) for v in neighbours]
        named.append(("average-over-whole-graph", average))
    else:  # exam
        named.append(("unconverted-unit", seg))
        named.append(("rise-only", rise_only))
        named.append(("unconverted-unit", seg * 6))
        named += [("neighbouring-segment", v) for v in neighbours]
        named.append(("average-over-whole-graph", average))
        if inverted is not None:
            named.append(("inverted-slope", inverted))
    distractors = _pick(key, named)

    answer, answer_text, answer_latex = _answer_fields(key)
    text = phrase(rng, patterns, a=a, b=b, unit=ctx["unit"], tw=ctx["t_word"]) + _ANSWER_FORM
    figure = _figure(ts, ds, ctx, 5 if scale == 60 else 1)
    figures.assert_no_leak(figure, [answer_text, *distractors])

    if scale == 60:
        working = [
            f"Between {ts[idx]} and {ts[idx + 1]} minutes the graph is one straight segment, so the "
            f"gradient is rise over run = {rise} / {run} = {rational_text(seg)} km per minute.",
            f"There are 60 minutes in an hour, so the speed is 60 times that: {answer_text} km per hour.",
        ]
    else:
        working = [
            f"The times {a} and {b} both fall inside one straight segment, from ({ts[idx]}, {ds[idx]}) "
            f"to ({ts[idx + 1]}, {ds[idx + 1]}).",
            f"Gradient = rise over run = {rise} / {run} = {answer_text} {ctx['unit']}"
            + (" (negative: the distance from home is falling)." if key < 0 else "."),
        ]
    text = _singular(text)
    working = [_singular(w) for w in working]
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        distractors=distractors,
        working=working,
        figure=figure,
    )


# ---------------------------------------------------------------- tiers

_EASY_PATTERNS = (
    "The graph shows the distance from home over time. What was the speed, in {unit}, from {a} to {b} {tw}?",
    "A distance-time graph is drawn below. Find the speed in {unit} during the interval from {a} {tw} to {b} {tw}.",
    "Use the graph to work out the speed, in {unit}, between time {a} and time {b} {tw}.",
)
_MEDIUM_PATTERNS = (
    "The graph shows a journey. Find the speed, in {unit}, between {a} and {b} {tw}.",
    "Use the graph to find how fast the journey was going, in {unit}, from {a} {tw} to {b} {tw}.",
    "Read the graph to state the speed, in {unit}, during the time from {a} to {b} {tw}.",
)
_SIGN = "Take movement away from home as positive velocity."
_HARD_PATTERNS = (
    "The graph shows the distance from home over time. "
    + _SIGN
    + " Find the velocity, in {unit}, between {a} and {b} {tw}.",
    _SIGN + " Use the graph to find the velocity, in {unit}, from {a} {tw} to {b} {tw}.",
    "A journey is shown on the graph. Moving away from home counts as positive velocity and "
    "moving towards home as negative velocity. What is the velocity, in {unit}, from {a} to {b} {tw}?",
)
_EXAM_PATTERNS = (
    "The graph shows a trip, with time in minutes and distance in kilometres. Find the speed, "
    "in kilometres per hour, between {a} and {b} minutes.",
    "Distance from home is graphed against time in minutes. Use the graph to find the speed in "
    "kilometres per hour from {a} minutes to {b} minutes.",
    "Read the graph (time in minutes, distance in km) and state the speed, in kilometres per hour, "
    "during the interval from {a} to {b} minutes.",
)


def _easy(rng):
    ctx_name = rng.choices(("hours", "seconds"), (65, 35))[0]
    ctx = _CONTEXTS[ctx_name]
    vmax = 9 if ctx_name == "hours" else 5
    for _attempt in range(400):
        idx = rng.randrange(3)
        speeds = [rng.randint(1, vmax) for _ in range(3)]
        dts = [rng.choice((1, 2, 3)) for _ in range(3)]
        if speeds[0] == speeds[1] or speeds[1] == speeds[2]:
            continue
        ts, ds = [0], [0]
        for v, dt in zip(speeds, dts):
            ts.append(ts[-1] + dt)
            ds.append(ds[-1] + v * dt)
        if ts[-1] > 9 or ds[-1] > 24:
            continue
        return _finish(rng, "easy", ts, ds, idx, ts[idx], ts[idx + 1], ctx, _EASY_PATTERNS)
    raise RuntimeError("distance_time/easy: could not draw a path")


def _medium(rng):
    ctx = _CONTEXTS[rng.choices(("hours", "seconds"), (65, 35))[0]]
    ts, ds, idx = _draw_path(
        rng,
        n=3,
        dt_target=(3, 4, 5),
        dt_other=(3, 4, 5),
        vmax=5,
        target_kinds=(("away", "stat"), (90, 10)),
        other_kinds=(("away", "stat"), (85, 15)),
        d0_max=0,
        dmax=30,
        tmax=15,
        need_all_kinds=False,
    )
    a, b = _inside_interval(rng, ts[idx], ts[idx + 1])
    return _finish(rng, "medium", ts, ds, idx, a, b, ctx, _MEDIUM_PATTERNS)


def _hard(rng):
    ctx = _CONTEXTS[rng.choices(("hours", "seconds"), (65, 35))[0]]
    ts, ds, idx = _draw_path(
        rng,
        n=4,
        dt_target=(3, 4),
        dt_other=(2, 3, 4),
        vmax=5,
        target_kinds=(("away", "stat", "return"), (43, 14, 43)),
        other_kinds=(("away", "stat", "return"), (2, 1, 2)),
        d0_max=8,
        dmax=30,
        tmax=16,
        need_all_kinds=True,
    )
    a, b = _inside_interval(rng, ts[idx], ts[idx + 1])
    return _finish(rng, "hard", ts, ds, idx, a, b, ctx, _HARD_PATTERNS)


def _exam(rng):
    ctx = _CONTEXTS["minutes"]
    for _attempt in range(800):
        idx = rng.randrange(3)
        dts = [rng.choice((15, 20, 25, 30) if i == idx else (10, 15, 20, 25, 30)) for i in range(3)]
        rises = [rng.randint(-(-3 * dt // 60), dt // 2) for dt in dts]  # 3..30 km per hour
        ts, ds = [0], [0]
        for dt, r in zip(dts, rises):
            ts.append(ts[-1] + dt)
            ds.append(ds[-1] + r)
        if ts[-1] > 90 or ds[-1] > 30:
            continue
        if any(_slope(ts, ds, i) == _slope(ts, ds, i + 1) for i in range(2)):
            continue
        a, b = _inside_interval(rng, ts[idx], ts[idx + 1], 5)
        return _finish(rng, "exam", ts, ds, idx, a, b, ctx, _EXAM_PATTERNS, scale=60)
    raise RuntimeError("distance_time/exam: could not draw a path")


@generator(
    TOPIC,
    level=13,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def pv1b_distance_time(rng, variant="hard"):
    """Speed (or velocity) during a stated interval of a straight-segment
    distance-time graph. See the module docstring for the four tiers."""
    if variant == "easy":
        return _easy(rng)
    if variant == "medium":
        return _medium(rng)
    if variant == "hard":
        return _hard(rng)
    if variant == "exam":
        return _exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# ---------------------------------------------------------------- reader

_NUM_RE = re.compile(r"\d+")


def read_distance_time(question) -> str:
    """Re-derive the key from the figure (segments and the time-axis label)
    plus the two interval times the prompt states; never the generator's
    parameters or the stored answer."""
    fig = question.figure
    a, b = (int(x) for x in _NUM_RE.findall(question.text)[:2])
    for seg in fig["segments"]:
        (t0, d0), (t1, d1) = seg["from"], seg["to"]
        if t0 <= a and b <= t1:
            speed = Fraction(d1 - d0, t1 - t0)
            if "minutes" in fig["x_label"]:
                speed *= 60
            return rational_text(speed)
    raise VerificationError("no segment covers the stated interval")


figures.register_reader(TOPIC, read_distance_time)
