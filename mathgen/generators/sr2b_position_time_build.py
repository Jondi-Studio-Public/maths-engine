"""Generates `sr2b_position_time` questions (easy, medium, hard) and the exam chain `sr2b_exam_position_time`.

Skill ``position-time-graph-velocity`` (``senior-mathematics``, answer ``number``, with a figure): a position-time graph
of straight segments for something moving along a line, with positions measured from a fixed point (so they can be
negative) and every vertex on a grid intersection. The questions go past reading one segment's gradient (the existing
distance-time skill does that): they combine segments.

* easy: the average velocity over the whole journey, which is the net change in position over the time. The path doubles
  back, so the total distance travelled is not the net change, and the start is not the origin.
* medium: the average speed over the whole journey, which is the total distance travelled over the time. The path
  doubles back, so the distance has to be added segment by segment, in absolute value.
* hard: the time at which the object first returns to its starting position. The return crosses the starting level
  inside a segment (not at a vertex), the start is not the origin, and the path first goes away and may stop.
* exam (``sr2b_exam_position_time``): the velocity on the last segment (signed), the time of the first return to the
  start, the total distance travelled, and the average speed (follows through from the distance).

Every answer is drawn from the vertex list, then re-derived from the polyline in a second way: the time of return by
finding where the interpolated position first equals the start, the distance and net change by summing the segments.
The picture is the existing coordinate-plane figure; the question text never states a position or time that is the
answer.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from .. import figures
from ..core import attempts, check, exact_decimal_string, generator, phrase
from ._sr2b_kit import decimal_dist, int_dist, number_fields, pick, places_needed, retry_short

TOPIC = "sr2b_position_time"
EXAM = "sr2b_exam_position_time"
ALT = "a position-time graph made of straight-line segments on a coordinate grid, with positions above and below zero"

_CONTEXTS = (
    dict(
        intro="The graph shows the position of a cyclist on a straight east-west path, in kilometres east of a kiosk "
        "(a negative position is west of the kiosk), against the time in hours.",
        x_label="Time (hours)",
        y_label="Position (km east of kiosk)",
        vel="km per hour",
        vsym="km/h",
        dsym="km",
        tword="hours",
        dword="kilometres",
    ),
    dict(
        intro="The graph shows the position of a remote-controlled car on a straight track, in metres to the right of a "
        "marker (a negative position is to the left of it), against the time in seconds.",
        x_label="Time (seconds)",
        y_label="Position (m right of marker)",
        vel="metres per second",
        vsym="m/s",
        dsym="m",
        tword="seconds",
        dword="metres",
    ),
)


# -- geometry -----------------------------------------------------------------------------------------------------


def _compositions(total: int, n: int) -> list[tuple[int, ...]]:
    return [c for c in product(range(1, total), repeat=n) if sum(c) == total]


def _times(dts) -> list[int]:
    ts = [0]
    for dt in dts:
        ts.append(ts[-1] + dt)
    return ts


def _vel(ts, ps, i) -> Fraction:
    return Fraction(ps[i + 1] - ps[i], ts[i + 1] - ts[i])


def _pos_at(ts, ps, t) -> Fraction:
    for i in range(len(ts) - 1):
        if ts[i] <= t <= ts[i + 1]:
            return ps[i] + _vel(ts, ps, i) * (t - ts[i])
    raise AssertionError(f"time {t} is off the graph")  # pragma: no cover


def _distance(ts, ps) -> int:
    return sum(abs(ps[i + 1] - ps[i]) for i in range(len(ps) - 1))


def _first_return(ts, ps) -> Fraction | None:
    """The first time after 0 at which the position equals the starting position (None if it never does), found by
    solving each segment's line and keeping the first root that is not the start itself."""
    p0 = ps[0]
    for i in range(len(ts) - 1):
        v = _vel(ts, ps, i)
        if v == 0:
            continue
        t = ts[i] + (p0 - ps[i]) / v
        if ts[i] <= t <= ts[i + 1] and t > 0:
            return t
    return None


def _turns(ps) -> bool:
    """Whether the path reverses direction at some vertex (so distance and net change differ)."""
    steps = [b - a for a, b in zip(ps, ps[1:]) if b != a]
    return any(x * y < 0 for x, y in zip(steps, steps[1:]))


def _figure(ts, ps, ctx):
    segments = [{"from": [ts[i], ps[i]], "to": [ts[i + 1], ps[i + 1]]} for i in range(len(ts) - 1)]
    span = max(ps) - min(ps)
    return figures.cartesian(
        x_range=(0, ts[-1]),
        y_range=(min(ps) - 1, max(ps) + 1),
        alt=ALT,
        x_grid=1,
        y_grid=1 if span <= 20 else 2,
        equal_scale=False,
        segments=segments,
        x_label=ctx["x_label"],
        y_label=ctx["y_label"],
        tick_labels=True,
    )


def _check_polyline(ts, ps):
    check(len(ts) == len(ps) and all(b > a for a, b in zip(ts, ts[1:])), "bad time list")
    check(all(abs(p) <= 12 for p in ps), "position off the grid")
    check(max(ps) - min(ps) <= 22, "graph too tall")


# -- easy and medium: averages over a doubling-back path -----------------------------------------------------------


def _draw_back_and_forth(rng):
    for _ in attempts(2000, "a path that doubles back with a start away from the origin"):
        total = rng.choice((4, 5, 10))
        n = 3 if total < 10 else rng.choice((3, 4))
        dts = rng.choice(_compositions(total, n))
        ts = _times(dts)
        ps = [rng.randint(-8, 8) for _ in range(n + 1)]
        if len(set(ps)) < n or not _turns(ps) or ps[0] == 0:
            continue
        if any(ps[i] == ps[i + 1] for i in range(n - 1)):
            continue
        if ps[-1] == ps[0] or _distance(ts, ps) == abs(ps[-1] - ps[0]):
            continue
        if max(ps) - min(ps) > 20:
            continue
        return ts, ps
    raise AssertionError  # pragma: no cover


def _average_velocity(ts, ps) -> Fraction:
    return Fraction(ps[-1] - ps[0], ts[-1])


def _average_speed(ts, ps) -> Fraction:
    return Fraction(_distance(ts, ps), ts[-1])


def _easy(rng):
    ctx = rng.choice(_CONTEXTS)
    ts, ps = _draw_back_and_forth(rng)
    _check_polyline(ts, ps)
    ans = _average_velocity(ts, ps)
    # second route: interpolate the polyline at both ends of the journey
    check((_pos_at(ts, ps, ts[-1]) - _pos_at(ts, ps, 0)) / ts[-1] == ans, "average velocity differs")
    dp = places_needed(ans)
    mean_vel = sum(_vel(ts, ps, i) for i in range(len(ts) - 1)) / (len(ts) - 1)
    cands = [
        decimal_dist(_average_speed(ts, ps), dp, "sr2b-average-speed-for-average-velocity"),
        decimal_dist(Fraction(ps[-1], ts[-1]), dp, "sr2b-start-position-ignored"),
        decimal_dist(ps[-1] - ps[0], dp, "sr2b-displacement-not-divided"),
        decimal_dist(mean_vel, dp, "sr2b-segment-velocities-averaged"),
        decimal_dist(-ans, dp, "sr2b-velocity-sign-reversed"),
    ]
    stem = f"{ctx['intro']} " + phrase(
        rng,
        (
            "Find the average velocity over the whole journey, from 0 to {T} {tw}, in {vel}.",
            "What is the average velocity from time 0 to time {T} {tw}? Give your answer in {vel}.",
        ),
        T=ts[-1],
        tw=ctx["tword"],
        vel=ctx["vel"],
    )
    fig = _figure(ts, ps, ctx)
    text = number_fields(ans, dp)["answer_text"]
    figures.assert_no_leak(fig, [text])
    return dict(
        text=stem,
        figure=fig,
        working=[
            f"Average velocity is the change in position divided by the time taken.",
            f"The position goes from {ps[0]} to {ps[-1]}, a change of {ps[-1] - ps[0]}, in {ts[-1]} {ctx['tword']}: {ps[-1] - ps[0]} ÷ {ts[-1]} = {text} {ctx['vsym']}.",
        ],
        distractors=pick(text, [c for c in cands if c.text != text]),
        **number_fields(ans, dp),
    )


def _medium(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(500, "an average speed that terminates"):
        ts, ps = _draw_back_and_forth(rng)
        if len({abs(_vel(ts, ps, i)) for i in range(len(ts) - 1)}) > 1:
            break
    _check_polyline(ts, ps)
    ans = _average_speed(ts, ps)
    # second route: add the absolute change of the interpolated position over each unit of time
    total = sum(abs(_pos_at(ts, ps, t + 1) - _pos_at(ts, ps, t)) for t in range(ts[-1]))
    check(total / ts[-1] == ans, "average speed differs")
    dp = places_needed(ans)
    mean_speed = sum(abs(_vel(ts, ps, i)) for i in range(len(ts) - 1)) / (len(ts) - 1)
    farthest = max(abs(p - ps[0]) for p in ps)
    cands = [
        decimal_dist(_average_velocity(ts, ps), dp, "sr2b-average-velocity-for-average-speed"),
        decimal_dist(_distance(ts, ps), dp, "sr2b-distance-not-divided"),
        decimal_dist(mean_speed, dp, "sr2b-segment-speeds-averaged"),
        decimal_dist(Fraction(farthest, ts[-1]), dp, "sr2b-farthest-point-only"),
        decimal_dist(Fraction(abs(ps[-1] - ps[0]), ts[-1]), dp, "sr2b-average-velocity-for-average-speed"),
    ]
    stem = f"{ctx['intro']} " + phrase(
        rng,
        (
            "Find the average speed over the whole journey, from 0 to {T} {tw}, in {vel}.",
            "What is the average speed for the whole journey, from time 0 to time {T} {tw}? Give your answer in {vel}.",
        ),
        T=ts[-1],
        tw=ctx["tword"],
        vel=ctx["vel"],
    )
    fig = _figure(ts, ps, ctx)
    text = number_fields(ans, dp)["answer_text"]
    figures.assert_no_leak(fig, [text])
    legs = " + ".join(str(abs(ps[i + 1] - ps[i])) for i in range(len(ps) - 1))
    return dict(
        text=stem,
        figure=fig,
        working=[
            "Average speed is the total distance travelled divided by the time taken. Each leg counts, whichever way it goes.",
            f"The distances of the legs are {legs} = {_distance(ts, ps)} {ctx['dsym']}; {_distance(ts, ps)} ÷ {ts[-1]} = {text} {ctx['vsym']}.",
        ],
        distractors=pick(text, [c for c in cands if c.text != text]),
        **number_fields(ans, dp),
    )


# -- hard and exam: the return to the starting position -------------------------------------------------------------


def _draw_return(rng, *, tail: bool, total_options=None):
    """A path that leaves its start (not the origin), may stop, then crosses the starting level inside a segment."""
    for _ in attempts(4000, "a path with a first return inside a segment"):
        p0 = rng.choice([v for v in range(-6, 7) if v != 0])
        side = rng.choice((-1, 1))
        t1 = rng.choice((2, 3, 4))
        out = rng.randint(2, 6)
        ts = [0, t1]
        ps = [p0, p0 + side * out]
        if rng.random() < 0.5:  # a stop, or a further slow rise, before turning
            dt = rng.choice((1, 2))
            ts.append(ts[-1] + dt)
            ps.append(ps[-1] + side * rng.choice((0, 0, 1, 2)))
        k = rng.choice((1, 2, 3))  # speed back towards the start
        m = Fraction(ps[-1] - p0, k) * side  # time needed to come back to the start level
        if m.denominator != 1 or m < 1:
            continue
        m2 = rng.choice((1, 2, 3))
        ts.append(ts[-1] + int(m) + m2)
        ps.append(p0 - side * k * m2)
        if tail:
            ts.append(ts[-1] + rng.choice((1, 2, 3, 4)))
            ps.append(ps[-1] + rng.choice((-3, -2, 0, 2, 3)))
        if any(ts[i] == ts[i + 1] for i in range(len(ts) - 1)) or max(abs(p) for p in ps) > 12:
            continue
        if (
            any(ps[i] == ps[i + 1] for i in range(len(ps) - 2))
            and sum(ps[i] == ps[i + 1] for i in range(len(ps) - 1)) > 1
        ):
            continue
        if max(ps) - min(ps) > 20 or ts[-1] > 20:
            continue
        if total_options and ts[-1] not in total_options:
            continue
        r = _first_return(ts, ps)
        if r is None or r.denominator != 1 or r in ts:
            continue
        return ts, ps, int(r)
    raise AssertionError  # pragma: no cover


def _return_distractors(ts, ps, r):
    vertices = [t for t in ts if 0 < t < ts[-1]]
    turn = ts[max(range(len(ps)), key=lambda i: abs(ps[i] - ps[0]))]
    origin = None
    for i in range(len(ts) - 1):
        v = _vel(ts, ps, i)
        if v != 0:
            t = ts[i] + (0 - ps[i]) / v
            if ts[i] <= t <= ts[i + 1] and t > 0 and t.denominator == 1:
                origin = int(t)
                break
    nxt = next(t for t in ts if t > r)
    cands = [
        int_dist(turn, "sr2b-turning-point-for-return"),
        int_dist(nxt, "sr2b-next-vertex-for-crossing"),
        int_dist(ts[-1], "sr2b-end-of-graph-for-return"),
    ]
    if origin is not None:
        cands.insert(1, int_dist(origin, "sr2b-origin-for-start-position"))
    cands.append(int_dist(r + 1, "sr2b-return-time-off-by-one"))
    return [c for c in cands if int(c.text) != r and int(c.text) > 0], vertices


def _hard(rng):
    ctx = rng.choice(_CONTEXTS)
    ts, ps, r = _draw_return(rng, tail=rng.random() < 0.4)
    _check_polyline(ts, ps)
    # second route: scan whole units of time for the first sign change of (position - start)
    found = None
    for t in range(1, ts[-1] + 1):
        if _pos_at(ts, ps, t - 1) != ps[0] and (_pos_at(ts, ps, t) - ps[0]) * (_pos_at(ts, ps, t - 1) - ps[0]) <= 0:
            found = t
            break
    check(found is not None and found - 1 < r <= found, "return time differs from the scan")
    check(_pos_at(ts, ps, r) == ps[0], "the position at the return time is not the start")
    cands, _ = _return_distractors(ts, ps, r)
    stem = f"{ctx['intro']} " + phrase(
        rng,
        (
            "At what time does it first return to its starting position? Give the time in {tw}.",
            "The journey begins at t = 0. After how many {tw} does it first get back to where it started?",
        ),
        tw=ctx["tword"],
    )
    fig = _figure(ts, ps, ctx)
    figures.assert_no_leak(fig, [str(r)])
    return dict(
        text=stem,
        figure=fig,
        working=[
            f"The starting position is {ps[0]}, so we look for the first time after 0 when the graph is back at {ps[0]} (not at 0).",
            f"The graph returns to {ps[0]} part-way along a segment, at t = {r}.",
        ],
        distractors=pick(str(r), cands),
        **number_fields(r, 0),
    )


@generator(
    TOPIC,
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def sr2b_position_time(rng, tier="easy"):
    """One question on a position-time graph; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator(EXAM, level=13, presets={"exam": dict(tier="exam")})
@retry_short
def sr2b_exam_position_time(rng, tier="exam"):
    """Exam chain: last-segment velocity, first return, total distance, average speed."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(500, "an exam path whose velocity and average speed terminate"):
        ts, ps, r = _draw_return(rng, tail=True, total_options=(4, 5, 10, 20))
        v_last = _vel(ts, ps, len(ts) - 2)
        try:
            places_needed(v_last)
            places_needed(Fraction(_distance(ts, ps), ts[-1]))
        except ValueError:
            continue
        if v_last == 0:
            continue
        break
    _check_polyline(ts, ps)
    n = len(ts) - 1
    dist = _distance(ts, ps)
    check(dist == sum(abs(_pos_at(ts, ps, t + 1) - _pos_at(ts, ps, t)) for t in range(ts[-1])), "distance differs")
    found_last = (_pos_at(ts, ps, ts[-1]) - _pos_at(ts, ps, ts[-2])) / (ts[-1] - ts[-2])
    check(found_last == v_last, "last velocity differs")
    avg = Fraction(dist, ts[-1])
    # part d is carried from part c, so it is stated to the places that any whole-number distance over this time needs
    dp_v, dp_s = places_needed(v_last), max(places_needed(avg), {4: 2, 5: 1, 10: 1, 20: 2}[ts[-1]])
    rise_last = ps[-1] - ps[-2]
    cands_a = [
        decimal_dist(abs(v_last), dp_v, "sr2b-velocity-sign-dropped")
        if v_last < 0
        else decimal_dist(-v_last, dp_v, "sr2b-velocity-sign-reversed"),
        decimal_dist(rise_last, dp_v, "sr2b-rise-not-divided"),
        decimal_dist(_vel(ts, ps, n - 2), dp_v, "sr2b-neighbouring-segment"),
        decimal_dist(Fraction(ps[-1] - ps[0], ts[-1]), dp_v, "sr2b-average-velocity-for-segment-velocity"),
    ]
    cands_b, _ = _return_distractors(ts, ps, r)
    net = abs(ps[-1] - ps[0])
    out_leg = max(abs(p - ps[0]) for p in ps)
    cands_c = [
        int_dist(net, "sr2b-displacement-for-distance"),
        int_dist(dist - out_leg, "sr2b-return-leg-lost"),
        int_dist(2 * out_leg, "sr2b-out-and-back-only"),
        int_dist(dist + abs(ps[0]), "sr2b-start-position-counted"),
    ]
    cands_c = [c for c in cands_c if int(c.text) != dist and int(c.text) > 0]
    cands_d = [
        decimal_dist(Fraction(ps[-1] - ps[0], ts[-1]), dp_s, "sr2b-average-velocity-for-average-speed"),
        decimal_dist(dist, dp_s, "sr2b-distance-not-divided"),
        decimal_dist(Fraction(dist, n), dp_s, "sr2b-distance-over-segments"),
        decimal_dist(Fraction(net, ts[-1]), dp_s, "sr2b-average-velocity-for-average-speed"),
    ]
    stem = f"{ctx['intro']} The journey lasts from t = 0 to t = {ts[-1]} {ctx['tword']}."
    fig = _figure(ts, ps, ctx)
    return dict(
        text=stem,
        figure=fig,
        parts=[
            dict(
                prompt=f"Find the velocity during the last segment of the journey, in {ctx['vel']}.",
                marks=1,
                sensible_range=(-100, 100),
                working=[
                    f"Gradient of the last segment = ({ps[-1]} - ({ps[-2]})) ÷ ({ts[-1]} - {ts[-2]}) = {number_fields(v_last, dp_v)['answer_text']}."
                ],
                distractors=pick(
                    number_fields(v_last, dp_v)["answer_text"],
                    [c for c in cands_a if c.text != number_fields(v_last, dp_v)["answer_text"]],
                ),
                **number_fields(v_last, dp_v),
            ),
            dict(
                prompt=f"At what time does it first return to its starting position? Give the time in {ctx['tword']}.",
                marks=2,
                sensible_range=(0, ts[-1]),
                working=[f"It starts at {ps[0]} and is first back at {ps[0]} when t = {r}."],
                distractors=pick(str(r), cands_b),
                **number_fields(r, 0),
            ),
            dict(
                prompt=f"Find the total distance travelled, in {ctx['dword']}.",
                marks=2,
                sensible_range=(0, None),
                working=[" + ".join(str(abs(ps[i + 1] - ps[i])) for i in range(n)) + f" = {dist}."],
                distractors=pick(str(dist), cands_c),
                **number_fields(dist, 0),
            ),
            dict(
                prompt=f"Hence find the average speed over the whole journey, in {ctx['vel']}.",
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_c/{ts[-1]}",
                working=[
                    f"Average speed = distance ÷ time = {dist} ÷ {ts[-1]} = {number_fields(avg, dp_s)['answer_text']}."
                ],
                distractors=pick(
                    number_fields(avg, dp_s)["answer_text"],
                    [c for c in cands_d if c.text != number_fields(avg, dp_s)["answer_text"]],
                ),
                **number_fields(avg, dp_s),
            ),
        ],
    )


# -- reader ---------------------------------------------------------------------------------------------------


def read_position_time(question) -> str:
    """Re-derive the key from the drawn segments and the wording alone (never from the stored answer)."""
    segs = question.figure["segments"]
    ts = [segs[0]["from"][0]] + [s["to"][0] for s in segs]
    ps = [segs[0]["from"][1]] + [s["to"][1] for s in segs]
    check(all(segs[i]["to"] == segs[i + 1]["from"] for i in range(len(segs) - 1)), "segments are not joined")
    if "average velocity" in question.text:
        return exact_decimal_string(Fraction(ps[-1] - ps[0], ts[-1] - ts[0]), 2)
    if "average speed" in question.text:
        return exact_decimal_string(Fraction(_distance(ts, ps), ts[-1] - ts[0]), 2)
    r = _first_return(ts, ps)
    check(r is not None, "no return found")
    return exact_decimal_string(r, 2)


figures.register_reader(TOPIC, read_position_time)
