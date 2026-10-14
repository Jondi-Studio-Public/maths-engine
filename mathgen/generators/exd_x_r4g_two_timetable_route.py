"""Exam chain for a journey on two timetables with a change at an interchange.

(a) the time the first service she catches reaches the interchange (four-digit 24-hour)    number, 1 mark
(b) the minutes she waits at the interchange for the connection she takes                   number, 2 marks
(c) the time she arrives at the destination (four-digit 24-hour)                            number, 2 marks
(d) the minutes the whole journey takes, from reaching the origin (follow-through (b))      number, 2 marks

She catches the first service she can and then the connection that arrives earliest. Each timetable is parsed back
out of the stem text and the route is searched again.
"""

from __future__ import annotations

from ..core import check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4g_kit import hhmm, number_part, retry
from .r4g_two_timetable_route_build import _draw, _stem, chosen, fmt, parse_timetable


@distinct_part_answers
def _build(rng):
    w = _draw(rng)
    o, c, d = w["places"]
    m1, m2 = w["modes"]
    stem = _stem(w)
    check(parse_timetable(stem, m1, o, c) == w["first"] and parse_timetable(stem, m2, c, d) == w["second"], "re-read")
    nxt, conn, first_dep = chosen(w)
    tr, ready = w["transfer"], w["ready"]
    wait = conn[0] - nxt[1]
    total = conn[1] - ready
    missed = next((s for s in w["second"] if nxt[1] <= s[0] < nxt[1] + tr), None)
    before = max((s for s in w["first"] if s[0] < ready), key=lambda s: s[0], default=None)
    text = (
        f"{stem} A traveller reaches {o} at {fmt(ready)}. She catches the first {m1} she can, then takes the "
        f"connection at {c} that gets her to {d} earliest. Give each time in 24-hour form as a four-digit number, such as 1342."
    )
    later = [s for s in w["first"] if s[0] > nxt[0]]
    parts = [
        number_part(
            f"At what time does the {m1} she catches reach {c}?",
            hhmm(nxt[1]),
            0,
            1,
            [f"The first {m1} she can catch leaves {o} at {fmt(nxt[0])} and reaches {c} at {fmt(nxt[1])}."],
            [
                (hhmm(nxt[0]), "r4g-departure-for-arrival"),
                (hhmm(before[1]) if before else hhmm(nxt[1] - 10), "r4g-service-just-missed"),
                (hhmm(later[0][1]), "r4g-later-service-chosen"),
            ],
            sensible_range=(1000, 2359),
        ),
        number_part(
            f"For how many minutes does she wait at {c} before the connection leaves?",
            wait,
            0,
            2,
            [
                f"She arrives at {fmt(nxt[1])}, so the connection must leave at {fmt(nxt[1] + tr)} or later: the {fmt(conn[0])} {m2}.",
                f"{fmt(conn[0])} is {wait} minutes after {fmt(nxt[1])}.",
            ],
            [
                (wait - tr if wait > tr else wait + tr, "r4g-change-time-for-wait"),
                (missed[0] - nxt[1] if missed else tr, "r4g-change-time-ignored"),
                (conn[0] - nxt[0], "r4g-departure-to-departure"),
                (conn[1] - nxt[1], "r4g-wait-and-ride-together"),
            ],
        ),
        number_part(
            f"At what time does she arrive at {d}?",
            hhmm(conn[1]),
            0,
            2,
            [f"The {m2} that leaves at {fmt(conn[0])} arrives at {fmt(conn[1])}."],
            [
                (hhmm(conn[0]), "r4g-departure-for-arrival"),
                (hhmm(next(s for s in w["second"] if s[0] >= nxt[1])[1]), "r4g-change-time-ignored"),
                (hhmm(first_dep[1]), "r4g-first-departing-not-earliest-arriving"),
                (hhmm(nxt[1]), "r4g-interchange-time-for-final"),
            ],
            sensible_range=(1000, 2359),
        ),
        number_part(
            f"How many minutes does the whole journey take, from reaching {o} until arriving at {d}? Use your waiting time from (b).",
            total,
            0,
            2,
            [
                f"Waiting at {o}: {nxt[0] - ready} min; riding to {c}: {nxt[1] - nxt[0]} min; waiting at {c}: {wait} min; riding to {d}: {conn[1] - conn[0]} min. Total {total} minutes."
            ],
            [
                (conn[1] - nxt[0], "r4g-wait-at-start-left-out"),
                ((nxt[1] - nxt[0]) + (conn[1] - conn[0]), "r4g-riding-time-only"),
                (hhmm(conn[1]) - hhmm(ready), "r4g-clock-numbers-subtracted"),
                (total + tr, "r4g-change-time-for-wait"),
            ],
            follow_through=f"part_b + {(nxt[1] - ready) + (conn[1] - conn[0])}",
        ),
    ]
    return dict(text=text, parts=parts)


@generator("exd_x_r4g_two_timetable_route", level=11, presets={"exam": {}})
def exd_x_r4g_two_timetable_route(rng):
    return retry(_build, rng)
