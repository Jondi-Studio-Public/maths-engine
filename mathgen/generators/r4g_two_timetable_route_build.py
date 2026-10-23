"""Planning a journey from two timetables: `r4g_two_timetable_route` (easy, medium, hard).

``two-timetable-route-planning`` (``senior-mathematics``, answer ``number``). Two services run end to end: the first
from the origin to an interchange, the second from the interchange to the destination. Both timetables are written
as text in the question (`leaves 1005 arrives 1038; ...`, 24-hour times, all between 1000 and 1959), and the
question states the shortest time to change at the interchange. A connection needs the second service to leave at
least that long after the first arrives.

* easy: the earliest arrival at the destination for a traveller who reaches the origin at a stated time (a
  four-digit 24-hour number such as 1342).
* medium: the whole journey time in minutes from reaching the origin to arriving, waiting included.
* hard: the latest departure from the origin that still arrives by a stated deadline. The timetables are drawn so
  that a quicker, later first service beats the first one that can be caught.

Every key is found by a brute-force search over every pair of services, re-read from the text of the question.
"""

from __future__ import annotations

import re

from ..core import attempts, check, generator
from ._r4g_kit import clock_text, hhmm, minutes, single_number, retry

PLACES = [
    ("Marlow", "Bexley", "Kingsford"),
    ("Ashton", "Carlow", "Eastvale"),
    ("Hartley", "Pemberton", "Yarrow"),
    ("Ridgeway", "Thornbury", "Lakeside"),
    ("Corrimal", "Wattle Flat", "Bay Heads"),
]
MODES = [("train", "bus"), ("train", "ferry"), ("bus", "ferry"), ("tram", "bus")]
TEN = 600  # 10:00 in minutes


def fmt(t) -> str:
    return clock_text(t)


def draw_services(rng, start, count, gaps, durations):
    """[(departure, arrival)] in minutes, strictly increasing departures."""
    out, t = [], start
    for _ in range(count):
        out.append((t, t + rng.choice(durations)))
        t += rng.randint(*gaps)
    return out


def timetable_text(mode, a, b, services):
    rows = "; ".join(f"leaves {fmt(d)} arrives {fmt(r)}" for d, r in services)
    return f"{mode.capitalize()} timetable, {a} to {b}: {rows}."


def parse_timetable(text, mode, a, b):
    head = f"{mode.capitalize()} timetable, {a} to {b}: "
    body = text.split(head, 1)[1].split(". ", 1)[0].rstrip(".")
    services = []
    for row in body.split("; "):
        m = re.fullmatch(r"leaves (\d{4}) arrives (\d{4})", row)
        services.append((minutes(int(m.group(1))), minutes(int(m.group(2)))))
    return services


def routes(first, second, transfer, ready):
    """[(first departure, interchange arrival, second departure, final arrival)] for every legal pair."""
    out = []
    for d1, a1 in first:
        if d1 < ready:
            continue
        for d2, a2 in second:
            if d2 >= a1 + transfer:
                out.append((d1, a1, d2, a2))
    return out


def earliest(first, second, transfer, ready):
    return min(routes(first, second, transfer, ready), key=lambda r: (r[3], r[0]))


def latest_departure(first, second, transfer, deadline):
    legal = [r for r in routes(first, second, transfer, 0) if r[3] <= deadline]
    return max(legal, key=lambda r: (r[0], -r[3]))


def _draw(rng, *, quick_later=False):
    # quick_later: the best overall route does not start with the first service that can be caught
    for _ in attempts(3000, "timetables with a connection trap"):
        o, c, d = rng.choice(PLACES)
        m1, m2 = rng.choice(MODES)
        transfer = rng.choice((5, 8, 10, 12))
        first = draw_services(rng, TEN + rng.randint(0, 20), rng.randint(5, 6), (14, 34), (26, 34, 44))
        second = draw_services(rng, first[0][1] + rng.randint(0, 12), rng.randint(6, 7), (10, 30), (18, 24, 32))
        if second[-1][1] > 1959 or first[-1][1] > 1900:
            continue
        i = rng.randint(1, len(first) - 3)
        ready = first[i][0] - rng.randint(1, 9)  # just missed service i - 1
        if i and ready <= first[i - 1][0]:
            continue
        legal = routes(first, second, transfer, ready)
        if not legal:
            continue
        best = min(legal, key=lambda r: (r[3], r[0]))
        next_first = next(s for s in first if s[0] >= ready)
        # trap: the first service that can be caught has a connection missed by less than the change time
        trap = any(next_first[1] <= s[0] < next_first[1] + transfer for s in second)
        if not trap:
            continue
        # the traveller catches next_first, so a connection must still be possible from it
        if not any(s[0] >= next_first[1] + transfer for s in second):
            continue
        if quick_later and best[0] == next_first[0]:
            continue
        return dict(places=(o, c, d), modes=(m1, m2), transfer=transfer, first=first, second=second, ready=ready)
    raise AssertionError("unreachable")


def _stem(w):
    o, c, d = w["places"]
    m1, m2 = w["modes"]
    return (
        f"{timetable_text(m1, o, c, w['first'])} {timetable_text(m2, c, d, w['second'])} "
        f"Allow at least {w['transfer']} minutes to change at {c}: the second service must leave at least that long after the first arrives."
    )


def chosen(w):
    """(first service caught, connection taken, first departing connection): the traveller catches the first service
    she can, then the connection that arrives earliest."""
    first, second, tr, ready = w["first"], w["second"], w["transfer"], w["ready"]
    nxt = next(s for s in first if s[0] >= ready)
    legal = [s for s in second if s[0] >= nxt[1] + tr]
    best = min(legal, key=lambda s: (s[1], s[0]))
    return nxt, best, legal[0]


def _easy(rng):
    w = _draw(rng)
    o, c, d = w["places"]
    m1, m2 = w["modes"]
    # re-read both timetables from the stem text and search again
    stem = _stem(w)
    first, second = parse_timetable(stem, m1, o, c), parse_timetable(stem, m2, c, d)
    check(first == w["first"] and second == w["second"], "timetable re-read differs")
    nxt, conn, first_dep = chosen(w)
    best = (nxt[0], nxt[1], conn[0], conn[1])
    no_transfer = min((s for s in second if s[0] >= nxt[1]), key=lambda s: (s[1], s[0]))
    text = (
        f"{stem} A traveller reaches {o} at {fmt(w['ready'])} and catches the first {m1} she can, then the best "
        f"connection at {c}. At what time does she arrive at {d}? Give the time in 24-hour form as a four-digit number, such as 1342."
    )
    slips = [
        (hhmm(no_transfer[1]), "r4g-change-time-ignored"),
        (hhmm(first_dep[1]), "r4g-first-departing-not-earliest-arriving"),
        (hhmm(best[2]), "r4g-departure-for-arrival"),
        (hhmm(nxt[1]), "r4g-interchange-time-for-final"),
    ]
    working = [
        f"The first {m1} she can catch leaves {o} at {fmt(nxt[0])} and reaches {c} at {fmt(nxt[1])}.",
        f"The best connection leaves {c} at {fmt(best[2])} and arrives at {d} at {fmt(best[3])}.",
    ]
    return single_number(text, hhmm(best[3]), 0, working, [(v, t) for v, t in slips])


def _medium(rng):
    w = _draw(rng)
    o, c, d = w["places"]
    m1, m2 = w["modes"]
    nxt, conn, first_dep = chosen(w)
    best = (nxt[0], nxt[1], conn[0], conn[1])
    no_transfer = min((s for s in w["second"] if s[0] >= nxt[1]), key=lambda s: (s[1], s[0]))
    total = best[3] - w["ready"]
    text = (
        f"{_stem(w)} A traveller reaches {o} at {fmt(w['ready'])}, takes the first {m1} she can catch and the "
        f"best connection at {c}, so she arrives at {d} as early as possible. How many minutes does the whole journey take, from reaching {o} until arriving at {d}?"
    )
    riding = (best[1] - best[0]) + (best[3] - best[2])
    slips = [
        (best[3] - best[0], "r4g-wait-at-start-left-out"),
        (riding, "r4g-riding-time-only"),
        (no_transfer[1] - w["ready"], "r4g-change-time-ignored"),
        (first_dep[1] - w["ready"], "r4g-first-departing-not-earliest-arriving"),
        (best[3] - best[2] + (best[1] - best[0]) + w["transfer"], "r4g-change-time-for-wait"),
    ]
    working = [
        f"She catches the {fmt(best[0])} {m1}, reaching {c} at {fmt(best[1])}, then the {fmt(best[2])} {m2}, arriving at {fmt(best[3])}.",
        f"From {fmt(w['ready'])} to {fmt(best[3])} is {total} minutes.",
    ]
    return single_number(text, total, 0, working, [(v, t) for v, t in slips if v > 0])


def _hard(rng):
    for _ in attempts(500, "a deadline with a quicker later service"):
        w = _draw(rng)
        first, second, tr = w["first"], w["second"], w["transfer"]
        deadline = rng.choice(sorted({r[3] for r in routes(first, second, tr, 0)}))
        deadline += rng.randint(0, 6)
        if deadline > 1959:
            continue
        pick = latest_departure(first, second, tr, deadline)
        alt = [r for r in routes(first, second, 0, 0) if r[3] <= deadline]
        if not alt or latest_departure(first, second, 0, deadline)[0] == pick[0]:
            continue
        break
    o, c, d = w["places"]
    m1, m2 = w["modes"]
    stem = _stem(w)
    check(parse_timetable(stem, m1, o, c) == first and parse_timetable(stem, m2, c, d) == second, "re-read differs")
    last_second = max((s for s in second if s[1] <= deadline), key=lambda s: s[0])
    prev = max((s for s in first if s[1] <= last_second[0] - tr), key=lambda s: s[0])
    later = [s for s in first if s[0] > pick[0]]
    text = (
        f"{stem} A traveller must arrive at {d} by {fmt(deadline)}. What is the latest time she can leave {o} on the "
        f"{m1}? Give the time in 24-hour form as a four-digit number, such as 1342."
    )
    slips = [
        (hhmm(latest_departure(first, second, 0, deadline)[0]), "r4g-change-time-ignored"),
        (hhmm(max(s for s in first if s[1] <= deadline)[0]), "r4g-first-service-ignores-connection"),
        (hhmm(pick[1]), "r4g-arrival-for-departure"),
        (hhmm(prev[0]), "r4g-latest-connection-only"),
        (hhmm(later[0][0]) if later else hhmm(pick[0] + 20), "r4g-later-service-chosen"),
    ]
    working = [
        f"Work backwards from {fmt(deadline)}: the {m2} that gets her to {d} in time and can still be reached leaves {c} at {fmt(pick[2])} (arrives {fmt(pick[3])}).",
        f"The {m1} must reach {c} by {fmt(pick[2] - w['transfer'])}; the latest one that does leaves {o} at {fmt(pick[0])}.",
    ]
    return single_number(text, hhmm(pick[0]), 0, working, slips)


_TIER = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "r4g_two_timetable_route",
    level=11,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def r4g_two_timetable_route(rng, tier="easy"):
    return retry(_TIER[tier], rng)
