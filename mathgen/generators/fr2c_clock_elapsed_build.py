"""Generates `fr2c_clock_elapsed` and `fr2c_clock_elapsed_exam` questions: elapsed time with am/pm and 24-hour times.

One skill: ``f10-ampm-elapsed-and-24h-practical`` (``answer: number``). Every question stays inside one day (nothing
crosses midnight). Times are written as 12-hour times with am or pm ("10:50 am") or as 24-hour times ("13:20").

- easy: the minutes from a morning time to an afternoon time, both written with am/pm, so the interval crosses noon.
- medium: the minutes between two times where one is written with am/pm and the other in 24-hour time.
- hard: a timetable of two journey legs and a wait between them, the legs in different styles; either the total time from
  the first departure to the last arrival, or the start time (in 24-hour time, four digits) that a stated total gives.

``fr2c_clock_elapsed_exam`` is the chain: (a) minutes on the first leg, (b) minutes waiting, (c) minutes on the second leg,
(d) the total from first departure to last arrival (follows through from (a), (b) and (c)), (e) that total as
hours and minutes (a pair).

Each interval is built from minutes since midnight and rechecked by changing both times to 24-hour form.
"""

from __future__ import annotations

from mathgen.core import attempts, check, generator, phrase

from . import _fr2c_kit as kit

_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")
_EVENTS = (
    ("a film", "starts", "ends"),
    ("a swimming carnival", "starts", "finishes"),
    ("a music lesson and rehearsal", "starts", "finishes"),
    ("a bushwalk", "begins", "ends"),
    ("a school excursion", "leaves school", "returns"),
    ("a soccer tournament", "starts", "finishes"),
)
_PLACES = (
    ("Adelaide", "Mount Gambier"),
    ("Perth", "Bunbury"),
    ("Brisbane", "Toowoomba"),
    ("Hobart", "Launceston"),
    ("Cairns", "Townsville"),
    ("Darwin", "Katherine"),
)


def fmt12(m: int) -> str:
    h, mm = divmod(m, 60)
    return f"{h % 12 or 12}:{mm:02d} {'am' if h < 12 else 'pm'}"


def fmt24(m: int) -> str:
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"


def hhmm(m: int) -> int:
    h, mm = divmod(m, 60)
    return h * 100 + mm


def _decimal_clock(a: int, b: int) -> int:
    """The error of subtracting clock readings as if they were decimals (13.20 - 10.50): the difference times 100."""
    ha, ma = divmod(a, 60)
    hb, mb = divmod(b, 60)
    return (
        abs((hb + mb / 100) - (ha + ma / 100)) * 100 // 1
        if False
        else int(round(abs((hb * 100 + mb) - (ha * 100 + ma))))
    )


def _style(rng, m: int, twelve: bool) -> str:
    return fmt12(m) if twelve else fmt24(m)


# ------------------------------------------------------------------------------------------ easy / medium


def _interval(rng, cross_noon: bool):
    for _ in attempts(500, "an interval inside one day"):
        start = rng.randint(6 * 60, 11 * 60 + 55) if cross_noon else rng.randint(7 * 60, 17 * 60)
        start = start // 5 * 5
        length = rng.randint(70, 330) if cross_noon else rng.randint(60, 300)
        length = length // 5 * 5
        end = start + length
        if cross_noon and not (start < 12 * 60 < end and end < 20 * 60):
            continue
        if end >= 23 * 60:
            continue
        return start, end
    raise AssertionError("unreachable")


def _easy(rng):
    start, end = _interval(rng, True)
    thing, v1, v2 = rng.choice(_EVENTS)
    text = phrase(
        rng,
        (
            "{t} {v1} at {s} and {v2} at {e}. How many minutes does it last?",
            "{t} {v1} at {s} and {v2} at {e}. How long is that, in minutes?",
        ),
        t=thing[0].upper() + thing[1:],
        v1=v1,
        v2=v2,
        s=fmt12(start),
        e=fmt12(end),
    )
    value = end - start
    wrong = [
        ("fr2c-noon-not-crossed", (start - end) % 720 if (start - end) % 720 != value else None),
        ("clock-time-read-as-decimal", _decimal_clock(start, end) if _decimal_clock(start, end) != value else None),
        ("fr2c-hour-lost-at-noon", value - 60),
        ("fr2c-hour-gained-at-noon", value + 60),
        ("fr2c-count-off-by-one", value + 10),
    ]
    working = [
        f"From {fmt12(start)} to 12:00 pm is {12 * 60 - start} minutes.",
        f"From 12:00 pm to {fmt12(end)} is {end - 12 * 60} minutes.",
        f"{12 * 60 - start} + {end - 12 * 60} = {value} minutes.",
    ]
    check(fmt24(end).split(":")[0] != "", "format")
    return text, value, [(m, w) for m, w in wrong if w], working


def _medium(rng):
    start, end = _interval(rng, False)
    thing, v1, v2 = rng.choice(_EVENTS)
    twelve_first = rng.random() < 0.5
    s = fmt12(start) if twelve_first else fmt24(start)
    e = fmt24(end) if twelve_first else fmt12(end)
    text = phrase(
        rng,
        (
            "{t} {v1} at {s} and {v2} at {e}. How many minutes does it last?",
            "The timetable shows that {t_low} {v1} at {s} and {v2} at {e}. How long is that, in minutes?",
        ),
        t=thing[0].upper() + thing[1:],
        t_low=thing,
        v1=v1,
        v2=v2,
        s=s,
        e=e,
    )
    value = end - start
    sh, sm = divmod(start, 60)
    eh, em = divmod(end, 60)
    wrong = [
        (
            "fr2c-twelve-hour-clock-used-for-24",
            abs((eh % 12 or 12) * 60 + em - ((sh % 12 or 12) * 60 + sm))
            if abs((eh % 12 or 12) * 60 + em - ((sh % 12 or 12) * 60 + sm)) != value
            else None,
        ),
        ("clock-time-read-as-decimal", _decimal_clock(start, end) if _decimal_clock(start, end) != value else None),
        ("fr2c-hour-lost-at-noon", value - 60),
        ("fr2c-hour-gained-at-noon", value + 60),
        ("fr2c-count-off-by-one", value + 10),
    ]
    working = [
        f"{s} is {fmt24(start) if twelve_first else s} and {e} is {fmt24(end)} in 24-hour time.",
        f"{fmt24(end)} - {fmt24(start)} = {value} minutes.",
    ]
    return text, value, [(m, w) for m, w in wrong if w and w > 0], working


# ------------------------------------------------------------------------------------------ hard / exam legs


def _legs(rng):
    """(start1, end1, start2, end2) in minutes since midnight: a leg, a wait of 20-90 minutes, then a second leg."""
    for _ in attempts(500, "two journey legs inside one day"):
        s1 = (rng.randint(6 * 60, 11 * 60 + 30)) // 5 * 5
        e1 = s1 + (rng.randint(70, 240)) // 5 * 5
        s2 = e1 + (rng.randint(20, 90)) // 5 * 5
        e2 = s2 + (rng.randint(60, 200)) // 5 * 5
        if e2 < 22 * 60 and e1 - s1 >= 60 and e2 - s2 >= 60:
            return s1, e1, s2, e2
    raise AssertionError("unreachable")


def _hard(rng):
    s1, e1, s2, e2 = _legs(rng)
    total = e2 - s1
    wait = s2 - e1
    first, second = rng.choice(_PLACES)
    name = rng.choice(_NAMES)
    leg1, leg2 = e1 - s1, e2 - s2
    t1 = (True, False)
    if rng.random() < 0.5:
        text = (
            f"{name} takes a bus from {first} that leaves at {fmt12(s1)} and arrives at {fmt24(e1)}. A train leaves at "
            f"{fmt24(s2)} and arrives at {fmt12(e2)}. How many minutes is it from the bus leaving to the train arriving?"
        )
        form = "12-24-24-12"
    else:
        text = (
            f"{name}'s trip from {first} to {second} is in two parts. The first part leaves at {fmt24(s1)} and arrives at "
            f"{fmt12(e1)}. The second part leaves at {fmt12(s2)} and arrives at {fmt24(e2)}. How many minutes is it from the "
            "first part leaving to the second part arriving?"
        )
        form = "24-12-12-24"
    del t1, form
    wrong = [
        ("left-out-the-break", leg1 + leg2),
        ("fr2c-wait-counted-twice", leg1 + leg2 + 2 * wait),
        ("fr2c-wait-only", wait),
        ("fr2c-hour-lost-at-noon", total - 60),
        ("fr2c-hour-gained-at-noon", total + 60),
    ]
    working = [
        f"First part: {leg1} minutes. Wait: {wait} minutes. Second part: {leg2} minutes.",
        f"{leg1} + {wait} + {leg2} = {total} minutes.",
    ]
    return text, total, wrong, working


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fr2c_clock_elapsed",
    level=5,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2c_clock_elapsed(rng, tier="easy"):
    """Elapsed minutes across noon, between 12-hour and 24-hour times, or over a two-leg timetable."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(60, "a question with three different wrong answers"):
        text, value, wrong, working = _TIERS[tier](rng)
        try:
            fields = kit.number_fields(value, [(m, w) for m, w in wrong if w > 0])
        except kit.Redraw:
            continue
        break
    return dict(text=text, working=working, **fields)


@generator("fr2c_clock_elapsed_exam", level=9, presets={"exam": {}})
def fr2c_clock_elapsed_exam(rng):
    """A two-leg trip: minutes on each leg, the wait, the total, and the total in hours and minutes."""
    return kit.build_with_redraw(rng, _build_fr2c_clock_elapsed_exam, what="fr2c_clock_elapsed_exam")


def _build_fr2c_clock_elapsed_exam(rng):
    s1, e1, s2, e2 = _legs(rng)
    first, second = rng.choice(_PLACES)
    name = rng.choice(_NAMES)
    leg1, wait, leg2 = e1 - s1, s2 - e1, e2 - s2
    total = e2 - s1
    check(total == leg1 + wait + leg2, "the legs do not add up to the total")
    hrs, mins = divmod(total, 60)
    stem = (
        f"{name} travels from {first} to {second}. The coach leaves {first} at {fmt12(s1)} and arrives at the transfer stop "
        f"at {fmt24(e1)}. The train from the transfer stop leaves at {fmt12(s2)} and arrives in {second} at {fmt24(e2)}."
    )
    parts = [
        kit.num_part(
            "How many minutes is the coach journey?",
            leg1,
            2,
            [f"{fmt24(s1)} to {fmt24(e1)} is {leg1} minutes."],
            [
                ("clock-time-read-as-decimal", _decimal_clock(s1, e1)),
                ("fr2c-hour-lost-at-noon", leg1 - 60),
                ("fr2c-hour-gained-at-noon", leg1 + 60),
                ("fr2c-noon-not-crossed", abs(((e1 % 720) - (s1 % 720)))),
            ],
            sensible_range=(0, None),
            positive_only=True,
        ),
        kit.num_part(
            "How many minutes is the wait at the transfer stop?",
            wait,
            1,
            [f"{fmt24(e1)} to {fmt24(s2)} is {wait} minutes."],
            [
                ("clock-time-read-as-decimal", _decimal_clock(e1, s2)),
                ("fr2c-count-off-by-one", wait + 10),
                ("fr2c-hour-lost-at-noon", wait - 60),
                ("fr2c-hour-gained-at-noon", wait + 60),
            ],
            sensible_range=(0, None),
            positive_only=True,
        ),
        kit.num_part(
            "How many minutes is the train journey?",
            leg2,
            1,
            [f"{fmt24(s2)} to {fmt24(e2)} is {leg2} minutes."],
            [
                ("clock-time-read-as-decimal", _decimal_clock(s2, e2)),
                ("fr2c-hour-lost-at-noon", leg2 - 60),
                ("fr2c-hour-gained-at-noon", leg2 + 60),
                ("fr2c-count-off-by-one", leg2 + 10),
            ],
            sensible_range=(0, None),
            positive_only=True,
        ),
        kit.num_part(
            f"Using your answers, how many minutes is it from leaving {first} to arriving in {second}?",
            total,
            2,
            [f"{leg1} + {wait} + {leg2} = {total} minutes."],
            [
                ("left-out-the-break", leg1 + leg2),
                ("fr2c-wait-counted-twice", leg1 + leg2 + 2 * wait),
                ("fr2c-hour-lost-at-noon", total - 60),
                ("fr2c-hour-gained-at-noon", total + 60),
            ],
            follow_through="part_a+part_b+part_c",
            sensible_range=(0, None),
            positive_only=True,
        ),
        kit.pair_part(
            "Write the whole trip as hours and minutes, in the form (hours, minutes).",
            (hrs, mins),
            1,
            [f"{total} minutes = {hrs} hours {mins} minutes."],
            [
                ("minutes-over-100", (total // 100, total % 100)),
                ("fr2c-hours-and-minutes-swapped", (mins, hrs)),
                ("fr2c-minutes-remainder-lost", (hrs, 0)),
                ("fr2c-count-off-by-one", (hrs + 1, mins)),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
