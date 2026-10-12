"""Exam-tier sources for four primary time skills: multi-step problems in an Australian setting.

Each topic returns a short chain of parts that share one stem, the way a national-assessment paper
sets a connected question: a first quantity is found from the stem, and the later parts use it.

- ``exd_x_pr2_trip_times`` (``elapsed-time``): the minutes from leaving to arriving, the minutes
  actually moving once a break is removed, then the length of a return trip with a different break.
- ``exd_x_pr2_timetable_connection`` (``f10-timetable-itinerary-duration``): a two-stage timetable;
  the wait at the changeover, the whole journey, then the time spent travelling.
- ``exd_x_pr2_calendar_span`` (``f10-describe-calendar-duration``): three dates in a year that is
  not a leap year; the days between the first pair, the second pair, then the complete weeks overall.
- ``exd_x_pr2_unit_totals`` (``f10-time-units-and-nearest-minute``): a compound duration converted
  to the smaller unit, scaled by a count of identical items, then converted back to the larger unit
  correct to 1 decimal place.

Every question is built backwards from the quantities it asks for, so each key is exact. Parts (b) and
(c) carry ``follow_through``, so a learner who slips in an early part still earns the later marks
for using their own answer correctly.
"""

from __future__ import annotations

import functools
import random
import re
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_NAMES = ["Priya", "Tom", "Mia", "Liam", "Aisha", "Noah", "Chloe", "Jack", "Zara", "Ethan"]


def _key_not_in_prompt(build: Callable[..., dict]) -> Callable[..., dict]:
    """Redraw (within a budget) until no part's answer appears as a number in the stem or any part prompt.

    A learner who sees the key printed in the question cannot tell it from a given value, and the grader
    reads the number nearest the end of a reply; clock minutes such as the 40 in "9:40 am" count as numbers."""

    @functools.wraps(build)
    def wrapper(rng: random.Random, tier: str = "exam") -> dict:
        for _ in attempts(200, "a question whose keys are not printed in its prompt"):
            try:
                question = build(rng, tier)
            except VerificationError:
                continue  # a draw with fewer than two usable distractors (short legs): redraw
            shown = " ".join([question["text"], *(part["prompt"] for part in question["parts"])])
            numbers = set(re.findall(r"\d+(?:\.\d+)?", shown))
            if not any(part["answer_text"] in numbers for part in question["parts"]):
                return question
        raise AssertionError("unreachable")  # attempts() raises first

    return wrapper


def _clock(minutes: int) -> str:
    """Minutes after midnight as a 12-hour clock time such as "8:50 am" or "12:05 pm"."""
    h24, m = divmod(minutes, 60)
    suffix = "am" if h24 < 12 else "pm"
    return f"{h24 % 12 or 12}:{m:02d} {suffix}"


def _options(
    key: Fraction,
    slips: Sequence[tuple[Fraction | int, str | None]],
    fmt: Callable[[Fraction], str],
    n: int = 3,
) -> list[Distractor]:
    """Tagged distractors for ``key`` from ``(value, catalogue id)`` slips, in order of likelihood.

    A slip is dropped when it is not positive or lands within the grader's tolerance of the key, and
    each remaining value is shown once. The value rides on the distractor, so a typed reply is matched
    by meaning."""
    key = Fraction(key)
    candidates = []
    seen = {key}
    for value, tag in slips:
        value = Fraction(value)
        if value <= 0 or any(abs(value - other) <= GRADER_TOLERANCE for other in seen):
            continue
        seen.add(value)
        exact = int(value) if value.denominator == 1 else value
        candidates.append(Distractor(fmt(value), tag, exact))
    kept = pick_distractors(fmt(key), candidates, n, tolerance=GRADER_TOLERANCE)
    check(len(kept) >= 2, "fewer than two usable distractors")
    return kept


def _whole(value: Fraction) -> str:
    return str(int(value))


def _int_part(prompt: str, answer: int, *, marks: int, distractors, follow_through=None, working=None) -> dict:
    """A part whose answer is a whole number of minutes, days, degrees or the like."""
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=latex_answer(str(answer)),
        marks=marks,
        sensible_range=(0, None),
        working=working or [],
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


# ---------------------------------------------------------------------------------------------------
# elapsed-time: a road trip with a break
# ---------------------------------------------------------------------------------------------------

# (from, to, shortest and longest plausible trip in minutes, a road trip with a stop)
_ROUTES = [
    ("Melbourne", "Albury", 200, 270),
    ("Sydney", "Canberra", 170, 240),
    ("Brisbane", "Byron Bay", 140, 210),
    ("Adelaide", "Mount Gambier", 270, 340),
    ("Perth", "Albany", 270, 340),
    ("Hobart", "Launceston", 140, 210),
    ("Sydney", "Coffs Harbour", 330, 400),
    ("Melbourne", "Mildura", 330, 400),
    ("Darwin", "Katherine", 200, 270),
    ("Canberra", "Wagga Wagga", 150, 220),
]

_TRIP_STEMS = [
    "{who}'s family drives from {a} to {b}. They leave at {t1}, stop for a {brk}-minute break along the way and "
    "arrive at {t2}.",
    "A coach tour leaves {a} at {t1} bound for {b}. It makes one {brk}-minute rest stop and pulls in to {b} at {t2}.",
    "A school minibus takes a netball team from {a} to {b}. It departs at {t1}, waits {brk} minutes at a roadhouse "
    "and reaches {b} at {t2}.",
]


@generator("exd_x_pr2_trip_times", level=6, presets={"exam": dict(tier="exam")})
@_key_not_in_prompt
def exd_x_pr2_trip_times(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: minutes elapsed, minutes moving once a break is removed, then a return trip."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a_place, b_place, lo, hi = rng.choice(_ROUTES)
    total = rng.choice([t for t in range(lo, hi + 1, 5) if t % 60 != 0])  # minutes must not line up
    brk = rng.randrange(15, 46, 5)
    brk2 = rng.choice([b for b in range(10, 41, 5) if b != brk])
    start = rng.randrange(6 * 60, 13 * 60, 5)
    end = start + total
    moving = total - brk
    back = moving + brk2
    check(moving >= 60 and end < 24 * 60, "trip does not fit the day")

    stem = rng.choice(_TRIP_STEMS).format(
        who=rng.choice(_NAMES), a=a_place, b=b_place, t1=_clock(start), t2=_clock(end), brk=brk
    )

    # Independent route to (a): clock hours and minutes, borrowing an hour when the minutes run short.
    h1, m1 = divmod(start, 60)
    h2, m2 = divmod(end, 60)
    dh, dm = h2 - h1, m2 - m1
    if dm < 0:
        dh, dm = dh - 1, dm + 60
    check(dh * 60 + dm == total, "elapsed time disagrees with the clock-face route")

    h_only = (h2 - h1) * 60
    a_slips = [
        (abs(m2 - m1), "minute-parts-only-subtraction"),
        (h_only + abs(m2 - m1), "smaller-minus-larger-minutes"),
        (h_only, "ignored-minute-parts"),
        (dh * 100 + dm, "base-100-time"),
    ]
    b_slips = [(total, "waiting-counted-as-travel"), (total + brk, "waiting-added-to-elapsed")]
    c_slips = [
        (moving + brk, "waiting-counted-twice"),
        (moving, "forgot-stop-time"),
        (total + brk2, "waiting-counted-twice"),
        (moving - brk2, "waiting-subtracted-not-added"),
    ]
    # `(moving + brk)` and `(total + brk2)` both need one id each; the second keeps its own value.
    return dict(
        text=stem,
        latex=f"plain:{stem}",
        parts=[
            _int_part(
                f"How many minutes pass between leaving {a_place} and arriving in {b_place}?",
                total,
                marks=2,
                distractors=_options(Fraction(total), a_slips, _whole),
                working=[f"From {_clock(start)} to {_clock(end)} is {dh} h {dm} min = {total} minutes."],
            ),
            _int_part(
                "How many of those minutes is the vehicle moving, not stopped?",
                moving,
                marks=1,
                follow_through=f"part_a - {brk}",
                distractors=_options(Fraction(moving), b_slips, _whole),
                working=[f"{total} - {brk} = {moving} minutes."],
            ),
            _int_part(
                f"The return trip has the same moving time but a {brk2}-minute stop. "
                "How many minutes does the return trip take altogether?",
                back,
                marks=2,
                follow_through=f"part_b + {brk2}",
                distractors=_options(Fraction(back), c_slips, _whole),
                working=[f"{moving} + {brk2} = {back} minutes."],
            ),
        ],
    )


# ---------------------------------------------------------------------------------------------------
# f10-timetable-itinerary-duration: a two-stage journey
# ---------------------------------------------------------------------------------------------------

_STAGES = [
    # (from, change point, to, first vehicle, second vehicle, first-leg minutes (lo, hi), second-leg minutes (lo, hi))
    ("Brisbane Roma Street", "Nambour", "Noosa", "train", "bus", (90, 110), (35, 55)),
    ("Sydney Central", "Wollongong", "Kiama", "train", "bus", (80, 100), (25, 45)),
    ("Melbourne Southern Cross", "Geelong", "Torquay", "train", "bus", (55, 75), (30, 50)),
    ("Perth", "Mandurah", "Bunbury", "train", "coach", (50, 60), (60, 90)),
    ("Adelaide", "Gawler", "Tanunda", "train", "bus", (50, 65), (20, 35)),
    ("Sydney Central", "Katoomba", "Blackheath", "train", "bus", (105, 120), (15, 25)),
    ("Hobart", "Kingston", "Huonville", "first bus", "second bus", (20, 35), (25, 40)),
    ("Cairns", "Port Douglas", "Daintree", "coach", "ferry", (65, 85), (25, 45)),
]

_TIMETABLE_STEMS = [
    "{who} plans a trip from {a} to {c} in two stages. Timetable: the {m1} leaves {a} at {t1} and arrives at {b} at "
    "{t2}. The {m2} leaves {b} at {t3} and arrives at {c} at {t4}.",
    "A journey from {a} to {c} uses two services. The {m1} departs {a} at {t1} and reaches {b} at {t2}. {who} "
    "changes there to the {m2}, which departs at {t3} and arrives at {c} at {t4}.",
]


@generator("exd_x_pr2_timetable_connection", level=6, presets={"exam": dict(tier="exam")})
@_key_not_in_prompt
def exd_x_pr2_timetable_connection(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the wait at a changeover, the whole journey, then the time spent travelling."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a, b, c, m1, m2, (lo1, hi1), (lo2, hi2) = rng.choice(_STAGES)
    leg1 = rng.randrange(lo1, hi1 + 1, 5)
    wait = rng.randrange(10, 56, 5)
    leg2 = rng.randrange(lo2, hi2 + 1, 5)
    t1 = rng.randrange(5 * 60, 15 * 60, 5)
    t2, t3 = t1 + leg1, t1 + leg1 + wait
    t4 = t3 + leg2
    whole = t4 - t1
    travelling = whole - wait
    check(t4 < 24 * 60 and travelling == leg1 + leg2, "journey does not fit the day")

    stem = rng.choice(_TIMETABLE_STEMS).format(
        who=rng.choice(_NAMES),
        a=a,
        b=b,
        c=c,
        m1=m1,
        m2=m2,
        t1=_clock(t1),
        t2=_clock(t2),
        t3=_clock(t3),
        t4=_clock(t4),
    )

    def mins(t: int) -> int:
        return t % 60

    a_slips = [
        (abs(mins(t3) - mins(t2)), "minute-parts-only-subtraction"),
        (leg2, "wrong-row-in-timetable"),
        (leg1, "wrong-row-in-timetable"),
        (t3 - t1, None),
    ]
    b_slips = [
        (leg1 + leg2, "forgot-stop-time"),
        (abs(mins(t4) - mins(t1)), "minute-parts-only-subtraction"),
        (t4 - t2, None),
    ]
    c_slips = [
        (whole, "waiting-counted-as-travel"),
        (whole + wait, "waiting-added-to-elapsed"),
        (leg1, None),
        (leg2, None),
    ]
    return dict(
        text=stem,
        latex=f"plain:{stem}",
        parts=[
            _int_part(
                f"How many minutes does the traveller wait at {b} for the {m2}?",
                wait,
                marks=1,
                distractors=_options(Fraction(wait), a_slips, _whole),
                working=[f"From {_clock(t2)} to {_clock(t3)} is {wait} minutes."],
            ),
            _int_part(
                f"How many minutes pass from leaving {a} to arriving at {c}?",
                whole,
                marks=2,
                distractors=_options(Fraction(whole), b_slips, _whole),
                working=[f"From {_clock(t1)} to {_clock(t4)} is {whole} minutes."],
            ),
            _int_part(
                f"Using your answers to (a) and (b), for how many minutes is the traveller actually on the {m1} "
                f"or the {m2}?"
                if m1 != m2
                else "Using your answers to (a) and (b), for how many minutes is the traveller actually on a bus?",
                travelling,
                marks=1,
                follow_through="part_b - part_a",
                distractors=_options(Fraction(travelling), c_slips, _whole),
                working=[f"{whole} - {wait} = {travelling} minutes."],
            ),
        ],
    )


# ---------------------------------------------------------------------------------------------------
# f10-describe-calendar-duration: three dates, days then complete weeks
# ---------------------------------------------------------------------------------------------------

_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # a year that is not a leap year
_YEAR = 2023  # only used to turn day numbers into dates; never shown

_EVENTS = [
    ("swimming carnival", "Entries for the {event} open on {d1} and close on {d2}. The {event} is held on {d3}."),
    ("school camp", "Payments for the {event} are due from {d1} to {d2}. The {event} starts on {d3}."),
    ("science fair", "Students may register projects for the {event} from {d1} until {d2}. The {event} is on {d3}."),
    ("athletics carnival", "Sign-up for the {event} opens on {d1} and shuts on {d2}. The {event} takes place on {d3}."),
]


def _date_text(ordinal: int) -> tuple[str, int]:
    """(``"25 January"``, month index) for a day number in a year that is not a leap year (1 is 1 January)."""
    d = date(_YEAR, 1, 1) + timedelta(days=ordinal - 1)
    return f"{d.day} {_MONTHS[d.month - 1]}", d.month - 1


@generator("exd_x_pr2_calendar_span", level=6, presets={"exam": dict(tier="exam")})
@_key_not_in_prompt
def exd_x_pr2_calendar_span(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: days between two pairs of dates, then the complete weeks across all three."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "three dates in different months with a part week left over"):
        o1 = rng.randint(1, 250)
        days_a = rng.randint(20, 75)
        days_b = rng.randint(20, 85)
        o3 = o1 + days_a + days_b
        if o3 > 365 or (days_a + days_b) % 7 == 0:
            continue
        o2 = o1 + days_a
        months = [_date_text(o)[1] for o in (o1, o2, o3)]
        if months[0] != months[1] and months[1] != months[2]:
            break
    d1, d2, d3 = (_date_text(o)[0] for o in (o1, o2, o3))

    # Independent route: add up whole month lengths between the dates.
    def span(first: int, second: int) -> int:
        (m1, m2) = (_date_text(first)[1], _date_text(second)[1])
        day1 = first - sum(_MONTH_DAYS[:m1])
        day2 = second - sum(_MONTH_DAYS[:m2])
        return sum(_MONTH_DAYS[m1:m2]) + day2 - day1

    check(span(o1, o2) == days_a and span(o2, o3) == days_b, "day counts disagree with month-length route")
    total = days_a + days_b
    weeks = total // 7
    event, template = rng.choice(_EVENTS)
    stem = template.format(event=event, d1=d1, d2=d2, d3=d3) + " The year is not a leap year."

    def day_slips(first: int, second: int, true_days: int) -> list:
        m1, m2 = _date_text(first)[1], _date_text(second)[1]
        day1 = first - sum(_MONTH_DAYS[:m1])
        day2 = second - sum(_MONTH_DAYS[:m2])
        return [
            (abs(day2 - day1), "ignored-month-boundary"),
            (true_days + 1, "inclusive-day-count"),
            (30 * (m2 - m1) + day2 - day1, "treats-every-month-as-30-days"),
            (true_days - 1, None),
            (true_days + 7, None),
        ]

    c_slips = [
        (weeks + 1, "rounded-up-complete-weeks"),
        (total, "forgot-to-convert-days-to-weeks"),
        (days_a // 7 + days_b // 7 + 1, None),
    ]
    return dict(
        text=stem,
        latex=f"plain:{stem}",
        parts=[
            _int_part(
                f"How many days after {d1} is {d2}?",
                days_a,
                marks=2,
                distractors=_options(Fraction(days_a), day_slips(o1, o2, days_a), _whole),
                working=[f"From {d1} to {d2} is {days_a} days."],
            ),
            _int_part(
                f"How many days after {d2} is the {event}?",
                days_b,
                marks=2,
                distractors=_options(Fraction(days_b), day_slips(o2, o3, days_b), _whole),
                working=[f"From {d2} to {d3} is {days_b} days."],
            ),
            _int_part(
                f"Using your answers to (a) and (b), how many complete weeks are there from {d1} to the {event}?",
                weeks,
                marks=2,
                follow_through="floor((part_a + part_b)/7)",
                distractors=_options(Fraction(weeks), c_slips, _whole),
                working=[f"{days_a} + {days_b} = {total} days; {total} ÷ 7 = {weeks} complete weeks."],
            ),
        ],
    )


# ---------------------------------------------------------------------------------------------------
# f10-time-units-and-nearest-minute: compound duration, scaled, converted back
# ---------------------------------------------------------------------------------------------------

# (big unit, small unit, small units per big unit, big range, small range, count range)
_UNIT_SETS = [
    ("hour", "minute", 60, (1, 3), (5, 55), (3, 8)),
    ("minute", "second", 60, (2, 5), (5, 55), (4, 9)),
    ("day", "hour", 24, (2, 5), (3, 21), (3, 6)),
]

_UNIT_STEMS = {
    ("hour", "minute"): [
        "Each episode of a wildlife documentary runs for {big} {bu} {small} {su}. A school screens {n} episodes back "
        "to back with no breaks.",
        "A film festival shows {n} short films one after another, each lasting {big} {bu} {small} {su}.",
    ],
    ("minute", "second"): [
        "Each song on a dance-class playlist lasts {big} {bu} {small} {su}. The teacher plays {n} songs one after "
        "another.",
        "A runner completes {n} laps of the school cross-country course, each lap taking {big} {bu} {small} {su}.",
    ],
    ("day", "hour"): [
        "A research ship's voyage from Hobart to Macquarie Island and back takes {big} {bu} {small} {su}. The ship "
        "makes {n} such voyages, one straight after the other.",
        "A truck driver's return run between Perth and Adelaide takes {big} {bu} {small} {su}. The driver makes {n} "
        "return runs in a row.",
    ],
}


def _plural(n: int, unit: str) -> str:
    return unit if n == 1 else unit + "s"


@generator("exd_x_pr2_unit_totals", level=6, presets={"exam": dict(tier="exam")})
@_key_not_in_prompt
def exd_x_pr2_unit_totals(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a compound time in the smaller unit, scaled by a count, then back to the larger unit."""
    check(tier == "exam", f"unknown tier {tier!r}")
    bu, su, factor, big_range, small_range, n_range = rng.choice(_UNIT_SETS)
    for _ in attempts(1000, "a total that rounds cleanly to one decimal place and not exactly"):
        big = rng.randint(*big_range)
        small = rng.randint(*small_range)
        n = rng.randint(*n_range)
        each = big * factor + small
        total = n * each
        exact = Fraction(total, factor)
        scaled = exact * 10
        # Not already a 1 d.p. number, and clear of the rounding boundary so every route rounds alike.
        if scaled.denominator != 1 and abs(scaled % 1 - Fraction(1, 2)) > Fraction(1, 20):
            break
    final = round_half_up(exact, 1)
    check(abs(final - exact) <= Fraction(1, 20), "final answer is not within half a tenth of the exact value")

    stem = rng.choice(_UNIT_STEMS[(bu, su)]).format(
        big=big, bu=_plural(big, bu), small=small, su=_plural(small, su), n=n
    )
    a_prompt = f"How long is each one, in {su}s?"
    b_prompt = f"What is the total time for all of them, in {su}s?"
    c_prompt = f"Using your answer to (b), write the total time in {bu}s, correct to 1 decimal place."

    a_slips = [
        (big * 100 + small, "base-100-time"),
        (big * factor, "dropped-smaller-unit"),
        (big + small, "added-unconverted-units"),
    ]
    b_slips = [
        (each, "forgot-to-multiply-by-count"),
        (n * big * factor, "dropped-smaller-unit"),
        (n * (big * 100 + small), "base-100-time"),
    ]
    quotient, remainder = divmod(total, factor)
    truncated = Fraction(int(exact * 10), 10)
    c_slips = [
        (Fraction(quotient) + Fraction(remainder, 100), "wrote-minutes-as-decimal-part"),
        (total * factor, "inverted-ratio"),
        (Fraction(total, 100), "base-100-time"),
        (truncated, "rounded-down-not-to-nearest"),
    ]

    def fmt1(v: Fraction) -> str:
        return stated_precision_string(round_half_up(v, 1), 1) if v.denominator != 1 or v else "0.0"

    def fmt2(v: Fraction) -> str:
        return stated_precision_string(v, 2)

    c_options = []
    for value, tag in c_slips:
        value = Fraction(value)
        shown = fmt2(value) if tag == "wrote-minutes-as-decimal-part" else fmt1(value)
        shown_value = Fraction(shown)
        if shown_value > 0 and abs(shown_value - final) > GRADER_TOLERANCE:
            c_options.append(Distractor(shown, tag, shown_value))
    seen: set[Fraction] = set()
    c_distractors = []
    for d in pick_distractors(fmt1(final), c_options, 3, tolerance=GRADER_TOLERANCE):
        if d.value not in seen:
            seen.add(d.value)
            c_distractors.append(d)
    check(len(c_distractors) >= 2, "fewer than two usable distractors for part (c)")
    check(all(abs(Fraction(d.text) - final) > GRADER_TOLERANCE for d in c_distractors), "distractor accepted")

    c_text = stated_precision_string(final, 1)
    return dict(
        text=stem,
        latex=f"plain:{stem}",
        parts=[
            _int_part(
                a_prompt,
                each,
                marks=2,
                distractors=_options(Fraction(each), a_slips, _whole),
                working=[f"{big} × {factor} + {small} = {each} {su}s."],
            ),
            _int_part(
                b_prompt,
                total,
                marks=1,
                follow_through=f"{n}*part_a",
                distractors=_options(Fraction(total), b_slips, _whole),
                working=[f"{n} × {each} = {total} {su}s."],
            ),
            dict(
                prompt=c_prompt,
                answer_kind="number",
                answer=numeric_answer(final),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b/{factor}",
                working=[f"{total} ÷ {factor} = {c_text} {bu}s (1 d.p.)."],
                distractors=c_distractors,
            ),
        ],
    )
