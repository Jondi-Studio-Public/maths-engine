"""Early-years calendar questions: one month drawn with the ``calendar`` figure kind.

- ``ey_calendar_span`` (skill ``ey-calendar-count-days``, ``answer: number``): easy counts the days in
  the month drawn; medium counts the weeks from a starred day to a ringed day exactly 14, 21 or 28
  days on; hard gives the span as weeks and days and asks for the days, so the weeks must be turned
  into days first.
- ``ey_calendar_weekday`` (skill ``ey-calendar-weekday-of-date``, ``answer: mcq``): easy names the
  weekday of a ringed date; medium of a date named in words (the learner finds it, then reads its
  column, with the week starting on Monday or Sunday); hard of the last day of the month or the 1st of
  the next one, which needs the month's length as well as its first column.
- ``ey_calendar_between`` (skill ``ey-calendar-days-between``, ``answer: number``): easy counts the
  days from one marked event to another within a week; medium finds the date one to three weeks after an
  event; hard counts days across several weeks, or from an event to the last day of the month.
- ``ey_calendar_order`` (skill ``ey-calendar-days-months-order``, ``answer: mcq``): easy gives the day
  after a weekday read from the column headings; medium the day after or before one at the edge of the
  week, where the order wraps to the other end of the row; hard the month after or before the one drawn.

Every month is a real month and year, so day counts and weekdays come from Python's ``calendar``
module, and the generator checks each keyed weekday a second way. The date numerals are the only
numbers drawn, so no label shows an answer. Each topic registers a figure reader that works from
``question.figure`` and the prompt alone.
"""

from __future__ import annotations

import calendar
import re

from .. import figures
from ..core import check, generator, phrase, pick_distractors
from ..figure_kinds.calendar import (
    MONTH_NAMES,
    WEEK_STARTS,
    WEEKDAY_NAMES,
    calendar_month,
    mark_day,
    week_order,
    weekday_of,
)
from ..slots import NAMES

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_EVENTS = ("Party", "Swim", "Zoo", "Picnic", "Library", "Beach", "Movie", "Dentist", "Footy", "Market", "Concert")


def _tier(tier: str) -> str:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return tier


def _month(rng, week_start: str | None = None) -> tuple[int, int, str]:
    return rng.randint(2024, 2031), rng.randint(1, 12), week_start or rng.choice(WEEK_STARTS)


def _unleaked(fig: dict, cands):
    """The candidates the figure's own text does not show."""
    out = []
    for c in cands:
        try:
            figures.assert_no_leak(fig, [c])
        except AssertionError:
            continue
        out.append(c)
    return out


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _number_result(text, answer: int, fig: dict, distractors: list[str], working: str) -> dict:
    answer_text = str(answer)
    check(len(distractors) >= 2, "too few distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    return dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[working],
        figure=fig,
    )


def _mcq_result(text, answer_text: str, fig: dict, cands: list[str], working: str) -> dict:
    distractors = pick_distractors(answer_text, _unleaked(fig, cands), n=3)
    check(len(distractors) == 3, "too few distractors")
    figures.assert_no_leak(fig, [answer_text, *distractors])
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=[working],
        figure=fig,
    )


# ====================================================================== count days, weeks

_DAYS_TEXT = (
    "How many days are in this month?",
    "{n} counts the days in this month. How many are there?",
    "How many days does this month have, {n}?",
)
_WEEKS_TEXT = (
    "{n} counts from Start to End. How many weeks is that?",
    "How many weeks from the Start to the End?",
    "The Start and the End are how many weeks apart?",
)
_WEEKS_DAYS_TEXT = (
    "{n} counts from Start to End: {span}. How many days?",
    "From Start to End is {span}. How many days is that?",
    "Start to End takes {span}. How many days?",
)
_SPAN_RE = re.compile(r"(\d+) weeks? and (\d+) days?")


def _span_marks(rng, days: int, gap: int):
    s = rng.randint(1, days - gap)
    return (
        s,
        s + gap,
        [{"day": s, "style": "star", "label": "Start"}, {"day": s + gap, "style": "ring", "label": "End"}],
    )


@generator("ey_calendar_span", level=1, presets=_PRESETS)
def ey_calendar_span(rng, tier="easy"):
    _tier(tier)
    year, month, ws = _month(rng)
    name = rng.choice(NAMES)
    days = calendar.monthrange(year, month)[1]
    if tier == "easy":
        fig = calendar_month(year=year, month=month, week_start=ws, alt="a calendar for one month")
        answer = days
        rows = -(-(fig["first_column"] + days) // 7)
        cands = [30, days - 1, days + 1, rows * 7, 31, 28]
        text = phrase(rng, _DAYS_TEXT, n=name)
        working = f"The last date on the calendar is {days}, so the month has {days} days."
    elif tier == "medium":
        weeks = rng.randint(2, 4 if days > 28 else 3)
        s, e, marks = _span_marks(rng, days, 7 * weeks)
        fig = calendar_month(year=year, month=month, week_start=ws, marks=marks, alt="a calendar with two marked days")
        answer = weeks
        cands = [7 * weeks, weeks + 1, weeks - 1, e]
        text = phrase(rng, _WEEKS_TEXT, n=name)
        working = f"From {s} to {e} is {e - s} days. 7 days make a week, so {e - s} days is {weeks} weeks."
    else:
        weeks, extra = rng.randint(1, 3), rng.randint(1, 6)
        gap = 7 * weeks + extra
        s, e, marks = _span_marks(rng, days, gap)
        fig = calendar_month(year=year, month=month, week_start=ws, marks=marks, alt="a calendar with two marked days")
        answer = gap
        cands = [weeks + extra, 7 * weeks, weeks * extra, gap + 1, gap - 1, int(f"{weeks}{extra}")]
        span = f"{_plural(weeks, 'week')} and {_plural(extra, 'day')}"
        text = phrase(rng, _WEEKS_DAYS_TEXT, n=name, span=span)
        working = f"{_plural(weeks, 'week')} is {7 * weeks} days. {7 * weeks} + {extra} = {gap} days."
    distractors = pick_distractors(str(answer), _unleaked(fig, [str(c) for c in cands if c > 0]))
    out = _number_result(text, answer, fig, distractors, working)
    check(_read_span(fig, text) == out["answer_text"], "the figure does not show the keyed answer")
    return out


def _read_span(fig: dict, text: str) -> str:
    span = _SPAN_RE.search(text)
    if span:
        gap = mark_day(fig, "End") - mark_day(fig, "Start")
        check(gap == 7 * int(span.group(1)) + int(span.group(2)), "the figure disagrees with the weeks and days")
        return str(gap)
    if "Start" in text:
        gap = mark_day(fig, "End") - mark_day(fig, "Start")
        check(gap % 7 == 0, "the span is not a whole number of weeks")
        return str(gap // 7)
    return str(fig["days"])


figures.register_reader("ey_calendar_span", lambda q: _read_span(q.figure, q.text))


# ====================================================================== weekday of a date

_RINGED_TEXT = (
    "What day of the week is the ringed date?",
    "{n} rings a date. What day of the week is it?",
    "Which day of the week is the ringed date?",
)
_DATE_TEXT = (
    "What day of the week is the {d}?",
    "{n} has a game on the {d}. What day of the week is that?",
    "Which day of the week is the {d}?",
)
_LAST_TEXT = (
    "What day of the week is the last day of the month?",
    "{n} looks at the last date. What day of the week is it?",
    "Which day of the week does this month end on?",
)
_NEXT_TEXT = (
    "What day of the week is the 1st of next month?",
    "{n} waits for next month. What day of the week is the 1st?",
    "Which day of the week does next month start on?",
)


def _weekday_cands(fig: dict, day: int, answer: str) -> list[str]:
    order = week_order(fig["week_start"])
    col = order.index(answer)
    naive = WEEKDAY_NAMES[(day - 1) % 7]  # as if every month began on a Monday
    return [naive, order[(col + 1) % 7], order[(col - 1) % 7], order[(col + 2) % 7], order[(col - 2) % 7]]


@generator("ey_calendar_weekday", level=2, presets=_PRESETS)
def ey_calendar_weekday(rng, tier="easy"):
    _tier(tier)
    name = rng.choice(NAMES)
    if tier == "easy":
        year, month, ws = _month(rng, "mon")
        day = rng.randint(2, 14)
        fig = calendar_month(
            year=year,
            month=month,
            week_start=ws,
            marks=[{"day": day, "style": "ring"}],
            alt="a calendar with a ringed date",
        )
        text = phrase(rng, _RINGED_TEXT, n=name)
        key_day = day
    elif tier == "medium":
        year, month, ws = _month(rng)
        days = calendar.monthrange(year, month)[1]
        day = rng.randint(8, days)
        fig = calendar_month(year=year, month=month, week_start=ws, alt="a calendar for one month")
        text = phrase(rng, _DATE_TEXT, n=name, d=_ordinal(day))
        key_day = day
    else:
        year, month, ws = _month(rng)
        days = calendar.monthrange(year, month)[1]
        fig = calendar_month(year=year, month=month, week_start=ws, alt="a calendar for one month")
        if rng.random() < 0.5:
            text = phrase(rng, _LAST_TEXT, n=name)
            key_day = days
            expected = WEEKDAY_NAMES[calendar.weekday(year, month, days)]
        else:
            text = phrase(rng, _NEXT_TEXT, n=name)
            key_day = days + 1
            ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
            expected = WEEKDAY_NAMES[calendar.weekday(ny, nm, 1)]
    answer_text = _read_weekday(fig, text)
    if tier == "hard":
        check(answer_text == expected, "the figure disagrees with the real calendar")
    else:
        check(answer_text == WEEKDAY_NAMES[calendar.weekday(year, month, key_day)], "weekday disagrees with calendar")
    cands = _weekday_cands(fig, key_day, answer_text)
    if tier == "hard":
        first = week_order(ws)[fig["first_column"]]
        last = weekday_of(fig, fig["days"])
        cands = [last if "next" in text else first, first if "next" not in text else last, *cands[1:]]
    return _mcq_result(
        text,
        answer_text,
        fig,
        cands,
        f"Find the date in its column and read the weekday above it: {answer_text}.",
    )


_DATE_RE = re.compile(r"the (\d+)(?:st|nd|rd|th)\b")


def _read_weekday(fig: dict, text: str) -> str:
    if "next month" in text:
        return week_order(fig["week_start"])[(fig["first_column"] + fig["days"]) % 7]
    if "last" in text or "end on" in text:
        return weekday_of(fig, fig["days"])
    ringed = [m["day"] for m in fig["marks"] if m["style"] == "ring"]
    if ringed:
        check(len(ringed) == 1, "expected one ringed date")
        return weekday_of(fig, ringed[0])
    date = _DATE_RE.search(text)
    check(date is not None, "no date in the prompt")
    return weekday_of(fig, int(date.group(1)))


figures.register_reader("ey_calendar_weekday", lambda q: _read_weekday(q.figure, q.text))


# ====================================================================== days between events

_BETWEEN_TEXT = (
    "How many days from the {a} to the {b}?",
    "{n} counts the days from the {a} to the {b}. How many?",
    "How many days is it from the {a} until the {b}?",
)
_AFTER_TEXT = (
    "What is the date {w} after the {a}?",
    "{w} after the {a} is which date?",
    "{n} counts on from the {a}. What is the date {w} later?",
)
_TO_END_TEXT = (
    "How many days from the {a} to the last day of the month?",
    "{n} counts from the {a} to the last day of the month. How many days?",
)
_BETWEEN_RE = re.compile(r"from the (\w+) (?:to|until) the (\w+)")
_AFTER_RE = re.compile(r"(\d) weeks?\b")


def _two_events(rng):
    a, b = rng.sample(_EVENTS, 2)
    return a, b


@generator("ey_calendar_between", level=2, presets=_PRESETS)
def ey_calendar_between(rng, tier="easy"):
    _tier(tier)
    year, month, ws = _month(rng)
    days = calendar.monthrange(year, month)[1]
    name = rng.choice(NAMES)
    a, b = _two_events(rng)
    if tier == "medium":
        weeks = rng.randint(1, 3)
        start = rng.randint(1, days - 7 * weeks)
        fig = calendar_month(
            year=year,
            month=month,
            week_start=ws,
            marks=[{"day": start, "style": "star", "label": a}],
            alt="a calendar with one marked event",
        )
        answer = start + 7 * weeks
        cands = [start + weeks, answer + 1, answer - 1, 7 * weeks, start + 7 * (weeks + 1)]
        text = phrase(rng, _AFTER_TEXT, n=name, a=a, w=_plural(weeks, "week"))
        working = f"{_plural(weeks, 'week')} is {7 * weeks} days. {start} + {7 * weeks} = {answer}."
    elif tier == "hard" and rng.random() < 0.4:
        start = rng.randint(1, days - 8)
        fig = calendar_month(
            year=year,
            month=month,
            week_start=ws,
            marks=[{"day": start, "style": "star", "label": a}],
            alt="a calendar with one marked event",
        )
        answer = days - start
        cands = [answer + 1, start, days, answer - 1]
        text = phrase(rng, _TO_END_TEXT, n=name, a=a)
        working = f"The month ends on the {days}. {days} - {start} = {answer} days."
    else:
        gap = rng.randint(2, 6) if tier == "easy" else rng.randint(8, 20)
        start = rng.randint(1, days - gap)
        fig = calendar_month(
            year=year,
            month=month,
            week_start=ws,
            marks=[{"day": start, "style": "star", "label": a}, {"day": start + gap, "style": "ring", "label": b}],
            alt="a calendar with two marked events",
        )
        answer = gap
        cands = [gap + 1, gap - 1, start + gap, start, 7 if tier == "easy" else 14, gap + 7]
        text = phrase(rng, _BETWEEN_TEXT, n=name, a=a, b=b)
        working = f"The {b} is on the {start + gap} and the {a} on the {start}. {start + gap} - {start} = {gap} days."
    distractors = pick_distractors(str(answer), _unleaked(fig, [str(c) for c in cands if c > 0]))
    out = _number_result(text, answer, fig, distractors, working)
    check(_read_between(fig, text) == out["answer_text"], "the figure does not show the keyed answer")
    return out


def _read_between(fig: dict, text: str) -> str:
    labels = {m["label"]: m["day"] for m in fig["marks"] if m.get("label")}
    pair = _BETWEEN_RE.search(text)
    if pair and pair.group(2) in labels:
        return str(labels[pair.group(2)] - labels[pair.group(1)])
    if "last day" in text:
        (day,) = labels.values()
        return str(fig["days"] - day)
    weeks = _AFTER_RE.search(text)
    check(weeks is not None, "no weeks in the prompt")
    (day,) = labels.values()
    return str(day + 7 * int(weeks.group(1)))


figures.register_reader("ey_calendar_between", lambda q: _read_between(q.figure, q.text))


# ====================================================================== order of days and months

_DAY_AFTER_TEXT = (
    "What day comes after {d}?",
    "{n} is thinking about the week. What day comes after {d}?",
    "Which day comes next after {d}?",
)
_DAY_BEFORE_TEXT = (
    "What day comes before {d}?",
    "{n} is thinking about the week. What day comes before {d}?",
    "Which day comes just before {d}?",
)
_MONTH_TEXT = {
    1: ("Which month comes next?", "{n} looks at this calendar. Which month comes next?"),
    -1: ("Which month came before this one?", "{n} looks at this calendar. Which month came before it?"),
}
_DAY_RE = re.compile(r"(after|before|next after|just before) (\w+day)\?")


@generator("ey_calendar_order", level=1, presets=_PRESETS)
def ey_calendar_order(rng, tier="easy"):
    _tier(tier)
    name = rng.choice(NAMES)
    year, month, ws = _month(rng, "mon" if tier == "easy" else None)
    if tier == "hard":
        step = rng.choice((1, -1))
        fig = calendar_month(year=year, month=month, week_start=ws, alt="a calendar for one month")
        text = phrase(rng, _MONTH_TEXT[step], n=name)
        answer_text = _read_order(fig, text)
        check(answer_text == MONTH_NAMES[(month - 1 + step) % 12], "month order slip")
        cands = [
            MONTH_NAMES[(month - 1 - step) % 12],
            MONTH_NAMES[(month - 1 + 2 * step) % 12],
            MONTH_NAMES[(month - 1 - 2 * step) % 12],
        ]
        return _mcq_result(
            text,
            answer_text,
            fig,
            cands,
            f"The months go in a loop. {answer_text} is {'after' if step == 1 else 'before'} {MONTH_NAMES[month - 1]}.",
        )
    order = week_order(ws)
    after = tier == "easy" or rng.random() < 0.5
    if tier == "easy":
        col = rng.randint(0, 5)
    else:
        col = (6 if after else 0) if rng.random() < 0.7 else rng.randint(0, 6)
    day = order[col]
    step = 1 if after else -1
    fig = calendar_month(year=year, month=month, week_start=ws, alt="a calendar for one month")
    text = phrase(rng, _DAY_AFTER_TEXT if after else _DAY_BEFORE_TEXT, n=name, d=day)
    answer_text = _read_order(fig, text)
    check(answer_text == WEEKDAY_NAMES[(WEEKDAY_NAMES.index(day) + step) % 7], "weekday order slip")
    cands = [
        order[(col - step) % 7],
        order[(col + 2 * step) % 7],
        order[(col + 3 * step) % 7],
        order[(col - 2 * step) % 7],
    ]
    if tier == "medium" and col in (0, 6):
        cands.insert(0, day)  # the order does not wrap: the learner repeats the day, or reads the other edge
    return _mcq_result(
        text,
        answer_text,
        fig,
        cands,
        f"The days go round in a loop: {answer_text} is {'after' if after else 'before'} {day}.",
    )


def _read_order(fig: dict, text: str) -> str:
    day = _DAY_RE.search(text)
    if day:
        order = week_order(fig["week_start"])
        step = 1 if day.group(1) in ("after", "next after") else -1
        return order[(order.index(day.group(2)) + step) % 7]
    step = -1 if "before" in text else 1
    return MONTH_NAMES[(fig["month"] - 1 + step) % 12]


figures.register_reader("ey_calendar_order", lambda q: _read_order(q.figure, q.text))
