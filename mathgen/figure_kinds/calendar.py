"""Calendar figure kinds: one month as a grid, and a strip of daily-event picture cards.

Every spec carries everything that is drawn; the front end draws it and never evaluates maths.
Each kind has pure ``read_*`` helpers that re-derive what a learner reads from the spec alone,
which a generator's plausibility reader calls.

Kinds: ``calendar``, ``day_strip``.

The calendar uses real calendar arithmetic (Python's :mod:`calendar`): the validator refuses a
spec whose day count or first weekday disagrees with the real month and year it names.
"""

from __future__ import annotations

import calendar as _calendar
from typing import Any, Iterable

from mathgen.figures import check, register_kind

# ------------------------------------------------------------------ calendar

MONTH_NAMES = (
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
)
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
WEEK_STARTS = ("mon", "sun")
MARK_STYLES = ("ring", "shade", "star")
MAX_LABEL = 12
MIN_YEAR, MAX_YEAR = 1900, 2100


def week_order(week_start: str) -> tuple[str, ...]:
    """The seven weekday names in column order, Monday first or Sunday first."""
    return WEEKDAY_NAMES if week_start == "mon" else (WEEKDAY_NAMES[-1], *WEEKDAY_NAMES[:-1])


def heading(name: str) -> str:
    """The three-letter column heading drawn for a weekday."""
    return name[:3]


def first_column(year: int, month: int, week_start: str) -> int:
    """The column (0-6) that day 1 falls in, from the real calendar."""
    weekday, _days = _calendar.monthrange(year, month)  # Monday = 0
    return weekday if week_start == "mon" else (weekday + 1) % 7


def calendar_month(
    *,
    year: int,
    month: int,
    alt: str,
    week_start: str = "mon",
    marks: Iterable[dict] = (),
    show_year: bool = False,
) -> dict:
    """One month of a real calendar as a grid.

    - ``year`` and ``month`` (1-12): the real month drawn. ``days`` and ``first_column`` are
      worked out from them by :mod:`calendar` (February has 29 days in a leap year) and the
      validator rejects any other value.
    - ``week_start``: ``"mon"`` or ``"sun"``, the weekday of the first column. Weekday headings are
      drawn as three letters (``Mon``) above the grid in that order.
    - ``marks``: ``{"day": int, "style": "ring" | "shade" | "star", "label"?: str}``. A ring is a
      circle round the date, a shade fills its square, a star sits in the corner of its square, so
      the styles differ without colour. ``label`` is a short event name (at most 12 characters)
      written in the square. A day has at most one mark.
    - ``show_year``: write the year beside the month name (default: the month name only).

    The date numerals are always drawn.
    """
    days = _calendar.monthrange(year, month)[1]
    spec = {
        "kind": "calendar",
        "alt": alt,
        "year": year,
        "month": month,
        "week_start": week_start,
        "days": days,
        "first_column": first_column(year, month, week_start),
        "marks": [dict(m) for m in marks],
        "show_year": bool(show_year),
    }
    return check(spec)


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_calendar(spec: dict) -> list[str]:
    year, month, start = spec.get("year"), spec.get("month"), spec.get("week_start")
    problems = []
    if not (_int(year) and MIN_YEAR <= year <= MAX_YEAR):
        problems.append(f"year must be an integer from {MIN_YEAR} to {MAX_YEAR}")
    if not (_int(month) and 1 <= month <= 12):
        problems.append("month must be an integer from 1 to 12")
    if start not in WEEK_STARTS:
        problems.append(f"week_start must be one of {WEEK_STARTS}")
    if not isinstance(spec.get("show_year"), bool):
        problems.append("show_year must be a bool")
    if problems:
        return problems
    days = _calendar.monthrange(year, month)[1]
    if spec.get("days") != days:
        problems.append(f"{MONTH_NAMES[month - 1]} {year} has {days} days, not {spec.get('days')!r}")
    if spec.get("first_column") != first_column(year, month, start):
        problems.append(f"the first of the month falls in column {first_column(year, month, start)}")
    marks = spec.get("marks")
    if not isinstance(marks, list) or len(marks) > days:
        return [*problems, "marks must be a list with at most one mark per day"]
    seen = set()
    for i, m in enumerate(marks):
        if not isinstance(m, dict):
            problems.append(f"marks[{i}] must be a dict")
            continue
        day = m.get("day")
        if not (_int(day) and 1 <= day <= days):
            problems.append(f"marks[{i}].day must be a date from 1 to {days}")
        elif day in seen:
            problems.append(f"marks[{i}]: day {day} is marked twice")
        else:
            seen.add(day)
        if m.get("style") not in MARK_STYLES:
            problems.append(f"marks[{i}].style must be one of {MARK_STYLES}")
        label = m.get("label", "")
        if not isinstance(label, str) or len(label) > MAX_LABEL:
            problems.append(f"marks[{i}].label must be a string of at most {MAX_LABEL} characters")
    return problems


def _calendar_texts(spec: dict) -> list[str]:
    """Everything drawn except the date numerals: the month, the headings and the event labels."""
    out = [MONTH_NAMES[spec["month"] - 1], *(heading(n) for n in week_order(spec["week_start"]))]
    if spec.get("show_year"):
        out.append(str(spec["year"]))
    out.extend(m["label"] for m in spec["marks"] if m.get("label"))
    return out


def month_name(spec: dict) -> str:
    """The month name the figure writes."""
    return MONTH_NAMES[spec["month"] - 1]


def column_of(spec: dict, day: int) -> int:
    """The column (0-6) a date sits in."""
    return (spec["first_column"] + day - 1) % 7


def row_of(spec: dict, day: int) -> int:
    """The row (0-5) a date sits in."""
    return (spec["first_column"] + day - 1) // 7


def rows(spec: dict) -> int:
    """How many week rows the grid needs."""
    return row_of(spec, spec["days"]) + 1


def weekday_of(spec: dict, day: int) -> str:
    """The weekday name of a date, read from its column and the headings."""
    return week_order(spec["week_start"])[column_of(spec, day)]


def mark_day(spec: dict, label: str) -> int:
    """The date of the mark carrying ``label``."""
    days = [m["day"] for m in spec["marks"] if m.get("label") == label]
    if len(days) != 1:
        raise ValueError(f"expected one mark labelled {label!r}, found {len(days)}")
    return days[0]


register_kind("calendar", _validate_calendar, texts=_calendar_texts)

# ------------------------------------------------------------------ day_strip

# Day order, earliest first. Two cards the same size of event never share a strip when their
# order is arguable (see CONFLICTS), so every strip has one right order.
ICON_ORDER = ("sunrise", "wake_up", "breakfast", "school", "lunch", "play", "dinner", "bath", "bed", "moon")
ICON_LABELS = {
    "sunrise": "Sunrise",
    "wake_up": "Wake up",
    "breakfast": "Breakfast",
    "school": "School",
    "lunch": "Lunch",
    "play": "Play",
    "dinner": "Dinner",
    "bath": "Bath",
    "bed": "Bed",
    "moon": "Moon",
}
DAY_PARTS = ("morning", "lunchtime", "afternoon", "night")
ICON_PART = {
    "sunrise": "morning",
    "wake_up": "morning",
    "breakfast": "morning",
    "school": "morning",
    "lunch": "lunchtime",
    "play": "afternoon",
    "dinner": "night",
    "bath": "night",
    "bed": "night",
    "moon": "night",
}
# Pairs whose order, or whose part of the day, a child could fairly argue: never drawn together.
CONFLICTS = (
    frozenset({"sunrise", "wake_up"}),
    frozenset({"bed", "moon"}),
    frozenset({"school", "lunch"}),
)
MAX_CARDS = 6
MAX_CARD_LABEL = 10


def day_strip(*, cards: Iterable[str], alt: str, numbered: bool = False) -> dict:
    """A row of up to six picture cards for daily events.

    - ``cards``: icon names from :data:`ICON_ORDER`, left to right as drawn (a scrambled order is
      the caller's choice). Each is drawn as a simple icon with its label written beneath.
    - ``numbered``: put a slot number (1, 2, 3 ...) above each card, counting the drawn positions
      from the left. The numbers say where a card IS in the row, not where it belongs in the day.

    Spec: ``{"kind": "day_strip", "alt", "numbered": bool,
    "cards": [{"icon": str, "label": str}]}``.
    """
    spec = {
        "kind": "day_strip",
        "alt": alt,
        "numbered": bool(numbered),
        "cards": [{"icon": c, "label": ICON_LABELS.get(c, "")} for c in cards],
    }
    return check(spec)


def _validate_day_strip(spec: dict) -> list[str]:
    cards = spec.get("cards")
    if not isinstance(spec.get("numbered"), bool):
        return ["numbered must be a bool"]
    if not isinstance(cards, list) or not 1 <= len(cards) <= MAX_CARDS:
        return [f"cards must hold 1 to {MAX_CARDS} cards"]
    problems = []
    icons = []
    for i, c in enumerate(cards):
        if not isinstance(c, dict) or c.get("icon") not in ICON_LABELS:
            problems.append(f"cards[{i}].icon must be one of {ICON_ORDER}")
            continue
        icons.append(c["icon"])
        label = c.get("label")
        if not isinstance(label, str) or not label or len(label) > MAX_CARD_LABEL:
            problems.append(f"cards[{i}].label must be 1 to {MAX_CARD_LABEL} characters")
    if len(set(icons)) != len(icons):
        problems.append("a card is drawn twice")
    for pair in CONFLICTS:
        if pair <= set(icons):
            problems.append(f"{sorted(pair)} cannot share a strip: their order is arguable")
    return problems


def _day_strip_texts(spec: dict) -> list[str]:
    return [c["label"] for c in spec["cards"] if c.get("label")]


def order_index(icon: str) -> int:
    """Where an event falls in the day (0 = earliest)."""
    return ICON_ORDER.index(icon)


def slot_of(spec: dict, label: str) -> int:
    """The 1-based slot a labelled card is drawn in."""
    slots = [i + 1 for i, c in enumerate(spec["cards"]) if c["label"] == label]
    if len(slots) != 1:
        raise ValueError(f"expected one card labelled {label!r}, found {len(slots)}")
    return slots[0]


def day_order(spec: dict) -> list[int]:
    """The 1-based slots listed in the order the events happen in the day."""
    return [i + 1 for i in sorted(range(len(spec["cards"])), key=lambda i: order_index(spec["cards"][i]["icon"]))]


def slots_in_part(spec: dict, part: str) -> list[int]:
    """The 1-based slots whose event happens in a part of the day."""
    return [i + 1 for i, c in enumerate(spec["cards"]) if ICON_PART[c["icon"]] == part]


register_kind("day_strip", _validate_day_strip, texts=_day_strip_texts)
