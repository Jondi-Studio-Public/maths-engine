"""Reading an analogue clock, from the figure alone.

- ``clock_half_quarter`` (reading the hour, half hour and quarter hour): a multiple choice over
  times written in words ("quarter to 4") or as digital times ("3:45"), one form per question
  so no two options are the same time. Easy reads o'clock and half past; medium adds quarter
  past and quarter to, where the hour hand sits nearly at the next numeral; hard shows four
  clocks A to D and asks which one shows a time given in words.
- ``clock_read_minute`` (reading to the nearest minute): a multiple choice over digital times.
  Easy reads five-minute times, medium reads one-minute times off the minute ticks, hard shows
  a Start and an End clock and asks how much time has passed.

The hour hand is drawn at ``30 * hour + 0.5 * minute`` degrees, so a quarter-to time has the
hour hand three quarters of the way to the next numeral. Every distractor is a time a learner
reaches by a named misreading: swapping the hands, naming the numeral the hour hand has passed
instead of the one it approaches, turning a quarter to into a quarter past, reading the minute
hand's numeral as the number of minutes. Each topic registers a reader that works from
``question.figure`` (and the time the prompt names, for the matching tier) alone.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.core import attempts, check, generator, pick_distractors
from mathgen.figure_kinds import measure

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

_QUARTER_MINUTES = (0, 15, 30, 45)


def _hr(h: int) -> int:
    """Wrap an hour into 1-12."""
    return (h - 1) % 12 + 1


def _words(h: int, m: int) -> str | None:
    """A quarter-hour time in words, or None if it has no such form."""
    if m == 0:
        return f"{h} o'clock"
    if m == 30:
        return f"half past {h}"
    if m == 15:
        return f"quarter past {h}"
    if m == 45:
        return f"quarter to {_hr(h + 1)}"
    return None


def _digital(h: int, m: int) -> str:
    return f"{h}:{m:02d}"


def _fmt(style: str, h: int, m: int) -> str | None:
    return _words(h, m) if style == "words" else _digital(h, m)


def _clock_fig(h: int, m: int, *, numerals: str, ticks: bool, caption: str = "", alt: str = "an analogue clock"):
    return measure.clock(hour=h, minute=m, alt=alt, numerals=numerals, minute_ticks=ticks, caption=caption)


def _valid_clock(h: int, m: int, *, numerals: str = "all", ticks: bool = True) -> bool:
    return not figures.validate(
        {
            "kind": "clock",
            "alt": "a clock",
            "hour": h,
            "minute": m,
            "hour_angle": measure.hand_angles(h, m)[0],
            "minute_angle": measure.hand_angles(h, m)[1],
            "numerals": numerals,
            "minute_ticks": ticks,
        }
    )


def _nearest5(minutes: float) -> int:
    return int(round(minutes / 5.0) * 5) % 60


def _swapped(h: int, m: int) -> tuple[int, int]:
    """The time a learner reads when the short hand is taken for minutes and the long hand for hours."""
    hour_hand_minutes = (30 * (h % 12) + 0.5 * m) / 6
    return (m // 5) or 12, _nearest5(hour_hand_minutes)


def _misreads(h: int, m: int, types: tuple[int, ...]) -> list[tuple[int, int]]:
    """Times a learner might name for a clock showing h:m, most characteristic misreading first."""
    out: list[tuple[int, int]] = []
    if m == 45:
        out.append((_hr(h - 1), 45))  # names the numeral the hour hand has passed, "quarter to 3" for 3:45
    out.append(_swapped(h, m))
    for t in types:  # the wrong kind of time on the right hour
        if t != m:
            out.append((h, t))
    out.append((_hr(h + 1), m))
    out.append((_hr(h - 1), m))
    for t in types:
        if t != m:
            out.append((_hr(h + 1), t))
            out.append((_hr(h - 1), t))
    return out


# ------------------------------------------------------------ half and quarter hour

_HQ_POOLS = {
    "easy": (
        (
            "The clock on the classroom wall is shown.",
            "Look at the clock.",
            "Here is a clock face.",
            "This clock hangs in the kitchen.",
            "A clock is shown below.",
            "Tom looks up at the clock in his bedroom.",
        ),
        (
            "What time does the clock show?",
            "Which time is shown on the clock?",
            "Choose the time that matches the clock.",
        ),
    ),
    "medium": (
        (
            "The clock above the shop door is shown.",
            "Read the clock on the train platform.",
            "Nina checks the hall clock.",
            "This clock face has two hands.",
            "The library clock is shown below.",
            "A clock at a bus stop is drawn here.",
        ),
        (
            "Look closely at both hands. What time is it?",
            "Which time do the two hands show?",
            "Say what time it is.",
        ),
    ),
    "hard": (
        (
            "Four clocks are drawn, A to D.",
            "Clocks A, B, C and D are shown.",
            "Here are four different clocks.",
            "Look at the four clocks A to D.",
        ),
        (
            "Which clock shows {t}?",
            "Find the clock that shows {t}.",
            "Which clock is showing {t}?",
        ),
    ),
}

_STYLE_HINT = {
    "words": ("Choose the answer written in words.", "The answers are written in words.", "Pick the time in words."),
    "digital": (
        "Choose the digital time.",
        "The answers are digital times.",
        "Pick the time as it would show on a digital clock.",
    ),
}

_HQ_NOTES = {
    0: "The long hand points straight up at the 12, so it is a whole hour; the short hand points at the hour.",
    15: "The long hand points at the 3: quarter past. The short hand has just passed the hour numeral.",
    30: "The long hand points at the 6: half past. The short hand is halfway between two numerals.",
    45: "The long hand points at the 9: quarter to. The short hand is almost at the next numeral, and the time is quarter to that next hour.",
}


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def _distractor_times(h, m, types, style, n, *, need_valid=False):
    cands = []
    for ch, cm in _misreads(h, m, types):
        if (ch, cm) == (h, m) or (ch, cm) in cands:
            continue
        if style == "words" and cm not in _QUARTER_MINUTES:
            continue
        if need_valid and not _valid_clock(ch, cm):
            continue
        cands.append((ch, cm))
    return cands[:n]


def _build_half_quarter(rng, tier: str):
    types = (0, 30) if tier == "easy" else _QUARTER_MINUTES
    if tier == "easy":
        m = _pick(rng, (0, 30))
    elif tier == "medium":
        m = _pick(rng, (15, 45, 15, 45, 0, 30))
    else:
        m = _pick(rng, (15, 45, 0, 30))
    style = _pick(rng, ("words", "digital"))
    # Quarter-to times read as "to the next hour", so the hour is drawn the hand is nearest, not read off.
    for _ in attempts(200, "a valid half/quarter clock"):
        h = rng.randrange(1, 13)
        if _valid_clock(h, m, numerals="all", ticks=(tier != "easy")):
            break
    numerals = "all"
    ticks = tier != "easy"
    if tier != "hard":
        fig = _clock_fig(h, m, numerals=numerals, ticks=ticks)
        answer = _fmt(style, h, m)
        near = _distractor_times(h, m, types, style, 3)
        check(len(near) == 3, "too few misreadings for the options")
        distractors = [_fmt(style, *t) for t in near]
        text = " ".join(
            [_pick(rng, _HQ_POOLS[tier][0]), _pick(rng, _HQ_POOLS[tier][1]), _pick(rng, _STYLE_HINT[style])]
        )
        return fig, text, answer, distractors, h, m
    # hard: four clocks, one of which shows the time named in the prompt
    target = _fmt(style, h, m)
    near = _distractor_times(h, m, types, style, 6, need_valid=True)
    check(len(near) >= 3, "too few legible misread clocks")
    # draw 3 of the candidates, keeping the most characteristic misreadings likelier
    chosen = near[:2] + [near[2 + rng.randrange(len(near) - 2)]]
    times = [(h, m), *chosen]
    check(len(set(times)) == 4, "two clocks show the same time")
    order = list(range(4))
    rng.shuffle(order)
    captions = "ABCD"
    clocks = [
        _clock_fig(*times[i], numerals=numerals, ticks=ticks, caption=captions[pos]) for pos, i in enumerate(order)
    ]
    fig = measure.clock_set(clocks=clocks, alt="four analogue clocks")
    answer = f"Clock {captions[order.index(0)]}"
    distractors = [f"Clock {c}" for c in captions if f"Clock {c}" != answer]
    text = " ".join([_pick(rng, _HQ_POOLS[tier][0]), _pick(rng, _HQ_POOLS[tier][1]).format(t=target)])
    return fig, text, answer, distractors, h, m


@generator("clock_half_quarter", level=3, presets=_PRESETS)
def clock_half_quarter(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, answer, distractors, h, m = _build_half_quarter(rng, tier)
    distractors = pick_distractors(answer, distractors, n=3)
    check(len(distractors) == 3, "the question needs three distinct distractors")
    figures.assert_no_leak(fig, [answer, *distractors])
    return dict(
        text=text,
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        distractors=distractors,
        working=[_HQ_NOTES[m]],
        figure=fig,
    )


_TIME_IN_TEXT = re.compile(r"(\d+) o'clock|half past (\d+)|quarter past (\d+)|quarter to (\d+)|(\d{1,2}):(\d\d)")


def _time_from_text(text: str) -> tuple[int, int]:
    mt = _TIME_IN_TEXT.search(text)
    check(mt is not None, "the prompt names no time")
    o, half, past, to, hh, mm = mt.groups()
    if o:
        return int(o), 0
    if half:
        return int(half), 30
    if past:
        return int(past), 15
    if to:
        return _hr(int(to) - 1), 45
    return int(hh), int(mm)


def _read_half_quarter(question) -> str:
    fig = question.figure
    if fig["kind"] == "clock_set":
        target = _time_from_text(question.text)
        hits = [c["caption"] for c in fig["clocks"] if measure.read_clock(c) == target]
        check(len(hits) == 1, "not exactly one clock shows the named time")
        return f"Clock {hits[0]}"
    h, m = measure.read_clock(fig)
    style = "digital" if any(":" in d for d in question.distractors) else "words"
    return _fmt(style, h, m)


figures.register_reader("clock_half_quarter", _read_half_quarter)


# ------------------------------------------------------------ to the nearest minute

_MIN_POOLS = {
    "easy": (
        (
            "The clock on the wall is shown below.",
            "Look at this clock.",
            "Mia checks the clock before school.",
            "This clock is in the hallway.",
            "A clock face is drawn here.",
            "The kitchen clock is shown.",
        ),
        (
            "What time is it? Choose the digital time.",
            "Which digital time matches the clock?",
            "Choose the digital time that the clock shows.",
            "Which digital time is on the clock?",
        ),
    ),
    "medium": (
        (
            "This clock has a mark for every minute.",
            "The station clock shows the time between two numerals.",
            "Read the minute hand against the minute marks.",
            "Ravi looks at the clock on the oven.",
            "The long hand of this clock is between two numerals.",
            "Lena reads the time from a clock with minute marks.",
        ),
        (
            "Which digital time does it show?",
            "What time is shown, to the nearest minute?",
            "Choose the digital time shown.",
            "Say the time as a digital time.",
        ),
    ),
    "hard": (
        (
            "The film starts at the time on the Start clock and ends at the time on the End clock.",
            "Ava's swimming lesson starts at the Start clock's time and finishes at the End clock's time.",
            "The bus leaves at the time shown on the Start clock and arrives at the time shown on the End clock.",
            "Dan begins his homework at the Start time and stops at the End time.",
            "A cake goes into the oven at the Start time and comes out at the End time.",
            "The match kicks off at the Start time and finishes at the End time.",
            "A train departs at the time on the Start clock and gets in at the time on the End clock.",
        ),
        (
            "How long does it take?",
            "How much time has passed?",
            "How long is it from the start to the end?",
            "How much time is there between the two clocks?",
        ),
    ),
}


def _mins(m: int) -> str:
    return f"{m} minute" if m == 1 else f"{m} minutes"


def _duration(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    if h == 0:
        return f"{m} minutes"
    hs = f"{h} hour" if h == 1 else f"{h} hours"
    return hs if m == 0 else f"{hs} {m} minutes"


def _read_minute_misreads(h: int, m: int, tier: str) -> list[tuple[int, int]]:
    out = []
    if m >= 5:
        out.append((h, m // 5))  # reads the numeral the minute hand points at as the number of minutes
    if m >= 30:
        out.append((_hr(h + 1), m))  # takes the numeral nearest the short hand for the hour
    if tier == "medium":
        out.append((h, m + 1 if m < 59 else m - 1))
        out.append((h, m - 1 if m > 0 else m + 1))
        out.append((h, (m // 5) * 5))  # reads the numeral the hand has passed and stops there
        out.append((h, m + 5 if m < 55 else m - 5))
    elif m:
        out.append((h, 60 - m))  # counts minutes to instead of minutes past
    out.append(_swapped(h, m))
    out.append((h, m + 5 if m < 55 else m - 5))
    out.append((_hr(h - 1), m))
    out.append((_hr(h + 1), m))
    return out


def _elapsed_misreads(h1, m1, h2, m2, d) -> list[int]:
    hours = (h2 - h1) % 12
    out = []
    if m2 < m1:
        out.append(60 * hours + (m1 - m2))  # subtracts the smaller minute from the larger without regrouping
    if hours:
        out.append(60 * hours)  # hours only
    out.append(abs(m2 - m1))  # minutes only
    out.extend([d + 10, d - 10, d + 60, d - 60, d + 5, d - 5])
    return out


def _build_read_minute(rng, tier: str):
    if tier == "easy":
        for _ in attempts(200, "a five-minute clock"):
            h, m = rng.randrange(1, 13), 5 * rng.randrange(12)
            if _valid_clock(h, m):
                break
        fig = _clock_fig(h, m, numerals="all", ticks=True)
        answer = _digital(h, m)
        cands = [_digital(*t) for t in _read_minute_misreads(h, m, tier)]
        notes = (
            f"The long hand points at the {m // 5 or 12}, which is {_mins(m)} past. The short hand shows the hour, {h}."
        )
    elif tier == "medium":
        numerals = _pick(rng, ("all", "quarters"))
        for _ in attempts(200, "a one-minute clock"):
            h, m = rng.randrange(1, 13), rng.randrange(1, 60)
            if m % 5 and _valid_clock(h, m, numerals=numerals):
                break
        fig = _clock_fig(h, m, numerals=numerals, ticks=True)
        answer = _digital(h, m)
        cands = [_digital(*t) for t in _read_minute_misreads(h, m, tier)]
        notes = f"Count minutes round from the 12: {_mins(m)} past. The short hand shows the hour, {h}."
    else:
        for _ in attempts(500, "an elapsed-time pair"):
            h1, m1 = rng.randrange(1, 13), 5 * rng.randrange(12)
            d = 5 * rng.randrange(5, 36)
            if d % 60 == 0:
                continue
            end = ((h1 % 12) * 60 + m1 + d) % 720  # minutes since 12:00 on the face
            h2, m2 = end // 60 or 12, end % 60
            if _valid_clock(h1, m1) and _valid_clock(h2, m2):
                break
        clocks = [
            _clock_fig(h1, m1, numerals="all", ticks=True, caption="Start"),
            _clock_fig(h2, m2, numerals="all", ticks=True, caption="End"),
        ]
        fig = measure.clock_set(clocks=clocks, alt="two analogue clocks")
        answer = _duration(d)
        cands = [_duration(x) for x in _elapsed_misreads(h1, m1, h2, m2, d) if x > 0]
        notes = f"From {_digital(h1, m1)} to {_digital(h2, m2)} is {answer}."
        text = " ".join(_pick(rng, pool) for pool in _MIN_POOLS["hard"])
        return fig, text, answer, cands, notes
    text = " ".join(_pick(rng, pool) for pool in _MIN_POOLS[tier])
    return fig, text, answer, cands, notes


@generator("clock_read_minute", level=4, presets=_PRESETS)
def clock_read_minute(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, answer, cands, notes = _build_read_minute(rng, tier)
    distractors = pick_distractors(answer, cands, n=3)
    check(len(distractors) == 3, "the question needs three distinct distractors")
    figures.assert_no_leak(fig, [answer, *distractors])
    return dict(
        text=text,
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        distractors=distractors,
        working=[notes],
        figure=fig,
    )


def _read_clock_minute(question) -> str:
    fig = question.figure
    if fig["kind"] == "clock_set":
        by_caption = {c["caption"]: measure.read_clock(c) for c in fig["clocks"]}
        (h1, m1), (h2, m2) = by_caption["Start"], by_caption["End"]
        return _duration(((h2 - h1) * 60 + (m2 - m1)) % 720)
    return _digital(*measure.read_clock(fig))


figures.register_reader("clock_read_minute", _read_clock_minute)
