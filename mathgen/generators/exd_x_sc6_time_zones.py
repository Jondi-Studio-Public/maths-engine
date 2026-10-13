"""Exam-tier sources for four time-zone skills. Each is a short chain in which a time difference is found first and
then used to move a clock reading from one place to another.

* ``exd_x_sc6_longitude``: the angle between two meridians, the time difference it gives in minutes, and a local solar time in the other city.
* ``exd_x_sc6_aus_zones``: the time difference between two Australian or neighbouring places after daylight saving,
  then the local arrival time of an eastbound flight and of the westbound return.
* ``exd_x_sc6_call_window``: the time difference, the length of the daily overlap of two "reasonable hours to call"
  windows, and the latest start of a call that has to fit inside the overlap.
* ``exd_x_sc6_travel_zones``: the zone difference recovered from a flight's clock times (with a calendar-day label),
  then the arrival time of the return flight and the home time when the traveller phones from the destination.

A clock time is answered as two number parts: the hour on the 24-hour clock, then the minutes past the hour. Both
follow through from an earlier part (``floor`` and a remainder express them).
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    check,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_HALF = Fraction(1, 2)


def _h(value: Fraction) -> str:
    return stated_precision_string(round_half_up(Fraction(value), 1), 1)


def _num(value: Fraction):
    value = round_half_up(Fraction(value), 1)
    return int(value) if value.denominator == 1 else float(value)


def _clock(hours: Fraction) -> str:
    hours = Fraction(hours)
    total_minutes = int(hours * 60)
    h24, minute = divmod(total_minutes, 60)
    h24 %= 24  # a reading past midnight (or before it) is a time on the neighbouring day
    h12 = h24 % 12 or 12
    return f"{h12}:{minute:02d} {'am' if h24 < 12 else 'pm'}"


def _cand(value: Fraction, misconception: str | None, *, low=0, high=48) -> Distractor | None:
    value = Fraction(value)
    if not low < value < high:
        return None
    text = _h(value)
    return Distractor(text, misconception, Fraction(text))


def _tag(true: Fraction, cands) -> list[Distractor]:
    key = _h(true)
    picked = pick_distractors(key, [c for c in cands if c is not None], 3)
    check(len(picked) >= 2, f"too few distractors for key {key}")
    check(all(Fraction(d.value) != Fraction(key) for d in picked), "a distractor equals the key")
    return picked


def _part(prompt: str, value: Fraction, marks: int, *, sensible=(0, None), **extra) -> dict:
    text = _h(value)
    return dict(
        prompt=prompt,
        answer_kind="number",
        answer=_num(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=sensible,
        **extra,
    )


def _int_part(prompt: str, value: int, marks: int, **extra) -> dict:
    return dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value),
        answer_text=str(int(value)),
        answer_latex=latex_answer(str(int(value))),
        marks=marks,
        **extra,
    )


def _clock_parts(
    prompt: str, label: str, total_hours: Fraction, ft_hours: str, cands, working: list[str]
) -> list[dict]:
    """Two parts asking for a clock time the way people write it: the hour on the 24-hour clock, then the minutes.

    `cands` are ``(hours, misconception)`` pairs for wrong clock readings; each gives a wrong hour and a wrong
    minute. `ft_hours` is the follow-through expression for the time in decimal hours, or None.
    """
    hour, minute = divmod(int(Fraction(total_hours) * 60), 60)
    hour_cands, minute_cands = [], []
    for value, misconception in cands:
        if value is None or not 0 < Fraction(value) < 24:
            continue
        h, m = divmod(int(round_half_up(Fraction(value) * 60, 0)), 60)
        if h != hour:
            hour_cands.append(Distractor(str(h), misconception, h))
        if m != minute:
            minute_cands.append(Distractor(str(m), misconception, m))
    hour_cands += [Distractor(str(h), None, h) for h in (hour + 1, hour - 1, hour + 2) if 0 <= h < 24]
    minute_cands += [Distractor(str(m), None, m) for m in (0, 30, 15, 45) if m != minute]
    picked_h = pick_distractors(str(hour), hour_cands, 3)
    picked_m = pick_distractors(str(minute), minute_cands, 3)
    hour_part = _int_part(
        f"{prompt} Give the hour, on the 24-hour clock.",
        hour,
        1,
        distractors=picked_h,
        working=working,
        **({"follow_through": f"floor({ft_hours})"} if ft_hours else {}),
    )
    minute_part = _int_part(
        f"Give the number of minutes past the hour for the time in part ({label}).",
        minute,
        1,
        distractors=picked_m,
        working=[f"The time is {hour}:{minute:02d} on the 24-hour clock."],
        **({"follow_through": f"60*(({ft_hours}) - floor({ft_hours}))"} if ft_hours else {}),
    )
    return [hour_part, minute_part]


def _fmt_utc(offset: Fraction) -> str:
    minutes = int(offset * 60)
    return f"+{minutes // 60}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------------------------------------------
# Longitude
# ---------------------------------------------------------------------------------------------------------------

# Approximate longitudes in degrees, east positive.
_LONGITUDE = {
    "Perth": Fraction("115.9"),
    "Adelaide": Fraction("138.6"),
    "Sydney": Fraction("151.2"),
    "Melbourne": Fraction("144.9"),
    "Brisbane": Fraction("153.0"),
    "Darwin": Fraction("130.8"),
    "Hobart": Fraction("147.3"),
    "Auckland": Fraction("174.8"),
    "Tokyo": Fraction("139.7"),
    "Singapore": Fraction("103.8"),
    "Dubai": Fraction("55.3"),
    "Cape Town": Fraction("18.4"),
    "London": Fraction("-0.1"),
    "Reykjavik": Fraction("-21.9"),
    "Honolulu": Fraction("-157.9"),
    "Los Angeles": Fraction("-118.2"),
    "Denver": Fraction("-105.0"),
    "New York": Fraction("-74.0"),
    "Santiago": Fraction("-70.7"),
    "Buenos Aires": Fraction("-58.4"),
}


def _lon_text(value: Fraction) -> str:
    return f"{stated_precision_string(abs(value), 1)}° {'E' if value >= 0 else 'W'}"


@generator("exd_x_sc6_longitude", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc6_longitude(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        first, second = rng.sample(sorted(_LONGITUDE), 2)
        l1, l2 = _LONGITUDE[first], _LONGITUDE[second]
        # Keep the pair on the same side of the 180th meridian, so the eastern place is also the one reached by the
        # shorter eastward arc and its clock reads later on the same calendar day.
        west, east = (first, second) if l1 < l2 else (second, first)
        sep = abs(l1 - l2)
        if 35 <= sep <= 170:
            break
    else:
        raise AssertionError("no city pair in 1000 draws")
    # Independent route: the meridian angle by the signed longitudes of the two places.
    check(sep == _LONGITUDE[east] - _LONGITUDE[west], "separation disagrees with signed subtraction")
    minutes = int(round_half_up(sep * 4, 0))  # 15 degrees is 60 minutes, so 1 degree is 4 minutes
    forward = rng.random() < 0.5  # True: the clock reading is given in the western city
    low, high = (6, min(20, 23 - Fraction(minutes, 60))) if forward else (max(6, Fraction(minutes, 60) + 1), 20)
    t0 = Fraction(rng.randint(int(2 * low) + 1, int(2 * high)), 2)
    t0m = int(t0 * 60)
    result_m = t0m + minutes if forward else t0m - minutes
    check(60 <= result_m <= 23 * 60, "clock time left the day")
    result = Fraction(result_m, 60)
    shown = [first, second]
    stem = (
        f"{first} is at longitude {_lon_text(l1)} and {second} is at longitude {_lon_text(l2)}. Local solar time "
        "changes by 1 hour for every 15° of longitude, and places further east are ahead."
    )
    mag = abs(l1) + abs(l2)
    diff = abs(abs(l1) - abs(l2))
    a_cands = [
        _cand(mag, "adds-longitudes-same-side", high=361),
        _cand(diff, "subtracts-longitudes-opposite-sides", high=361),
        _cand(180 - sep, None, high=361),
        _cand(sep / 15, None, high=361),
    ]
    b_cands = [
        Distractor(str(int(round_half_up(sep, 0))), "forgets-to-divide-by-15", int(round_half_up(sep, 0))),
        Distractor(str(int(round_half_up(sep / 4, 0))), None, int(round_half_up(sep / 4, 0))),
        Distractor(str(minutes + 60), "off-by-one-hour", minutes + 60),
        Distractor(str(minutes - 60), "off-by-one-hour", minutes - 60) if minutes > 60 else None,
        Distractor(str(int(round_half_up(sep * 15, 0))), None, int(round_half_up(sep * 15, 0))),
    ]
    sign = 1 if forward else -1
    other = west if forward else east
    target = east if forward else west
    c_cands = [
        (Fraction(t0m - sign * minutes, 60), "wrong-zone-direction"),
        (Fraction(t0m + sign * int(round_half_up(sep, 0)), 60), "forgets-to-divide-by-15"),
        (result + 1, "off-by-one-hour"),
        (result - 1, "off-by-one-hour"),
    ]
    start = f"A webinar starts at {_clock(t0)} local solar time in {west if forward else east}."
    ft_hours = f"({t0m} {'+' if forward else '-'} part_b)/60"
    step = f"{_clock(t0)} {'+' if forward else '−'} {minutes} minutes = {_clock(result)}."
    parts = [
        _part(
            f"Find the angle between the meridians of {shown[0]} and {shown[1]}, in degrees, "
            "correct to 1 decimal place.",
            sep,
            2,
            sensible=(0, 180),
            working=[f"The smaller angle between {_lon_text(l1)} and {_lon_text(l2)} is {_h(sep)}°."],
            distractors=_tag(sep, a_cands),
        ),
        _int_part(
            f"Hence find the difference between the local solar times of {first} and {second}, in minutes, "
            "to the nearest minute.",
            minutes,
            1,
            follow_through="part_a * 4",
            sensible_range=(0, None),
            working=[f"Each degree is 4 minutes: {_h(sep)} × 4 = {minutes} minutes; {east} is ahead of {west}."],
            distractors=pick_distractors(str(minutes), [c for c in b_cands if c is not None], 3),
        ),
    ]
    parts += _clock_parts(
        f"{start} Find the local solar time in {target} at that moment.",
        "c",
        result,
        ft_hours,
        c_cands,
        [f"{target} is {'ahead of' if forward else 'behind'} {other}: {step}"],
    )
    return dict(text=stem, parts=parts)


# ---------------------------------------------------------------------------------------------------------------
# Routes shared by the three Australian generators
# ---------------------------------------------------------------------------------------------------------------

# place -> (standard UTC offset in hours, observes daylight saving). Every place that observes it is described as
# currently observing it, so the allowance applies.
_ZONES = {
    "Perth": (Fraction(8), False),
    "Darwin": (Fraction(19, 2), False),
    "Brisbane": (Fraction(10), False),
    "Port Moresby": (Fraction(10), False),
    "Jakarta": (Fraction(7), False),
    "Adelaide": (Fraction(19, 2), True),
    "Sydney": (Fraction(10), True),
    "Melbourne": (Fraction(10), True),
    "Auckland": (Fraction(12), True),
}
# (western place, eastern place, flight hours eastbound, flight hours westbound)
_ROUTES = [
    ("Perth", "Sydney", Fraction(4), Fraction(5)),
    ("Perth", "Melbourne", Fraction(7, 2), Fraction(9, 2)),
    ("Perth", "Adelaide", Fraction(3), Fraction(7, 2)),
    ("Perth", "Brisbane", Fraction(9, 2), Fraction(5)),
    ("Darwin", "Sydney", Fraction(4), Fraction(9, 2)),
    ("Darwin", "Brisbane", Fraction(3), Fraction(7, 2)),
    ("Brisbane", "Auckland", Fraction(7, 2), Fraction(4)),
    ("Sydney", "Auckland", Fraction(3), Fraction(7, 2)),
    ("Melbourne", "Auckland", Fraction(7, 2), Fraction(4)),
    ("Adelaide", "Sydney", Fraction(2), Fraction(2)),
    ("Brisbane", "Sydney", Fraction(3, 2), Fraction(3, 2)),
    ("Brisbane", "Melbourne", Fraction(2), Fraction(2)),
    ("Jakarta", "Perth", Fraction(9, 2), Fraction(9, 2)),
    ("Port Moresby", "Sydney", Fraction(9, 2), Fraction(9, 2)),
]


def _eff(place: str) -> Fraction:
    offset, observes = _ZONES[place]
    return offset + (1 if observes else 0)


def _std(place: str) -> Fraction:
    return _ZONES[place][0]


def _dst_sentence(place: str) -> str:
    standard = f"{place}'s standard time zone offset is UTC{_fmt_utc(_std(place))}."
    if _ZONES[place][1]:
        return f"{standard} {place} is currently observing daylight saving."
    return f"{standard} {place} does not observe daylight saving."


def _zone_stem(west: str, east: str) -> str:
    note = " Daylight saving advances local clocks by 1 hour." if _ZONES[west][1] or _ZONES[east][1] else ""
    return f"{_dst_sentence(west)} {_dst_sentence(east)}{note}"


def _route(rng: random.Random, standard_only: bool = False):
    routes = [r for r in _ROUTES if _std_diff(r[0], r[1]) > 0] if standard_only else _ROUTES
    west, east, d_east, d_west = rng.choice(routes)
    d = _eff(east) - _eff(west)
    check(d > 0, f"{east} is not ahead of {west}")
    return west, east, d_east, d_west, d


def _half_hours(rng: random.Random, lo: int, hi: int) -> Fraction:
    return Fraction(rng.randint(2 * lo, 2 * hi), 2)


def _std_diff(west: str, east: str) -> Fraction:
    return _std(east) - _std(west)


@generator("exd_x_sc6_aus_zones", level=12, presets={"exam": dict(tier="exam")})
def exd_x_sc6_aus_zones(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    west, east, d_east, d_west, d = _route(rng)
    for _ in range(1000):
        dep1 = _half_hours(rng, 5, 17)
        dep2 = _half_hours(rng, 5, 17)
        arr1 = dep1 + d_east + d
        arr2 = dep2 + d_west - d
        if arr1 < 24 and 0 < arr2 < 24:
            break
    else:
        raise AssertionError("no departure times in 1000 draws")
    std = _std_diff(west, east)
    # Independent route: the effective offsets from the stated sentences.
    check(_eff(east) - _eff(west) == d, "effective offsets disagree")
    a_cands = [
        _cand(std, "ignores-daylight-saving", low=-1),
        _cand(d + 1, "applies-daylight-saving-to-wrong-place"),
        _cand(abs(d - 1), "subtracts-daylight-saving"),
        _cand(d + _HALF, None),
        _cand(2 * d, None),
        _cand(_eff(east) + _eff(west), None),
    ]
    b_cands = [
        (dep1 + d_east, "forgets-zone-difference"),
        (dep1 + d_east - d, "wrong-zone-direction"),
        (dep1 + d_east + std, "ignores-daylight-saving"),
    ]
    c_cands = [
        (dep2 + d_west, "forgets-zone-difference"),
        (dep2 + d_west + d, "wrong-zone-direction"),
        (dep2 + d_west - std, "ignores-daylight-saving"),
    ]
    parts = [
        _part(
            f"By how many hours is the local time in {east} ahead of the local time in {west}? "
            "Give your answer in hours, to 1 decimal place.",
            d,
            2,
            working=[
                f"{west}: UTC{_fmt_utc(_eff(west))}. {east}: UTC{_fmt_utc(_eff(east))}.",
                f"Difference = {_h(d)} hours.",
            ],
            distractors=_tag(d, a_cands),
        )
    ]
    parts += _clock_parts(
        f"A flight leaves {west} at {_clock(dep1)} local time and takes {_h(d_east)} hours to reach {east}. "
        f"Find the local time in {east} when it lands.",
        "b",
        arr1,
        f"{dep1} + {d_east} + part_a",
        b_cands,
        [f"Landing time in {west}'s time = {_clock(dep1 + d_east)}; add {_h(d)} hours: {_clock(arr1)}."],
    )
    parts += _clock_parts(
        f"The return flight leaves {east} at {_clock(dep2)} local time and takes {_h(d_west)} hours. "
        f"Find the local time in {west} when it lands.",
        "d",
        arr2,
        f"{dep2} + {d_west} - part_a",
        c_cands,
        [f"{_clock(dep2 + d_west)} in {east}'s time; subtract {_h(d)} hours: {_clock(arr2)}."],
    )
    return dict(text=_zone_stem(west, east), parts=parts)


_WINDOWS = [(8, 22), (9, 21), (7, 21), (8, 20), (9, 20)]


@generator("exd_x_sc6_call_window", level=12, presets={"exam": dict(tier="exam")})
def exd_x_sc6_call_window(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    west, east, _, _, d = _route(rng)
    for _ in range(1000):
        a1, a2 = rng.choice(_WINDOWS)
        b1, b2 = rng.choice(_WINDOWS)
        lo = max(a1, b1 - d)
        hi = min(a2, b2 - d)
        overlap = hi - lo
        m = Fraction(rng.choice([1, 3, 2, 5, 4, 6]), 2)
        if overlap >= m + 1 and overlap != a2 - a1 and overlap != b2 - b1:
            break
    else:
        raise AssertionError("no windows in 1000 draws")
    latest = hi - m
    # Independent route: shift east's window into west's clock by subtracting d, then intersect hour by hour.
    free = [h for h in range(24 * 2) if a1 * 2 <= h < a2 * 2 and (b1 - d) * 2 <= h < (b2 - d) * 2]
    check(Fraction(len(free), 2) == overlap, "overlap disagrees with hour-by-hour count")
    std = _std_diff(west, east)
    std_overlap = max(Fraction(0), min(a2, b2 - std) - max(a1, b1 - std))
    a_cands = [
        _cand(std, "ignores-daylight-saving", low=-1),
        _cand(d + 1, "applies-daylight-saving-to-wrong-place"),
        _cand(abs(d - 1), "subtracts-daylight-saving"),
        _cand(d + _HALF, None),
        _cand(2 * d, None),
    ]
    b_cands = [
        _cand(Fraction(a2 - a1), "ignores-time-difference"),
        _cand(Fraction(a2 - a1 + b2 - b1), "adds-instead-of-intersects"),
        _cand(max(Fraction(0), min(a2, b2 + d) - max(a1, b1 + d)), "wrong-zone-direction"),
        _cand(std_overlap, "ignores-daylight-saving"),
    ]
    c_cands = [
        (hi, "forgets-call-length"),
        (hi - m + d, None),
        (lo + m, None),
        (Fraction(b2) - m, None),
    ]
    stem = (
        f"{_zone_stem(west, east)} In {west} the reasonable hours to call are {_clock(a1)} to {_clock(a2)}, local "
        f"time. In {east} they are {_clock(b1)} to {_clock(b2)}, local time."
    )
    parts = [
        _part(
            f"By how many hours is the local time in {east} ahead of the local time in {west}? "
            "Give your answer in hours, to 1 decimal place.",
            d,
            2,
            working=[f"Effective offsets differ by {_h(d)} hours."],
            distractors=_tag(d, a_cands),
        ),
        _part(
            "Find the length of the daily window, in hours, during which both places' reasonable hours to call "
            "overlap. Give your answer to 1 decimal place.",
            overlap,
            2,
            follow_through=f"Max(0, Min({a2}, {b2} - part_a) - Max({a1}, {b1} - part_a))",
            working=[f"In {west}'s time the windows are {a1} to {a2} and {_h(b1 - d)} to {_h(b2 - d)}."],
            sensible=(0, None),
            distractors=_tag(overlap, b_cands),
        ),
    ]
    parts += _clock_parts(
        f"A conference call lasting {_h(m)} hours must start and finish inside the overlap. "
        f"Find the latest time it can start, in {west}'s local time.",
        "c",
        latest,
        f"Min({a2}, {b2} - part_a) - {m}",
        c_cands,
        [f"The overlap closes at {_clock(hi)}; start no later than {_clock(latest)}."],
    )
    return dict(text=stem, parts=parts)


@generator("exd_x_sc6_travel_zones", level=12, presets={"exam": dict(tier="exam")})
def exd_x_sc6_travel_zones(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    west, east, d_east, d_west, _ = _route(rng, standard_only=True)
    d = _std_diff(west, east)  # both cities on standard time: the zone difference the learner has to recover
    check(d > 0, "no zone difference")
    eastbound = rng.random() < 0.5
    origin, dest = (west, east) if eastbound else (east, west)
    dur = d_east if eastbound else d_west
    sign = 1 if eastbound else -1
    for _ in range(1000):
        dep = _half_hours(rng, 6, 23)
        arrival_total = dep + dur + sign * d  # clock reading at the destination, counted from the origin's midnight
        day = arrival_total // 24  # 0 same day, 1 next day, -1 previous day
        arr_clock = arrival_total - 24 * day
        if day != 0 or rng.random() < 0.3:
            break
    day_label = {0: "on the same day", 1: "on the next day", -1: "on the previous day"}[int(day)]
    dep_back = _half_hours(rng, 6, 16)
    dur_back = d_west if eastbound else d_east
    back_arrival = dep_back + dur_back - sign * d
    if not 0 < back_arrival < 24:
        dep_back = _half_hours(rng, 6, 12) if back_arrival >= 24 else _half_hours(rng, 12, 18)
        back_arrival = dep_back + dur_back - sign * d
    call_time = _half_hours(rng, 8, 20)
    call_home = call_time - sign * d
    check(0 < back_arrival < 24 and 0 < call_home < 24, "return times fell outside the day")
    # Independent route: rebuild the clock reading from the recovered difference.
    check(dep + dur + sign * d - 24 * day == arr_clock, "arrival does not reproduce")
    direction = "east" if eastbound else "west"
    stem = (
        f"Both {origin} and {dest} are on standard time. A traveller leaves {origin} at {_clock(dep)} local time "
        f"and flies {direction} to {dest}. The flight takes {_h(dur)} hours and lands at {_clock(arr_clock)} "
        f"{dest} local time, {day_label}."
    )
    word_back = "east" if not eastbound else "west"
    a_cands = [
        _cand(dur, "reports-duration-not-offset"),
        _cand(d + 1, "off-by-one-hour"),
        _cand(d - 1, "off-by-one-hour"),
        _cand(abs(arr_clock - dep - dur), "omits-day-adjustment") if day else None,
        _cand(d + _HALF, None),
    ]
    b_cands = [
        (dep_back + dur_back, "forgets-zone-difference"),
        (dep_back + dur_back + sign * d, "wrong-zone-direction"),
    ]
    c_cands = [
        (call_time, "forgets-zone-difference"),
        (call_time + sign * d, "wrong-zone-direction"),
        (call_time - sign * d - 1, "off-by-one-hour"),
    ]
    sign_text = "+" if -sign == 1 else "-"
    parts = [
        _part(
            f"By how many hours do the clocks in {origin} and {dest} differ? Give your answer in hours, to "
            "1 decimal place.",
            d,
            2,
            working=[
                f"Landing time on {origin}'s clock = {_clock(dep + dur)}; {dest} reads {_clock(arr_clock)} "
                f"{day_label.replace('on the ', '')}.",
                f"The difference is {_h(d)} hours.",
            ],
            distractors=_tag(d, a_cands),
        )
    ]
    parts += _clock_parts(
        f"On the return the traveller leaves {dest} at {_clock(dep_back)} local time, flying {word_back}. "
        f"This flight takes {_h(dur_back)} hours. Find the local time in {origin} when it lands.",
        "b",
        back_arrival,
        f"{dep_back} + {dur_back} {sign_text} part_a",
        b_cands,
        [f"{_clock(dep_back + dur_back)} on {dest}'s clock; adjust by {_h(d)} hours: {_clock(back_arrival)}."],
    )
    parts += _clock_parts(
        f"During the stay, at {_clock(call_time)} local time in {dest}, the traveller phones home to {origin}. "
        f"What is the local time in {origin} at that moment?",
        "d",
        call_home,
        f"{call_time} {'-' if sign == 1 else '+'} part_a",
        c_cands,
        [f"{origin} is {'behind' if sign == 1 else 'ahead of'} {dest} by {_h(d)} hours: {_clock(call_home)}."],
    )
    return dict(text=stem, parts=parts)
