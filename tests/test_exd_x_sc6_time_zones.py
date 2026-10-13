"""Exam tiers of four time-zone skills: longitude, Australian zones with daylight saving, call windows and travel."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.engine import grade
from tests.exd_x_sc6_support import (
    assert_answer_space,
    assert_follow_through,
    assert_grading,
    assert_new_shapes,
    assert_no_text_defects,
    assert_source_wired,
    canonical,
    exam,
    frac,
    skill_of,
)

LONGITUDE = skill_of("senior-find-time-differences-between-two-places-on-earth")
AUS = skill_of("senior-solve-problems-involving-time-zones-in-australia-and")
CALL = skill_of("senior-solve-problems-associated-with-time-zones-for-example")
TRAVEL = skill_of("senior-solve-problems-relating-to-travelling-east-and-west")
ALL = [
    (LONGITUDE, "exd_x_sc6_longitude"),
    (AUS, "exd_x_sc6_aus_zones"),
    (CALL, "exd_x_sc6_call_window"),
    (TRAVEL, "exd_x_sc6_travel_zones"),
]
_CLOCK = re.compile(r"(\d{1,2}):(\d\d) (am|pm)")


def hours(text: str) -> Fraction:
    """A clock reading such as '3:30 pm' as hours after midnight."""
    m = _CLOCK.search(text)
    h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
    return Fraction(h) + Fraction(int(m.group(2)), 60)


def fmt1(value: Fraction) -> str:
    """Round half up to 1 decimal place and print with exactly one."""
    tenths = int(value * 10 + Fraction(1, 2))
    return f"{tenths // 10}.{tenths % 10}"


def hm(total_hours: Fraction) -> tuple[str, str]:
    """Hour on the 24-hour clock and minutes past it, each as the digits a learner types."""
    total = int(total_hours * 60 + Fraction(1, 2))
    return str(total // 60), str(total % 60)


def _offsets(stem: str) -> dict[str, Fraction]:
    out = {}
    for m in re.finditer(
        r"(\w[\w ]*?)'s standard time zone offset is UTC\+(\d+):(\d\d)\. \1 (is currently observing|does not observe) daylight saving",
        stem,
    ):
        place, h, mi, dst = m.groups()
        out[place] = Fraction(int(h)) + Fraction(int(mi), 60) + (1 if dst.startswith("is") else 0)
    return out


# ---------------------------------------------------------------- longitude


def _longitudes(stem: str) -> dict[str, Fraction]:
    return {
        m.group(1): Fraction(m.group(2)) * (1 if m.group(3) == "E" else -1)
        for m in re.finditer(r"([A-Z][A-Za-z ]+?) is at longitude (\d+\.\d)° ([EW])", stem)
    }


def oracle_longitude(q) -> dict[str, str]:
    lon = _longitudes(q.stem)
    (n1, l1), (n2, l2) = lon.items()
    sep = abs(l1 - l2)
    minutes = int(sep * 4 + Fraction(1, 2))
    east = n1 if l1 > l2 else n2
    t0 = hours(q.parts[2].prompt)
    target = re.search(r"local solar time in ([A-Z][A-Za-z ]+?) at that moment", q.parts[2].prompt).group(1)
    result = t0 + Fraction(minutes, 60) if target == east else t0 - Fraction(minutes, 60)
    h, m = hm(result)
    return {"a": fmt1(sep), "b": str(minutes), "c": h, "d": m}


# ---------------------------------------------------------------- Australian zones


def oracle_aus(q) -> dict[str, str]:
    off = _offsets(q.stem)
    ahead = re.search(r"local time in (\w[\w ]*?) ahead of the local time in (\w[\w ]*?)\?", q.parts[0].prompt)
    east, west = ahead.groups()
    d = off[east] - off[west]
    dep1 = hours(q.parts[1].prompt)
    dur1 = Fraction(re.search(r"takes ([\d.]+) hours", q.parts[1].prompt).group(1))
    dep2 = hours(q.parts[3].prompt)
    dur2 = Fraction(re.search(r"takes ([\d.]+) hours", q.parts[3].prompt).group(1))
    (h1, m1), (h2, m2) = hm(dep1 + dur1 + d), hm(dep2 + dur2 - d)
    return {"a": fmt1(d), "b": h1, "c": m1, "d": h2, "e": m2}


# ---------------------------------------------------------------- call windows


def oracle_call(q) -> dict[str, str]:
    off = _offsets(q.stem)
    east, west = re.search(
        r"local time in (\w[\w ]*?) ahead of the local time in (\w[\w ]*?)\?", q.parts[0].prompt
    ).groups()
    d = off[east] - off[west]
    windows = {}
    for m in re.finditer(
        r"In ([A-Z][\w ]*?) (?:the reasonable hours to call are|they are) (\d+:\d\d [ap]m) to (\d+:\d\d [ap]m)", q.stem
    ):
        windows[m.group(1)] = (hours(m.group(2)), hours(m.group(3)))
    if len(windows) < 2:  # second window is introduced as "In X they are"
        windows = {}
        pairs = re.findall(r"(\d+:\d\d [ap]m) to (\d+:\d\d [ap]m)", q.stem)
        windows[west], windows[east] = [(hours(a), hours(b)) for a, b in pairs]
    # Count the free half hours in `west` time directly.
    free = [
        h / 2
        for h in range(48)
        if windows[west][0] <= Fraction(h, 2) < windows[west][1]
        and windows[east][0] - d <= Fraction(h, 2) < windows[east][1] - d
    ]
    overlap = Fraction(len(free), 2)
    m_len = Fraction(re.search(r"lasting ([\d.]+) hours", q.parts[2].prompt).group(1))
    latest = max(free) + Fraction(1, 2) - m_len
    h, m = hm(Fraction(latest))
    return {"a": fmt1(d), "b": fmt1(overlap), "c": h, "d": m}


# ---------------------------------------------------------------- travel


def oracle_travel(q) -> dict[str, str]:
    stem = q.stem
    origin, dest = re.search(r"leaves ([A-Z][\w ]*?) at .*? flies (?:east|west) to ([A-Z][\w ]*?)\.", stem).groups()
    east = " flies east " in stem
    dep = hours(stem)
    dur = Fraction(re.search(r"flight takes ([\d.]+) hours", stem).group(1))
    arr = hours(stem[stem.index("lands at") :])
    day = {"on the same day": 0, "on the next day": 1, "on the previous day": -1}[
        re.search(r"(on the (?:same|next|previous) day)", stem).group(1)
    ]
    sign = 1 if east else -1
    d = abs(arr + 24 * day - dep - dur)
    dep_b = hours(q.parts[1].prompt)
    dur_b = Fraction(re.search(r"takes ([\d.]+) hours", q.parts[1].prompt).group(1))
    flying_east_back = "flying east" in q.parts[1].prompt
    assert flying_east_back == (not east)
    back = dep_b + dur_b + (d if flying_east_back else -d)
    call = hours(q.parts[3].prompt)
    home = call - sign * d
    assert origin in q.parts[1].prompt and dest in q.parts[1].prompt
    (h1, m1), (h2, m2) = hm(back), hm(home)
    return {"a": fmt1(d), "b": h1, "c": m1, "d": h2, "e": m2}


ORACLES = {
    "exd_x_sc6_longitude": oracle_longitude,
    "exd_x_sc6_aus_zones": oracle_aus,
    "exd_x_sc6_call_window": oracle_call,
    "exd_x_sc6_travel_zones": oracle_travel,
}


@pytest.mark.parametrize(("skill", "topic"), ALL)
def test_source_wired(skill, topic):
    assert_source_wired(skill, topic)


@pytest.mark.parametrize(("skill", "topic"), ALL)
def test_shapes_and_text(skill, topic):
    assert_new_shapes(skill)
    assert_no_text_defects(skill)


@pytest.mark.parametrize(("skill", "topic"), ALL)
def test_answer_space(skill, topic):
    assert_answer_space(skill, minimum=100)


@pytest.mark.parametrize(("skill", "topic"), ALL)
def test_chain_shape(skill, topic):
    for seed in range(40):
        q = exam(skill, seed)
        assert [p.label for p in q.parts] == list("abcde")[: len(q.parts)]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert sum(p.follow_through is not None for p in q.parts) >= 2


@pytest.mark.parametrize(("skill", "topic"), ALL)
def test_key_rederived_from_prompt(skill, topic):
    oracle = ORACLES[topic]
    for seed in range(300):
        q = exam(skill, seed)
        got = {k: frac(v) for k, v in oracle(q).items()}
        want = {k: frac(v) for k, v in canonical(q).items()}
        assert got == want, (seed, q.stem, [p.prompt for p in q.parts])


@pytest.mark.parametrize(("skill", "topic"), ALL)
def test_grader_accepts_key_rejects_distractors(skill, topic):
    assert_grading(skill)


# ---------------------------------------------------------------- follow-through


def test_longitude_follow_through():
    checked = 0
    for seed in range(150):
        q = exam(LONGITUDE, seed)
        # A wrong angle in (a) carries into the minutes in (b) and the clock time in (c) and (d); an impossible angle
        # (over 180 degrees) carries nothing, as in a marked exam.
        wrongs = [d for d in q.parts[0].distractors if 0 < frac(d) < 180]
        if not wrongs:
            continue
        wrong = wrongs[0]
        minutes = int(frac(wrong) * 4 + Fraction(1, 2))
        t0 = hours(q.parts[2].prompt)
        target = re.search(r"local solar time in ([A-Z][A-Za-z ]+?) at that moment", q.parts[2].prompt).group(1)
        lon = _longitudes(q.stem)
        east = max(lon, key=lon.get)
        c = t0 + Fraction(minutes, 60) if target == east else t0 - Fraction(minutes, 60)
        if not 0 < c < 48:
            continue
        h, m = hm(c)
        if m == str(minutes):  # a reply that merely repeats the earlier one earns no follow-through
            continue
        assert_follow_through(LONGITUDE, q, {"a": wrong, "b": str(minutes), "c": h, "d": m}, "a")
        checked += 1
    assert checked >= 100


def test_aus_follow_through():
    checked = 0
    for seed in range(150):
        q = exam(AUS, seed)
        wrong = next(d for d in q.parts[0].distractors if frac(d) != 0)
        d_wrong = frac(wrong)
        dep1 = hours(q.parts[1].prompt)
        dur1 = Fraction(re.search(r"takes ([\d.]+) hours", q.parts[1].prompt).group(1))
        dep2 = hours(q.parts[3].prompt)
        dur2 = Fraction(re.search(r"takes ([\d.]+) hours", q.parts[3].prompt).group(1))
        b, c = dep1 + dur1 + d_wrong, dep2 + dur2 - d_wrong
        if not (0 < b < 48 and 0 < c < 48):
            continue
        (h1, m1), (h2, m2) = hm(b), hm(c)
        assert_follow_through(AUS, q, {"a": wrong, "b": h1, "c": m1, "d": h2, "e": m2}, "a")
        checked += 1
    assert checked >= 100


def test_call_window_follow_through():
    checked = 0
    for seed in range(150):
        q = exam(CALL, seed)
        wrong = next(d for d in q.parts[0].distractors if frac(d) != 0)
        d = frac(wrong)
        pairs = [(hours(a), hours(b)) for a, b in re.findall(r"(\d+:\d\d [ap]m) to (\d+:\d\d [ap]m)", q.stem)]
        (a1, a2), (b1, b2) = pairs
        m_len = Fraction(re.search(r"lasting ([\d.]+) hours", q.parts[2].prompt).group(1))
        overlap = max(Fraction(0), min(a2, b2 - d) - max(a1, b1 - d))
        latest = min(a2, b2 - d) - m_len
        if overlap == 0 or not 0 < latest < 48:
            continue
        h, m = hm(latest)
        replies = {"a": wrong, "b": fmt1(overlap), "c": h, "d": m}
        assert_follow_through(CALL, q, replies, "a")
        checked += 1
    assert checked >= 100


def test_travel_follow_through():
    checked = 0
    for seed in range(150):
        q = exam(TRAVEL, seed)
        wrong = next(d for d in q.parts[0].distractors if frac(d) != 0)
        d = frac(wrong)
        east = " flies east " in q.stem
        sign = 1 if east else -1
        dep_b = hours(q.parts[1].prompt)
        dur_b = Fraction(re.search(r"takes ([\d.]+) hours", q.parts[1].prompt).group(1))
        back = dep_b + dur_b + (d if "flying east" in q.parts[1].prompt else -d)
        home = hours(q.parts[3].prompt) - sign * d
        if not (0 < back < 48 and 0 < home < 48):
            continue
        (h1, m1), (h2, m2) = hm(back), hm(home)
        assert_follow_through(TRAVEL, q, {"a": wrong, "b": h1, "c": m1, "d": h2, "e": m2}, "a")
        checked += 1
    assert checked >= 100


@pytest.mark.parametrize(("skill", "topic"), ALL)
def test_all_wrong_earns_nothing(skill, topic):
    for seed in range(30):
        assert grade(skill, exam(skill, seed), {"a": "99.9", "b": "98.8", "c": "97.7"}).marks == 0


@pytest.mark.parametrize(("skill", "topic"), ALL)
def test_deterministic(skill, topic):
    assert exam(skill, 7).to_dict() == exam(skill, 7).to_dict()
