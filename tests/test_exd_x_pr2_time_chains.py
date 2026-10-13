"""Exam sources of four primary time skills: connected multi-step problems with follow-through marking."""

from __future__ import annotations

import re
from datetime import date

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILLS = {s.id: s for s in load_skills()[1]}
_CLOCK = re.compile(r"(\d{1,2}):(\d\d) (am|pm)")
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]
_MONTHS += ["November", "December"]


def _minutes(match) -> int:
    h, m, ap = int(match.group(1)), int(match.group(2)), match.group(3)
    return (h % 12 + (12 if ap == "pm" else 0)) * 60 + m


def _clock_times(text: str) -> list[int]:
    return [_minutes(m) for m in _CLOCK.finditer(text)]


def _trip(q):
    t1, t2 = _clock_times(q.stem)
    brk = int(
        re.search(r"(\d+)-minute|waits (\d+) minutes", q.stem).group(1) or re.search(r"waits (\d+)", q.stem).group(1)
    )
    brk2 = int(re.search(r"a (\d+)-minute stop", q.parts[2].prompt).group(1))
    total = t2 - t1
    return {"a": total, "b": total - brk, "c": total - brk + brk2}, {"b": f"a-{brk}", "c": f"b+{brk2}"}


def _timetable(q):
    t1, t2, t3, t4 = _clock_times(q.stem)
    return {"a": t3 - t2, "b": t4 - t1, "c": (t4 - t1) - (t3 - t2)}, {}


def _dates(text: str) -> list[date]:
    found = re.findall(r"(\d+) (" + "|".join(_MONTHS) + ")", text)
    return [date(2023, _MONTHS.index(mo) + 1, int(d)) for d, mo in found]


def _calendar(q):
    d1, d2, d3 = _dates(q.stem)
    a, b = (d2 - d1).days, (d3 - d2).days
    return {"a": a, "b": b, "c": (a + b) // 7}, {}


_UNIT_WORDS = {"hour": ("minute", 60), "minute": ("second", 60), "day": ("hour", 24)}


def _unit_parse(q):
    m = re.search(r"(\d+) (days?|hours?|minutes?) (\d+) (hours?|minutes?|seconds?)", q.stem)
    big, bu, small = int(m.group(1)), m.group(2).rstrip("s"), int(m.group(3))
    rest = re.sub(re.escape(m.group(0)), "", q.stem)
    n = int(re.search(r"(\d+)", rest).group(1))
    su, factor = _UNIT_WORDS[bu]
    return big, small, n, factor


def _tenths(total: int, factor: int) -> str:
    """total / factor to 1 decimal place, halves rounded up, by integer arithmetic."""
    t = (total * 20 + factor) // (2 * factor)
    return f"{t // 10}.{t % 10}"


def _units(q):
    big, small, n, factor = _unit_parse(q)
    each = big * factor + small
    total = n * each
    return {"a": each, "b": total, "c": _tenths(total, factor)}, {}


CASES = {
    "exd_x_pr2_trip_times": ("elapsed-time", _trip),
    "exd_x_pr2_timetable_connection": ("f10-timetable-itinerary-duration", _timetable),
    "exd_x_pr2_calendar_span": ("f10-describe-calendar-duration", _calendar),
    "exd_x_pr2_unit_totals": ("f10-time-units-and-nearest-minute", _units),
}
HARD_TOPICS = {
    "exd_x_pr2_trip_times": ("elapsed_time", {}),
    "exd_x_pr2_timetable_connection": ("timetable_itinerary_duration", {"tier": "hard"}),
    "exd_x_pr2_calendar_span": ("describe_calendar_duration", {"tier": "hard"}),
    "exd_x_pr2_unit_totals": ("time_units_and_nearest_minute", {"tier": "hard"}),
}
TOPICS = list(CASES)
FT_PARTS = {
    "exd_x_pr2_trip_times": [False, True, True],
    "exd_x_pr2_timetable_connection": [False, False, True],
    "exd_x_pr2_calendar_span": [False, False, True],
    "exd_x_pr2_unit_totals": [False, True, True],
}


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


@pytest.mark.parametrize("topic", TOPICS)
def test_source_wired(topic):
    skill = _SKILLS[CASES[topic][0]]
    exam = [s for s in skill.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(topic, ["exam"])]
    assert exam[0].params == {"tier": "exam"}


@pytest.mark.parametrize("topic", TOPICS)
def test_prompt_variety_and_new_shapes(topic):
    exam = {prompt_shape(_exam(topic, s)) for s in range(60)}
    hard_topic, params = HARD_TOPICS[topic]
    hard = {prompt_shape(make(hard_topic, seed=s, **params)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


@pytest.mark.parametrize("topic", TOPICS)
def test_shape_and_no_text_defects(topic):
    for seed in range(60):
        q = _exam(topic, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == FT_PARTS[topic]
        assert all(p.answer_kind == "number" and p.sensible_range == (0, None) for p in q.parts)
        assert find_text_defects(q) == []


@pytest.mark.parametrize("topic", TOPICS)
def test_key_rederived_from_prompt(topic):
    oracle = CASES[topic][1]
    for seed in range(300):
        q = _exam(topic, seed)
        expected, _ = oracle(q)
        assert {k: str(v) for k, v in expected.items()} == _canonical(q), (seed, q.text)


@pytest.mark.parametrize("topic", TOPICS)
def test_key_not_in_prompt_and_prompts_do_not_end_on_equation(topic):
    for seed in range(100):
        q = _exam(topic, seed)
        for p in q.parts:
            assert not p.prompt.rstrip().endswith("=")
        # no part's key appears as a number in the shared stem
        shown = " ".join([q.stem, *(p.prompt for p in q.parts)])
        numbers = set(re.findall(r"\d+(?:\.\d+)?", shown))
        for p in q.parts:
            assert p.answer_text not in numbers, (seed, p.label, q.stem)


@pytest.mark.parametrize("topic", TOPICS)
def test_grader_accepts_key_and_rejects_distractors(topic):
    skill = _SKILLS[CASES[topic][0]]
    for seed in range(120):
        q = _exam(topic, seed)
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (seed, q.answer_text)
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                result = grade(skill, q, {**canon, p.label: str(d)})
                assert result.parts[p.label].marks == 0, (seed, p.label, str(d))
                assert not result.correct


@pytest.mark.parametrize("topic", TOPICS)
def test_answer_space(topic):
    answers = [_exam(topic, s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) / len(answers) <= 0.05


@pytest.mark.parametrize("topic", TOPICS)
def test_deterministic(topic):
    assert _exam(topic, 7).to_dict() == _exam(topic, 7).to_dict()


# -- follow-through: a wrong earlier answer used correctly earns the later marks ------------------


def _assert_follow_through(skill, q, reply, wrong_label, later):
    result = grade(skill, q, reply)
    assert result.parts[wrong_label].marks == 0
    for label in later:
        part = result.parts[label]
        assert part.marks == part.max_marks, (q.text, label, reply)
        assert part.follow_through is True
    assert not result.correct


def test_trip_follow_through():
    skill = _SKILLS["elapsed-time"]
    checked = 0
    for seed in range(80):
        q = _exam("exd_x_pr2_trip_times", seed)
        _, steps = _trip(q)
        brk, brk2 = int(steps["b"][2:]), int(steps["c"][2:])
        for d in q.parts[0].distractors:
            a = int(str(d))
            if a - brk > 0:
                reply = {"a": str(a), "b": str(a - brk), "c": str(a - brk + brk2)}
                _assert_follow_through(skill, q, reply, "a", "bc")
                checked += 1
                break
    assert checked >= 40


def test_timetable_follow_through():
    skill = _SKILLS["f10-timetable-itinerary-duration"]
    checked = 0
    for seed in range(80):
        q = _exam("exd_x_pr2_timetable_connection", seed)
        canon = _canonical(q)
        for d in q.parts[1].distractors:
            c = int(str(d)) - int(canon["a"])
            if c > 0 and str(c) not in (canon["a"], str(d)):
                reply = {"a": canon["a"], "b": str(d), "c": str(c)}
                _assert_follow_through(skill, q, reply, "b", "c")
                checked += 1
                break
    assert checked >= 40


def test_calendar_follow_through():
    skill = _SKILLS["f10-describe-calendar-duration"]
    checked = 0
    for seed in range(80):
        q = _exam("exd_x_pr2_calendar_span", seed)
        canon = _canonical(q)
        for d in q.parts[0].distractors:
            weeks = (int(str(d)) + int(canon["b"])) // 7
            if weeks > 0 and str(weeks) not in (str(d), canon["b"], canon["c"]):
                reply = {"a": str(d), "b": canon["b"], "c": str(weeks)}
                _assert_follow_through(skill, q, reply, "a", "c")
                checked += 1
                break
    assert checked >= 40


def test_units_follow_through():
    skill = _SKILLS["f10-time-units-and-nearest-minute"]
    checked = 0
    for seed in range(80):
        q = _exam("exd_x_pr2_unit_totals", seed)
        _, _, n, factor = _unit_parse(q)
        d = q.parts[0].distractors[0]
        b = n * int(str(d))
        c = _tenths(b, factor)
        reply = {"a": str(d), "b": str(b), "c": c}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        checked += 1
    assert checked == 80


@pytest.mark.parametrize("topic", TOPICS)
def test_all_wrong_earns_zero(topic):
    skill = _SKILLS[CASES[topic][0]]
    for seed in range(40):
        q = _exam(topic, seed)
        result = grade(skill, q, {"a": "9999", "b": "1", "c": "2"})
        assert result.marks == 0 and not result.correct
