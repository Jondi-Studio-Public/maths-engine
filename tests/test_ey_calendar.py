"""The ``calendar`` and ``day_strip`` figure kinds and the five early-years skills built on them.

Each skill is generated through its own sources at every tier and 60 seeds: the figure is valid and does
not show the key, the registered reader re-derives the key from the figure and prompt, the real grader
accepts the key and rejects every distractor, and every prompt fits the read-aloud cap. The kinds refuse
bad specs, and the calendar agrees with Python's own ``calendar`` module.
"""

from __future__ import annotations

import calendar as pycal
import functools
import re
import xml.etree.ElementTree as ET

import pytest

from mathgen import figures, make, svg
from mathgen.figure_kinds import calendar as kinds
from mathgen.grading.typed import grade, mcq_options
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills
from scripts import skill_contract

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

SKILLS = {
    "ey-calendar-count-days": ("ey_calendar_span", "number", ["ac9m1m03"]),
    "ey-calendar-weekday-of-date": ("ey_calendar_weekday", "mcq", ["ac9m1m03", "ac9m2m03"]),
    "ey-calendar-days-between": ("ey_calendar_between", "number", ["ac9m2m03"]),
    "ey-day-strip-sequence": ("ey_day_strip", "number", ["ac9mfm02"]),
    "ey-calendar-days-months-order": ("ey_calendar_order", "mcq", ["ac9mfm02", "ac9m1m03"]),
}


@functools.lru_cache(maxsize=1)
def _skills():
    return {s.id: s for s in load_skills()[1]}


@functools.lru_cache(maxsize=None)
def _q(skill_id, tier, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=tier, **(spec.params or {}))


def _is_right(skill_id, q, text):
    if SKILLS[skill_id][1] == "mcq":
        return grade("mcq", q, mcq_options(q).index(text)).correct
    return grade("number", q, text).correct


# ------------------------------------------------------------------ the calendar kind


@pytest.mark.parametrize("year,month", [(2024, 2), (2025, 2), (2026, 10), (2028, 2), (2100, 2), (2026, 12)])
@pytest.mark.parametrize("week_start", ["mon", "sun"])
def test_calendar_matches_the_python_calendar(year, month, week_start):
    fig = kinds.calendar_month(year=year, month=month, week_start=week_start, alt="a month")
    assert fig["days"] == pycal.monthrange(year, month)[1]
    for day in range(1, fig["days"] + 1):
        assert kinds.weekday_of(fig, day) == kinds.WEEKDAY_NAMES[pycal.weekday(year, month, day)]
    assert kinds.week_order(week_start)[0] == ("Monday" if week_start == "mon" else "Sunday")


def test_february_has_28_or_29_days():
    assert kinds.calendar_month(year=2025, month=2, alt="x")["days"] == 28
    assert kinds.calendar_month(year=2028, month=2, alt="x")["days"] == 29
    assert kinds.calendar_month(year=2100, month=2, alt="x")["days"] == 28  # not a leap year


def _good(**over):
    spec = kinds.calendar_month(
        year=2026, month=10, alt="a month", marks=[{"day": 9, "style": "star", "label": "Party"}]
    )
    spec.update(over)
    return spec


@pytest.mark.parametrize(
    "bad",
    [
        {"days": 30},  # October has 31
        {"first_column": 0},  # the 1st of October 2026 is a Thursday
        {"week_start": "tue"},
        {"month": 13},
        {"year": "2026"},
        {"marks": [{"day": 32, "style": "ring"}]},
        {"marks": [{"day": 0, "style": "ring"}]},
        {"marks": [{"day": 5, "style": "colour"}]},
        {"marks": [{"day": 5, "style": "ring", "label": "A very long label"}]},
        {"marks": [{"day": 5, "style": "ring"}, {"day": 5, "style": "star"}]},
        {"marks": "none"},
        {"show_year": "yes"},
        {"alt": ""},
    ],
)
def test_bad_calendar_specs_are_refused(bad):
    assert figures.validate(_good(**bad)), bad


def test_calendar_text_leaves_out_the_date_numerals_but_keeps_labels():
    fig = kinds.calendar_month(
        year=2026, month=10, alt="x", marks=[{"day": 9, "style": "star", "label": "Party"}], show_year=True
    )
    shown = figures.labels(fig)
    assert "October" in shown and "Party" in shown and "Thu" in shown and "2026" in shown
    assert "9" not in shown


def test_calendar_marks_are_not_told_apart_by_colour_alone():
    fig = kinds.calendar_month(
        year=2026,
        month=10,
        alt="x",
        marks=[{"day": 3, "style": "ring"}, {"day": 4, "style": "shade"}, {"day": 5, "style": "star"}],
    )
    text = svg.figure_svg(fig)
    assert text.count("<circle") == 1 and "<polygon" in text  # a ring and a star
    assert 'fill="#b0b0b0"' in text  # a shaded square
    assert all(h[0:2] == h[2:4] == h[4:6] for h in re.findall(r"#([0-9a-fA-F]{6})", text))  # greys only


def test_calendar_labels_and_text_are_escaped():
    fig = kinds.calendar_month(year=2026, month=10, alt="<b>", marks=[{"day": 3, "style": "ring", "label": "<&>"}])
    root = ET.fromstring(svg.figure_svg(fig))
    assert "<&>" in [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]


# ------------------------------------------------------------------ the day strip kind


def test_day_strip_validates_cards():
    ok = kinds.day_strip(cards=["wake_up", "breakfast", "lunch"], alt="a day", numbered=True)
    assert figures.validate(ok) == []
    assert [c["label"] for c in ok["cards"]] == ["Wake up", "Breakfast", "Lunch"]


@pytest.mark.parametrize(
    "bad",
    [
        {"cards": []},
        {"cards": [{"icon": "rocket", "label": "Rocket"}]},
        {"cards": [{"icon": "bed", "label": ""}]},
        {"cards": [{"icon": "bed", "label": "Bed time now"}]},
        {"cards": [{"icon": "bed", "label": "Bed"}, {"icon": "bed", "label": "Bed"}]},
        {"cards": [{"icon": i, "label": "x"} for i in ("bed", "bath", "dinner", "play", "lunch", "school", "wake_up")]},
        {"cards": [{"icon": "bed", "label": "Bed"}, {"icon": "moon", "label": "Moon"}]},  # arguable order
        {"cards": [{"icon": "school", "label": "School"}, {"icon": "lunch", "label": "Lunch"}]},
        {"numbered": 1},
    ],
)
def test_bad_day_strip_specs_are_refused(bad):
    spec = kinds.day_strip(cards=["wake_up", "lunch"], alt="a day")
    spec.update(bad)
    assert figures.validate(spec), bad


def test_day_strip_helpers_read_the_order_and_parts():
    fig = kinds.day_strip(cards=["dinner", "wake_up", "play", "breakfast"], alt="a day", numbered=True)
    assert kinds.day_order(fig) == [2, 4, 3, 1]
    assert kinds.slots_in_part(fig, "night") == [1] and kinds.slots_in_part(fig, "morning") == [2, 4]
    assert kinds.slot_of(fig, "Play") == 3
    assert [kinds.ICON_ORDER.index(i) for i in kinds.ICON_ORDER] == list(range(10))
    assert set(kinds.ICON_PART.values()) == set(kinds.DAY_PARTS)


def test_every_icon_renders_with_its_label_and_numbers():
    for icon in kinds.ICON_ORDER:
        fig = kinds.day_strip(cards=[icon], alt="one card", numbered=True)
        root = ET.fromstring(svg.figure_svg(fig))
        texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
        assert texts == ["1", kinds.ICON_LABELS[icon]]


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_skill_wiring(skill_id):
    topic, answer, codes = SKILLS[skill_id]
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.domain == "primary-geometry-time"
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False and skill.read_aloud is True and skill.learner_band == "early"
    assert skill.answer == answer
    assert [(s.topic, s.difficulty) for s in skill.sources] == [(topic, t) for t in TIERS]
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
    assert set(codes) <= set(skill.tags) and "exam" not in {s.difficulty for s in skill.sources}
    assert len(skill.misconceptions) >= 4 and all(m.description for m in skill.misconceptions)
    steps = [skill.difficulty_progression[t].steps for t in TIERS]
    assert steps == sorted(steps)


# ------------------------------------------------------------------ the questions


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_valid_does_not_leak_and_the_reader_agrees(skill_id, tier):
    reader = figures.get_reader(SKILLS[skill_id][0])
    assert reader is not None
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert reader(q) == q.answer_text, (tier, seed, q.text)
        svg.figure_svg(q.figure)


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_the_key_and_rejects_every_distractor(skill_id, tier):
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert _is_right(skill_id, q, q.answer_text), (tier, seed)
        assert len(q.distractors) >= 2 and q.answer_text not in q.distractors
        for d in q.distractors:
            assert not _is_right(skill_id, q, d), (tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("tier", TIERS)
def test_every_prompt_is_short_and_varied(skill_id, tier):
    texts = {_q(skill_id, tier, seed).text for seed in SEEDS}
    assert max(len(t.split()) for t in texts) <= READ_ALOUD_MAX_WORDS
    assert len(texts) >= 6


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(skill_id, family):
    skill_contract.FAMILIES[family](skill_id)


@pytest.mark.parametrize("topic", sorted({v[0] for v in SKILLS.values()}))
def test_unknown_tier_raises(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, tier="bogus")


# ------------------------------------------------------------------ tier shape


def test_count_days_tiers_ask_different_things():
    for seed in SEEDS:
        easy, medium, hard = (_q("ey-calendar-count-days", t, seed) for t in TIERS)
        assert easy.figure["marks"] == [] and 28 <= int(easy.answer_text) <= 31
        assert [m["label"] for m in medium.figure["marks"]] == ["Start", "End"]
        assert int(medium.answer_text) in (2, 3, 4) and "weeks" in medium.text
        gap = {m["label"]: m["day"] for m in hard.figure["marks"]}
        assert int(hard.answer_text) == gap["End"] - gap["Start"] and "week" in hard.text and "day" in hard.text


def test_count_days_easy_includes_february_and_31_day_months():
    answers = {_q("ey-calendar-count-days", "easy", s).answer_text for s in SEEDS}
    assert {"30", "31"} <= answers and answers & {"28", "29"}


def test_weekday_hard_uses_the_real_calendar_for_next_month():
    seen = set()
    for seed in SEEDS:
        q = _q("ey-calendar-weekday-of-date", "hard", seed)
        fig = q.figure
        year, month = fig["year"], fig["month"]
        if "last" in q.text or "end on" in q.text:
            seen.add("last")
            want = kinds.WEEKDAY_NAMES[pycal.weekday(year, month, pycal.monthrange(year, month)[1])]
        else:
            seen.add("next")
            ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
            want = kinds.WEEKDAY_NAMES[pycal.weekday(ny, nm, 1)]
        assert q.answer_text == want
    assert seen == {"last", "next"}


def test_weekday_medium_and_hard_use_both_week_starts():
    for tier in ("medium", "hard"):
        starts = {_q("ey-calendar-weekday-of-date", tier, s).figure["week_start"] for s in SEEDS}
        assert starts == {"mon", "sun"}
    assert {_q("ey-calendar-weekday-of-date", "easy", s).figure["week_start"] for s in SEEDS} == {"mon"}


def test_between_tiers():
    for seed in SEEDS:
        easy = _q("ey-calendar-days-between", "easy", seed)
        assert 2 <= int(easy.answer_text) <= 6 and len(easy.figure["marks"]) == 2
        medium = _q("ey-calendar-days-between", "medium", seed)
        (mark,) = medium.figure["marks"]
        assert int(medium.answer_text) - mark["day"] in (7, 14, 21) and int(medium.answer_text) <= medium.figure["days"]
        hard = _q("ey-calendar-days-between", "hard", seed)
        assert int(hard.answer_text) >= 8 or "last day" in hard.text


def test_between_hard_has_both_shapes():
    shapes = {"last day" in _q("ey-calendar-days-between", "hard", s).text for s in SEEDS}
    assert shapes == {True, False}


def test_day_strip_tiers():
    for seed in SEEDS:
        easy = _q("ey-day-strip-sequence", "easy", seed)
        icons = [c["icon"] for c in easy.figure["cards"]]
        assert icons == sorted(icons, key=kinds.ICON_ORDER.index) and 3 <= len(icons) <= 4
        medium = _q("ey-day-strip-sequence", "medium", seed)
        assert (
            4 <= len(medium.figure["cards"]) <= 5
            and kinds.day_order(medium.figure) != list(range(1, 6))[: len(medium.figure["cards"])]
        )
        hard = _q("ey-day-strip-sequence", "hard", seed)
        n = len(hard.figure["cards"])
        assert 4 <= n <= 6 and kinds.day_order(hard.figure) != list(range(1, n + 1))
        assert all(c["icon"] in kinds.ICON_LABELS for c in hard.figure["cards"])


def test_day_strip_medium_has_exactly_one_card_in_the_part():
    parts = set()
    for seed in SEEDS:
        q = _q("ey-day-strip-sequence", "medium", seed)
        for part in kinds.DAY_PARTS:
            if {
                "morning": "in the morning",
                "lunchtime": "at lunchtime",
                "afternoon": "in the afternoon",
                "night": "at night",
            }[part] in q.text:
                parts.add(part)
                assert len(kinds.slots_in_part(q.figure, part)) == 1
    assert parts == set(kinds.DAY_PARTS)


def test_day_strip_hard_distractor_is_the_slot_for_the_place():
    hits = 0
    for seed in SEEDS:
        q = _q("ey-day-strip-sequence", "hard", seed)
        place = ["first", "second", "third", "fourth", "fifth"]
        word = next((w for w in place if w in q.text), None)
        if word and str(place.index(word) + 1) in q.distractors:
            hits += 1
    assert hits >= 20


def test_order_wraps_at_the_ends_of_the_week():
    wraps = set()
    for seed in SEEDS:
        q = _q("ey-calendar-days-months-order", "medium", seed)
        if ("after Saturday" in q.text and q.figure["week_start"] == "sun") or "after Sunday" in q.text:
            wraps.add("after")
            assert q.answer_text in ("Sunday", "Monday")
        if "before Sunday" in q.text or "before Monday" in q.text:
            wraps.add("before")
    assert wraps == {"after", "before"}


def test_order_hard_wraps_december_and_january():
    answers = {_q("ey-calendar-days-months-order", "hard", s).answer_text for s in range(300)}
    assert {"January", "December"} <= answers


def test_mcq_key_position_is_balanced():
    for skill_id in ("ey-calendar-weekday-of-date", "ey-calendar-days-months-order"):
        for tier in TIERS:
            counts: dict[int, int] = {}
            for seed in range(200):
                q = make(SKILLS[skill_id][0], seed=seed, difficulty=tier)
                pos = mcq_options(q).index(q.answer_text)
                counts[pos] = counts.get(pos, 0) + 1
            assert max(counts.values()) / 200 < 0.45, (skill_id, tier, counts)
