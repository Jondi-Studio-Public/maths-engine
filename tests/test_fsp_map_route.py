"""Tests `f10-map-grid-routes-bearings`: topic `fsp_map_route` (easy, medium, hard) and the exam chain `fsp_map_route_exam`.

The oracle reads the places off the drawn points (grid positions) and the scale and place names off the question text,
then works out the distance (counting squares) or the bearing (atan2 from north, clockwise).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pytest

import mathgen.generators.fsp_maps_build as maps
from mathgen import figures, make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILL_ID = "f10-map-grid-routes-bearings"
TIERS = ("easy", "medium", "hard")
SEEDS = range(80)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make("fsp_map_route", seed=seed, difficulty=tier)


def _exam(seed):
    return make("fsp_map_route_exam", seed=seed, difficulty="exam")


def _in_order(text, places):
    found = sorted((m.start(), name) for name in places for m in re.finditer(rf"\b{name}\b", text))
    return list(dict.fromkeys(name for _pos, name in found))


def _scale(text):
    return int(re.search(r"(\d+) km", text).group(1))


def test_skill_wiring():
    assert _SKILL.answer == "number" and _SKILL.status == "active"
    assert _SKILL.needs_figure is True and _SKILL.capability_gates == ["figure_stimulus"]
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        ("fsp_map_route", "easy"),
        ("fsp_map_route", "medium"),
        ("fsp_map_route", "hard"),
        ("fsp_map_route_exam", "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert {m.id for m in _SKILL.misconceptions} <= set(CATALOGUE)


def test_bearing_from_atan2_matches_the_compass_table():
    for (dx, dy), expected in maps._EIGHT.items():
        assert maps.bearing(3 * dx, 3 * dy) == expected
    assert maps.bearing(2, 1) == 63


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_a_numbered_grid_map_and_the_reader_agrees(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.figure["kind"] == "cartesian" and not figures.validate(q.figure)
        assert q.figure["y_label"] == "North"
        places = maps.map_places(q.figure)
        assert len(places) == 4 and len(set(places.values())) == 4
        assert figures.get_reader("fsp_map_route")(q) == q.answer_text, (tier, seed, q.text)
        assert not find_text_defects(q, include_working=True)


def test_easy_distance_is_squares_between_times_scale():
    seen = Counter()
    for seed in range(200):
        q = _q("easy", seed)
        places = maps.map_places(q.figure)
        s = _scale(q.text)
        ref = re.search(r"grid reference \((\d+), (\d+)\)", q.text)
        names = "|".join(map(re.escape, places))
        start_name = re.search(rf"(?:directly \w+ of|from) the ({names})\b", q.text).group(1)
        start = places[start_name]
        others = [n for n in _in_order(q.text, places) if n != start_name]
        end = (int(ref.group(1)), int(ref.group(2))) if ref else places[others[0]]
        assert end in places.values()
        assert start[0] == end[0] or start[1] == end[1]
        direction = re.search(r"directly (\w+)", q.text).group(1)
        dx, dy = maps._DIRS[direction]
        steps = abs(end[0] - start[0]) + abs(end[1] - start[1])
        assert (end[0] - start[0], end[1] - start[1]) == (dx * steps, dy * steps)
        assert int(q.answer_text) == steps * s
        seen["ref" if ref else "name"] += 1
    assert min(seen.values()) >= 60


def test_medium_route_is_the_sum_of_two_axis_legs():
    for seed in SEEDS:
        q = _q("medium", seed)
        places = maps.map_places(q.figure)
        a, b, c = (places[n] for n in _in_order(q.text, places)[:3])
        assert a[0] == b[0] or a[1] == b[1]
        assert b[0] == c[0] or b[1] == c[1]
        legs = abs(b[0] - a[0]) + abs(b[1] - a[1]) + abs(c[0] - b[0]) + abs(c[1] - b[1])
        assert int(q.answer_text) == legs * _scale_from_medium(q.text), (seed, q.text)


def _scale_from_medium(text):
    return int(re.search(r"(\d+) km", text).group(1))


def test_hard_bearing_of_the_start_from_the_finish():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        places = maps.map_places(q.figure)
        asked = re.search(r"bearing of the (\w+) from the (\w+)", q.text)
        a, c = places[asked.group(1)], places[asked.group(2)]
        angle = math.degrees(math.atan2(a[0] - c[0], a[1] - c[1])) % 360
        assert int(q.answer_text) == round(angle) % 360 and abs(angle - round(angle)) < 1e-9
        assert int(q.answer_text) % 45 == 0
        assert str((int(q.answer_text) + 180) % 360) in q.distractors or len(q.distractors) == 3
        # the route runs start, middle, finish and the question asks about the start and the finish
        route = _in_order(q.text, places)
        assert route[0] == asked.group(1) and route[2] == asked.group(2)
        seen[int(q.answer_text)] += 1
    assert len(seen) >= 6


def test_tiers_differ_in_shape():
    assert all("bearing" not in _q(t, s).text for t in ("easy", "medium") for s in SEEDS)
    assert all("bearing" in _q("hard", s).text for s in SEEDS)
    assert any("grid reference" in _q("easy", s).text for s in SEEDS)
    assert all(
        "then" in _q("medium", s).text or "and then" in _q("medium", s).text or "on to" in _q("medium", s).text
        for s in SEEDS
    )


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) == 3
        for text in q.distractors:
            result = grade(_SKILL, q, text)
            assert not result.correct, (tier, seed, text)
            assert result.misconception in CATALOGUE


@pytest.mark.parametrize("tier", TIERS)
def test_prompts_vary(tier):
    assert len({_q(tier, s).text for s in range(60)}) >= 20


def test_figure_does_not_leak_the_answer():
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text])


# -- the exam chain -------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 1, 1, 2]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through for p in q.parts] == [None, None, "part_a + part_b", None]


def test_exam_answers_recomputed_from_the_map():
    for seed in range(120):
        q = _exam(seed)
        places = maps.map_places(q.figure)
        a, b, c = (places[n] for n in _in_order(q.stem, places)[:3])
        s = _scale(q.stem)
        first, second = abs(b[0] - a[0]) * s, abs(c[1] - b[1]) * s
        assert b[1] == a[1] and c[0] == b[0] and c[0] > a[0] and c[1] > a[1]
        angle = round(math.degrees(math.atan2(a[0] - c[0], a[1] - c[1]))) % 360
        assert [int(p.answer_text) for p in q.parts] == [first, second, first + second, angle], (seed, q.text)


def test_exam_full_marks_and_distractors_rejected():
    for seed in range(60):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert 2 <= len(p.distractors) <= 3
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_exam_follow_through_from_a_wrong_first_leg():
    for seed in range(60):
        q = _exam(seed)
        wrong = int(q.parts[0].distractors[0])
        second = int(q.parts[1].answer_text)
        canon = {p.label: p.answer_text for p in q.parts}
        result = grade(_SKILL, q, {**canon, "a": str(wrong), "c": str(wrong + second)})
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["c"].follow_through


def test_exam_prompts_vary():
    assert len({_exam(s).text for s in range(60)}) >= 20
