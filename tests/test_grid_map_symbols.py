"""Map, plan and convention skills drawn on a square grid (`mathgen/generators/grid_map_symbols_build.py`).

Three active multiple-choice skills behind the figure gate: ``senior-interpret-commonly-used-symbols-and-
abbreviations-in-scale`` (``grid_plan_symbols``), ``senior-use-symbols-and-conventions-for-the-representation-of``
(``grid_plan_conventions``) and ``f10-map-feature-and-landmark-representation`` (``grid_map_landmarks``). Every tier of
every source is generated through the skill's own sources, so what is tested is what a learner is served.
"""

import functools
import re

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.generators import grid_map_symbols_build as gen
from mathgen.grading import grade
from mathgen.skills import load_skills, validate

SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

SKILLS = {
    "senior-interpret-commonly-used-symbols-and-abbreviations-in-scale": "grid_plan_symbols",
    "senior-use-symbols-and-conventions-for-the-representation-of": "grid_plan_conventions",
    "f10-map-feature-and-landmark-representation": "grid_map_landmarks",
}
CASES = [(sid, tier) for sid in SKILLS for tier in TIERS]


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


@functools.lru_cache(maxsize=None)
def _q(skill_id, tier, seed):
    src = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ------------------------------------------------------------------ the skill files


def test_my_skills_validate_with_their_prerequisites():
    """The validator generates every source; run it on these skills and the one prerequisite outside them."""
    domains, skills = load_skills()
    chosen = [s for s in skills if s.id in SKILLS or s.id == "f10-position-location-language"]
    assert len(chosen) == len(SKILLS) + 1
    assert validate(domains, chosen).errors == []


@pytest.mark.parametrize("skill_id", SKILLS)
def test_skill_is_active_behind_the_figure_gate(skill_id):
    skill = _skills()[skill_id]
    assert skill.answer == "mcq"
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert [s.difficulty for s in skill.sources] == TIERS
    assert {s.topic for s in skill.sources} == {SKILLS[skill_id]}
    assert len(skill.misconceptions) >= 4
    framework = "acara-f10-v9" if skill_id.startswith("f10-") else "acara-senior-v8-4"
    assert framework in skill.tags and any(re.fullmatch(r"ac(m|9m)\w+", t) for t in skill.tags)


# ------------------------------------------------------------------ every figure, every tier


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_figure_is_square_valid_does_not_leak_and_the_reader_agrees_with_the_key(skill_id, tier):
    reader = figures.get_reader(SKILLS[skill_id])
    assert reader is not None
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        fig = q.figure
        assert fig is not None and fig["kind"] == "grid_drawing" and fig["grid"]["type"] == "square", (seed, q.text)
        assert figures.validate(fig) == [], (seed, figures.validate(fig))
        figures.assert_no_leak(fig, [q.answer_text])
        assert gen._leaks(fig, q.answer_text) is False, (seed, q.answer_text)
        assert not any(ch.isdigit() for ch in fig["alt"]), fig["alt"]
        assert reader(q) == q.answer_text, (skill_id, tier, seed, q.text, q.answer_text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_the_key_grades_correct_and_every_distractor_wrong(skill_id, tier):
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert grade("mcq", q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3, (seed, q.distractors)
        assert q.answer_text not in q.distractors
        for d in q.distractors:
            assert not grade("mcq", q, d).correct, (skill_id, tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_deterministic_and_varied(skill_id, tier):
    seen = set()
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        again = make(q.topic, seed=seed, difficulty=tier)
        assert q.to_json() == again.to_json()
        seen.add((q.text, q.answer_text, repr(q.figure)))
    assert len(seen) >= 55, (skill_id, tier, len(seen))


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_the_prompt_is_plain_text_and_the_unknown_tier_is_refused(skill_id, tier):
    q = _q(skill_id, tier, 0)
    assert "\\(" not in q.text and "\\(" not in q.latex
    with pytest.raises(ValueError):
        make(SKILLS[skill_id], seed=0, difficulty=None, tier="impossible")


def _answers(skill_id, tier):
    return [_q(skill_id, tier, seed) for seed in SEEDS]


# ------------------------------------------------------------------ plan symbols and abbreviations

PLAN = "senior-interpret-commonly-used-symbols-and-abbreviations-in-scale"


def test_plan_tiers_ask_for_different_reasoning():
    easy, medium, hard = (_answers(PLAN, t) for t in TIERS)
    # easy: a symbol decides the room; no fixture labels are on the plan
    for q in easy:
        assert q.text in (*gen._TOILETS_ASKS, *gen._WINDOW_ASKS)
        assert not any(lb["text"] in gen.FIXTURES for lb in q.figure["labels"])
        assert q.answer_text in {gen.room_full(lb["text"]) for lb in q.figure["labels"] if gen._is_room(lb["text"])}
    # medium: a full name is decoded to a fixture abbreviation, found on the plan, then a room abbreviation decoded
    for q in medium:
        assert q.text.startswith("In which room is the ")
        assert {lb["text"] for lb in q.figure["labels"]} >= set(gen.FIXTURES)
    # hard: the answer is a count that combines symbols and abbreviations
    for q in hard:
        assert re.fullmatch(r"\d rooms?", q.answer_text)
        assert q.text in (*gen._WET_ASKS, *gen._PLUMBER_ASKS)


def test_plan_rooms_are_named_by_abbreviation_and_answered_in_full():
    for q in _answers(PLAN, "easy") + _answers(PLAN, "medium"):
        shown = {lb["text"] for lb in q.figure["labels"]}
        assert q.answer_text not in shown
        assert all(d not in shown for d in q.distractors)
        assert {gen.room_full(a) for a in shown if gen._is_room(a)} >= set(q.distractors) | {q.answer_text}


def test_plan_easy_has_exactly_one_toilets_room_or_one_dark_room():
    for q in _answers(PLAN, "easy"):
        rooms = gen._plan_rooms_of(q.figure)
        toilets = gen._rooms_with(q.figure, rooms, "toilets")
        windows = gen._rooms_with(q.figure, rooms, "window")
        if "toilets" in q.text:
            assert len(toilets) == 1
        else:
            assert len(rooms) - len(windows) == 1


def test_plan_legend_is_dropped_only_when_it_would_give_the_answer_away():
    seen = set()
    for q in _answers(PLAN, "easy") + _answers(PLAN, "medium"):
        legend = q.figure.get("legend", False)
        seen.add((q.answer_text == "Toilet", legend))
        if q.answer_text == "Toilet":
            assert legend is False
        else:
            assert legend is True
    # both cases occur: the Toilet answer without a legend, and some other answer with one
    assert seen == {(True, False), (False, True)}


def test_plan_hard_distractors_are_real_miscounts():
    for q in _answers(PLAN, "hard"):
        rooms = gen._plan_rooms_of(q.figure)
        wet = [r["abbr"] for r in rooms if r["abbr"] in gen.WET]
        windowed = gen._rooms_with(q.figure, rooms, "window")
        if q.text in gen._WET_ASKS:
            n = len([a for a in wet if a in windowed])
            assert gen._rooms(n) == q.answer_text
            assert gen._rooms(len(wet)) in q.distractors or gen._rooms(len(wet) - n) in q.distractors


# ------------------------------------------------------------------ symbols and conventions

CONV = "senior-use-symbols-and-conventions-for-the-representation-of"


def test_conventions_tiers_ask_for_different_reasoning():
    easy, medium, hard = (_answers(CONV, t) for t in TIERS)
    for q in easy:  # the dashed outline decides it
        assert sum(p["style"] == "dashed" for p in q.figure["polygons"]) == 1
        assert q.answer_text in gen.FEATURES.values()
    for q in medium:  # a compass direction, from the north arrow
        assert q.figure.get("north") is True and q.answer_text in gen.COMPASS
        assert not any(p["style"] == "dashed" for p in q.figure["polygons"])
    for q in hard:  # dimension labels, one dashed and one solid shape, and a ratio scale
        assert re.fullmatch(r"\d+ m\u00b2", q.answer_text) and len(q.figure["polygons"]) == 2
        assert {p["style"] for p in q.figure["polygons"]} == {"solid", "dashed"}
        assert re.fullmatch(r"1 : \d+", q.figure["scale"]["text"])
        assert all(lb["text"].endswith(" m") for lb in q.figure["labels"])


def test_conventions_dimension_labels_match_the_drawn_shapes_at_the_stated_scale():
    for q in _answers(CONV, "hard"):
        metres = int(q.figure["scale"]["text"].split(":")[1]) // 100
        for poly in q.figure["polygons"]:
            x0, y0, x1, y1 = gen._box(poly["vertices"])
            assert gen._dim(q.figure, [(x0 + x1) / 2, y1 + 0.5]) == (x1 - x0) * metres
            assert gen._dim(q.figure, [x1 + 1, (y0 + y1) / 2]) == (y1 - y0) * metres


def test_conventions_hard_never_labels_which_shape_is_which():
    """The shapes carry dimensions only: which is proposed is known from the dashed outline, not from a label."""
    for q in _answers(CONV, "hard"):
        assert not any(lb["text"] in gen.FEATURES for lb in q.figure["labels"])


def test_conventions_medium_directions_lie_on_a_compass_line_and_distractors_are_misreadings():
    for q in _answers(CONV, "medium"):
        d = q.answer_text
        assert d in gen.COMPASS
        reverse = gen._BY_VECTOR[tuple(-v for v in gen.COMPASS[d])]
        assert reverse in q.distractors or d in ("north", "south", "east", "west")


# ------------------------------------------------------------------ map landmarks

MAP = "f10-map-feature-and-landmark-representation"


def test_map_tiers_ask_for_different_reasoning():
    easy, medium, hard = (_answers(MAP, t) for t in TIERS)
    for q in easy:  # a square reference names a landmark; no legend, so the symbol has to be recognised
        assert re.fullmatch(r"Which landmark is in square [A-G][1-6]\?", q.text)
        assert "legend" not in q.figure
    kinds = set()
    for q in medium:
        if q.text.startswith("What is the grid reference"):
            kinds.add("reference")
            assert re.fullmatch(r"[A-G][1-6]", q.answer_text) and q.figure["legend"] is True
        else:
            kinds.add("direction")
            assert q.text.startswith("Look ") and "legend" not in q.figure
    assert kinds == {"reference", "direction"}
    for q in hard:  # two moves from a named landmark
        assert re.fullmatch(r"Start at the .+\. Go \d squares? \w+, then \d squares? \w+\. Which square.*", q.text)
        assert re.fullmatch(r"[A-G][1-6]", q.answer_text) and q.figure["legend"] is True


def test_map_has_column_letters_row_numbers_a_north_arrow_and_one_symbol_per_kind():
    for tier in TIERS:
        for q in _answers(MAP, tier):
            fig = q.figure
            assert fig["north"] is True
            assert fig["column_labels"] == list(gen.LETTERS[: fig["grid"]["width"]])
            assert fig["row_labels"] == [str(i) for i in range(1, fig["grid"]["height"] + 1)]
            symbols = [s["symbol"] for s in fig["symbols"]]
            assert len(symbols) == len(set(symbols))
            cells = [(int(s["at"][0]), int(s["at"][1])) for s in fig["symbols"]]
            assert len(cells) == len(set(cells))


def test_map_easy_distractors_are_the_landmarks_in_the_misread_squares():
    for q in _answers(MAP, "easy"):
        fig = q.figure
        c, r = gen._cell_of(fig, q.text.split("square ")[1].rstrip("?"))
        height = fig["grid"]["height"]
        by_name = {gen.grid.SYMBOLS[s].capitalize(): cell for cell, s in gen._symbol_cells(fig).items()}
        cells = {d: by_name[d] for d in q.distractors}
        misread = {(r, c), (c, height - 1 - r), (c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)}
        assert set(cells.values()) <= misread
        assert cells and (c, r) not in cells.values()


def test_map_reference_distractors_are_named_misreadings():
    for q in _answers(MAP, "medium"):
        if not q.text.startswith("What is the grid reference"):
            continue
        fig = q.figure
        c, r = gen._cell_of(fig, q.answer_text)
        height = fig["grid"]["height"]
        misreadings = {
            gen._ref(c, height - 1 - r),  # rows counted from the top
            gen._ref(r, c) if r < fig["grid"]["width"] else None,  # column and row swapped
            gen._ref(c + 1, r) if c + 1 < fig["grid"]["width"] else None,
            gen._ref(c - 1, r) if c else None,
            gen._ref(c, r + 1) if r + 1 < height else None,
        }
        assert set(q.distractors) <= misreadings


def test_map_direction_has_one_landmark_each_way_and_distractors_are_the_other_three():
    for q in _answers(MAP, "medium"):
        if not q.text.startswith("Look "):
            continue
        fig = q.figure
        start = gen._cell_of(fig, re.search(r"square (\w+)\.", q.text).group(1))
        cells = gen._symbol_cells(fig)
        rays = {"north": 0, "south": 0, "east": 0, "west": 0}
        for x, y in cells:
            if (x, y) == start:
                continue
            if x == start[0]:
                rays["north" if y > start[1] else "south"] += 1
            elif y == start[1]:
                rays["east" if x > start[0] else "west"] += 1
        assert set(rays.values()) == {1}, (q.text, rays)
        on_lines = {
            gen.grid.SYMBOLS[s].capitalize()
            for (x, y), s in cells.items()
            if (x, y) != start and (x == start[0] or y == start[1])
        }
        assert set(q.distractors) | {q.answer_text} == on_lines


def test_map_hard_distractors_are_real_wrong_routes():
    for q in _answers(MAP, "hard"):
        fig = q.figure
        m = re.match(r"Start at the (.+?)\. Go (\d+) squares? (\w+), then (\d+) squares? (\w+)\.", q.text)
        sx, sy = gen._symbol_named(fig, m.group(1))
        steps = {m.group(3): int(m.group(2)), m.group(5): int(m.group(4))}
        dx = steps.get("east", 0) - steps.get("west", 0)
        dy = steps.get("north", 0) - steps.get("south", 0)
        a, b = abs(dx), abs(dy)
        sgx, sgy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
        routes = {
            (sx - dx, sy + dy),
            (sx + dx, sy - dy),
            (sx + sgx * b, sy + sgy * a),
            (sx + sgx * (a - 1), sy + sgy * (b - 1)),
            (sx - dx, sy - dy),
        }
        wrong = {gen._cell_of(fig, d) for d in q.distractors}
        assert wrong <= routes
        assert gen._cell_of(fig, q.answer_text) == (sx + dx, sy + dy)


def test_map_answers_that_are_landmark_names_never_have_a_legend():
    for tier in ("easy", "medium"):
        for q in _answers(MAP, tier):
            if not re.fullmatch(r"[A-G][1-6]", q.answer_text):
                assert "legend" not in q.figure
            if q.figure.get("legend"):
                assert re.fullmatch(r"[A-G][1-6]", q.answer_text)


def test_map_hard_start_landmark_is_found_through_the_legend_and_is_on_the_map():
    for q in _answers(MAP, "hard"):
        name = re.match(r"Start at the (.+?)\.", q.text).group(1)
        assert name in {gen.grid.SYMBOLS[s["symbol"]] for s in q.figure["symbols"]}
