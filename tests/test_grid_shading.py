"""Grid shading skills: elevations, plans and nets, scale plans, tessellations, and an enlargement read off a grid.

Four deferred skills (``answer: grid_cells``, the learner shades cells on a canvas) and one source of the active
``f10-enlargement-invariants`` skill (a multi-part question with a number part and a points part). Every tier of every
source is generated through the skill's own sources, so what is tested is what a learner is served.

The answers are re-derived here by routes that do not share code with the generators: a cube view by projecting along
a viewing direction with a cross product, a reflection by moving cell centres, a scale plan by the area it must have,
an enlargement by ratios of distances from the centre.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction

import pytest

from mathgen import figures, make
from mathgen.core import Part
from mathgen.generators import grid_shading_build as gs
from mathgen.grading import grade
from mathgen.grading.grid import cell_text, is_cube_net, orientations
from mathgen.skills import load_skills

SEEDS = range(60)
FIGURE_SKILLS = {
    # skill id -> (topic, tiers)
    "senior-sketch-elevation-views-of-different-models": ("grid_elevation_views", ["easy", "medium", "hard"]),
    "f10-two-dimensional-object-representations": ("grid_plan_views", ["easy", "medium", "hard"]),
    "senior-construct-scale-drawings-by-hand-and-by-using": ("grid_scale_plan", ["easy", "medium", "hard"]),
    "f10-tessellation-transform-combinations": ("grid_tessellation_images", ["easy", "medium", "hard"]),
}
ENLARGEMENT = "f10-enlargement-invariants"
CASES = [(sid, tier) for sid, (_t, tiers) in FIGURE_SKILLS.items() for tier in tiers]


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


def _source(skill_id, tier=None, key=None):
    return next(
        s
        for s in _skills()[skill_id].sources
        if (key is not None and s.key == key) or (key is None and s.difficulty == tier and s.key is None)
    )


@functools.lru_cache(maxsize=None)
def _q(skill_id, tier, seed):
    src = _source(skill_id, tier)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@functools.lru_cache(maxsize=None)
def _enlargement(seed):
    src = _source(ENLARGEMENT, key="grid-image")
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _cells(q):
    return {tuple(c) for c in q.answer["cells"]}


def _canvas_size(q):
    canvas = q.figure["canvas"]
    if canvas == "grid":
        return q.figure["grid"]["width"], q.figure["grid"]["height"]
    return canvas["width"], canvas["height"]


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", FIGURE_SKILLS)
def test_the_shading_skills_are_deferred_behind_the_visual_gate_with_working_sources(skill_id):
    topic, tiers = FIGURE_SKILLS[skill_id]
    skill = _skills()[skill_id]
    assert skill.status == "deferred" and skill.deferred_reason == "visual"
    assert skill.capability_gates == ["visual_construction"]
    assert skill.answer == "grid_cells" and skill.needs_figure is True and skill.placement_eligible is False
    assert [s.difficulty for s in skill.sources] == tiers
    assert all(s.topic == topic for s in skill.sources)
    assert skill.misconceptions and skill.prerequisites
    assert all(p in _skills() for p in skill.prerequisites)


def test_the_enlargement_skill_stays_active_with_one_grid_source_that_has_its_own_key():
    skill = _skills()[ENLARGEMENT]
    assert skill.status == "active" and skill.answer == "number"
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    keys = [s.key for s in skill.sources if s.topic == "grid_enlargement_image"]
    assert keys == ["grid-image"]
    other = [s for s in skill.sources if s.topic != "grid_enlargement_image"]
    assert [s.difficulty for s in other] == ["easy", "medium", "hard", "exam"]  # the sources already there
    ids = [m.id for m in skill.misconceptions]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------------ every figure, every tier


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_figure_is_valid_does_not_leak_and_the_reader_agrees_with_the_key(skill_id, tier):
    topic = FIGURE_SKILLS[skill_id][0]
    reader = figures.get_reader(topic)
    assert reader is not None
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert q.figure["kind"] == "grid_drawing" and q.answer_kind == "grid_cells"
        assert figures.validate(q.figure) == [], (seed, figures.validate(q.figure))
        figures.assert_no_leak(q.figure, [q.answer_text])
        assert not any(ch.isdigit() for ch in q.figure["alt"]), q.figure["alt"]
        read = reader(q)
        if q.answer["match"] == "exact":
            assert read == q.answer_text, (skill_id, tier, seed, q.text)
        else:  # a placement left open: the reader draws the shape at the corner, which is still right
            assert grade("grid_cells", q, read).correct, (skill_id, tier, seed, q.text, read)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_the_key_grades_correct_and_every_distractor_wrong_under_the_questions_own_match(skill_id, tier):
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert grade("grid_cells", q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        key = _cells(q)
        w, h = q.answer["grid"]
        for d in q.distractors:
            result = grade("grid_cells", q, d)
            assert not result.correct, (skill_id, tier, seed, d, q.answer_text, q.answer["match"])
            assert result.given_normalised is not None  # it was read as a drawing, then judged wrong
            cells = {tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", d)}
            assert cells != key and cells, (seed, d)
            assert all(0 <= x < w and 0 <= y < h for x, y in cells), (seed, d)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_deterministic_and_varied(skill_id, tier):
    seen = set()
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        src = _source(skill_id, tier)
        again = make(q.topic, seed=seed, difficulty=q.difficulty, **(src.params or {}))
        assert q.to_json() == again.to_json()
        seen.add((q.text, q.answer_text, repr(q.figure)))
    assert len(seen) >= 40, (skill_id, tier, len(seen))


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_the_answer_sits_on_the_canvas_and_the_match_mode_is_the_one_the_tier_needs(skill_id, tier):
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        w, h = _canvas_size(q)
        assert q.answer["grid"] == [w, h]
        assert all(0 <= x < w and 0 <= y < h for x, y in _cells(q))
        assert q.text and "\\(" not in q.text and "plain:" in q.answer_latex


# ------------------------------------------------------------------ cube views: an independent projection


def _project(cubes, direction, extent):
    """The cells (column, layer) seen by a viewer looking along ``direction`` (a vector in the ground plane).

    The viewer's right hand is ``direction x up``, so the column of a cube is its distance along that right-hand
    vector, counted from the leftmost position the grid allows."""
    dx, dy = direction
    right = (dy, -dx)  # (dx, dy, 0) x (0, 0, 1)
    w, d = extent
    corners = [(0, 0), (w - 1, 0), (0, d - 1), (w - 1, d - 1)]
    lowest = min(x * right[0] + y * right[1] for x, y in corners)
    return {(x * right[0] + y * right[1] - lowest, z) for x, y, z in cubes}


_LOOKING = {"front": (0, 1), "back": (0, -1), "right": (-1, 0), "left": (1, 0)}


def _stack_questions(skill_id, tiers):
    for tier in tiers:
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            if q.figure["grid"]["type"] == "isometric" and len(q.figure["cubes"]) > 1:
                yield tier, seed, q


def test_elevations_agree_with_a_projection_along_the_viewing_direction():
    seen = set()
    for tier, seed, q in _stack_questions(
        "senior-sketch-elevation-views-of-different-models", ["easy", "medium", "hard"]
    ):
        grid, cubes = q.figure["grid"], q.figure["cubes"]
        side = gs._VIEW_WORDS.findall(q.text)[-1]
        seen.add((tier, side))
        want = _project(cubes, _LOOKING[side], (grid["width"], grid["height"]))
        assert _cells(q) == want, (tier, seed, side, q.text)
    assert seen == {("easy", "front"), ("medium", "right"), ("medium", "left"), ("hard", "back")}


@pytest.mark.parametrize(
    "skill_id", ["senior-sketch-elevation-views-of-different-models", "f10-two-dimensional-object-representations"]
)
def test_every_stack_is_readable_and_the_canvas_is_exactly_the_size_of_the_view(skill_id):
    """No column hides behind a taller one and nothing is ambiguous, and the view fills its canvas edge to edge, so
    the position of the drawing is pinned to the bottom-left cell."""
    for tier, seed, q in _stack_questions(skill_id, ["easy", "medium", "hard"]):
        cubes = [tuple(c) for c in q.figure["cubes"]]
        heights = {}
        for x, y, z in cubes:
            heights[(x, y)] = max(heights.get((x, y), 0), z + 1)
        assert all(len({(x, y, z) for x, y, z in cubes if (x, y) == col}) == v for col, v in heights.items())  # solid
        grid = q.figure["grid"]
        every = [(x, y) for x in range(grid["width"]) for y in range(grid["height"])]
        for a in every:  # no spot, built on or bare, behind a taller column that is in front of it or to its right
            for b in every:
                if a != b and b[0] >= a[0] and b[1] <= a[1]:
                    assert heights.get(b, 0) <= heights.get(a, 0), (tier, seed, a, b)
        cells = _cells(q)
        w, h = _canvas_size(q)
        assert (max(x for x, _ in cells) + 1, max(y for _, y in cells) + 1) == (w, h), (tier, seed)
        assert (min(x for x, _ in cells), min(y for _, y in cells)) == (0, 0)
        assert "bottom-left" in q.text


def test_the_reversed_hard_question_gives_the_mirror_image_of_the_view_it_shows():
    count = 0
    for seed in SEEDS:
        q = _q("senior-sketch-elevation-views-of-different-models", "hard", seed)
        if q.figure["grid"]["type"] != "square":
            continue
        count += 1
        given = {tuple(c) for c in q.figure["cells"][0]["cells"]}
        w = q.figure["grid"]["width"]
        assert _cells(q) == {(w - 1 - x, z) for x, z in given}
        assert _cells(q) != given  # not a symmetric view, so copying it is a real, wrong answer
        named = set(re.findall(r"(?:the|from the) (front|back|left|right)(?: view|[.,])", q.text))
        assert named in ({"front", "back"}, {"left", "right"}), q.text  # the two opposite sides, no others
        assert cell_text(given) in q.distractors  # drawing the view it was given is offered as a wrong answer
    assert count >= 15


def test_the_elevation_tiers_differ_in_the_reasoning_they_need():
    sides = {t: set() for t in ("easy", "medium", "hard")}
    for tier in sides:
        for seed in SEEDS:
            q = _q("senior-sketch-elevation-views-of-different-models", tier, seed)
            sides[tier].add(
                gs._VIEW_WORDS.findall(q.text)[-1] + (":given" if q.figure["grid"]["type"] == "square" else "")
            )
    assert sides["easy"] == {"front"}
    assert sides["medium"] == {"left", "right"}
    assert sides["hard"] == {"back", "front:given", "back:given", "left:given", "right:given"}


# ------------------------------------------------------------------ plans, plans with heights and nets


def test_plan_views_are_the_columns_seen_from_above_with_the_front_at_the_bottom():
    for seed in SEEDS:
        q = _q("f10-two-dimensional-object-representations", "easy", seed)
        columns = {(x, y) for x, y, _z in q.figure["cubes"]}
        assert _cells(q) == columns
        w, d = _canvas_size(q)
        assert len(columns) < w * d  # at least one bare spot, so the plan is not just a rectangle
        assert "plan view" in q.text and "front" in q.text


def test_plan_with_heights_gives_the_view_that_the_numbers_on_the_plan_describe():
    for seed in SEEDS:
        q = _q("f10-two-dimensional-object-representations", "medium", seed)
        fig = q.figure
        heights = {(int(lb["at"][0]), int(lb["at"][1])): int(lb["text"]) for lb in fig["labels"]}
        # each shaded square carries its own number and every number is on a shaded square
        shaded = {tuple(c) for c in fig["cells"][0]["cells"]}
        assert shaded == set(heights) and all(1 <= v <= 4 for v in heights.values())
        side = gs._VIEW_WORDS.findall(q.text)[-1]
        cubes = [(x, y, z) for (x, y), v in heights.items() for z in range(v)]
        w, d = fig["grid"]["width"], fig["grid"]["height"]
        assert _cells(q) == _project(cubes, _LOOKING[side], (w, d)), (seed, side)
        top = max(heights.values())
        assert _canvas_size(q) == ((w if side == "front" else d), top)


def test_the_nets_are_exactly_the_eleven_shapes_that_fold_into_a_cube():
    shapes = gs._net_shapes()
    free = {frozenset(min(orientations(s), key=sorted)) for s in shapes}
    assert len(free) == 11
    assert all(len(s) == 6 and is_cube_net(s) for s in shapes)
    # on a 5 by 2 canvas only the net with two rows of three offset by two fits (turned or flipped)
    tight = {frozenset(min(orientations(n), key=sorted)) for n in gs._nets_on(5, 2)}
    assert len(tight) == 1


def test_net_questions_accept_any_net_that_fits_and_the_distractors_are_not_nets():
    sizes = set()
    for seed in SEEDS:
        q = _q("f10-two-dimensional-object-representations", "hard", seed)
        assert q.answer["match"] == "cube_net"
        w, h = _canvas_size(q)
        sizes.add((w, h))
        assert is_cube_net(_cells(q)) and figures.get_reader("grid_plan_views")(q)
        for other in gs._nets_on(w, h)[:: max(1, len(gs._nets_on(w, h)) // 7)]:
            assert grade("grid_cells", q, cell_text(other)).correct  # a different net is right too
        for d in q.distractors:
            cells = {tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", d)}
            assert not is_cube_net(cells)
        assert f"{w} cells wide and {h} cells" in q.text or f"({w} cells wide, {h} cells tall)" in q.text
    assert (6, 6) in sizes and len(sizes) >= 5


def test_the_plan_tiers_differ_in_the_reasoning_they_need():
    easy = _q("f10-two-dimensional-object-representations", "easy", 1)
    medium = _q("f10-two-dimensional-object-representations", "medium", 1)
    hard = _q("f10-two-dimensional-object-representations", "hard", 1)
    assert easy.figure["grid"]["type"] == "isometric" and easy.answer["match"] == "exact"
    assert medium.figure["grid"]["type"] == "square" and medium.figure["labels"] and medium.answer["match"] == "exact"
    assert hard.answer["match"] == "cube_net"


# ------------------------------------------------------------------ scale plans: by the area they must have


def _scale_facts(q):
    text = q.text
    per_cm = gs._scale_cm(q.figure["scale"]["text"])
    pairs = [
        (Fraction(a), Fraction(b))
        for a, b in re.findall(r"([\d.]+) m across the page and ([\d.]+) m up the page", text)
    ]
    return per_cm, pairs


def test_a_scale_plan_has_the_area_the_real_dimensions_and_the_scale_give():
    for tier in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _q("senior-construct-scale-drawings-by-hand-and-by-using", tier, seed)
            per_cm, pairs = _scale_facts(q)
            real = pairs[0][0] * pairs[0][1] - (pairs[1][0] * pairs[1][1] if len(pairs) == 2 else 0)  # square metres
            drawn_cm2 = len(_cells(q))  # a cell is 1 cm by 1 cm
            assert real * 10_000 == drawn_cm2 * per_cm**2, (tier, seed, q.text)
            assert re.search(r"1 cm (on|of) the drawing", q.text) and q.figure["scale"]["text"] not in q.text


def test_the_wording_says_which_freedom_the_learner_has_and_the_match_mode_agrees():
    modes = {}
    for tier in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _q("senior-construct-scale-drawings-by-hand-and-by-using", tier, seed)
            mode = q.answer["match"]
            modes.setdefault(tier, set()).add(mode)
            if mode == "congruent":
                assert "turn or flip" in q.text
            else:
                assert mode == "translate" and "do not" not in q.text
                assert "keep it the same way round" in q.text and "turn or flip" not in q.text
            lshape = "L-shaped" in q.text
            assert lshape == (len(_scale_facts(q)[1]) == 2)
    assert modes["easy"] == {"translate"} and modes["medium"] == {"translate"}
    assert modes["hard"] == {"translate", "congruent"}


def test_scale_tiers_differ_in_the_reasoning_they_need():
    easy_scales = {
        _q("senior-construct-scale-drawings-by-hand-and-by-using", "easy", s).figure["scale"]["text"] for s in SEEDS
    }
    hard_scales = {
        _q("senior-construct-scale-drawings-by-hand-and-by-using", "hard", s).figure["scale"]["text"] for s in SEEDS
    }
    assert all(t.startswith("1 cm : ") and t.endswith(" m") for t in easy_scales)  # whole metres per centimetre
    assert any(" : " in t and "cm :" not in t for t in hard_scales)  # a ratio such as 1 : 200
    assert any(t.endswith(" cm") for t in hard_scales)  # a scale in centimetres, with real lengths in metres
    shapes = {
        tier: {"L-shaped" in _q("senior-construct-scale-drawings-by-hand-and-by-using", tier, s).text for s in SEEDS}
        for tier in ("easy", "medium")
    }
    assert shapes == {"easy": {False}, "medium": {True}}


def test_a_turned_scale_plan_is_right_when_the_orientation_is_open_and_wrong_when_it_is_not():
    checked = {"congruent": 0, "translate": 0}
    for seed in SEEDS:
        q = _q("senior-construct-scale-drawings-by-hand-and-by-using", "hard", seed)
        cells = _cells(q)
        turned = {(y, x) for x, y in cells}
        mode = q.answer["match"]
        if (
            turned == cells
            or max(x for x, _ in turned) >= q.answer["grid"][0]
            or max(y for _, y in turned) >= q.answer["grid"][1]
        ):
            continue
        ok = grade("grid_cells", q, cell_text(turned)).correct
        assert ok == (mode == "congruent")
        checked[mode] += 1
    assert checked["congruent"] >= 5


# ------------------------------------------------------------------ tessellations: by moving cell centres


def _centre_reflect(cells, line):
    """Reflect cells by moving their centres, then name the cell each lands in."""
    axis, m = gs._line_of(line)
    out = set()
    for x, y in cells:
        cx, cy = Fraction(2 * x + 1, 2), Fraction(2 * y + 1, 2)
        if axis == "v":
            cx = 2 * m - cx
        else:
            cy = 2 * m - cy
        out.add((math.floor(cx), math.floor(cy)))
    return out


def test_easy_and_medium_images_follow_the_wording_step_by_step():
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = _q("f10-tessellation-transform-combinations", tier, seed)
            fig = q.figure
            cells = {tuple(c) for c in fig["cells"][0]["cells"]}
            lines = list(fig.get("lines", []))
            steps = re.findall(r"(reflect|translate)[^,.]*", q.text, re.IGNORECASE)
            assert 1 <= len(steps) <= 2 and (tier == "easy") == (len(steps) == 1), (tier, seed, q.text)
            for step in re.finditer(r"(reflect|translate)([^,.]*)", q.text, re.IGNORECASE):
                if step.group(1).lower() == "reflect":
                    which = re.search(r"mirror line (\d)", step.group(2))
                    cells = _centre_reflect(cells, lines[int(which.group(1)) - 1] if which else lines[0])
                else:
                    dx = dy = 0
                    for n, word in re.findall(r"(\d+) squares? (to the left|to the right|up|down)", step.group(2)):
                        n = int(n)
                        dx += {"to the right": n, "to the left": -n}.get(word, 0)
                        dy += {"up": n, "down": -n}.get(word, 0)
                    cells = {(x + dx, y + dy) for x, y in cells}
            assert cells == _cells(q), (tier, seed, q.text)
            given = {tuple(c) for c in fig["cells"][0]["cells"]}
            assert not given & _cells(q) and len(given) == len(_cells(q))
            assert "already drawn" in q.text or "are given" in q.text or "already" in q.text


def test_every_tessellation_question_says_the_drawn_squares_are_given():
    for tier in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _q("f10-tessellation-transform-combinations", tier, seed)
            assert q.figure["canvas"] == "grid"
            assert re.search(r"already drawn|are given|given", q.text), q.text
            given = [tuple(c) for g in q.figure["cells"] for c in g["cells"]]
            assert not set(given) & _cells(q)
            assert all(g["style"] == "hatched" for g in q.figure["cells"])


def test_hard_tessellations_continue_the_pattern_by_the_same_rule():
    kinds = set()
    for seed in SEEDS:
        q = _q("f10-tessellation-transform-combinations", "hard", seed)
        t0, t1 = ({tuple(c) for c in g["cells"]} for g in q.figure["cells"])
        lines = q.figure.get("lines", [])
        asked = _cells(q)
        assert "next two tiles" in q.text and len(asked) == 2 * len(t0)
        if lines:
            kinds.add("reflections")
            (a, m1), (b, m2) = (gs._line_of(ln) for ln in lines)
            assert a == b and abs(m2 - m1) >= 3
            t2 = _centre_reflect(t1, lines[1])
            third = (
                {"points": [[2 * m2 - m1, 0], [2 * m2 - m1, 1]]}
                if a == "v"
                else {"points": [[0, 2 * m2 - m1], [1, 2 * m2 - m1]]}
            )
            t3 = _centre_reflect(t2, third)
            assert asked == t2 | t3
            assert t1 == _centre_reflect(t0, lines[0])  # the second tile really is the first, reflected
            unflipped = {(x + (min(a for a, _ in t1) - min(a for a, _ in t0)), y) for x, y in t0}
            if a == "h":
                unflipped = {(x, y + (min(b for _, b in t1) - min(b for _, b in t0))) for x, y in t0}
            assert t1 != unflipped  # the second tile is the first turned over, not just moved along
        else:
            kinds.add("translations")
            dx = min(x for x, _ in t1) - min(x for x, _ in t0)
            dy = min(y for _, y in t1) - min(y for _, y in t0)
            assert t1 == {(x + dx, y + dy) for x, y in t0}
            assert asked == {(x + k * dx, y + k * dy) for x, y in t0 for k in (2, 3)}
    assert kinds == {"reflections", "translations"}


def test_the_tessellation_tiers_differ_in_the_reasoning_they_need():
    for seed in SEEDS:
        easy = _q("f10-tessellation-transform-combinations", "easy", seed)
        medium = _q("f10-tessellation-transform-combinations", "medium", seed)
        hard = _q("f10-tessellation-transform-combinations", "hard", seed)
        assert ", then " not in easy.text and ", then " in medium.text
        assert "next two tiles" in hard.text and len(hard.figure["cells"]) == 2
        assert len(easy.figure["cells"]) == len(medium.figure["cells"]) == 1


def test_a_slid_tile_is_offered_as_a_wrong_answer_for_a_reflection():
    slid = 0
    for seed in SEEDS:
        q = _q("f10-tessellation-transform-combinations", "easy", seed)
        if "Reflect" not in q.text:
            continue
        tile = {tuple(c) for c in q.figure["cells"][0]["cells"]}
        image = _cells(q)
        # the same shape, not turned over, at the place the reflection lands
        dx, dy = min(x for x, _ in image) - min(x for x, _ in tile), min(y for _, y in image) - min(y for _, y in tile)
        unturned = {(x + dx, y + dy) for x, y in tile}
        assert unturned != image
        slid += cell_text(unturned) in q.distractors
    assert slid >= 5


# ------------------------------------------------------------------ the enlargement, by ratios from the centre


def _enlargement_facts(q):
    fig = q.figure
    verts = [tuple(v) for v in fig["polygons"][0]["vertices"]]
    at = {lb["text"]: tuple(lb["at"]) for lb in fig["labels"]}
    k = Fraction(re.search(r"scale factor of (\d+(?:/\d+)?)", q.text).group(1))
    return verts, at["O"], k


def test_the_enlargement_is_a_multi_part_question_with_a_number_part_then_a_points_part():
    for seed in SEEDS:
        q = _enlargement(seed)
        assert [p.label for p in q.parts] == ["a", "b"] and all(isinstance(p, Part) for p in q.parts)
        assert [p.answer_kind for p in q.parts] == ["number", "points"]
        assert q.figure["kind"] == "grid_drawing" and figures.validate(q.figure) == []
        assert q.stem and q.text.startswith(q.stem) and "(a)" in q.text and "(b)" in q.text
        assert q.answer_text == f"(a) {q.parts[0].answer_text}; (b) {q.parts[1].answer_text}"
        assert q.parts[1].answer["ordered"] is True


def test_the_image_vertices_are_the_original_ones_scaled_from_the_centre():
    factors = set()
    for seed in SEEDS:
        q = _enlargement(seed)
        verts, centre, k = _enlargement_facts(q)
        factors.add(k)
        image = [tuple(p) for p in q.parts[1].answer["points"]]
        assert len(image) == len(verts) and k in (2, 3, Fraction(1, 2))
        for (vx, vy), (ix, iy) in zip(verts, image):
            assert (ix - centre[0]) * 1 == k * (vx - centre[0]) and (iy - centre[1]) == k * (vy - centre[1])
            assert (vx, vy) != centre
        # the centre is a point of the grid that is not the origin, so enlarging from (0, 0) is wrong
        assert centre != (0, 0)
        assert all(x >= 0 and y >= 0 for x, y in image)
        grid = q.figure["grid"]
        assert all(x <= grid["width"] and y <= grid["height"] for x, y in image)
    assert factors == {Fraction(2), Fraction(3), Fraction(1, 2)}


def test_part_a_is_the_perimeter_or_a_side_of_the_image_worked_from_the_figure():
    kinds = set()
    for seed in SEEDS:
        q = _enlargement(seed)
        verts, _centre, k = _enlargement_facts(q)
        lengths = [math.dist(a, b) for a, b in zip(verts, verts[1:] + verts[:1])]
        prompt = q.parts[0].prompt
        if "perimeter" in prompt or "all the way round" in prompt:
            kinds.add("perimeter")
            want = sum(lengths) * k
        else:
            kinds.add("side")
            name = re.search(r"([A-D])'([A-D])'|matches ([A-D])([A-D])", prompt)
            first = name.group(1) or name.group(3)
            want = lengths["ABCD".index(first)] * k
        assert q.parts[0].answer == want == int(want)
    assert kinds == {"perimeter", "side"}


def test_the_enlargement_reader_the_key_and_every_distractor_grade_as_they_should():
    reader = figures.get_reader("grid_enlargement_image")
    for seed in SEEDS:
        q = _enlargement(seed)
        figures.assert_no_leak(q.figure, [q.answer_text, *(p.answer_text for p in q.parts)])
        assert reader(q) == q.answer_text
        assert grade("number", q, q.answer_text).correct
        for part in q.parts:
            assert len(part.distractors) == 3 and len(set(part.distractors)) == 3
            for d in part.distractors:
                assert not grade(part.answer_kind, part, d).correct, (seed, part.label, d, part.answer_text)
        # a part answered wrongly loses only that part
        wrong_b = {"a": q.parts[0].answer_text, "b": q.parts[1].distractors[0]}
        marked = grade("number", q, wrong_b)
        assert marked.parts["a"].correct and not marked.parts["b"].correct and not marked.correct


def test_the_points_distractors_are_real_slips():
    seen = set()
    for seed in SEEDS:
        q = _enlargement(seed)
        verts, centre, k = _enlargement_facts(q)
        b = q.parts[1]
        from_origin = gs.points_answer_text([(k * x, k * y) for x, y in verts])
        no_centre = gs.points_answer_text([(k * (x - centre[0]), k * (y - centre[1])) for x, y in verts])
        extra_factor = gs.points_answer_text([(x + k * (x - centre[0]), y + k * (y - centre[1])) for x, y in verts])
        for name, text in (("origin", from_origin), ("no-centre", no_centre), ("extra", extra_factor)):
            if text in b.distractors:
                seen.add(name)
    assert seen == {"origin", "no-centre", "extra"}


def test_the_enlargement_prompts_vary_and_say_what_unit_to_answer_in():
    shapes = set()
    for seed in SEEDS:
        q = _enlargement(seed)
        assert "cm" in q.parts[0].prompt
        shapes.add(re.sub(r"\d+(/\d+)?", "#", q.stem))
    assert len(shapes) >= 3
