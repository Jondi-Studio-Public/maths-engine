"""Grid solids, nets and composite shapes: the five skills of `mathgen/generators/grid_solids_build.py`.

Every tier of every source is generated through the skill's own sources, so what is tested is what a learner is
served: each figure validates and does not leak, the registered reader re-derives the key from the figure and the
sentence, the key grades correct and every distractor grades wrong, and the tiers ask for different reasoning.
"""

import functools
import re

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import mcq_options
from mathgen.figure_kinds import grid
from mathgen.generators import grid_solids_build as build
from mathgen.grading.grid import is_cube_net
from mathgen.grading.typed import grade
from mathgen.skills import load_skills, validate

SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

# skill id -> (answer kind, topic)
SKILLS = {
    "senior-interpret-diagrams-of-three-dimensional-objects": ("number", "grid_cube_stack"),
    "senior-recognise-the-properties-of-common-two-dimensional-geometric": ("number", "grid_solid_properties"),
    "senior-interpret-different-forms-of-two-dimensional-representations-of": ("mcq", "grid_net_views"),
    "f10-composite-shape-representation": ("number", "grid_composite_area"),
    "f10-nets-build-solids": ("mcq", "grid_nets_build_solids"),
}
CASES = [(sid, tier) for sid in SKILLS for tier in TIERS]
# the tiers that are figure-led; the rest are words only
TEXT_ONLY = {("senior-recognise-the-properties-of-common-two-dimensional-geometric", t) for t in ("medium", "hard")}


@functools.lru_cache(maxsize=1)
def _loaded():
    domains, skills = load_skills()
    return domains, skills, {s.id: s for s in skills}


def _source(skill_id, tier):
    topic = SKILLS[skill_id][1]
    return next(s for s in _loaded()[2][skill_id].sources if s.difficulty == tier and s.topic == topic)


@functools.lru_cache(maxsize=None)
def _gen(skill_id, tier, seed):
    src = _source(skill_id, tier)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _grade_given(skill_id, q, text):
    """Grade an option the way a client does: an mcq option is sent by its index."""
    if SKILLS[skill_id][0] == "mcq":
        return grade("mcq", q, str(mcq_options(q).index(text)))
    return grade("number", q, text)


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", SKILLS)
def test_skill_is_active_behind_the_figure_gate(skill_id):
    answer, topic = SKILLS[skill_id]
    skill = _loaded()[2][skill_id]
    assert skill.answer == answer
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    # f10-nets-build-solids also carries #75's fsp_nets tiers and exam chain (tests/test_fsp_nets.py)
    assert [s.difficulty for s in skill.sources if s.topic == topic] == TIERS
    assert all(s.topic == topic for s in skill.sources) or skill_id == "f10-nets-build-solids"
    assert skill.misconceptions and skill.prerequisites


@pytest.mark.parametrize("skill_id", SKILLS)
def test_the_library_validates_with_the_new_skills(skill_id):
    domains, skills, _by_id = _loaded()
    report = validate(domains, skills)
    assert [e for e in report.errors if skill_id in e] == []


def test_prerequisites_are_active_skills_and_the_chain_is_not_circular():
    by_id = _loaded()[2]
    for sid in SKILLS:
        for pre in by_id[sid].prerequisites:
            assert pre in by_id and by_id[pre].status == "active", (sid, pre)
    assert (
        "f10-nets-build-solids"
        in by_id["senior-interpret-different-forms-of-two-dimensional-representations-of"].prerequisites
    )


# ------------------------------------------------------------------ every question, every tier


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_figure_is_valid_does_not_leak_and_the_reader_agrees_with_the_key(skill_id, tier):
    _answer, topic = SKILLS[skill_id]
    reader = figures.get_reader(topic)
    assert reader is not None
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        if (skill_id, tier) in TEXT_ONLY:
            assert q.figure is None
        else:
            assert q.figure is not None and q.figure["kind"] == "grid_drawing", (seed, q.text)
            assert figures.validate(q.figure) == [], (seed, figures.validate(q.figure))
            figures.assert_no_leak(q.figure, [q.answer_text])
            assert q.figure["alt"] and not any(ch.isdigit() for ch in q.figure["alt"])
        assert reader(q) == q.answer_text, (skill_id, tier, seed, q.text, q.answer_text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_the_key_grades_correct_and_every_distractor_wrong(skill_id, tier):
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert _grade_given(skill_id, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3, (seed, q.distractors)
        assert q.answer_text not in q.distractors
        for d in q.distractors:
            assert not _grade_given(skill_id, q, d).correct, (skill_id, tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_deterministic_and_varied(skill_id, tier):
    seen = set()
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        again = make(q.topic, seed=seed, difficulty=q.difficulty)
        assert q.to_json() == again.to_json()
        assert "\\(" not in q.text and "\\(" not in q.latex
        seen.add((q.text, q.answer_text, repr(q.figure)))
    assert len(seen) >= 40, (skill_id, tier, len(seen))


@pytest.mark.parametrize("skill_id,tier", [c for c in CASES if SKILLS[c[0]][0] == "number"])
def test_number_distractors_clear_the_grader_by_construction(skill_id, tier):
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        answer = float(q.answer_text)
        for d in q.distractors:
            assert abs(float(d) - answer) >= 0.5, (seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id,tier", [c for c in CASES if c not in TEXT_ONLY])
def test_every_stack_stays_inside_its_grid(skill_id, tier):
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        fig = q.figure
        if "cubes" in fig:
            assert grid.tops_all_visible(fig["cubes"]) or SKILLS[skill_id][1] == "grid_solid_properties"


# ------------------------------------------------------------------ nets: every cube net and non-net is checked


def test_there_are_eleven_cube_nets_and_no_non_net_folds():
    nets, others = build.cube_nets(), build.non_nets()
    assert len(nets) == 11
    assert all(is_cube_net(s) for s in nets) and not any(is_cube_net(s) for s in others)
    assert len(others) >= 15


def _option_drawings(fig):
    return build._drawings_by_letter(fig)


@pytest.mark.parametrize("skill_id,tier", [("f10-nets-build-solids", "medium")])
def test_exactly_one_drawing_is_a_cube_net(skill_id, tier):
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        drawings = _option_drawings(q.figure)
        assert sorted(drawings) == list("ABCD")
        assert all(len(c) == 6 for c in drawings.values())
        nets = [k for k, c in drawings.items() if is_cube_net(c)]
        assert nets == [q.answer_text.removeprefix("Net ")]
        assert sorted(q.distractors) == sorted(f"Net {k}" for k in "ABCD" if k not in nets)


def test_the_not_a_net_question_has_three_cube_nets_and_one_that_is_not():
    for seed in SEEDS:
        q = _gen("senior-interpret-different-forms-of-two-dimensional-representations-of", "medium", seed)
        drawings = _option_drawings(q.figure)
        flags = {k: is_cube_net(c) for k, c in drawings.items()}
        assert sum(flags.values()) == 3
        assert [k for k, v in flags.items() if not v] == [q.answer_text.removeprefix("Net ")]
        assert "NOT a net" in q.text


def test_drawings_in_one_question_are_different_shapes_and_do_not_overlap():
    for skill_id in ("f10-nets-build-solids", "senior-interpret-different-forms-of-two-dimensional-representations-of"):
        for seed in SEEDS:
            fig = _gen(skill_id, "medium", seed).figure
            cells = [tuple(c) for g in fig["cells"] for c in g["cells"]]
            assert len(cells) == len(set(cells)) == 24
            drawings = list(_option_drawings(fig).values())
            for i, a in enumerate(drawings):
                for b in drawings[i + 1 :]:
                    assert not build.congruent(a, b)


# an oracle for the folding that shares no code with the rolling routine: fold the faces up in 3D
def _fold_in_3d(cells):
    """The outward normal of each cell's face when the net is folded, by hinging each neighbour up along the edge."""
    neg = lambda v: tuple(-c for c in v)  # noqa: E731
    start = min(cells)
    state = {start: ((0, 0, -1), (1, 0, 0), (0, 1, 0))}  # normal, in-plane x direction, in-plane y direction
    todo = [start]
    while todo:
        cell = todo.pop()
        n, ex, ey = state[cell]
        for (dx, dy), nxt in (
            ((1, 0), (ex, neg(n), ey)),
            ((-1, 0), (neg(ex), n, ey)),
            ((0, 1), (ey, ex, neg(n))),
            ((0, -1), (neg(ey), ex, n)),
        ):
            other = (cell[0] + dx, cell[1] + dy)
            if other in cells and other not in state:
                state[other] = nxt
                todo.append(other)
    return {c: s[0] for c, s in state.items()}


def test_the_opposite_face_agrees_with_an_independent_3d_fold():
    for shape in build.cube_nets():
        for oriented in build.orientations(shape):
            normals = _fold_in_3d(set(oriented))
            assert len(set(normals.values())) == 6
            for cell in oriented:
                partner = build.opposite_cell(oriented, cell)
                assert normals[partner] == tuple(-c for c in normals[cell]), (sorted(oriented), cell)


def test_hard_net_question_names_the_opposite_face_and_the_wrong_answers_touch_or_lie_in_line():
    for seed in SEEDS:
        q = _gen("f10-nets-build-solids", "hard", seed)
        by_cell = {(int(lb["at"][0]), int(lb["at"][1])): lb["text"] for lb in q.figure["labels"]}
        assert sorted(by_cell.values()) == list("ABCDEF")
        marked = re.search(r"opposite face (\w)", q.text).group(1)
        marked_cell = next(c for c, t in by_cell.items() if t == marked)
        answer_cell = next(c for c, t in by_cell.items() if f"Face {t}" == q.answer_text)
        assert answer_cell != marked_cell
        assert abs(answer_cell[0] - marked_cell[0]) + abs(answer_cell[1] - marked_cell[1]) > 1  # opposite never touches
        for d in q.distractors:
            assert d != f"Face {marked}"


def test_net_easy_draws_each_of_the_four_solids_and_the_cube_net_is_a_net():
    seen = set()
    for seed in range(120):
        q = make("grid_nets_build_solids", seed=seed, difficulty="easy")
        seen.add(q.answer_text)
        if q.answer_text == "a cube":
            cells = {(min(x for x, _ in p["vertices"]), min(y for _, y in p["vertices"])) for p in q.figure["polygons"]}
            assert is_cube_net(cells)
    assert seen == {"a cube", "a cuboid", "a square pyramid", "a triangular prism"}


# ------------------------------------------------------------------ the tiers differ in reasoning


def test_cube_stack_tiers_differ_in_what_must_be_done():
    easy = [_gen("senior-interpret-diagrams-of-three-dimensional-objects", "easy", s) for s in SEEDS]
    medium = [_gen("senior-interpret-diagrams-of-three-dimensional-objects", "medium", s) for s in SEEDS]
    hard = [_gen("senior-interpret-diagrams-of-three-dimensional-objects", "hard", s) for s in SEEDS]
    assert max(q.figure["grid"]["layers"] for q in easy) <= 2
    assert all(q.figure["grid"]["layers"] == 3 for q in medium)
    assert all(q.figure["grid"]["layers"] >= 3 for q in hard)
    assert all("how many more" in q.text.lower() or "surface area" in q.text.lower() for q in hard)
    assert {"more" if "more" in q.text else "surface" for q in hard} == {"more", "surface"}
    assert all("more" not in q.text and "surface area" not in q.text for q in easy + medium)


def test_cube_stacks_have_no_hidden_cubes_and_the_surface_area_counts_every_face():
    for skill_id in (
        "senior-interpret-diagrams-of-three-dimensional-objects",
        "senior-interpret-different-forms-of-two-dimensional-representations-of",
    ):
        for seed in SEEDS:
            q = _gen(skill_id, "hard" if "diagrams" in skill_id else "easy", seed)
            if "cubes" in q.figure:
                assert grid.tops_all_visible(q.figure["cubes"])
    for seed in SEEDS:
        q = _gen("senior-interpret-diagrams-of-three-dimensional-objects", "hard", seed)
        if "surface area" in q.text:
            cubes = {tuple(c) for c in q.figure["cubes"]}
            exposed = sum(
                (n not in cubes)
                for x, y, z in cubes
                for n in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1))
            )
            assert str(exposed) == q.answer_text


def test_property_tiers_are_figure_then_words_then_euler():
    easy = [_gen("senior-recognise-the-properties-of-common-two-dimensional-geometric", "easy", s) for s in SEEDS]
    medium = [_gen("senior-recognise-the-properties-of-common-two-dimensional-geometric", "medium", s) for s in SEEDS]
    hard = [_gen("senior-recognise-the-properties-of-common-two-dimensional-geometric", "hard", s) for s in SEEDS]
    assert all(q.figure is not None for q in easy)
    assert all(q.figure is None for q in medium + hard)
    assert any("diagonals" in q.text for q in medium) and any(" prism" in q.text or "pyramid" in q.text for q in medium)
    assert any("Euler" in q.text for q in hard) and any("Euler" not in q.text for q in hard)


def test_property_answers_match_the_solids():
    for family, n, counts in (("prism", 5, (7, 15, 10)), ("pyramid", 6, (7, 12, 7))):
        c = build._counts(family, n)
        assert (c["faces"], c["edges"], c["vertices"]) == counts
        assert c["vertices"] + c["faces"] - c["edges"] == 2


def test_composite_tiers_count_then_split_then_estimate():
    easy = [_gen("f10-composite-shape-representation", "easy", s) for s in SEEDS]
    medium = [_gen("f10-composite-shape-representation", "medium", s) for s in SEEDS]
    hard = [_gen("f10-composite-shape-representation", "hard", s) for s in SEEDS]
    assert all("polygons" in q.figure and "cells" not in q.figure for q in easy + medium)
    assert all(len(q.figure["polygons"]) == 1 for q in easy + medium)
    assert all(any(len(set(map(tuple, p["vertices"]))) for p in q.figure["polygons"]) for q in easy)

    # medium outlines have a slanted edge: a triangle to split off; easy outlines are all right angles
    def slanted(q):
        v = q.figure["polygons"][0]["vertices"]
        return any(a[0] != b[0] and a[1] != b[1] for a, b in zip(v, v[1:] + v[:1]))

    assert not any(slanted(q) for q in easy) and all(slanted(q) for q in medium)
    assert all("cells" in q.figure and "polygons" not in q.figure for q in hard)
    styles = {tuple(g["style"] for g in q.figure["cells"]) for q in hard}
    assert styles == {("shaded", "hatched", "dotted"), ("shaded", "hatched")}


def test_composite_hard_rule_is_stated_and_the_answer_follows_it():
    for seed in SEEDS:
        q = _gen("f10-composite-shape-representation", "hard", seed)
        count = {g["style"]: len(g["cells"]) for g in q.figure["cells"]}
        if "dotted" in count:
            assert "ignoring every dotted square" in q.text
            assert q.answer_text == str(count["shaded"] + count["hatched"])
            assert q.answer_text not in {str(sum(count.values())), str(count["shaded"])}
        else:
            assert "half a square" in q.text and count["hatched"] % 2 == 0
            assert float(q.answer_text) == count["shaded"] + count["hatched"] / 2


def test_composite_medium_area_is_a_rectangle_plus_or_minus_a_triangle():
    for seed in SEEDS:
        q = _gen("f10-composite-shape-representation", "medium", seed)
        poly = q.figure["polygons"][0]["vertices"]
        total = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))) / 2
        assert float(q.answer_text) == total
        assert "square units" in q.text


def test_net_view_tiers_differ_in_what_must_be_done():
    sid = "senior-interpret-different-forms-of-two-dimensional-representations-of"
    easy = [_gen(sid, "easy", s) for s in SEEDS]
    medium = [_gen(sid, "medium", s) for s in SEEDS]
    hard = [_gen(sid, "hard", s) for s in SEEDS]
    assert all("cubes" in q.figure for q in easy)
    assert all("cells" in q.figure and "labels" in q.figure for q in medium)
    assert all("polygons" in q.figure and len(q.figure["polygons"]) == 6 for q in hard)
    assert {("surface" if "surface area" in q.text else "volume") for q in hard} == {"surface", "volume"}
    for q in easy:
        assert q.answer_text.startswith("Column heights, left to right:")
        assert len({*q.distractors, q.answer_text}) == 4


def test_hard_net_units_are_asked_for_and_in_every_option():
    sid = "senior-interpret-different-forms-of-two-dimensional-representations-of"
    for seed in SEEDS:
        q = _gen(sid, "hard", seed)
        unit = "square units" if "surface area" in q.text else "cubic units"
        assert q.answer_text.endswith(unit) and all(d.endswith(unit) for d in q.distractors)
