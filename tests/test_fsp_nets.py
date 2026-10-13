"""Tests `f10-nets-build-solids`: topic `fsp_nets` (easy, medium, hard) and the exam chain `fsp_nets_exam`.

The oracle for the single-answer tiers is the registered figure reader, which rebuilds the key from the drawn
segments alone; the facts about cube nets (eleven of them, found by two independent folds) are checked directly.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.fsp_nets_build as nets
from mathgen import figures, make
from mathgen.engine import grade, load_skills
from mathgen.generators import _fsp_kit as kit
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILL_ID = "f10-nets-build-solids"
TIERS = ("easy", "medium", "hard")
SEEDS = range(80)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make("fsp_nets", seed=seed, difficulty=tier)


def _exam(seed):
    return make("fsp_nets_exam", seed=seed, difficulty="exam")


# -- the geometry underneath ----------------------------------------------------------------------------------------


def test_exactly_eleven_free_hexominoes_fold_into_a_cube():
    valid, invalid = kit.hexomino_nets()
    assert len({nets._canonical(c) for c in valid}) == 11
    assert {nets._canonical(c) for c in valid} == nets._ELEVEN
    assert not {nets._canonical(c) for c in invalid} & nets._ELEVEN
    # both folding methods agree on every loop-free hexomino
    for cells in valid + invalid:
        faces_a = {frozenset(kit.cube_corner_map(cells)[c] for c in _cell_corners(cell)) for cell in cells}
        assert (len(faces_a) == 6) == kit.folds_into_cube(cells)


def _cell_corners(cell):
    x, y = cell
    return [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]


def test_every_cube_net_has_the_twelve_cube_edges():
    for cells in kit.hexomino_nets()[0]:
        nets._check_fold_consistent(cells, kit.cube_corner_map(cells))


# -- wiring ---------------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "mcq" and _SKILL.status == "active"
    assert _SKILL.needs_figure is True and _SKILL.capability_gates == ["figure_stimulus"]
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        ("fsp_nets", "easy"),
        ("fsp_nets", "medium"),
        ("fsp_nets", "hard"),
        ("fsp_nets_exam", "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    ids = {m.id for m in _SKILL.misconceptions}
    assert ids <= set(CATALOGUE)


# -- single-answer tiers --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_a_diagram_and_the_reader_agrees(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.figure["kind"] == "diagram" and not figures.validate(q.figure)
        assert figures.get_reader("fsp_nets")(q) == q.answer_text, (tier, seed, q.text)
        assert not find_text_defects(q, include_working=True)


def test_easy_covers_every_drawn_solid():
    seen = Counter(_q("easy", seed).answer_text for seed in range(300))
    assert set(seen) == set(nets._DRAWN), seen


def test_easy_only_the_key_fits_the_net():
    for seed in SEEDS:
        q = _q("easy", seed)
        signature = nets._figure_signature(q.figure)
        assert nets._solid_signature(q.answer_text) == signature
        for d in q.distractors:
            assert nets._solid_signature(d) != signature, (seed, d)


def test_easy_never_offers_a_rectangular_prism_against_a_cube():
    for seed in range(300):
        q = _q("easy", seed)
        if q.answer_text == "Cube":
            assert "Rectangular prism" not in q.distractors


def test_medium_keys_match_the_squares_drawn():
    seen = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        faces = kit.read_faces(q.figure)
        assert all(kit.is_square(f) for f in faces)
        cells = nets._cells_from_squares(q.figure, faces)
        if len(cells) == 5:
            expect = nets._FEW
        elif nets._is_known_cube_net(cells):
            expect = nets._YES
        else:
            expect = nets._OVERLAP
        assert q.answer_text == expect, (seed, q.text)
        seen[expect] += 1
    assert min(seen.values()) >= 15 and len(seen) == 3


def test_hard_key_is_the_set_of_corners_that_meet():
    for seed in SEEDS:
        q = _q("hard", seed)
        letters = {p["label"] for p in q.figure["points"] if p.get("label")}
        assert len(letters) == len(q.figure["points"])
        start = re.search(r"[Cc]orner ([A-Z])\b", q.text).group(1)
        key = q.answer_text.split(" and ")
        assert len(key) == 2 and start not in key and set(key) <= letters
        for d in q.distractors:
            assert len(d.split(" and ")) == 2


def test_tiers_differ_in_shape():
    easy = {len(_q("easy", s).figure["points"]) for s in SEEDS}
    medium = [_q("medium", s) for s in SEEDS]
    hard = [_q("hard", s) for s in SEEDS]
    assert len(easy) >= 4
    assert all(not any(p.get("label") for p in q.figure["points"]) for q in medium)
    assert all(all(p.get("label") for p in q.figure["points"]) for q in hard)
    assert {q.answer_text for q in medium} <= {nets._YES, nets._OVERLAP, nets._FEW}
    assert all(" and " in q.answer_text and "No" not in q.answer_text for q in hard)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
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
        assert [p.marks for p in q.parts] == [1, 2, 2, 1]
        assert [p.answer_kind for p in q.parts] == ["mcq", "number", "number", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert q.figure["not_to_scale"] is True


def test_exam_parts_follow_from_the_measurements():
    for seed in range(120):
        q = _exam(seed)
        l, perimeter, h = nets._exam_measurements(q.figure)
        w = perimeter // 2 - l
        assert (l, w, h) != (w, l, h) and len({l, w, h}) == 3
        assert {p.label: p.answer_text for p in q.parts} == {
            "a": "Rectangular prism",
            "b": str(w),
            "c": str(2 * (l * w + l * h + w * h)),
            "d": str(l * w * h),
        }
        assert nets._read_exam(q) == q.answer_text


def test_exam_full_marks_and_distractors_rejected():
    for seed in range(60):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert 2 <= len(p.distractors) <= 3
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_exam_follow_through_from_a_wrong_width():
    for seed in range(60):
        q = _exam(seed)
        l, _perimeter, h = nets._exam_measurements(q.figure)
        wrong_b = q.parts[1].distractors[0]
        w = int(wrong_b)
        reply = {
            "a": q.parts[0].answer_text,
            "b": wrong_b,
            "c": str(2 * (l * w + l * h + w * h)),
            "d": str(l * w * h),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["c"].follow_through
        assert result.parts["d"].marks == q.parts[3].marks and result.parts["d"].follow_through


def test_exam_prompts_vary():
    assert len({_exam(s).text for s in range(60)}) >= 20
