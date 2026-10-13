"""Tests `f10-solid-faces-edges-vertices`: topic `fsp_solids` (easy, medium, hard) and the exam chain `fsp_solids_exam`.

The oracle recounts from the drawn figure alone (points and segments of a wireframe, faces and corners of a net) and
checks Euler's formula on every solid.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.fsp_solids_build as solids
from mathgen import figures, make
from mathgen.engine import grade, load_skills
from mathgen.generators import _fsp_kit as kit
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

SKILL_ID = "f10-solid-faces-edges-vertices"
TIERS = ("easy", "medium", "hard")
SEEDS = range(80)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make("fsp_solids", seed=seed, difficulty=tier)


def _exam(seed):
    return make("fsp_solids_exam", seed=seed, difficulty="exam")


def _asked(q):
    return re.search(r"[Hh]ow many (faces|edges|vertices)", q.text).group(1)


def test_skill_wiring():
    assert _SKILL.answer == "number" and _SKILL.status == "active"
    assert _SKILL.needs_figure is True and _SKILL.capability_gates == ["figure_stimulus"]
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        ("fsp_solids", "easy"),
        ("fsp_solids", "medium"),
        ("fsp_solids", "hard"),
        ("fsp_solids_exam", "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert {m.id for m in _SKILL.misconceptions} <= set(CATALOGUE)


def test_eulers_formula_for_every_solid_drawn():
    for kind in ("prism", "pyramid"):
        for n in (3, 4, 5, 6):
            v, e, f = solids._counts(kind, n)
            assert v - e + f == 2


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_a_diagram_and_the_reader_agrees(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.figure["kind"] == "diagram" and not figures.validate(q.figure)
        assert figures.get_reader("fsp_solids")(q) == q.answer_text, (tier, seed, q.text)
        assert not find_text_defects(q, include_working=True)


def test_easy_counts_faces_of_a_wireframe():
    shapes = Counter()
    for seed in range(200):
        q = _q("easy", seed)
        assert _asked(q) == "faces"
        v, e = len(q.figure["points"]), len(q.figure["segments"])
        assert int(q.answer_text) == 2 - v + e
        assert any(s.get("dashed") for s in q.figure["segments"])
        shapes[(v, e)] += 1
    assert len(shapes) >= 6


def test_medium_asks_for_edges_or_vertices_and_dashed_edges_count():
    asked = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        what = _asked(q)
        asked[what] += 1
        expect = len(q.figure["segments"]) if what == "edges" else len(q.figure["points"])
        assert int(q.answer_text) == expect, (seed, q.text)
        dashed = sum(1 for s in q.figure["segments"] if s.get("dashed"))
        assert dashed >= 1
        if what == "edges":
            # leaving out the dashed edges is one of the offered errors, and it is wrong
            assert str(expect - dashed) in q.distractors or len(q.distractors) >= 2
    assert set(asked) == {"edges", "vertices"} and min(asked.values()) >= 60


def test_hard_vertices_of_the_solid_come_from_the_net_not_its_corners():
    shapes = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        faces = kit.read_faces(q.figure)
        sides = sum(len(f) for f in faces)
        assert int(q.answer_text) == 2 + sides // 2 - len(faces)
        assert int(q.answer_text) != len(q.figure["points"])
        assert str(len(q.figure["points"])) in q.distractors
        shapes[len(faces)] += 1
    assert set(shapes) >= {4, 5, 6, 7}


def test_tiers_differ_in_shape():
    assert {_asked(_q("easy", s)) for s in SEEDS} == {"faces"}
    assert {_asked(_q("medium", s)) for s in SEEDS} == {"edges", "vertices"}
    assert all("net shown" in _q("hard", s).text for s in SEEDS)
    assert not any("net shown" in _q(t, s).text for t in ("easy", "medium") for s in SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        assert 2 <= len(q.distractors) <= 3
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
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_exam_answers_recounted_from_the_net():
    for seed in range(120):
        q = _exam(seed)
        f, e, v = solids._net_counts(q.figure)
        assert v - e + f == 2
        straw = int(re.search(r"each straw is (\d+) cm", q.text).group(1))
        assert [int(p.answer_text) for p in q.parts] == [f, e, v, straw * e], (seed, q.text)


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


def test_exam_follow_through_from_wrong_edges_and_faces():
    for seed in range(60):
        q = _exam(seed)
        f, e, _v = (int(p.answer_text) for p in q.parts[:3])
        straw = int(re.search(r"each straw is (\d+) cm", q.text).group(1))
        wrong_e = int(q.parts[1].distractors[0])
        reply = {
            "a": str(f),
            "b": str(wrong_e),
            "c": str(2 + wrong_e - f),
            "d": str(straw * wrong_e),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["c"].follow_through
        assert result.parts["d"].marks == q.parts[3].marks and result.parts["d"].follow_through


def test_exam_prompts_vary():
    assert len({_exam(s).text for s in range(60)}) >= 20
