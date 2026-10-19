"""Early-years position, directions, steps, turns and simple maps, drawn on a ``grid_drawing``.

Each skill is generated through its own sources at every tier and 60 seeds: the figure is valid and does not show
the key, the registered reader re-derives the key from the figure and prompt, the real grader accepts the key and
rejects every distractor, and every prompt fits the read-aloud cap. The pictures sit in cells the figure's own
coordinates name, so the tests also check the layout the questions are built on.
"""

from __future__ import annotations

import functools
import re

import pytest

from mathgen import figures, make, svg
from mathgen.generators import ey_position_build as build
from mathgen.grading.typed import grade, mcq_options
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills
from scripts import skill_contract

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

SKILLS = {
    "ey-position-words-scene": ("ey_position_words", "mcq", ["ac9mfsp02", "ac9m1sp02"]),
    "ey-follow-directions-grid": ("ey_follow_directions", "mcq", ["ac9m1sp02", "ac9m2sp02"]),
    "ey-count-steps-on-grid": ("ey_count_steps", "number", ["ac9m1sp02", "ac9m2sp02"]),
    "ey-quarter-half-turns": ("ey_turns", "mcq", ["ac9m2m05"]),
    "ey-simple-map-reading": ("ey_map_reading", "mcq", ["ac9m2sp02"]),
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
        return grade("mcq", q, str(mcq_options(q).index(text))).correct
    return grade("number", q, text).correct


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


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("tier", TIERS)
def test_every_tagged_distractor_is_a_listed_misconception(skill_id, tier):
    listed = {m.id for m in _skills()[skill_id].misconceptions}
    tags = set()
    for seed in SEEDS:
        q = _q(skill_id, tier, seed)
        assert set(q.option_notes) <= set(q.distractors), (tier, seed)
        assert q.option_notes, (tier, seed)
        tags |= set(q.option_notes.values())
    assert tags and tags <= listed and tags <= set(CATALOGUE), (tags - listed, tags - set(CATALOGUE))


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
        assert len(q.distractors) >= 3 and q.answer_text not in q.distractors
        for d in q.distractors:
            assert not _is_right(skill_id, q, d), (tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("tier", TIERS)
def test_every_prompt_is_short_and_varied(skill_id, tier):
    texts = {_q(skill_id, tier, seed).text for seed in SEEDS}
    assert max(len(t.split()) for t in texts) <= READ_ALOUD_MAX_WORDS
    assert len(texts) >= 2
    figs = {repr(_q(skill_id, tier, seed).figure) for seed in SEEDS}
    assert len(figs) >= 40


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(skill_id, family):
    skill_contract.FAMILIES[family](skill_id)


@pytest.mark.parametrize("topic", sorted({v[0] for v in SKILLS.values()}))
def test_unknown_tier_raises(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, tier="bogus")


@pytest.mark.parametrize("skill_id", ["ey-position-words-scene", "ey-follow-directions-grid", "ey-simple-map-reading"])
def test_mcq_key_position_is_balanced(skill_id):
    for tier in TIERS:
        counts: dict[int, int] = {}
        for seed in range(200):
            q = make(SKILLS[skill_id][0], seed=seed, difficulty=tier)
            pos = mcq_options(q).index(q.answer_text)
            counts[pos] = counts.get(pos, 0) + 1
        assert max(counts.values()) / 200 < 0.5, (skill_id, tier, counts)


def test_pictures_are_named_in_plain_words_and_never_labelled():
    assert set(build.WORD.values()) == set(build.KEY_OF)
    for skill_id in SKILLS:
        for tier in TIERS:
            for seed in range(10):
                fig = _q(skill_id, tier, seed).figure
                for label in fig.get("labels", []):
                    assert label["text"] in ("Start", "You")
                assert not fig.get("legend")


# ------------------------------------------------------------------ tier shape


def test_position_words_easy_is_one_line_of_five():
    for seed in SEEDS:
        fig = _q("ey-position-words-scene", "easy", seed).figure
        grid = fig["grid"]
        assert sorted((grid["width"], grid["height"])) == [1, 5] and len(fig["symbols"]) == 5


def test_position_words_medium_has_the_neighbour_on_both_sides():
    for seed in SEEDS:
        q = _q("ey-position-words-scene", "medium", seed)
        pics = build.symbols_of(q.figure)
        assert 5 <= len(pics) <= 6 and q.answer_text in pics.values()
        assert "just" in q.text or "directly" in q.text


def test_position_words_hard_uses_all_three_kinds():
    kinds = set()
    for seed in range(120):
        text = make("ey_position_words", seed=seed, difficulty="hard").text
        kinds.add("two" if " and just " in text else "between" if "between" in text else "where")
    assert kinds == {"two", "between", "where"}


def test_where_options_are_the_four_ways():
    seen = set()
    for seed in range(120):
        q = make("ey_position_words", seed=seed, difficulty="hard")
        if "Where is" in q.text:
            seen.add(frozenset([q.answer_text, *q.distractors]))
    assert seen == {frozenset(["above", "below", "to the left", "to the right"])}


def test_follow_directions_moves_get_longer():
    for seed in SEEDS:
        easy, medium, hard = (_q("ey-follow-directions-grid", t, seed) for t in TIERS)
        counts = [len(build._MOVE_RE.findall(q.text)) for q in (easy, medium, hard)]
        assert counts == [1, 2, 3], (seed, counts)


def test_follow_directions_hard_goes_back_along_the_first_move():
    for seed in SEEDS:
        moves = build._MOVE_RE.findall(_q("ey-follow-directions-grid", "hard", seed).text)
        assert build.OPPOSITE[moves[0][0]] == moves[2][0]


def test_follow_directions_wrong_options_are_on_the_grid_and_distinct_from_start():
    for tier in TIERS:
        for seed in SEEDS:
            q = _q("ey-follow-directions-grid", tier, seed)
            (label,) = q.figure["labels"]
            start = (int(label["at"][0]), int(label["at"][1]))
            assert start not in build.symbols_of(q.figure)


def test_count_steps_path_shapes():
    for seed in SEEDS:
        easy = _q("ey-count-steps-on-grid", "easy", seed).figure["path"]["cells"]
        assert len({c[0] for c in easy}) == 1 or len({c[1] for c in easy}) == 1
        medium = _q("ey-count-steps-on-grid", "medium", seed).figure["path"]["cells"]
        assert len({c[0] for c in medium}) > 1 and len({c[1] for c in medium}) > 1
        hard = _q("ey-count-steps-on-grid", "hard", seed)
        assert "path" not in hard.figure and 3 <= int(hard.answer_text) <= 8


def test_count_steps_hard_distractors_include_the_corner_cut():
    hits = 0
    for seed in SEEDS:
        q = _q("ey-count-steps-on-grid", "hard", seed)
        pics = build.symbols_of(q.figure)
        where = {w: c for c, w in pics.items()}
        a, b = build._names_in(q.text)[:2]
        dx, dy = abs(where[a][0] - where[b][0]), abs(where[a][1] - where[b][1])
        assert dx != dy and str(dx + dy) == q.answer_text
        hits += str(max(dx, dy)) in q.distractors
    assert hits >= 40


def test_turns_figure_faces_the_picture_the_arrow_points_at():
    for seed in SEEDS:
        for tier in TIERS:
            fig = _q("ey-quarter-half-turns", tier, seed).figure
            a, b = fig["path"]["cells"]
            assert a == [2, 2] and abs(b[0] - 2) + abs(b[1] - 2) == 1
            assert sorted(build._SEAT) == sorted(build.symbols_of(fig))
        assert _q("ey-quarter-half-turns", "easy", seed).figure["path"]["cells"][1] == [2, 3]  # facing up


def test_turns_medium_and_hard_face_every_way_and_use_both_hands():
    faces, texts = set(), set()
    for seed in range(120):
        fig = make("ey_turns", seed=seed, difficulty="medium").figure
        faces.add(tuple(fig["path"]["cells"][1]))
        texts.add(re.sub(r"[^a-z ]", "", make("ey_turns", seed=seed, difficulty="medium").text.lower()))
    assert len(faces) == 4
    joined = " ".join(texts)
    assert "left" in joined and "right" in joined and "half turn" in joined


def test_turns_hard_has_two_turn_and_which_turn_questions():
    kinds = set()
    for seed in range(120):
        text = make("ey_turns", seed=seed, difficulty="hard").text
        kinds.add("which" if "Which turn" in text or "which turn" in text.lower() else "two")
    assert kinds == {"which", "two"}


def test_turns_two_turns_never_cancel():
    for seed in range(120):
        q = make("ey_turns", seed=seed, difficulty="hard")
        turns = build._TURN_RE.findall(q.text)
        if len(turns) == 2:
            net = sum(2 if size == "half" else 1 if side == "right" else -1 for size, side in turns)
            assert net % 4 != 0


def test_map_easy_answer_is_alone_at_its_edge():
    for seed in SEEDS:
        q = _q("ey-simple-map-reading", "easy", seed)
        cells = [(int(s["at"][0]), int(s["at"][1])) for s in q.figure["symbols"]]
        assert len({c[0] for c in cells}) == 5 and len({c[1] for c in cells}) == 5


def test_map_medium_has_a_single_closest_or_farthest():
    for seed in SEEDS:
        q = _q("ey-simple-map-reading", "medium", seed)
        pics = build.symbols_of(q.figure)
        (x,) = build._names_in(q.text)
        ref = next(c for c, w in pics.items() if w == x)
        dists = sorted(build._dist(ref, c) for c in pics if c != ref)
        pick = dists[-2:] if "farthest" in q.text else dists[:2]
        assert pick[0] != pick[1]


def test_map_hard_closest_pair_is_unique_and_a_corner_cut_is_offered():
    offered = 0
    for seed in SEEDS:
        q = _q("ey-simple-map-reading", "hard", seed)
        pics = build.symbols_of(q.figure)
        cells = list(pics)
        dist = sorted(build._dist(a, b) for i, a in enumerate(cells) for b in cells[i + 1 :])
        assert dist[0] < dist[1] and dist[0] >= 2
        offered += len(q.distractors) == 3
    assert offered == len(SEEDS)


# ------------------------------------------------------------------ the helpers refuse nonsense


def test_the_figure_refuses_a_picture_off_the_grid():
    with pytest.raises(Exception):
        build._figure(3, 3, {(3, 0): "house"}, alt="x")
    with pytest.raises(Exception):
        build._figure(3, 3, {(0, 0): "house"}, alt="x", path=[(0, 0), (2, 2)])


def test_a_question_needs_three_distractors():
    fig = build._figure(2, 1, {(0, 0): "house", (1, 0): "tree"}, alt="two pictures")
    with pytest.raises(Exception):
        build._mcq("What is next to the house?", "tree", [build.Distractor("house")], fig, ["x"])
