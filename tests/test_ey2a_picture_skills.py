"""Five early-years picture skills: counting to 20, ten frames, part-part-whole, comparing and sharing.

Each skill draws counters (or a shape cut into equal parts) and asks for one number that is not in the stem. Every tier
of every skill is generated through the skill's own sources; the figure is valid, must not leak the answer, must agree
with the key through the registered reader, every wrong option must grade wrong with the real grader, and the tiers
differ in shape.
"""

import functools
import re

import pytest

from mathgen import figures, make
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills

TIERS = ["easy", "medium", "hard"]
SEEDS = range(120)

# skill id -> topic
SKILLS = {
    "count-principles-to-20": "ey2a_count_principles",
    "ten-frame-numbers": "ey2a_ten_frame_numbers",
    "part-part-whole-pictures": "ey2a_part_part_whole_pictures",
    "compare-match-quantities-pictures": "ey2a_compare_match_quantities",
    "equal-groups-sharing-halves-pictures": "ey2a_equal_groups_sharing_halves",
}
CASES = [(sid, topic, tier) for sid, topic in SKILLS.items() for tier in TIERS]


@functools.cache
def _skills():
    return {s.id: s for s in load_skills()[1]}


def _questions(topic, tier):
    return [make(topic, seed=seed, difficulty=tier) for seed in SEEDS]


@pytest.mark.parametrize("sid", SKILLS)
def test_skill_is_active_early_and_sources_use_the_topic(sid):
    skill = _skills()[sid]
    assert skill.status == "active"
    assert skill.answer == "number"
    assert [src.topic for src in skill.sources] == [SKILLS[sid]] * 3
    assert [src.difficulty for src in skill.sources] == TIERS
    assert "figure_stimulus" in skill.capability_gates


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_deterministic_and_answer_grades_correct(sid, topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()
        assert grade("number", q, q.answer_text).correct
        assert q.answer == int(q.answer_text)


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_every_distractor_grades_wrong_and_is_tagged_from_the_catalogue(sid, topic, tier):
    for q in _questions(topic, tier):
        assert len(q.distractors) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors
        for d in q.distractors:
            assert not grade("number", q, d).correct, (q.text, d)
            assert d == str(int(d))  # whole numbers, like the answer
        for tag in (q.option_notes or {}).values():
            assert tag in CATALOGUE
            assert tag.startswith("ey2a-")


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_figure_is_valid_does_not_leak_and_agrees_with_the_key(sid, topic, tier):
    reader = figures.get_reader(topic)
    assert reader is not None
    for q in _questions(topic, tier):
        assert figures.validate(q.figure) == []
        assert q.figure["kind"] in {"counters", "fraction_model"}
        figures.assert_no_leak(q.figure, [q.answer_text])
        assert reader(q) == q.answer_text, (q.text, q.figure)


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_stems_are_short_plain_and_do_not_state_the_answer(sid, topic, tier):
    for q in _questions(topic, tier):
        assert len(q.text.split()) <= 18, q.text
        assert q.text.count("?") == 1
        assert "colour" not in q.text and "color" not in q.text  # ringed vs plain, never by hue


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_pool_is_varied(sid, topic, tier):
    qs = _questions(topic, tier)
    assert len({q.answer_text for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 4
    assert len({str(q.figure) for q in qs}) >= 40


def test_count_principles_tiers_differ_in_what_is_done_with_the_count():
    easy = _questions("ey2a_count_principles", "easy")
    medium = _questions("ey2a_count_principles", "medium")
    hard = _questions("ey2a_count_principles", "hard")
    assert all(q.answer == len(q.figure["dots"]) and 11 <= q.answer <= 20 for q in easy)
    assert all(q.answer in (len(q.figure["dots"]) + 1, len(q.figure["dots"]) - 1) for q in medium)
    assert all({d["colour"] for d in q.figure["dots"]} == {1, 2} for q in hard)
    # the hard answer is never the number of counters, and counting every counter is an offered error
    assert all(q.answer != len(q.figure["dots"]) for q in hard)


def test_ten_frame_tiers_use_one_then_two_frames_and_hard_frames_are_never_full():
    easy = _questions("ey2a_ten_frame_numbers", "easy")
    medium = _questions("ey2a_ten_frame_numbers", "medium")
    hard = _questions("ey2a_ten_frame_numbers", "hard")
    assert all(len(q.figure["frames"]) == 1 for q in easy)
    assert all(len(q.figure["frames"]) == 2 for q in medium + hard)
    assert all(sum(1 for b in q.figure["frames"][0] if b) == 10 for q in medium if "move" not in q.text)
    for q in hard:
        assert all(sum(1 for b in f if b) < 10 for f in q.figure["frames"])
    # hard counters sit in scattered boxes: not every frame is filled from the top left
    assert any(f != sorted(f, reverse=True) for q in hard for f in q.figure["frames"])


def test_part_part_whole_tiers_ask_whole_then_missing_part():
    easy = _questions("ey2a_part_part_whole_pictures", "easy")
    medium = _questions("ey2a_part_part_whole_pictures", "medium")
    hard = _questions("ey2a_part_part_whole_pictures", "hard")
    assert all({b for b in q.figure["frames"][0] if b} == {1, 2} for q in easy)
    assert all("plain?" in q.text and {b for b in q.figure["frames"][0] if b} == {2} for q in medium)
    assert {("plain?" in q.text) for q in hard} == {True, False}
    assert all(len(q.figure["frames"]) == 2 for q in hard)


def test_compare_tiers_use_two_then_three_rings():
    easy = _questions("ey2a_compare_match_quantities", "easy")
    medium = _questions("ey2a_compare_match_quantities", "medium")
    hard = _questions("ey2a_compare_match_quantities", "hard")
    assert all(len(q.figure["groups"]) == 2 for q in easy + medium)
    assert all(len(q.figure["groups"]) == 3 and len(set(q.figure["groups"])) == 3 for q in hard)
    assert all(q.answer == max(q.figure["groups"]) for q in easy)
    assert all(q.answer == abs(q.figure["groups"][0] - q.figure["groups"][1]) for q in medium)
    # the stem never says which ring has more
    assert all(not re.search(r"\b(left|right|first|second) (ring|group)", q.text) for q in easy + medium + hard)


def test_sharing_halves_tiers_cover_every_operation_and_stay_within_twenty():
    easy = _questions("ey2a_equal_groups_sharing_halves", "easy")
    medium = _questions("ey2a_equal_groups_sharing_halves", "medium")
    hard = _questions("ey2a_equal_groups_sharing_halves", "hard")
    assert all(q.figure["layout"] == "groups" and len(set(q.figure["groups"])) == 1 for q in easy)
    assert {q.figure["kind"] for q in medium + hard} == {"counters", "fraction_model"}
    assert any("half" in q.text for q in medium) and any("Share" in q.text or "shares" in q.text for q in medium)
    assert any("quarter" in q.text for q in hard) and any("left over" in q.text for q in hard)
    for q in easy + medium + hard:
        total = (
            sum(q.figure["groups"])
            if q.figure["kind"] == "counters" and q.figure["layout"] == "groups"
            else len(q.figure["dots"])
            if q.figure["kind"] == "counters"
            else q.figure["models"][0]["parts"]
        )
        assert total <= 20
