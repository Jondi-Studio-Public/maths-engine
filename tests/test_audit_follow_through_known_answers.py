"""Follow-through carried wholly from known answers (canonical or a catalogued distractor) earns its marks even when
it lands on the same value as the earlier reply; the same junk number in every box still earns nothing."""

from __future__ import annotations

import pytest

from mathgen import engine

_SKILLS = {s.id: s for s in engine.load_skills()[1]}


def _exam_question(skill_id, seed):
    skill = _SKILLS[skill_id]
    source = next(i for i, s in enumerate(skill.sources) if s.difficulty == "exam")
    return skill, engine.make_for_skill(skill, seed, source=source)


def _marks(skill, question, reply):
    result = engine.grade(skill, question, reply)
    return {label: part.correct for label, part in result.parts.items()}


def test_distractor_then_its_follow_through_on_the_same_value():
    skill, question = _exam_question("senior-use-examples-and-counter-examples", 1)
    assert "104" in question.parts[0].distractors
    assert _marks(skill, question, {"a": "104", "b": "104"}) == {"a": False, "b": True}


def test_chain_of_follow_through_through_a_max():
    skill, question = _exam_question("max-shear-force-udl", 1)
    assert "33" in question.parts[0].distractors
    assert _marks(skill, question, {"a": "33", "b": "33", "c": "33"}) == {"a": False, "b": True, "c": True}


def test_follow_through_from_a_negative_one_distractor():
    skill, question = _exam_question("jacobi-iteration-2x2", 2)
    assert "-1" in question.parts[1].distractors
    got = _marks(skill, question, {"a": "0", "b": "-1", "c": "-1", "d": "17/7"})
    assert got["c"] is True


@pytest.mark.parametrize("junk", ["7", "59", "1"])
def test_the_same_unknown_number_in_every_box_is_still_refused(junk):
    skill, question = _exam_question("max-shear-force-udl", 1)
    got = _marks(skill, question, {"a": junk, "b": junk, "c": junk})
    assert got["b"] is False
    assert got["c"] is False
