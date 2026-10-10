"""mathgen.engine: the public load / make / grade API."""

import pytest

from mathgen import engine


@pytest.fixture(scope="module")
def skills():
    return {s.id: s for s in engine.load_skills()[1]}


def test_make_for_skill_is_deterministic_and_graded_correct(skills):
    skill = skills["simplify-surds"]
    a = engine.make_for_skill(skill, seed=7)
    b = engine.make_for_skill(skill, seed=7)
    assert a.text == b.text and a.answer_text == b.answer_text
    assert engine.grade(skill, a, a.answer_text).correct


def test_wrong_answer_is_rejected(skills):
    skill = skills["simplify-surds"]
    q = engine.make_for_skill(skill, seed=3)
    assert not engine.grade(skill, q, "definitely not the answer").correct


def test_source_can_be_picked_by_index_label_or_object(skills):
    skill = next(s for s in skills.values() if len(s.sources) > 1 and s.sources[-1].difficulty)
    last = skill.sources[-1]
    by_index = engine.pick_source(skill, len(skill.sources) - 1)
    by_label = engine.pick_source(skill, last.difficulty)
    assert by_index is last
    assert by_label.difficulty == last.difficulty
    assert engine.pick_source(skill, last) is last
    with pytest.raises(KeyError):
        engine.pick_source(skill, "no-such-difficulty")


def test_every_answer_kind_in_the_library_has_a_grader(skills):
    from mathgen.grading import GRADERS

    kinds = {s.answer for s in skills.values() if s.status == "active"} - {"rubric"}
    assert kinds <= set(GRADERS)


def test_missing_default_gate_registry_means_no_gates_not_a_crash(monkeypatch, tmp_path):
    from mathgen import capabilities

    monkeypatch.setenv("MATHS_CAPABILITY_GATES_FILE", str(tmp_path / "absent.yaml"))
    registry = capabilities.load_capability_gates()
    assert registry == {}
    assert not capabilities.gate_passed("technology_use", registry)  # unknown gate: fails closed
    assert capabilities.skill_gates_satisfied([], registry)
