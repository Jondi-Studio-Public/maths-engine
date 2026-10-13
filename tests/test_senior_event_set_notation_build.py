"""Build tests for `senior-use-set-language-and-notation-for-events-including` (`answer: set`): the skill's shape,
the shared skill contract families, the real `set` grader on every canonical answer and every distractor, the empty
set (mutually exclusive events) and the tier ladder."""

from __future__ import annotations

import pytest

from mathgen import make
from mathgen.grading.typed import grade_set
from mathgen.skills import load_skills
from scripts.measure_tiers import classify_skill
from scripts.skill_contract import FAMILIES

SKILL_ID = "senior-use-set-language-and-notation-for-events-including"
TOPIC = "senior_event_set_notation"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(120)

_SKILLS = {s.id: s for s in load_skills()[1]}
SKILL = _SKILLS[SKILL_ID]


def _questions(tier):
    return [make(TOPIC, seed=s, difficulty=tier) for s in SEEDS]


def _reply(item):
    return "(" + ", ".join(str(x) for x in item) + ")" if isinstance(item, list) else str(item)


def test_skill_shape():
    assert SKILL.status == "active"
    assert SKILL.answer == "set"
    assert SKILL.domain == "statistics-probability"
    assert [src.difficulty for src in SKILL.sources] == list(TIERS)
    assert SKILL.sources[-1].stages == ["exam"]
    assert all(src.topic == TOPIC for src in SKILL.sources)
    assert all(p in _SKILLS for p in SKILL.prerequisites)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_shared_skill_contract(family):
    FAMILIES[family](SKILL_ID)


@pytest.mark.parametrize("tier", TIERS)
def test_the_set_grader_accepts_the_answer_in_any_order(tier):
    for q in _questions(tier):
        assert q.answer_kind == "set"
        assert grade_set(q, q.answer_text).correct, q.text
        assert grade_set(q, ", ".join(_reply(i) for i in reversed(q.answer))).correct, q.text


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_rejected_and_tagged(tier):
    declared = {m.id for m in SKILL.misconceptions}
    for q in _questions(tier):
        assert q.distractors, q.text
        for text in q.distractors:
            assert not grade_set(q, text).correct, (q.text, text)
            assert q.option_notes[text] in declared, (q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_a_missing_or_extra_outcome_is_rejected(tier):
    for q in _questions(tier):
        items = [_reply(i) for i in q.answer]
        if items:
            assert not grade_set(q, ", ".join(items[:-1])).correct, q.text
        assert not grade_set(q, ", ".join(items + ["zz"])).correct, q.text


@pytest.mark.parametrize("tier", ("medium", "hard", "exam"))
def test_the_empty_set_is_an_answer_from_the_worded_tiers_and_is_accepted_in_every_spelling(tier):
    empties = [q for q in (make(TOPIC, seed=s, difficulty=tier) for s in range(400)) if not q.answer]
    assert empties, "mutually exclusive events should appear in the worded tiers"
    for q in empties:
        assert q.answer_text == "{}"
        for reply in ("{}", "{ }", "∅", "the empty set"):
            assert grade_set(q, reply).correct, (q.text, reply)
        assert not grade_set(q, "{1}").correct


def test_the_listed_tier_never_asks_for_a_trivial_answer():
    for q in _questions("easy"):
        assert q.answer, q.text


def test_the_tiers_escalate():
    verdict, adds = classify_skill(SKILL, 60)
    assert verdict == "escalates", adds
    assert adds == [False, True, True, True]


def test_questions_are_deterministic():
    for tier in TIERS:
        assert make(TOPIC, seed=11, difficulty=tier).to_json() == make(TOPIC, seed=11, difficulty=tier).to_json()


def test_the_latex_prompt_uses_real_set_symbols():
    for tier in TIERS:
        for q in _questions(tier)[:30]:
            assert "∩" not in q.latex and "∪" not in q.latex, q.latex
            if "∩" in q.text:
                assert r"\cap" in q.latex
            if "∪" in q.text:
                assert r"\cup" in q.latex
