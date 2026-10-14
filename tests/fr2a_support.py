"""Shared checks for the `test_fr2a_*` files (not a test module itself)."""

from __future__ import annotations

from functools import cache

from mathgen import figures, make
from mathgen.engine import load_skills
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TIERS = ("easy", "medium", "hard")


@cache
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def q(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam", tier="exam")


def canonical(question) -> dict[str, str]:
    return {p.label: p.answer_text for p in question.parts}


def assert_wired(skill_id: str, topic: str, exam_topic: str | None, answer: str, figure: bool) -> None:
    sk = skill(skill_id)
    assert sk.status == "active" and sk.answer == answer
    assert [(s.topic, s.difficulty) for s in sk.sources] == [(topic, t) for t in TIERS] + (
        [(exam_topic, "exam")] if exam_topic else []
    )
    if exam_topic:
        assert sk.sources[-1].stages == ["exam"] and sk.sources[-1].params == {"tier": "exam"}
    assert sk.needs_figure is figure
    assert (sk.capability_gates == ["figure_stimulus"]) is figure
    assert {m.id for m in sk.misconceptions} <= set(CATALOGUE)


def assert_single_answer_contract(skill_id: str, topic: str, seeds, tiers=TIERS, reader: bool = True, minimum: int = 3):
    """Deterministic, clean text, a valid figure the reader agrees with, key accepted, every distractor rejected."""
    sk = skill(skill_id)
    for tier in tiers:
        for seed in seeds:
            question = q(topic, tier, seed)
            assert question.to_json() == q(topic, tier, seed).to_json()
            assert find_text_defects(question, include_working=True) == [], (tier, seed, question.text)
            assert question.answer_kind == sk.answer
            assert grade(sk.answer, question, question.answer_text).correct, (tier, seed, question.answer_text)
            assert len(question.distractors) >= minimum and len(set(question.distractors)) == len(question.distractors)
            for d in question.distractors:
                result = grade(sk.answer, question, d)
                assert not result.correct, (tier, seed, d, question.answer_text)
                assert question.option_notes.get(d) in CATALOGUE, (tier, seed, d)
            if question.figure is not None:
                assert figures.validate(question.figure) == []
                figures.assert_no_leak(question.figure, [question.answer_text])
                if reader:
                    got = figures.get_reader(topic)(question)
                    assert got == question.answer_text, (tier, seed, got, question.answer_text)


def assert_exam_contract(skill_id: str, topic: str, seeds, labels="abcd") -> None:
    sk = skill(skill_id)
    for seed in seeds:
        question = exam(topic, seed)
        assert question.to_json() == exam(topic, seed).to_json()
        assert find_text_defects(question) == [], (seed, question.text)
        assert [p.label for p in question.parts] == list(labels)
        full = grade(sk.answer, question, canonical(question))
        assert full.correct and full.marks == full.max_marks, (seed, canonical(question))
        for p in question.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            assert all(p.option_notes.get(d) in CATALOGUE for d in p.distractors), (seed, p.label)
            for d in p.distractors:
                result = grade(sk.answer, question, {**canonical(question), p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d, p.answer_text)
        if question.figure is not None:
            assert figures.validate(question.figure) == []


def assert_follow_through(skill_id: str, question, wrong: dict[str, str], expect: dict[str, str]) -> None:
    """Early parts wrong, later ones right for the learner's own work: the early parts earn nothing and every later
    part earns full marks flagged as follow-through."""
    sk = skill(skill_id)
    result = grade(sk.answer, question, {**wrong, **expect})
    for label in wrong:
        assert result.parts[label].marks == 0, (question.text, label)
    for label in expect:
        part = result.parts[label]
        assert part.marks == part.max_marks and part.follow_through is True, (question.text, label, expect[label])
