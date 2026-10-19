"""Shared checks for the ``test_s2c3_*`` files: grading, distractors, text quality and exam-chain behaviour."""

from __future__ import annotations

from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects


def skill_by_id(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def tier_question(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def assert_key_accepted_and_distractors_rejected(skill, topic: str, tier: str, seeds=range(60), minimum: int = 2):
    for seed in seeds:
        q = tier_question(topic, tier, seed)
        assert grade(skill, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= minimum, (seed, q.text)
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


def assert_text_quality_and_determinism(topic: str, tier: str, seeds=range(40)):
    for seed in seeds:
        q = tier_question(topic, tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == tier_question(topic, tier, seed).to_json()


def assert_exam_grading(skill, topic: str, marks: list[int], kinds: list[str], seeds=range(40)):
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == [chr(97 + i) for i in range(len(marks))]
        assert [p.marks for p in q.parts] == marks
        assert [p.answer_kind for p in q.parts] == kinds
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE


def assert_follow_through(skill, topic: str, label: str, wrong_replies: dict, expected: callable, seeds=range(30)):
    """Reply wrongly to the earlier parts (`wrong_replies`, label -> text) and check part `label` still earns its marks
    when the reply is right for those wrong values: `expected(question, replies)` gives the text it should be."""
    checked = 0
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, **{k: v(q) if callable(v) else v for k, v in wrong_replies.items()}}
        replies[label] = expected(q, replies)
        result = grade(skill, q, replies)
        part = result.parts[label]
        assert part.marks == next(p.marks for p in q.parts if p.label == label), (seed, replies, result)
        assert part.follow_through is True, (seed, replies)
        checked += 1
    assert checked == len(list(seeds))


def spread(topic: str, tier: str, seeds=range(80), minimum_texts: int = 70, minimum_answers: int = 8):
    texts = [tier_question(topic, tier, s).text for s in seeds]
    answers = [tier_question(topic, tier, s).answer_text for s in seeds]
    assert len(set(texts)) >= minimum_texts, len(set(texts))
    assert len(set(answers)) >= minimum_answers, Counter(answers).most_common(4)
