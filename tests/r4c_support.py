"""Shared checks for the `r4c_*` abstract-algebra skills (`uc2-*`): the same contract tests run on every one.

Each skill's own test file adds an oracle that re-derives the key from the question TEXT by a different route.
"""

from __future__ import annotations

from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TIERS = ("easy", "medium", "hard")


def skill(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def q(topic, tier, seed):
    return make(topic, seed=seed, difficulty=tier)


def exam(topic, seed):
    return make("exd_x_" + topic, seed=seed, difficulty="exam")


def check_wiring(skill_id, topic, kind):
    s = skill(skill_id)
    assert s.answer == kind
    assert [(src.topic, src.difficulty) for src in s.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        ("exd_x_" + topic, "exam"),
    ]
    assert s.sources[-1].stages == ["exam"]
    assert "university-year-2" in s.tags and tuple(s.year_level) == ("university",)
    assert s.prerequisites


def check_keys_and_distractors(skill_id, topic, kind, seeds=60):
    s = skill(skill_id)
    for tier in TIERS:
        for seed in range(seeds):
            question = q(topic, tier, seed)
            assert question.answer_kind == kind
            assert grade(s, question, question.answer_text).correct, (tier, seed, question.answer_text)
            assert len(question.distractors) >= 2, (tier, seed)
            tagged = 0
            for text in question.distractors:
                assert not grade(s, question, text).correct, (tier, seed, text, question.answer_text)
                note = question.option_notes.get(text)
                if note is not None:
                    assert note in CATALOGUE, note
                    tagged += 1
            assert tagged >= 1, (tier, seed)


def check_text_and_determinism(topic, seeds=40):
    for tier in TIERS:
        for seed in range(seeds):
            question = q(topic, tier, seed)
            assert find_text_defects(question) == [], (tier, seed, find_text_defects(question))
            assert question.to_json() == q(topic, tier, seed).to_json()
    for seed in range(seeds):
        question = exam(topic, seed)
        assert find_text_defects(question) == [], (seed, find_text_defects(question))
        assert question.to_json() == exam(topic, seed).to_json()


def check_answer_space(topic, minimum, seeds=80, text_fraction=0.6):
    for tier in TIERS:
        answers = [q(topic, tier, seed).answer_text for seed in range(seeds)]
        texts = [q(topic, tier, seed).text for seed in range(seeds)]
        assert len(set(texts)) >= seeds * text_fraction, (tier, len(set(texts)))
        assert len(set(answers)) >= minimum[tier], (tier, Counter(answers).most_common(4))
        assert max(Counter(answers).values()) <= seeds * 0.55, (tier, Counter(answers).most_common(3))


def check_exam_shape(skill_id, topic, labels, marks, kinds, follow_through):
    s = skill(skill_id)
    for seed in range(60):
        question = exam(topic, seed)
        assert [p.label for p in question.parts] == list(labels)
        assert [p.marks for p in question.parts] == list(marks)
        assert [p.answer_kind for p in question.parts] == list(kinds)
        assert [bool(p.follow_through) for p in question.parts] == list(follow_through)
        replies = {p.label: p.answer_text for p in question.parts}
        full = grade(s, question, replies)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, full)
        for p in question.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(s, question, {**replies, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                note = p.option_notes.get(text)
                assert note is None or note in CATALOGUE


def check_follow_through(skill_id, topic, label, make_replies, expected, seeds=60):
    """Part `label` earns its marks for the learner's own earlier replies. `make_replies(question)` returns the
    replies dict to send (earlier parts wrong); `expected(question)` is the text that part `label` should then be
    right with, worked out by the test itself."""
    s = skill(skill_id)
    checked = 0
    for seed in range(seeds):
        question = exam(topic, seed)
        replies = make_replies(question)
        if replies is None:
            continue
        part = next(p for p in question.parts if p.label == label)
        replies = {**replies, label: expected(question, replies)}
        result = grade(s, question, replies)
        assert result.parts[label].marks == part.marks, (seed, label, replies, result.parts[label])
        assert result.parts[label].follow_through is True, (seed, label, replies)
        checked += 1
    assert checked >= seeds // 2
