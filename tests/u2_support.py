"""Shared checks for the `u2_*` first-year university skills: every key is accepted, every distractor is rejected and
tagged, text is clean and deterministic, and an exam chain pays full marks, zero for a distractor and follow-through."""

from __future__ import annotations

from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TIERS = ("easy", "medium", "hard")


def skill(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def check_wiring(sk, topic, exam_topic, answer_kind):
    assert sk.answer == answer_kind
    assert sk.status == "active"
    assert [(s.topic, s.difficulty) for s in sk.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam_topic, "exam"),
    ]
    assert sk.sources[-1].stages == ["exam"]


def check_tier(sk, topic, tier, seeds=80, minimum_options=2):
    """Key accepted, every distractor rejected and tagged, text clean, same seed same question."""
    for seed in range(seeds):
        q = make(topic, seed=seed, difficulty=tier)
        assert grade(sk, q, q.answer_text).correct, (tier, seed, q.text, q.answer_text)
        assert len(q.distractors) >= minimum_options, (tier, seed)
        for text in q.distractors:
            assert not grade(sk, q, text).correct, (tier, seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE, (tier, seed, text)
        assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()


def check_exam(sk, exam_topic, labels, marks, follow_through, seeds=60):
    """Chain shape, full marks for the key, zero for each distractor in its own part, determinism."""
    for seed in range(seeds):
        q = make(exam_topic, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == list(labels)
        assert [p.marks for p in q.parts] == list(marks)
        assert [bool(p.follow_through) for p in q.parts] == list(follow_through)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == make(exam_topic, seed=seed, difficulty="exam").to_json()
        canon = {p.label: p.answer_text for p in q.parts}
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, full)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(sk, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE


def check_follow_through(sk, exam_topic, label, later, wrong, later_value, seeds=40):
    """A wrong reply to `label` (`wrong` may be a function of the question) still earns `later` its marks when
    `later_value(q, wrong)` is right for it."""
    done = 0
    for seed in range(seeds):
        q = make(exam_topic, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        bad = wrong(q) if callable(wrong) else wrong
        reply = {**canon, label: str(bad), later: str(later_value(q, bad))}
        result = grade(sk, q, reply)
        assert result.parts[label].marks == 0, (seed, reply)
        part = result.parts[later]
        assert part.marks == next(p.marks for p in q.parts if p.label == later), (seed, reply, part)
        assert part.follow_through is True, (seed, reply)
        done += 1
    assert done == seeds


def answer_variety(topic, difficulty, seeds=60, minimum=4):
    answers = {make(topic, seed=s, difficulty=difficulty).answer_text for s in range(seeds)}
    assert len(answers) >= minimum, answers


def distinct_stems(topic, tier, seeds=60, minimum=40):
    stems = {make(topic, seed=s, difficulty=tier).text for s in range(seeds)}
    assert len(stems) >= minimum
