"""Shared checks for the `test_fr2c_*.py` files: every skill's key is accepted and every wrong option rejected."""

from __future__ import annotations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def check_single(skill_id: str, topic: str, tier: str, seeds=range(60), *, min_options: int = 3):
    """Key accepted, every distractor rejected and tagged, text clean, same decimal places for numbers."""
    sk = skill(skill_id)
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        assert grade(sk, q, q.answer_text).correct, (topic, tier, seed, q.text)
        assert len(q.distractors) >= min_options, (topic, tier, seed, q.distractors)
        for text in q.distractors:
            result = grade(sk, q, text)
            assert not result.correct, (topic, tier, seed, text)
            assert result.misconception in CATALOGUE, (topic, tier, seed, text, result.misconception)
        assert not find_text_defects(q, include_working=True), (topic, tier, seed)
        if q.answer_kind == "number":
            places = len(q.answer_text.partition(".")[2])
            assert all(len(d.partition(".")[2]) == places for d in q.distractors), (seed, q.answer_text, q.distractors)
        if q.answer_kind == "mcq":
            assert len(q.distractors) + 1 >= 4, (topic, tier, seed)


def check_parts(skill_id: str, topic: str, seeds=range(40), *, marks=None, kinds=None, follow=None):
    """Full marks for the key, no marks for any wrong option of any part, the shape of the chain as stated."""
    sk = skill(skill_id)
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        assert 3 <= len(q.parts) <= 5
        if marks:
            assert [p.marks for p in q.parts] == marks, (seed, [p.marks for p in q.parts])
        if kinds:
            assert [p.answer_kind for p in q.parts] == kinds, (seed, [p.answer_kind for p in q.parts])
        if follow:
            assert [bool(p.follow_through) for p in q.parts] == follow
        canonical = {p.label: p.answer_text for p in q.parts}
        result = grade(sk, q, canonical)
        assert result.correct and result.marks == result.max_marks, (seed, canonical)
        assert not find_text_defects(q, include_working=True), (topic, seed)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            if p.answer_kind == "mcq":
                assert len(p.distractors) >= 3
            for d in p.distractors:
                wrong = grade(sk, q, {**canonical, p.label: d})
                assert wrong.parts[p.label].marks == 0, (seed, p.label, d)
                assert wrong.parts[p.label].misconception in CATALOGUE or wrong.parts[p.label].misconception is None


def check_follow_through(skill_id: str, topic: str, label: str, replace, later: str, seeds=range(25)):
    """A wrong part ``label`` (``replace(part)`` gives its reply) still earns the marks of part ``later`` when the later
    reply is right for the learner's own earlier reply."""
    sk = skill(skill_id)
    earned = 0
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        parts = {p.label: p for p in q.parts}
        canonical = {p.label: p.answer_text for p in q.parts}
        reply, later_reply = replace(q, parts)
        if reply is None:
            continue
        result = grade(sk, q, {**canonical, label: reply, later: later_reply})
        assert result.parts[label].marks == 0
        if result.parts[later].marks:
            assert result.parts[later].follow_through, (seed, later_reply)
            earned += 1
    assert earned >= 5, earned
