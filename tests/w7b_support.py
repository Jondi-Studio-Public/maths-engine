"""Shared checks for the `w7b_*` skill tests: every tier's key grades correct, every distractor of every part grades
wrong and carries a catalogue id, text is clean and deterministic, and exam chains follow through."""

from __future__ import annotations

from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE  # noqa: F401
from mathgen.text_quality import find_text_defects, prompt_shape

TIERS = ("easy", "medium", "hard")


def skill_of(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def single(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def check_wiring(skill, topic: str, exam_topic: str, answer: str) -> None:
    assert skill.answer == answer
    assert [(s.topic, s.difficulty) for s in skill.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam_topic, "exam"),
    ]
    assert skill.sources[-1].stages == ["exam"]


def check_single_tier(skill, topic: str, tier: str, seeds: int = 80) -> None:
    for seed in range(seeds):
        q = single(topic, tier, seed)
        assert not q.parts
        assert grade(skill, q, q.answer_text).correct, (tier, seed, q.answer_text)
        assert len(q.distractors) >= 2, (tier, seed)
        assert len(set(q.distractors)) == len(q.distractors)
        for text in q.distractors:
            assert not grade(skill, q, text).correct, (tier, seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE, (tier, seed, text)
            assert text in q.distractor_values
        assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))
        assert q.to_json() == single(topic, tier, seed).to_json()


def check_exam_chain(skill, exam_topic: str, marks: list[int], kinds: list[str], seeds: int = 60) -> None:
    for seed in range(seeds):
        q = exam(exam_topic, seed)
        assert [p.label for p in q.parts] == [chr(97 + i) for i in range(len(marks))]
        assert [p.marks for p in q.parts] == marks
        assert [p.answer_kind for p in q.parts] == kinds
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == exam(exam_topic, seed).to_json()
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, full)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE, (seed, p.label, text)


def check_follow_through(skill, exam_topic: str, label: str, wrong_for, seeds: int = 40) -> int:
    """Re-grade part `label` after replacing earlier answers by `wrong_for(question, canonical replies)` (a dict of
    replacements); the later part must still earn its marks as follow-through. Returns how many cases were checked."""
    checked = 0
    for seed in range(seeds):
        q = exam(exam_topic, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        replace = wrong_for(q, canon)
        if replace is None:
            continue
        result = grade(skill, q, {**canon, **replace})
        part = next(p for p in q.parts if p.label == label)
        assert result.parts[label].marks == part.marks, (seed, replace, result.parts[label])
        assert result.parts[label].follow_through is True, (seed, replace)
        checked += 1
    return checked


def distinct_shapes(topic: str, tier: str, seeds: int = 60) -> set[str]:
    return {prompt_shape(make(topic, seed=s, difficulty=tier)) for s in range(seeds)}


def check_pools(topic: str, tiers=TIERS, min_prompts: int = 20, seeds: int = 60) -> None:
    pools = {}
    for tier in tiers:
        qs = [single(topic, tier, s) for s in range(seeds)]
        assert len({q.text for q in qs}) >= min_prompts, tier
        pools[tier] = {q.text for q in qs}
        assert len(distinct_shapes(topic, tier, seeds)) >= 3, tier
    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert pools[a] != pools[b]
    easy, hard = distinct_shapes(topic, "easy", seeds), distinct_shapes(topic, "hard", seeds)
    assert hard - easy, "the hard tier adds no prompt shape over the easy tier"


def answer_spread(topic: str, tier: str, seeds: int = 80) -> Counter:
    return Counter(single(topic, tier, s).answer_text for s in range(seeds))
