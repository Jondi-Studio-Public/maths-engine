"""Shared assertions for the `test_sr2b_*` skill test files (not a test module itself)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("easy", "medium", "hard")


def skill_yaml(skill_id: str) -> dict:
    (path,) = list((ROOT / "skills").rglob(f"{skill_id}.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def skill_object(skill_id: str):
    _, skills = load_skills()
    return next(s for s in skills if s.id == skill_id)


def tier_question(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def exam_question(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def assert_wired(skill_id: str, topic: str, exam_topic: str, answer: str, prerequisites: list[str]) -> None:
    data = skill_yaml(skill_id)
    assert data["answer"] == answer
    assert data["prerequisites"] == prerequisites
    by_difficulty = {s["difficulty"]: s for s in data["sources"]}
    for tier in TIERS:
        assert by_difficulty[tier]["topic"] == topic
        assert not by_difficulty[tier]["params"]
    assert by_difficulty["exam"]["topic"] == exam_topic
    assert by_difficulty["exam"]["stages"] == ["exam"]
    assert not by_difficulty["exam"]["params"]
    sk = skill_object(skill_id)
    assert sk.status == "active"


def assert_clean_and_deterministic(topic: str, tier: str, seeds=range(40)) -> None:
    for seed in seeds:
        q = tier_question(topic, tier, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == tier_question(topic, tier, seed).to_json()


def assert_exam_clean_and_deterministic(topic: str, seeds=range(40)) -> None:
    for seed in seeds:
        q = exam_question(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam_question(topic, seed).to_json()


def assert_tier_shapes(topic: str, exam_topic: str, minimum_hard=2) -> None:
    """The tiers differ in the kind of prompt they set and the exam adds a shape beyond the hard tier."""
    shapes = {t: {prompt_shape(tier_question(topic, t, s)) for s in range(60)} for t in TIERS}
    assert len(shapes["hard"]) >= minimum_hard, shapes["hard"]
    assert shapes["hard"] - shapes["easy"], "hard sets no prompt shape that easy does not"
    exam = {prompt_shape(exam_question(exam_topic, s)) for s in range(60)}
    assert exam - shapes["hard"], "the exam tier adds no prompt shape beyond the hard tier"


def assert_answer_space(topic: str, tier: str, seeds=range(200), distinct=12, top=0.25) -> None:
    counts = Counter(tier_question(topic, tier, s).answer_text for s in seeds)
    assert len(counts) >= distinct, (tier, len(counts))
    value, n = counts.most_common(1)[0]
    assert n / len(seeds) <= top, (tier, value, n)


def assert_exam_answer_space(topic: str, seeds=range(200), distinct=60, top=0.05) -> None:
    counts = Counter(tuple(p.answer_text for p in exam_question(topic, s).parts) for s in seeds)
    assert len(counts) >= distinct, len(counts)
    value, n = counts.most_common(1)[0]
    assert n / len(seeds) <= top, (value, n)


def assert_single_graded(skill_id: str, topic: str, tier: str, seeds=range(60)) -> None:
    """The key is graded correct, every distractor is graded wrong, and each carries a catalogue tag."""
    sk = skill_object(skill_id)
    for seed in seeds:
        q = tier_question(topic, tier, seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, (seed, q.distractors)
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not grade(sk, q, d).correct, (seed, q.text, q.answer_text, d)
            assert q.option_notes.get(d) in CATALOGUE, (seed, d, q.option_notes.get(d))


def assert_chain_shape(topic: str, kinds, marks, follow_through) -> None:
    for seed in range(40):
        q = exam_question(topic, seed)
        assert [p.label for p in q.parts] == list("abcd"[: len(kinds)])
        assert [p.answer_kind for p in q.parts] == list(kinds)
        assert [p.marks for p in q.parts] == list(marks)
        assert [bool(p.follow_through) for p in q.parts] == list(follow_through)
        for p in q.parts:
            if p.answer_kind == "number":
                assert p.sensible_range is not None, (seed, p.label)


def assert_exam_graded(skill_id: str, topic: str, seeds=range(60)) -> None:
    """The composed key earns every mark, every distractor of every part earns nothing for its part, and every
    distractor is tagged."""
    sk = skill_object(skill_id)
    for seed in seeds:
        q = exam_question(topic, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        full = grade(sk, q, canon)
        assert full.correct and full.marks == full.max_marks, (seed, q.answer_text)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (seed, p.label, d)
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d, p.answer_text)
        junk = grade(sk, q, {p.label: f"{7919 * (i + 1)}.5" for i, p in enumerate(q.parts)})
        assert junk.marks == 0, (seed, junk.marks)


def part_marks(skill_id: str, q, replies: dict[str, str], label: str) -> int:
    """The marks the reply `replies[label]` earns for its part when the other replies are as given."""
    return grade(skill_object(skill_id), q, replies).parts[label].marks
