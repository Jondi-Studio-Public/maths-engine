"""Shared assertions for the `test_exd_x_sc3_*` exam-tier test files (not a test module itself)."""

from __future__ import annotations

from fractions import Fraction
from functools import cache

from mathgen import make
from mathgen.core import round_half_up, stated_precision_string
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


@cache
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam", tier="exam")


def hard(skill_id: str, seed: int):
    source = next(s for s in skill(skill_id).sources if s.difficulty == "hard")
    return make(source.topic, seed=seed, difficulty="hard", **source.params)


def typed(value, places: int) -> str:
    """`value` rounded half up to `places` places and written the way a learner would type it."""
    return stated_precision_string(round_half_up(Fraction(value), places), places)


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def assert_registered(skill_id: str, topic: str) -> None:
    last = skill(skill_id).sources[-1]
    assert last.topic == topic
    assert last.difficulty == "exam" and list(last.stages) == ["exam"]
    assert last.params == {"tier": "exam"}


def assert_clean_and_deterministic(topic: str) -> None:
    for seed in range(60):
        q = exam(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam(topic, seed).to_json()


def assert_shapes(skill_id: str, topic: str) -> None:
    shapes = {prompt_shape(exam(topic, s)) for s in range(60)}
    hard_shapes = {prompt_shape(hard(skill_id, s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard_shapes, "the exam tier adds no prompt shape beyond the hard tier"


def assert_key_accepted_distractors_rejected(skill_id: str, topic: str, seeds=range(120)) -> None:
    sk = skill(skill_id)
    for seed in seeds:
        q = exam(topic, seed)
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (seed, q.answer_text)
        canon = canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label, p.distractors)
            assert all(p.option_notes.get(d) for d in p.distractors), (seed, p.label)
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d, p.answer_text)


def assert_answer_space(topic: str, minimum: int = 100, ceiling: float = 0.05, seeds: int = 300) -> None:
    from collections import Counter

    counts = Counter(exam(topic, s).answer_text for s in range(seeds))
    assert len(counts) >= minimum, len(counts)
    assert max(counts.values()) / seeds <= ceiling, counts.most_common(3)


def assert_follow_through(skill_id: str, q, wrong: dict[str, str], expect: dict[str, str]) -> None:
    """Reply `wrong` for the early parts and `expect` (their own work carried forward) for the later ones: the early
    parts earn nothing, every later part earns full marks and is flagged as follow-through."""
    sk = skill(skill_id)
    result = grade(sk, q, {**wrong, **expect})
    for label in wrong:
        assert result.parts[label].marks == 0, (q.text, label)
    for label in expect:
        part = result.parts[label]
        assert part.marks == part.max_marks, (q.text, label, expect[label])
        assert part.follow_through is True, (q.text, label)
    assert not result.correct
