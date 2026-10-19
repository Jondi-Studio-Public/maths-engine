"""Shared assertions for the `exd_x_r3e_*` multi-part exam-tier test files (not a test module itself)."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def skill_yaml(skill_id: str) -> dict:
    (path,) = list((ROOT / "skills").rglob(f"{skill_id}.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def skill_object(skill_id: str):
    _, skills = load_skills()
    return next(s for s in skills if s.id == skill_id)


def exam_question(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam", tier="exam")


def hard_question(skill_id: str, seed: int):
    """A question from the hardest non-exam tier the skill has (its hard tier, or the highest tier below it)."""
    sources = [s for s in skill_yaml(skill_id)["sources"] if s["difficulty"] != "exam"]
    order = {"easy": 0, "medium": 1, "hard": 2}
    hard = max(sources, key=lambda s: order.get(s["difficulty"], 0))
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (hard.get("params") or {}).items()}
    return make(hard["topic"], seed=seed, difficulty=hard["difficulty"], **params)


def assert_registered(skill_id: str, topic: str) -> None:
    exam = [s for s in skill_yaml(skill_id)["sources"] if s["difficulty"] == "exam" and s.get("key") != "exam-single"]
    assert len(exam) == 1
    assert exam[0]["topic"] == topic
    assert exam[0]["stages"] == ["exam"]
    assert exam[0]["params"] == {"tier": "exam"}


def assert_clean_and_deterministic(topic: str, seeds=range(60)) -> None:
    for seed in seeds:
        q = exam_question(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam_question(topic, seed).to_json()


def assert_shapes(skill_id: str, topic: str) -> None:
    exam = {prompt_shape(exam_question(topic, s)) for s in range(60)}
    hard = {prompt_shape(hard_question(skill_id, s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam tier adds no prompt shape beyond the hard tier"


def assert_answer_space(topic: str, *, seeds=300, minimum=100, top_share=0.05) -> None:
    tuples = Counter(tuple(p.answer_text for p in exam_question(topic, s).parts) for s in range(seeds))
    assert len(tuples) >= minimum, len(tuples)
    assert max(tuples.values()) / seeds <= top_share, tuples.most_common(3)


def assert_chain_shape(topic: str, labels: str, kinds: list[str], marks: list[int]) -> None:
    for seed in range(40):
        q = exam_question(topic, seed)
        assert [p.label for p in q.parts] == list(labels)
        assert [p.answer_kind for p in q.parts] == kinds
        assert [p.marks for p in q.parts] == marks
        assert q.parts[0].follow_through is None
        assert all(p.follow_through for p in q.parts[1:] if p.follow_through is not None)
        assert sum(1 for p in q.parts if p.follow_through) >= 1


def reply(q, **overrides) -> dict:
    out = {p.label: p.answer_text for p in q.parts}
    out.update(overrides)
    return out


def assert_full_marks_and_distractors(skill_id: str, topic: str, seeds=range(100)) -> None:
    """The key earns full marks; every distractor of every part earns nothing for that part (the other parts
    answered correctly); every distractor is tagged with a catalogue id and carries its value."""
    skill = skill_object(skill_id)
    for seed in seeds:
        q = exam_question(topic, seed)
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (seed, q.answer_text)
        for part in q.parts:
            assert len(part.distractors) >= 2, (seed, part.label, part.distractors)
            for d in part.distractors:
                assert part.option_notes.get(d) in CATALOGUE, (seed, part.label, d, part.option_notes)
                assert d in part.distractor_values, (seed, part.label, d)
                got = grade(skill, q, reply(q, **{part.label: d}))
                assert got.parts[part.label].marks == 0, (seed, part.label, d, part.answer_text)
                assert got.parts[part.label].misconception == part.option_notes[d], (seed, part.label, d)
                for other in q.parts:
                    if other.label != part.label:
                        assert got.parts[other.label].marks == other.marks, (seed, part.label, d, other.label)


def assert_text_has_no_working_words(topic: str) -> None:
    for seed in range(60):
        q = exam_question(topic, seed)
        low = q.text.lower()
        assert "exam-style" not in low and "show working" not in low and "show your working" not in low, q.text
