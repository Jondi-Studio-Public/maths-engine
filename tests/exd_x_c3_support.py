"""Shared assertions for the `exd_x_c3_*` exam-tier test files (not a test module itself)."""

from __future__ import annotations

from pathlib import Path

import yaml

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parent.parent


def skill(skill_id: str) -> dict:
    (path,) = list((ROOT / "skills").rglob(f"{skill_id}.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def exam_question(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam", tier="exam")


def hard_question(skill_id: str, seed: int):
    hard = [s for s in skill(skill_id)["sources"] if s["difficulty"] == "hard"][0]
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (hard.get("params") or {}).items()}
    return make(hard["topic"], seed=seed, difficulty="hard", **params)


def assert_registered_as_exam_source(skill_id: str, topic: str) -> None:
    last = skill(skill_id)["sources"][-1]
    assert last["topic"] == topic
    assert last["difficulty"] == "exam" and last["stages"] == ["exam"]
    assert last["params"] == {"tier": "exam"}


def assert_clean_text_and_deterministic(topic: str) -> None:
    for seed in range(60):
        q = exam_question(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam_question(topic, seed).to_json()


def assert_shapes(skill_id: str, topic: str) -> None:
    exam = {prompt_shape(exam_question(topic, s)) for s in range(60)}
    hard = {prompt_shape(hard_question(skill_id, s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam tier adds no prompt shape beyond the hard tier"


def assert_grader(skill_id: str, topic: str, seeds=range(120)) -> None:
    kind = skill(skill_id)["answer"]
    grader = GRADERS[kind]
    bad = []
    for seed in seeds:
        q = exam_question(topic, seed)
        if not grader(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        assert len(q.distractors) == 3, (seed, q.distractors)
        for d in q.distractors:
            if grader(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]
