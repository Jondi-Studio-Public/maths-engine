"""Shared checks for the `test_exd_x_r2j_*` files: wiring, prompt shapes, answer space and part-by-part grading.

Nothing here imports a generator module. Each test file re-derives its own answers from the rendered prompt.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from mathgen.engine import grade, load_skills, make_for_skill
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape


@lru_cache(maxsize=None)
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def exam_q(skill_id: str, seed: int):
    return make_for_skill(skill(skill_id), seed, source="exam")


def hard_q(skill_id: str, seed: int):
    return make_for_skill(skill(skill_id), seed, source="hard")


def canon(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def check_wiring(skill_id: str, topic: str) -> None:
    exam = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [topic]
    assert [list(s.stages) for s in exam] == [["exam"]]


def check_shapes(skill_id: str, n: int = 60) -> None:
    shapes = {prompt_shape(exam_q(skill_id, s)) for s in range(n)}
    assert len(shapes) >= 3, shapes
    hard = {prompt_shape(hard_q(skill_id, s)) for s in range(n)}
    assert shapes - hard, "the exam tier adds no prompt shape over the hard tier"


def check_no_defects(skill_id: str, n: int = 60) -> None:
    for seed in range(n):
        q = exam_q(skill_id, seed)
        assert find_text_defects(q) == [], (seed, q.text)


def check_answer_space(skill_id: str, n: int = 300, minimum: int = 100, share: float = 0.05) -> None:
    tuples = Counter(exam_q(skill_id, s).answer_text for s in range(n))
    assert len(tuples) >= minimum, len(tuples)
    assert max(tuples.values()) / n <= share, tuples.most_common(3)


def check_grading(skill_id: str, n: int = 100) -> None:
    """The composed key earns full marks; every distractor of every part earns that part nothing; every
    distractor carries a catalogue id."""
    sk = skill(skill_id)
    for seed in range(n):
        q = exam_q(skill_id, seed)
        assert len(q.parts) >= 3, q.text
        result = grade(sk, q, canon(q))
        assert result.correct and result.marks == result.max_marks, (seed, q.text)
        assert result.marks == sum(p.marks for p in q.parts)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                tag = p.option_notes.get(d)
                assert tag in CATALOGUE, (seed, p.label, d, tag)
                reply = {**canon(q), p.label: d}
                part_result = grade(sk, q, reply).parts[p.label]
                assert part_result.marks == 0, (seed, p.label, d, q.text)
        # one wrong earlier answer never earns a later part its marks when that later part is also wrong
        assert not grade(sk, q, {p.label: "0.123" for p in q.parts}).correct


def check_determinism(skill_id: str) -> None:
    for seed in (0, 1, 7, 99):
        assert exam_q(skill_id, seed).to_json() == exam_q(skill_id, seed).to_json()
