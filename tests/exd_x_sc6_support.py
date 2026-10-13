"""Shared checks for the `test_exd_x_sc6_*` files: exam-tier chains graded part by part.

Nothing here imports a generator module. Each test file re-derives its keys from the rendered prompt.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade, load_skills, make_for_skill
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILLS = {s.id: s for s in load_skills()[1]}


def skill_of(skill_id: str):
    return _SKILLS[skill_id]


def exam(skill, seed: int):
    return make_for_skill(skill, seed, source="exam")


def frac(text: str) -> Fraction:
    return Fraction(text.strip().replace(" ", ""))


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def assert_source_wired(skill, topic: str) -> None:
    exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam_sources] == [topic]
    assert exam_sources[0].stages == ["exam"]


def assert_new_shapes(skill, n: int = 60) -> None:
    exam_shapes = {prompt_shape(exam(skill, s)) for s in range(n)}
    hardest = [src for src in skill.sources if src.difficulty != "exam"][-1]
    hard_shapes = {prompt_shape(make_for_skill(skill, s, source=hardest)) for s in range(n)}
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard_shapes


def assert_no_text_defects(skill, n: int = 60) -> None:
    for seed in range(n):
        assert find_text_defects(exam(skill, seed)) == [], seed


def assert_answer_space(skill, n: int = 300, minimum: int = 100, ceiling: float = 0.05) -> None:
    answers = [exam(skill, s).answer_text for s in range(n)]
    assert len(set(answers)) >= minimum
    assert max(answers.count(a) for a in set(answers)) / n <= ceiling


def assert_grading(skill, seeds=range(120)) -> None:
    """The composed key earns full marks; every distractor of every part earns nothing for that part."""
    tagged = 0
    for seed in seeds:
        q = exam(skill, seed)
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (seed, q.answer_text)
        key = canonical(q)
        for part in q.parts:
            assert len(part.distractors) >= 2, (seed, part.label)
            for d in part.distractors:
                result = grade(skill, q, {**key, part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)
            tagged += len(part.option_notes)
    assert tagged >= len(list(seeds))


def assert_follow_through(skill, q, replies: dict[str, str], expect_zero: str) -> None:
    """`replies` is a full set of replies whose later parts follow correctly from the wrong `expect_zero` part."""
    result = grade(skill, q, replies)
    assert result.parts[expect_zero].marks == 0, (q.stem, replies)
    for label, part in result.parts.items():
        if label != expect_zero:
            assert part.marks == part.max_marks, (q.stem, replies, label)
    assert not result.correct


def rhs_free(prompt: str, key: str) -> bool:
    """True when the key does not appear as a standalone number in the prompt text."""
    return re.search(rf"(?<![\d.]){re.escape(key)}(?![\d.])", prompt) is None
