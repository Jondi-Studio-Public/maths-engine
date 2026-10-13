"""Shared helpers for the `test_exd_x_u4_*` files: checks that every multi-part exam tier here must pass.

Nothing here imports a generator module. Each test file re-derives its own key from the rendered prompt; these
helpers only cover what is the same for all of them (prompt variety, full marks, distractor rejection, answer space).
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def canonical_reply(q) -> dict:
    return {p.label: p.answer_text for p in q.parts}


def shapes(topic: str, n: int = 60, **kwargs) -> set[str]:
    return {prompt_shape(make(topic, seed=s, **kwargs)) for s in range(n)}


def check_variety(topic: str, hard_topic: str, hard_params: dict | None = None, n: int = 60) -> None:
    exam_shapes = {prompt_shape(exam(topic, s)) for s in range(n)}
    hard_shapes = shapes(hard_topic, n, difficulty="hard", **(hard_params or {}))
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes


def check_no_text_defects(topic: str, n: int = 60) -> None:
    for seed in range(n):
        assert find_text_defects(exam(topic, seed)) == []


def check_full_marks(skill_id: str, topic: str, n: int = 120) -> None:
    sk = skill(skill_id)
    for seed in range(n):
        q = exam(topic, seed)
        result = grade(sk, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (seed, q.answer_text)
        assert grade(sk, q, canonical_reply(q)).correct


def check_distractors_rejected(skill_id: str, topic: str, n: int = 120, minimum: int = 300) -> None:
    sk = skill(skill_id)
    shown = 0
    for seed in range(n):
        q = exam(topic, seed)
        canon = canonical_reply(q)
        for p in q.parts:
            assert p.distractors, (seed, p.label)
            for d in p.distractors:
                shown += 1
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= minimum, shown


def check_tagged(topic: str, n: int = 60) -> None:
    """Every part offers at least one distractor tagged with a catalogue id and a value, and most parts two."""
    two_or_more = total = 0
    for seed in range(n):
        for p in exam(topic, seed).parts:
            tagged = [d for d in p.distractors if d in p.option_notes and d in p.distractor_values]
            assert tagged, (seed, p.label, p.distractors, p.option_notes)
            total += 1
            two_or_more += len(tagged) >= 2
    assert two_or_more >= 0.9 * total, (two_or_more, total)


def answer_space(topic: str, n: int = 300) -> Counter:
    return Counter(exam(topic, s).answer_text for s in range(n))


def check_answer_space(topic: str, n: int = 300) -> None:
    counts = answer_space(topic, n)
    assert len(counts) >= 100, len(counts)
    assert max(counts.values()) <= 0.05 * n, counts.most_common(3)


def frac(text: str) -> Fraction:
    return Fraction(text.strip())
