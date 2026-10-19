"""Shared checks for the `r3h_*` skills (specialist differential equations and Methods applied calculus)."""

from __future__ import annotations

import functools

import mathgen.generators  # noqa: F401  (registers every generator)
from mathgen import make
from mathgen.grading import grade
from mathgen.grading.parts import follow_through_target, reply_value

SEEDS = range(1, 9)
TIERS = ("easy", "medium", "hard")


@functools.lru_cache(maxsize=None)
def _one(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def questions(topic: str, tier: str, seeds=SEEDS):
    return [_one(topic, tier, s) for s in seeds]


def check_grading(q) -> None:
    """The canonical answer is marked right and every distractor wrong, part by part."""
    if q.parts:
        right = {p.label: p.answer_text for p in q.parts}
        result = grade(q.parts[0].answer_kind, q, right)
        assert result.correct, (q.seed, right, result)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.seed, p.label)
            for d in p.distractors:
                wrong = dict(right, **{p.label: str(d)})
                assert not grade(q.parts[0].answer_kind, q, wrong).correct, (q.seed, p.label, str(d))
        return
    assert grade(q.answer_kind, q, q.answer_text).correct, (q.seed, q.answer_text)
    assert len(q.distractors) >= 2, q.seed
    for d in q.distractors:
        assert not grade(q.answer_kind, q, str(d)).correct, (q.seed, str(d), q.answer_text)


def check_follow_through(q) -> None:
    """Each follow-through part reproduces its own canonical answer from the earlier parts' answers."""
    values = {}
    for p in q.parts:
        if p.follow_through:
            target = follow_through_target(p, values)
            assert target is not None, (q.seed, p.label)
            assert grade(p.answer_kind, target, p.answer_text).correct or p.answer_kind == "number", (q.seed, p.label)
        values[p.label] = reply_value(p.answer_kind, p.answer_text)


def check_text(q) -> None:
    texts = [q.text] + [p.prompt for p in q.parts]
    for t in texts:
        assert t and t == t.strip()
        assert "  " not in t and "None" not in t and "{" not in t and "}" not in t, t
        assert "^1 " not in t and "(1t)" not in t, t
        assert t.count("(") == t.count(")"), t


def add_common_tests(namespace: dict, practice: str, exam: str) -> None:
    """Put the checks every `r3h` skill shares into a test module's namespace."""
    import pytest

    def topic_for(tier):
        return exam if tier == "exam" else practice

    @pytest.mark.parametrize("tier", (*TIERS, "exam"))
    def test_answers_and_distractors_grade(tier):
        for q in questions(topic_for(tier), tier):
            check_grading(q)
            check_text(q)

    @pytest.mark.parametrize("tier", (*TIERS, "exam"))
    def test_variety_and_determinism(tier):
        qs = questions(topic_for(tier), tier, range(1, 25))
        assert len({q.text for q in qs}) >= 8, len({q.text for q in qs})
        a = make(topic_for(tier), seed=7, difficulty=tier).to_dict()
        assert a == make(topic_for(tier), seed=7, difficulty=tier).to_dict()

    def test_exam_chain_shape():
        for q in questions(exam, "exam"):
            assert 3 <= len(q.parts) <= 4
            assert any(p.follow_through for p in q.parts)
            assert all(p.marks >= 1 for p in q.parts)
            check_follow_through(q)

    def test_tiers_are_distinct():
        shapes = {t: {q.text for q in questions(practice, t, range(1, 13))} for t in TIERS}
        assert not (shapes["easy"] & shapes["hard"])
        assert not (shapes["easy"] & shapes["medium"])
        assert not (shapes["medium"] & shapes["hard"])

    def test_single_part_tiers_have_no_parts():
        for t in TIERS:
            assert all(not q.parts for q in questions(practice, t, range(1, 6)))

    for fn in (
        test_answers_and_distractors_grade,
        test_variety_and_determinism,
        test_exam_chain_shape,
        test_tiers_are_distinct,
        test_single_part_tiers_have_no_parts,
    ):
        namespace[fn.__name__] = fn
