"""Shared checks for the `test_exd_x_r2c_*` files: multi-part exam chains.

Nothing here imports a generator module. A test file passes its topic and skill id and supplies its own
independent re-derivation of every part's answer from the rendered question.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape


@lru_cache(maxsize=None)
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


@lru_cache(maxsize=None)
def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def check_wired(skill_id: str, topic: str) -> None:
    sources = [s for s in skill(skill_id).sources if s.difficulty == "exam" and s.effective_key == "exam"]
    assert [s.topic for s in sources] == [topic]
    assert list(sources[0].stages) == ["exam"]


def check_shapes(topic: str, hard_topic: str, hard_kwargs: dict | None = None, n: int = 60) -> None:
    shapes = {prompt_shape(exam(topic, s)) for s in range(n)}
    assert len(shapes) >= 3, shapes
    hard = {prompt_shape(make(hard_topic, seed=s, difficulty="hard", **(hard_kwargs or {}))) for s in range(n)}
    assert shapes - hard, (shapes, hard)


def check_clean_and_deterministic(topic: str, n: int = 60) -> None:
    for seed in range(n):
        q = exam(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()
        assert len(q.parts) in (3, 4)
        assert q.parts[0].follow_through is None
        assert any(p.follow_through for p in q.parts[1:])


def check_answer_space(topic: str, n: int = 300, distinct: int = 100, share: float = 0.05) -> None:
    tuples = Counter(tuple(sorted(canonical(exam(topic, s)).items())) for s in range(n))
    assert len(tuples) >= distinct, len(tuples)
    assert max(tuples.values()) / n <= share, tuples.most_common(3)


def check_grading(skill_id: str, topic: str, n: int = 120) -> None:
    """The composed answer earns full marks; every distractor, with the other parts right, earns nothing."""
    sk = skill(skill_id)
    for seed in range(n):
        q = exam(topic, seed)
        result = grade(sk, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (seed, q.answer_text)
        canon = canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label, p.distractors)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (seed, p.label, d)
                graded = grade(sk, q, {**canon, p.label: d})
                assert graded.parts[p.label].marks == 0, (seed, p.label, d, p.answer_text)


def check_follow_through(skill_id: str, topic: str, carried, n: int = 60) -> None:
    """`carried(q)` returns (label, wrong reply, {later label: reply that is right for that wrong reply}).

    The wrong part earns nothing and every later part named in the dict earns its full marks, at least one of them
    as follow-through.
    """
    sk = skill(skill_id)
    for seed in range(n):
        q = exam(topic, seed)
        label, wrong, later = carried(q)
        replies = {**canonical(q), label: wrong, **later}
        result = grade(sk, q, replies)
        assert result.parts[label].marks == 0, (seed, label, wrong)
        by_label = {p.label: p for p in q.parts}
        for other, reply in later.items():
            assert result.parts[other].marks == by_label[other].marks, (seed, other, reply, q.answer_text)
        # A later reply that happens to equal the canonical answer earns its marks without the flag.
        assert any(result.parts[other].follow_through for other in later), (seed, later)
        assert not result.correct
