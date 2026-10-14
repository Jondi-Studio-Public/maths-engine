"""Shared checks for the `test_r4f_*` files: typed-step proof skills with single-answer tiers and an exam chain.

Nothing here imports a generator module. Each test file supplies an `oracle(question)` that re-derives the answer from
the rendered stem alone (by brute force, calculus or coordinates, never by the generator's own construction).
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


@lru_cache(maxsize=None)
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def parse(text: str, **names):
    """A learner-facing expression ("3a/2 - b", "2*3^(2*k + 1)") as a SymPy expression."""
    text = text.replace("×", "*").replace("·", "*").replace("−", "-")
    return parse_expr(text, local_dict=names, transformations=_TRANSFORMS)


def equal(left, right) -> bool:
    return sympy.simplify(sympy.sympify(left) - sympy.sympify(right)) == 0


def questions(topic: str, tier: str, n: int):
    return [make(topic, seed=s, difficulty=tier) for s in range(n)]


def check_wired(skill_id: str, topic: str, exam_topic: str) -> None:
    sources = skill(skill_id).sources
    assert [s.difficulty for s in sources] == ["easy", "medium", "hard", "exam"]
    assert [s.topic for s in sources] == [topic, topic, topic, exam_topic]
    assert list(sources[-1].stages) == ["exam"]
    assert sources[-1].params == {"tier": "exam"}


def check_single_clean(skill_id: str, topic: str, tier: str, n: int = 60) -> None:
    """The key grades correct, every distractor grades wrong and is tagged, no text defects, deterministic."""
    sk = skill(skill_id)
    for seed in range(n):
        q = make(topic, seed=seed, difficulty=tier)
        assert find_text_defects(q, include_working=True) == [], (tier, seed, q.text)
        assert grade(sk, q, q.answer_text).correct, (tier, seed, q.text, q.answer_text)
        assert len(q.distractors) >= 2, (tier, seed)
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors
        for d in q.distractors:
            assert not grade(sk, q, d).correct, (tier, seed, q.text, q.answer_text, d)
        assert set(q.option_notes) >= set(q.distractors), (tier, seed)
        again = make(topic, seed=seed, difficulty=tier)
        assert again.to_dict() == q.to_dict()


def check_tiers_differ(topic: str, n: int = 60) -> None:
    shapes = {
        t: {prompt_shape(make(topic, seed=s, difficulty=t)) for s in range(n)} for t in ("easy", "medium", "hard")
    }
    assert shapes["easy"] != shapes["medium"] and shapes["medium"] != shapes["hard"]
    assert not shapes["easy"] & shapes["medium"], "easy and medium share a prompt shape"
    assert not shapes["medium"] & shapes["hard"], "medium and hard share a prompt shape"


def check_answer_space_single(topic: str, tier: str, n: int = 200, floor: int = 12) -> None:
    answers = {make(topic, seed=s, difficulty=tier).answer_text for s in range(n)}
    assert len(answers) >= floor, (tier, len(answers))


def exam_questions(topic: str, n: int):
    return [make(topic, seed=s, difficulty="exam") for s in range(n)]


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def check_exam_clean(skill_id: str, topic: str, n: int = 40) -> None:
    """The composed answer earns every mark; each distractor of each part earns that part nothing; text is clean."""
    sk = skill(skill_id)
    for q in exam_questions(topic, n):
        assert find_text_defects(q, include_working=True) == [], q.text
        result = grade(sk, q, canonical(q))
        assert result.correct and result.marks == result.max_marks, (q.text, canonical(q))
        assert [p.label for p in q.parts] == list("abcd")
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            assert p.answer_text not in p.distractors
            assert len(set(p.distractors)) == len(p.distractors)
            assert set(p.option_notes) >= set(p.distractors), (q.text, p.label)
            for d in p.distractors:
                got = grade(sk, q, {**canonical(q), p.label: d})
                assert got.parts[p.label].marks == 0, (q.text, p.label, d)
        wrong = grade(sk, q, {p.label: "999999" for p in q.parts})
        assert wrong.marks == 0 and not wrong.correct
        assert make(topic, seed=3, difficulty="exam").to_dict() == make(topic, seed=3, difficulty="exam").to_dict()


def check_exam_oracle(topic: str, oracle, n: int = 120) -> None:
    bad = []
    for q in exam_questions(topic, n):
        expected = oracle(q)
        got = canonical(q)
        for label, value in expected.items():
            if not _same_answer(q, label, value, got[label]):
                bad.append((q.text, label, value, got[label]))
    assert bad == [], bad[:2]


def _same_answer(q, label, expected, given: str) -> bool:
    part = next(p for p in q.parts if p.label == label)
    if part.answer_kind == "expression":
        return equal(parse(given, a=sympy.Symbol("a"), b=sympy.Symbol("b")), expected)
    return Fraction(given) == Fraction(expected)


def check_exam_shape(topic: str, floor_distinct: int, n: int = 300, cap: float = 0.08) -> None:
    from collections import Counter

    tuples = Counter(tuple(canonical(q).values()) for q in exam_questions(topic, n))
    assert len(tuples) >= floor_distinct, len(tuples)
    assert max(tuples.values()) / n <= cap, tuples.most_common(3)
