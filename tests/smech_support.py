"""Shared checks for the senior mechanics and residual skills (`tests/test_smech_*.py`).

Every skill in the batch has four tiers (easy, medium, hard, exam). The helpers here grade through
`mathgen.engine.grade`, never a grader directly, and check the same things for each skill: the canonical answer is
accepted, every distractor of every part is rejected, an exam chain follows an earlier wrong answer through, and the
tiers differ in the shape of the question.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills

TIERS = ("easy", "medium", "hard")


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def sources(sk) -> dict[str, str]:
    """difficulty -> topic for the skill's sources."""
    return {s.difficulty: s.topic for s in sk.sources}


def questions(sk, difficulty: str, seeds):
    topic = sources(sk)[difficulty]
    return [make(topic, seed=s, difficulty=difficulty) for s in seeds]


def frac(text: str) -> Fraction:
    """The exact number a printed answer or stem figure states."""
    return Fraction(text.replace("−", "-").replace(",", "").strip())


def assert_single_answer_rules(sk, difficulty: str, seeds, *, minimum_distractors: int = 2):
    """The canonical answer grades correct and every distractor grades wrong, through the engine."""
    for q in questions(sk, difficulty, seeds):
        assert not q.parts
        assert grade(sk, q, q.answer_text).correct, (q.text, q.answer_text)
        assert len(q.distractors) >= minimum_distractors, q.text
        assert len(set(q.distractors)) == len(q.distractors) and q.answer_text not in q.distractors, q.text
        for d in q.distractors:
            assert not grade(sk, q, d).correct, (q.text, q.answer_text, d)


def decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def assert_number_distractors_match_places(q_or_part):
    """Same number of decimal places as the answer, so no option is told apart by its length."""
    places = decimals(q_or_part.answer_text)
    for d in q_or_part.distractors:
        assert decimals(d) == places, (q_or_part.answer_text, d)


def assert_exam_rules(sk, seeds, *, labels: str, marks: list[int]):
    """Full marks for the composed answer, zero for a distractor in its own part, and follow-through intact."""
    topic = sources(sk)["exam"]
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == list(labels)
        assert [p.marks for p in q.parts] == marks
        assert q.parts[0].follow_through is None and any(p.follow_through for p in q.parts[1:])
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), q.text
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def fixed(value, places: int) -> str:
    """`value` (a Fraction or float) rounded half-up and shown at exactly `places` decimal places."""
    from decimal import ROUND_HALF_UP, Decimal, getcontext

    getcontext().prec = 40
    d = Decimal(value.numerator) / Decimal(value.denominator) if isinstance(value, Fraction) else Decimal(repr(value))
    return str(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
