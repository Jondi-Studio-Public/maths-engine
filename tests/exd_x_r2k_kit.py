"""Shared checks for the `test_exd_x_r2k_*` files: wiring, prompt shapes, answer space and part-by-part grading.

Nothing here imports a generator module. Each test file re-derives its own answers from the rendered prompt.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
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


def half_up(value, places: int) -> Fraction:
    """`value` (a Fraction, int or float) rounded half up to `places` decimal places, exactly."""
    exact = Fraction(repr(value)) if isinstance(value, float) else Fraction(value)
    scale = 10**places
    return Fraction(int(abs(exact) * scale + Fraction(1, 2)) * (1 if exact >= 0 else -1), scale)


def fixed(value, places: int) -> str:
    """`value` rounded half up and written with exactly `places` places."""
    v = half_up(value, places)
    if places == 0:
        return str(int(v))
    sign = "-" if v < 0 else ""
    v = abs(v)
    whole, frac = divmod(int(v * 10**places), 10**places)
    return f"{sign}{whole}.{frac:0{places}d}"


def frac_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def check_wiring(skill_id: str, topic: str) -> None:
    exam = [s for s in skill(skill_id).sources if s.difficulty == "exam"][:1]  # a second exam source may follow
    assert [s.topic for s in exam] == [topic]
    assert [list(s.stages) for s in exam] == [["exam"]]
    assert [s.params for s in exam] == [{"tier": "exam"}]


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


def check_grading(skill_id: str, n: int = 120, min_parts: int = 3) -> None:
    """The composed key earns full marks; every distractor of every part earns that part nothing; every tagged
    distractor carries a catalogue id; every part has at least one distractor."""
    sk = skill(skill_id)
    for seed in range(n):
        q = exam_q(skill_id, seed)
        assert len(q.parts) >= min_parts, q.text
        result = grade(sk, q, canon(q))
        assert result.correct and result.marks == result.max_marks, (seed, q.text)
        assert result.marks == sum(p.marks for p in q.parts)
        assert len({p.answer_kind for p in q.parts}) >= 1
        for p in q.parts:
            assert p.distractors, (seed, p.label)
            for d in p.distractors:
                tag = p.option_notes.get(d)
                assert tag is None or tag in CATALOGUE, (seed, p.label, d, tag)
                reply = {**canon(q), p.label: d}
                assert grade(sk, q, reply).parts[p.label].marks == 0, (seed, p.label, d, q.text)
        assert not grade(sk, q, {p.label: "999999" for p in q.parts}).correct


def check_tags(skill_id: str, n: int = 120) -> None:
    """Every number, fraction and list part with two or more wrong answers tags most of them."""
    tagged = total = 0
    for seed in range(n):
        for p in exam_q(skill_id, seed).parts:
            for d in p.distractors:
                total += 1
                tagged += d in p.option_notes
    assert total and tagged / total >= 0.9, (tagged, total)


def check_determinism(skill_id: str) -> None:
    for seed in (0, 1, 7, 99):
        assert exam_q(skill_id, seed).to_json() == exam_q(skill_id, seed).to_json()


def check_follow_through(skill_id: str, wrong_chain, n: int = 100) -> None:
    """`wrong_chain(q)` returns (label of the wrong part, {label: reply}) for a chain that starts with a wrong
    answer and follows through correctly. The wrong part earns nothing; every later part earns its marks."""
    sk = skill(skill_id)
    for seed in range(n):
        q = exam_q(skill_id, seed)
        label, replies = wrong_chain(q)
        result = grade(sk, q, replies)
        assert result.parts[label].marks == 0, (seed, label, replies)
        later = [p for p in q.parts if p.label > label and p.follow_through]
        assert later, (seed, label)
        for p in later:
            assert result.parts[p.label].marks == p.marks, (seed, p.label, replies, q.text)
            assert result.parts[p.label].follow_through, (seed, p.label)
        assert not result.correct
