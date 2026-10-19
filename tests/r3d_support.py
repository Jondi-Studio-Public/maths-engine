"""Shared checks for the motion-by-calculus skills (`tests/test_r3d_*.py`).

Every skill in the batch has four tiers (easy, medium, hard, exam). The helpers grade through `mathgen.engine.grade`,
never a grader directly, and check the same things for each skill: the canonical answer is accepted, every distractor
of every part is rejected and tagged with a catalogue id, an exam chain follows an earlier wrong answer through, and
the generated text has no defects.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, getcontext
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TIERS = ("easy", "medium", "hard")


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def sources(sk) -> dict[str, str]:
    """difficulty -> topic for the skill's sources."""
    return {s.difficulty: s.topic for s in sk.sources}


def question(sk, difficulty: str, seed: int):
    return make(sources(sk)[difficulty], seed=seed, difficulty=difficulty)


def questions(sk, difficulty: str, seeds):
    return [question(sk, difficulty, s) for s in seeds]


def frac(text: str) -> Fraction:
    """The exact number a printed figure states."""
    return Fraction(text.replace("−", "-").replace(",", "").strip())


def decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def fixed(value, places: int) -> str:
    """`value` (a Fraction, int or float) rounded half-up and shown at exactly `places` decimal places."""
    getcontext().prec = 60
    if isinstance(value, Fraction):
        d = Decimal(value.numerator) / Decimal(value.denominator)
    else:
        d = Decimal(repr(value))
    return str(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def assert_single_answer_rules(sk, difficulty: str, seeds, *, minimum_distractors: int = 2):
    """The canonical answer grades correct and every distractor grades wrong, through the engine."""
    for q in questions(sk, difficulty, seeds):
        assert not q.parts
        assert grade(sk, q, q.answer_text).correct, (q.text, q.answer_text)
        assert len(q.distractors) >= minimum_distractors, q.text
        assert len(set(q.distractors)) == len(q.distractors) and q.answer_text not in q.distractors, q.text
        for d in q.distractors:
            assert not grade(sk, q, d).correct, (q.text, q.answer_text, d)
            assert q.option_notes[d] in CATALOGUE, (q.text, d)
        assert find_text_defects(q) == [], (q.text, find_text_defects(q))


def assert_number_distractors_match_places(q_or_part):
    """Same number of decimal places as the answer, so no option is told apart by its length."""
    if getattr(q_or_part, "answer_kind", "number") != "number":
        return
    places = decimals(q_or_part.answer_text)
    for d in q_or_part.distractors:
        assert decimals(d) == places, (q_or_part.answer_text, d)


def assert_exam_rules(sk, seeds, *, labels: str, marks: list[int] | list[list[int]]):
    """Full marks for the composed answer, zero for a distractor in its own part, and follow-through intact."""
    topic = sources(sk)["exam"]
    allowed = marks if isinstance(marks[0], list) else [marks]
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        assert [p.label for p in q.parts] == list(labels)
        assert [p.marks for p in q.parts] in allowed
        got = [p.marks for p in q.parts]
        assert q.parts[0].follow_through is None and any(p.follow_through for p in q.parts[1:])
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(got), q.text
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)
                assert p.option_notes[d] in CATALOGUE, (q.text, p.label, d)
        assert find_text_defects(q) == [], (q.text, find_text_defects(q))


def assert_deterministic_and_varied(topic: str, difficulty: str, *, seeds: int = 60, distinct: int = 30):
    assert make(topic, seed=5, difficulty=difficulty).to_dict() == make(topic, seed=5, difficulty=difficulty).to_dict()
    assert len({make(topic, seed=s, difficulty=difficulty).text for s in range(seeds)}) >= distinct


def pi_multiple(text: str) -> Fraction:
    """'7π/60' -> 7/60, 'π/12' -> 1/12, '2π' -> 2."""
    m = re.fullmatch(r"(-?\d*)π(?:/(\d+))?", text)
    num = m.group(1)
    num = 1 if num == "" else (-1 if num == "-" else int(num))
    return Fraction(num, int(m.group(2) or 1))


def assert_states(q_or_part, expected, context=""):
    """The printed answer is `expected` rounded to the answer's own number of decimal places (half-up)."""
    places = decimals(q_or_part.answer_text)
    got = float(frac(q_or_part.answer_text))
    assert abs(got - float(expected)) <= 0.5 * 10**-places + 1e-9, (context, q_or_part.answer_text, expected)


def assert_follow_through_credit(sk, topic: str, seeds, *, min_checked: int):
    """A later part worked correctly from an earlier part's catalogued wrong answer earns its marks, flagged as follow-through."""
    from mathgen.grading.parts import VALUE_KINDS, follow_through_target, reply_value

    checked = 0
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        canon = {p.label: p.answer_text for p in q.parts}
        for part in q.parts:
            if not part.follow_through:
                continue
            for earlier in q.parts[: q.parts.index(part)]:
                if earlier.answer_kind not in VALUE_KINDS:
                    continue
                for wrong in earlier.distractors:
                    replies = {**canon, earlier.label: wrong}
                    values = {
                        p.label: reply_value(p.answer_kind, replies[p.label])
                        for p in q.parts
                        if p.answer_kind in VALUE_KINDS
                    }
                    target = follow_through_target(part, values)
                    if target is None or target.answer_text in (part.answer_text, *part.distractors):
                        continue
                    carried = reply_value(part.answer_kind, target.answer_text)
                    if carried == 0:
                        continue  # a difference that works out to nothing is not marked as following through
                    if any(v is not None and v == carried for v in values.values()):
                        continue  # writing an earlier reply again is not following through (the engine says so)
                    result = grade(sk, q, {**replies, part.label: target.answer_text})
                    assert result.parts[part.label].marks == part.marks, (q.text, part.label, wrong, target.answer_text)
                    checked += 1
    assert checked >= min_checked, checked
