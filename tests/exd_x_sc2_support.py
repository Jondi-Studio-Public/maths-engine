"""Shared checks for the `test_exd_x_sc2_*` files: variety, wiring, grading and follow-through of multi-part exam tiers.

Nothing here imports a generator module. Each test file supplies an oracle that re-derives every part's answer from the
RENDERED stem and part prompts alone, so the keys are checked against what a learner actually reads.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def text_of(q) -> str:
    """The stem and every part prompt, as the learner reads them."""
    return " ".join([q.stem, *(p.prompt for p in q.parts)])


def decimal_text(value: Fraction, places: int) -> str:
    """`value` rounded half up to `places` decimal places and printed with exactly that many."""
    quantum = Decimal(1).scaleb(-places)
    exact = Decimal(value.numerator) / Decimal(value.denominator)
    return str(exact.quantize(quantum, rounding=ROUND_HALF_UP))


def check_no_text_defects(topic: str, seeds: int = 60) -> None:
    for seed in range(seeds):
        q = exam(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)


def check_variety(topic: str, hard_maker, seeds: int = 60) -> None:
    """At least three prompt shapes, and one that the skill's hard tier never shows."""
    exam_shapes = {prompt_shape(exam(topic, s)) for s in range(seeds)}
    hard_shapes = {prompt_shape(hard_maker(s)) for s in range(seeds)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes, (exam_shapes, hard_shapes)


def check_wired(skill_id: str, topic: str) -> None:
    sources = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert [s.topic for s in sources] == [topic]
    assert list(sources[0].stages) == ["exam"]


def check_oracle(topic: str, oracle, seeds: int = 300) -> None:
    bad = []
    for seed in range(seeds):
        q = exam(topic, seed)
        want = oracle(q)
        got = {p.label: p.answer_text for p in q.parts}
        if want != got:
            bad.append((seed, want, got))
    assert bad == [], bad[:3]


def check_answer_space(topic: str, seeds: int = 300, minimum: int = 100) -> None:
    answers = [exam(topic, s).answer_text for s in range(seeds)]
    assert len(set(answers)) >= minimum, len(set(answers))
    for label in "abc":
        values = [p.answer_text for s in range(seeds) for p in exam(topic, s).parts if p.label == label]
        top = max(values.count(v) for v in set(values))
        assert top <= 0.25 * len(values), (label, top)


def check_deterministic(topic: str) -> None:
    for seed in (0, 1, 7, 123):
        assert exam(topic, seed).to_json() == exam(topic, seed).to_json()


def check_key_accepted_distractors_rejected(skill_id: str, topic: str, seeds: int = 120) -> None:
    sk = skill(skill_id)
    for seed in range(seeds):
        q = exam(topic, seed)
        result = grade(sk, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(p.marks for p in q.parts), (seed, q.text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label, p.distractors)
            for d in p.distractors:
                reply = {**canon, p.label: d}
                part_result = grade(sk, q, reply).parts[p.label]
                assert part_result.marks == 0, (seed, p.label, d, p.answer_text)


def check_follow_through(skill_id: str, topic: str, wrong_chain, seeds: int = 120, minimum: int = 40) -> None:
    """`wrong_chain(q)` returns a reply to every part with a wrong answer to (a) and every later part worked out
    correctly from the learner's own earlier answers, or None when no such reply can be typed exactly (for example
    when the downstream value is not a whole number but the part is graded as a whole number). A seed whose replies
    repeat a number is skipped, because a reply that only repeats an earlier answer earns no follow-through.

    Part (a) earns nothing, every later part earns full marks, and every later part whose reply differs from its
    canonical answer is marked as follow-through."""
    sk = skill(skill_id)
    checked = 0
    for seed in range(seeds):
        q = exam(topic, seed)
        replies = wrong_chain(q)
        if replies is None or len({str(Fraction(v)) for v in replies.values()}) < len(replies):
            continue
        result = grade(sk, q, replies)
        assert result.parts["a"].marks == 0, (seed, replies)
        for part in q.parts[1:]:
            got = result.parts[part.label]
            assert got.marks == part.marks, (seed, part.label, replies, got.marks)
            if Fraction(replies[part.label]) != Fraction(part.answer_text):
                assert got.follow_through is True, (seed, part.label)
        assert not result.correct
        checked += 1
    assert checked >= minimum, checked


def check_all_wrong_earns_nothing(skill_id: str, topic: str, junk: dict, seeds: int = 40) -> None:
    sk = skill(skill_id)
    for seed in range(seeds):
        q = exam(topic, seed)
        result = grade(sk, q, junk)
        assert result.marks == 0 and not result.correct, (seed, junk)
