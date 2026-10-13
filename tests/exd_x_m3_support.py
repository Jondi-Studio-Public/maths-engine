"""Shared checks for the `test_exd_x_m3_*` files.

Each test file supplies an oracle that re-derives every part's answer from the RENDERED stem and prompts, using
only the standard library and exact arithmetic, so nothing here imports a generator module.
"""

from __future__ import annotations

from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def exam_questions(topic: str, n: int):
    return [make(topic, seed=s, difficulty="exam") for s in range(n)]


def check_source_wired(skill_id: str, topic: str) -> None:
    exam = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert len(exam) == 1 and exam[0].topic == topic and list(exam[0].stages) == ["exam"]
    assert exam[0].params == {"tier": "exam"}


def check_shapes(skill_id: str, topic: str, n: int = 60) -> None:
    """At least three distinct prompt shapes, at least one of them absent from the skill's hard tier."""
    exam = {prompt_shape(q) for q in exam_questions(topic, n)}
    hard = {
        prompt_shape(make(src.topic, seed=s, **src.params))
        for src in skill(skill_id).sources
        if src.difficulty == "hard"
        for s in range(n)
    }
    assert len(exam) >= 3, exam
    assert exam - hard, (exam, hard)


def check_text_and_determinism(topic: str, n: int = 60) -> None:
    for seed in range(n):
        q = make(topic, seed=seed, difficulty="exam")
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()


def check_answer_space(topic: str, n: int = 300) -> None:
    """Enough distinct answers: some part takes at least 100 values, and the composed answer is never repeated often."""
    qs = exam_questions(topic, n)
    best = 0
    for label in [p.label for p in qs[0].parts]:
        counts = Counter(q.parts[ord(label) - ord("a")].answer_text for q in qs)
        best = max(best, len(counts))
    assert best >= 100, best
    whole = Counter(q.answer_text for q in qs)
    assert len(whole) >= 100 and max(whole.values()) <= 0.05 * n, whole.most_common(3)


def check_oracle(topic: str, oracle, n: int = 300) -> None:
    """`oracle(question) -> {label: answer text}` is computed from the rendered text alone."""
    bad = []
    for q in exam_questions(topic, n):
        want = oracle(q)
        got = {p.label: p.answer_text for p in q.parts}
        if want != got:
            bad.append((q.seed, q.text, want, got))
    assert bad == [], bad[:3]


def check_grading(skill_id: str, topic: str, n: int = 100) -> None:
    """The composed key earns full marks; every distractor of every part earns nothing for its part."""
    sk = skill(skill_id)
    for q in exam_questions(topic, n):
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (q.seed, q.answer_text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.seed, p.label)
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.seed, p.label, d)


def check_tagged(topic: str, n: int = 100) -> None:
    """Every part carries tagged distractors with values, and tags exist in the catalogue."""
    from mathgen.misconceptions import CATALOGUE

    tagged = 0
    for q in exam_questions(topic, n):
        for p in q.parts:
            assert set(p.option_notes.values()) <= set(CATALOGUE)
            assert set(p.option_notes) <= set(p.distractors)
            assert set(p.distractor_values) == set(p.distractors), (q.seed, p.label)
            tagged += len(p.option_notes)
    assert tagged > 0


def check_follow_through(skill_id: str, topic: str, wrong_first, downstream, ft_labels, n: int = 60) -> None:
    """A wrong (a) used correctly in later parts earns those parts' marks.

    `wrong_first(q)` is a wrong reply for (a); `downstream(q, reply_a)` gives the replies a learner who carries
    that reply forward would write for the other parts, worked out independently of the generator.
    """
    sk = skill(skill_id)
    for q in exam_questions(topic, n):
        a_reply = wrong_first(q)
        replies = {"a": a_reply, **downstream(q, a_reply)}
        result = grade(sk, q, replies)
        assert result.parts["a"].marks == 0, (q.seed, a_reply)
        for p in q.parts[1:]:
            part = result.parts[p.label]
            assert part.marks == part.max_marks, (q.seed, p.label, replies)
            if p.label in ft_labels:
                assert part.follow_through is True, (q.seed, p.label)
        assert not result.correct and 0 < result.marks < result.max_marks
    junk = grade(sk, q, {p.label: "9999" for p in q.parts})
    assert junk.marks == 0


def fmt(value, dp: int) -> str:
    """`value` (a Fraction, int or Decimal) rounded half up to `dp` places, at exactly that many places."""
    from decimal import ROUND_HALF_UP, Decimal, localcontext
    from fractions import Fraction

    value = Fraction(value)
    with localcontext() as ctx:
        ctx.prec = 80
        exact = Decimal(value.numerator) / Decimal(value.denominator)
        return str(exact.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))
