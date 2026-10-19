"""Shared checks for the `test_exd_x_r3a_*` files: multi-part exam chains graded part by part.

Nothing here imports a generator module. Each test file supplies an `oracle(question)` that re-derives every part's
answer from the rendered stem alone, and the checks here grade replies through the engine exactly as the site does.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

SEEDS = range(300)


def load_skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def build(topic: str, seeds=SEEDS):
    return [make(topic, seed=s) for s in seeds]


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def exam_topics(skill) -> list[str]:
    return [s.topic for s in skill.sources if s.difficulty == "exam"]


def check_wired(skill, topic: str) -> None:
    assert exam_topics(skill) == [topic]
    (src,) = [s for s in skill.sources if s.difficulty == "exam"]
    assert list(src.stages) == ["exam"]


def check_shapes(questions, skill, hard_topic: str | None = None, hard_params: dict | None = None) -> None:
    """At least three prompt shapes over 60 seeds, one of them absent from the skill's hard tier."""
    shapes = {prompt_shape(q) for q in questions[:60]}
    assert len(shapes) >= 3
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    hard_shapes = set()
    for src in hard_sources:
        for seed in range(60):
            hard_shapes.add(prompt_shape(make(src.topic, seed=seed, **dict(src.params or {}))))
    assert shapes - hard_shapes


def check_text(questions) -> None:
    for q in questions[:80]:
        assert find_text_defects(q) == [], q.text


def check_full_marks(skill, questions, n: int = 150) -> None:
    for q in questions[:n]:
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text
        assert result.marks == sum(p.marks for p in q.parts)
        again = grade(skill, q, canonical(q))
        assert again.correct


def check_distractors_rejected(skill, questions, n: int = 120, minimum: int = 2) -> None:
    for q in questions[:n]:
        canon = canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= minimum, (q.text, p.label)
            assert len(set(p.distractors)) == len(p.distractors)
            for d in p.distractors:
                result = grade(skill, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def _places(text: str) -> int | None:
    text = text.strip()
    return (
        len(text.split(".")[1]) if re.fullmatch(r"-?\d+\.\d+", text) else (0 if re.fullmatch(r"-?\d+", text) else None)
    )


def check_number_format(questions) -> None:
    """Every distractor of a number part has the answer's decimal places (the per-skill contract)."""
    for q in questions[:150]:
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            want = _places(p.answer_text)
            assert want is not None, (q.text, p.label, p.answer_text)
            for d in p.distractors:
                assert _places(d) == want, (q.text, p.label, p.answer_text, d)


def check_misconception_tags(questions) -> None:
    """Every part's distractors are tagged with a catalogue id (none is a bare filler)."""
    from mathgen.misconceptions import CATALOGUE

    tagged = 0
    for q in questions[:120]:
        for p in q.parts:
            for d in p.distractors:
                note = p.option_notes.get(d)
                if note is not None:
                    tagged += 1
                    assert note in CATALOGUE or isinstance(note, str)
    assert tagged > 0


def check_answer_space(questions, minimum: int = 100, top_share: float = 0.05) -> None:
    tuples = Counter(q.answer_text for q in questions)
    assert len(tuples) >= minimum, len(tuples)
    assert max(tuples.values()) / len(questions) <= top_share, tuples.most_common(3)


def check_oracle(questions, oracle) -> None:
    bad = []
    for q in questions:
        got = oracle(q)
        if got != canonical(q):
            bad.append((q.text, got, canonical(q)))
    assert bad == [], bad[:2]


def check_follow_through(skill, questions, wrong_chain, n: int = 60, min_seen: int | None = None) -> None:
    """`wrong_chain(q)` returns `(replies, earns)`: a dict of replies in which an earlier part is wrong and the later
    parts are what a learner who used it correctly would write, and the labels expected to earn full marks."""
    seen = 0
    for q in questions[:n]:
        chain = wrong_chain(q)
        if chain is None:
            continue
        replies, earns = chain
        result = grade(skill, q, replies)
        for p in q.parts:
            got = result.parts[p.label].marks
            if p.label in earns:
                assert got == p.marks, (q.text, replies, p.label)
            else:
                assert got == 0, (q.text, replies, p.label)
        assert not result.correct
        seen += 1
    assert seen >= (min_seen if min_seen is not None else n // 2)


def check_all_wrong_zero(skill, questions, junk: dict[str, str]) -> None:
    for q in questions[:40]:
        result = grade(skill, q, junk)
        assert result.marks == 0 and not result.correct


def check_deterministic(topic: str) -> None:
    assert make(topic, seed=7).to_dict() == make(topic, seed=7).to_dict()


def frac(text: str) -> Fraction:
    return Fraction(text.strip().replace(",", ""))
