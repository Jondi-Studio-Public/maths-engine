"""Shared checks for the `test_exd_x_r3b_*` files: wiring, shapes, grading, tagging and answer space.

Nothing here imports a generator module. A file supplies its own oracle (the key re-derived from the rendered
stem) and its own follow-through cases; these helpers cover what every multi-part exam source must satisfy.
"""

from __future__ import annotations

import re
from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILLS = {s.id: s for s in load_skills()[1]}


def skill(skill_id: str):
    return _SKILLS[skill_id]


def exam_questions(topic: str, n: int, start: int = 0):
    return [make(topic, seed=s, difficulty="exam") for s in range(start, start + n)]


def check_wiring(topic: str, skill_id: str) -> None:
    exam = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [topic], exam
    assert all(list(s.stages) == ["exam"] for s in exam)


def check_shapes(topic: str, skill_id: str, n: int = 60) -> None:
    """At least three distinct prompt shapes, at least one the skill's hard tier never shows."""
    hard = [s for s in skill(skill_id).sources if s.difficulty == "hard"]
    hard_shapes = {prompt_shape(make(src.topic, seed=seed, **(src.params or {}))) for src in hard for seed in range(n)}
    shapes = {prompt_shape(q) for q in exam_questions(topic, n)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard_shapes, "the exam tier shows no prompt shape the hard tier lacks"


def check_structure(topic: str, labels: str, marks: list[int], n: int = 40) -> None:
    for q in exam_questions(topic, n):
        assert "".join(p.label for p in q.parts) == labels
        assert [p.marks for p in q.parts] == marks
        assert all(p.follow_through is None for p in q.parts[:1])
        assert any(p.follow_through for p in q.parts[1:])


def check_grading(topic: str, skill_id: str, n: int = 120) -> None:
    """The composed key earns full marks; every distractor of every part earns that part nothing."""
    sk = skill(skill_id)
    failures = []
    for q in exam_questions(topic, n):
        full = grade(sk, q, q.answer_text)
        if not (full.correct and full.marks == full.max_marks):
            failures.append(("key", q.seed, q.answer_text))
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            if len(p.distractors) < 2:
                failures.append(("few distractors", q.seed, p.label))
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                if result.parts[p.label].marks != 0:
                    failures.append(("distractor accepted", q.seed, p.label, d))
    assert failures == [], failures[:4]


def check_tags(topic: str, n: int = 60) -> None:
    """Every distractor carries a catalogue id, and a number part's options share the key's decimal places."""
    for q in exam_questions(topic, n):
        for p in q.parts:
            for d in p.distractors:
                note = p.option_notes.get(d)
                assert note in CATALOGUE, (q.seed, p.label, d, note)
            if p.answer_kind == "number":
                places = len(p.answer_text.split(".")[1]) if "." in p.answer_text else None
                for d in p.distractors:
                    got = len(d.split(".")[1]) if "." in d else None
                    assert got == places, (q.seed, p.label, p.answer_text, d)
                    assert re.fullmatch(r"-?\d+(\.\d+)?", d), d


def check_answer_space(topic: str, n: int = 300) -> None:
    """At least 100 distinct answer tuples over 300 seeds, none above 5%."""
    tuples = Counter(tuple(p.answer_text for p in q.parts) for q in exam_questions(topic, n))
    assert len(tuples) >= 100, len(tuples)
    assert max(tuples.values()) / n <= 0.05, tuples.most_common(3)


def check_clean_and_deterministic(topic: str, n: int = 60) -> None:
    for seed in range(n):
        q = make(topic, seed=seed, difficulty="exam")
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == make(topic, seed=seed, difficulty="exam").to_json()
