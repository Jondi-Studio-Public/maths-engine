"""Shared checks for the `w5_*` skill tests. Everything in `tests/w7b_support.py` is re-exported; this adds an exam-chain
check for parts whose grader kind may vary with the value (a whole number is a `number`, a fraction a `fraction`)."""

from __future__ import annotations

from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects
from tests.w7b_support import *  # noqa: F401,F403
from tests.w7b_support import exam


def check_exam_chain_kinds(skill, exam_topic: str, marks: list[int], kinds: list[set[str]], seeds: int = 60) -> None:
    """Every seed: the parts carry `marks` and one of the allowed kinds, the canonical replies earn full marks, every
    distractor of every part earns nothing and is tagged from the catalogue, the text is clean and the draw repeats."""
    for seed in range(seeds):
        q = exam(exam_topic, seed)
        assert [p.label for p in q.parts] == [chr(97 + i) for i in range(len(marks))]
        assert [p.marks for p in q.parts] == marks
        for part, allowed in zip(q.parts, kinds):
            assert part.answer_kind in allowed, (seed, part.label, part.answer_kind)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == exam(exam_topic, seed).to_json()
        full = grade(skill, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(marks), (seed, full)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(skill, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE, (seed, p.label, text)
