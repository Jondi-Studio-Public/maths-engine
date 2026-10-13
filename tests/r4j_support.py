"""Shared checks for the `r4j_*` skill tests: grading every canonical answer and every distractor, and follow-through."""

from __future__ import annotations

from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE

SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₋ₙ−", "0123456789-n-")


def plain(text: str) -> str:
    """Subscript digits to digits, and the true minus sign to a hyphen."""
    return text.translate(SUBSCRIPTS)


def check_single(q, kind):
    """Canonical answer right, every distractor wrong, every distractor tagged with a catalogue id."""
    assert grade(kind, q, q.answer_text).correct, (q.topic, q.seed, q.answer_text)
    assert len(q.distractors) >= 2, (q.topic, q.seed)
    for d in q.distractors:
        assert not grade(kind, q, str(d)).correct, (q.topic, q.seed, "distractor accepted", d)
    for tag in (getattr(q, "option_notes", None) or {}).values():
        assert tag in CATALOGUE, tag


def check_parts(q, count=4):
    """Every part's canonical answer right, every part's distractors wrong, all together graded correct."""
    assert q.parts and len(q.parts) == count
    replies = {p.label: p.answer_text for p in q.parts}
    result = grade(q.parts[0].answer_kind, q, replies)
    assert result.correct and result.marks == result.max_marks, (q.topic, q.seed)
    assert sum(p.marks for p in q.parts) == result.max_marks
    for p in q.parts:
        assert grade(p.answer_kind, p, p.answer_text).correct, (q.topic, q.seed, p.label, p.answer_text)
        assert len(p.distractors) >= 2, (q.topic, q.seed, p.label)
        for d in p.distractors:
            assert not grade(p.answer_kind, p, str(d)).correct, (q.topic, q.seed, p.label, "distractor accepted", d)
        for tag in (p.option_notes or {}).values():
            assert tag in CATALOGUE, tag


def check_carried(q, wrong_label, wrong_reply, final_reply, final_label="d"):
    """A wrong earlier reply, carried correctly into the last part, still earns the last part's marks as follow-through."""
    final = next(p for p in q.parts if p.label == final_label)
    replies = {p.label: p.answer_text for p in q.parts}
    replies[wrong_label], replies[final_label] = wrong_reply, final_reply
    result = grade(q.parts[0].answer_kind, q, replies)
    got = result.parts[final_label]
    assert got.marks == final.marks and got.follow_through, (q.topic, q.seed, wrong_reply, final_reply)
    assert not result.parts[wrong_label].correct
