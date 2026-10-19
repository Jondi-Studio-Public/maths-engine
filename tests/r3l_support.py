"""Shared checks for the `r3l_*` skill tests: grading every canonical answer and every distractor, and follow-through."""

from __future__ import annotations

import sympy

from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE

SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₋ₙ−", "0123456789-n-")


def plain(text: str) -> str:
    """Subscript digits to digits, and the true minus sign to a hyphen."""
    return text.translate(SUBSCRIPTS)


def parse_matrix(text: str) -> sympy.Matrix:
    """The first `[[...], ...]` literal in `text`, entries integers or fractions like 3/2."""
    start = text.index("[[")
    depth, end = 0, start
    for i in range(start, len(text)):
        depth += {"[": 1, "]": -1}.get(text[i], 0)
        if depth == 0:
            end = i + 1
            break
    rows = [r for r in text[start + 1 : end - 1].split("],")]
    return sympy.Matrix([[sympy.Rational(v.strip()) for v in r.strip(" []").split(",")] for r in rows])


def check_single(q, kind):
    """Canonical answer right, every distractor wrong, every distractor tagged with a catalogue id."""
    assert grade(kind, q, q.answer_text).correct, (q.topic, q.seed, q.answer_text)
    assert len(q.distractors) >= 2, (q.topic, q.seed)
    for d in q.distractors:
        assert not grade(kind, q, str(d)).correct, (q.topic, q.seed, "distractor accepted", d)
    for tag in (getattr(q, "option_notes", None) or {}).values():
        assert tag in CATALOGUE, tag


def check_parts(q):
    """Every part's canonical answer right, every part's distractors wrong, all together graded correct."""
    assert q.parts and len(q.parts) >= 3
    replies = {p.label: p.answer_text for p in q.parts}
    result = grade(q.parts[0].answer_kind, q, replies)
    assert result.correct and result.marks == result.max_marks, (q.topic, q.seed)
    assert sum(p.marks for p in q.parts) == result.max_marks
    for p in q.parts:
        assert grade(p.answer_kind, p, p.answer_text).correct, (q.topic, q.seed, p.label, p.answer_text)
        assert len(p.distractors) >= 2, (q.topic, q.seed, p.label)
        for d in p.distractors:
            assert not grade(p.answer_kind, p, str(d)).correct, (q.topic, q.seed, p.label, "distractor accepted", d)


def _values(q):
    out = {}
    for p in q.parts:
        if p.answer_kind in ("number", "fraction"):
            out[p.label] = sympy.Rational(p.answer_text)
        elif p.answer_kind == "expression":
            out[p.label] = sympy.sympify(
                p.answer_text.replace("^", "**"), locals={"k": sympy.Symbol("k"), "n": sympy.Symbol("n")}
            )
    return out


def follow_through_value(part, values: dict):
    """Evaluate `part.follow_through` at `values` (label -> SymPy value)."""
    names = {f"part_{label}": v for label, v in values.items()}
    expr = sympy.sympify(
        part.follow_through.replace("^", "**"), locals={**names, "n": sympy.Symbol("n"), "k": sympy.Symbol("k")}
    )
    return sympy.nsimplify(expr)


def check_follow_through(q, wrong_label, wrong_value, final_label="d"):
    """A wrong earlier answer, carried correctly into the last part, still earns the last part's marks."""
    final = next(p for p in q.parts if p.label == final_label)
    values = _values(q)
    values[wrong_label] = sympy.nsimplify(wrong_value)
    carried = follow_through_value(final, values)
    replies = {p.label: p.answer_text for p in q.parts}
    replies[wrong_label] = str(wrong_value)
    replies[final_label] = str(carried)
    result = grade(q.parts[0].answer_kind, q, replies)
    got = result.parts[final_label]
    assert got.marks == final.marks, (q.topic, q.seed, wrong_value, carried)
    assert got.follow_through, (q.topic, q.seed)
    assert not result.parts[wrong_label].correct
