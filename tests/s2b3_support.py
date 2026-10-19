"""Shared helpers for the second-year inner-product-space skill tests (`tests/test_s2b3_*.py`).

Each test file reads the data of a question back out of its TEXT and re-derives the key with SymPy, so the oracle shares
no code with the generator. The helpers here do the reading and the generic checks (the key is accepted, every
distractor of every part is rejected and tagged, text quality, determinism).
"""

from __future__ import annotations

import ast
import re

import sympy

from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects


def skill(skill_id):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def stem(q):
    return q.stem if q.parts else q.text


_VECTOR_RE = re.compile(r"\((-?\d+(?:, -?\d+)+)\)")
_MATRIX_RE = re.compile(r"\[\[[^A-Za-z]*?\]\]")


def int_vectors(text):
    """Every parenthesised list of integers in `text`, in order, as a list of ints."""
    return [[int(x) for x in m.group(1).split(", ")] for m in _VECTOR_RE.finditer(text)]


def named_vector(text, name):
    match = re.search(rf"\b{name} = \((-?\d+(?:, -?\d+)+)\)", text)
    return [int(x) for x in match.group(1).split(", ")]


def matrix_in(text):
    return sympy.Matrix(ast.literal_eval(_MATRIX_RE.search(text).group(0)))


def answer_matrix(q_or_part):
    """The canonical answer of a `matrix` question or part as a SymPy matrix (rows as stored)."""
    rows = q_or_part.answer
    out = []
    for row in rows:
        out.append([_entry(e) for e in row])
    return sympy.Matrix(out)


def _entry(e):
    if isinstance(e, dict):
        return sympy.Rational(e["p"], e["q"])
    if isinstance(e, str):
        return sympy.sympify(e)
    return sympy.Integer(e)


def answer_vector(q_or_part):
    return list(answer_matrix(q_or_part))


def same_matrix(left, right):
    return left.shape == right.shape and (left - right).applyfunc(sympy.simplify) == sympy.zeros(*left.shape)


def check_single_tier(sk, make_question, seeds=60):
    """The key is accepted, every distractor is rejected, tagged in the catalogue and stored with its value."""
    for seed in range(seeds):
        q = make_question(seed)
        assert grade(sk, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(sk, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == make_question(seed).to_json()


def check_exam_grading(sk, make_exam, seeds=40):
    """Full marks for the canonical reply; zero for a part's own distractor (the other parts correct)."""
    for seed in range(seeds):
        q = make_exam(seed)
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (seed, full)
        canon = {p.label: p.answer_text for p in q.parts}
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == make_exam(seed).to_json()
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for text in p.distractors:
                result = grade(sk, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text, p.answer_text)
                assert p.option_notes[text] in CATALOGUE


def check_wiring(sk, topic, exam_topic, prerequisites):
    assert sk.answer == "matrix"
    assert sk.status == "active"
    assert [(s.topic, s.difficulty) for s in sk.sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam_topic, "exam"),
    ]
    assert sk.sources[-1].stages == ["exam"]
    for pre in prerequisites:
        assert pre in sk.prerequisites
    assert "university-year-2" in sk.tags
