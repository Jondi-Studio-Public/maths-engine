"""Exam tier for two-by-two-eigenvalues: the eigenvalues of sA + tI for a full 2x2 matrix A.

(a) several prompt shapes, some absent from the hard tier; (b) the key re-derived from the RENDERED PROMPT with
SymPy over 300 seeds; (c) the skill's grader accepts the key and rejects every distractor over 150 seeds.
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u2_eigen_shifted"
GRADER = GRADERS["list"]


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _independent(text: str) -> list[int]:
    rows = re.search(r"\[\[(-?\d+), (-?\d+)\], \[(-?\d+), (-?\d+)\]\]", text)
    a, b, c, d = (int(v) for v in rows.groups())
    m = re.search(r"M = (-?)(\d*)A ([+-]) (\d*)I", text)
    scale = (-1 if m.group(1) else 1) * (int(m.group(2)) if m.group(2) else 1)
    shift = (1 if m.group(3) == "+" else -1) * (int(m.group(4)) if m.group(4) else 1)
    matrix = scale * sympy.Matrix([[a, b], [c, d]]) + shift * sympy.eye(2)
    lam = sympy.Symbol("lam")
    roots = sympy.solve(sympy.det(matrix - lam * sympy.eye(2)), lam)
    return sorted(int(r) for r in roots)


def test_matrix_is_not_triangular():
    for seed in range(100):
        rows = re.search(r"\[\[(-?\d+), (-?\d+)\], \[(-?\d+), (-?\d+)\]\]", _q(seed).text)
        assert int(rows.group(3)) != 0 and int(rows.group(2)) != 0


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        if list(q.answer) != _independent(q.text):
            bad.append((seed, q.text, q.answer_text))
    assert bad == []


def test_grader_accepts_key_and_rejects_every_distractor():
    bad = []
    for seed in range(150):
        q = _q(seed)
        assert len(q.distractors) >= 2 and q.answer_kind == "list"
        assert all(note in CATALOGUE for note in q.option_notes.values())
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        reversed_key = ", ".join(reversed(q.answer_text.split(", ")))
        if not GRADER(q, reversed_key).correct:
            bad.append(("reordered key rejected", seed, reversed_key))
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == []
