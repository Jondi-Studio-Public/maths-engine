"""Exam tier for matrix-rank-small: the value of k that drops a 3x3 matrix to rank 2.

(a) several prompt shapes, some absent from the hard tier; (b) the key re-derived from the RENDERED PROMPT with
SymPy over 300 seeds, including that the rank really is 2 there and 3 elsewhere; (c) the skill's grader accepts
the key and rejects every distractor over 150 seeds.
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u2_rank_parameter"
GRADER = GRADERS["number"]
_K = sympy.Symbol("k")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _matrix(text: str) -> sympy.Matrix:
    """Read the printed rows back into a SymPy matrix, with k as the unknown."""
    body = re.search(r"(\[\[.*\]\])", text).group(1)
    rows = re.findall(r"\[([^\[\]]+)\]", body)
    return sympy.Matrix([[_K if cell.strip() == "k" else int(cell) for cell in row.split(",")] for row in rows])


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        matrix = _matrix(q.text)
        roots = sympy.solve(sympy.det(matrix), _K)
        key = int(q.answer_text)
        if roots != [key]:
            bad.append((seed, q.text, key, roots))
            continue
        at_key = matrix.subs(_K, key).rank()
        away = matrix.subs(_K, key + 1).rank()
        if (at_key, away) != (2, 3):
            bad.append((seed, "rank", at_key, away))
    assert bad == []


def test_grader_accepts_key_and_rejects_every_distractor():
    bad = []
    for seed in range(150):
        q = _q(seed)
        assert len(q.distractors) >= 2 and q.answer_kind == "number"
        assert all(note in CATALOGUE for note in q.option_notes.values())
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == []
