"""Exam source for `matrix-solve-system`: (A B) x = b, A x = B c and A x = x + b."""

from __future__ import annotations

import re

from sympy import Matrix, eye

from mathgen import make
from mathgen import skills as sm
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

SKILL_ID = "matrix-solve-system"
TOPIC = "exd_x_g4_matrix_solve_system"


def _skill():
    return {s.id: s for s in sm.load_skills()[1]}[SKILL_ID]


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def test_yaml_has_the_exam_source_last():
    src = _skill().sources[-1]
    assert (src.topic, src.difficulty, list(src.stages)) == (TOPIC, "exam", ["exam"])
    assert len(_skill().sources) == 4


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_make(seed)) == []


def test_prompt_shapes_are_varied_and_new_over_hard():
    hard = next(s for s in _skill().sources if s.difficulty == "hard")
    hard_shapes = {
        prompt_shape(
            make(
                hard.topic,
                seed=s,
                difficulty="hard",
                **{k: tuple(v) if isinstance(v, list) else v for k, v in hard.params.items()},
            )
        )
        for s in range(60)
    }
    shapes = {prompt_shape(_make(s)) for s in range(60)}
    assert len(shapes) >= 3
    assert len(shapes - hard_shapes) >= 3


def test_deterministic():
    assert _make(7).to_json() == _make(7).to_json()


_MAT = re.compile(r"\[(-?\d+) (-?\d+); (-?\d+) (-?\d+)\]")
_VEC = re.compile(r"\((-?\d+), (-?\d+)\)")


def _oracle(text: str):
    """Re-derive (x, y) from the rendered prompt with exact sympy linear algebra."""
    mats = [Matrix(2, 2, [int(g) for g in m.groups()]) for m in _MAT.finditer(text)]
    vec = Matrix([int(g) for g in _VEC.findall(text)[-1]])
    if text.startswith("Solve (A B) x = b"):
        A, B = mats
        sol = (A * B).LUsolve(vec)
    elif text.startswith("Solve A x = B c"):
        A, B = mats
        sol = A.LUsolve(B * vec)
    else:
        assert text.startswith("Find the vector x = (x, y) such that A x = x + b"), text
        (A,) = mats
        sol = (A - eye(2)).LUsolve(vec)
    return {"x": int(sol[0]), "y": int(sol[1])}


def test_key_matches_independent_oracle_over_300_seeds():
    bad = [(s, _make(s).text) for s in range(300) if _oracle(_make(s).text) != _make(s).answer]
    assert bad == []


def test_all_three_shapes_occur():
    heads = {_make(s).text[:12] for s in range(60)}
    assert len(heads) == 3


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    grade = GRADERS["pair"]
    bad = []
    for seed in range(150):
        q = _make(seed)
        assert len(q.distractors) >= 2
        if not grade(q, q.answer_text).correct:
            bad.append(("key", seed))
        bad += [("distractor", seed, d) for d in q.distractors if grade(q, d).correct]
    assert bad == []
