"""Tests the rearrangement-first hard tier of `solve-quadratics-formula`.

Solve-quadratics-formula rearrangement-first hard tier."""

from __future__ import annotations

from mathgen import make


def test_formula_rearrange_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("quadratic", seed=seed, kind="formula_rearrange")
        assert q.answer_text not in q.distractors
        assert "x(" in q.text  # unrearranged form: x(ax + b) = -c
        assert " = 0" not in q.text  # not already in standard form


def test_formula_rearrange_reachable_from_hard_source():
    q = make("quadratic", seed=3, difficulty="hard", kind="formula_rearrange")
    assert q.answer_text
