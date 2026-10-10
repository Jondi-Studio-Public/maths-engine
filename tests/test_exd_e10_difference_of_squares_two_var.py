"""Tests the two-variable hard tier of `difference-of-squares`.

Difference-of-squares two-variable hard tier."""

from __future__ import annotations

from mathgen import make


def test_two_var_matches_own_answer_over_many_seeds():
    for seed in range(120):
        q = make("difference_of_squares", seed=seed, kind="two_var")
        assert q.answer_text not in q.distractors
        assert "y" in q.text


def test_two_var_answer_form_is_expanded_or_factored():
    for seed in range(60):
        q = make("difference_of_squares", seed=seed, kind="two_var")
        assert q.answer_form in ("expanded", "factored")


def test_two_var_reachable_from_hard_source():
    q = make("difference_of_squares", seed=5, difficulty="hard", kind="two_var")
    assert "y" in q.text
