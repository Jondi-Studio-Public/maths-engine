"""Tests the from_two_terms hard tier of `senior-use-recursion-to-generate-an-arithmetic-sequence`.

Senior-use-recursion-to-generate-an-arithmetic-sequence from_two_terms hard tier."""

from __future__ import annotations

from mathgen import make


def test_from_two_terms_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("arithmetic_recursion_generate", seed=seed, variant="from_two_terms")
        assert q.answer_text not in q.distractors
        assert "t1" not in q.text.split(".")[0]  # t1 is not stated directly in the given data


def test_from_two_terms_reachable_from_hard_source():
    q = make("arithmetic_recursion_generate", seed=3, difficulty="hard", variant="from_two_terms")
    assert q.answer_text
