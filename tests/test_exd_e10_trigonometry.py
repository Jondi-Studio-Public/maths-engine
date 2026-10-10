"""Tests the `triangle_area_reverse` kind of the `trigonometry` generator, the hard tier of `triangle-area-sine`.

Triangle-area-sine hard tier.

right-triangle-angle's "pythagoras_first" style was REVERTED: it was a counterfeit escalation (opp+hyp gives
the angle directly via sin, so "find the adjacent side first" was a prompt-only detour, not a new reasoning
shape).. Only triangle-area-sine's hard tier is tested here now.
"""

from __future__ import annotations

from mathgen import make


def test_triangle_area_reverse_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("trigonometry", seed=seed, kinds=("triangle_area_reverse",))
        assert q.answer_text not in q.distractors
        assert "area" in q.text and q.text.strip().startswith("Triangle")


def test_triangle_area_reverse_reachable_from_hard_source():
    q = make("trigonometry", seed=2, difficulty="hard", kinds=("triangle_area_reverse",))
    assert q.answer_text
