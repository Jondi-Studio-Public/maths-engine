"""The remainder-theorem exam chain states that `a` is an integer, so it offers no fractional option."""

from __future__ import annotations

from mathgen import make


def test_remainder_theorem_options_are_integers():
    for seed in range(1, 121):
        q = make("exd_x_sc1_poly_remainder_unknown", seed=seed, difficulty="exam")
        for part in q.parts:
            assert len(part.distractors) >= 2, (seed, part.label)
            for d in part.distractors:
                assert "/" not in str(getattr(d, "text", d)), (seed, part.label, d)
