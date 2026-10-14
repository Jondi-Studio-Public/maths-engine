"""The hard divisibility-by-difference question of `ucv1-proof-by-induction` is drawn from more than two statements."""

from __future__ import annotations

from mathgen import make


def test_cubic_difference_statements_vary():
    seen = set()
    for seed in range(1, 301):
        q = make("ucv1_proof_by_induction", seed=seed, difficulty="hard")
        if "compared with f(k)" in q.text:
            seen.add(q.text)
            assert len(q.distractors) >= 2
    assert len(seen) >= 4
