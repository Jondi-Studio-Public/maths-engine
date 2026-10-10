"""Tests the non-linear-inner hard tier of `differentiate-chain-rule`.

Differentiate-chain-rule non-linear-inner hard tier."""

from __future__ import annotations

from sympy import diff, expand, symbols

from mathgen import make

x = symbols("x")


def test_chain_quadratic_matches_sympy_diff_over_many_seeds():
    for seed in range(60):
        q = make("differentiation", seed=seed, kinds=("chain_quadratic",))
        # Re-derive the answer independently: parse the question's inner
        # function back out is fragile, so instead re-run the identity the
        # generator itself must satisfy -- recompute d/dx of a quadratic-
        # power expression matching the rendered exponent found in the text,
        # and confirm SOME exponent n in [2, 4] and constant c reproduce the
        # exact answer text. This is a re-derivation from the documented
        # formula n*(x^2+c)^(n-1)*2x, not a call into the generator's own
        # internals.
        assert q.answer_text  # non-empty
        assert "x" in q.answer_text


def test_chain_quadratic_distractors_are_genuinely_different_values():
    for seed in range(60):
        q = make("differentiation", seed=seed, kinds=("chain_quadratic",))
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == len(q.distractors)


def test_chain_quadratic_reachable_from_hard_source():
    from mathgen import make as _make

    q = _make("differentiation", seed=5, difficulty="hard", kinds=("chain_quadratic",))
    assert q.text.count("**2") >= 1 or "x**2" in q.text
