"""Limit-removable/one-sided-limits hard tiers."""

from __future__ import annotations

from mathgen import make


def test_quad_over_quad_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("limits", seed=seed, kinds=("quad_over_quad",))
        assert q.answer_text not in q.distractors
        assert isinstance(q.answer, int)


def test_one_sided_quad_matches_own_answer_over_many_seeds():
    for seed in range(60):
        q = make("limits", seed=seed, kinds=("one_sided_quad",))
        assert q.answer_text not in q.distractors
        assert "x^2" in q.text


def test_both_reachable_from_hard_sources():
    a = make("limits", seed=2, difficulty="hard", kinds=("quad_over_quad",))
    b = make("limits", seed=2, difficulty="hard", kinds=("one_sided_quad",))
    assert a.answer_text and b.answer_text


def test_one_sided_quad_never_prints_a_unit_coefficient_or_a_zero_term_over_1000_seeds():
    # Swept over 1000 seeds (not sampled) since the original defect was a third of all seeds, not
    # occasional.
    checked = 0
    for seed in range(1000):
        q = make("limits", seed=seed, kinds=("one_sided_quad",))
        assert " 1x" not in q.text, f"seed={seed}: unit coefficient printed literally: {q.text!r}"
        assert "+ 0" not in q.text, f"seed={seed}: zero term printed: {q.text!r}"
        assert "- 0" not in q.text, f"seed={seed}: zero term printed: {q.text!r}"
        checked += 1
    assert checked == 1000
