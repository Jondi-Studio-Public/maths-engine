"""Fractions' hard-tier `three_term` mode chains three fractions with two operators -- a shape the two-term
easy/medium tiers cannot reach. Never changes the default (non-three_term) rendering, so any output pinned for the default
stays byte-identical."""

import mathgen


def test_three_term_renders_three_fractions():
    for seed in range(15):
        q = mathgen.make("fractions", seed=seed, max_den=20, ops=("+", "-", "*", "/"), three_term=True)
        assert q.text.count("/") == 3


def test_three_term_default_is_false_and_unchanged():
    a = mathgen.make("fractions", seed=5, max_den=20, ops=("+", "-", "*"))
    b = mathgen.make("fractions", seed=5, max_den=20, ops=("+", "-", "*"), three_term=False)
    assert a.text == b.text
    assert a.answer == b.answer
