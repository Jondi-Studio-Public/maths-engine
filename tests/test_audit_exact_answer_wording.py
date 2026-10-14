"""Exam parts whose answer is an exact fraction must say so, or a correct decimal is rejected without warning."""

from __future__ import annotations

import pytest

from mathgen import make

# (topic, labels of the fraction parts that must carry the wording)
_CASES = (
    ("exd_x_u6_hypergeometric_acceptance", "abc"),
    ("exd_x_u5_lagrange_weighted", "abcd"),
    ("exd_x_r2c_limit_at_infinity", "bcd"),
    ("exd_x_r2c_limit_removable", "d"),
    ("exd_x_uc_triple_integral", "abcd"),
    ("exd_x_r2a_rules_tangent", "bd"),
    ("exd_x_r2a_first_principles_tangent", "bd"),
    ("exd_x_r2a_parallel_tangent_normal", "ad"),
    ("exd_x_r2d_lt_reflection_gradient", "abc"),
    ("exd_x_r2k_bernoulli_suitability", "ab"),
)


@pytest.mark.parametrize(("topic", "labels"), _CASES)
def test_fraction_parts_ask_for_an_exact_answer(topic, labels):
    for seed in range(1, 13):
        q = make(topic, seed=seed, difficulty="exam")
        by_label = {p.label: p for p in q.parts}
        for label in labels:
            part = by_label[label]
            assert part.answer_kind == "fraction", (topic, seed, label)
            prompt = part.prompt.lower()
            assert "fraction" in prompt, (topic, seed, label, part.prompt)
