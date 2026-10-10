"""Tests the `scale_product_divide_back` and decimal-power missing-factor variants of
`mental_mult_div_extension`, each of which verifies its own answer."""

import mathgen


def test_scale_product_divide_back_verifies():
    for seed in range(30):
        q = mathgen.make("mental_mult_div_extension", seed=seed, difficulty="hard", variant="scale_product_divide_back")
        assert "÷" in q.text
        assert "×" in q.text  # the stated fact still uses ×
        assert str(q.answer) not in q.distractors


def test_decimal_power_missing_factor_verifies():
    for seed in range(30):
        q = mathgen.make(
            "powers_ten_multiply_divide", seed=seed, difficulty="hard", variant="decimal_power_missing_factor"
        )
        assert "___" in q.text
        assert q.answer in (10, 100, 1000)
        assert str(q.answer) not in q.distractors
