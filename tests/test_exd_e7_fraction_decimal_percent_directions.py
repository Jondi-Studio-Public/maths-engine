"""Fraction_decimal_percent's hard-tier `reverse_directions` mode adds decimal->percent and percent->decimal
conversions, directions the easy/medium tiers never ask for."""

import mathgen


def test_reverse_directions_can_reach_new_modes():
    texts = {
        mathgen.make(
            "fraction_decimal_percent", seed=seed, denominators=(2, 4, 5, 10, 20, 25), reverse_directions=True
        ).text
        for seed in range(40)
    }
    assert any("equal to the decimal" in t for t in texts)
    assert any(t.startswith("Which decimal is equal to") and "%" in t for t in texts)


def test_reverse_directions_default_is_false_and_unchanged():
    a = mathgen.make("fraction_decimal_percent", seed=5, denominators=(2, 4, 5, 10, 20, 25))
    b = mathgen.make("fraction_decimal_percent", seed=5, denominators=(2, 4, 5, 10, 20, 25), reverse_directions=False)
    assert a.text == b.text
    assert a.answer == b.answer
