"""Tests the shared `mathgen.core` helpers `money_cents_rounded_2dp`, `prompt_safe_str`, `value_safe_str` and
`pairwise_distinct`.

`money_cents_rounded_2dp`, `prompt_safe_str` and `value_safe_str`, and `pairwise_distinct`."""

from fractions import Fraction

import pytest
from sympy import Symbol, sympify

from mathgen.core import (
    VerificationError,
    check_pairwise_distinct,
    money_cents_rounded_2dp,
    pairwise_distinct,
    prompt_safe_str,
    round_half_up,
    stated_precision_string,
    value_safe_str,
)


# --- money_cents_rounded_2dp --------------------------------------------------
# Helper: `Fraction`
# cents (possibly non-integer) -> half-up-rounded 2dp dollar string. This is
# exactly the `round_half_up` + `stated_precision_string` composition that was
# duplicated verbatim as `_money_rounded` in both
# `mathgen/generators/p12a_financial_model_build.py` and
# `mathgen/generators/p13a_direct_proportion_scale_build.py`.


class TestMoneyCentsRounded2dp:
    def test_exact_integer_cents(self):
        assert money_cents_rounded_2dp(450) == "4.50"

    def test_exact_integer_cents_fraction(self):
        assert money_cents_rounded_2dp(Fraction(450)) == "4.50"

    def test_matches_manual_composition(self):
        cents = Fraction(4517, 3)  # a misconception's own non-integer cents arithmetic
        expected = stated_precision_string(round_half_up(cents / 100, 2), 2)
        assert money_cents_rounded_2dp(cents) == expected

    def test_rounds_half_up_not_to_even(self):
        # 450.5 cents -> half-up (away from zero, since non-negative here) -> 451 cents -> "4.51"
        assert money_cents_rounded_2dp(Fraction(4505, 10)) == "4.51"

    def test_zero_cents(self):
        assert money_cents_rounded_2dp(0) == "0.00"

    def test_negative_cents_rounds_away_from_zero(self):
        # round_half_up ties away from zero: -450.5 cents -> -451 cents -> "-4.51"
        assert money_cents_rounded_2dp(Fraction(-4505, 10)) == "-4.51"

    def test_no_currency_symbol(self):
        assert "$" not in money_cents_rounded_2dp(199)


# --- prompt_safe_str / value_safe_str -----------------------------------------
# Helper: a shared sympy-to-display-string helper --
# `mathgen.core.prompt_safe_str(expr)` (`**` -> `^`, `*` -> `·`) and
# `mathgen.core.value_safe_str(expr)` (`**` -> `^` only) -- matching
# `p13c_antideriv_notation_build.py`'s own `_disp`/`_value` and
# `differential_equation`'s `_ode_exponential` (`str(ans).replace('**', '^')`).


class TestPromptSafeStr:
    def test_double_star_becomes_caret(self):
        x = Symbol("x")
        assert prompt_safe_str(x**2) == "x^2"

    def test_star_becomes_multiplication_dot(self):
        x = Symbol("x")
        assert "·" in prompt_safe_str(3 * x)
        assert "*" not in prompt_safe_str(3 * x)

    def test_combined_power_and_product(self):
        x = Symbol("x")
        expr = sympify("2*x**3")
        result = prompt_safe_str(expr)
        assert "**" not in result
        assert "*" not in result
        assert "^" in result
        assert "·" in result

    def test_plain_string_input(self):
        assert prompt_safe_str("2*x**2") == "2·x^2"

    def test_no_operators_untouched(self):
        x = Symbol("x")
        assert prompt_safe_str(x + 1) == str(x + 1)


class TestValueSafeStr:
    def test_double_star_becomes_caret(self):
        x = Symbol("x")
        assert value_safe_str(x**2) == "x^2"

    def test_bare_star_is_untouched(self):
        # unlike prompt_safe_str, a bare "*" is left alone in a value string
        x = Symbol("x")
        result = value_safe_str(3 * x)
        assert "**" not in result
        assert "·" not in result

    def test_matches_ode_exponential_convention(self):
        x = Symbol("x")
        expr = sympify("2*x**2")
        assert value_safe_str(expr) == str(expr).replace("**", "^")

    def test_plain_string_input(self):
        assert value_safe_str("2*x**2") == "2*x^2"


# --- pairwise_distinct ---------------------------------------------------------
# Helper: `pairwise_distinct(*values) -> bool` -- a
# non-raising sibling of `check_pairwise_distinct` for a redraw-until-distinct
# guard loop, sharing its comparison logic exactly.


class TestPairwiseDistinct:
    def test_all_distinct_values_true(self):
        assert pairwise_distinct(1, 2, 3) is True

    def test_a_collision_is_false(self):
        assert pairwise_distinct(1, 2, 2) is False

    def test_single_value_is_trivially_true(self):
        assert pairwise_distinct(5) is True

    def test_no_values_is_trivially_true(self):
        assert pairwise_distinct() is True

    def test_symbolic_collision_detected(self):
        x = Symbol("x")
        # (x + 1)**2 and x**2 + 2*x + 1 are the same value, different spelling
        assert pairwise_distinct(x**2 + 2 * x + 1, (x + 1) ** 2) is False

    def test_string_fraction_values(self):
        assert pairwise_distinct("1/2", "0.5") is False
        assert pairwise_distinct("1/2", "1/3", "1/4") is True

    def test_shares_logic_with_check_pairwise_distinct_on_success(self):
        # check_pairwise_distinct must not raise when pairwise_distinct is True
        check_pairwise_distinct(1, 2, 3)

    def test_shares_logic_with_check_pairwise_distinct_on_failure(self):
        # check_pairwise_distinct must raise exactly when pairwise_distinct is False
        with pytest.raises(VerificationError):
            check_pairwise_distinct(1, 2, 2)

    def test_agrees_with_check_pairwise_distinct_across_cases(self):
        cases = [
            (1, 2, 3),
            (1, 2, 2),
            ("a", "b"),
            (Fraction(1, 2), Fraction(2, 4)),
            (Fraction(1, 2), Fraction(1, 3)),
        ]
        for values in cases:
            ok = pairwise_distinct(*values)
            if ok:
                check_pairwise_distinct(*values)
            else:
                with pytest.raises(VerificationError):
                    check_pairwise_distinct(*values)
