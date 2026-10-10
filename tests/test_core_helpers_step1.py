"""Tests for the eleven shared helpers added to `mathgen/core.py` (step 1 of the mathgen/core.py
duplication-survey refactor): `fixed_point_string`, `exact_decimal_string`, `rounded_decimal_string`,
`numeric_answer`, `round_half_up`, `check_conversion`, `check_combine`, `format_pi_multiple`,
`reversed_and_adjacent_swap`, `rounds_unambiguously` and `currency_from_cents`.

This is step 1 only: the ~40 generator call sites that hand-roll these
patterns today are migrated separately later. These tests exercise the
helpers in isolation, against known values and against the specific old
behaviour (e.g. the "+infinity" tie-break) that each helper must, or must
not, reproduce.
"""

from fractions import Fraction

import pytest

from mathgen.core import (
    VerificationError,
    check_combine,
    check_conversion,
    currency_from_cents,
    exact_decimal_string,
    fixed_point_string,
    format_pi_multiple,
    latex_pi_multiple,
    numeric_answer,
    reversed_and_adjacent_swap,
    round_half_up,
    rounded_decimal_string,
    rounds_unambiguously,
)


# --- fixed_point_string ------------------------------------------------------


class TestFixedPointString:
    def test_negative_two_places(self):
        assert fixed_point_string(-305, 2) == "-3.05"

    def test_zero_places_is_a_bare_integer(self):
        assert fixed_point_string(7, 0) == "7"

    def test_zero_value_two_places(self):
        assert fixed_point_string(0, 2) == "0.00"

    def test_positive_two_places(self):
        assert fixed_point_string(1234, 2) == "12.34"


# --- exact_decimal_string -----------------------------------------------------


class TestExactDecimalString:
    def test_exact_case(self):
        assert exact_decimal_string(Fraction(1, 4), 2) == "0.25"

    def test_uses_the_shortest_terminating_form(self):
        assert exact_decimal_string(Fraction(1, 2), 4) == "0.5"

    def test_non_terminating_raises(self):
        with pytest.raises(ValueError):
            exact_decimal_string(Fraction(1, 3), 4)

    def test_negative_exact_case(self):
        assert exact_decimal_string(Fraction(-3, 4), 2) == "-0.75"


# --- rounded_decimal_string ----------------------------------------------------


class TestRoundedDecimalString:
    def test_exact_case_matches_exact_decimal_string(self):
        value = Fraction(1, 4)
        assert rounded_decimal_string(value, 2) == exact_decimal_string(value, 2)

    def test_non_exact_case_rounds_half_up_with_old_plus_infinity_tie_break(self):
        # 1/8 = 0.125 exactly at 3dp, but rounding to 2dp lands exactly on the
        # .5-of-the-next-digit boundary: 0.125 * 100 = 12.5, so the OLD
        # "+infinity" formula (floor(value*scale + 1/2)) rounds up to 13,
        # giving "0.13" -- this must reproduce that behaviour exactly.
        assert rounded_decimal_string(Fraction(1, 8), 2) == "0.13"

    def test_non_exact_non_boundary_case(self):
        # 1/3 = 0.333... -> "0.33" at 2dp.
        assert rounded_decimal_string(Fraction(1, 3), 2) == "0.33"

    def test_negative_tie_keeps_old_plus_infinity_rule_not_round_half_up(self):
        # The one input that discriminates this from round_half_up: a
        # negative tie. -1/8 = -0.125; the OLD "+infinity" formula is
        # floor(-0.125*100 + 1/2) = floor(-12.5 + 0.5) = floor(-12.0) = -12
        # -> "-0.12". round_half_up (ties away from zero) would instead give
        # "-0.13". This pins the design decision that rounded_decimal_string
        # deliberately does NOT route through round_half_up.
        assert rounded_decimal_string(Fraction(-1, 8), 2) == "-0.12"


# --- numeric_answer -----------------------------------------------------------


class TestNumericAnswer:
    def test_whole_fraction_becomes_int(self):
        result = numeric_answer(Fraction(6, 1))
        assert result == 6
        assert isinstance(result, int)

    def test_non_whole_fraction_becomes_float(self):
        result = numeric_answer(Fraction(1, 3))
        assert isinstance(result, float)
        assert result == pytest.approx(1 / 3)


# --- round_half_up ------------------------------------------------------------


class TestRoundHalfUp:
    def test_negative_tie_rounds_away_from_zero(self):
        # -0.25 to 1dp: ties-away-from-zero gives -0.3, NOT the -0.2 that the
        # old "+infinity" formula (floor(value*scale + 1/2)) would give.
        assert round_half_up(Fraction(-1, 4), 1) == Fraction(-3, 10)

    def test_positive_tie_rounds_up(self):
        assert round_half_up(Fraction(1, 4), 1) == Fraction(3, 10)

    def test_ordinary_non_tie_case(self):
        assert round_half_up(Fraction(1, 3), 2) == Fraction(33, 100)

    def test_places_zero(self):
        assert round_half_up(Fraction(5, 2)) == Fraction(3, 1)

    def test_places_zero_negative_tie(self):
        assert round_half_up(Fraction(-5, 2)) == Fraction(-3, 1)

    def test_places_two(self):
        assert round_half_up(Fraction(1, 8), 2) == Fraction(13, 100)


# --- check_conversion ---------------------------------------------------------


class TestCheckConversion:
    FACTORS = {"m": Fraction(1), "cm": Fraction(1, 100)}

    def test_passing_case(self):
        # 250 cm -> m: pivot = 250 * (1/100) = 2.5 m
        check_conversion(250, "cm", "m", Fraction(5, 2), self.FACTORS)

    def test_failing_case_raises(self):
        with pytest.raises(VerificationError):
            check_conversion(250, "cm", "m", Fraction(3), self.FACTORS)


# --- check_combine -------------------------------------------------------------


class TestCheckCombine:
    FACTORS = {"m": Fraction(1), "cm": Fraction(1, 100)}

    def test_passing_sum(self):
        values = {"m": Fraction(2), "cm": Fraction(50)}
        # 50 cm -> 0.5 m; 2 + 0.5 = 2.5
        check_combine(values, "cm", "m", "sum", Fraction(5, 2), self.FACTORS)

    def test_passing_difference(self):
        values = {"m": Fraction(2), "cm": Fraction(50)}
        # |2 - 0.5| = 1.5
        check_combine(values, "cm", "m", "difference", Fraction(3, 2), self.FACTORS)

    def test_failing_case_raises(self):
        values = {"m": Fraction(2), "cm": Fraction(50)}
        with pytest.raises(VerificationError):
            check_combine(values, "cm", "m", "sum", Fraction(10), self.FACTORS)

    def test_unknown_operation_raises_value_error(self):
        values = {"m": Fraction(2), "cm": Fraction(50)}
        with pytest.raises(ValueError):
            check_combine(values, "cm", "m", "product", Fraction(1), self.FACTORS)


# --- format_pi_multiple --------------------------------------------------------


class TestFormatPiMultiple:
    def test_proper_fraction(self):
        assert format_pi_multiple(Fraction(5, 6)) == "5pi/6"

    def test_unit_coefficient(self):
        assert format_pi_multiple(Fraction(1, 1)) == "pi"

    def test_negative_half(self):
        assert format_pi_multiple(Fraction(-1, 2)) == "-pi/2"

    def test_zero(self):
        assert format_pi_multiple(Fraction(0, 1)) == "0"

    def test_bare_int_input(self):
        assert format_pi_multiple(2) == "2pi"


# --- latex_pi_multiple ----------------------------------------------------------


class TestLatexPiMultiple:
    """The LaTeX twin of `format_pi_multiple`: same value, KaTeX-renderable.
    Every case here is the same input as the `format_pi_multiple` case above
    it, so the pair cannot drift."""

    def test_proper_fraction(self):
        assert latex_pi_multiple(Fraction(5, 6)) == r"\frac{5\pi}{6}"

    def test_unit_coefficient(self):
        assert latex_pi_multiple(Fraction(1, 1)) == r"\pi"

    def test_unit_numerator_fraction_drops_the_one(self):
        assert latex_pi_multiple(Fraction(1, 2)) == r"\frac{\pi}{2}"

    def test_negative_half(self):
        assert latex_pi_multiple(Fraction(-1, 2)) == r"-\frac{\pi}{2}"

    def test_negative_whole(self):
        assert latex_pi_multiple(Fraction(-3, 1)) == r"-3\pi"

    def test_zero(self):
        assert latex_pi_multiple(Fraction(0, 1)) == "0"

    def test_bare_int_input(self):
        assert latex_pi_multiple(2) == r"2\pi"

    def test_katex_renderable_shape(self):
        """No bare "pi" word survives -- that italic-letters defect is the
        whole reason this helper exists."""
        for coeff in (Fraction(1, 1), Fraction(13, 12), Fraction(-1, 2), 6):
            assert "pi" not in latex_pi_multiple(coeff).replace(r"\pi", "")


# --- reversed_and_adjacent_swap -------------------------------------------------


class TestReversedAndAdjacentSwap:
    def test_four_element_list(self):
        reversed_list, adjacent_swap = reversed_and_adjacent_swap([1, 2, 3, 4])
        assert reversed_list == [4, 3, 2, 1]
        # mid = max(1, 4 // 2) = 2; swap indices 1 and 2 -> [1, 3, 2, 4]
        assert adjacent_swap == [1, 3, 2, 4]

    def test_five_element_list(self):
        reversed_list, adjacent_swap = reversed_and_adjacent_swap([1, 2, 3, 4, 5])
        assert reversed_list == [5, 4, 3, 2, 1]
        # mid = max(1, 5 // 2) = 2; swap indices 1 and 2 -> [1, 3, 2, 4, 5]
        assert adjacent_swap == [1, 3, 2, 4, 5]

    def test_does_not_mutate_the_input(self):
        original = [1, 2, 3, 4]
        reversed_and_adjacent_swap(original)
        assert original == [1, 2, 3, 4]


# --- rounds_unambiguously -------------------------------------------------------


class TestRoundsUnambiguously:
    def test_far_from_boundary_is_true(self):
        assert rounds_unambiguously(1.23, 1) is True

    def test_on_the_boundary_is_false(self):
        # 1.25 to 1dp: 1.25 * 10 = 12.5, exactly on the .5 boundary.
        assert rounds_unambiguously(1.25, 1) is False

    def test_within_margin_of_boundary_is_false(self):
        assert rounds_unambiguously(1.249, 1, margin=0.05) is False


# --- currency_from_cents ---------------------------------------------------------


class TestCurrencyFromCents:
    def test_positive_default(self):
        assert currency_from_cents(450, "$") == "$4.50"

    def test_negative_default_allows_negative(self):
        assert currency_from_cents(-450, "$") == "-$4.50"

    def test_allow_negative_false_rejects_negative(self):
        with pytest.raises(VerificationError):
            currency_from_cents(-450, "$", allow_negative=False)

    def test_allow_negative_false_accepts_non_negative(self):
        assert currency_from_cents(450, "$", allow_negative=False) == "$4.50"

    def test_custom_symbol(self):
        assert currency_from_cents(450, "A$") == "A$4.50"
