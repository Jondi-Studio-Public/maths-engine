"""Tests for the shared helpers added to `mathgen/core.py`: `pairs_equal` (and the `symbolically_equal`
fix it depends on), `latex_answer` and `stated_precision_tolerance`. Also covers `check_pairwise_distinct`,
which centralises the hand-rolled `_all_distinct` checks of several generator files, and
`stated_precision_string`, which centralises `p10e_exact_approx_repeated_calc_build.py`'s former local
`_stated_precision_string`.
"""

from fractions import Fraction

import pytest
from sympy import Rational, Symbol, sympify

from mathgen.grading.typed import grade_number
from mathgen.core import (
    Question,
    VerificationError,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    exact_decimal_string,
    latex_answer,
    latex_prompt,
    pairs_equal,
    stated_precision_string,
    stated_precision_tolerance,
    symbolically_equal,
)


# --- symbolically_equal / pairs_equal ---------------------------------------


class TestSymbolicallyEqualScopingFix:
    """The real defect: `_probe_differs` sympified into its own locals and
    `symbolically_equal` went on to use the original, un-sympified strings, so `simplify(str - str)` raised."""

    def test_plain_equal_number_strings_no_longer_raise(self):
        assert symbolically_equal("18.4", "18.4") is True

    def test_plain_different_number_strings(self):
        assert symbolically_equal("18.4", "18.5") is False

    def test_pair_strings_no_longer_raise(self):
        assert symbolically_equal("(18.4, 2.5)", "(18.4, 2.5)") is True

    def test_pair_strings_different(self):
        assert symbolically_equal("(18.4, 2.5)", "(18.4, 2.6)") is False

    def test_integers_still_work(self):
        assert symbolically_equal("3", "3") is True
        assert symbolically_equal("3", "4") is False

    def test_symbolic_expressions_still_work(self):
        assert symbolically_equal("x**2 - 1", "(x - 1)*(x + 1)") is True
        assert symbolically_equal("x + 1", "x + 2") is False

    def test_sympy_expr_inputs_still_work(self):
        x = Symbol("x")
        assert symbolically_equal(x**2 - 1, (x - 1) * (x + 1)) is True


class TestPairsEqual:
    def test_equal_string_pairs(self):
        assert pairs_equal("(18.4, 2.5)", "(18.4, 2.5)") is True

    def test_different_string_pairs(self):
        assert pairs_equal("(18.4, 2.5)", "(18.4, 2.6)") is False

    def test_equal_tuples(self):
        assert pairs_equal((1, 2), (1, 2)) is True

    def test_reordered_pair_is_not_equal(self):
        # order matters for an ordered (x, y)-shaped answer.
        assert pairs_equal((1, 2), (2, 1)) is False

    def test_different_length_is_not_equal_not_an_exception(self):
        assert pairs_equal((1, 2, 3), (1, 2)) is False

    def test_sequence_vs_non_sequence_is_not_equal_not_an_exception(self):
        assert pairs_equal((1, 2), "18.4") is False
        assert pairs_equal("18.4", (1, 2)) is False

    def test_equivalent_but_differently_spelled_elements(self):
        # each element compared by value, matching symbolically_equal.
        assert pairs_equal(("1/2", "3"), (Rational(1, 2), 3)) is True

    def test_tolerance_path(self):
        assert pairs_equal((1.001, 2.0), (1.0, 2.0), tol=0.01) is True
        assert pairs_equal((1.1, 2.0), (1.0, 2.0), tol=0.01) is False

    def test_non_sympifiable_element_is_not_equal_not_an_exception(self):
        assert pairs_equal(("12 cm", 2), ("12 cm", 2)) is False


class TestCheckDistinctNoLongerCrashesOnPairs:
    """This is the shape of defect to guard against: a distractor equal
    to its own answer used to escape `check_distinct` because the check
    itself raised (or, for scalar strings, also raised) instead of
    detecting the collision."""

    def test_equal_pair_distractor_is_caught(self):
        with pytest.raises(Exception):
            check_distinct("(18.4, 2.5)", "(18.4, 2.5)")

    def test_equal_number_string_distractor_is_caught(self):
        with pytest.raises(Exception):
            check_distinct("18.4", "18.4")

    def test_genuinely_different_pair_distractor_passes(self):
        check_distinct("(18.4, 2.5)", "(18.4, 2.6)")

    def test_check_equivalent_still_works_for_pairs(self):
        check_equivalent("(18.4, 2.5)", "(18.4, 2.5)")


# --- check_pairwise_distinct -------------------------------------------------


class TestCheckPairwiseDistinct:
    """`check_pairwise_distinct` centralises the `_all_distinct` hand-rolled, near-verbatim, across six generator
    files -- full pairwise `n choose 2` comparison, not `check_distinct`'s `n - 1` comparison against a single
    privileged answer."""

    def test_all_distinct_values_pass(self):
        check_pairwise_distinct(1, 2, 3, 4)

    def test_two_distractors_equal_each_other_but_not_the_answer_is_caught(self):
        # The motivating case: check_distinct(answer, d1, d2) would pass here
        # because neither d1 nor d2 equals the answer -- only check_pairwise_distinct
        # sees that d1 and d2 collide with each other.
        answer, d1, d2 = 5, 7, 7
        with pytest.raises(VerificationError):
            check_pairwise_distinct(answer, d1, d2)

    def test_check_distinct_alone_would_miss_that_case(self):
        answer, d1, d2 = 5, 7, 7
        check_distinct(answer, d1, d2)  # does not raise -- this is the gap

    def test_candidate_equal_to_answer_is_still_caught(self):
        with pytest.raises(VerificationError):
            check_pairwise_distinct(5, 5, 6)

    def test_single_value_trivially_passes(self):
        check_pairwise_distinct(5)

    def test_no_values_trivially_passes(self):
        check_pairwise_distinct()

    def test_pair_shaped_values_are_compared_by_value(self):
        check_pairwise_distinct("(1, 2)", "(1, 3)", "(2, 1)")
        with pytest.raises(VerificationError):
            check_pairwise_distinct("(1, 2)", "(1, 2)")

    def test_custom_message_is_used(self):
        with pytest.raises(VerificationError, match="custom collision message"):
            check_pairwise_distinct(3, 3, msg="custom collision message")


# --- latex_answer ------------------------------------------------------------


class TestLatexAnswer:
    def test_surd(self):
        assert latex_answer("8*sqrt(7)").replace(" ", "") == r"8\sqrt{7}"

    def test_fraction_object(self):
        assert latex_answer(Fraction(3, 4)) == r"\frac{3}{4}"

    def test_fraction_string(self):
        assert latex_answer("3/4") == r"\frac{3}{4}"

    def test_relation_survives_as_a_relation(self):
        out = latex_answer("x = 3")
        assert out == "x = 3"
        # never mangled into "x3" (no bare concatenation of the two sides).
        assert "x3" not in out.replace(" ", "")

    def test_multi_relation_list(self):
        out = latex_answer("x = 3, y = -5")
        assert "x = 3" in out
        assert "y = -5" in out

    def test_pair_renders_as_a_pair_not_a_subtraction(self):
        out = latex_answer("(2, -5)")
        assert "-5" in out
        # a subtraction would render as "2 - 5"; a pair keeps the comma.
        assert "," in out

    def test_already_latex_passed_through_unchanged(self):
        assert latex_answer(r"\frac{3}{4}") == r"\frac{3}{4}"
        assert latex_answer(r"8\sqrt{7}") == r"8\sqrt{7}"

    def test_already_plain_prefixed_passed_through_unchanged(self):
        assert latex_answer("plain:Obtuse") == "plain:Obtuse"

    def test_idempotent(self):
        once = latex_answer("Obtuse")
        assert latex_answer(once) == once

    def test_non_maths_word(self):
        assert latex_answer("Obtuse") == "plain:Obtuse"

    def test_non_maths_word_list(self):
        assert latex_answer("Monday, Tuesday") == "plain:Monday, Tuesday"

    def test_unparseable_text_falls_back_to_plain(self):
        assert latex_answer("12 cm") == "plain:12 cm"

    def test_negative_sign_preserved(self):
        assert latex_answer("-5") == "-5"

    def test_negative_fraction_sign_preserved(self):
        out = latex_answer("-1/2")
        assert out.replace(" ", "").startswith("-")
        assert r"\frac{1}{2}" in out

    def test_stated_precision_not_normalised_away(self):
        # a bare decimal is already valid LaTeX; sympifying it must not
        # collapse a trailing zero that states the answer's precision.
        assert latex_answer("18.40") == "18.40"

    def test_greek_letter_symbol_is_not_plain(self):
        assert latex_answer("theta") == r"\theta"

    def test_single_letter_variable_is_not_plain(self):
        assert latex_answer("x") == "x"

    def test_subscripted_variable_is_not_plain(self):
        assert latex_answer("x_1") == "x_{1}"

    def test_empty_string(self):
        assert latex_answer("") == "plain:"

    def test_default_evaluate_distributes_a_factored_form(self):
        # documents today's default: `Number * Add` auto-distributes.
        out = latex_answer("2*(x + 1)")
        assert out.replace(" ", "") == "2x+2"

    def test_evaluate_false_preserves_a_factored_answer_form(self):
        # `answer_form="factored"` needs the product kept intact.
        out = latex_answer("2*(x + 1)", evaluate=False).replace(" ", "")
        assert out == r"2\left(x+1\right)"

    def test_evaluate_false_preserves_a_three_factor_product(self):
        out = latex_answer("3*(x - 2)*(x + 4)", evaluate=False)
        assert "x + 4" in out or "x+4" in out.replace(" ", "")
        assert "x - 2" in out or "x-2" in out.replace(" ", "")
        # not expanded to a quadratic -- no x^2/x^{2} term appears.
        assert "x^{2}" not in out and "x^2" not in out

    # plain=True ---------------------------

    def test_plain_true_wraps_outright(self):
        assert latex_answer("Acute", plain=True) == "plain:Acute"

    def test_plain_true_idempotent_on_already_prefixed(self):
        assert latex_answer("plain:Acute", plain=True) == "plain:Acute"

    def test_plain_true_skips_sympy_class_name_collision(self):
        # Without plain=True, "Triangle"/"Circle"/"Categorical"/"Equivalent"
        # resolve to a live sympy class via sympify and produce garbage
        # LaTeX rather than a plain fallback -- this is exactly why the
        # caller must say plain=True rather than rely on auto-detection.
        assert latex_answer("Triangle", plain=True) == "plain:Triangle"
        assert latex_answer("Circle", plain=True) == "plain:Circle"
        assert latex_answer("Categorical", plain=True) == "plain:Categorical"
        assert latex_answer("Equivalent", plain=True) == "plain:Equivalent"

    def test_plain_true_skips_function_call_misparse(self):
        # "Unstable (repelling)" parses as Function("Unstable")(repelling)
        # without plain=True.
        assert latex_answer("Unstable (repelling)", plain=True) == "plain:Unstable (repelling)"

    def test_plain_true_on_a_relation_shaped_label_string(self):
        # Every clause of "left = 1, right = 2" sympifies cleanly on its own
        # (both "left" and "right" are valid symbol names), so the relation
        # branch never falls back to plain by itself -- plain=True is the
        # only way to get the right answer here.
        assert latex_answer("left = 1, right = 2", plain=True) == "plain:left = 1, right = 2"

    def test_plain_true_on_a_number_is_still_wrapped(self):
        # plain=True is an unconditional caller decision; it does not
        # inspect the value first.
        assert latex_answer("5", plain=True) == "plain:5"


# --- latex_prompt ------------------------------


class TestLatexPrompt:
    def test_pure_math_passthrough_byte_identical(self):
        # `addition`/`subtraction` are plain-number topics whose output has latex == text --
        # this must never change.
        assert latex_prompt("56 + 71") == "56 + 71"

    def test_engineering_notation_stem_lifts_the_expression_only(self):
        text = (
            "Write 3.59 × 10^8 in engineering notation (a power of "
            "ten that is a multiple of 3). Include the SI-prefix name."
        )
        out = latex_prompt(text)
        assert out != text
        assert r"\times" in out
        assert "10^{8}" in out
        # the prose survives inside \text{}, unmangled.
        assert r"\text{Write }" in out
        assert "engineering notation" in out
        assert "SI-prefix" in out  # hyphenated word left intact, not split

    def test_bare_rounding_instruction_digit_not_lifted(self):
        text = "Give your answer to 2 decimal places"
        out = latex_prompt(text)
        # the whole phrase must still read as one \text{} run so
        # test_question_latex_instructions.py can find "decimal places"
        # (and, separately, the 2) as contiguous, unbroken plain text.
        assert out == r"\text{Give your answer to 2 decimal places}"

    def test_dollar_and_percent_escaped(self):
        out = latex_prompt("The item costs $38, a discount of 80%")
        assert r"\$38" in out
        assert r"80\%" in out

    def test_idempotent(self):
        text = (
            "Write 3.59 × 10^8 in engineering notation (a power of "
            "ten that is a multiple of 3). Include the SI-prefix name."
        )
        once = latex_prompt(text)
        assert latex_prompt(once) == once
        assert latex_prompt("56 + 71") == latex_prompt(latex_prompt("56 + 71"))

    def test_hyphenated_word_left_in_text_not_split_as_subtraction(self):
        # "SI-prefix": the parser cannot tell a word-joining hyphen from a
        # minus sign here, so the whole word stays inside \text{}, exactly
        # as it read before -- never split into `SI` minus `prefix`.
        out = latex_prompt("Include the SI-prefix name")
        assert out == r"\text{Include the SI-prefix name}"

    def test_already_latex_passed_through_unchanged(self):
        assert latex_prompt(r"\text{Solve for } x") == r"\text{Solve for } x"

    def test_empty_string(self):
        assert latex_prompt("") == ""

    def test_unicode_operators_converted(self):
        out = latex_prompt("x ≤ 5 and y ≥ -3")
        assert r"\le" in out
        assert r"\ge" in out

    def test_degree_symbol_becomes_caret_circ(self):
        out = latex_prompt("Turn through 90° clockwise")
        assert r"90^\circ" in out

    def test_algebraic_caret_never_left_dangling(self):
        # `2^x`: the exponent is a variable, not a digit, so this parser
        # cannot resolve it -- the caret must fall back into \text{} (and
        # get escaped like any other special character), never survive as
        # a bare trailing `^` with no argument (a KaTeX parse error).
        out = latex_prompt("2^x = 16. Using the definition of a logarithm, find the value of x.")
        assert not out.rstrip().endswith("^")
        assert "^\\text{" not in out  # no dangling caret immediately before \text{
        assert r"\textasciicircum{}" in out

    def test_numeric_caret_exponent_still_lifts(self):
        out = latex_prompt("Simplify x^8 × x^(-1), giving your answer as a single power of x.")
        assert "x^{8}" in out
        assert "x^{-1}" in out


class TestLatexAnswerBareKeywordCorruption:
    """`sympify` parses Python's `is`/`and`/`or`/`not`/`in`/`if`/ `else` as real operators, so an un-mathsy
    answer containing one of them as a bare word is not merely mis-rendered -- it is silently replaced ("a and
    b" -> "b", "10 is even" -> "\\text{False}"). `latex_answer` must detect a standalone keyword token and
    fall back to `plain:` before ever sympifying."""

    # --- defect rows: must not corrupt --------------------------------

    def test_is_even_not_replaced_with_false(self):
        out = latex_answer("10 is even")
        assert out != r"\text{False}"
        assert out == "plain:10 is even"

    def test_is_odd_not_replaced_with_false(self):
        out = latex_answer("7 is odd")
        assert out != r"\text{False}"
        assert out == "plain:7 is odd"

    def test_is_not_zero_not_replaced_with_true(self):
        out = latex_answer("x is not zero")
        assert out != r"\text{True}"
        assert out == "plain:x is not zero"

    def test_and_does_not_silently_drop_first_term(self):
        out = latex_answer("a and b")
        assert out != "b"
        assert out == "plain:a and b"

    def test_or_does_not_silently_drop_second_term(self):
        out = latex_answer("p or q")
        assert out != "p"
        assert out == "plain:p or q"

    def test_not_does_not_become_false(self):
        out = latex_answer("not p")
        assert out != r"\text{False}"
        assert out == "plain:not p"

    # --- preservation rows: must be unchanged --------------------------

    def test_surd_unchanged(self):
        assert latex_answer("8*sqrt(7)").replace(" ", "") == r"8\sqrt{7}"

    def test_fraction_string_unchanged(self):
        assert latex_answer("3/4") == r"\frac{3}{4}"

    def test_relation_unchanged(self):
        assert latex_answer("x = 3") == "x = 3"

    def test_pair_unchanged(self):
        out = latex_answer("(2, -5)")
        assert "-5" in out
        assert "," in out

    def test_plain_word_unchanged(self):
        assert latex_answer("Obtuse") == "plain:Obtuse"

    def test_relation_with_bare_or_falls_back_to_plain_unchanged(self):
        # "or" sits inside a string the relation branch attempts to parse
        # as a single "lhs = rhs" clause ("x", "-5 or x = 5"); the rhs is
        # not valid Python (an embedded assignment cannot be an expression)
        # so the relation branch's own except already falls back to plain
        # -- this must keep doing that, not change to anything else, and
        # the new keyword check must never fire here (it would be reached
        # only via the final fallback, which this input never reaches).
        assert latex_answer("x = -5 or x = 5") == "plain:x = -5 or x = 5"

    # --- word-boundary-only: keyword as a substring, not a token -------

    @pytest.mark.parametrize(
        "word",
        [
            "distance",
            "android",
            "Normal",
            "points",
            "origin",
            "inverse",
            "notation",
            "orthogonal",
            "isosceles",
            "island",
        ],
    )
    def test_keyword_substring_is_not_mistaken_for_a_token(self, word):
        # each of these contains "is"/"and"/"or"/"not"/"in" as a substring
        # but never as a standalone word; must render exactly as it did
        # before the keyword check existed (a plain-word fallback today).
        assert latex_answer(word) == f"plain:{word}"


# --- stated_precision_tolerance ---------------------------------------------


class TestStatedPrecisionTolerance:
    def test_zero_dp(self):
        assert stated_precision_tolerance(0) == Fraction(1, 2)

    def test_one_dp(self):
        assert stated_precision_tolerance(1) == Fraction(1, 20)

    def test_two_dp(self):
        assert stated_precision_tolerance(2) == Fraction(1, 200)

    def test_negative_dp_rejected(self):
        with pytest.raises(ValueError):
            stated_precision_tolerance(-1)

    def test_not_the_spoken_transcript_tolerance(self):
        from mathgen.core import GRADER_TOLERANCE

        assert stated_precision_tolerance(2) != GRADER_TOLERANCE


# --- stated_precision_string -------------------------------------------------


class TestStatedPrecisionString:
    """Display `value` at EXACTLY `d` decimal
    places, never fewer -- unlike `exact_decimal_string`, the shortest
    terminating string within a maximum, which would collapse `0.10` to
    `0.1`."""

    def test_under_displaying_value_keeps_its_trailing_zero(self):
        assert stated_precision_string(Fraction(1, 10), 2) == "0.10"

    def test_exact_decimal_string_collapses_the_same_value(self):
        # The hazard this helper exists to avoid: exact_decimal_string
        # returns the SHORTEST terminating string, not a fixed width, so it
        # renders the very same value one digit shorter.
        assert exact_decimal_string(Fraction(1, 10), 2) == "0.1"

    def test_value_already_at_full_width_is_unchanged(self):
        assert stated_precision_string(Fraction(345, 100), 2) == "3.45"

    def test_negative_value(self):
        assert stated_precision_string(Fraction(-305, 100), 2) == "-3.05"

    def test_zero_places_gives_a_bare_integer(self):
        assert stated_precision_string(Fraction(7), 0) == "7"

    def test_off_grid_value_raises(self):
        # 1/3 does not terminate at 2 decimal places at all -- this is a
        # signal the caller constructed a value it should not have, not an
        # expected runtime path to catch and continue past (same contract
        # as exact_decimal_string's own ValueError, but this helper raises
        # the project's VerificationError via `check` instead).
        with pytest.raises(VerificationError):
            stated_precision_string(Fraction(1, 3), 2)


class TestStatedPrecisionStringGradingHazard:
    """A canonical answer built through `exact_decimal_string` at a stated precision it under-displays is
    therefore graded with a tolerance ten times too loose -- proved RED here first, then GREEN once the
    canonical is built through `stated_precision_string` instead.
    """

    @staticmethod
    def _question(answer_text: str) -> Question:
        return Question(
            topic="stated_precision_hazard_probe",
            level=1,
            seed=0,
            text="Work out the discrepancy. Give your answer to 2 decimal places.",
            latex="plain:Work out the discrepancy. Give your answer to 2 decimal places.",
            answer=0.1,
            answer_text=answer_text,
            answer_latex=answer_text,
        )

    def test_exact_decimal_string_canonical_false_accepts_a_full_rounding_unit_away(self):
        # 0.10 collapses to "0.1" -- one displayed decimal place, so
        # grade_number derives a 0.05 tolerance, ten times looser than the
        # 0.005 a genuinely-2dp-stated answer should get.
        canonical = exact_decimal_string(Fraction(1, 10), 2)
        assert canonical == "0.1"
        question = self._question(canonical)
        # "0.11" is a full rounding unit away at the 2dp the question
        # states, but sits inside the too-loose 0.05 tolerance derived from
        # "0.1"'s single displayed place. Proved RED: this is the hazard,
        # reproduced against the real grader, not merely asserted.
        result = grade_number(question, "0.11")
        assert result.correct, (
            "expected the pre-fix hazard to reproduce: an exact_decimal_string "
            "canonical that under-displays its stated precision should "
            "false-accept a value a full rounding unit away"
        )

    def test_stated_precision_string_canonical_rejects_the_same_wrong_value(self):
        canonical = stated_precision_string(Fraction(1, 10), 2)
        assert canonical == "0.10"
        question = self._question(canonical)
        result = grade_number(question, "0.11")
        assert not result.correct, (
            "stated_precision_string's fixed-width display should derive the "
            "correct, tighter 2dp tolerance and reject a value a full "
            "rounding unit away"
        )

    def test_stated_precision_string_canonical_still_accepts_itself(self):
        canonical = stated_precision_string(Fraction(1, 10), 2)
        question = self._question(canonical)
        assert grade_number(question, canonical).correct
