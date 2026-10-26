"""Round 4 typed-grader fixes (review D, C4): displayed keys, hedges, labels, LaTeX, grouping, ln|u|."""

from functools import lru_cache

import pytest

from mathgen import engine, service
from mathgen.core import display_question, mcq_options
from mathgen.grading import grade as grade_kind


@lru_cache(maxsize=1)
def _skills():
    return {s.id: s for s in engine.load_skills()[1]}


def _grade(skill_id, source, seed, reply):
    skill = _skills()[skill_id]
    return engine.grade(skill, engine.make_for_skill(skill, seed, source=source), reply).correct


def _accepts(skill_id, source, seed, *replies):
    for reply in replies:
        assert _grade(skill_id, source, seed, reply), (skill_id, reply)


def _refuses(skill_id, source, seed, *replies):
    for reply in replies:
        assert not _grade(skill_id, source, seed, reply), (skill_id, reply)


# -- D3: the displayed key is accepted ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id, source, seed",
    [
        ("senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm", 0, 11),
        ("senior-convert-between-cartesian-and-polar-form", 0, 5),
        ("senior-determine-and-examine-the-n-th-roots-of-acmsm088", 0, 5),
        ("prism-capacity-three-units", 0, 5),
        ("uc2-curl-of-field", 1, 5),
    ],
)
def test_the_displayed_key_is_accepted(skill_id, source, seed):
    skill = _skills()[skill_id]
    question = engine.make_for_skill(skill, seed, source=source)
    shown = display_question(question)["answer_text"]
    assert engine.grade(skill, question, shown).correct
    assert engine.grade(skill, question, question.answer_text).correct


def test_pair_labels_fold_greek_and_superscripts_but_keep_their_values():
    _accepts("senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm", 0, 11, "R = 65, α = 14.3", "α = 14.3, R = 65")
    _refuses("senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm", 0, 11, "R = 14.3, α = 65", "R = 65, α = 15.3")
    _accepts("prism-capacity-three-units", 0, 5, "m³ = 16.8, cm³ = 16800000, L = 16800")
    _refuses("prism-capacity-three-units", 0, 5, "m³ = 16800, cm³ = 16.8, L = 16800")


def test_every_typed_key_is_accepted_as_displayed():
    """The library-wide sweep: grade(display_question(q)["answer_text"]) is correct for every typed skill."""
    skip = {
        "mcq", "open_response", "step_chain", "grid_cells", "bars", "polygon", "number_line_set", "line_graph",
        "points", "half_planes", "curve", "matrix", "table",
    }  # fmt: skip
    failures = []
    for skill in _skills().values():
        if skill.answer in skip:
            continue
        for source in range(len(skill.sources)):
            for seed in (5, 11):
                question = engine.make_for_skill(skill, seed, source=source)
                if getattr(question, "parts", None):
                    continue
                shown = display_question(question)["answer_text"]
                if not engine.grade(skill, question, shown).correct:
                    failures.append((skill.id, source, seed, question.answer_text, shown))
    assert not failures, failures[:10]


# -- D4: MCQ option text as displayed -------------------------------------------------------------------------


def test_mcq_option_text_as_displayed_is_accepted():
    skill_id = "senior-understand-the-notion-of-independence-of-an-event"
    skill = _skills()[skill_id]
    checked = 0
    for source in range(len(skill.sources)):
        question = engine.make_for_skill(skill, 11, source=source)
        stored, shown = mcq_options(question), display_question(question)["options"]
        if stored == shown:
            continue
        right = stored.index(question.answer_text)
        wrong = (right + 1) % len(stored)
        assert engine.grade(skill, question, shown[right]).correct
        assert not engine.grade(skill, question, shown[wrong]).correct
        checked += 1
    assert checked


def test_service_matches_displayed_option_text():
    skill = _skills()["senior-understand-the-notion-of-independence-of-an-event"]
    question = engine.make_for_skill(skill, 11, source=0)
    shown = display_question(question)["options"]
    stored = mcq_options(question)
    assert shown != stored
    index = stored.index(question.answer_text)
    assert service._option_reply(question, shown[index]) == index
    assert service._option_reply(question, stored[index]) == index


# -- D5: a lone i, j or k is no unit of a real answer ---------------------------------------------------------


def test_lone_imaginary_unit_is_refused_on_a_real_key():
    _accepts("complex-imaginary-part", 0, 0, "-4", "z = -4")
    _refuses("complex-imaginary-part", 0, 0, "-4i", "-4j", "i-4")
    _accepts("senior-consider-complex-numbers-as-points-in-a-plane", 0, 0, "x = -5, y = -6", "(-5, -6)")
    _refuses("senior-consider-complex-numbers-as-points-in-a-plane", 0, 0, "x = -5, y = -6i")
    _refuses("senior-consider-conjugate-roots-for-polynomials-with-real-coefficients", 0, 0, "6i", "i6")
    _accepts("senior-consider-conjugate-roots-for-polynomials-with-real-coefficients", 0, 0, "6")


# -- D6 and scope 5: hedge words, and a false equation before the value ----------------------------------------

HEDGES = [
    "less than 9", "more than 9", "fewer than 9", "under 9", "over 9", "at least 9", "at most 9", "9 or more",
    "9 or less", "9 or fewer", "almost 9", "nearly 9", "nope 9", "nah 9", "idk 9", "dunno 9", "guessing 9",
]  # fmt: skip


@pytest.mark.parametrize("reply", HEDGES)
def test_inequality_and_doubt_words_are_hedges(reply):
    _refuses("cyclic-group-element-order", 0, 0, reply)


def test_a_no_before_a_comma_still_answers():
    _accepts("cyclic-group-element-order", 0, 0, "nope, 9", "no, it's 9")


def test_plain_replies_still_pass_and_approximation_words_are_left_alone():
    _accepts("cyclic-group-element-order", 0, 0, "9", "x = 9", "order = 9", "9.", "= 9", "area = 3 x 3 = 9")
    _accepts("cyclic-group-element-order", 0, 0, "about 9", "≈9")  # unchanged: an estimate stem may need them


def test_a_false_equation_before_the_value_is_refused():
    _refuses("cyclic-group-element-order", 0, 0, "10 = 9", "2+2 = 9", "5*2 = 9", "x = 10 = 9")
    _accepts("cyclic-group-element-order", 0, 0, "3 x 3 = 9", "4 + 5 = 9", "x = 9", "9 = 9")


# -- scope 6: a label that is the question's own, LaTeX, capital X ---------------------------------------------


def test_fraction_keys_take_the_questions_own_label():
    _accepts("f10-relative-frequency-estimate", 0, 0, "8/21", "P(green) = 8/21", "p(green)=8/21")
    _refuses("f10-relative-frequency-estimate", 0, 0, "P(red) = 8/21", "P(green) = 9/21", "x = 8/21")
    _accepts("gradient-from-graph", 1, 0, "-3/2", "m = -3/2")
    _refuses("gradient-from-graph", 1, 0, "m = -3/3", "x = -3/2")
    _accepts("discrete-expected-value", 0, 0, "3", "E[X] = 3", "E(X) = 3")
    _refuses("discrete-expected-value", 0, 0, "E[X] = 4")
    _accepts("senior-interpret-the-derivative-as-the-slope-or-gradient", 1, 0, "-3", "f'(-1) = -3", "f’(-1) = -3")
    _refuses("senior-interpret-the-derivative-as-the-slope-or-gradient", 1, 0, "f'(-2) = -3", "f'(-1) = -4")


def test_a_mismatched_function_label_is_refused():
    _accepts("function-notation", 0, 0, "-6", "f(1) = -6", "f (1) = -6")
    _refuses("function-notation", 0, 0, "f(2) = -6", "f(0) = -6", "g(1) = -6", "f(1) = -5")


def test_latex_in_expression_answers():
    _accepts("simplify-surds", 0, 7, "6\\sqrt{21}", "6 \\cdot \\sqrt{21}", "\\frac{12}{2}\\sqrt{21}")
    _refuses("simplify-surds", 0, 7, "6\\sqrt{22}", "6\\sqrt{21}x", "\\sqrt{")


def test_capital_variable_in_an_expression_answer():
    _accepts("ucv1-quotient-ring-operation", 0, 7, "-8*X", "-8x")
    _refuses("ucv1-quotient-ring-operation", 0, 7, "-8*Y", "-7*X")


# -- D10, D12: unfinished replies and unit letters of nothing -------------------------------------------------


@pytest.mark.parametrize("reply", ["9+", "9-", "(9", "== 9", "9d", "9 d", "9 A", "9 t"])
def test_unfinished_replies_and_stray_unit_letters_are_refused(reply):
    _refuses("cyclic-group-element-order", 0, 0, reply)


def test_unfinished_numeric_expression_keys_are_refused():
    _accepts("ucv1-residue-simple-pole", 0, 21, "7", "(7)")
    _refuses("ucv1-residue-simple-pole", 0, 21, "7-", "7+", "(7")


def test_unit_letters_stay_where_the_question_or_key_shows_them():
    _refuses("right-triangle-angle", 0, 0, "31 A", "31 t")
    _accepts("right-triangle-angle", 0, 0, "31.0", "31.0°", "θ = 31.0°")
    _refuses("bending-moment-point-load", 0, 0, "90 A")
    _accepts("bending-moment-point-load", 0, 0, "90", "M = 90 kN·m")


# -- D11: grouped thousands on pair and fraction keys ----------------------------------------------------------


def test_grouped_thousands_on_pair_and_fraction_keys():
    _accepts("prism-capacity-three-units", 0, 21, "m3 = 20.3, cm3 = 20 300 000, L = 20 300")
    _accepts("prism-capacity-three-units", 0, 21, "m3 = 20.3, cm3 = 20,300,000, L = 20,300")
    _refuses("prism-capacity-three-units", 0, 21, "m3 = 20.3, cm3 = 20 300 001, L = 20 300")
    _accepts("ucv1-taylor-remainder-bound", 0, 21, "1/122 880", "1/122,880", "1/122880")
    _refuses("ucv1-taylor-remainder-bound", 0, 21, "1/122 881", "1/12 288")


# -- D17: a percent on a dollar answer ------------------------------------------------------------------------


def test_percent_on_a_dollar_answer_is_refused_even_when_the_stem_quotes_a_percent():
    _accepts("f10-rational-percentage-financial-model", 2, 11, "1640.00", "$1640.00", "1640")
    _refuses("f10-rational-percentage-financial-model", 2, 11, "1640.00%", "1640%")


def test_percentile_is_no_percentage():
    _accepts("ucv1-bootstrap-summary", 0, 11, "mean = 253/10, p10 = 3")
    _refuses("ucv1-bootstrap-summary", 0, 11, "mean = 253/10, p10 = 3%")


# -- scope 9: form is enforced where the stem names it --------------------------------------------------------


def test_stated_precision_is_enforced_where_the_stem_names_it():
    skill_id = "f10-round-decimals-accuracy"
    skill = _skills()[skill_id]
    question = next(
        q
        for q in (engine.make_for_skill(skill, seed, source=2) for seed in range(40))
        if "nearest tenth" in q.text and q.answer_text.endswith(".0")
    )
    whole = question.answer_text[:-2]
    assert engine.grade(skill, question, question.answer_text).correct
    assert not engine.grade(skill, question, whole).correct


def test_form_is_not_enforced_where_the_stem_does_not_name_it():
    # "20% of a whole is what fraction": no "simplest form" in the stem, so 2/10 and 20/100 are the same value.
    skill = _skills()["f10-percentage-equivalence-comparison"]
    question = next(
        q for q in (engine.make_for_skill(skill, seed, source=0) for seed in range(40)) if q.answer_text == "1/5"
    )
    for reply in ("1/5", "2/10", "20/100"):
        assert engine.grade(skill, question, reply).correct


# -- C4: ln|u| where u > 0 ------------------------------------------------------------------------------------


def test_log_bars_may_be_dropped_or_added_only_where_the_argument_is_positive():
    _accepts("integrate-f-prime-over-f", 0, 0, "ln(x^2 + 6) + C", "ln|x^2 + 6| + C", "ln(abs(x^2+6))+C",
             "ln(\\|x^2+6\\|) + C")  # fmt: skip
    _refuses("integrate-f-prime-over-f", 0, 0, "ln|x^2 + 7| + C", "ln|x^2 + 6|")
    _accepts("integrate-f-prime-over-f", 1, 1, "3 ln(cos(x) + 4) + C", "3 ln|cos(x) + 4| + C")
    _refuses("integrate-f-prime-over-f", 1, 1, "3 ln|cos(x) + 5| + C")


def test_log_bars_with_a_stem_domain_and_the_log_power_law():
    _accepts("uc2-homogeneous-substitution", 0, 3, "-2x/(4ln(x) + 1)", "-2x/(4ln|x|+1)", "-2x/(1+ln(x^4))")
    _refuses("uc2-homogeneous-substitution", 0, 3, "-2x/(1+ln(x^3))")


def test_dropping_the_bars_where_the_argument_can_be_negative_stays_wrong():
    _accepts("integrate-reciprocal-linear", 0, 0, "-9 ln|x| + C", "-9*ln(abs(x)) + C")
    _refuses("integrate-reciprocal-linear", 0, 0, "-9 ln(x) + C", "-9 ln|x|")


# -- A3: stem-and-leaf notation --------------------------------------------------------------------------------


def test_stem_and_leaf_plot_is_shown_and_read_in_standard_notation():
    skill = _skills()["construct-stem-and-leaf-plots"]
    question = engine.make_for_skill(skill, 4, source=0)
    assert question.answer_text == "2 | 3 7 8 9, 3 | 1 4 8 9, 4 | 0 3, 5 | 0 3 8"
    assert "written like 2 | x y z" in question.text
    assert engine.grade(skill, question, display_question(question)["answer_text"]).correct
    _accepts("construct-stem-and-leaf-plots", 0, 4,
             "2 | 3 7 8 9, 3 | 1 4 8 9, 4 | 0 3, 5 | 0 3 8",
             "2: 3 7 8 9, 3: 1 4 8 9, 4: 0 3, 5: 0 3 8",
             "2 | 3 7 8 9;\n3 | 1 4 8 9\n4 | 0 3, 5 | 0 3 8.",
             "Key: 2 | 3 means 23\n2 | 3 7 8 9, 3 | 1 4 8 9, 4 | 0 3, 5 | 0 3 8",
             "5 | 0 3 8, 4 | 0 3, 3 | 1 4 8 9, 2 | 3 7 8 9",
             "2 Leaf = 3 7 8 9, 3 Leaf = 1 4 8 9, 4 Leaf = 0 3, 5 Leaf = 0 3 8",
             "3 7 8 9, 1 4 8 9, 0 3, 0 3 8")  # fmt: skip
    _refuses("construct-stem-and-leaf-plots", 0, 4,
             "2 | 3 7 8 9, 3 | 1 4 8 9, 4 | 0 3, 5 | 0 3 9",
             "2 | 3 7 8 9, 3 | 1 4 8 9, 4 | 0 3",
             "2 | 3 7 8 9, 2 | 3 7 8 9, 4 | 0 3, 5 | 0 3 8",
             "2 | 3 7 8 9, 3 | 1 4 8 9, 4 | 0 3, 6 | 0 3 8")  # fmt: skip


def test_a_full_stop_closes_a_table_reply_but_a_percent_or_bang_is_not_in_the_key():
    skill = _skills()["uc2-group-table-check"]
    question = engine.make_for_skill(skill, 11, source=0)
    shown = question.answer_text
    for reply in (shown, shown + "."):
        assert engine.grade(skill, question, reply).correct, reply
    for reply in (shown + "%", shown + "!"):
        assert not engine.grade(skill, question, reply).correct, reply


# -- D17: a percent sign the key lacks is refused in a pair --------------------------------------------------


def test_a_pair_reply_with_a_percent_sign_the_key_lacks_is_refused():
    _accepts("ucv1-bootstrap-summary", 0, 11, "mean = 253/10, p10 = 3", "p10 = 3, mean = 25.3")
    _refuses(
        "ucv1-bootstrap-summary", 0, 11, "mean = 25.3%, p10 = 3", "p10 = 3%, mean = 25.3", "mean = 253/10%, p10 = 3"
    )


# -- A16: a mixed number for a pair component whose key is a fraction -------------------------------------


def test_a_mixed_number_is_one_component_of_a_pair_when_the_key_is_a_fraction():
    _accepts(
        "graph-f-squared-features", 2, 3, "(-7/3, 1)", "(-2 1/3, 1)", "-2 1/3, 1", "[-2 1/3, 1]", "x = -2 1/3, y = 1"
    )
    _refuses("graph-f-squared-features", 2, 3, "(-2 2/3, 1)", "(2 1/3, 1)", "(1, -2 1/3)", "(-2 1/3, 1/3)", "(-2 1/3)")


# -- list grader: pi multiples, duplicates, stray symbols -----------------------------------------------------


def _synthetic(kind, answer, answer_text, text):
    from types import SimpleNamespace

    return SimpleNamespace(
        answer=answer, answer_text=answer_text, answer_kind=kind, parts=None, text=text, distractors=[]
    )


def test_a_list_of_pi_multiples_reads_pi_or_the_symbol_in_any_spelling():
    question = _synthetic(
        "list", ["pi/4", "3pi/4"], "π/4, 3π/4", "Solve sin(x) = √2/2 for 0 < x < π. Give exact values."
    )
    for good in ("π/4, 3π/4", "3π/4, π/4", "pi/4, 3pi/4", "3*pi/4, pi/4", "(3/4)π, π/4"):
        assert grade_kind("list", question, good).correct, good
    for bad in ("π/4", "π/4, 3π/4, 3π/4", "π/4, 3π/5", "0.785, 2.356", "π/4, π/4", "π/4, 3π"):
        assert not grade_kind("list", question, bad).correct, bad


def test_a_repeated_value_is_refused_unless_the_key_can_repeat():
    _accepts("spot-bias", 0, 4, "3")
    _refuses("spot-bias", 0, 4, "3, 3")
    _accepts("finite-set-operations", 0, 1, "{3}")
    _refuses("finite-set-operations", 0, 1, "{3, 3}", "3, 3")
    # a quadratic's double root typed twice is still read as the one root (tests/test_bh2_grade2.py)
    _accepts("solve-quadratics-factorable", 1, 15, "x = -9", "-9, -9", "x = -9 or x = -9")
    _refuses("solve-quadratics-factorable", 1, 15, "-9, -8")
    question = _synthetic("list", [3, 3, 5], "3, 3, 5", "List the mode and the values.")
    assert grade_kind("list", question, "3, 3, 5").correct
    assert not grade_kind("list", question, "3, 5, 5").correct


def test_a_list_reply_with_trailing_junk_is_refused():
    from fractions import Fraction

    question = _synthetic(
        "list", [Fraction(3, 4), Fraction(5, 4)], "3/4, 5/4", "Solve 16x^2 - 32x + 15 = 0 for x. Give the solutions."
    )
    for good in ("3/4, 5/4", "5/4, 3/4", "3/4 and 5/4"):
        assert grade_kind("list", question, good).correct, good
    for bad in ("3/4_x, 5/4", "3/4 _x, 5/4", "3/4_, 5/4", "3/4, 5/4#", "3/4, 5/4~"):
        assert not grade_kind("list", question, bad).correct, bad
    metres = _synthetic("list", [12, 15], "12 m, 15 m", "Find the two lengths.")
    assert grade_kind("list", metres, "12 m, 15 m").correct


# -- grader items 1-3: ratio form, spaced numerals, tonnes ------------------------------------------------


def test_a_ratio_key_does_not_accept_a_bare_decimal_but_keeps_the_fraction():
    skill = _skills()["ratio-rate-proportion"]
    question = engine.make_for_skill(skill, 1, source=0)
    assert question.answer_text == "1:2"
    for good in ("1:2", "1/2"):
        assert engine.grade(skill, question, good).correct, good
    for bad in ("0.5", ".5", "2:4"):
        assert not engine.grade(skill, question, bad).correct, bad
    # a stem that asks for a decimal keeps its decimal
    from mathgen.grading.typed import _ratio_form_is_acceptable

    assert _ratio_form_is_acceptable("12:5", "12/5")
    assert not _ratio_form_is_acceptable("12:5", "2.4", "Simplify the ratio 96:40.")
    assert _ratio_form_is_acceptable("12:5", "2.4", "Write the ratio 96:40 as a decimal.")


def test_a_tonne_symbol_is_a_unit_only_where_the_question_names_tonnes():
    tonnes = _synthetic("number", 5, "5", "Convert 5000 kg to tonnes.")
    for good in ("5", "5 t", "5 tonnes"):
        assert grade_kind("number", tonnes, good).correct, good
    assert not grade_kind("number", tonnes, "5 kg").correct
    kilograms = _synthetic("number", 5000, "5000", "A lorry weighs 5 t. Write this in kilograms.")
    assert grade_kind("number", kilograms, "5000 kg").correct
    assert not grade_kind("number", kilograms, "5000 t").correct
    assert not grade_kind("number", _synthetic("number", 5, "5", "Convert 5000 g to kg."), "5 t").correct


# -- A14: a space-separated list of single digits or letters ---------------------------------------------


def test_single_character_outcomes_may_be_separated_by_spaces():
    _accepts("ey-list-possible-outcomes", 1, 1, "1 2 3", "3 2 1", "{1 2 3}", "1, 2, 3")
    _refuses("ey-list-possible-outcomes", 1, 1, "1 2", "1 2 3 3", "1 2 4", "123")
    _accepts("ey-list-possible-outcomes", 2, 3, "A B M O T W", "a b m o t w")
    _refuses("ey-list-possible-outcomes", 2, 3, "A B M O T")
    # not for longer items: "red green" is not two outcomes
    _refuses("ey-list-possible-outcomes", 0, 2, "orange yellow")


# -- review of PR 147: labelled sets, currency prefixes, tonnes and degrees, letter case --------------------


def test_a_labelled_set_with_trailing_punctuation_is_read_as_the_set():
    _accepts(
        "uc2-units-and-zero-divisors-zn",
        0,
        0,
        "{0, 10, 20, 30, 40}",
        "x = {0, 10, 20, 30, 40}",
        "A = {0, 10, 20, 30, 40}",
        "x ∈ {0, 10, 20, 30, 40}",
        "S = {0, 10, 20, 30, 40}",
        "the set is {0, 10, 20, 30, 40}",
        "{0, 10, 20, 30, 40},",
    )
    _refuses(
        "uc2-units-and-zero-divisors-zn",
        0,
        0,
        "{0, 10, 20, 30",
        "x = {0, 10, 20, 30",
        "x = {0, 10, 20, 30, 40,",
        "{0, 10, 20, 30, 40+",
    )
    _accepts(
        "uc2-permutation-cycle-notation",
        0,
        0,
        "x = {(1, 5, 2, 4), (3, 6)}",
        "x = (1, 5, 2, 4), (3, 6)",
        "x = (1,5,2,4),(3,6)",
        "{(1, 5, 2, 4), (3, 6)},",
    )
    _refuses("uc2-permutation-cycle-notation", 0, 0, "x = (1, 5, 2, 4), (3, 6", "x = (1, 5, 2, 4), (3, 7)")
    _accepts("f10-partition-one-two-digit-numbers", 0, 0, "x = {(3, 3), (2, 13), (1, 23), (0, 33)}")


def test_only_the_currency_prefixes_the_library_uses_pass_before_a_dollar_sign():
    skill_id = "senior-use-currency-exchange-rates-to-determine-the-cost"
    _accepts(skill_id, 0, 0, "A$138.00", "C$138.00", "AU$138.00", "$138.00", "138")
    _refuses(skill_id, 0, 0, "a$138.00", "Q$138.00", "x$138.00", "z$138.00")


def test_degrees_are_no_mass_even_where_the_stem_asks_for_tonnes():
    tonnes = _synthetic("number", 5, "5", "Convert 5000 kg to tonnes.")
    for good in ("5", "5 t", "5 tonnes"):
        assert grade_kind("number", tonnes, good).correct, good
    for bad in ("5°", "5 °", "5 degrees", "5 radians"):
        assert not grade_kind("number", tonnes, bad).correct, bad
    skill = _skills()["senior-estimate-the-mass-of-different-objects"]
    question = engine.make_for_skill(skill, 0, source=3)
    for bad in ("0.297°", "0.297 °", "0.297 degrees"):
        assert not engine.grade(skill, question, bad).correct, bad


def test_a_reply_may_write_a_canonical_letter_in_either_case_but_unit_letters_stay_refused():
    _accepts("build-sample-space", 0, 0, "{(H, 1), (H, 2), (H, 3), (T, 1), (T, 2), (T, 3)}")
    _accepts("build-sample-space", 0, 0, "{(h, 1), (h, 2), (h, 3), (t, 1), (t, 2), (t, 3)}")
    _refuses("build-sample-space", 0, 0, "{(h, 1), (h, 2), (h, 3), (t, 1), (t, 2), (t, 4)}")
    group = _synthetic("number", 9, "9", "What is the order of the group?")
    for bad in ("9 t", "9 T", "9 A", "9 d"):
        assert not grade_kind("number", group, bad).correct, bad


def test_a_function_letter_may_be_written_in_either_case_but_not_another_call_or_letter():
    for skill_id, source, seed in (("function-notation", 0, 0),):
        _accepts(skill_id, source, seed, "f(1) = -6", "F(1) = -6", "F (1) = -6", "-6")
        _refuses(skill_id, source, seed, "f(2) = -6", "F(2) = -6", "g(1) = -6", "G(1) = -6")


# -- re-review of PR 147: unit case after a joiner, capitals the question uses ------------------------------


def test_a_capital_after_a_unit_joiner_is_no_label_letter():
    _accepts("bending-moment-point-load", 0, 0, "M = 90 kN·m", "90 kN·m")
    _refuses("bending-moment-point-load", 0, 0, "M = 90 kN·M", "90 kN·M")
    _accepts("moment-of-a-force", 0, 0, "M = 10.0 N·m")
    _refuses("moment-of-a-force", 0, 0, "M = 10.0 N·M")
    _accepts("max-bending-moment-udl", 0, 0, "M_max = 144 kN·m")
    _refuses("max-bending-moment-udl", 0, 0, "M_max = 144 kN·M")


def test_a_capital_for_a_lower_case_symbol_is_refused_where_the_question_uses_the_capital():
    _accepts("vector-proof-geometry", 0, 0, "-a + b")
    _refuses("vector-proof-geometry", 0, 0, "-A + B", "-A + b")
    _refuses("uc2-conditional-density-expectation", 0, 0, "1/(1 - Y)")
    _refuses("uc2-transformation-of-random-variable", 0, 0, "-Y^3/64 + 9*Y^2/64 - 27*Y/64 + 27/64")
    _refuses("newton-cooling-shifted-exponential", 0, 0, "28 - 24e^(-3T/20)")


def test_a_capital_mixed_with_the_questions_own_lower_case_letters_is_refused():
    _accepts("rearrange-simple-formulas", 1, 1, "(-b + y)/a", "(y - b)/a")
    _refuses("rearrange-simple-formulas", 1, 1, "(-b + y)/A")
    _accepts("rearrange-simple-formulas", 2, 1, "(E - m*v**2/2)/(h*m)")
    _refuses("rearrange-simple-formulas", 2, 1, "(E - m*V**2/2)/(h*m)", "(E - m*v**2/2)/(H*m)")
    _accepts("ucv1-quotient-ring-operation", 0, 7, "-8*X", "-8x")
