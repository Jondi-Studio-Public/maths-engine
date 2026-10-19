"""Regression tests for the round-2 generator and skill-source text defects (bug hunt c3 and c4, lane genb).

Each test renders a skill through the engine and the learner display pass, then asserts on the displayed text
over several seeds and every tier: working that states something false, template splices, ASCII leftovers,
the letter x used as a times sign, redundant brackets, and money written in two styles.
"""

import re
from functools import lru_cache

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import engine
from mathgen.core import display_question

SEEDS = range(24)


@lru_cache(maxsize=1)
def _skills():
    _errors, skills = engine.load_skills()
    return {s.id: s for s in skills}


def _questions(skill_id, sources=None, seeds=SEEDS):
    skill = _skills()[skill_id]
    for index, source in enumerate(skill.sources):
        if sources is not None and index not in sources:
            continue
        for seed in seeds:
            question = engine.make_for_skill(skill, seed, source=index)
            yield index, seed, question, display_question(question)


def _lines(skill_id, sources=None, seeds=SEEDS):
    """Every learner-visible line of every question: (source, seed, kind, text)."""
    for index, seed, _question, shown in _questions(skill_id, sources, seeds):
        yield index, seed, "text", shown["text"]
        yield index, seed, "key", shown["answer_text"]
        for line in shown["working"]:
            yield index, seed, "working", line
        for part in shown.get("parts") or []:
            yield index, seed, "part text", part["text"]
            for line in part["working"]:
                yield index, seed, "part working", line


def _forbid(skill_id, pattern, sources=None, kinds=None, seeds=SEEDS):
    regex = re.compile(pattern)
    hits = [
        (index, seed, kind, text)
        for index, seed, kind, text in _lines(skill_id, sources, seeds)
        if (kinds is None or kind in kinds) and regex.search(text)
    ]
    assert not hits, hits[:3]


# ---------------------------------------------------------------- working that is false or spliced


def test_truss_chain_answer_line_has_no_spliced_instruction():
    _forbid("pythagoras-truss-chain", r"Answer: Find")
    for _index, _seed, question, shown in _questions("pythagoras-truss-chain"):
        assert shown["working"][-1] == f"Answer: {question.answer_text} m."


@pytest.mark.parametrize(
    "skill_id",
    ["equation-from-table", "senior-convert-sums-mathrm-a-cos-mathrm-x-mathrm"],
)
def test_no_working_line_repeats_its_left_hand_side(skill_id):
    _forbid(skill_id, r"(\b\w+) = [^=]*= \1 =", kinds=("working", "part working"))


def test_complex_conjugate_working_shows_the_negative_radicand():
    skill_id = "senior-determine-complex-conjugate-solutions-of-real-quadratic-equations"
    x = sympy.Symbol("x")
    for index, seed, _question, shown in _questions(skill_id, sources=[0, 1]):
        equation = re.search(r"The equation (.+) = 0 has", shown["text"]).group(1)
        poly = sympy.Poly(
            parse_expr(
                equation.replace("^", "**"),
                local_dict={"x": x},
                transformations=standard_transformations + (implicit_multiplication_application,),
            ),
            x,
        )
        a, b, c = poly.all_coeffs()
        shown_radicand = int(re.search(r"√\((-?\d+)\)", shown["working"][0]).group(1))
        assert shown_radicand == b * b - 4 * a * c, (index, seed, shown["working"][0], equation)


def test_inverse_function_working_restates_the_stated_function():
    for index, seed, _question, shown in _questions(
        "senior-consider-inverses-of-one-to-one-function", sources=[0, 1, 2]
    ):
        stated = re.search(r"f\(x\) = (.+)\.$", shown["text"]).group(1)
        assert shown["working"][0].startswith(f"Let y = {stated}."), (index, seed, shown["working"][0], stated)
        assert shown["working"][1] == f"y = {stated}", (index, seed, shown["working"][1], stated)


def test_arithmetic_recursion_working_lists_every_generated_term():
    skill_id = "senior-use-recursion-to-generate-an-arithmetic-sequence"
    for index, seed, question, shown in _questions(skill_id, sources=[0, 1]):
        terms = question.answer_text.split(", ")
        steps = [line for line in shown["working"] if re.match(r"t\(\d+\) = ", line)]
        assert [line.split(" = ")[0] for line in steps] == [f"t({n + 2})" for n in range(len(terms))], (index, seed)
        assert [line.split(" = ")[-1] for line in steps] == terms, (index, seed)


def test_quadratic_variation_working_has_no_nested_fraction():
    _forbid("senior-examine-examples-of-quadratically-related-variables", r"/ \d+/\d+|÷ \d+/\d+\)?")


def test_composition_domain_working_never_cites_an_undefined_k():
    _forbid("senior-determine-when-the-composition-of-two-functions-is", r"\bk = ")


def test_cubic_double_root_working_has_no_k_in_the_easy_tier_and_no_letter_x_times():
    _forbid("f10-cubic-double-root-graph", r"k is positive", sources=[0])
    _forbid("f10-cubic-double-root-graph", r"\) x \(|\d x \(|k x \(|\(1\) ")


def test_inequality_region_working_names_the_boundary_rule():
    _forbid("f10-match-inequality-to-line", r"the line\.", kinds=("working",))


def test_plot_linear_points_working_says_which_way_the_point_is():
    _forbid("plot-linear-points", r"across|up or down", sources=[0], kinds=("working",))


def test_equation_from_graph_exam_working_brackets_fractions_and_drops_redundant_brackets():
    _forbid("equation-from-graph", r"÷ -?\d+/\d+ |= -?\d+/\d+x |\(\d+\) \+ c|- \(\d+\)\|", sources=[3])
    _forbid("equation-from-graph", r"- \(\d+\)\) /|- \(\d+\)\)", sources=[0])


def test_geometric_recursive_definition_check_brackets_a_negative_divisor():
    _forbid("senior-recognise-and-use-the-recursive-definition-of-a", r"/ -|÷ -")


def test_rate_of_change_exam_working_is_real_working():
    _forbid("rate-of-change", r"Working by the method", sources=[3])
    for _index, _seed, _question, shown in _questions("rate-of-change", sources=[3]):
        assert len(shown["working"]) >= 2


def test_linear_equation_classifier_working_is_not_an_internal_label():
    _forbid("senior-identify-and-solve-linear-equations", r"Independent check", sources=[0, 1, 2])


def test_congruence_criteria_working_has_no_internal_diagnostics():
    _forbid("f10-congruence-similarity-criteria", r"Sides stated in both|Only the answer's own", sources=[0, 1, 2])


# ---------------------------------------------------------------- zero and unit noise


def test_zero_root_is_a_factor_x_not_x_minus_zero():
    _forbid("solve-quadratics-factorable", r"\(x [-+] 0\)", seeds=range(40))


def test_quadratic_formula_standard_form_has_no_bracketed_coefficients():
    _forbid("solve-quadratics-formula", r"\(-?\d+\)x|\(-?\d+\) = 0|\+ \(-", sources=[3])


def test_one_step_equations_do_not_divide_by_one():
    _forbid("one-step-equations", r"/1 =", sources=[0])


def test_word_problems_do_not_add_zero():
    _forbid("solve-equation-word-problems", r"add 0|\+ 0\b")


def test_degree_discriminant_working_does_not_print_zero_s():
    _forbid("senior-find-turning-points-and-zeros-of-quadratics-and", r"zero\(s\)")


def test_concavity_polynomials_use_juxtaposition_and_have_no_zero_constant():
    _forbid(
        "senior-understand-the-concepts-of-concavity-and-points-of",
        r"·|\+ 0\b|\+ 0$",
        sources=[0, 1, 2],
        kinds=("text",),
    )


def test_difference_quotient_polynomials_use_juxtaposition():
    _forbid("senior-interpret-the-difference-quotient-frac-f-left-x", r"·", kinds=("text",))


def test_money_cents_working_has_no_plus_zero_or_letter_x():
    _forbid("f10-money-dollars-cents-representation", r"\+ 0\b| x ", sources=[0])


def test_recurrence_stems_do_not_write_unit_coefficient_or_zero_constant():
    _forbid("senior-recognise-that-a-sequence-generated-by-a-first", r"= 1 × u\(n\)|\+ 0[ ,.]", kinds=("text",))


def test_simplify_multi_operation_stem_gives_an_instruction():
    for index, seed, _question, shown in _questions("simplify-multi-operation-expressions", sources=[0, 1, 2]):
        assert shown["text"].startswith("Simplify "), (index, seed, shown["text"])


# ---------------------------------------------------------------- the letter x (and ASCII) as notation


@pytest.mark.parametrize(
    "skill_id",
    [
        "compare-functions-representations",
        "f10-cubic-double-root-graph",
        "f10-parabola-equation-from-graph",
        "f10-quadratic-parameter-effects",
        "senior-graph-dilation-read-factor",
        "senior-determine-if-a-function-is-one-to-one",
        "angle-degrees-to-radians-decimal",
        "angle-radians-to-degrees-decimal",
        "solve-linear-equations-graphically",
        "f10-quadrilateral-properties",
        "f10-enlargement-invariants",
        "senior-solve-practical-problems-that-involve-finding-the-point",
        "senior-solve-practical-problems-involving-the-trigonometry-of-right",
        "f10-graphical-inequality-regions",
        "f10-graphical-systems-of-lines",
        "hybrid-functions",
    ],
)
def test_letter_x_is_never_a_times_sign(skill_id):
    _forbid(
        skill_id,
        r"[\d)\]] x (?=[\d(−]|-\d)|\b[a-zA-Z] x (?=[\d(])|\d [a-z] [+=]|-\d+ x =|\bx \d|\bdeg\b|radians x|degrees x",
        seeds=range(40),
    )


def test_limiting_behaviour_text_uses_arrow_and_inequality_symbols():
    _forbid("senior-understand-the-limiting-behaviour-as-n-rightarrow-infty", r"->|<=|>=")


def test_geometric_ratio_working_has_no_float_noise():
    _forbid("senior-use-the-formula-t-n-r-n-1", r"\b\d+\.0\b", sources=[1])


def test_periodic_motion_does_not_write_a_unit_coefficient():
    _forbid("senior-model-periodic-motion-using-sine-and-cosine-functions", r"[= ]-?1(sin|cos)", kinds=("text",))
    _forbid("senior-examine-amplitude-changes-and-the-graphs-of-y", r"-1π|\b1π", kinds=("working",))


def test_power_function_working_does_not_print_a_fraction_with_a_negative_denominator():
    _forbid("power-functions-rational-index", r"/\(-|/1\b", sources=[2], kinds=("working",))


def test_inequality_working_does_not_subtract_a_negative():
    for skill_id in ("inequality-solutions-number-line", "two-step-inequalities", "inequalities-negative-mult-div"):
        _forbid(skill_id, r"[Ss]ubtract \(?-")


def test_trig_power_graph_working_does_not_capitalise_a_variable():
    skill_id = "senior-recognise-features-of-the-graphs-of-y-x-acmmm014"
    _forbid(skill_id, r"\bX\b|with n is a positive whole number")


def test_per_period_rate_is_not_written_as_a_ratio_of_a_percentage():
    _forbid("senior-use-technology-to-compare-numerically-and-graphically-the", r"\d/\d+%")


def test_locus_working_uses_the_conjugate_bar_the_stem_uses():
    _forbid("locus-circle-centre-radius", r"z-bar|c-bar")


# ---------------------------------------------------------------- brackets, articles, wording


def test_redundant_brackets_around_positive_numbers_are_gone():
    cases = {
        "linear-models": r"[-+×] \(\d+\) |\(\d+\)[a-z]\b|\(\d+\) ×",
        "senior-determine-the-slope-and-intercepts-of-a-straight": r"- \(\d+\)\)",
        "midpoint-two-points": r"\(\d+\)/2",
        "senior-use-arithmetic-sequences-in-contexts-involving-discrete-linear": r"× \(\d+\)\)",
    }
    for skill_id, pattern in cases.items():
        _forbid(skill_id, pattern, kinds=("working", "part working"))


def test_an_before_a_price_that_starts_with_a_vowel_sound():
    _forbid("senior-develop-a-linear-formula-from-a-word-description", r"\ba \$(8|11|18)\b", seeds=range(40))


def test_number_sentence_story_does_not_put_boxes_in_a_box():
    _forbid("f10-add-subtract-number-sentences", r"boxes in a box|boxes were taken out", seeds=range(60))


def test_dms_distractors_are_real_minutes():
    for index, seed, question, _shown in _questions("angle-radians-to-dms", seeds=range(40)):
        for distractor in question.distractors:
            minutes = int(re.search(r"°(\d+)′", distractor).group(1))
            assert minutes < 60, (index, seed, distractor)


def test_systems_exam_distractors_have_no_placeholder_value():
    for index, seed, question, _shown in _questions("f10-graphical-systems-of-lines", sources=[3], seeds=range(60)):
        for part in question.parts:
            assert "99" not in part.distractors, (index, seed, part.distractors)


# ---------------------------------------------------------------- c4 student text


def test_limit_estimate_working_uses_the_stated_decimal():
    skill_id = "senior-estimate-the-limit-of-frac-a-h-1"
    for index, seed, question, shown in _questions(skill_id, sources=[3]):
        stated = question.parts[0].answer_text
        working = " ".join(shown["working"])
        assert f"The limit is about {stated}" in working, (index, seed)
        assert not re.search(r"\d+/\d+ ×|1/\(\d+/\d+\)|\bL x ", working), (index, seed, working)
        assert "→ 0" in shown["parts"][0]["text"]


def test_geometric_model_ordinals_are_correct():
    skill_id = "senior-use-geometric-sequences-to-model-and-analyse-numerically"
    _forbid(
        skill_id,
        r"\b\d*(?<![1])[3]th\b|\b1th\b|\b2th\b|\b\d+(?<!1)1th\b",
        sources=[0, 1, 2],
        kinds=("text",),
        seeds=range(40),
    )
    texts = " ".join(text for _i, _s, kind, text in _lines(skill_id, sources=[0, 1], seeds=range(40)) if kind == "text")
    assert "3rd" in texts


def test_circular_table_stem_is_grammatical():
    _forbid("counting-with-restrictions", r"can \d+ people are ", seeds=range(60))


def test_money_in_one_question_uses_one_style():
    skill_id = "senior-use-technology-to-investigate-the-effect-of-the"
    for index, seed, _question, shown in _questions(skill_id, sources=[3]):
        for line in shown["working"]:
            assert not re.search(r"\$\d{4,}(?![\d,])", line), (index, seed, line)
    for skill_id in (
        "flat-loan-repayment",
        "simple-interest",
        "senior-investigate-the-effect-of-the-interest-rate-and",
        "senior-use-a-recurrence-relation-to-model-an-annuity",
        "senior-use-technology-and-a-recurrence-relation-to-model",
    ):
        _forbid(skill_id, r"\$\d{4,}(?![\d,])(?!\.\d{3})", kinds=("text", "working", "part text", "part working"))


def test_recurrence_exam_working_uses_ascii_minus_and_comma_money():
    _forbid("senior-use-technology-and-a-recurrence-relation-to-model", r"−|\$\d{4,}(?![\d,])", sources=[3])
