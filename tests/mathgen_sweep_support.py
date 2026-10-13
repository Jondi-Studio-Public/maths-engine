"""Shared by the generator sweep files `tests/test_mathgen_*.py`."""

import os

from mathgen import make

# Seeds for the text and working-text checks. Set MATHS_DEEP_SEEDS to raise them
# for a deeper run; leaving it unset keeps 60.
TEXT_SEEDS = int(os.environ.get("MATHS_DEEP_SEEDS", "60"))
WORKING_SEEDS = int(os.environ.get("MATHS_DEEP_SEEDS", "60"))


# ---------------------------------------------------------------------------
# New kinds/params introduced in this change: (topic, select-param name,
# value) triples, each fuzzed explicitly so every new variant gets its own
# coverage regardless of whether it's ever chosen by a bare `make(topic)`.
# select-param is "kinds" for topics that pick internally via
# rng.choice(list(kinds)), "kind" for topics with a direct kind= dispatch,
# and "ops" for the arithmetic op-selection topics.
NEW_KIND_VARIANTS = [
    ("linear_equation", "kind", "brackets"),
    ("linear_equation", "kind", "both_sides"),
    ("quadratic", "kind", "formula"),
    ("quadratic", "kind", "vertex"),
    ("trigonometry", "kinds", "cosine_rule"),
    ("trigonometry", "kinds", "triangle_area"),
    ("vectors", "kinds", "add"),
    ("differentiation", "kinds", "product_rule"),
    ("differentiation", "kinds", "quotient_rule"),
    ("differentiation", "kinds", "second_derivative"),
    ("differentiation", "kinds", "stationary_points"),
    ("integration", "kinds", "definite"),
    ("integration", "kinds", "substitution"),
    ("limits", "kinds", "one_sided"),
    ("limits", "kinds", "degree_comparison"),
    ("engineering", "kinds", "hookes_law"),
    ("engineering", "kinds", "youngs_modulus"),
    ("structural", "kinds", "bending_moment_point_load"),
    ("structural", "kinds", "cantilever_udl"),
    ("fractions", "ops", "/"),
    ("like_terms", "kind", "positive"),
    ("like_terms", "kind", "signed"),
    ("expand_bracket", "kind", "single"),
    ("expand_bracket", "kind", "distributive"),
    ("simplify_expression", "kind", "brackets"),
    ("simplify_expression", "kind", "multi"),
    ("expression_parts", "kind", "coefficient"),
    ("expression_parts", "kind", "constant"),
    ("expression_parts", "kind", "term_count"),
    ("gradient", "kind", "points"),
    ("gradient", "kind", "table"),
    ("gradient", "kind", "context"),
    ("linear_from_table", "kind", "equation"),
    ("linear_from_table", "kind", "intercept"),
    ("sequence", "kind", "common_diff"),
    ("sequence", "kind", "nth_term_rule"),
    ("sequence", "kind", "use_rule"),
    ("ratio", "kind", "simplify"),
    ("ratio", "kind", "share"),
    ("ratio", "kind", "unit_rate"),
    ("inequality", "kind", "one_step"),
    ("inequality", "kind", "neg_coef"),
    ("inequality", "kind", "two_step"),
    ("decimals", "ops", "/"),
    ("integer_arithmetic", "ops", "/"),
    ("rearrange", "n_steps", 1),
    ("rearrange", "n_steps", 2),
    ("rearrange", "hard", True),
    ("factor_gcf", "n_terms", 2),
    ("factor_gcf", "n_terms", 3),
    ("factor_gcf", "var_factor", True),
    ("interpret_notation", "kinds", "mult"),
    ("interpret_notation", "kinds", "square"),
    ("interpret_notation", "kinds", "distribute"),
    ("interpret_notation", "kinds", "divide"),
    ("interpret_notation", "kinds", "product"),
    ("interpret_notation", "kinds", "bracket_sq"),
    ("interpret_notation", "kinds", "coef_sq"),
    ("like_terms_select", "n_vars", 1),
    ("like_terms_select", "n_vars", 2),
    # scientific_notation_operate picks its op via rng.choice internally,
    # so a bare make() already reaches all four with enough seeds -- these
    # rows force per-op coverage regardless, matching the existing
    # ("fractions", "ops", "/") convention.
    ("scientific_notation_operate", "ops", "add"),
    ("scientific_notation_operate", "ops", "subtract"),
    ("scientific_notation_operate", "ops", "multiply"),
    ("scientific_notation_operate", "ops", "divide"),
]


def make_kind(topic, param, value, seed):
    kwargs = {param: (value,)} if param in ("kinds", "ops") else {param: value}
    return make(topic, seed=seed, **kwargs)
