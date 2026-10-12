"""Prompt-shape variety and tier escalation.

Part of the generator sweep, split across `tests/test_mathgen_*.py` so the CI
shards can balance it (`scripts/ci_shard.py` balances files by size). Verification
is built into the generators themselves (each independently re-checks its
answer with SymPy and raises VerificationError on mismatch), so constructing
thousands of questions IS the correctness test; the assertions here cover the
output contract on top of that. Set `MATHS_DEEP_SEEDS` for a deeper run of the
text checks.
"""

import pytest

from mathgen import make, topics
from mathgen.core import PRESETS, REGISTRY
from mathgen.text_quality import prompt_shape

TOPICS = [t for _lvl, t in topics()]
# Topics registered by a build module (`mathgen/generators/*_build.py`): these
# must meet the learner-facing text bar. Legacy topics keep their existing
# notation (their `latex` field is what the UI renders) and are out of scope
# until someone owns them.
BUILD_TOPICS = sorted(t for t, (_lvl, fn) in REGISTRY.items() if fn.__module__.rsplit(".", 1)[-1].endswith("_build"))

# ---------------------------------------------------------------------------
# Prompt-shape variety: `mathgen.text_quality.prompt_shape` masks digits AND every
# `mathgen.slots` bank value -- a strictly finer mask than
# `_distinct_prompts` in test_mathgen_options.py, which only dedupes raw TEXT. Every
# build source needs >= 3 distinct shapes over 60 seeds.
#
# `ONE_SHAPE_SOURCES` is the PERMANENT allowlist, same shape as
# `TEXT_DEFECTS_GRANDFATHERED`: a source whose structure is bounded to
# fewer than 3 shapes and always will be -- a preset that pins the
# structural choice (`convert_units_for_rates`'s `mode`,
# `cartesian_polar_convert`'s `variant`, ...), or a provably <=2-outcome
# classification (`factors_multiples_divisibility`: "does a divide b" is
# yes/no, one fixed template either way -- already `GENUINE_TWO_OPTION_TOPICS`
# in test_mathgen_answers.py for the same reason). Every entry was confirmed by reading the
# generator, not just its measured count.
#
# `VARIETY_BACKLOG` is the SHRINK-ONLY backlog, same tripwire shape as
# `OPTION_SHAPE_BACKLOG`: every one of these BUILD_TOPICS sources predates
# `mathgen.core.phrase()`, so almost
# every one is one sentence with the numbers swapped -- exactly the
# "authoring habit, not difficulty" gap. Measured at 622 of 1052 BUILD_TOPICS x
# declared-preset (incl. None) combinations; 27 are the permanent
# ONE_SHAPE_SOURCES above, the remaining 595 are this backlog, worked down one
# skill/domain at a time as cleanups adopt `phrase()`. Large by measurement, not
# by mistake.

ONE_SHAPE_SOURCES: dict[tuple[str, str | None], str] = {
    (
        "cartesian_polar_convert",
        "easy",
    ): "variant is FIXED by the preset (axis/triple/reverse), never rng-chosen: permanently one template per tier",
    (
        "cartesian_polar_convert",
        "hard",
    ): "variant is FIXED by the preset (axis/triple/reverse), never rng-chosen: permanently one template per tier",
    (
        "cartesian_polar_convert",
        "medium",
    ): "variant is FIXED by the preset (axis/triple/reverse), never rng-chosen: permanently one template per tier",
    (
        "cartesian_polar_convert",
        None,
    ): "variant is FIXED by the preset (axis/triple/reverse), never rng-chosen: permanently one template per tier",
    (
        "convert_between_metric_units_of_area",
        "easy",
    ): "direction is rng-chosen but bounded to exactly 2 possible outcomes per mode",
    (
        "convert_between_metric_units_of_area",
        "medium",
    ): "direction is rng-chosen but bounded to exactly 2 possible outcomes per mode",
    (
        "convert_between_metric_units_of_area",
        None,
    ): "direction is rng-chosen but bounded to exactly 2 possible outcomes per mode",
    (
        "convert_energy_units",
        "easy",
    ): "mode is FIXED by the preset (j_kj/kj_mj/kwh_kj), never rng-chosen: permanently one template per tier",
    (
        "convert_energy_units",
        "hard",
    ): "mode is FIXED by the preset (j_kj/kj_mj/kwh_kj), never rng-chosen: permanently one template per tier",
    (
        "convert_energy_units",
        "medium",
    ): "mode is FIXED by the preset (j_kj/kj_mj/kwh_kj), never rng-chosen: permanently one template per tier",
    (
        "convert_energy_units",
        None,
    ): "mode is FIXED by the preset (j_kj/kj_mj/kwh_kj), never rng-chosen: permanently one template per tier",
    (
        "convert_units_for_rates",
        "easy",
    ): "mode is FIXED by the preset (kmh_to_ms/ms_to_kmh/km_per_min_to_ms), never rng-chosen: permanently one template per tier",
    (
        "convert_units_for_rates",
        "hard",
    ): "mode is FIXED by the preset (kmh_to_ms/ms_to_kmh/km_per_min_to_ms), never rng-chosen: permanently one template per tier",
    (
        "convert_units_for_rates",
        "medium",
    ): "mode is FIXED by the preset (kmh_to_ms/ms_to_kmh/km_per_min_to_ms), never rng-chosen: permanently one template per tier",
    (
        "convert_units_for_rates",
        None,
    ): "mode is FIXED by the preset (kmh_to_ms/ms_to_kmh/km_per_min_to_ms), never rng-chosen: permanently one template per tier",
    (
        "factors_multiples_divisibility",
        "easy",
    ): "genuine 2-option classification (a divides b: yes/no), one fixed template",
    (
        "factors_multiples_divisibility",
        "hard",
    ): "genuine 2-option classification (a divides b: yes/no), one fixed template",
    (
        "factors_multiples_divisibility",
        "medium",
    ): "genuine 2-option classification (a divides b: yes/no), one fixed template",
    (
        "factors_multiples_divisibility",
        None,
    ): "genuine 2-option classification (a divides b: yes/no), one fixed template",
    (
        "grams_milligrams_convert",
        "easy",
    ): "variant is FIXED by the preset (g_to_mg/mg_to_g), never rng-chosen: permanently one template per tier",
    (
        "grams_milligrams_convert",
        "medium",
    ): "variant is FIXED by the preset (g_to_mg/mg_to_g), never rng-chosen: permanently one template per tier",
    (
        "grams_milligrams_convert",
        None,
    ): "variant is FIXED by the preset (g_to_mg/mg_to_g), never rng-chosen: permanently one template per tier",
    ("prime_composite_square_properties", "easy"): "genuine 2-option classification, one fixed template",
    (
        "prime_factorisation_exponent",
        "easy",
    ): "one fixed template ('Write N as a product of prime powers...'), only the number varies",
    (
        "prime_factorisation_exponent",
        "hard",
    ): "one fixed template ('Write N as a product of prime powers...'), only the number varies",
    (
        "prime_factorisation_exponent",
        "medium",
    ): "one fixed template ('Write N as a product of prime powers...'), only the number varies",
    (
        "prime_factorisation_exponent",
        None,
    ): "one fixed template ('Write N as a product of prime powers...'), only the number varies",
}

VARIETY_BACKLOG: dict[tuple[str, str | None], str] = {
    (
        "additive_multiplicative_patterns",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "additive_multiplicative_patterns",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "additive_multiplicative_patterns",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "alternate_segment_theorem",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "alternate_segment_theorem",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "alternate_segment_theorem",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angle_in_semicircle",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angle_in_semicircle",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angle_in_semicircle",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angle_in_semicircle",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angle_of_inclination_gradient",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angle_of_inclination_gradient",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angle_of_inclination_gradient",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angles_same_arc",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angles_same_arc",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "angles_same_arc",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "annulus_cross_section_area",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "annulus_cross_section_area",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "annulus_cross_section_area",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "annulus_cross_section_area",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "applied_pythagoras",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "applied_pythagoras",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "applied_pythagoras",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "apply_approximation_strategy",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "apply_approximation_strategy",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "arc_length_sector_area",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "arc_length_sector_area",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "arc_length_sector_area",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "area_between_curves_simple_cases",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "arithmetic_recursion_generate",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "arithmetic_recursion_generate",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "arithmetic_recursion_generate",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "arithmetic_recursion_generate",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "aux_angle_r_cos_conversion",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "aux_angle_r_cos_conversion",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "aux_angle_r_cos_conversion",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "aux_angle_r_cos_conversion",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "bearings_problem",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "bearings_problem",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "bearings_problem",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "bearings_problem",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_perimeters_of_familiar_shapes",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_perimeters_of_familiar_shapes",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_perimeters_of_familiar_shapes",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_surface_areas_of_standard_solids",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_surface_areas_of_standard_solids",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_surface_areas_of_standard_solids",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_surface_areas_of_standard_solids",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_total_change_by_integrating_instantaneous_or_marginal",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_total_change_by_integrating_instantaneous_or_marginal",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_total_change_by_integrating_instantaneous_or_marginal",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "calculate_total_change_by_integrating_instantaneous_or_marginal",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "chords_equal_length_equal_angles",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "chords_equal_length_equal_angles",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "chords_equal_length_equal_angles",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "chords_equal_length_equal_angles",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_and_prism_duration_rates",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_and_prism_duration_rates",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_and_prism_duration_rates",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_and_prism_duration_rates",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_measures",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_measures",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_measures",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_measures",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_reverse_from_area",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_reverse_from_area",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_reverse_from_area",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_reverse_from_area",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_theorem_converses_classify",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_theorem_converses_classify",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "circle_theorem_converses_classify",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "compare_measurement_attributes",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "compare_measurement_attributes",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "complex_number_as_point",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "conditional_probability_notation",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "conditional_probability_notation",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "cone_volume",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "cone_volume",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "cone_volume",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "cone_volume",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "conjugate_roots_real_coefficients",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "coordinate_movement",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "coordinate_movement",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "coordinate_movement",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "coordinate_movement",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "cosine_rule_angle",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "counterexample_identify",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "counterexample_identify",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_estimate_add_subtract",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_estimate_add_subtract",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_estimate_add_subtract",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_estimate_add_subtract",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_metric_representation",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_metric_representation",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_metric_representation",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_metric_representation",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_tenths_hundredths_compose",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_tenths_hundredths_compose",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_tenths_hundredths_compose",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "decimal_tenths_hundredths_compose",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "define_and_use_multiplication_by_a_scalar_of",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "define_and_use_multiplication_by_a_scalar_of",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "define_and_use_multiplication_by_a_scalar_of",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "define_and_use_unit_vectors_and_the_perpendicular",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "define_and_use_unit_vectors_and_the_perpendicular",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "define_and_use_unit_vectors_and_the_perpendicular",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "define_and_use_unit_vectors_and_the_perpendicular",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_a_vector_equation_of_a_straight_line",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_a_vector_equation_of_a_straight_line",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_a_vector_equation_of_a_straight_line",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_a_vector_equation_of_a_straight_line",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_if_one_to_one",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_the_optimum_assignment_s_by_inspection_for",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_the_optimum_assignment_s_by_inspection_for",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_the_optimum_assignment_s_by_inspection_for",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_the_optimum_assignment_s_by_inspection_for",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_vector_and_cartesian_equations_of_a_plane",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_vector_and_cartesian_equations_of_a_plane",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_vector_and_cartesian_equations_of_a_plane",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_volumes_of_solids_of_revolution_about_either",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_volumes_of_solids_of_revolution_about_either",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_volumes_of_solids_of_revolution_about_either",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "determine_volumes_of_solids_of_revolution_about_either",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "distance_two_points",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "distance_two_points",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "distance_two_points",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "distance_two_points",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "division_remainder_context",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "division_remainder_context",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "division_remainder_context",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "elevation_depression_problem",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "elevation_depression_problem",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "elevation_depression_problem",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "elevation_depression_problem",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "engineering_notation",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "engineering_notation",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "engineering_notation",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "engineering_notation",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "equivalent_fraction_related_denominator",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "equivalent_fraction_related_denominator",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "equivalent_fraction_related_denominator",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "estimate_counts_and_check_reasonableness",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "estimate_counts_and_check_reasonableness",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "examine_examples_of_vectors_including_displacement_and_velocity",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "examine_examples_of_vectors_including_displacement_and_velocity",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "examine_properties_of_parallel_and_perpendicular_vectors_and",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "examine_properties_of_parallel_and_perpendicular_vectors_and",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "examine_properties_of_parallel_and_perpendicular_vectors_and",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "examine_properties_of_parallel_and_perpendicular_vectors_and",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "expand_binomial_product_monic",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "expand_binomial_product_monic",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "expand_binomial_product_monic",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "exponent_laws_integer_variable",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "exponent_laws_positive_zero",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "exponent_laws_positive_zero",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "exponent_laws_positive_zero",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "exponent_laws_positive_zero",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "factorise_quadratic",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "factorise_quadratic",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_surface_area_of_familiar_solids",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_surface_area_of_familiar_solids",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_surface_area_of_familiar_solids",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_surface_area_of_familiar_solids",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_area_of_irregular_figures",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_area_of_irregular_figures",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_area_of_irregular_figures",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_area_of_irregular_figures",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_equation_of_a_quadratic_given_sufficient",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_equation_of_a_quadratic_given_sufficient",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_equation_of_a_quadratic_given_sufficient",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_equation_of_a_quadratic_given_sufficient",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_equation_of_a_straight_line_given",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_equation_of_a_straight_line_given",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_surface_area_of_pyramids_such_as",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_surface_area_of_pyramids_such_as",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_surface_area_of_pyramids_such_as",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "find_the_surface_area_of_pyramids_such_as",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "finite_sample_space_probability",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "finite_sample_space_probability",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "finite_sample_space_probability",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "finite_sample_space_probability",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "fraction_halving",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "fraction_halving",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "fraction_halving",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "fraction_halving",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "frustum_volume",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "frustum_volume",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "frustum_volume",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "frustum_volume",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "geometric_nth_term_formula",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "geometric_nth_term_formula",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "geometric_nth_term_formula",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "geometric_nth_term_formula",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "geometric_recursion_generate",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "geometric_recursion_generate",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "geometric_recursion_generate",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "geometric_recursion_generate",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "imaginary_number_i_root",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "imaginary_number_i_root",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "imaginary_number_i_root",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "imaginary_number_i_root",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "index_laws_review",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "index_laws_review",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "index_laws_review",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "index_laws_review",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integer_numberline_order",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integer_numberline_order",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integer_numberline_order",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integer_numberline_order",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integrate_by_parts",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integrate_by_parts",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integrate_expressions_of_the_form_frac_pm1_sqrt",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integrate_expressions_of_the_form_frac_pm1_sqrt",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "integrate_expressions_of_the_form_frac_pm1_sqrt",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "interpolation_extrapolation_classify",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "interpolation_extrapolation_classify",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "interpolation_extrapolation_classify",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "interval_estimate_proportion_margin",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "interval_estimate_proportion_margin",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "interval_estimate_proportion_margin",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "interval_estimate_proportion_margin",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "inverse_one_to_one_function",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "irregular_composite_area",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "irregular_composite_area",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "irregular_composite_area",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "irregular_composite_area",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "journey_time_or_cost",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "journey_time_or_cost",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "journey_time_or_cost",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "journey_time_or_cost",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "large_number_ordering",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "large_number_ordering",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "leading_digit_estimate",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "leading_digit_estimate",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "leading_digit_estimate",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "leading_digit_estimate",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_composition",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_composition",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_composition",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_dilation_rotation_reflection",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_dilation_rotation_reflection",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_dilation_rotation_reflection",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_inverse",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_inverse",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "linear_transform_inverse",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "logarithms_as_indices",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "logarithms_as_indices",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "logarithms_as_indices",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "logarithms_as_indices",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "measurement_error_interpretation",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "measurement_error_interpretation",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "measurement_error_interpretation",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "mental_mult_div_extension",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "mental_mult_div_extension",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "mental_mult_div_extension",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "mental_mult_div_extension",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoint_coordinate_scalar",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoint_coordinate_scalar",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoint_coordinate_scalar",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoint_coordinate_scalar",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoint_two_points",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoint_two_points",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoint_two_points",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoint_two_points",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoints_of_the_sides_of_a_quadrilateral_join",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoints_of_the_sides_of_a_quadrilateral_join",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoints_of_the_sides_of_a_quadrilateral_join",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "midpoints_of_the_sides_of_a_quadrilateral_join",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "multidigit_multiplication",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "multidigit_multiplication",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "multidigit_multiplication",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "multidigit_multiplication",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "multiplication_division_inverse_unknown",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "natural_logarithm_define",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "natural_logarithm_define",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "natural_logarithm_define",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "natural_logarithm_define",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "nth_roots_of_complex_number",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "nth_roots_of_complex_number",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "nth_roots_of_complex_number",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "nth_roots_of_complex_number",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "nth_roots_of_unity",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "nth_roots_of_unity",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "nth_roots_of_unity",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "nth_roots_of_unity",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "number_algorithms_decisions",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "number_algorithms_decisions",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_1000",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_1000",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_1000",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_1000",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_120",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_120",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_120",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_120",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_twenty",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_twenty",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_twenty",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "order_to_twenty",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10a_acceleration_to_position_line",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10a_acceleration_to_position_line",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_critical_path_float",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_critical_path_float",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_critical_path_float",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_critical_path_float",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_critical_path_locate",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_critical_path_locate",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_critical_path_locate",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_critical_path_locate",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_government_allowance",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_government_allowance",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_government_allowance",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_government_allowance",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_personal_budget",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_personal_budget",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_personal_budget",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_personal_budget",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_population_compounding",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_population_compounding",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_population_compounding",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10c_population_compounding",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_angle_relationships",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_angle_relationships",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_angle_relationships",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_exact_approx_repeated_calc",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_exact_approx_repeated_calc",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_exact_approx_repeated_calc",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_logarithmic_scales",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_logarithmic_scales",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_logarithmic_scales",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p10e_logarithmic_scales",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_complete_rate_calculations",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_direct_proportion",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_direct_proportion",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_direct_proportion",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_direct_proportion",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_inverse_proportion",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_inverse_proportion",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_inverse_proportion",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_inverse_proportion",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_quadratic_variation",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_quadratic_variation",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_quadratic_variation",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_examine_quadratic_variation",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p7d_use_rates_to_compare",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_confidence_interval_mean",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_confidence_interval_mean",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_confidence_interval_mean",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_confidence_interval_mean",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_confidence_interval_proportion",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_confidence_interval_proportion",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_confidence_interval_proportion",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_confidence_interval_proportion",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_normality_approx_phat",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_normality_approx_phat",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_normality_approx_phat",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p8c_normality_approx_phat",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_decimal_order_numberline",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_decimal_order_numberline",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_decimal_order_numberline",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_decimal_order_numberline",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_factor_multiple_divisibility_algorithm",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_factor_multiple_divisibility_algorithm",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_factor_multiple_divisibility_algorithm",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_factor_multiple_divisibility_algorithm",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_round_decimal_accuracy",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_round_decimal_accuracy",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_round_decimal_accuracy",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_terminating_recurring_decimal",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_terminating_recurring_decimal",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_terminating_recurring_decimal",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "p9e_terminating_recurring_decimal",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "parallelogram_area",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "parallelogram_area",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "parallelogram_area",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "parallelogram_area",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "pascals_triangle_properties",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "pascals_triangle_properties",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "pascals_triangle_properties",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "pascals_triangle_properties",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "percentage_of_amount_review",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "percentage_of_amount_review",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "percentage_of_amount_review",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "percentage_to_fraction",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "percentage_to_fraction",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "percentage_to_fraction",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "percentage_to_fraction",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "powers_ten_multiply_divide",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "powers_ten_multiply_divide",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "powers_ten_multiply_divide",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "powers_ten_multiply_divide",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "prism_capacity_conversion",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "probability_complement_union",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "probability_complement_union",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "proof_by_contradiction_identify",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "proof_by_contradiction_identify",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "proof_by_contradiction_identify",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "pythagoras_truss_chain",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "pythagoras_truss_chain",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "pythagoras_truss_chain",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "pythagoras_truss_chain",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "quadratic_complex_conjugate_solutions",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "quadratic_complex_conjugate_solutions",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "quadratic_complex_conjugate_solutions",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "quantify_compare_to_20",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "quantify_compare_to_20",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "radicals_fractional_indices",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "radicals_fractional_indices",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "radicals_fractional_indices",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "radicals_fractional_indices",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "rational_four_operations",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "rational_percentage_estimation",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "rational_percentage_estimation",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "rational_percentage_estimation",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "rational_percentage_estimation",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "reasonableness_verdict",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "reasonableness_verdict",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "reasonableness_verdict",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "reasonableness_verdict",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "rectangle_mixed_unit_area_perimeter",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "regular_irregular_area_perimeter",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "regular_irregular_area_perimeter",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "regular_irregular_area_perimeter",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "regular_irregular_area_perimeter",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "represent_a_scalar_multiple_of_a_vector",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "represent_a_scalar_multiple_of_a_vector",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "represent_a_scalar_multiple_of_a_vector",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "represent_a_scalar_multiple_of_a_vector",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "review_metric_units_of_length",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "review_metric_units_of_length",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "review_metric_units_of_length",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "review_metric_units_of_length",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scale_factor_similar_areas",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scale_factor_similar_areas",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scale_factor_similar_areas",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scale_factor_similar_areas",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scale_factor_similar_solids",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scale_factor_similar_solids",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scale_factor_similar_solids",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scale_factor_similar_solids",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scientific_notation_measurements",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scientific_notation_measurements",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scientific_notation_measurements",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "scientific_notation_measurements",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "shape_features_classification",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "significant_figures",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "significant_figures",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "significant_figures",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "significant_figures",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "similar_figure_scale_factor_length",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "similar_figure_scale_factor_length",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "similar_figure_scale_factor_length",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "similar_figure_scale_factor_length",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "similarity_conditions_classify",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "similarity_conditions_classify",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "similarity_conditions_classify",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "similarity_conditions_classify",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "simple_polynomial_equation",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "simple_polynomial_equation",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "simple_polynomial_equation",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "simple_polynomial_equation",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "sine_cosine_ratio_triangle",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "sine_cosine_ratio_triangle",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "sine_cosine_ratio_triangle",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "sine_cosine_ratio_triangle",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "sine_rule_ambiguous",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "sine_rule_angle",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "sine_rule_solve",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "skip_count_pattern_sequences",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "skip_count_pattern_sequences",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "skip_count_pattern_sequences",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "skip_count_pattern_sequences",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_indices_with_logarithms",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_indices_with_logarithms",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_indices_with_logarithms",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_indices_with_logarithms",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_linear_equation_fraction_both_sides",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_small_scale_network_flow_problems_including_the",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_small_scale_network_flow_problems_including_the",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_small_scale_network_flow_problems_including_the",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "solve_small_scale_network_flow_problems_including_the",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "square_roots_perfect_squares",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "square_roots_perfect_squares",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "square_roots_perfect_squares",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "square_roots_perfect_squares",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_rationalise_binomial",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_rationalise_binomial",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_rationalise_binomial",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_rationalise_binomial",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_simplify_algebraic_fraction",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_simplify_surds",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_simplify_surds",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_simplify_surds",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t1_simplify_surds",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_degrees_radians_pi_form",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_degrees_radians_pi_form",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_degrees_radians_pi_form",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_degrees_to_radians_decimal",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_degrees_to_radians_decimal",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_degrees_to_radians_decimal",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_degrees_to_radians_decimal",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_radians_to_degrees_decimal",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_radians_to_degrees_decimal",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_radians_to_degrees_decimal",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_radians_to_degrees_decimal",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_radians_to_dms",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_radians_to_dms",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "t4_radians_to_dms",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "tan_shift_complementary_identity",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "tan_shift_complementary_identity",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "tan_shift_complementary_identity",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "tan_shift_complementary_identity",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_angle_at_the_centre_subtended_by_an_arc",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_angle_at_the_centre_subtended_by_an_arc",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_angle_at_the_centre_subtended_by_an_arc",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_angle_at_the_centre_subtended_by_an_arc",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_diagonals_of_a_parallelogram_meet_at_right",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_diagonals_of_a_parallelogram_meet_at_right",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_diagonals_of_a_parallelogram_meet_at_right",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_opposite_angles_of_a_cyclic_quadrilateral_are",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "the_opposite_angles_of_a_cyclic_quadrilateral_are",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "time_units_and_nearest_minute",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "translations_and_their_representation_as_column_vectors",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "translations_and_their_representation_as_column_vectors",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "translations_and_their_representation_as_column_vectors",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "translations_and_their_representation_as_column_vectors",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "triangle_area_base_height",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "triangle_area_base_height",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "triangle_area_base_height",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "triangle_area_base_height",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "triangular_prism_volume",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "triangular_prism_volume",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "triangular_prism_volume",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unit_circle_tan_evaluate",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unit_circle_tan_evaluate",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unit_circle_tan_evaluate",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unit_circle_tan_evaluate",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unit_fractions_combine_whole",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unit_fractions_combine_whole",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unit_fractions_combine_whole",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unmarked_partial_measurement",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "unmarked_partial_measurement",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_addition_of_the_area_of_the_faces",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_addition_of_the_area_of_the_faces",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_addition_of_the_area_of_the_faces",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_addition_of_the_area_of_the_faces",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_formulas_to_calculate_areas",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_formulas_to_calculate_areas",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_formulas_to_calculate_areas",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_formulas_to_calculate_areas",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_formulas_to_find_the_volume",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_formulas_to_find_the_volume",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_formulas_to_find_the_volume",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_forward_and_backward_scanning_to_determine_the",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_forward_and_backward_scanning_to_determine_the",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_forward_and_backward_scanning_to_determine_the",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_forward_and_backward_scanning_to_determine_the",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_metric_units_of_area",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_metric_units_of_volume",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_metric_units_of_volume",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_metric_units_of_volume",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_ordered_pair_notation_and_column_vector_notation",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_ordered_pair_notation_and_column_vector_notation",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_ordered_pair_notation_and_column_vector_notation",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_ordered_pair_notation_and_column_vector_notation",
        None,
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_partial_fractions_where_necessary_for_integration_in",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_partial_fractions_where_necessary_for_integration_in",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_partial_fractions_where_necessary_for_integration_in",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_partial_fractions_where_necessary_for_integration_in",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_the_critical_path_to_determine_the_minimum",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_the_critical_path_to_determine_the_minimum",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_the_critical_path_to_determine_the_minimum",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_the_critical_path_to_determine_the_minimum",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_units_of_time_conversions_between_units_fractional",
        "easy",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_vector_equations_of_curves_in_two_or",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_vector_equations_of_curves_in_two_or",
        "hard",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_vector_equations_of_curves_in_two_or",
        "medium",
    ): "pre-phrase() single template (2 shapes/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "use_vector_equations_of_curves_in_two_or",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "volume_of_pyramids_and_spheres",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "volume_of_pyramids_and_spheres",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "volume_of_pyramids_and_spheres",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "volume_of_pyramids_and_spheres",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "volume_surface_area_prisms_cylinders",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "volume_surface_area_prisms_cylinders",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "volume_surface_area_prisms_cylinders",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "volume_surface_area_prisms_cylinders",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "when_two_chords_of_a_circle_intersect_the_product",
        "easy",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "when_two_chords_of_a_circle_intersect_the_product",
        "hard",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "when_two_chords_of_a_circle_intersect_the_product",
        "medium",
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
    (
        "when_two_chords_of_a_circle_intersect_the_product",
        None,
    ): "pre-phrase() single template (1 shape/60 seeds); a cleanup varies it via mathgen.core.phrase",
}

_SHAPE_SEEDS = 60


def _distinct_shapes(topic, difficulty):
    return {prompt_shape(make(topic, seed=seed, difficulty=difficulty)) for seed in range(_SHAPE_SEEDS)}


@pytest.mark.parametrize("topic", BUILD_TOPICS)
def test_prompt_variety(topic):
    for difficulty in [*sorted(PRESETS.get(topic, {})), None]:
        key = (topic, difficulty)
        if key in ONE_SHAPE_SOURCES or key in VARIETY_BACKLOG:
            continue
        shapes = _distinct_shapes(topic, difficulty)
        assert len(shapes) >= 3, (
            f"{topic} [{difficulty}] yields only {len(shapes)} distinct prompt "
            f"shapes in {_SHAPE_SEEDS} seeds (at least 3 are required, masked by "
            f"mathgen.text_quality.prompt_shape). Allowlist with a reason in "
            f"ONE_SHAPE_SOURCES if genuinely bounded forever, or VARIETY_BACKLOG "
            f"if it should eventually vary"
        )


@pytest.mark.parametrize("topic,difficulty", sorted(ONE_SHAPE_SOURCES, key=lambda key: (key[0], key[1] or "~")))
def test_one_shape_source_still_bounded(topic, difficulty):
    """Forces the allowlist to stay honest: a source that gained a third
    shape (a preset rewired, a new branch added) must be removed, matching
    `test_grandfathered_text_defect_still_present`'s mechanics."""
    assert topic in BUILD_TOPICS, f"{topic} is not a build topic; drop it from ONE_SHAPE_SOURCES"
    shapes = _distinct_shapes(topic, difficulty)
    assert len(shapes) < 3, (
        f"{topic} [{difficulty}] now yields {len(shapes)} distinct shapes: "
        f"remove it from ONE_SHAPE_SOURCES (it is no longer permanently bounded)"
    )


@pytest.mark.parametrize("topic,difficulty", sorted(VARIETY_BACKLOG, key=lambda key: (key[0], key[1] or "~")))
def test_variety_backlog_still_thin(topic, difficulty):
    """Forces the backlog to shrink: a fixed source must be removed, matching
    `test_option_shape_backlog_still_dirty`'s mechanics."""
    assert topic in BUILD_TOPICS, f"{topic} is not a build topic; drop it from VARIETY_BACKLOG"
    shapes = _distinct_shapes(topic, difficulty)
    assert len(shapes) < 3, (
        f"{topic} [{difficulty}] now yields {len(shapes)} distinct shapes: remove it from VARIETY_BACKLOG"
    )


# ---------------------------------------------------------------------------
# Tier escalation. `scripts.measure_tiers.classify_skill` is the ONE classification
# function this test and `scripts/measure_tiers.py` both call -- imported,
# never reimplemented, so the gate and the measurement can never quietly
# diverge on what "escalates" means.
#
# Keyed by SKILL ID via `mathgen.skills.load_skills()`, never by BUILD_TOPICS:
# the confirmed range-only examples ("Solve 4x = 48", "4(x + 9) = 148") are the
# `linear_equation` topic in `mathgen/generators/algebra.py`, a legacy
# non-`_build` module a BUILD_TOPICS-keyed test structurally cannot see.
# Loaded ONCE at module scope here, beside `TOPICS`/`REGISTRY` at the top of
# this file -- never inside a fixture, and never via a full-library `validate()`,
# which is slow.
from mathgen import skills as _skills_mod  # noqa: E402
from scripts.measure_tiers import DEFAULT_SEEDS as _TIER_SEEDS  # noqa: E402
from scripts.measure_tiers import classify_skill as _classify_skill  # noqa: E402
from scripts.measure_tiers import qualifying_skills as _qualifying_skills  # noqa: E402

_TIER_DOMAINS, _TIER_ALL_SKILLS = _skills_mod.load_skills()
_TIER_QUALIFYING_SKILLS = {s.id: s for s in _qualifying_skills(_TIER_ALL_SKILLS)}

# Confirmed range-only by `scripts/measure_tiers.py` at 60 seeds (the 20-seed
# default under-detected range-only tiers). Shrink-only: a skill whose hard tier
# is fixed (a new preset, a re-pointed YAML source -- never a mutated one) is
# removed here, never merely left skipped.
TIER_ESCALATION_BACKLOG: dict[str, str] = {
    "decimal-arithmetic": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "f10-decimal-notation-tenths-hundredths": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "f10-decimal-order-numberline": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "f10-fraction-percent-quantity-strategy": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "f10-multidigit-multiplication": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "f10-percentage-equivalence-comparison": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "f10-rational-percentage-estimation": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "f10-round-decimals-accuracy": "investigated: hard tier's real escalation (force_five, force_carry in p9e_decimal_accuracy_order_build.py) is genuine but instrument-blind -- mathgen.text_quality.prompt_shape masks every digit run including the decimal point, so a forced boundary digit or forced carry never changes the rendered shape. Stays range-only until a text-visible, non-cosmetic escalation is found; not built.",
    "f10-terminating-recurring-decimals": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "jacobi-iteration-2x2": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "senior-find-the-line-of-best-fit-by-eye": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "senior-review-calculating-a-percentage-of-a-given-amount": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "significant-figures": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
    "simplify-radicals-of-monomials": "confirmed range-only by scripts/measure_tiers.py at 60 seeds (120 agrees)",
    "two-by-two-eigenvalues": "confirmed range-only by scripts/measure_tiers.py at 60 seeds",
}

_TIER_TESTABLE_IDS = sorted(set(_TIER_QUALIFYING_SKILLS) - set(TIER_ESCALATION_BACKLOG))


@pytest.mark.parametrize("skill_id", _TIER_TESTABLE_IDS)
def test_tier_escalation(skill_id):
    skill = _TIER_QUALIFYING_SKILLS[skill_id]
    verdict, _adds = _classify_skill(skill, _TIER_SEEDS)
    assert verdict == "escalates", (
        f"{skill_id}: hard tier adds no prompt shape over the union of every "
        f"lower tier (scripts/measure_tiers.py, {_TIER_SEEDS} seeds) -- either "
        f"a genuine range-only hard tier (add it to TIER_ESCALATION_BACKLOG "
        f"with the skill/seed evidence) or a fixable one"
    )


@pytest.mark.parametrize("skill_id", sorted(TIER_ESCALATION_BACKLOG))
def test_tier_escalation_backlog_still_range_only(skill_id):
    """Forces the backlog to shrink: a fixed skill must be removed, matching
    `test_option_shape_backlog_still_dirty`'s mechanics."""
    skill = _TIER_QUALIFYING_SKILLS.get(skill_id)
    assert skill is not None, (
        f"{skill_id} is not a qualifying skill (active, testable, >= 2 sources); drop it from TIER_ESCALATION_BACKLOG"
    )
    verdict, _adds = _classify_skill(skill, _TIER_SEEDS)
    assert verdict == "range-only", f"{skill_id} now escalates: remove it from TIER_ESCALATION_BACKLOG"
