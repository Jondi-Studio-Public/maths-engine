"""Output contract: pinned outputs, registry, fuzz, determinism, presets, answer_form,
and helper reuse.

Part of the generator sweep (`tests/test_mathgen_*.py`). Verification
is built into the generators themselves (each independently re-checks its
answer with SymPy and raises VerificationError on mismatch), so constructing
thousands of questions IS the correctness test; the assertions here cover the
output contract on top of that. Set `MATHS_DEEP_SEEDS` for a deeper run of the
text checks.

The per-topic checks here (`check_*`) take the questions already built, and
`tests/mathgen_sweep_cases.py` builds each topic's grid once and runs every check
on it, so a topic is generated once instead of once per check.
"""

import hashlib
import json

import pytest

from mathgen import make, topics
from mathgen.core import PRESETS, REGISTRY
from tests.mathgen_sweep_support import NEW_KIND_VARIANTS, make_kind, take

TOPICS = [t for _lvl, t in topics()]
# Topics registered by a build module (`mathgen/generators/*_build.py`): these
# must meet the learner-facing text bar. Legacy topics keep their existing
# notation (their `latex` field is what the UI renders) and are out of scope
# until someone owns them.
BUILD_TOPICS = sorted(t for t, (_lvl, fn) in REGISTRY.items() if fn.__module__.rsplit(".", 1)[-1].endswith("_build"))
# dsolve/checkodesol is ~27 ms/question; fewer fuzz iterations there.
N_FUZZ = {"differential_equation": 60}

# ---------------------------------------------------------------------------
# Backward-compatibility pin: sha256[:16] of make(topic, seed[difficulty])
# .to_json(), computed against the generators as they stood immediately
# before the "kinds"/params extensions in this change landed. Any change to
# an existing (topic, seed[difficulty]) question's output fails this test
# -- new kinds/params are only reachable by opting in explicitly, so they
# cannot perturb these hashes.
PINNED_HASHES = {
    "addition": {
        "0": "29289f5ae3d0b756",
        "1": "265060418cdd9aeb",
        "7": "27234f053fb81c2d",
        "42": "214efd44c1428f64",
        "3:easy": "32a778a923ce7c18",
        "3:medium": "a2211284da78bea3",
        "3:hard": "062df6dbe05be104",
    },
    "subtraction": {
        "0": "1fa25864e6c2f9ca",
        "1": "3102eeca5b589dbf",
        "7": "d11d880d1ca05fd8",
        "42": "5b4dadcbdfa52ef2",
        "3:easy": "1b433c8266389c79",
        "3:medium": "beba1b6f1fc8482c",
        "3:hard": "433eb99cf60d80d7",
    },
    "multiplication": {
        "0": "376c5287aa112feb",
        "1": "6ef3300e7ce262a6",
        "7": "e7b538db4c9e0f8d",
        "42": "e7f614be9b0eb4b4",
        "3:easy": "98ed76da8f578ca5",
        "3:medium": "dbdd93719fe19efa",
        "3:hard": "515033015f5348cb",
    },
    "division": {
        "0": "37a22540552812cf",
        "1": "ba76b635dc0373b0",
        "7": "1c467d677cd08e75",
        "42": "b73db071056f4c0c",
        "3:easy": "a32b1457b7a735b8",
        "3:medium": "42874555febec247",
        "3:hard": "b89a68c840fa5f0c",
    },
    "fractions": {
        "0": "eae2e56b9d0c2bae",
        "1": "a85fc0ee9ee59529",
        "7": "e1165e4034049a69",
        "42": "cebb83cca96e33f5",
        "3:easy": "0c94e36ec2edf2d2",
        "3:medium": "7ef01cef6d9a6f1a",
        "3:hard": "ec13ffb7d103123f",
    },
    "powers": {
        "0": "9aff4b6502ea7594",
        "1": "86c5696b94936b19",
        "7": "e21ff7918da91f73",
        "42": "96b376780cced85a",
        "3:easy": "0e8b763db46fd8cc",
        "3:medium": "6b195ba4353869a5",
        "3:hard": "1cad5f60925ec59c",
    },
    # Re-pinned once for the typed answer contract (engine commit aea1e1f): the
    # topic's distractors now carry misconception tags, so its output gained
    # `option_notes` and `distractor_values`.
    "linear_equation": {
        "0": "93bad5c9299b1615",
        "1": "173a59f5b1c7c484",
        "7": "e4a4763946120e77",
        "42": "2ba4e3d6d0a34537",
        "3:easy": "d3cb84b2a9545d59",
        "3:medium": "19e7a5d6850c8c0f",
        "3:hard": "141547b9cb64954d",
    },
    # Updated once for the sign-formatting fix in mathgen/generators/algebra.py
    # (simultaneous's `side()` no longer prints "1x"/"-1x" for a unit x/y
    # coefficient).
    "simultaneous": {
        "0": "eb24cfee1e0d5fb1",
        "1": "9c4e42537dbd746b",
        "7": "0e57702cb5431eda",
        "42": "8098cf1d09894e4a",
        "3:easy": "b3309f2691e12bf0",
        "3:medium": "9c269736a454fa72",
        "3:hard": "455d312fb0243ba9",
    },
    # Re-pinned once for the typed answer contract (engine commit aea1e1f): the
    # topic's distractors now carry misconception tags, so its output gained
    # `option_notes` and `distractor_values`.
    "quadratic": {
        "0": "b658f0e6cd2a967c",
        "1": "013464ec2665ceda",
        "7": "96d23e56f85fff02",
        "42": "5d57078c7e754b06",
        "3:easy": "29063a45947d1bd4",
        "3:medium": "f93fbbad0cb9a114",
        "3:hard": "f2d4eca5b4cc745a",
    },
    # Re-pinned once for the typed answer contract (engine commit aea1e1f): the
    # topic's distractors now carry misconception tags, so its output gained
    # `option_notes` and `distractor_values`.
    # Re-pinned (bug hunt round 2): only `working` changed -- zero terms and unit coefficients are no longer
    # written out, so no "+ -3(2)", "+ 0" or "1(3)" noise. Seeds 0, 1 and 42 were already clean.
    "function_evaluation": {
        "0": "291346ace9e7ead3",
        "1": "8898d6f7117d2d48",
        "7": "5fe096e5d9c0f614",
        "42": "2aa37dcce84a2627",
        "3:easy": "3a541f5a6c0415dd",
        "3:medium": "83d289b33bfdee6e",
        "3:hard": "e3f3f4983e8d7f0f",
    },
    # Updated once for the medium/hard tier-collision fix in
    # mathgen/generators/geometry.py (hard's k now draws from randint(4, 6)
    # instead of randint(1, 3), disjoint from medium's range) -- only
    # "3:hard" moved, since seed 3 under the hard preset draws a
    # k-dependent kind (side_hyp); the bare seeds 0/1/7/42 use the
    # function's own defaults (k_hi=3, new default k_lo=1, unaffected) and
    # "3:easy"/"3:medium" are untouched presets.
    "trigonometry": {
        "0": "139567d7d3e0e97e",
        "1": "6bcd02ad051e4f51",
        "7": "3af00f21a7050e58",
        "42": "7d6e9c0109bcfddb",
        "3:easy": "57f03bdd0ea52b74",
        "3:medium": "c8cea9b556d35cce",
        "3:hard": "2b026b4bb80b59c2",
    },
    # Updated once for the new `answer_alt` field on vector-magnitude
    # questions.
    "vectors": {
        "0": "7198d95700295c41",
        "1": "eeaa585bc5df4b6c",
        "7": "f4bba27512f24967",
        "42": "8e99eb4bd6bf8d99",
        "3:easy": "01b210be42e74e9d",
        "3:medium": "c5c9a4578a2b2d17",
        "3:hard": "905249c232f57057",
    },
    "matrices": {
        "0": "ff52a569ad6652ff",
        "1": "b115b23a732c214b",
        "7": "f476a9682f58c1d6",
        "42": "a37f0c1058f61bea",
        "3:easy": "3de32027efd2d01f",
        "3:medium": "8cca35f953cbecae",
        "3:hard": "f14d2e1cd1300424",
    },
    "differentiation": {
        "0": "103b01f2568dc46a",
        "1": "de91e16dfc7e7880",
        "7": "be96fe6a7e7ab8cd",
        "42": "28f73babea794a56",
        "3:easy": "0d88dde0c720af62",
        "3:medium": "e24cb248562da28b",
        "3:hard": "80f1c6b73ff02b8c",
    },
    "integration": {
        "0": "f9f42d311565ac68",
        "1": "e5ff41694909f8d1",
        "7": "35cd4993405153d6",
        "42": "7400a61c431a0a2e",
        "3:easy": "1bb9007b1b42725f",
        "3:medium": "bcabf2fe65b7e2b4",
        "3:hard": "d777278f5520c95d",
    },
    # Re-pinned once for the typed answer contract (engine commit aea1e1f): the
    # topic's distractors now carry misconception tags, so its output gained
    # `option_notes` and `distractor_values`.
    "limits": {
        "0": "a4653ae129540800",
        "1": "164837ba6d33331d",
        "7": "8d07438b48d7a073",
        "42": "51a9eb3187d5107c",
        "3:easy": "13056ebbbc62a6c6",
        "3:medium": "0ed8fe1df145afa5",
        "3:hard": "0a4a9fb85cced6b3",
    },
    # The "exponential" branch was rebuilt from a
    # general solution ("y = C*e^(kx)") to a particular solution from a
    # stated initial condition, so every bare/preset draw that lands on
    # "exponential" changes. Verified by hand which pinned entries actually
    # draw that kind (`rng.choice(("exponential", "separable_poly"))`, no
    # shipped skill exercises this bare/preset path -- both skills override
    # `kinds` explicitly): bare seeds 0 and 7 draw "exponential" and moved;
    # bare seeds 1 and 42, and all three "3:easy"/"3:medium"/"3:hard" preset
    # draws, draw "separable_poly" and are byte-identical to before --
    # confirmed by printing each draw's `q.text` before editing
    # the entry, not just trusting the hash diff.
    "differential_equation": {
        "0": "24bb3f0d61865018",
        "1": "dfb1b7120968fba4",
        "7": "0a3b83ebe4cc0a37",
        "42": "ade5bff6cd53b983",
        "3:easy": "b1347a950725a075",
        "3:medium": "93b7fde8738b045a",
        "3:hard": "30d9f2e73465eb71",
    },
    "partial_derivatives": {
        "0": "fb8b810e5cedf8a2",
        "1": "9aa2261e25ca4bbc",
        "7": "c8dfe801a8ce8d98",
        "42": "6070312b8441ec9c",
        "3:easy": "5f6842365f79bcc7",
        "3:medium": "f4d9e432ebc54add",
        "3:hard": "0cb37bd73e07e5b2",
    },
    "engineering": {
        "0": "f7fc59a9a3f52c28",
        "1": "0701d34b4bafcfcc",
        "7": "e5d9f34fb904c7ed",
        "42": "122277b5c17cdecb",
        "3:easy": "031c4ff4a8dd7a3e",
        "3:medium": "82345d8813f66647",
        "3:hard": "8b6e13b09b543846",
    },
    "structural": {
        "0": "146b442c682902b9",
        "1": "e70c26a48e48a497",
        "7": "4b8048c2f04aa3c1",
        "42": "a4a7db5ee1a58d6f",
        "3:easy": "33c5e15ea2f8236d",
        "3:medium": "a90a5261924ccd7c",
        "3:hard": "576b8afc95fb3271",
    },
    # First pin for this topic, taken after the `:g` exponent-notation
    # fix (mathgen/generators/curriculum_expansion.py) -- there was no prior
    # entry to preserve. No presets, so no "seed:tier" keys.
    "scientific_notation": {
        "0": "5699a991208686bd",
        "1": "c32c27b7112eae67",
        "7": "0748b3ced4a72d4f",
        "42": "c482e0211cc44e29",
    },
}


@pytest.mark.parametrize("topic", sorted(PINNED_HASHES))
def test_backward_compatible_outputs_pinned(topic):
    """Every pre-existing (topic, seed[difficulty]) pair must still produce
    byte-identical output after the generator extensions in this change."""
    for key, expected in PINNED_HASHES[topic].items():
        if ":" in key:
            seed_str, preset = key.split(":")
            q = make(topic, seed=int(seed_str), difficulty=preset)
        else:
            q = make(topic, seed=int(key))
        got = hashlib.sha256(q.to_json().encode()).hexdigest()[:16]
        assert got == expected, f"{topic} seed={key}: output changed for a pre-existing question"


# ---------------------------------------------------------------------------
# `option_notes` (option text -> a misconception/near_miss id) is an optional
# field on `mathgen/core.py`'s Question that only misconception-tagged generators
# populate. Proven two ways: the key must never appear in the output of a
# registered topic outside OPTION_NOTES_TOPICS (the property that decides whether
# a pinned hash can move, and unlike a fresh per-topic pinned-hash table it stays
# valid as the registry grows), and the field itself round-trips exactly like
# `answer_alt` -- omitted when falsy, present verbatim when a generator does set
# it. `test_backward_compatible_outputs_pinned` above is the persistent
# byte-level guard for the topics it already pins.


# Topics whose generators tag their distractors with a misconception id
# (`mathgen.answers`: each `Distractor`'s misconception becomes an `option_notes`
# entry), so their top-level output carries `option_notes` by design. The list only
# shrinks: a topic that stops tagging must be removed (see the test below it).
OPTION_NOTES_TOPICS = frozenset(
    {
        "ds_display_choice_easy",
        "ds_census_sample_statements",
        "ds_display_choice_medium",
        "ds_display_choice_hard",
        "ds_freq_table_easy",
        "ds_shape_var_read",
        "ds_classify_ordered",
        "ds_freq_table_cumulative",
        "ds_sample_vs_pop_gap",
        "ds_shape_var_shape",
        "ds_freq_table_derived",
        "ds_shape_var_compare",
        "w6b_trig_equation_interval",
        "grid_latlong_typed",
        "pw1_piecewise_cost_graph",
        "nv_bits_hex_counts",
        "nv_summary_choice",
        "w3_flat_depreciation",
        "w3_reducing_depreciation",
        "w3_compare_depreciation",
        "w3_interest_only_loan",
        "w3_annuity_table",
        "w3_profit_markup",
        "w3_commission_pay",
        "w3_supply_demand",
        "w3_fair_game",
        "w3_hcf_lcm_words",
        "w3_ci_tiers",
        "w3_si_tiers",
        "w3_wage_tiers",
        "w3_ot_tiers",
        "w3_unit_tiers",
        "w3_flatloan_tiers",
        "w3_movavg_tiers",
        "w3_mean_tiers",
        "w3_netpath_tiers",
        "w3_percentage_change",
        "w3_classify_easy",
        "w10_proof_irrational_surd",
        "w10_proof_parity",
        "w10_proof_am_gm",
        "w10_proof_triangle",
        "w3_ests_lsts",
        "w3_bearings_general",
        "w6a_aux_angle",
        "dg_graph_terms",
        "dg_construct_columns",
        "dg_construct_dotplot",
        "dg_construct_stem",
        "dg_compare_displays",
        "dg_real_data_graphs",
        "dg_grid_map",
        "ey_place_objects",
        "eyz_named_solids",
        "eyz_angle_estimate",
        "eyz_symmetry_count",
        "eyz_prism_pyramid_counts",
        "eyz_estimate_collection",
        "eyz_angle_kind",
        "eyz_twos_facts",
        "gt_inverse_trig",
        "gt_f_squared",
        "gt_sqrt_function",
        "sc_complex_euler",
        "eyz_chance_words",
        "eyz_list_outcomes",
        "eyz_right_angle_compare",
        "gt_graph_transform",
        "gt_modulus",
        "sc_conditional_counts",
        "sc_integrate_tan_sec",
        "sc_simulated_proportions",
        "sc_slope_field",
        "sc_trapezium",
        "sc_whole_units",
        "w6a_line_conic",
        "w6a_boolean_logic",
        "w6b_complex_polygon_roots",
        "w6b_complex_rotation",
        "nx_complex_arg_ratio",
        "nx_trig_sub_hsc",
        "w6b_sinusoid_features",
        "w7f_turn_contexts",
        "w7f_trial_variants",
        "w7a_graph_types",
        "w7a_network_cut",
        "w7a_dp_table",
        "w6c_cofactor",
        "w6c_taylor",
        "w6c_hermitian",
        "w6c_svd",
        "w7c_quadric",
        "w7c_coding",
        "fy1la_column_space",
        "fy1la_consistency",
        "fy1la_eigenvalues",
        "fy1la_general_solution",
        "fy1la_independence",
        "fy1la_rref",
        "fy1la_span",
        "w5c_gradient_vector",
        "w5c_z_test_mean",
        "w5a_cauchy_riemann",
        "w5a_complex_log",
        "w5a_contour",
        "w5a_laurent",
        "w5a_mobius",
        "w5b_metric_balls",
        "w5b_interior_closure",
        "w5a_real_integrals",
        "w5a_residues",
        "w5b_contour_param",
        "w5b_pde_discriminant",
        "enlargement_invariants",
        "exd_x_pr7_large_numbers_mixed_forms",
        "exd_x_pr7_order_to_1000_regrouped",
        "exd_x_pr7_signed_temperatures_order",
        "exd_x_u2_eigen_shifted",
        "fsp_map_route",
        "fsp_nets",
        "fsp_plan_views",
        "fsp_scale_plan",
        "fsp_solids",
        "exd_x_u2_rank_parameter",
        "ey2a_compare_match_quantities",
        "ey2a_count_principles",
        "ey2a_equal_groups_sharing_halves",
        "ey2a_part_part_whole_pictures",
        "ey2a_ten_frame_numbers",
        "eyz_angle_estimate",
        "eyz_angle_kind",
        "eyz_chance_words",
        "eyz_estimate_collection",
        "eyz_list_outcomes",
        "eyz_named_solids",
        "eyz_prism_pyramid_counts",
        "eyz_right_angle_compare",
        "eyz_symmetry_count",
        "eyz_twos_facts",
        "ey_compare_displays",
        "ey_count_steps",
        "ey_follow_directions",
        "ey_map_reading",
        "ey_one_to_one_data",
        "ey_position_words",
        "ey_sort_objects",
        "ey_tally_tables",
        "ey_turns",
        "f10_collect_sort_categorical",
        "f10_geometric_construction_algorithm",
        "f10_grid_reference_pathway",
        "f10_linear_conjecture",
        "f10_partition_tens_ones",
        "f10_stem_and_leaf",
        "f10_symmetry_lines_rotation",
        "f10_trig_ratios_similarity",
        "f10_two_way_table",
        "f10a_conditional_language",
        "f10a_data_display_selection",
        "f10c_congruence_criteria",
        "f10c_quadrilateral_properties",
        "fpm_compound_frequencies",
        "fpm_expected_vs_observed",
        "fpm_function_model",
        "fpm_linear_model",
        "fpm_relative_frequency",
        "fpm_sample_size",
        "fpm_translate_equation",
        "fy1ms_chain_rule",
        "fy1ms_distances_3d",
        "fy1ms_lines_planes",
        "fy1ms_polar_integral",
        "fy1ms_t_test_two_sample",
        "fy1ms_tangent_plane",
        "fy1ms_uniform_exponential",
        "fy1ms_z_test_proportion",
        "fr2a_box_coordinates",
        "fr2a_describe_transform",
        "fr2a_matching_view",
        "fr2a_tessellation_vertex",
        "fr2a_transform_image",
        "fr2b_expanded_notation",
        "fr2b_fraction_decimal_links",
        "fr2b_integer_situations",
        "fr2b_mixed_numerals_number_line",
        "fr2b_numerals_place_value",
        "fr2b_written_division",
        "fr2c_choose_unit_convert",
        "fr2c_clock_elapsed",
        "fr2c_cubic_equation",
        "fr2c_grid_area_perimeter",
        "fr2c_growing_patterns",
        "fr2c_informal_capacity_mass",
        "fr2c_parabola_effects",
        "fr2c_parabola_equation",
        "fr2d_conditional_frequency",
        "fr2d_order_events",
        "fr2d_read_simulation",
        "fr2d_trial_tallies",
        "fr2e_choose_model",
        "fr2e_conjecture",
        "fr2e_money_multistep",
        "fr2e_number_algorithm",
        "fr2e_percent_multistep",
        "fr2e_spatial_algorithm",
        "fraction_decimal_percent",
        "function_evaluation",
        "ir2a_investigation_step_order",
        "ir2a_question_to_data_needed",
        "ir2a_spot_bias",
        "laplace_transform",
        "fy1ca_ivt_bisection",
        "fy1ca_piecewise_continuity",
        "fy1ca_related_rates",
        "fy1ca_applied_optimisation",
        "fy1ca_arc_length",
        "fy1ca_shells_surface",
        "fy1ca_average_value_mvt",
        "fy1ca_hyperbolic",
        "fy1so_taylor_polynomial",
        "fy1so_taylor_remainder",
        "fy1so_series_tests",
        "fy1so_alternating_series",
        "fy1so_ode_integrating_factor",
        "fy1so_ode_separable",
        "fy1so_ode_undetermined",
        "limits",
        "lin_ineq_negative",
        "lin_ineq_one_step",
        "lin_ineq_two_step",
        "lin_ineq_word",
        "line_integral",
        "linear_equation",
        "mat_inv_2x2",
        "mat_inv_3x3",
        "percentage_change",
        "quadratic",
        "r4a_burnside_counting",
        "r4a_cosets_and_index",
        "r4a_dihedral_group_compositions",
        "r4a_group_table_check",
        "r4a_homomorphism_kernel_image",
        "r4a_lagranges_theorem_applications",
        "r4a_orbit_stabiliser_count",
        "r4a_permutation_cycle_notation",
        "r4a_permutation_sign_and_order",
        "r4b_fixed_point_convergence",
        "r4b_gauss_legendre",
        "r4b_lagrange_interpolation",
        "r4b_lu_factorisation",
        "r4b_newton_divided_differences",
        "r4b_power_method",
        "r4b_rk4_step",
        "r4b_secant_iteration",
        "r4c_finite_field_arithmetic_gfp",
        "r4c_first_isomorphism_theorem_orders",
        "r4c_irreducibility_test_over_zp",
        "r4c_normal_subgroup_test",
        "r4c_quotient_group_order_and_structure",
        "r4c_ring_ideal_gcd_in_polynomial_ring",
        "r4c_subgroup_lattice_count",
        "r4c_symmetric_group_conjugacy",
        "r4c_units_and_zero_divisors_zn",
        "r4e_accumulation_function",
        "r4e_curve_features",
        "r4e_hybrid_functions",
        "r4e_power_functions",
        "r4e_riemann_limit",
        "r4e_simpsons_rule",
        "r4f_am_gm",
        "r4f_induction_divisibility",
        "r4f_induction_inequality",
        "r4f_vector_proof",
        "r4g_back_to_back_stem",
        "r4g_grouped_frequency",
        "r4g_longitude_local_time",
        "r4g_moving_average_smoothing",
        "r4g_sector_degrees",
        "r4g_transform_linearity",
        "r4g_two_rate_tariff",
        "r4g_two_timetable_route",
        "r4h_gaussian_partial_pivoting",
        "r4h_improved_euler_step",
        "r4h_interpolation_error_bound",
        "r4h_kkt_conditions",
        "r4h_lagrange_nonlinear_constraint",
        "r4h_matrix_norm_condition",
        "r4h_newton_optimisation",
        "r4h_ode_method_order_stability",
        "r4h_richardson_extrapolation",
        "r4i_anova",
        "r4i_clt_sums",
        "r4i_cramer_rao",
        "r4i_estimator_bias_mse",
        "r4i_gamma_beta",
        "r4i_markov_chebyshev",
        "r4i_poisson_process",
        "r4i_regression_slope",
        "r4j_dijkstra_trace",
        "r4j_forced_recurrence",
        "r4j_generating_function_coefficient",
        "r4j_generating_function_partition_counting",
        "r4j_partial_order_hasse",
        "r4j_tree_counting_prufer",
        "r3b_differentiate_ln",
        "r3b_exact_definite_integral",
        "r3b_integral_properties",
        "r3b_integrate_exponential",
        "r3b_integrate_f_prime_over_f",
        "r3b_integrate_reciprocal",
        "r3b_integrate_trig_powers",
        "r3b_partial_fractions",
        "r3f_ci_coverage",
        "r3f_empirical_rule",
        "r3f_linear_combination",
        "r3f_pdf_cdf",
        "r3f_piecewise_pdf",
        "r3f_test_errors_power",
        "r3g_absolute_value",
        "r3g_ambiguous_case",
        "r3g_compound_angle",
        "r3g_multiple_angle",
        "r3g_product_sum",
        "r3g_quadratic_forms",
        "r3g_t_formulae",
        "r3g_trig_equations",
        "r3h_cooling",
        "r3h_euler_nonlinear",
        "r3h_families",
        "r3h_logistic",
        "r3h_methods_optimisation",
        "r3h_separable_de",
        "r3h_three_variable",
        "r3h_total_distance",
        "r3k_conditional_density",
        "r3k_geometric",
        "r3k_joint_density",
        "r3k_markov",
        "r3k_mgf",
        "r3k_mle",
        "r3k_transformation",
        "s2a1_coordconv",
        "s2a1_cylindrical",
        "s2a1_jacobian_cov",
        "s2a1_region_integral",
        "s2a1_reverse_order",
        "s2a1_spherical",
        "s2a2_conservative3d",
        "s2a2_curl",
        "s2a2_divergence",
        "s2a2_scalar_surface",
        "s2a2_surface_area",
        "s2a2_surface_normal",
        "s2a2_vector_identities",
        "s2a3_divergence_flux",
        "s2a3_flux_graph",
        "s2a3_flux_pieces",
        "s2a3_green_area",
        "s2a3_green_circulation",
        "s2a3_stokes_circulation",
        "s2a3_theorem_verification",
        "s2b1_basis_from_equations",
        "s2b1_change_of_basis_matrix",
        "s2b1_coordinates_in_basis",
        "s2b1_kernel_and_image",
        "s2b1_matrix_of_linear_map",
        "s2b1_subspace_test",
        "s2b2_complex_eigenvalues_2x2",
        "s2b2_diagonalisability_test",
        "s2b2_jordan_form",
        "s2b2_matrix_exponential",
        "s2b2_orthogonal_diagonalisation",
        "s2b2_quadratic_form_definiteness",
        "s2b2_recurrence_diagonalisation",
        "s2b3_gram_schmidt",
        "s2b3_least_squares",
        "s2b3_orthogonal_complement",
        "s2b3_projection",
        "s2b3_qr",
        "s2c1_bernoulli_equation",
        "s2c1_euler_cauchy",
        "s2c1_exact_equation",
        "s2c1_forced_oscillator_resonance",
        "s2c1_homogeneous_substitution",
        "s2c1_second_order_ivp",
        "s2c1_variation_of_parameters",
        "s2c2_frobenius_indicial",
        "s2c2_linear_system_complex_eigen",
        "s2c2_linear_system_real_eigen",
        "s2c2_linear_system_repeated_eigen",
        "s2c2_nonlinear_equilibria_linearisation",
        "s2c2_power_series_solution",
        "s2c3_fourier_half_range",
        "s2c3_fourier_series",
        "s2c3_heat_equation",
        "s2c3_laplace_rectangle",
        "s2c3_laplace_step",
        "s2c3_sturm_liouville",
        "s2c3_wave_equation",
        "senior_dividend_portfolio_table",
        "senior_event_set_notation",
        "senior_percentaged_two_way_table",
        "senior_reducing_balance_schedule",
        "snet_adjacency",
        "snet_degrees",
        "w4_graph_bridges",
        "snet_eulerian",
        "snet_hamiltonian",
        "snet_route_types",
        "smech_connected_particles",
        "smech_equilibrium",
        "smech_inclined_plane",
        "smech_newton_second_law",
        "smech_residual_plots",
        "smech_residuals",
        "sequence_term_from_rule",
        "sr2a_bipartite",
        "sr2a_box_plots",
        "sr2a_histogram_shape",
        "sr2a_network_model",
        "sr2a_planar_faces",
        "sr2a_scatter_assoc",
        "sr2a_scatter_fit",
        "sr2a_time_series",
        "sr2b_composite_domain",
        "sr2b_counting_restrictions",
        "sr2b_exact_trig_values",
        "sr2b_position_time",
        "sr2b_rational_features",
        "sr2b_scale_costs",
        "sr2b_signed_area",
        "sr2b_trig_graph_features",
        "sr2l_apollonius",
        "sr2l_arg_ray",
        "sr2l_bisector",
        "sr2l_circle_centre",
        "sr2l_circle_pick",
        "sr2l_re_im",
        "transformations_invariants",
        "triple_integral",
        "u2_clt_sample_mean",
        "u2_complex_equations",
        "u2_conic_features",
        "u2_functions_injective_surjective",
        "u2_log_differentiation",
        "u2_parametric_polar",
        "u2_sequence_limits",
        "u2_three_set_ie",
        "ucv1_chi_square_test",
        "ucv1_diagonalisation",
        "ucv1_eigenvectors",
        "ucv1_improper_integral",
        "ucv1_lhopital",
        "ucv1_null_space",
        "ucv1_one_sample_t_test",
        "ucv1_power_series_convergence",
        "ucv1_proof_by_induction",
        "ucv1_series_convergence",
        "ucv1_binomial_terms",
        "ucv1_complex_exponential",
        "ucv1_complex_poly_roots",
        "ucv1_equivalence_relations",
        "ucv1_reduction_formulae",
        "ucv1_trig_substitution",
        "ucv1_weak_solution_regularity",
        "salg_binomial_coefficient",
        "salg_binomial_expand",
        "salg_de_moivre_power",
        "salg_polar_multiply_divide",
        "salg_reciprocal_trig_values",
        "salg_trig_identities",
        "y910_classify_number_sets",
        "y910_geometric_proof_reasons",
        "y910_graph_inequality_match",
        "y910_graph_inequality_region",
        "y910_graph_system_solution",
        "y910_line_plane_angle",
        "r3d_shm_calculus",
        "r3d_shm_speed",
        "r3d_accel_position",
        "r3d_accel_velocity",
        "r3d_vertical_resistance",
        "r3d_projectile_cartesian",
        "r3a_binomial_at_least",
        "r3a_binomial_find_np",
        "r3a_binomial_moments",
        "r3a_discrete_unknown",
        "r3a_linear_transform",
        "r3a_sampling_moments",
        "r3c_costing",
        "r3c_dominance",
        "r3c_leslie",
        "r3c_matrix_arithmetic",
        "r3c_permutation",
        "r3c_transition_states",
        "r3c_transition_steady_state",
        "r3e_bnpl",
        "r3e_cpi",
        "r3e_credit_card",
        "r3e_gst_invoice_bas",
        "r3e_hecs_help",
        "r3e_income_tax",
        "r3e_payg_payslip",
        "r3e_superannuation",
        "r3e_utility_bill",
        "r3i_complex_factor",
        "r3i_domain_range",
        "r3i_line_sphere",
        "r3i_parametric_conics",
        "r3i_pigeonhole",
        "r3i_poly_roots",
        "r3i_reflect_rotate",
        "r3i_series_sums",
        "r3j_classify_variable",
        "r3j_heron",
        "r3j_imperial",
        "r3j_recurrence",
        "r3j_salary",
        "r3j_sample_space",
        "r3j_slope_context",
        "r3j_speed",
        "r3l_chromatic_polynomial_hall",
        "r3l_linear_recurrence_closed_form",
        "r3l_lp_corner_points",
        "r3l_lp_duality",
        "r3l_simplex_final_tableau",
        "r3l_simplex_one_pivot",
        "r3l_transportation_assignment",
        "w5b_e_limit_table",
        "w5b_precedence_table",
        # build-b batch
        "w7b_relative_velocity",
        "w7b_unit_cost_depreciation",
        "w7b_number_systems_primes",
        "w7b_floor_ceiling_estimation",
        # w5 algebra batch
        "w5_arithmetic_sequences",
        "w5_check_substitution",
        "w5_domain_range_intro",
        "w5_equivalent_expressions",
        "w5_function_concept",
        "w5_identity_no_solution",
        "w5_inequality_symbols",
        "w5_inverse_operations",
        "w5_meaning_of_equality",
        "w5_proportional_relations",
        "w5_solution_set",
        "w5_trig_all_solutions",
        "w5_variables_roles",
        "w5_quadratic_inequality",
        "w5_rational_inequality",
        "w5_circle_complete_square",
        "w5_circle_forms",
        "w5_circle_point_position",
        "w5_circle_tangent",
        "w5_circle_unknown",
        "w5b_sylow_counting",
        "w5b_mm1_queue",
        "w5b_eoq_inventory",
        "w6_limit_degree",
        "w6_subst_rewrite_x",
        "w6_subst_exp_trig",
        "w6_subst_definite",
        "w6_projectile_resistance",
        "w8_data_acquisition",
        "w8_ogive_read",
        "w8_polygon_read",
        "w9_plot_points",
        "w9_line_draw",
        "w9_numberline_inequality",
        "w9_shade_region",
        "w9_transform_shape",
        "w9_parabola_draw",
        "dr1_line_graph",
        "dr1_graph_from_data",
        "dr1_scatterplot",
        "dr1_time_series",
        "vm_pseudocode_trace",
        "vm_power_graph_features",
    }
)


OPTION_NOTES_SEEDS = 3


def check_option_notes_absent_from_every_topics_output(topic, questions):
    """A topic outside OPTION_NOTES_TOPICS must never carry a top-level
    `option_notes` key. Breaking mathgen/core.py's `to_dict()` pop for it (or
    defaulting it to something non-empty) turns this red for every registered
    topic not listed. A multi-part question serialises each part's own
    `option_notes` inside `parts`, so only the top-level keys are compared.

    `questions`: `make(topic, seed=s)` for s in range(n), n >= `OPTION_NOTES_SEEDS`."""
    carries = False
    for q in take(questions, OPTION_NOTES_SEEDS):
        assert ("option_notes" in q.to_dict()) == ("option_notes" in json.loads(q.to_json()))
        carries = carries or "option_notes" in q.to_dict()
    if topic not in OPTION_NOTES_TOPICS:
        assert not carries, f"{topic} now emits option_notes: add it to OPTION_NOTES_TOPICS"
    else:
        assert carries, f"{topic} no longer emits option_notes at seeds 0-2: remove it from OPTION_NOTES_TOPICS"


def test_question_option_notes_round_trips_like_answer_alt():
    """`option_notes` behaves exactly like `answer_alt` on `Question`:
    omitted from `to_dict()` when empty/unset, present verbatim when set.
    Exercised directly on the dataclass since no generator sets it yet."""
    from mathgen.core import Question

    base = dict(topic="t", level=1, seed=1, text="q", latex="q", answer=1, answer_text="1")
    empty = Question(**base)
    assert "option_notes" not in empty.to_dict()

    noted = Question(**base, option_notes={"41": "wrong-op"})
    assert noted.to_dict()["option_notes"] == {"41": "wrong-op"}


# New topics with no kind/ops variant worth enumerating separately (single
# behaviour, or already covered by NEW_KIND_VARIANTS above) -- just needs to
# be included in the generic per-topic fuzz/determinism/preset coverage.
NEW_FOUNDATION_TOPICS = [
    "order_of_operations",
    "decimals",
    "integer_arithmetic",
    "like_terms",
    "expand_bracket",
    "simplify_expression",
    "expression_parts",
    "expression_from_words",
    "gradient",
    "linear_from_table",
    "sequence",
    "ratio",
    "formula_plugin",
    "word_problem",
    "coordinates",
    "inequality",
    "rearrange",
    "factor_gcf",
    "interpret_notation",
    "like_terms_select",
]


@pytest.mark.parametrize("topic,param,value", NEW_KIND_VARIANTS)
def test_new_kind_variants_fuzz(topic, param, value):
    for seed in range(50):
        q = make_kind(topic, param, value, seed)
        assert q.text and q.latex and q.answer_text
        assert q.answer is not None
        d = json.loads(q.to_json())
        assert d["topic"] == topic


def test_all_20_levels_registered():
    # Levels 1-20 must all be covered; new foundational topics (added after
    # the original 20) may reuse a level 1-10 rather than each needing a
    # unique one, so this checks coverage, not a 1:1 count. Level 21 is the
    # transformation-invariants pair (`f10_*_invariants_build.py`); level 22 is the
    # second-year university generators (`s2c1_*_build.py`).
    assert set(lvl for lvl, _t in topics()) == set(range(1, 23))
    assert len(topics()) == len(REGISTRY)


def fuzz_seeds(topic):
    return N_FUZZ.get(topic, 200)


def check_fuzz_generate_and_verify(topic, questions):
    for seed, q in enumerate(take(questions, fuzz_seeds(topic))):
        # Contract: required fields present and non-empty.
        assert q.text and q.latex and q.answer_text
        assert q.answer is not None
        assert q.level == REGISTRY[topic][0]
        # Contract: JSON round-trips.
        d = json.loads(q.to_json())
        assert d["topic"] == topic and d["seed"] == seed


@pytest.mark.parametrize("topic", TOPICS)
def test_determinism(topic):
    for seed in (0, 99, 12345):
        assert make(topic, seed=seed).to_json() == make(topic, seed=seed).to_json()


DISTRACTOR_SEEDS = 80


def check_distractors_are_valid(topic, questions):
    for q in take(questions, DISTRACTOR_SEEDS):
        # Never colliding with the right answer, never duplicated.
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))


@pytest.mark.parametrize("topic,preset", [(t, p) for t, ps in sorted(PRESETS.items()) for p in sorted(ps)])
def test_presets_generate(topic, preset):
    for seed in range(25):
        q = make(topic, seed=seed, difficulty=preset)
        assert q.difficulty == preset
        assert q.text and q.answer_text


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        make("addition", seed=1, difficulty="nightmare")


def test_explicit_params_override_preset():
    # easy addition capped at 20, but explicit hi/lo wins.
    q = make("addition", seed=1, difficulty="easy", lo=1000, hi=1000)
    assert q.text == "1000 + 1000"


# ---------------------------------------------------------------------------
# Helper reuse. `mathgen/core.py` is the shared toolbox; a contributor who
# re-implements one of its helpers costs the next one the same thinking again. Enforced here for the ONE pattern with proven
# signal, and deliberately not for the four that were measured and rejected.
#
# `simplify(<expr>) == 0` IS `symbolically_equal` / `check_equivalent`, written
# out by hand. 49 occurrences were migrated with an AST rewrite (output proved
# byte-identical) and the last straggler fixed, so there is NO grandfathered
# list here and none is ever to be added: the check fails on the first new one.
#
# MEASURED AND REJECTED, do not add these back:
#   `check(abs(a - b) < tol)` -- 42 generators, every one inspected was a
#       deliberate approximate check with a chosen epsilon (a rounded value, a
#       nudge probe, an off-by-one proof), not symbolic equality.
#   `f"{a} + {b}"` with a possibly-negative b -- 70 matches, and the rendered
#       output is already covered correctly by the `signed_after_operator` text
#       pattern, which sees the actual string rather than guessing from source.
# Both would have been noise, the same verdict the `double_space` pattern got.
_HANDROLLED_MSG = (
    "{where}: `simplify(...) == 0` is `symbolically_equal` hand-rolled.\n"
    "Use `check_equivalent(actual, expected, msg)` to verify an answer, or\n"
    "`symbolically_equal(a, b)` if you need the bool. Both are in mathgen.core.\n"
    "If you need a helper that does NOT exist there, do not write it locally:\n"
    "say so in the pull request instead (see CONTRIBUTING.md, 'Adding a generator')."
)


def _handrolled_equality_sites():
    """Every `simplify(<expr>) == 0` under mathgen/, as "path:line"."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "mathgen"
    sites = []
    for path in sorted(root.rglob("*.py")):
        # core.py DEFINES the helper: `symbolically_equal` is itself
        # `simplify(...) == 0`, so flagging it would flag the fix.
        if path.name == "core.py" and path.parent == root:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
                continue
            left, right = node.left, node.comparators[0]
            if (
                isinstance(left, ast.Call)
                and getattr(left.func, "id", None) == "simplify"
                and isinstance(right, ast.Constant)
                and right.value == 0
            ):
                sites.append(f"{path.relative_to(root.parent).as_posix()}:{node.lineno}")
    return sites


def test_no_hand_rolled_symbolic_equality():
    sites = _handrolled_equality_sites()
    assert not sites, "\n".join(_HANDROLLED_MSG.format(where=s) for s in sites)


def test_the_hand_rolled_detector_can_actually_fire():
    """A detector that cannot fail is not a detector.

    `test_no_hand_rolled_symbolic_equality` passes on a clean tree, which is
    indistinguishable from a broken scan. This feeds the same AST walk a source
    string that does contain the pattern.
    """
    import ast

    def sites_in(src):
        found = []
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
                continue
            left, right = node.left, node.comparators[0]
            if (
                isinstance(left, ast.Call)
                and getattr(left.func, "id", None) == "simplify"
                and isinstance(right, ast.Constant)
                and right.value == 0
            ):
                found.append(node.lineno)
        return found

    assert sites_in("check(simplify(a - b) == 0)\n") == [1]
    # and stays silent on the shapes that were measured and deliberately allowed
    assert sites_in("check(abs(a - b) < 1e-6)\n") == []
    assert sites_in("check(simplify(a - b) == c)\n") == []
    assert sites_in("x = simplify(a - b)\n") == []


# ---------------------------------------------------------------------------
# `answer_form` (None | "expanded" | "factored"). The required
# answer shape is a property of what the question ASKS, declared explicitly on
# the Question -- never inferred from a "(" in the canonical string. Mirrors
# the `option_notes` inertness/round-trip pattern above.

ANSWER_FORM_TOPICS = {
    "expand_bracket": "expanded",
    "expand_binomial_product_monic": "expanded",
    "expand_quadratic_and_cubic_polynomials_from_factors": "expanded",
    "factor_gcf": "factored",
    "factorise_cubic_linear_factor": "factored",
    # difference_of_squares genuinely asks either
    # direction per question -- "expand" or "factorise" -- so its answer_form
    # varies question to question rather than being fixed per topic. A set of
    # allowed values (checked below via `in`) is this dict's one exception to
    # "one topic, one form"; every other topic here still asserts equality.
    "difference_of_squares": {"expanded", "factored"},
    # Both always set "expanded" -- coefficient
    # handling/alignment skills, never a factored-form answer.
    "polynomial_add_subtract": "expanded",
    "polynomial_linear_combination": "expanded",
    # Always "factored" -- the whole point of
    # the skill is stating the factorisation, never the expanded trinomial.
    "factorise_quadratic": "factored",
    # Each of these three topics declares a single fixed answer_form on every
    # question regardless of difficulty tier (unlike difference_of_squares
    # above), so a plain string is correct here, not a set.
    "t1_simplify_surds": "simplified_surd",
    "t1_rationalise_binomial": "rationalised",
    "t1_simplify_algebraic_fraction": "expanded",
    # The exam source is a sum of two
    # rationalised binomial-denominator fractions, combined into one
    # expression -- same declared form as the single-fraction topic above.
    "t1_rationalise_binomial_sum": "rationalised",
    # Exam-style algebra skills. Each declares a single fixed answer_form on every
    # question regardless of tier.
    "exd_e2_algebraic_fraction_mul_div": "expanded",
    "exd_e2_expand_binomial_trinomial": "expanded",
    "exd_e2_simplify_radicals_of_monomials": "expanded",
    "exd_e2_algebraic_fraction_add_sub": "single_fraction",
    # Topics that declare `answer_form` in their generator code (`answer_form="..."`
    # in the module); grep the generator before adding one here.
    "exd_x_a2_complex_quadratic": "expanded",
    "exd_x_f2_cubic_factor": "factored",
    "exd_x_a1_factorise_quadratic": "factored",
    "exd_x_a2_expand_binomial_sums": "expanded",
    "exd_x_a5_expand_monomial_multiplier": "expanded",
    "exd_x_a5_factor_monomial_gcf": "factored",
    "exd_x_a5_fraction_cancel_multistep": "expanded",
    "exd_x_a5_fractional_coefficients": "expanded",
    "exd_x_a5_nested_brackets": "expanded",
    "exd_x_a5_special_products_combined": "expanded",
    "exd_x_a2_expand_cubic_shapes": "expanded",
    "simplify_expression": "expanded",
    # Eigenvector answers are graded up to a scalar multiple, so the question declares it.
    "ucv1_eigenvectors": "scalar_multiple",
    "salg_binomial_expand": "expanded",
}

# Topics that set `answer_form` on only some of their question kinds: when it is
# present it must be one of these values, and it may be absent.
ANSWER_FORM_SOMETIMES = {
    "exd_x_f2_compose": {"rationalised", "single_fraction"},
}


ANSWER_FORM_SEEDS = 3


def check_answer_form_only_present_for_declared_topics(topic, questions):
    """Every OTHER registered topic must never carry `answer_form` in its
    output -- proof this can fail: with ANSWER_FORM_TOPICS emptied, the six
    topics that do set it immediately turn this red.

    `questions`: `make(topic, seed=s)` for s in range(n), n >= `ANSWER_FORM_SEEDS`."""
    expected_present = topic in ANSWER_FORM_TOPICS
    for q in take(questions, ANSWER_FORM_SEEDS):
        # A multi-part question serialises each part's own `answer_form` inside
        # `parts`, so the top-level keys are what is compared.
        present = "answer_form" in q.to_dict()
        assert present == ("answer_form" in json.loads(q.to_json())), topic
        if topic in ANSWER_FORM_SOMETIMES:
            if present:
                assert q.to_dict()["answer_form"] in ANSWER_FORM_SOMETIMES[topic], topic
            continue
        assert present == expected_present, topic
        if expected_present:
            expected = ANSWER_FORM_TOPICS[topic]
            actual = q.to_dict()["answer_form"]
            if isinstance(expected, set):
                assert actual in expected, topic
            else:
                assert actual == expected, topic


def test_question_answer_form_round_trips_like_answer_alt():
    """`answer_form` behaves exactly like `answer_alt`/`option_notes` on
    `Question`: omitted from `to_dict()` when unset, present verbatim when
    set. Exercised directly on the dataclass, independent of any generator."""
    from mathgen.core import Question

    base = dict(topic="t", level=1, seed=1, text="q", latex="q", answer=1, answer_text="1")
    unset = Question(**base)
    assert unset.answer_form is None
    assert "answer_form" not in unset.to_dict()
    assert "answer_form" not in unset.to_json()

    declared = Question(**base, answer_form="factored")
    assert declared.to_dict()["answer_form"] == "factored"
    assert '"answer_form": "factored"' in declared.to_json()


@pytest.mark.parametrize(
    "topic,difficulty",
    [
        ("expand_bracket", "easy"),
        ("expand_bracket", "medium"),
        ("expand_bracket", "hard"),
        ("expand_binomial_product_monic", "easy"),
        ("expand_binomial_product_monic", "medium"),
        ("expand_binomial_product_monic", "hard"),
        ("expand_quadratic_and_cubic_polynomials_from_factors", "easy"),
        ("expand_quadratic_and_cubic_polynomials_from_factors", "medium"),
        ("expand_quadratic_and_cubic_polynomials_from_factors", "hard"),
    ],
)
def test_expanded_generators_reject_the_question_copied_verbatim(topic, difficulty):
    """The three false-accept skills (false-accept cases): a learner who copies
    the factored question into the answer box must be rejected. Reproduced
    with `sympy.factor(canonical)` (never a regex over the prompt -- a regex
    over "Expand 4(5x - 7)" drops the leading coefficient and misses
    expand-single-brackets entirely) against the real grader."""
    import sympy
    from mathgen.grading import typed as grading

    params = {"kind": "single"} if topic == "expand_bracket" else {}
    for seed in range(10):
        q = make(topic, seed=seed, difficulty=difficulty, **params)
        assert q.answer_form == "expanded"
        factored = str(sympy.factor(sympy.sympify(q.answer)))
        result = grading.grade("expression", q, factored)
        if factored == str(sympy.sympify(q.answer)):
            continue  # already-factored canonical (e.g. constant/monomial); nothing to reject
        assert not result.correct, (
            f"{topic}/{difficulty} seed={seed}: factored copy {factored!r} of canonical "
            f"{q.answer_text!r} must be rejected"
        )


@pytest.mark.parametrize(
    "topic,params",
    [
        ("factor_gcf", {}),
        ("factor_gcf", {"var_factor": True}),
        ("factorise_cubic_linear_factor", {}),
    ],
)
def test_factoring_generators_reject_expanded_form(topic, params):
    """The two factoring skills still reject the fully expanded
    form -- plain sympy equivalence would otherwise accept it, which defeats
    a "factorise this" question. The companion contract -- accepting an equal
    ALTERNATIVE factorisation such as x(x^2 + 5x + 4) for x^3 + 5x^2 + 4x, not
    just the exact canonical spelling -- is proved directly against the
    grader."""
    import sympy
    from mathgen.grading import typed as grading

    for seed in range(10):
        q = make(topic, seed=seed, **params)
        assert q.answer_form == "factored"
        expanded = str(sympy.expand(sympy.sympify(q.answer)))
        assert not grading.grade("expression", q, expanded).correct, (
            f"{topic} seed={seed}: expanded form {expanded!r} of {q.answer_text!r} must be rejected"
        )
