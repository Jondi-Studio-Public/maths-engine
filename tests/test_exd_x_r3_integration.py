"""Round-3 integration checks across all 65 exam chains (`exd_x_r3*`): the pins for the integrator's fixes.

* the misconception catalogue the chains share has no catch-all ids and every tag a chain uses exists;
* no two parts of one question share an answer (a repeat is a free mark, and lets a chain be answered by copying);
* a degenerate reply (every part 0, every part 1, the first key repeated) earns nothing a part's own key would not;
* the fixes made in integration: the 3 x 3 determinant draw that left (d) without distractors, the map scale whose
  length in cm repeated the length in km, an even difference of degrees that repeated (b) as (c), a flow upgrade that
  changed nothing, and the vowel sentence in a stem about copies of one letter.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pytest

from mathgen import make
from mathgen.generators._exd_r3_distinct import answers_distinct
from mathgen.grading import typed
from mathgen.grading.parts import grade_parts
from mathgen.misconceptions import CATALOGUE

TOPICS = [
    "exd_x_r3f_quotient_ring_roots",
    "exd_x_r3b_second_derivative",
    "exd_x_r3e_inverse_trig_tangent",
    "exd_x_r3a_right_triangle_chain",
    "exd_x_r3a_overtime_week",
    "exd_x_r3e_fee_discount_gst",
    "exd_x_r3c_network_float",
    "exd_x_r3d_compound_two_readings",
    "exd_x_r3e_assignment_reduction",
    "exd_x_r3f_network_flow",
    "exd_x_r3c_annuity_payout",
    "exd_x_r3b_arith_seq",
    "exd_x_r3d_bounce_series_chain",
    "exd_x_r3a_connector_reroute",
    "exd_x_r3b_critical_path",
    "exd_x_r3c_loan_first_period",
    "exd_x_r3b_simple_interest",
    "exd_x_r3c_trig_identity_integral",
    "exd_x_r3d_riemann_error_chain",
    "exd_x_r3f_inverse_trig_chain",
    "exd_x_r3e_trig_identity_integral",
    "exd_x_r3d_limit_growth",
    "exd_x_r3c_cramer_line",
    "exd_x_r3a_determinant_chain",
    "exd_x_r3b_det3",
    "exd_x_r3e_product_measure_complement",
    "exd_x_r3b_gauss_seidel",
    "exd_x_r3f_initial_value_chain",
    "exd_x_r3a_survey_conditional",
    "exd_x_r3d_combinations_selection",
    "exd_x_r3b_complex_conjugate",
    "exd_x_r3c_complex_cube",
    "exd_x_r3d_complex_product_modulus",
    "exd_x_r3e_complex_quotient",
    "exd_x_r3e_code_counting",
    "exd_x_r3f_permutation_roles",
    "exd_x_r3c_semicircle_triangle",
    "exd_x_r3f_complex_parallelogram",
    "exd_x_r3c_dose_tablets",
    "exd_x_r3a_complex_powers",
    "exd_x_r3b_nth_roots",
    "exd_x_r3c_club_overlaps",
    "exd_x_r3d_quotient_limits_chain",
    "exd_x_r3f_meridian_voyage",
    "exd_x_r3a_bearings_triangle",
    "exd_x_r3a_vowel_arrangements",
    "exd_x_r3d_tangent_pair_chain",
    "exd_x_r3e_circle_centre_chain",
    "exd_x_r3f_rhombus_chain",
    "exd_x_r3a_cyclic_chain",
    "exd_x_r3b_factorial",
    "exd_x_r3e_map_route",
    "exd_x_r3b_similar_scale",
    "exd_x_r3a_ftc_power_chain",
    "exd_x_r3d_regression_summary_chain",
    "exd_x_r3b_proportion_ci",
    "exd_x_r3a_stochastic_two_step",
    "exd_x_r3e_vector_combination",
    "exd_x_r3c_unit_vector_perp",
    "exd_x_r3f_vector_perpendicular",
    "exd_x_r3c_vector_resultant",
    "exd_x_r3d_vector_parallel_scalar",
    "exd_x_r3e_vector_points",
    "exd_x_r3f_vector_curve",
    "exd_x_r3d_vector_projection",
]

ROOT = Path(__file__).resolve().parent.parent
SEEDS = range(1, 61)
DEGENERATE_SEEDS = range(1, 16)

#: Ids removed or merged in integration: a generator must not tag with them again.
RETIRED = {
    "multiplied-instead-of-divided",
    "adds-instead-of-multiplies",
    "gave-total-not-difference",
    "reversed-sign",
    "wrong-axis-component",
    "difference-sign-reversed",
    "difference-taken-backwards",
    "vector-difference-order-reversed",
    "i-squared-taken-as-one",
    "forgot-i-squared-is-minus-one",
    "complement-reported-for-probability",
    "complement-reported-for-proportion",
    "gave-complement-instead-of-event",
    "failing-count-reported",
    "complement-count-reported-for-at-least-one",
    "reports-est-not-float",
    "summed-all-durations",
    "restricted-position-ignored",
    "lower-limit-forgotten",
    "evaluation-not-reduced-mod-p",
    "reads-imaginary-coefficient-as-real-part",
    "reported-real-part-for-imaginary",
    "vector-reported-as-position",
}


def test_catalogue_has_65_topics_and_no_retired_ids():
    assert len(TOPICS) == 65 and len(set(TOPICS)) == 65
    assert not RETIRED & set(CATALOGUE)


def test_no_chain_generator_still_names_a_retired_id():
    unknown = {}
    for path in glob.glob(str(ROOT / "mathgen" / "generators" / "*r3*.py")):
        text = Path(path).read_text()
        for tag in re.findall(r'["\']([a-z][a-z0-9]*(?:-[a-z0-9]+)+)["\']', text):
            if tag in RETIRED:
                unknown[tag] = Path(path).name
    assert not unknown


@pytest.mark.parametrize("topic", TOPICS)
def test_no_two_parts_share_an_answer(topic):
    for seed in SEEDS:
        q = make(topic, seed=seed)
        assert answers_distinct(q.parts), (topic, seed, q.answer_text)


@pytest.mark.parametrize("topic", TOPICS)
def test_every_tag_used_is_in_the_catalogue(topic):
    for seed in SEEDS:
        for part in make(topic, seed=seed).parts:
            for tag in (part.option_notes or {}).values():
                assert tag in CATALOGUE, (topic, tag)
                assert tag not in RETIRED, (topic, tag)


@pytest.mark.parametrize("topic", TOPICS)
def test_degenerate_replies_earn_nothing_a_part_key_would_not(topic):
    for seed in DEGENERATE_SEEDS:
        q = make(topic, seed=seed)
        labels = [p.label for p in q.parts]
        n = len(labels)
        replies = (
            ["0"] * n,
            ["1"] * n,
            ["2"] * n,
            ["-1"] * n,
            [str(q.parts[0].answer_text)] * n,
            [str(q.parts[-1].answer_text)] * n,
        )
        for reply in replies:
            result = grade_parts(q, dict(zip(labels, reply)))
            for part, given in zip(q.parts, reply):
                earned = result.parts[part.label].marks
                if earned > 0:
                    assert typed.grade(part.answer_kind, part, given).correct, (topic, seed, part.label, reply)


def test_3x3_determinant_always_has_distractors_for_the_singular_k():
    for seed in (39, 688, 768, 1032, 1929, *range(1, 400)):
        make("exd_x_r3b_det3", seed=seed)


def test_map_route_second_scale_never_makes_cm_equal_km():
    for seed in range(1, 200):
        q = make("exd_x_r3e_map_route", seed=seed)
        assert "1 : 100000. Find its length" not in q.text


def test_limit_growth_difference_of_degrees_is_odd_so_c_differs_from_b():
    for seed in range(1, 120):
        q = make("exd_x_r3d_limit_growth", seed=seed)
        b, c = (p.answer for p in q.parts[1:])
        assert b != c and b == -c


def test_flow_upgrade_always_raises_the_maximum_flow():
    for seed in range(1, 150):
        q = make("exd_x_r3f_network_flow", seed=seed)
        assert int(q.parts[3].answer_text) > 0


def test_vowel_sentence_only_when_the_restricted_group_is_the_vowels():
    seen = set()
    for seed in range(1, 150):
        q = make("exd_x_r3a_vowel_arrangements", seed=seed)
        about_vowels = "vowels" in q.parts[1].prompt or "vowels" in q.parts[0].prompt
        assert ("are vowels and the others are consonants" in q.stem) == about_vowels, q.text
        seen.add(about_vowels)
    assert seen == {True, False}
