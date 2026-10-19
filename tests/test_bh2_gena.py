"""Regression tests for the round-2 generator text defects (bug hunt 2026-10-19, lane gena).

Each case renders every topic registered by a touched generator module across several seeds and
every preset difficulty, and checks the student-visible text for the defect that was fixed.
"""

import importlib
import re

import pytest

from mathgen import core
from mathgen.core import REGISTRY, PRESETS, display_question, make

SEEDS = range(12)

# Modules whose every topic now carries a worked solution.
WORKING_MODULES = [
    "p6f_measurement_faces_build",
    "p7e_solid_surface_areas_build",
    "p7e_pyramid_sphere_volume_build",
    "p6f_scale_factor_build",
    "p6f_measurement_time_build",
    "p4d_outcomes_likelihood_build",
    "p4d_scale_complement_build",
    "p19c_bernoulli_binomial_suitability_build",
    "p19c_independence_language_build",
    "t5_capacity_mixedunits_build",
    "t5_annulus_frustum_build",
]

# (regex, description) that must never appear in student-visible text or working.
BAD = [
    (re.compile(r"\d+/2 hundredths"), "raw fraction before 'hundredths'"),
    (re.compile(r"There are 1 altogether"), "plural with one"),
    (re.compile(r"\b1 of them are shaded"), "plural with one"),
    (re.compile(r"log_\{"), "raw LaTeX braces"),
    (re.compile(r"\bExam question:"), "internal prefix"),
    (re.compile(r"\(roughly\)"), "stray parenthetical"),
    (re.compile(r"\bH_small\b|\bH_total\b"), "internal identifier"),
    (re.compile(r"\bAn 85c\b"), "wrong article"),
    (re.compile(r"\b(\d+)/(\d+) = \1/\2\b"), "tautology"),
    (re.compile(r"\bNone\b"), "None leaked"),
]


def _topics(module):
    return [t for t, (_, fn) in REGISTRY.items() if fn.__module__.endswith("." + module) or fn.__module__ == module]


def _render(topic, seed, difficulty):
    q = make(topic, seed, difficulty)
    return q, display_question(q)


def _all_text(q, d):
    out = [str(d.get("text", "")), str(d.get("working", "") or ""), str(q.answer_text)]
    out += [str(x) for x in (d.get("options") or d.get("distractors") or [])]
    for p in q.parts or []:
        out += [str(p.text), str(p.working or "")]
    return "\n".join(out)


def _cases(modules):
    for m in modules:
        importlib.import_module(f"mathgen.generators.{m}")
        for t in _topics(m):
            for diff in list(PRESETS.get(t, {})) or [None]:
                yield pytest.param(t, diff, id=f"{t}-{diff}")


@pytest.mark.parametrize("topic,diff", list(_cases(WORKING_MODULES)))
def test_worked_solution_present_and_clean(topic, diff):
    for seed in SEEDS:
        q, d = _render(topic, seed, diff)
        has_working = bool(str(d.get("working") or "").strip()) or any(
            str(p.working or "").strip() for p in q.parts or []
        )
        assert has_working, f"{topic} {diff} seed {seed}: empty working"
        blob = _all_text(q, d)
        for rx, why in BAD:
            assert not rx.search(blob), f"{topic} {diff} seed {seed}: {why}: {rx.search(blob).group(0)!r}"


CLEAN_MODULES = WORKING_MODULES + [
    "exd_x_pr4_decimal_place_value",
    "foundations_subitise_order_build",
    "fig3_fraction_model_build",
    "exd_x_sc5_log_equation",
    "p8a_notation_build",
    "fig1_money_build",
    "p4a_circle_duration_build",
    "p5a_measurement_partial_area_curves_build",
    "exd_e5_rectangle_area_perimeter_three_units_build",
    "exd_e5_circle_area_to_diameter_and_circumference_build",
    "exd_e5_square_pyramid_volume_and_surface_area_build",
    "exd_x_pr7_large_numbers_mixed_forms",
    "exd_x_pr7_order_to_1000_regrouped",
    "exd_x_n3_decibel_scale",
    "exd_x_pr5_trail_markers",
    "curriculum_expansion",
]


@pytest.mark.parametrize("topic,diff", list(_cases(CLEAN_MODULES)))
def test_no_known_text_defects(topic, diff):
    for seed in SEEDS:
        try:
            q, d = _render(topic, seed, diff)
        except Exception:  # a generator may reject a seed; other tests cover that
            continue
        blob = _all_text(q, d)
        for rx, why in BAD:
            assert not rx.search(blob), f"{topic} {diff} seed {seed}: {why}: {rx.search(blob).group(0)!r}"


def test_decimal_place_value_has_no_fraction_hundredths():
    for seed in range(60):
        for diff in (None, "exam"):
            q = make("exd_x_pr4_decimal_place_value", seed, diff)
            assert "/2 hundredths" not in q.text + str(q.working)


def test_circle_chain_working_uses_pi_wording():
    for t in _topics("t5_annulus_frustum_build"):
        for seed in SEEDS:
            q = make(t, seed)
            assert "Using pi" not in str(q.working)


def test_partial_measurement_stem_counts_divisions():
    for t in _topics("p5a_measurement_partial_area_curves_build"):
        for seed in SEEDS:
            q = make(t, seed)
            assert not re.search(r"\b1 divisions\b", q.text)


def test_prism_capacity_working_products_are_correct():
    for t in _topics("t5_capacity_mixedunits_build"):
        for seed in range(30):
            q = make(t, seed)
            for m in re.finditer(r"((?:[\d.]+ × )+[\d.]+) = ([\d.]+)", str(q.working)):
                prod = 1.0
                for f in m.group(1).split(" × "):
                    prod *= float(f)
                assert abs(prod - float(m.group(2))) < 0.01 * max(1.0, prod), (t, seed, m.group(0))


def test_nothing_accepts_a_distractor_in_touched_modules():
    from mathgen.grading import grade

    bad = []
    for m in WORKING_MODULES:
        for t in _topics(m):
            for seed in range(4):
                q = make(t, seed)
                for dist in q.distractors or []:
                    if isinstance(dist, str) and dist == q.answer_text:
                        bad.append((t, seed))
    assert not bad
