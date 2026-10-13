"""Focused contracts for the bounded Fourier/dynamics university batch."""

from __future__ import annotations

import yaml

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make
from mathgen.core import PRESETS, REGISTRY


BATCH = {
    "fourier_coefficient_simple": "expression",
    "constant_coefficient_pde_classification": "mcq",
    "autonomous_equilibrium_stability": "mcq",
    "simple_harmonic_angular_frequency": "number",
    "wave_frequency_from_speed_wavelength": "number",
}


def test_batch_is_registered_deterministic_and_gradable():
    for topic, kind in BATCH.items():
        assert topic in REGISTRY
        assert set(PRESETS[topic]) == {"easy", "medium", "hard"}
        for seed in range(40):
            first = make(topic, seed=seed)
            second = make(topic, seed=seed)
            assert first.to_json() == second.to_json()
            assert first.answer_text and first.answer_text not in first.distractors
            assert len(first.distractors) >= 2
            assert len(first.distractors) == len(set(first.distractors))
            given = str(grading.mcq_options(first).index(first.answer_text)) if kind == "mcq" else first.answer_text
            assert grading.grade(kind, first, given).correct


def test_all_difficulty_presets_generate():
    for topic in BATCH:
        for difficulty in ("easy", "medium", "hard"):
            for seed in range(12):
                question = make(topic, seed=seed, difficulty=difficulty)
                assert question.difficulty == difficulty
                assert question.answer is not None
                assert ",;" not in question.latex
                assert ",quad" not in question.latex


def test_classification_generators_cover_all_outcomes():
    pde = {make("constant_coefficient_pde_classification", seed=seed).answer_text for seed in range(120)}
    stability = {make("autonomous_equilibrium_stability", seed=seed).answer_text for seed in range(120)}
    assert pde == {"Elliptic", "Parabolic", "Hyperbolic"}
    assert stability == {"Stable (attracting)", "Unstable (repelling)"}


def test_batch_skills_are_active_aligned_and_reachable():
    _domains, loaded = skills_mod.load_skills()
    by_id = {skill.id: skill for skill in loaded}
    expected = {
        "fourier-coefficient-simple": ("fourier_coefficient_simple", {"fourier-analysis-and-pdes"}),
        "classify-constant-coefficient-pde": ("constant_coefficient_pde_classification", {"fourier-analysis-and-pdes"}),
        "autonomous-equilibrium-stability": (
            "autonomous_equilibrium_stability",
            {"ordinary-differential-equations", "modelling-and-dynamical-systems"},
        ),
        "simple-harmonic-angular-frequency": (
            "simple_harmonic_angular_frequency",
            {"ordinary-differential-equations", "mathematical-physics"},
        ),
        "wave-frequency-from-speed-wavelength": ("wave_frequency_from_speed_wavelength", {"mathematical-physics"}),
    }
    for skill_id, (topic, areas) in expected.items():
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert set(skill.framework_areas) == areas
        assert all(source.topic == topic for source in skill.sources if source.difficulty != "exam")
        assert all(prerequisite in by_id for prerequisite in skill.prerequisites)
        assert {alignment.area for alignment in skill.alignments} == areas
        assert all(alignment.framework == "university-core-v1" for alignment in skill.alignments)
