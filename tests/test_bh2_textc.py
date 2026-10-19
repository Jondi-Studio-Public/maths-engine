"""Regression tests for the student-text defects fixed by the bug-hunt-2026-10-19 textc lane.

Each test renders skills over several seeds and asserts on the text a learner reads (after ``paper_notation``).
"""

from __future__ import annotations

import re

import pytest

from mathgen import engine
from mathgen.core import display_question, paper_notation

_SKILLS = {s.id: s for s in engine.load_skills()[1]}
_SEEDS = range(4)


def _texts(skill_id: str, *, working_only: bool = False):
    """Every learner-visible string of every source of a skill: (source key, seed, kind, text)."""
    skill = _SKILLS[skill_id]
    for src in skill.sources:
        key = engine._key_of(skill, src)
        for seed in _SEEDS:
            q = engine.make_for_skill(skill, seed, source=src)
            shown = display_question(q)
            rows = [("text", shown["text"]), ("key", shown["answer_text"])]
            rows += [("option", o) for o in shown["options"]]
            rows += [("working", w) for w in shown["working"]]
            for part in shown.get("parts", []):
                rows += [("text", part["text"]), ("key", part["answer_text"])]
                rows += [("option", o) for o in part["options"]]
                rows += [("working", w) for w in part["working"]]
            for kind, text in rows:
                if not working_only or kind == "working":
                    yield key, seed, kind, text


# Skills whose working was empty on every seed of some tier.
_EMPTY_WORKING = [
    "partial-derivative-x",
    "partial-derivative-y",
    "solve-ode-separable-poly",
    "solve-ode-exponential",
    "solve-simultaneous-equations",
    "matrix-solve-system",
    "matrix-determinant-3x3",
    "moment-of-a-force",
    "senior-consider-constant-and-non-constant-force",
    "senior-examine-momentum-force-resultant-force-action-and-reaction",
    "senior-use-implicit-differentiation-to-determine-the-gradient-of",
]


@pytest.mark.parametrize("skill_id", _EMPTY_WORKING)
def test_every_tier_has_a_worked_solution(skill_id):
    skill = _SKILLS[skill_id]
    for src in skill.sources:
        for seed in _SEEDS:
            q = engine.make_for_skill(skill, seed, source=src)
            if q.parts:
                assert all(p.working for p in q.parts), (skill_id, engine._key_of(skill, src), seed)
            else:
                assert q.working, (skill_id, engine._key_of(skill, src), seed)


# Noise a learner should never meet: double signs, "+-", "1x", "0x", leaked None, Python "**", singular/plural and
# article slips, and LaTeX braces after a caret.
_NOISE = re.compile(
    r"- -\d|\+ -\d|--|\+-|\b1x\b(?!\^)|\b0x\b|\bNone\b|\*\*|[^\w\s]\{[^}]*\}"
    r"|\b1 (?:days|parts|points|pieces)\b|\ba 8\d?\b|\^\{"
)

_CLEAN_SKILLS = [
    "beam-equilibrium-seesaw",
    "condition-number-diagonal",
    "definite-integral",
    "directional-derivative",
    "hookes-law",
    "jacobian-two-variable",
    "limit-removable",
    "matrix-determinant-2x2",
    "matrix-determinant-3x3",
    "matrix-solve-system",
    "moment-of-a-force",
    "partial-derivative-x",
    "partial-derivative-y",
    "senior-consider-constant-and-non-constant-force",
    "senior-consider-position-of-vectors-as-a-function-of",
    "senior-examine-momentum-force-resultant-force-action-and-reaction",
    "senior-use-forward-and-backward-scanning-to-determine-the",
    "senior-use-implicit-differentiation-to-determine-the-gradient-of",
    "senior-use-matrices-for-storing-and-displaying-information-that",
    "senior-use-partial-fractions-where-necessary-for-integration-in",
    "solve-ode-exponential",
    "solve-ode-separable-poly",
    "solve-simultaneous-equations",
    "square-pyramid-volume-and-surface-area",
    "strain-from-extension",
    "stress-from-force-area",
    "uc2-irreducibility-test-over-zp",
    "uc2-multivariable-taylor-hessian",
    "uc2-nonhomogeneous-recurrence-polynomial-or-exponential-forcing",
    "ucv1-average-value-mvt",
    "ucv1-completeness-cauchy-check",
    "ucv1-complex-equations-division",
    "ucv1-contour-integral-residue",
    "ucv1-coordinate-basis",
    "ucv1-derivative-definition",
    "ucv1-diagonal-operator-norm",
    "ucv1-eigenvectors",
    "ucv1-finite-subgroup-test",
    "ucv1-fourier-transform-gaussian",
    "ucv1-general-solution-parametric",
    "ucv1-heat-equation-mode",
    "ucv1-inverse-regularised-solve",
    "ucv1-isomorphism-invariant",
    "ucv1-iterative-linear-solver-step",
    "ucv1-lhopital-rule",
    "ucv1-linear-oscillator-classification",
    "ucv1-manifold-chart-jacobian",
    "ucv1-markov-hitting-probability",
    "ucv1-ode-separable-product",
    "ucv1-piecewise-continuity",
    "ucv1-second-order-characteristic-roots",
    "ucv1-sequence-limits-squeeze-monotone",
    "ucv1-series-comparison-tests",
    "ucv1-simplex-homology",
    "ucv1-stochastic-euler-step",
    "ucv1-structure-classification",
    "ucv1-symmetry-conservation",
    "ucv1-taylor-polynomial",
    "ucv1-topology-basis-membership",
    "ucv1-weak-solution-regularity",
    "youngs-modulus",
]


@pytest.mark.parametrize("skill_id", _CLEAN_SKILLS)
def test_no_double_signs_or_coefficient_noise(skill_id):
    for key, seed, kind, text in _texts(skill_id):
        match = _NOISE.search(text)
        assert match is None, (skill_id, key, seed, kind, text)


@pytest.mark.parametrize("skill_id", ["ucv1-residue-simple-pole", "ucv1-contour-integral-residue"])
def test_imaginary_unit_is_lower_case_i(skill_id):
    for key, seed, kind, text in _texts(skill_id):
        assert not re.search(r"(?<![\w^])I(?![\w])", text), (skill_id, key, seed, kind, text)


def test_definite_integral_working_has_no_dict():
    for key, seed, kind, text in _texts("definite-integral", working_only=True):
        assert "{" not in text and "'p'" not in text, (key, seed, text)


def test_partial_fractions_writes_ln_not_log():
    for key, seed, kind, text in _texts("senior-use-partial-fractions-where-necessary-for-integration-in"):
        assert "log(" not in text, (key, seed, kind, text)


def test_square_pyramid_answer_has_no_identifier():
    for key, seed, kind, text in _texts("square-pyramid-volume-and-surface-area"):
        assert "surface_area" not in text, (key, seed, kind, text)


@pytest.mark.parametrize(
    "raw, shown",
    [
        ("7*ln|x| + C", "7 ln|x| + C"),
        ("-7*ln|4*x + 9|/4 + C", "-7 ln|4x + 9|/4 + C"),
        ("2*x + 16*ln|x - 6| + C", "2x + 16 ln|x - 6| + C"),
    ],
)
def test_log_of_absolute_value_is_a_function_call(raw, shown):
    assert paper_notation(raw) == shown


def test_taylor_question_ends_with_question_mark():
    skill = _SKILLS["ucv1-taylor-polynomial"]
    for src in skill.sources:
        for seed in range(12):
            q = engine.make_for_skill(skill, seed, source=src)
            if q.text.startswith("What is"):
                assert q.text.rstrip().endswith("?"), q.text


def test_eigenvector_working_has_no_unit_coefficients():
    for key, seed, kind, text in _texts("ucv1-eigenvectors", working_only=True):
        assert not re.search(r"\b1[λIv]\b|[+-] 1[λIv]\b|= -1v", text), (key, seed, text)
