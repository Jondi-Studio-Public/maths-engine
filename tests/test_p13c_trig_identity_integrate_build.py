"""Tests `p13c_trig_identity_integrate` (ACMSM116): integration using trigonometric identities."""

import re

import pytest
from sympy import Symbol, sympify

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check, check_equivalent

SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p13c_trig_identity_integrate"
SKILL = "senior-integrate-using-the-trigonometric-identities-mathrm-s-mathrm"

_grade_expression = GRADERS["expression"]

x = Symbol("x")


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = _grade_expression(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: canonical {q.answer_text!r} not accepted"
        for d in q.distractors:
            result = _grade_expression(q, d)
            assert not result.correct, f"{tier} seed={seed}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_value(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        wrong = f"{q.answer} + 999999 + C"
        result = _grade_expression(q, wrong)
        assert not result.correct, f"{tier} seed={seed}: deliberately wrong {wrong!r} accepted"


# --- The +C grading finding, proven directly through the real grader -------
# (mathgen/grading/typed.py::_expected_expression's docstring): the DISPLAYED
# answer_text ("F(x) + C") is authoritative whenever it differs from the
# internal `answer` (bare F(x)) by exactly one free symbol -- so a learner
# who drops "+ C" is compared against the full "F(x) + C" and correctly
# rejected, not silently accepted against the bare internal value. This is
# a SEPARATE generator module from
# `p13c_antideriv_notation_build.py`; the finding is re-confirmed here, not
# assumed to carry over.


@pytest.mark.parametrize("tier", TIERS)
def test_plus_c_grading_finding_direct(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # The canonical, WITH "+ C", is accepted.
        assert _grade_expression(q, q.answer_text).correct, f"{tier} seed={seed}: canonical with +C not accepted"
        # The bare internal answer, WITHOUT "+ C", is REJECTED -- the
        # "dropped +C" misconception firing through the real grader, not
        # just in-generator.
        assert not _grade_expression(q, q.answer).correct, (
            f"{tier} seed={seed}: bare answer (missing +C) was wrongly accepted"
        )
        # Any single constant-looking letter ("D") stands for the constant when the answer has one;
        # a number or a second term in its place is still wrong.
        wrong_letter = f"{q.answer} + D"
        assert _grade_expression(q, wrong_letter).correct == q.answer_text.rstrip().endswith("+ C"), (
            f"{tier} seed={seed}: +D graded wrongly"
        )
        assert not _grade_expression(q, f"{q.answer} + 5").correct, f"{tier} seed={seed}: +5 accepted as the constant"
        checked += 1
    assert checked == len(SEEDS)


# --- sec/tan parsing finding, confirmed directly against the real grader ---
# (mathgen/grading/typed.py::_parse_expression has no restricted local_dict/
# global_dict, so every sympy function name -- including sec -- parses
# through it). This generator never emits `sec` in a canonical answer or
# distractor (every antiderivative here is a sin/cos/tan/x expression), so
# the finding is recorded for the record rather than load-bearing, but it is
# still confirmed directly rather than assumed.


def test_sec_and_tan_parse_through_the_real_grader():
    from mathgen.grading.typed import _parse_expression
    import sympy

    tan_expr = _parse_expression("tan(x) - x + C")
    assert tan_expr is not None
    sec_expr = _parse_expression("sec(x)**2")
    assert sec_expr is not None
    # sec(x)**2 == 1 + tan(x)**2 (the very identity this skill teaches) --
    # confirms `sec` parses to the real sympy function, not a bare symbol
    # "sec" multiplied implicitly by "(x)**2".
    assert sympy.simplify(sec_expr - (1 + sympy.tan(sympy.Symbol("x")) ** 2)) == 0


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "should fail")
    with pytest.raises(VerificationError):
        check(False, "should fail")


# --- Distinct rendered questions -------------------------------------------


# --- Answer kind: an expression, with the answer satisfying the notation ---


@pytest.mark.parametrize("tier", TIERS)
def test_answer_text_is_the_answer_plus_c(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert q.answer_text.endswith(" + C"), (
            f"{tier} seed={seed}: answer_text {q.answer_text!r} does not end in ' + C'"
        )
        answer_part = q.answer_text[: -len(" + C")]
        assert sympify(answer_part.replace("^", "**")) == sympify(q.answer.replace("^", "**")), (
            f"{tier} seed={seed}: answer_text {q.answer_text!r} does not match answer {q.answer!r}"
        )
        # The internal answer never itself carries "+ C" or "C" as a token.
        parsed = sympify(q.answer.replace("^", "**"))
        assert Symbol("C") not in parsed.free_symbols, (
            f"{tier} seed={seed}: internal answer unexpectedly carries a free C"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_formatted_like_the_answer(tier):
    # Every distractor is an expression string (optionally with "+ C"), same
    # shape family as the canonical -- never a bare number, never wrapped in
    # a different label. Distractors are VALUE strings (only "**" rewritten
    # to "^", a bare "*" left alone), never PROMPT strings.
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert re.match(r"^-?[0-9A-Za-z*/().\s+^-]+$", d), (
                f"{tier} seed={seed}: distractor not expression-shaped: {d!r}"
            )
            assert "**" not in d, f"{tier} seed={seed}: distractor still has Python ** operator: {d!r}"


# --- Boundary against integrate-trig: every question here has a squared or
# tan^2 trig term, never a bare first-power sin(kx)/cos(kx) integrand -----


@pytest.mark.parametrize("tier", TIERS)
def test_every_question_has_a_squared_or_tan_squared_term(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "^2(" in q.text, f"{tier} seed={seed}: no squared trig term in prompt: {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_never_a_bare_linear_sin_cos_integrand(tier):
    """Boundary against integrate-trig (skills/integration/integrate-trig.yaml,
    "Find ∫(25cos(5x)) dx" -- a bare first-power sin/cos integrand, no
    identity needed). This generator's integrand string is always built via
    `_sq_disp`/`join_terms` with an explicit "^2(" marker; a bare
    "sin(kx)"/"cos(kx)" with no "^2" would be that skill's territory, never
    produced here.
    """
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        # Every sin(/cos(/tan( occurrence in the integrand must be
        # immediately preceded by "^2" (the squared marker), i.e. no bare
        # first-power trig call appears in the prompt's integrand.
        integrand_part = re.search(r"∫ \((.*)\) dx", q.text)
        assert integrand_part is not None, f"{tier} seed={seed}: no ∫(...) dx integrand found"
        integrand_text = integrand_part.group(1)
        for fn in ("sin(", "cos(", "tan("):
            for m in re.finditer(re.escape(fn), integrand_text):
                preceding = integrand_text[: m.start()]
                assert preceding.endswith("^2"), (
                    f"{tier} seed={seed}: bare (non-squared) {fn} in integrand: {integrand_text!r}"
                )


# --- Tier-specific identity shape checks ------------------------------------


def test_easy_tier_is_a_single_sin_or_cos_squared_term():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert ("sin^2(" in q.text) != ("cos^2(" in q.text), (
            f"seed={seed}: easy tier should have exactly one of sin^2/cos^2: {q.text!r}"
        )
        assert "tan^2(" not in q.text
        assert "+" not in re.search(r"∫ \((.*)\) dx", q.text).group(1)
        checked += 1
    assert checked == len(SEEDS)


def test_medium_tier_is_a_single_tan_squared_term():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        assert "tan^2(" in q.text
        assert "sin^2(" not in q.text and "cos^2(" not in q.text
        checked += 1
    assert checked == len(SEEDS)


def test_hard_tier_combines_sin_squared_and_cos_squared_with_shared_frequency():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert "sin^2(" in q.text and "cos^2(" in q.text and "tan^2(" not in q.text, (
            f"seed={seed}: hard tier should combine sin^2 and cos^2 only: {q.text!r}"
        )
        sin_freq = re.search(r"sin\^2\((\d*)x\)", q.text).group(1) or "1"
        cos_freq = re.search(r"cos\^2\((\d*)x\)", q.text).group(1) or "1"
        assert sin_freq == cos_freq, f"seed={seed}: hard tier's sin^2/cos^2 should share one frequency: {q.text!r}"
        checked += 1
    assert checked == len(SEEDS)


def test_exam_tier_mixes_cos_squared_and_tan_squared_with_two_frequencies():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "cos^2(" in q.text and "tan^2(" in q.text and "sin^2(" not in q.text, (
            f"seed={seed}: exam tier should mix cos^2 and tan^2 only: {q.text!r}"
        )
        cos_freq = re.search(r"cos\^2\((\d*)x\)", q.text).group(1) or "1"
        tan_freq = re.search(r"tan\^2\((\d*)x\)", q.text).group(1) or "1"
        assert cos_freq != tan_freq, f"seed={seed}: exam tier should use TWO DIFFERENT frequencies: {q.text!r}"
        checked += 1
    assert checked == len(SEEDS)


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["integrate-polynomial"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == ("exd_x_r3e_trig_identity_integral" if source.difficulty == "exam" else TOPIC)
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
    assert skill.sources[3].difficulty == "exam"
