"""The exam source on ``senior-review-indices-including-fractional-indices-and-the-index`` (topic
``index_laws_review``, variant ``exam_algebraic_fraction``).

Exam target: ``((256x**4)/(81x**8))**(3/4)`` -- an algebraic
fraction (numerator and denominator both a numeric perfect-power coefficient
times a power of ``x``) raised to a fractional index. The independent oracle
below parses only what a learner reads (``question.text``), rebuilds the
ORIGINAL nested expression with a positive symbol and lets sympy's own
simplification derive the answer -- it never imports anything from
``mathgen.generators.p3c_indices_build``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from sympy import Rational, Symbol, simplify, sympify

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "index_laws_review"
DIFFICULTY = "exam"
SEEDS = range(60)

# The nested-fraction pattern a rendered exam question always carries, e.g.
# "((256x^4)/(81x^8))^(3/4)". Used both by the oracle below and by the
# "typed the original expression back" grader check -- extracted from
# `question.text`, the surface a learner actually reads, never from the
# generator's own internals.
_QUESTION_RE = re.compile(r"\(\((\d+)x\^(\d+)\)/\((\d+)x\^(\d+)\)\)\^\((\d+)/(\d+)\)")


def _extract(question):
    m = _QUESTION_RE.search(question.text)
    assert m, f"question text does not match the expected nested-fraction shape: {question.text!r}"
    num_coef, a, den_coef, b, n, d = (int(v) for v in m.groups())
    return num_coef, a, den_coef, b, n, d


# --- independent oracle -------------------------------------------------
#
# Parses `question.text` only, rebuilds the ORIGINAL unsimplified nested
# expression with a POSITIVE symbol (the standard convention for this class
# of "simplify to a single power" item -- see the generator's own docstring
# for why an assumption-free x leaves a genuine branch-cut ambiguity), and
# lets sympy's own simplification derive the answer -- a route independent
# of the generator's backwards construction.


def _oracle_answer(question):
    num_coef, a, den_coef, b, n, d = _extract(question)
    xp = Symbol("x", positive=True)
    original = ((num_coef * xp**a) / (den_coef * xp**b)) ** Rational(n, d)
    return simplify(original)


def test_oracle_matches_generator_answer_every_seed():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        oracle = _oracle_answer(question)
        # Re-sympify the oracle's OWN string, fresh, with no positivity
        # assumption -- the same thing `mathgen.grading._expected_expression`
        # does to `question.answer` -- so the comparison is apples to
        # apples: two plain, assumption-free `x` symbols. Comparing the
        # positive-assumption oracle object directly against `question.answer`
        # sympified fresh would spuriously disagree: sympy treats
        # `Symbol("x", positive=True)` and `Symbol("x")` as different symbols
        # even though both display as "x".
        oracle_replayed = sympify(str(oracle))
        generated = sympify(question.answer)
        assert simplify(oracle_replayed - generated) == 0, (
            f"seed {seed}: oracle {oracle} disagrees with generator answer {generated} for {question.text!r}"
        )


def test_independent_check_fires_on_corruption_not_on_clean_seed():
    """`check_equivalent` (the same helper the generator itself calls) must
    raise on a corrupted pair and must NOT raise on a genuinely matching
    pair, for the same expression shape this generator builds."""
    xp = Symbol("x", positive=True)
    original = (625 * xp**4 / (81 * xp**8)) ** Rational(3, 4)
    correct = Rational(125, 27) * xp**-3
    corrupted = Rational(125, 27) * xp**-2  # wrong exponent
    check_equivalent(original, correct, "should not raise: this pair is genuinely equal")
    with pytest.raises(VerificationError):
        check_equivalent(original, corrupted, "should raise: this pair is genuinely different")


# --- determinism / distractor hygiene ------------------------------------


def test_generator_is_deterministic_across_seeds():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        repeat = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        assert question.to_json() == repeat.to_json()


def test_distractors_distinct_nonempty_never_equal_answer():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) == 3


# --- real grader: accept/reject, unconditionally over every seed --------


def test_real_grader_accepts_canonical_every_seed():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"seed {seed}: canonical {question.answer_text!r} rejected"


def test_real_grader_rejects_every_distractor_every_seed():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"seed {seed}: distractor {distractor!r} incorrectly accepted"


def test_real_grader_rejects_a_deliberately_wrong_answer_every_seed():
    """A value-wrong answer (answer's coefficient off by one, same exponent)
    must be rejected on every seed."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        ans = sympify(question.answer)
        wrong = str(ans + 1)  # shifts the value; never accidentally equal
        result = grade_expression(question, wrong)
        assert not result.correct, f"seed {seed}: deliberately wrong {wrong!r} accepted"


def test_real_grader_rejects_the_typed_original_unsimplified_expression_every_seed():
    """Checked directly against the real grader, unconditionally, over every seed -- not sampled, because sympy's
    ability to auto-combine ``(x**e)**(n/d)`` for an assumption-free ``x`` can depend on whether the inner
    exponent ``e`` is even or odd."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        num_coef, a, den_coef, b, n, d = _extract(question)
        original_typed = f"(({num_coef}x^{a})/({den_coef}x^{b}))^({n}/{d})"
        result = grade_expression(question, original_typed)
        assert not result.correct, (
            f"seed {seed}: typed original unsimplified expression {original_typed!r} "
            f"was ACCEPTED -- contradicts the skill's own 'single simplified power' "
            f"contract; this generator relies on this rejection instead of declaring "
            f"answer_form"
        )


# --- exactness: no residual surd, integer exponent, reduced coefficient --


def test_answer_is_an_exact_integer_power_no_residual_surd_every_seed():
    x_sym = sympify("x")
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        ans = sympify(question.answer)
        assert ans.free_symbols <= {x_sym}
        # No sqrt/irrational leftover: every numeric atom is rational.
        for atom in ans.atoms():
            if atom.is_number:
                assert atom.is_rational, f"seed {seed}: irrational atom {atom} in {ans}"
        # Structural check: rebuild as coefficient * x**exponent and confirm
        # the x exponent is a plain integer (never a bare surd/fractional
        # power of x reaching the learner).
        coeff, rest = ans.as_coeff_Mul()
        if rest == 1:
            exponent = 0
        else:
            base, exponent = rest.as_base_exp()
            assert base == x_sym, f"seed {seed}: unexpected base {base} in {ans}"
        assert exponent.is_Integer, f"seed {seed}: non-integer x exponent {exponent} in {ans}"


def test_no_bare_x_to_the_power_one_in_rendered_text():
    """A learner never sees 'x^1' on a
    real exam paper. base_exp is drawn from [2, 3] specifically to avoid it."""
    bare_x1 = re.compile(r"x\^1(?!\d)")
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        assert not bare_x1.search(question.text), f"seed {seed}: {question.text!r}"


# --- text quality / variety ----------------------------------------------


def test_find_text_defects_empty_every_seed():
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        defects = find_text_defects(question)
        assert defects == [], f"seed {seed}: {defects} for {question.text!r}"


def test_at_least_three_distinct_prompt_shapes():
    shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=DIFFICULTY)) for seed in SEEDS}
    assert len(shapes) >= 3, f"only {len(shapes)} distinct shapes: {sorted(shapes)}"


def test_at_least_20_distinct_rendered_questions():
    texts = {make(TOPIC, seed=seed, difficulty=DIFFICULTY).text for seed in SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct rendered questions over {len(SEEDS)} seeds"


def test_exam_wrong_root_misconception_is_exercised_when_d_is_4():
    """`exam-wrong-root-index` (the "fourth root mistaken for a square root"
    misconception) is only exact when d == 4 (see the generator's own
    docstring: (num_coef/den_coef)**(n/2) instead of the correct n/d).
    Confirm it is genuinely exercised -- appears among some d == 4
    question's actual rendered distractors -- not dead weight in the skill
    YAML that `pick_distractors(n=3)` always drops."""
    found_d4 = False
    exercised = False
    for seed in range(200):
        question = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        num_coef, _a, den_coef, _b, n, d = _extract(question)
        if d != 4:
            continue
        found_d4 = True
        wrong_root_value = Rational(num_coef, den_coef) ** Rational(n, 2)
        for distractor in question.distractors:
            expr = sympify(distractor.replace("^", "**"))
            coeff, _rest = expr.as_coeff_Mul()
            if coeff == wrong_root_value:
                exercised = True
    assert found_d4, "no d == 4 seed found in range(200)"
    assert exercised, "exam-wrong-root-index never appeared in a d == 4 seed's distractors in range(200)"


# --- skill contract --------------------------------------------------------


def test_skill_contract_has_four_sources_with_exam_tier():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["senior-review-indices-including-fractional-indices-and-the-index"]

    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources} == {TOPIC}
    exam_source = skill.sources[-1]
    assert exam_source.difficulty == "exam"
    assert exam_source.stages == ["exam"]
    assert exam_source.params == {"variant": "exam_algebraic_fraction"}
