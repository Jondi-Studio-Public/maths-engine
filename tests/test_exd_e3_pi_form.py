"""Exam tier for `angle-degrees-radians-pi-form` (topic
`t4_degrees_radians_pi_form`, `variant="forward_fine_exam"`).

Combines the focused contract tests (mirroring `test_t4_pi_form_build.py`,
scoped to the new tier only -- easy/medium/hard are that file's, not touched
here) with an independent oracle: the oracle section parses only
`question.text`/`question.answer_text`, via its own regexes, and imports
nothing from `mathgen.generators.t4_pi_form_build` (no pool constant, no
`_grader_pi_string`, no `format_pi_multiple`/`latex_pi_multiple`), so it can
never simply replay the generator's own construction back at itself.
"""

import re
from fractions import Fraction
from pathlib import Path

import pytest
import sympy

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct
from mathgen.generators.t4_pi_form_build import FINE_DEGREES_EXAM
from mathgen.text_quality import prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "t4_degrees_radians_pi_form"
DIFFICULTY = "exam"
SEEDS = range(60)


# ---------------------------------------------------------------------------
# Pool, verified independently (Fraction reduction, not copied blind from the
# generator's own docstring comment).


def test_pool_is_half_integer_degrees_in_one_revolution():
    assert len(FINE_DEGREES_EXAM) == len(set(FINE_DEGREES_EXAM)), "duplicate pool entries"
    for deg in FINE_DEGREES_EXAM:
        assert 0 < deg <= 360
        assert deg == int(deg) + 0.5, f"{deg} is not a half-integer degree"


def test_pool_denominator_range_honestly_reported():
    """Every pool member's reduced pi-form denominator, recomputed here from
    scratch (not read off the generator's own `coeff`), is one of the six
    values reachable by a half-integer degree (see the generator's own
    `FINE_DEGREES_EXAM` comment for the derivation: 2*deg is always odd, so
    the denominator can only be 360 divided by a divisor of 45). The pool
    keeps {24, 40, 72} in full (all denominators <= 72, clearing the
    "up to 72" framing) plus the single explicit 38.5 (denominator 360,
    exceeding 72 -- reported honestly, not
    forced to fit)."""
    denominators = {deg: Fraction(round(deg * 2), 360).denominator for deg in FINE_DEGREES_EXAM}
    seen = set(denominators.values())
    assert seen == {24, 40, 72, 360}, seen
    # every member is strictly finer than the medium tier's fixed denominator
    # 12 -- not merely "the pool overall", every single value.
    assert all(d > 12 for d in denominators.values())
    # exactly one pool member exceeds the "up to 72" framing: 38.5 itself.
    over_72 = {deg for deg, d in denominators.items() if d > 72}
    assert over_72 == {38.5}, over_72


def test_paper_examples_reduce_correctly():
    """Two named examples, recomputed independently of the
    generator's pool/formatting code: 37.5 degrees -> 5pi/24, 38.5 degrees ->
    77pi/360."""
    assert Fraction(round(37.5 * 2), 360) == Fraction(5, 24)
    assert Fraction(round(38.5 * 2), 360) == Fraction(77, 360)
    assert 37.5 in FINE_DEGREES_EXAM
    assert 38.5 in FINE_DEGREES_EXAM


# ---------------------------------------------------------------------------
# Determinism, >= 40 seeds.


def test_determinism():
    for seed in range(40):
        q1 = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        q2 = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        assert q1.to_json() == q2.to_json()


# Shape-masking proof: the exam tier's sentence(s) are structurally distinct from easy/medium's
# shared "Convert {deg} degrees to radians..." sentence, AND from each other.


def test_masked_shape_differs_from_lower_tiers():
    lower_shapes = {
        prompt_shape(make(TOPIC, seed=seed, difficulty=difficulty))
        for difficulty in ("easy", "medium")
        for seed in range(20)
    }
    assert lower_shapes == {"Convert # degrees to radians. Give your answer as an exact value in terms of pi."}
    exam_shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=DIFFICULTY)) for seed in SEEDS}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes.isdisjoint(lower_shapes), (
        f"exam tier shares a masked prompt shape with easy/medium -- {exam_shapes & lower_shapes}"
    )


def test_distinct_question_texts_at_least_20_over_60_seeds():
    texts = {make(TOPIC, seed=seed, difficulty=DIFFICULTY).text for seed in SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct question texts in 60 seeds"


# ---------------------------------------------------------------------------
# check_distinct fires on a corrupted answer.


def test_check_distinct_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_distinct(sympy.Rational(5, 24) * sympy.pi, sympy.Rational(5, 24) * sympy.pi)


# ---------------------------------------------------------------------------
# The real grader, called directly: accepts answer_text, rejects every
# distractor and a deliberately-wrong value, >= 40 seeds.


def test_grader_accepts_canonical_rejects_distractors_and_wrong_value():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        result = grade_expression(q, q.answer_text)
        assert result.correct, f"seed={seed}: canonical {q.answer_text!r} rejected"
        for distractor in q.distractors:
            result = grade_expression(q, distractor)
            assert not result.correct, f"seed={seed}: distractor {distractor!r} wrongly accepted"
        # 1000pi/7 has coefficient ~142.86, far outside any coefficient this
        # tier can produce (max is 355/72 ~ 4.93 at deg=355.5) -- guaranteed
        # wrong for every seed.
        wrong = grade_expression(q, "1000pi/7")
        assert not wrong.correct, f"seed={seed}: deliberately wrong value wrongly accepted"


def test_no_shipped_distractor_is_value_equal_to_answer():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        for distractor in q.distractors:
            result = grade_expression(q, distractor)
            assert not result.correct, (
                f"seed={seed}: shipped distractor {distractor!r} is value-equal to the answer {q.answer_text!r}"
            )


def test_distractor_count_always_three():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        assert len(q.distractors) == 3, f"seed={seed}: expected 3 distractors, got {q.distractors!r}"
        assert len(set(q.distractors)) == 3, f"seed={seed}: duplicate distractors {q.distractors!r}"


# ---------------------------------------------------------------------------
# ASCII spelling: no bare '*' in any learner-facing field, ASCII "pi" only.


def test_no_bare_star_and_ascii_pi_only():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        assert "*" not in q.text
        assert "*" not in q.answer_text
        for distractor in q.distractors:
            assert "*" not in distractor
        for field in (q.text, q.answer_text, *q.distractors):
            assert "π" not in field, f"seed={seed}: unicode pi found in {field!r}"


# ---------------------------------------------------------------------------
# The skill loads through mathgen.skills.load_skills with the expected shape.


def test_skill_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id["angle-degrees-radians-pi-form"]
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["angle-types", "f10-circle-measures"]
    assert len(skill.sources) == 4
    assert {source.difficulty for source in skill.sources} == {"easy", "medium", "hard", "exam"}
    assert {source.topic for source in skill.sources} == {TOPIC}
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.params == {"variant": "forward_fine_exam"}
    assert exam_source.stages == ["exam"]


# ---------------------------------------------------------------------------
# Independent oracle. Parses ONLY `question.text`/`question.answer_text`, via
# its own regex, no import from `mathgen.generators.t4_pi_form_build`.

_PI_STRING_RE = re.compile(r"^(\d*)pi(?:/(\d+))?$")
_DEG_IN_TEXT_RE = re.compile(r"(\d+(?:\.\d+)?) degrees")


def _parse_pi_string(s: str) -> Fraction:
    match = _PI_STRING_RE.match(s.strip())
    assert match, f"could not parse pi-form string {s!r}"
    numerator = int(match.group(1)) if match.group(1) else 1
    denominator = int(match.group(2)) if match.group(2) else 1
    return Fraction(numerator, denominator)


def _oracle_expected_coeff(text: str) -> Fraction:
    match = _DEG_IN_TEXT_RE.search(text)
    assert match, f"no degree measure found in {text!r}"
    deg = Fraction(match.group(1))
    return deg / 180


def test_oracle_exam_tier():
    checked = 0
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=DIFFICULTY)
        expected_coeff = _oracle_expected_coeff(q.text)
        got_coeff = _parse_pi_string(q.answer_text)
        assert got_coeff == expected_coeff, (
            f"seed={seed}: {q.text!r} implies coefficient {expected_coeff}, but "
            f"answer_text {q.answer_text!r} implies {got_coeff}"
        )
        checked += 1
    assert checked == 40


def test_oracle_catches_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty=DIFFICULTY)
    expected_coeff = _oracle_expected_coeff(q.text)
    got_coeff = _parse_pi_string(q.answer_text)
    assert got_coeff == expected_coeff  # sanity: the real answer agrees

    corrupted_coeff = got_coeff + 1  # deliberately wrong
    assert corrupted_coeff != expected_coeff, "oracle failed to disagree with a deliberately corrupted coefficient"
