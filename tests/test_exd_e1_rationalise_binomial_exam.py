"""Tests for `rationalise-binomial-denominator`'s exam source: a SUM of two independent binomial-denominator
fractions, each rationalised by its own conjugate, combined into one single simplified expression.

Generator: `mathgen.generators.t1_rationalise_binomial_build.rationalise_binomial_sum`, registered under the
NEW topic `t1_rationalise_binomial_sum`. The new topic is checked directly by the tests here.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import prompt_shape

TOPIC = "t1_rationalise_binomial_sum"
SEEDS = range(60)
WIDE_SEEDS = range(200)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# Two templates (`_SUM_TEMPLATES` in the generator), both:
# "1/(a1 [+-] √b1) + 1/(a2 [+-] √b2)" then either "by rationalising each
# denominator. Give your answer as a single simplified expression with a
# rational denominator." or "as a single fraction with a rational
# denominator."
_QUESTION_RE = re.compile(
    r"^(?:Simplify|Write) 1/\((\d+) ([+-]) √(\d+)\) \+ 1/\((\d+) ([+-]) √(\d+)\) "
    r"(?:by rationalising each denominator\. Give your answer as a single simplified "
    r"expression with a rational denominator\.|as a single fraction with a rational "
    r"denominator\.)$"
)


def _all_questions(seeds=SEEDS):
    return [make(TOPIC, seed=s) for s in seeds]


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS)


def _denominator_has_surviving_radical(expr) -> bool:
    """True if `sympy.fraction(expr)`'s denominator contains a `Pow` with a non-integer rational exponent
    anywhere in it. Built from scratch here, never imported from `mathgen.grading` or from the generator
    module -- the independent-oracle discipline."""
    _, denominator = sympy.fraction(expr)
    for node in sympy.preorder_traversal(denominator):
        if isinstance(node, sympy.Pow):
            exponent = node.exp
            if getattr(exponent, "is_Rational", False) and not getattr(exponent, "is_Integer", False):
                return True
    return False


def _oracle_value(a1, s1, b1, a2, s2, b2):
    """Independently recomputes the combined value straight from the two
    ORIGINAL fractions parsed out of `question.text` -- never touches the
    generator's own D1/D2/const/coef1/coef2 arithmetic, and does not import
    `mathgen.generators.t1_rationalise_binomial_build`."""
    return sympy.Rational(1) / (sympy.Integer(a1) + s1 * sympy.sqrt(b1)) + sympy.Rational(1) / (
        sympy.Integer(a2) + s2 * sympy.sqrt(b2)
    )


def _parse_question_text(text: str):
    m = _QUESTION_RE.match(text)
    assert m, f"unrecognised prompt shape {text!r}"
    a1, sign1_text, b1, a2, sign2_text, b2 = m.groups()
    sign1 = 1 if sign1_text == "+" else -1
    sign2 = 1 if sign2_text == "+" else -1
    return int(a1), sign1, int(b1), int(a2), sign2, int(b2)


class TestRationaliseBinomialSumExam:
    def test_determinism_across_seeds(self):
        for seed in range(40):
            q1 = make(TOPIC, seed=seed)
            q2 = make(TOPIC, seed=seed)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors
            assert q1.answer_form == q2.answer_form

    def test_question_text_is_recognised_shape(self):
        for q in _all_questions(WIDE_SEEDS):
            assert _QUESTION_RE.match(q.text), f"unrecognised prompt shape {q.text!r}"

    def test_answer_form_is_rationalised(self):
        for q in _all_questions():
            assert q.answer_form == "rationalised", f"{q.text!r} should be answer_form=rationalised"

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for q in _all_questions():
            assert len(q.distractors) > 0, f"no distractors for {q.text!r}"
            for d in q.distractors:
                assert d != q.answer_text, f"distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"duplicate distractors in {q.text!r}"

    def test_three_misconceptions_present_every_question(self):
        # All three candidates are constructed unconditionally every draw;
        # confirm pick_distractors never had to drop one over a wide sweep.
        for q in _all_questions(WIDE_SEEDS):
            assert len(q.distractors) == 3, f"expected 3 distractors, got {q.distractors} for {q.text!r}"

    def test_real_grader_accepts_canonical_answer(self):
        for q in _all_questions():
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"grader rejected canonical {q.answer_text!r} for {q.text!r}"

    def test_real_grader_rejects_every_distractor(self):
        for q in _all_questions(WIDE_SEEDS):
            for d in q.distractors:
                result = GRADERS["expression"](q, d)
                assert not result.correct, f"grader accepted distractor {d!r} for {q.text!r}"

    def test_real_grader_rejects_deliberately_wrong_answer(self):
        for q in _all_questions(range(20)):
            wrong = "(9999 + 9999sqrt(9999) - 9999sqrt(9998))/9999"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    def test_at_least_20_distinct_questions(self):
        texts = {q.text for q in _all_questions(WIDE_SEEDS)}
        assert len(texts) >= 20, f"only {len(texts)} distinct questions (need >= 20) over {len(WIDE_SEEDS)} seeds"

    def test_answer_and_radical_distractors_formatted_alike(self):
        # Every option carrying a radical is shown "(numerator)/denominator",
        # the same shape as the answer -- the half-rationalised distractor is
        # the one deliberate exception (it is NOT a single fraction, which is
        # exactly its own misconception, so it is exempt from this shape
        # check by definition and covered by its own test below instead).
        for q in _all_questions():
            answer_has_paren = q.answer_text.startswith("(")
            for d in q.distractors:
                if "1/(" in d:
                    continue  # the half-rationalised distractor
                assert d.startswith("(") == answer_has_paren, (
                    f"option {d!r} does not match answer shape {q.answer_text!r} for {q.text!r}"
                )

    def test_prompt_variety(self):
        # Hand-asserted: at least 3 distinct prompt shapes over 60 seeds.
        shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
        assert len(shapes) >= 3, f"only {len(shapes)} distinct prompt shapes in 60 seeds (prompt variety requires >= 3)"

    def test_skill_loads_with_expected_exam_source(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "rationalise-binomial-denominator")
        assert skill.domain == "secondary-number"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False

        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1, f"expected exactly one exam source, got {exam_sources}"
        exam_source = exam_sources[0]
        assert exam_source.topic == TOPIC
        assert exam_source.stages == ["exam"], f"exam source must be stages=[exam], got {exam_source.stages}"

        # The three original sources are untouched.
        other_sources = [s for s in skill.sources if s.difficulty != "exam"]
        assert len(other_sources) == 3
        for s in other_sources:
            assert s.topic == "t1_rationalise_binomial"
        assert {s.difficulty for s in other_sources} == {"easy", "medium", "hard"}

        misconception_ids = {m.id for m in skill.misconceptions}
        assert {"half-rationalised", "wrong-conjugate-sign-one-term", "mismatched-denominator"} <= misconception_ids


class TestHalfRationalisedDistractor:
    """The 'applied the conjugate to only one of the two fractions'
    misconception is deliberately VALUE-EQUAL to the canonical answer (a
    rationalised fraction and its un-rationalised original are the same real
    number) and is caught only by `answer_form="rationalised"`'s form check,
    never by value comparison. Proved by construction, since a
    does-not-simplify-shaped candidate is allowed here."""

    def _half_rationalised(self, q):
        candidates = [d for d in q.distractors if "1/(" in d]
        assert len(candidates) == 1, f"expected exactly one half-rationalised candidate in {q.distractors}"
        return candidates[0]

    def test_half_rationalised_is_value_equal_to_canonical(self):
        for q in _all_questions(range(30)):
            half = self._half_rationalised(q)
            answer_value = _parse(q.answer_text)
            half_value = _parse(half)
            assert sympy.simplify(answer_value - half_value) == 0, (
                f"half-rationalised {half!r} should be value-equal to canonical {q.answer_text!r} "
                f"by construction for {q.text!r}"
            )

    def test_half_rationalised_still_rejected_by_real_grader(self):
        for q in _all_questions(range(30)):
            half = self._half_rationalised(q)
            result = GRADERS["expression"](q, half)
            assert not result.correct, (
                f"real grader accepted the half-rationalised (value-equal, wrong-form) "
                f"candidate {half!r} for {q.text!r} -- answer_form=rationalised should reject it"
            )


class TestIndependentOracle:
    """From-scratch recomputation of the combined value straight from the two original fractions parsed out of
    `question.text` -- never imports `mathgen.generators.t1_rationalise_binomial_build`."""

    def test_oracle_agrees_with_answer_text(self):
        examined = 0
        challenged = []
        for q in _all_questions(SEEDS):
            examined += 1
            a1, s1, b1, a2, s2, b2 = _parse_question_text(q.text)
            expected = _oracle_value(a1, s1, b1, a2, s2, b2)
            given = _parse(q.answer_text)
            if sympy.simplify(expected - given) != 0:
                challenged.append({"issue": f"oracle={expected}, answer_text stated {q.answer_text}", "text": q.text})
                continue
            if _denominator_has_surviving_radical(given):
                challenged.append(
                    {
                        "issue": "answer_text's denominator still carries a radical",
                        "text": q.text,
                        "answer_text": q.answer_text,
                    }
                )
        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"rationalise-binomial-denominator (sum) oracle: examined {examined}, challenged 0")

    def test_oracle_fires_on_corruption(self):
        # The independent check above is not vacuous: a deliberately
        # corrupted value (the true combined value plus 1) must disagree
        # with the oracle's own from-scratch recomputation -- proving the
        # `sympy.simplify(expected - given) != 0` branch in the test above
        # is reachable, not dead code that would pass on any input.
        checked = 0
        for q in _all_questions(range(10)):
            a1, s1, b1, a2, s2, b2 = _parse_question_text(q.text)
            expected = _oracle_value(a1, s1, b1, a2, s2, b2)
            given = _parse(q.answer_text)
            assert sympy.simplify(expected - given) == 0  # sanity: the real answer agrees first
            corrupted = given + 1
            assert sympy.simplify(expected - corrupted) != 0, (
                f"corrupted value {corrupted} should disagree with the oracle for {q.text!r}"
            )
            checked += 1
        assert checked >= 8, f"only {checked} corruption cases exercised (need enough to trust the proof)"


class TestGeneratorInternalChecksFireOnCorruption:
    """`check_equivalent`/`check_distinct` inside the generator itself raise
    `VerificationError` on a genuine mismatch -- exercised directly against
    `mathgen.core`, not just trusted because the un-corrupted path passes."""

    def test_check_equivalent_raises_on_mismatch(self):
        from mathgen.core import check_equivalent

        try:
            check_equivalent(sympy.Integer(1), sympy.Integer(2), "deliberate mismatch")
        except VerificationError:
            pass
        else:
            raise AssertionError("check_equivalent should have raised on a genuine mismatch")

    def test_check_distinct_raises_when_a_candidate_equals_the_answer(self):
        from mathgen.core import check_distinct

        try:
            check_distinct(sympy.Integer(5), sympy.Integer(6), sympy.Integer(5), msg="deliberate collision")
        except VerificationError:
            pass
        else:
            raise AssertionError("check_distinct should have raised when a candidate equals the answer")
