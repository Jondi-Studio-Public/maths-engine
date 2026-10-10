"""Tests `simplify-surds`: answers, rejection of a learner's own wrong form and skill loading.

Tests for simplify-surds."""

import re

import pytest
import sympy
from sympy import sqrt

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent
from mathgen.generators.t1_simplify_surds_build import SQUAREFREE, largest_square_factor

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)
PROOF_SEEDS = range(60)  # 60 seeds, matching the module's own sweep width.

_EASY_RE = re.compile(r"^Simplify √(\d+)\. Give your answer in the form a√b\.$")
_MEDIUM_RE = re.compile(r"^Simplify √(\d+) \+ √(\d+)\. Give your answer in the form a√b\.$")
_HARD_RE = re.compile(r"^Simplify (\d+)√(\d+) - (\d+)√(\d+)\. Give your answer in the form a√b\.$")
# Every option is str(sympy_expr) of either a plain integer, "N*sqrt(M)"
# (coefficient magnitude != 1), or "sqrt(M)"/"-sqrt(M)" (sympy drops an
# explicit +-1 coefficient on its own str()) -- never a stray "**", "^" or
# unicode radical.
_OPTION_RE = re.compile(r"^-?\d+$|^-?\d+\*sqrt\(\d+\)$|^-?sqrt\(\d+\)$")


def _all_questions(tier, seeds=SEEDS):
    return [make("t1_simplify_surds", seed=s, difficulty=tier) for s in seeds]


def _doubled(answer_text: str) -> str:
    """A deliberately wrong but well-FORMED answer: twice the canonical value.
    Always a genuinely different value (the canonical is never zero) and,
    since sympy's own str() re-simplifies the doubled expression, always
    lands back in the same clean str(sympy_expr) shape."""
    return str(sympy.sympify(answer_text.replace("^", "**")) * 2)


class TestSimplifySurds:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("t1_simplify_surds", seed=seed, difficulty=tier)
            q2 = make("t1_simplify_surds", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors
            assert q1.answer_form == q2.answer_form

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for q in _all_questions(tier):
            assert len(q.distractors) == 2, f"{tier}: expected 2 distractors for {q.text!r}, got {q.distractors}"
            for d in q.distractors:
                assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_form_is_simplified_surd(self, tier):
        for q in _all_questions(tier):
            assert q.answer_form == "simplified_surd", f"{tier}: {q.text!r} has answer_form {q.answer_form!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_answer(self, tier):
        for q in _all_questions(tier):
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"{tier}: grader rejected canonical {q.answer_text!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_every_distractor(self, tier):
        for q in _all_questions(tier):
            for d in q.distractors:
                result = GRADERS["expression"](q, d)
                assert not result.correct, f"{tier}: grader accepted distractor {d!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_deliberately_wrong_answer(self, tier):
        for q in _all_questions(tier, seeds=range(10)):
            wrong = _doubled(q.answer_text)
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_options_formatted_alike(self, tier):
        # No option identifiable by shape alone beyond what the skill itself
        # teaches: every option is str(sympy_expr), never a stray "**"/"^" or
        # the unicode radical (that only ever appears in prompt text).
        for q in _all_questions(tier):
            options = [q.answer_text] + list(q.distractors)
            for o in options:
                assert _OPTION_RE.match(o), f"{tier}: option {o!r} not in expected str(sympy_expr) shape: {q.text!r}"
                assert "**" not in o and "^" not in o and "√" not in o, f"{tier}: malformed option {o!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_satisfies_its_own_declared_form(self, tier):
        # The canonical answer_text itself must already be fully simplified:
        # any sqrt(...) radicand in it must be square-free.
        for q in _all_questions(tier):
            m = re.search(r"sqrt\((\d+)\)", q.answer_text)
            if m:
                radicand = int(m.group(1))
                k, mm = largest_square_factor(radicand)
                assert k == 1 and mm == radicand, (
                    f"{tier}: canonical answer {q.answer_text!r} for {q.text!r} is not fully simplified "
                    f"(sqrt({radicand}) still has an extractable factor {k})"
                )

    def test_squarefree_bases_are_genuinely_squarefree(self):
        for m in SQUAREFREE:
            k, mm = largest_square_factor(m)
            assert (k, mm) == (1, m), f"{m} is not square-free: largest_square_factor gave ({k}, {mm})"

    def test_independent_checks_fire_on_corrupted_input(self):
        # check_equivalent/check_distinct, exercised directly against a
        # deliberately corrupted pair.
        with pytest.raises(VerificationError):
            check_equivalent(3 * sqrt(2), 4 * sqrt(2), "deliberately wrong")
        with pytest.raises(VerificationError):
            check_distinct(3 * sqrt(2), 3 * sqrt(2), msg="deliberately colliding")


class TestOwnFormRejectedProof:
    """'s mandatory proof: grading the question's OWN un-simplified form back at it is REJECTED -- built as plain
    TEXT from the numbers STATED IN question.text (parsed with a regex), never sympy's own str of a sqrt call
    (which auto-simplifies on construction -- str(sqrt(792)) is already "6*sqrt(22)", not "sqrt(792)" -- and
    so would silently reproduce the canonical instead of the raw unsimplified spelling a learner who stopped
    partway through actually types). This is what proves the `simplified_surd` back door fixed (the raw-text
    check in mathgen/grading/typed.py:_simplified_surd_ok) does not silently reopen here: every one of these
    values is numerically EQUAL to the canonical, so only the form check can be rejecting it."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_own_unsimplified_form_is_rejected(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            checked += 1
            if tier == "easy":
                m = _EASY_RE.match(q.text)
                assert m, f"unrecognised easy prompt shape: {q.text!r}"
                n = m.group(1)
                wrong = f"sqrt({n})"
            elif tier == "medium":
                m = _MEDIUM_RE.match(q.text)
                assert m, f"unrecognised medium prompt shape: {q.text!r}"
                r1, r2 = m.group(1), m.group(2)
                wrong = f"sqrt({r1}) + sqrt({r2})"
            else:
                m = _HARD_RE.match(q.text)
                assert m, f"unrecognised hard prompt shape: {q.text!r}"
                c1, r1, c2, r2 = m.groups()
                wrong = f"{c1}*sqrt({r1}) - {c2}*sqrt({r2})"
            # Sanity: this "wrong" text really is numerically equal to the
            # canonical -- only the form check can be what rejects it.
            assert sympy.simplify(sympy.sympify(wrong) - sympy.sympify(q.answer_text)) == 0, (
                f"{tier}: {wrong!r} is not even numerically equal to canonical {q.answer_text!r} -- "
                "test construction bug, not a grading finding"
            )
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, (
                f"{tier}: question {q.text!r} (answer {q.answer_text!r}) accepted its own un-simplified form {wrong!r}"
            )
        assert checked == len(list(PROOF_SEEDS)), f"{tier}: only {checked} questions proof-tested"
        print(f"{tier}: own-unsimplified-form-rejected proof: {checked}/{checked} passed, 0 failed")


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "simplify-surds")
        assert skill.domain == "secondary-number"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"f10-square-roots-perfect-squares", "prime-factorisation"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == "t1_simplify_surds"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
