"""Tests for f10-real-number-applied-problems."""

import re

import pytest
import sympy

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent, check_pairwise_distinct

TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)

_EASY_RE = re.compile(
    r"^(?:A running track is shaped like a rectangle (\d+) m by (\d+) m, but one of the \2 m ends "
    r"is replaced by a semicircular curve of diameter \2 m\..*|"
    r"A garden bed is a rectangle (\d+) m by (\d+) m with one of its \4 m ends replaced by a "
    r"semicircular curve of diameter \4 m\..*|"
    r"A swimming lane is bounded by two straight sides of length (\d+) m, one straight end of "
    r"length (\d+) m, and a semicircular end of diameter \6 m\..*)$"
)
_MEDIUM_RE = re.compile(
    r"^(?:A path runs along the diagonal of a rectangular park (\d+) m by (\d+) m,.*|"
    r"Fencing is needed along the diagonal of a rectangular block (\d+) m by (\d+) m,.*|"
    r"A cable runs along the diagonal of a rectangular field (\d+) m by (\d+) m,.*)$"
)
_HARD_RE = re.compile(r"rectangular plot (\d+) m by (\d+) m")
_HARD_L_RE = re.compile(r"each ([\d.]+) m long")
_EXAM_RE = re.compile(
    r"radius (\d+) m\..*?(?:total perimeter|perimeter) is (?:stated to be|exactly) (\S+) \+ \1·π metres"
)


def _all_questions(tier, seeds=SEEDS):
    return [make("p18a_real_number_applied", seed=s, difficulty=tier) for s in seeds]


class TestDeterminismAndBasics:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("p18a_real_number_applied", seed=seed, difficulty=tier)
            q2 = make("p18a_real_number_applied", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for q in _all_questions(tier):
            assert len(q.distractors) == 3, f"{tier}: expected 3 distractors, got {q.distractors} for {q.text!r}"
            for d in q.distractors:
                assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_three_prompt_shapes(self, tier):
        regex = {"easy": _EASY_RE, "medium": _MEDIUM_RE}.get(tier)
        if regex is None:
            # hard/exam use 2-3 wordings, each after a fixed "Exam-style
            # question."/no prefix; measure distinct SECOND-clause openers
            # (the first clause is either the shared exam prefix or empty).
            def _opener(text):
                parts = text.split(". ")
                return parts[1].split(",")[0] if len(parts) > 1 else parts[0].split(",")[0]

            openers = {_opener(q.text) for q in _all_questions(tier)}
            assert len(openers) >= 3, f"{tier}: only {len(openers)} distinct openers: {openers}"
            return
        matched = [regex.match(q.text) for q in _all_questions(tier)]
        assert all(matched), f"{tier}: unrecognised prompt shape among {sum(1 for m in matched if not m)} questions"

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
            wrong = str(sympy.sympify(q.answer_text) + 1)
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_options_formatted_alike(self, tier):
        for q in _all_questions(tier):
            options = [q.answer_text] + list(q.distractors)
            for o in options:
                assert "**" not in o and "^" not in o and "√" not in o, f"{tier}: malformed option {o!r}: {q.text!r}"

    def test_answer_concentration_under_15_percent(self):
        from collections import Counter

        for tier in TIERS:
            counts = Counter(q.answer_text for q in _all_questions(tier, seeds=range(200)))
            top = counts.most_common(1)[0][1]
            assert top / 200 < 0.15, f"{tier}: top answer share {top / 200:.3f} >= 0.15"


class TestTierSpecificShape:
    def test_easy_answer_is_a_plus_b_pi(self):
        for q in _all_questions("easy"):
            expr = sympy.sympify(q.answer_text)
            assert expr.has(sympy.pi), f"easy answer {q.answer_text!r} has no pi term: {q.text!r}"

    def test_medium_answer_is_a_plus_sqrt_b(self):
        for q in _all_questions("medium"):
            expr = sympy.sympify(q.answer_text)
            assert expr.has(sympy.sqrt(2)) or any(
                a.is_Pow and a.exp == sympy.Rational(1, 2) for a in expr.atoms(sympy.Pow)
            ), f"medium answer {q.answer_text!r} has no surd term: {q.text!r}"

    def test_hard_answer_is_a_decimal_to_two_places(self):
        for q in _all_questions("hard"):
            assert re.fullmatch(r"-?\d+\.\d{2}", q.answer_text), f"hard answer not 2dp: {q.answer_text!r}"

    def test_hard_never_rounds_pi_early_in_construction(self):
        # Confirm the module's own docstring claim: recompute one hard-tier
        # seed's exact pre-rounding value with sympy's OWN pi and its own
        # `sqrt(a**2+b**2)/2` derivation of the radius from the stated
        # rectangular plot (never re-using the generator's own float route),
        # and confirm it agrees to the stated precision.
        from fractions import Fraction

        from mathgen.core import round_half_up

        for q in _all_questions("hard", seeds=range(10)):
            m = _HARD_RE.search(q.text)
            lm = _HARD_L_RE.search(q.text)
            assert m and lm, f"could not parse hard question: {q.text!r}"
            a, b = int(m.group(1)), int(m.group(2))
            L = Fraction(lm.group(1))
            r = sympy.sqrt(a * a + b * b) / 2
            sym_total = 2 * sympy.Rational(L.numerator, L.denominator) + r * sympy.pi
            recomputed = round_half_up(Fraction(str(sym_total.evalf(30))), 2)
            assert str(recomputed) == str(Fraction(q.answer_text)) or abs(
                recomputed - Fraction(q.answer_text)
            ) <= Fraction(1, 200), f"hard: sympy-pi re-derivation {recomputed} disagrees with {q.answer_text!r}"

    def test_exam_answer_is_rational_and_solves_forward_formula(self):
        for q in _all_questions("exam"):
            m = _EXAM_RE.search(q.text)
            assert m, f"could not parse exam question: {q.text!r}"
            C = int(m.group(1))
            R = sympy.Rational(m.group(2))
            L = sympy.sympify(q.answer_text)
            forward_total = 2 * L + C * sympy.pi
            rational_part = sympy.simplify(forward_total - C * sympy.pi)
            assert rational_part == R, f"exam: L={L} does not satisfy R={R}, C={C} in {q.text!r}"


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-real-number-applied-problems")
        assert skill.domain == "secondary-number"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert list(skill.prerequisites) == ["classify-real-numbers"]
        stages_covered = set()
        difficulties = set()
        for source in skill.sources:
            assert source.topic == "p18a_real_number_applied"
            stages_covered.update(source.stages)
            difficulties.add(source.difficulty)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        assert difficulties == {"easy", "medium", "hard", "exam"}


class TestIndependentChecksFireOnCorruption:
    def test_check_equivalent_fires(self):
        with pytest.raises(VerificationError):
            check_equivalent(2 + sympy.pi, 3 + sympy.pi, "deliberately wrong")

    def test_check_pairwise_distinct_fires(self):
        with pytest.raises(VerificationError):
            check_pairwise_distinct(2 + sympy.pi, 2 + sympy.pi, msg="deliberately colliding")
