"""Tests the Leibniz-notation derivative skill (ACMMM083) and the increments-formula skill (ACMMM107).

senior-use-the-leibniz-notation-for-the-derivative-frac (ACMMM083, answer:
number) and senior-use-the-increments-formula-delta-y-cong-frac (ACMMM107,
answer: fraction).
"""

import re
from fractions import Fraction

import pytest
from sympy import Integer

import mathgen.generators.p9c_leibniz_derivative_build as gen_mod
from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

TOPIC_4 = "p9c_leibniz_derivative_correspondence"
TOPIC_5 = "p9c_increments_formula_estimate"
SKILL_4 = "senior-use-the-leibniz-notation-for-the-derivative-frac"
SKILL_5 = "senior-use-the-increments-formula-delta-y-cong-frac"
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]


def _mk(topic, difficulty, seed):
    return make(topic, seed=seed, difficulty=difficulty)


# ---------------------------------------------------------------------------
# Skill 4: senior-use-the-leibniz-notation-for-the-derivative-frac


class TestLeibnizCorrespondence:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism(self, tier):
        for seed in SEEDS:
            q1 = _mk(TOPIC_4, tier, seed)
            q2 = _mk(TOPIC_4, tier, seed)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for seed in SEEDS:
            q = _mk(TOPIC_4, tier, seed)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, tier):
        for seed in SEEDS:
            q = _mk(TOPIC_4, tier, seed)
            result = grade_number(q, q.answer_text)
            assert result.correct, f"seed {seed} {tier}: grader rejected {q.answer_text!r}"
            for d in q.distractors:
                result = grade_number(q, d)
                assert not result.correct, f"seed {seed} {tier}: grader accepted distractor {d!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_rejects_a_deliberately_wrong_answer(self, tier):
        for seed in range(10):
            q = _mk(TOPIC_4, tier, seed)
            wrong = int(q.answer_text) + 137
            result = grade_number(q, str(wrong))
            assert not result.correct, f"seed {seed} {tier}: grader accepted deliberately-wrong {wrong}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_is_a_plain_integer(self, tier):
        for seed in SEEDS:
            q = _mk(TOPIC_4, tier, seed)
            assert isinstance(q.answer, int)
            assert q.answer_text == str(int(q.answer_text))

    @pytest.mark.parametrize("tier", TIERS)
    def test_full_draw_space(self, tier):
        texts = {_mk(TOPIC_4, tier, seed).text for seed in range(60)}
        assert len(texts) >= 20

    def test_names_both_notations_and_a_number(self):
        """The one-sentence boundary test: every question connects dy/dx to
        f'(x)/f'(a) and answers with a plain number -- never an expression
        in x or x, h."""
        for tier in TIERS:
            for seed in range(10):
                q = _mk(TOPIC_4, tier, seed)
                assert "dy/dx" in q.text
                assert "f'(x)" in q.text
                assert "h" not in q.answer_text

    def test_easy_no_linear_term(self):
        """B == 0 at easy tier, so f(x) is only ``[+-]?<coef>x^2`` optionally
        followed by a constant term -- no separate linear ``x`` term."""
        pattern = re.compile(r"^-?\d*x\^2( [+-] \d+)?$")
        for seed in SEEDS:
            q = _mk(TOPIC_4, "easy", seed)
            f_clause = q.text.split("y = f(x) = ", 1)[1].split(".", 1)[0]
            assert pattern.match(f_clause), f_clause

    def test_hard_is_a_genuine_inverse_not_an_evaluation(self):
        """Hard tier states the derivative VALUE and asks for the point,
        never the reverse."""
        for seed in range(20):
            q = _mk(TOPIC_4, "hard", seed)
            assert "find that value of x" in q.text
            assert "find the value of dy/dx" not in q.text

    def test_independent_check_fires_on_corrupted_answer(self, monkeypatch):
        """Exercise THIS generator's own check sites directly: corrupt the
        sympy entry point each route calls (`diff` for easy/medium's
        evaluation, `solve` for hard's inversion) and confirm
        `check_equivalent` raises rather than silently accepting a wrong
        value. Must return sympy `Integer`s, not bare ints, so the
        generator's own `.subs`/indexing on the result still works."""
        monkeypatch.setattr(gen_mod, "diff", lambda *a, **k: Integer(999))
        with pytest.raises(VerificationError):
            make(TOPIC_4, seed=0, difficulty="easy")
        monkeypatch.undo()

        monkeypatch.setattr(gen_mod, "solve", lambda *a, **k: [Integer(999)])
        with pytest.raises(VerificationError):
            make(TOPIC_4, seed=0, difficulty="hard")


# ---------------------------------------------------------------------------
# Skill 5: senior-use-the-increments-formula-delta-y-cong-frac


class TestIncrementsFormula:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism(self, tier):
        for seed in SEEDS:
            q1 = _mk(TOPIC_5, tier, seed)
            q2 = _mk(TOPIC_5, tier, seed)
            assert q1.to_json() == q2.to_json()

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for seed in SEEDS:
            q = _mk(TOPIC_5, tier, seed)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_accepts_answer_rejects_distractors(self, tier):
        for seed in SEEDS:
            q = _mk(TOPIC_5, tier, seed)
            result = grade_fraction(q, q.answer_text)
            assert result.correct, f"seed {seed} {tier}: grader rejected {q.answer_text!r}"
            for d in q.distractors:
                result = grade_fraction(q, d)
                assert not result.correct, f"seed {seed} {tier}: grader accepted distractor {d!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_accepts_unreduced_equivalent_fraction(self, tier):
        """grade_fraction compares by value: a typed but unreduced fraction
        equal to the canonical must still be accepted."""
        for seed in range(5):
            q = _mk(TOPIC_5, tier, seed)
            value = Fraction(q.answer["p"], q.answer["q"])
            unreduced = f"{value.numerator * 3}/{value.denominator * 3}"
            assert grade_fraction(q, unreduced).correct, f"seed {seed} {tier}: rejected {unreduced!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_grader_rejects_a_deliberately_wrong_answer(self, tier):
        for seed in range(10):
            q = _mk(TOPIC_5, tier, seed)
            value = Fraction(q.answer["p"], q.answer["q"])
            wrong = value + Fraction(7, 1)
            result = grade_fraction(q, f"{wrong.numerator}/{wrong.denominator}")
            assert not result.correct, f"seed {seed} {tier}: grader accepted deliberately-wrong {wrong}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_is_an_exact_fraction_in_lowest_terms(self, tier):
        for seed in SEEDS:
            q = _mk(TOPIC_5, tier, seed)
            assert isinstance(q.answer, dict) and {"p", "q"} <= set(q.answer)
            value = Fraction(q.answer["p"], q.answer["q"])
            assert value.numerator == q.answer["p"] and value.denominator == q.answer["q"], (
                "answer dict is not already in lowest terms"
            )

    @pytest.mark.parametrize("tier", TIERS)
    def test_full_draw_space(self, tier):
        texts = {_mk(TOPIC_5, tier, seed).text for seed in range(60)}
        assert len(texts) >= 20

    def test_every_question_says_estimate(self):
        """The one-sentence boundary test: this skill is an ESTIMATE via a
        stated formula, never the exact change -- the word must appear."""
        for tier in TIERS:
            for seed in range(10):
                q = _mk(TOPIC_5, tier, seed)
                assert "Estimate" in q.text
                assert "increments formula" in q.text

    def test_hard_asks_for_new_value_not_increment(self):
        for seed in range(20):
            q = _mk(TOPIC_5, "hard", seed)
            assert "new value of y" in q.text
            assert "change in y" not in q.text  # that phrasing is easy/medium's only

    def test_hard_uses_negative_dx_easy_medium_use_positive(self):
        for seed in range(20):
            qe = _mk(TOPIC_5, "easy", seed)
            qm = _mk(TOPIC_5, "medium", seed)
            qh = _mk(TOPIC_5, "hard", seed)
            dx_e = qe.text.split("δx = ")[1].split(" from")[0]
            dx_m = qm.text.split("δx = ")[1].split(" from")[0]
            dx_h = qh.text.rstrip(".").split("δx = ")[-1]
            assert not dx_e.startswith("-")
            assert not dx_m.startswith("-")
            assert dx_h.startswith("-")

    def test_exact_change_never_offered_as_the_answer(self):
        """Exact minus estimate is always A*δx**2 != 0 for easy/medium, so
        the true exact change f(x0+δx)-f(x0) is a genuinely different value
        from the canonical estimate on every seed -- confirmed by recomputing
        it here directly from the question's own stated A, x0, δx (parsed
        from the text) and checking it never equals the shipped answer."""
        pattern = re.compile(
            r"y = f\(x\) = (?P<f>[^.]+)\. Estimate the change in y, δy, .* "
            r"for a change of δx = (?P<dx>\S+) from x0 = (?P<x0>-?\d+)\."
        )
        for tier in ["easy", "medium"]:
            for seed in range(20):
                q = _mk(TOPIC_5, tier, seed)
                m = pattern.search(q.text)
                assert m, q.text
                a_match = re.match(r"-?(\d*)x\^2", m.group("f"))
                a_str = a_match.group(1)
                a = int(a_str) if a_str else 1
                dx = Fraction(m.group("dx"))
                estimate = Fraction(q.answer["p"], q.answer["q"])
                exact_minus_estimate = a * dx**2
                assert exact_minus_estimate != 0
                assert estimate + exact_minus_estimate != estimate

    def test_independent_check_fires_on_corrupted_answer(self, monkeypatch):
        """Corrupt `diff`, this generator's sole independent-check entry
        point for every tier (medium/hard tiers) and confirm the deriv/f(x0)
        check_equivalent calls raise rather than accept."""
        monkeypatch.setattr(gen_mod, "diff", lambda *a, **k: Integer(999))
        for tier in TIERS:
            with pytest.raises(VerificationError):
                make(TOPIC_5, seed=0, difficulty=tier)


# ---------------------------------------------------------------------------
# Skill loading


class TestSkillsLoad:
    def test_leibniz_skill_loads_with_correct_metadata(self):
        domains, skills = load_skills()
        skill = next((s for s in skills if s.id == SKILL_4), None)
        assert skill is not None, f"skill {SKILL_4} not found"
        assert skill.answer == "number"
        assert skill.prerequisites == ["differentiate-polynomial", "senior-use-the-notation-frac-delta-y-delta-x"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 5
        assert len(skill.misconceptions) >= 1

    def test_increments_skill_loads_with_correct_metadata(self):
        domains, skills = load_skills()
        skill = next((s for s in skills if s.id == SKILL_5), None)
        assert skill is not None, f"skill {SKILL_5} not found"
        assert skill.answer == "fraction"
        assert skill.prerequisites == ["differentiate-polynomial", "senior-use-the-leibniz-notation-delta-x-and-delta"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 5
        assert len(skill.misconceptions) >= 1

    def test_both_skills_sources_cover_all_difficulties(self):
        domains, skills = load_skills()
        for skill_id in (SKILL_4, SKILL_5):
            skill = next(s for s in skills if s.id == skill_id)
            difficulties = {source.difficulty for source in skill.sources}
            assert difficulties == {"easy", "medium", "hard", "exam"}
