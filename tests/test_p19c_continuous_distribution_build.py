"""Tests for: probability-density integral (ACMMM165,
``senior-understand-the-concepts-of-a-probability-density-function``) and continuous-uniform E/Var/SD
(ACMMM166, ``senior-recognise-the-expected-value-variance-and-standard-deviation``)."""

from __future__ import annotations

from fractions import Fraction

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import REGISTRY, stated_precision_tolerance
from mathgen.text_quality import prompt_shape

SEEDS = range(60)

PD_TOPIC = "p19c_density_probability"
PD_EXAM_SOURCE_TOPIC = "exd_x_r2k_density"  # the skill's exam source is the multi-part chain, not PD_TOPIC
PD_GRADER = GRADERS["number"]
PD_SKILL = "senior-understand-the-concepts-of-a-probability-density-function"

CUM_TOPIC = "p19c_continuous_uniform_moments"
CUM_GRADER = GRADERS["ordered_list"]
CUM_SKILL = "senior-recognise-the-expected-value-variance-and-standard-deviation"


# --------------------------------------------------------------------------
# p19c_density_probability
# --------------------------------------------------------------------------


class _PDCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            result = PD_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = PD_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            dp = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
            tol = stated_precision_tolerance(dp)
            wrong = float(q.answer) + float(tol) * 3
            assert not PD_GRADER(q, f"{wrong:.{dp}f}").correct

    def test_probability_in_zero_one(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            value = Fraction(q.answer_text)
            assert 0 < value < 1

    def test_always_named_as_density_and_probability(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            assert "probability density function" in q.text
            assert "probability that X" in q.text.lower() or "probability that x" in q.text.lower()
            # Never collapses into a bare-integral phrasing with no
            # probability language (the definite-integral boundary).
            assert "evaluate the definite integral" not in q.text.lower() or "probability" in q.text.lower()

    def test_never_mentions_cdf_or_type_classification(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            lowered = q.text.lower()
            assert "cumulative distribution" not in lowered
            assert "what type" not in lowered
            assert "classify" not in lowered

    def test_stated_three_decimal_places(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            assert "3 decimal places" in q.text
            assert len(q.answer_text.split(".")[1]) == 3

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(PD_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(PD_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == PD_SKILL)
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == [
            "senior-examine-simple-examples-of-non-uniform-discrete-random",
            "integrate-polynomial",
        ]
        expected = PD_EXAM_SOURCE_TOPIC if self.TIER == "exam" else PD_TOPIC
        assert any(src.topic == expected and src.difficulty == self.TIER for src in skill.sources)


class TestPDEasy(_PDCommon):
    TIER = "easy"

    def test_uniform_density_constant(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            # f(x) = <constant>, no "x" in the density expression itself.
            frag = q.text.split("f(x) = ")[1].split(" for ")[0].split(",")[0]
            assert "x" not in frag


class TestPDMedium(_PDCommon):
    TIER = "medium"

    def test_linear_density(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            assert "^2" not in q.text.split("f(x) = ")[1].split(" for ")[0].split(", defined")[0]


class TestPDHard(_PDCommon):
    TIER = "hard"

    def test_quadratic_density(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            assert "^2" in q.text


class TestPDExam(_PDCommon):
    TIER = "exam"

    def test_boundary_bound_touches_support_endpoint(self):
        for seed in SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty=self.TIER)
            assert (
                "support of X is exactly" in q.text
                or "equal to 0 outside it" in q.text
                or "can never take a value outside" in q.text
            )


def test_density_probability_topic_registered_once():
    _, fn = REGISTRY[PD_TOPIC]
    assert fn.__module__.endswith("p19c_continuous_distribution_build")


# --------------------------------------------------------------------------
# p19c_continuous_uniform_moments
# --------------------------------------------------------------------------


class _CUMCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            result = CUM_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = CUM_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            # Deliberately-wrong triple: perturb E(X) by well over the 2 d.p. tolerance.
            e0, var0, sd0 = q.answer
            wrong_text = f"({float(e0) + 1.0}, {var0}, {sd0})"
            assert not CUM_GRADER(q, wrong_text).correct

    def test_answer_is_ordered_triple_of_three_numbers(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            assert isinstance(q.answer, list)
            assert len(q.answer) == 3

    def test_closed_form_uniform_formulas_hold(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            e, var, sd = (Fraction(str(v)) for v in q.answer)
            # Var and SD are consistent with each other to within rounding
            # at the stated 2 d.p. precision.
            assert abs(float(sd) * float(sd) - float(var)) < 0.05

    def test_never_shows_a_discrete_probability_table(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            assert "P(X =" not in q.text

    def test_stated_two_decimal_places(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            assert "2 decimal places" in q.text

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(CUM_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(CUM_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == CUM_SKILL)
        assert skill.answer == "ordered_list"
        assert skill.placement_eligible is False
        assert skill.prerequisites == [
            "senior-understand-the-concepts-of-a-probability-density-function",
            "integrate-polynomial",
        ]
        assert any(src.topic == CUM_TOPIC and src.difficulty == self.TIER for src in skill.sources)


class TestCUMEasy(_CUMCommon):
    TIER = "easy"

    def test_nonnegative_interval(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            e = float(q.answer[0])
            assert e >= 0


class TestCUMMedium(_CUMCommon):
    TIER = "medium"


class TestCUMHard(_CUMCommon):
    TIER = "hard"

    def test_negative_lower_bound(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            assert "-" in q.text.split("Determine")[0].split("Compute")[0].split("distributed")[-1]


class TestCUMExam(_CUMCommon):
    TIER = "exam"

    def test_interval_stated_indirectly_via_scenario(self):
        for seed in SEEDS:
            q = make(CUM_TOPIC, seed=seed, difficulty=self.TIER)
            assert "[" not in q.text
            assert "at least" in q.text or "guaranteed" in q.text or "minimum" in q.text or "never less than" in q.text


def test_continuous_uniform_moments_topic_registered_once():
    _, fn = REGISTRY[CUM_TOPIC]
    assert fn.__module__.endswith("p19c_continuous_distribution_build")


SWEEP_SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")


class TestPDFInRangeDistractors:
    """PDF minor: every rendered distractor value parses to a number in the
    open interval (0, 1), every tier, 200 seeds -- guards against a
    previously observed negative-distractor rate (200/200) and medium's >1 rate (36/200),
    asserting both are 0/200."""

    def test_no_negative_distractor_any_tier(self):
        for tier in TIERS:
            neg_count = 0
            for seed in SWEEP_SEEDS:
                q = make(PD_TOPIC, seed=seed, difficulty=tier)
                if any(Fraction(d) < 0 for d in q.distractors):
                    neg_count += 1
            assert neg_count == 0, f"{tier}: {neg_count}/200 draws had a negative distractor"

    def test_no_gt1_distractor_medium_tier(self):
        gt1_count = 0
        for seed in SWEEP_SEEDS:
            q = make(PD_TOPIC, seed=seed, difficulty="medium")
            if any(Fraction(d) > 1 for d in q.distractors):
                gt1_count += 1
        assert gt1_count == 0, f"medium: {gt1_count}/200 draws had a >1 distractor"

    def test_every_distractor_in_open_interval_zero_one_all_tiers(self):
        for tier in TIERS:
            for seed in SWEEP_SEEDS:
                q = make(PD_TOPIC, seed=seed, difficulty=tier)
                for d in q.distractors:
                    v = Fraction(d)
                    assert 0 < v < 1, f"{tier} seed {seed}: distractor {d!r} out of (0, 1)"

    def test_at_least_three_distractors_every_draw_all_tiers(self):
        for tier in TIERS:
            for seed in SWEEP_SEEDS:
                q = make(PD_TOPIC, seed=seed, difficulty=tier)
                assert len(q.distractors) >= 3, f"{tier} seed {seed}: only {len(q.distractors)} distractors"


class TestCUMColumnWiseDistinct:
    """Continuous-uniform moments: for every seed/tier, the four rendered
    E values are pairwise distinct, the four Var values are pairwise
    distinct, the four SD values are pairwise distinct (no column-wise
    repeat, hence no majority-vote tell) -- guards against a
    previously observed modal-column rate (193/186/200/200 across tiers), asserting it
    is 0/200."""

    def _columns(self, q):
        key_e, key_var, key_sd = q.answer_text.strip("()").split(", ")
        columns = {"E": [key_e], "Var": [key_var], "SD": [key_sd]}
        for dist in q.distractors:
            de, dvar, dsd = dist.strip("()").split(", ")
            columns["E"].append(de)
            columns["Var"].append(dvar)
            columns["SD"].append(dsd)
        return columns

    def test_no_column_wise_majority_any_tier(self):
        for tier in TIERS:
            modal_count = {"E": 0, "Var": 0, "SD": 0}
            for seed in SWEEP_SEEDS:
                q = make(CUM_TOPIC, seed=seed, difficulty=tier)
                columns = self._columns(q)
                for label, values in columns.items():
                    if values.count(values[0]) >= 2:
                        modal_count[label] += 1
            for label, count in modal_count.items():
                assert count == 0, f"{tier}: {label} column has a majority-vote tell in {count}/200 draws"

    def test_all_four_column_values_pairwise_distinct_every_draw(self):
        for tier in TIERS:
            for seed in SWEEP_SEEDS:
                q = make(CUM_TOPIC, seed=seed, difficulty=tier)
                columns = self._columns(q)
                for label, values in columns.items():
                    assert len(set(values)) == 4, f"{tier} seed {seed}: {label} column not pairwise distinct: {values}"


# ---------------------------------------------------------------------------
# concentration on 0.125 (an exact-half interior offset always cubes to
# 1/8), which exceeded the composition's 15%-of-200-draws ceiling.
# ---------------------------------------------------------------------------


def test_pd_exam_top_answer_share_under_15_percent():
    from collections import Counter

    counts = Counter(make(PD_TOPIC, seed=seed, difficulty="exam").answer_text for seed in range(200))
    top_share = counts.most_common(1)[0][1] / 200
    assert top_share <= 0.15, f"exam-tier top answer share {top_share:.2%} exceeds the 15% ceiling ({counts})"
