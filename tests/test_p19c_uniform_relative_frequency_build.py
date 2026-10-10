"""Tests for: uniform discrete outcome probability (ACMMM138,
``senior-recognise-uniform-discrete-random-variables-and-use-them``) and relative frequency as a point
estimate (ACMEM152, ``senior-identify-relative-frequency-as-probability``)."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import REGISTRY, stated_precision_tolerance
from mathgen.text_quality import prompt_shape

SEEDS = range(60)
SEEDS_200 = range(200)

UD_TOPIC = "p19c_uniform_discrete_outcome"
UD_GRADER = GRADERS["fraction"]
UD_SKILL = "senior-recognise-uniform-discrete-random-variables-and-use-them"

RF_TOPIC = "p19c_relative_frequency_estimate"
RF_GRADER = GRADERS["number"]
RF_SKILL = "senior-identify-relative-frequency-as-probability"


# --------------------------------------------------------------------------
# p19c_uniform_discrete_outcome
# --------------------------------------------------------------------------


class _UDCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            result = UD_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = UD_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            answer_value = Fraction(q.answer["p"], q.answer["q"])
            wrong = answer_value + Fraction(1, 97)
            assert not UD_GRADER(q, str(wrong)).correct

    def test_answer_is_1_over_n_in_lowest_terms(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer["p"] == 1
            assert q.answer["q"] >= 4
            frac = Fraction(q.answer["p"], q.answer["q"])
            assert frac.numerator == q.answer["p"] and frac.denominator == q.answer["q"]

    def test_never_emits_a_distribution_table(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            assert "P(X" not in q.text

    def test_never_asks_equally_likely_classification(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            assert "equally likely?" not in q.text.lower()
            assert "not equally likely" not in q.text.lower()

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(UD_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(UD_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == UD_SKILL)
        assert skill.answer == "fraction"
        assert skill.placement_eligible is False
        assert any(src.topic == UD_TOPIC and src.difficulty == self.TIER for src in skill.sources)


class TestUDEasy(_UDCommon):
    TIER = "easy"


class TestUDMedium(_UDCommon):
    TIER = "medium"

    def test_n_is_derived_from_two_stated_factors(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            # The old "in total" wording assertion is replaced by its purpose -- the product n is
            # never stated, so it must be derived from the two factors.
            n = q.answer["q"]
            assert not re.search(rf"{n}", q.text)


class TestUDHard(_UDCommon):
    TIER = "hard"

    def test_answer_is_1_over_n_in_lowest_terms(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer["q"] >= 4
            assert q.answer["p"] < q.answer["q"]
            frac = Fraction(q.answer["p"], q.answer["q"])
            assert frac.numerator == q.answer["p"] and frac.denominator == q.answer["q"]


class TestUDExam(_UDCommon):
    TIER = "exam"

    def test_compares_two_spinners(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            assert "Two spinners" in q.text
            assert "equal sectors" in q.text
            assert "different sizes" in q.text

    def test_answer_is_1_over_n_in_lowest_terms(self):
        for seed in SEEDS:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            # m/n can reduce below the stated n (e.g. m=4, n=6 -> 2/3), so
            # unlike the other tiers' bare 1/n the denominator is no longer
            # guaranteed >= 4 -- only that it is a genuine proper fraction
            # already in lowest terms.
            assert q.answer["q"] >= 2
            assert q.answer["p"] < q.answer["q"]
            frac = Fraction(q.answer["p"], q.answer["q"])
            assert frac.numerator == q.answer["p"] and frac.denominator == q.answer["q"]

    def test_exam_answer_is_m_over_n_and_does_not_degenerate_to_m_equals_1(self):
        # >= 200 seeds: the new m/n shape must be a genuine
        # "specified event of m outcomes" framing, not construction quietly
        # always drawing m == 1 (which would just be the OLD 1/n kernel
        # wearing new distractors).
        numerators = [make(UD_TOPIC, seed=seed, difficulty=self.TIER).answer["p"] for seed in SEEDS_200]
        assert any(p > 1 for p in numerators), "exam-tier m/n degenerated to m == 1 (bare 1/n) on every seed"
        # And it must not degenerate the OTHER way either: m/n stays a
        # proper fraction less than 1 (m < n) on every draw.
        for seed in SEEDS_200:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer["p"] < q.answer["q"], f"seed {seed}: m/n was not < 1"

    def test_exam_answer_is_not_structurally_the_largest_option_every_draw(self):
        # At least one seed must have some distractor above the answer, and at least one seed some
        # distractor below it.
        any_distractor_above = False
        any_distractor_below = False
        for seed in SEEDS_200:
            q = make(UD_TOPIC, seed=seed, difficulty=self.TIER)
            answer_value = Fraction(q.answer["p"], q.answer["q"])
            values = []
            for d in q.distractors:
                if "/" in d:
                    p_str, q_str = d.split("/")
                    values.append(Fraction(int(p_str), int(q_str)))
                else:
                    values.append(Fraction(int(d)))
            if any(v > answer_value for v in values):
                any_distractor_above = True
            if any(v < answer_value for v in values):
                any_distractor_below = True
            assert values, f"seed {seed}: no distractors at all"
        assert any_distractor_above, "answer was the largest option on every seed"
        assert any_distractor_below, "answer was the smallest option on every seed"


def test_exam_tier_uneven_degrees_always_sum_to_360():
    # >= 200 seeds: the exam-tier spinner-degree
    # composition must sum to exactly 360 on every draw, never an
    # illustrative approximation.
    import re

    for seed in SEEDS_200:
        q = make(UD_TOPIC, seed=seed, difficulty="exam")
        m = re.search(
            r"sectors of different sizes \((.*?)\), the sizes "
            r"adding to 360 degrees in total",
            q.text,
        )
        assert m, f"seed {seed}: could not find the degree listing in: {q.text!r}"
        degrees = [int(x) for x in re.findall(r"(\d+) degrees", m.group(1))]
        assert sum(degrees) == 360, f"seed {seed}: degrees {degrees} sum to {sum(degrees)}, not 360"


def test_uniform_discrete_distractors_never_bare_n_nor_constant_half():
    for tier in ["easy", "medium", "hard", "exam"]:
        for seed in SEEDS_200:
            q = make(UD_TOPIC, seed=seed, difficulty=tier)
            n = q.answer["q"]
            for d in q.distractors:
                if "/" in d:
                    p_str, q_str = d.split("/")
                    value = Fraction(int(p_str), int(q_str))
                else:
                    value = Fraction(int(d))
                assert value != Fraction(n), f"[{tier}, seed {seed}] distractor {d!r} equals bare n"
                assert value != Fraction(1, 2), f"[{tier}, seed {seed}] distractor {d!r} equals 1/2"


def test_uniform_discrete_outcome_topic_registered_once():
    _, fn = REGISTRY[UD_TOPIC]
    assert fn.__module__.endswith("p19c_uniform_relative_frequency_build")


# --------------------------------------------------------------------------
# p19c_relative_frequency_estimate
# --------------------------------------------------------------------------


class _RFCommon:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            result = RF_GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected {q.answer_text!r}"
            for dist in q.distractors:
                result = RF_GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            dp = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
            tol = stated_precision_tolerance(dp)
            wrong = float(q.answer) + float(tol) * 3
            assert not RF_GRADER(q, f"{wrong:.{dp}f}").correct

    def test_never_shows_a_second_category(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            # Exactly one stated count out of one stated total: no second
            # named category, no table, no "or"/"and not" combinator.
            assert " or " not in q.text
            # No frequency-table shape: never more than one distinct named
            # outcome category in the same question.
            assert q.text.count("were ") <= 2

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(RF_TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(RF_TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == RF_SKILL)
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert any(src.topic == RF_TOPIC and src.difficulty == self.TIER for src in skill.sources)


class TestRFEasy(_RFCommon):
    TIER = "easy"

    def test_two_decimal_places_stated(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert "2 decimal places" in q.text
            assert len(q.answer_text.split(".")[1]) == 2


class TestRFMedium(_RFCommon):
    TIER = "medium"

    def test_two_decimal_places_stated(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert "2 decimal places" in q.text


class TestRFHard(_RFCommon):
    TIER = "hard"

    def test_two_decimal_places_stated(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert "2 decimal places" in q.text


class TestRFExam(_RFCommon):
    TIER = "exam"

    def test_percentage_format_escalation(self):
        for seed in SEEDS:
            q = make(RF_TOPIC, seed=seed, difficulty=self.TIER)
            assert "PERCENTAGE" in q.text
            assert len(q.answer_text.split(".")[1]) == 1


def test_relative_frequency_estimate_topic_registered_once():
    _, fn = REGISTRY[RF_TOPIC]
    assert fn.__module__.endswith("p19c_uniform_relative_frequency_build")


def test_relative_frequency_distractors_stay_in_range_and_off_the_key():
    # >= 200 seeds/tier: every relative-
    # frequency distractor must be a genuine (0, 1] probability -- or
    # (0, 100] for the exam-tier percentage -- and land at least the stated
    # margin away from the key in the FORMATTED (dp-place) representation,
    # never a near-miss collision such as the old constant wrong_half
    # colliding with a key near 0.5.
    for tier in ["easy", "medium", "hard", "exam"]:
        for seed in SEEDS_200:
            q = make(RF_TOPIC, seed=seed, difficulty=tier)
            dp = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
            range_max = 100.0 if tier == "exam" else 1.0
            margin = float(10 * stated_precision_tolerance(dp))
            answer_value = float(q.answer)
            for d in q.distractors:
                v = float(d)
                assert 0 < v <= range_max, f"[{tier}, seed {seed}] distractor {d!r} out of range"
                assert abs(v - answer_value) >= margin, (
                    f"[{tier}, seed {seed}] distractor {d!r} within margin of key {answer_value}"
                )


# ---------------------------------------------------------------------------
# -- medium-tier text-pool collapse and an out-of-range exam-tier distractor.
# ---------------------------------------------------------------------------


def test_medium_tier_derived_count_pool_no_longer_collapses():
    """Was 46/200 distinct texts (top text recurring 11x) before widening
    the wheel_split/grid_seats/tournament_teams ranges; must now clear a
    much healthier floor."""
    texts = [make(UD_TOPIC, seed=seed, difficulty="medium").text for seed in range(200)]
    distinct = len(set(texts))
    assert distinct >= 90, f"medium-tier distinct texts over 200 seeds: {distinct} (was 46)"
    from collections import Counter

    top_count = Counter(texts).most_common(1)[0][1]
    assert top_count <= 5, f"medium-tier top-recurring text seen {top_count}x over 200 seeds"


def test_exam_tier_distractors_never_exceed_a_valid_probability():
    """The `Fraction(m, n2)` misconception ('used the other spinner's own
    count') could exceed 1 when m > n2 -- e.g. 15/8 -- an impossible
    probability that is trivially eliminable on sight. Every distractor must
    be a genuine probability in (0, 1)."""
    for seed in range(200):
        q = make(UD_TOPIC, seed=seed, difficulty="exam")
        for d in q.distractors:
            v = Fraction(d)
            assert 0 < v < 1, f"seed {seed}: exam-tier distractor {d!r} out of (0, 1)"


# ---------------------------------------------------------------------------
# variant always graded a bare 1/n (200/200 seeds), the exact same shape as
# the easy tier, just with more prose. Every hard_read sub-variant now grades
# a complement or a conditional fraction instead.
# ---------------------------------------------------------------------------


def test_hard_tier_answer_is_not_always_bare_1_over_n():
    """A healthy majority of draws must now grade a different fraction shape (a complement (n-1)/n or a
    conditional 1/(n-1) with n - 1 >= 4, so its own p == 1 is a different denominator, not the old bare-1/n
    shape)."""
    not_bare = 0
    for seed in range(200):
        q = make(UD_TOPIC, seed=seed, difficulty="hard")
        if q.answer["p"] != 1:
            not_bare += 1
    share = not_bare / 200
    assert share >= 0.60, f"hard-tier non-bare-1/n share over 200 seeds: {share:.2%} (measured for this fix: 66.5%)"


def test_hard_tier_answer_is_not_structurally_the_largest_option_every_draw():
    """At least one seed's distractors include something above the key and at least one seed's include something
    below."""
    any_above = False
    any_below = False
    for seed in range(200):
        q = make(UD_TOPIC, seed=seed, difficulty="hard")
        answer_value = Fraction(q.answer["p"], q.answer["q"])
        values = [Fraction(d) for d in q.distractors]
        if any(v > answer_value for v in values):
            any_above = True
        if any(v < answer_value for v in values):
            any_below = True
    assert any_above, "hard-tier answer was the largest option on every seed"
    assert any_below, "hard-tier answer was the smallest option on every seed"


def test_hard_tier_distractors_never_exceed_a_valid_probability():
    """Analogous to `test_exam_tier_distractors_never_exceed_a_valid_probability`:
    every hard-tier distractor must be a genuine probability in (0, 1),
    distinct from the key, over 200 seeds."""
    for seed in range(200):
        q = make(UD_TOPIC, seed=seed, difficulty="hard")
        answer_value = Fraction(q.answer["p"], q.answer["q"])
        for d in q.distractors:
            v = Fraction(d)
            assert 0 < v < 1, f"seed {seed}: hard-tier distractor {d!r} out of (0, 1)"
            assert v != answer_value, f"seed {seed}: hard-tier distractor {d!r} equals the key"


# ---------------------------------------------------------------------------
# sub-variant built its wrong_candidates in a FIXED order with a FIXED
# above/below split, so `pick_distractors` (which never shuffles) offered the
# same three survivors every draw, putting the key at a CONSTANT rank among
# the sorted options (measured: rank 3 of 4 on 200/200 seeds, or rank 2 of 4
# under a different seeding convention -- either way, a mechanically
# exploitable "always pick this rank" tell, independent of reading the
# question). The fix stratifies each sub-variant's candidates into
# above-key/below-key pools and draws a genuinely random above/below split
# each draw (`_stratified_wrong_fractions`).
# ---------------------------------------------------------------------------


def test_hard_tier_key_rank_is_not_constant():
    """Over 200 seeds, sort `[answer_text] + distractors` as `Fraction` values and record the key's 1-indexed
    rank (ascending; rank 1 = smallest). Measured for this fix (4-option draws only, the overwhelming
    majority): rank 1 24.5%, rank 2 24%, rank 3 28.5%, rank 4 23% -- no rank above 40% of 200 draws."""
    rank_counts: dict[int, int] = {}
    for seed in range(200):
        q = make(UD_TOPIC, seed=seed, difficulty="hard")
        answer_value = Fraction(q.answer["p"], q.answer["q"])
        values = sorted([answer_value] + [Fraction(d) for d in q.distractors])
        rank = values.index(answer_value) + 1
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    top_share = max(rank_counts.values()) / 200
    assert top_share <= 0.40, (
        f"hard-tier key-rank distribution over 200 seeds is too concentrated: "
        f"{rank_counts} (top rank share {top_share:.2%}, must be <= 40%)"
    )
    assert len(rank_counts) >= 2, f"hard-tier key rank never varied: {rank_counts}"


# ---------------------------------------------------------------------------
# Easy/medium put the key at
# the median value rank on 200/200 seeds (exam 76.5%); medium stated N
# directly on 200/200 (contradicting the contract) with only 103/200 unique
# prompts; 51/800 easy, 14/800 medium and 49/800 exam draws showed only three
# distinct options. Each test below is swept and was RED before the fix.
# ---------------------------------------------------------------------------


def _key_rank_counts(tier, seeds=200):
    counts: dict[int, int] = {}
    for seed in range(seeds):
        q = make(UD_TOPIC, seed=seed, difficulty=tier)
        key = Fraction(q.answer["p"], q.answer["q"])
        values = sorted([key] + [Fraction(d) for d in q.distractors])
        rank = values.index(key) + 1
        counts[rank] = counts.get(rank, 0) + 1
    return counts


def test_easy_medium_exam_key_rank_is_spread_over_every_rank():
    for tier in ["easy", "medium", "exam"]:
        counts = _key_rank_counts(tier)
        assert set(counts) == {1, 2, 3, 4}, f"{tier}: not every rank occurs: {counts}"
        assert max(counts.values()) / 200 <= 0.40, f"{tier}: key rank too concentrated: {counts}"


def test_easy_medium_exam_distractors_are_possible_values_off_the_key():
    for tier in ["easy", "medium", "exam"]:
        for seed in range(200):
            q = make(UD_TOPIC, seed=seed, difficulty=tier)
            key = Fraction(q.answer["p"], q.answer["q"])
            for d in q.distractors:
                assert 0 < Fraction(d) <= 1 and Fraction(d) != key, f"[{tier}, seed {seed}] {d!r}"


def test_medium_prompt_never_states_the_product_n():
    import re

    for seed in range(200):
        q = make(UD_TOPIC, seed=seed, difficulty="medium")
        n = q.answer["q"]
        assert not re.search(rf"(?<![\d/]){n}(?![\d/])", q.text), f"seed {seed}: medium prompt states N={n}: {q.text!r}"


def test_medium_prompt_gives_exactly_the_two_factors():
    import re

    for seed in range(200):
        q = make(UD_TOPIC, seed=seed, difficulty="medium")
        nums = [int(x) for x in re.findall(r"\d+", q.text)]
        assert len(nums) == 2 and nums[0] * nums[1] == q.answer["q"], f"seed {seed}: {q.text!r}"


def test_medium_unique_prompts_at_least_150_of_200():
    texts = {make(UD_TOPIC, seed=seed, difficulty="medium").text for seed in range(200)}
    assert len(texts) >= 150, f"medium unique prompts over 200 seeds: {len(texts)} (was 103)"


def test_every_tier_always_offers_four_distinct_options():
    for tier in ["easy", "medium", "hard", "exam"]:
        for seed in range(800):
            q = make(UD_TOPIC, seed=seed, difficulty=tier)
            options = {Fraction(q.answer_text)} | {Fraction(d) for d in q.distractors}
            assert len(q.distractors) == 3 and len(options) == 4, (
                f"[{tier}, seed {seed}] options {[q.answer_text, *q.distractors]}"
            )


def test_exam_prompt_sizes_are_truly_distinct():
    import re

    for seed in range(200):
        q = make(UD_TOPIC, seed=seed, difficulty="exam")
        m = re.search(r"sectors of different sizes \((.*?)\), the sizes", q.text)
        sizes = [int(x) for x in re.findall(r"(\d+) degrees", m.group(1))]
        assert len(set(sizes)) == len(sizes) and sum(sizes) == 360, f"seed {seed}: {sizes}"


# Wording cleanup
# --------------------------------------------------------------------------


def test_2063_uniform_medium_never_says_in_total():
    for seed in SEEDS_200:
        q = make(UD_TOPIC, seed=seed, difficulty="medium")
        assert "in total" not in q.text, q.text


def test_2063_uniform_exam_introduces_one_spinner_before_the_other():
    for seed in SEEDS_200:
        q = make(UD_TOPIC, seed=seed, difficulty="exam")
        assert "one spinner" in q.text and "the other spinner" in q.text, q.text
        assert q.text.index("one spinner") < q.text.index("the other spinner"), q.text
