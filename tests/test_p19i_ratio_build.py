"""See mathgen/generators/p19i_ratio_build.py for the tier design and misconceptions."""

import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make

ROOT = Path(__file__).resolve().parents[1]
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

TOPIC_A = "p19i_ratio_fraction_share"
TOPIC_B = "p19i_ratio_simplest_form"


def _value(q):
    return Fraction(q.answer["p"], q.answer["q"])


# --- determinism / distractor hygiene, both topics --------------------------


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        q1 = make(topic, seed=seed, difficulty=difficulty)
        q2 = make(topic, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_and_not_the_answer_by_value(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3, (topic, difficulty, seed, q.distractors)
        assert all(d for d in q.distractors)
        answer = _value(q)
        dvals = [Fraction(d.replace(":", "/")) for d in q.distractors]
        assert all(v != answer for v in dvals), (topic, difficulty, seed, q.distractors)
        assert len(set(dvals)) == len(dvals), (topic, difficulty, seed, q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_answer_text_is_a_bare_fraction(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        assert ":" not in q.answer_text, q.answer_text
        assert "/" in q.answer_text
        for d in q.distractors:
            assert ":" not in d, d


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_answer_text_is_colon_notation(difficulty):
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        assert "/" not in q.answer_text, q.answer_text
        assert ":" in q.answer_text
        for d in q.distractors:
            assert "/" not in d, d


def test_skill_b_answer_is_always_in_lowest_terms():
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(TOPIC_B, seed=seed, difficulty=difficulty)
            p, q_ = q.answer["p"], q.answer["q"]
            assert math.gcd(p, q_) == 1, (difficulty, seed, q.answer_text)


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_answer_rejects_distractors_and_a_wrong_value(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert grade_fraction(q, q.answer_text).correct, (topic, difficulty, seed)
        for d in q.distractors:
            assert not grade_fraction(q, d).correct, (topic, difficulty, seed, d)
        wrong = Fraction(q.answer["p"] + 1000, q.answer["q"] + 1)
        assert not grade_fraction(q, f"{wrong.numerator}/{wrong.denominator}").correct


# --- variety ------------------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{topic}/{difficulty}: only {len(texts)} distinct questions"


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_3_distinct_prompt_shapes_over_60_seeds(topic, difficulty):
    from mathgen.text_quality import prompt_shape

    shapes = {prompt_shape(make(topic, seed=seed, difficulty=difficulty)) for seed in range(60)}
    assert len(shapes) >= 3, f"{topic}/{difficulty}: only {len(shapes)} distinct prompt shapes"


# --- 200-seed answer-concentration instrument --------------------------


@pytest.mark.parametrize("topic", [TOPIC_A, TOPIC_B])
@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_concentration_stays_under_15_percent(topic, difficulty):
    counts = Counter(_value(make(topic, seed=seed, difficulty=difficulty)) for seed in range(200))
    top_key, top_count = counts.most_common(1)[0]
    assert top_count / 200 <= 0.15, (
        f"{topic}/{difficulty}: {top_key} appears in {top_count}/200 draws, over the 15% cap"
    )


# --- tier-specific invariants: Skill A -----------------------------------


def test_skill_a_exam_states_at_least_three_numbers():
    """Every exam-tier question states at least three numbers -- the two
    relevant counts plus one irrelevant number (a third item's count, or
    the total shown as a redundant red herring)."""
    import re

    for seed in range(60):
        q = make(TOPIC_A, seed=seed, difficulty="exam")
        nums = [int(n) for n in re.findall(r"\d+", q.text)]
        assert len(nums) >= 3, q.text


def _exam_total(text):
    """The unstated whole, re-derived from the printed numbers.
    Two shapes: three stated group counts, or one count + a two-group ratio
    + a third count. Returns None when the prompt states a total itself."""
    import re

    if re.search(r"total|altogether|in all|combined", text):
        return None
    ratio = re.search(
        r"(\d+) [a-z ]+?\. The ratio of [a-z ]+? to [a-z ]+? is (\d+):(\d+), "
        r"and there are (\d+) ",
        text,
    )
    if ratio:
        s, a, b, t = (int(g) for g in ratio.groups())
        assert s % b == 0
        return s // b * (a + b) + t
    counts = re.search(r"(\d+) [a-z ]+?, (\d+) [a-z ]+? and (\d+) [a-z ]+?\.", text)
    assert counts, text
    return sum(int(g) for g in counts.groups())


def test_skill_a_exam_never_states_the_total_and_needs_a_combine_step():
    """Exam is a step beyond hard. The total is never printed, the key denominator (and the unreduced total) is
    not a printed number, and the total is only reachable by combining sub-group counts."""
    import re

    questions = [make(TOPIC_A, seed=seed, difficulty="exam") for seed in range(200)]
    stated = sum(1 for q in questions if _exam_total(q.text) is None)
    assert stated == 0, f"{stated}/200 exam prompts state a total"
    for seed, q in enumerate(questions):
        total = _exam_total(q.text)
        printed = {int(n) for n in re.findall(r"\d+", q.text)}
        assert total not in printed, (seed, q.text)
        assert q.answer["q"] not in printed, (seed, q.text)
        # the answer recomputes from the printed numbers
        assert total % q.answer["q"] == 0, (seed, q.text)
        assert Fraction(q.answer["p"] * (total // q.answer["q"]), total) == _value(q)


def _hard_has_derived_step(q):
    """marker: hard states a total and ONE group's count, and asks for 'the rest' -- exactly two numbers, so the
    asked group's count is never stated and must be derived by subtraction."""
    import re

    nums = [int(n) for n in re.findall(r"\d+", q.text)]
    if len(nums) != 2 or "the rest" not in q.text:
        return False
    total, given = nums
    return total > given and Fraction(total - given, total) == _value(q)


def test_skill_a_hard_requires_a_derived_missing_count():
    """Hard is not a wording change over easy/medium: the asked group's
    count is absent from the text and equals total - given."""
    for seed in range(200):
        q = make(TOPIC_A, seed=seed, difficulty="hard")
        assert _hard_has_derived_step(q), (seed, q.text)


def test_skill_a_easy_medium_never_carry_the_hard_marker():
    for difficulty in ("easy", "medium"):
        for seed in range(60):
            assert not _hard_has_derived_step(make(TOPIC_A, seed=seed, difficulty=difficulty))


def test_skill_a_hard_distractors_include_the_stated_groups_own_share():
    """The stated group's own share given/total is still the wrong-group distractor and is still offered by
    construction."""
    import re

    offered_count = 0
    for seed in range(60):
        q = make(TOPIC_A, seed=seed, difficulty="hard")
        total, given = (int(n) for n in re.findall(r"\d+", q.text))
        offered = {Fraction(*(int(p) for p in str(d).split("/"))) for d in q.distractors}
        assert _value(q) not in offered
        offered_count += Fraction(given, total) in offered
    assert offered_count >= 30, offered_count


# Shares are proper fractions; the key's rank is spread -----


def _skill_a_options(q):
    return [Fraction(*(int(p) for p in str(o).split("/"))) for o in [q.answer_text, *q.distractors]]


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_no_distractor_is_above_one_or_nonpositive(difficulty):
    """A share of a whole is a proper fraction: no option may exceed 1."""
    bad = []
    for seed in range(200):
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        if any(not (0 < v < 1) for v in _skill_a_options(q)):
            bad.append((seed, q.answer_text, q.distractors))
    assert not bad, f"{difficulty}: {len(bad)}/200 items offer an option outside (0, 1): {bad[:3]}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_key_rank_is_spread_over_all_four_positions(difficulty):
    """The key's rank among the four options (1 = smallest) must not
    concentrate. The target was 25-35% per rank, which four ranks summing
    to 100% cannot all satisfy except at exactly 25%; the sweep asserts the
    achievable band: every rank at most 35% and at least 18% of 200 seeds."""
    ranks = Counter()
    for seed in range(200):
        options = _skill_a_options(make(TOPIC_A, seed=seed, difficulty=difficulty))
        ranks[sorted(options).index(options[0]) + 1] += 1
    for r in (1, 2, 3, 4):
        assert 0.18 <= ranks[r] / 200 <= 0.35, f"{difficulty}: rank counts {dict(ranks)}"


# --- tier-specific invariants: Skill B -----------------------------------


def test_skill_b_easy_ratio_genuinely_reduces():
    """easy counts always share a factor of 2-9, so 'simplify' is never a
    no-op (never already coprime)."""
    import re

    for seed in range(60):
        q = make(TOPIC_B, seed=seed, difficulty="easy")
        first_sentence = q.text.split(".", 1)[0]
        nums = [int(n) for n in re.findall(r"\d+", first_sentence)]
        assert len(nums) == 2, q.text
        assert math.gcd(*nums) > 1, q.text


def test_skill_b_exam_states_one_decimal_place_measurements():
    import re

    for seed in range(60):
        q = make(TOPIC_B, seed=seed, difficulty="exam")
        decimals = re.findall(r"\d+\.\d", q.text)
        assert len(decimals) >= 2, q.text


# --- skill contract -------------------------------------------------------


def test_skill_a_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["senior-understand-the-relationship-between-fractions-and-ratio"]

    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert skill.prerequisites == []
    assert {source.topic for source in skill.sources} == {TOPIC_A}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.stages for source in skill.sources][-1] == ["exam"]


def test_skill_b_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["senior-demonstrate-an-understanding-of-the-elementary-ideas-and"]

    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert skill.prerequisites == ["order-of-operations"]
    assert {s.topic for s in skill.sources if s.difficulty != "exam"} == {TOPIC_B}
    assert [s.topic for s in skill.sources if s.difficulty == "exam"] == ["exd_x_r3i_ratio_batch"]
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.stages for source in skill.sources][-1] == ["exam"]
