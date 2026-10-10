"""See mathgen/generators/p13a_divide_ratio_build.py for the tier design and misconceptions."""

from collections import Counter
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p13a_divide_ratio"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


def _pair(q):
    return tuple(q.answer.values())


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_and_not_the_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3
        assert all(d for d in q.distractors)
        assert len(q.distractors) == len(set(q.distractors))
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_an_unlabelled_ordered_pair_of_integers(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        share_a, share_b = _pair(q)
        assert isinstance(share_a, int) and isinstance(share_b, int)
        assert q.answer_text == f"({share_a}, {share_b})"
        assert "=" not in q.answer_text  # unlabelled: no "label = value" segment


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_answer_rejects_distractors_and_a_wrong_value(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_pair(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_pair(q, d).correct
        share_a, share_b = _pair(q)
        deliberately_wrong = f"({share_a + 1000}, {share_b + 1000})"
        assert not grade_pair(q, deliberately_wrong).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_grader_rejects_the_reversed_order_pair(difficulty):
    """The grading-path finding: `answer_text` carries no
    `label = value` segments, so `_labelled_pair_verdict` never engages and
    `grade_pair` falls to its POSITIONAL comparison -- a swapped-order pair
    is a genuinely different (wrong) answer, never accepted either way."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        share_a, share_b = _pair(q)
        if share_a == share_b:
            continue  # would not be a distinguishing case; never happens here (p != q)
        reversed_pair = f"({share_b}, {share_a})"
        assert not grade_pair(q, reversed_pair).correct


# --- variety ------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions in 60 seeds"


# --- 200-seed answer-concentration instrument (over the PAIR) -----------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_pair_concentration_stays_under_25_percent(difficulty):
    counts = Counter(_pair(make(TOPIC, seed=seed, difficulty=difficulty)) for seed in range(200))
    top_pair, top_count = counts.most_common(1)[0]
    assert top_count / 200 <= 0.25, f"{difficulty}: pair {top_pair} appears in {top_count}/200 draws, over the 25% cap"


# --- tier-specific invariants --------------------------------------------


def test_easy_shares_sum_to_the_stated_total():
    import re

    pattern = re.compile(r"(\d+).*ratio (\d+):(\d+)")
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        m = pattern.search(q.text)
        assert m, q.text
        total = int(m.group(1))
        share_a, share_b = _pair(q)
        assert share_a + share_b == total


def test_medium_shares_produce_the_stated_difference():
    import re

    pattern = re.compile(r"ratio (\d+):(\d+).*?(\d+)", re.DOTALL)
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = pattern.search(q.text)
        assert m, q.text
        diff = int(m.group(3))
        share_a, share_b = _pair(q)
        assert abs(share_a - share_b) == diff


def test_hard_first_component_matches_the_given_share():
    import re

    pattern = re.compile(r"ratio (\d+):(\d+).*?(\d+)", re.DOTALL)
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = pattern.search(q.text)
        assert m, q.text
        given_share = int(m.group(3))
        share_a, _share_b = _pair(q)
        assert share_a == given_share


def test_exam_never_states_the_ratio_as_whole_numbers_only():
    """The whole boundary against hard: the exam tier's stated ratio always
    carries a decimal point or a slash (fraction), never two bare integers."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        ratio_part = q.text.split("ratio", 1)[1]
        assert "." in ratio_part or "/" in ratio_part, q.text


def test_exam_shares_sum_to_the_stated_total():
    import re

    pattern = re.compile(r"^\D*(\d+)")
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        m = pattern.search(q.text)
        assert m, q.text
        total = int(m.group(1))
        share_a, share_b = _pair(q)
        assert share_a + share_b == total


# --- skill contract -------------------------------------------------------


def test_skill_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["divide-quantity-in-ratio"]

    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert sorted(skill.prerequisites) == ["fraction-of-quantity", "ratio-rate-proportion"]
    assert {source.topic for source in skill.sources} == {TOPIC}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.stages for source in skill.sources] == [
        [0, 1, 2],
        [3, 4],
        [5, 6, "maintenance"],
        ["exam"],
    ]
    assert len(skill.misconceptions) == 3
