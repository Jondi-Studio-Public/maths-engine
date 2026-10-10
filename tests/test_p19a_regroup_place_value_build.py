"""See mathgen/generators/p19a_regroup_place_value_build.py for the tier design and misconceptions."""

from collections import Counter
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p19a_regroup_place_value"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)


def _triple(q):
    return q.answer["hundreds"], q.answer["tens"], q.answer["ones"]


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_across_seeds(difficulty):
    for seed in range(40):
        q1 = make(TOPIC, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_present_distinct_nonempty_and_not_the_answer(difficulty):
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) >= 2
        assert all(d for d in q.distractors)
        assert len(q.distractors) == len(set(q.distractors))
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_a_labelled_triple_of_ints(difficulty):
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        H, T, O = _triple(q)
        assert isinstance(H, int) and isinstance(T, int) and isinstance(O, int)
        assert q.answer_text == f"hundreds = {H}, tens = {T}, ones = {O}"
        assert 0 <= T <= 9 and 0 <= O <= 9


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_answer_rejects_distractors_and_a_wrong_value(difficulty):
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_pair(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_pair(q, d).correct
        H, T, O = _triple(q)
        deliberately_wrong = f"hundreds = {H + 3}, tens = {T}, ones = {O}"
        assert not grade_pair(q, deliberately_wrong).correct


def test_grader_probe_labelled_any_order_bare_positional_and_concatenated_rejected():
    """The exact probe: the labelled form in any
    label order, a bare positional triple, a "H hundreds, T tens, O ones"
    phrase, and a run-on concatenated numeral."""
    q = make(TOPIC, seed=3, difficulty="hard")
    H, T, O = _triple(q)
    assert grade_pair(q, f"hundreds = {H}, tens = {T}, ones = {O}").correct
    assert grade_pair(q, f"tens = {T}, ones = {O}, hundreds = {H}").correct
    assert grade_pair(q, f"{H}, {T}, {O}").correct
    assert grade_pair(q, f"{H} hundreds, {T} tens, {O} ones").correct
    concatenated = f"{H}{T}{O}"
    assert not grade_pair(q, concatenated).correct


# --- variety ------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions in 60 seeds"


# --- 200-seed answer-concentration + zero/three-digit-case instruments -----


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_triple_concentration_and_case_shares(difficulty):
    triples = [_triple(make(TOPIC, seed=seed, difficulty=difficulty)) for seed in range(200)]
    counts = Counter(triples)
    top_triple, top_count = counts.most_common(1)[0]
    assert top_count / 200 < 0.10, (
        f"{difficulty}: triple {top_triple} appears in {top_count}/200 draws, at/over the 10% cap"
    )
    zero_case = sum(1 for (H, T, O) in triples if T == 0 or O == 0)
    three_digit_case = sum(1 for (H, T, O) in triples if H > 0)
    assert zero_case / 200 >= 0.25, f"{difficulty}: zero-digit case only {zero_case}/200"
    assert three_digit_case >= 1, f"{difficulty}: no three-digit result in 200 draws"


# --- tier-specific invariants --------------------------------------------


def test_easy_hundreds_never_stated_in_prompt():
    import re

    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert not re.search(r"\d+ hundreds?", q.text), q.text


def test_medium_hundreds_never_stated_in_prompt():
    import re

    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        assert not re.search(r"\d+ hundreds?", q.text), q.text


def test_medium_stated_tens_always_at_least_ten():
    import re

    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = re.search(r"(\d+) tens", q.text)
        assert m, q.text
        assert int(m.group(1)) >= 10


def test_hard_states_hundreds_explicitly():
    import re

    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert re.search(r"\d+ hundreds?", q.text), q.text


# --- skill contract -------------------------------------------------------


def test_skill_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["f10-regroup-zero-place-value"]

    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert sorted(skill.prerequisites) == ["place-value-to-100"]
    assert {source.topic for source in skill.sources} == {TOPIC}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard"]
    assert [source.stages for source in skill.sources] == [
        [0, 1, 2],
        [3, 4],
        [5, 6, "maintenance"],
    ]
    assert len(skill.misconceptions) == 3
