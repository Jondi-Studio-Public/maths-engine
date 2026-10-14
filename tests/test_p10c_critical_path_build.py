"""Tests the critical-path location (`mcq`) and float-time (`number`) skills built on earliest and latest start
times.

Critical-path location and float times.

Tests for:
- senior-use-ests-and-lsts-to-locate-the-critical (ACMGM106), answer: mcq
- senior-calculate-float-times-for-non-critical-activities (ACMGM108), answer: number
"""

import re

import pytest

from mathgen.grading.typed import grade_mcq, grade_number, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p10c_critical_path_build import (
    ACTIVITIES,
    PREDECESSORS,
    _backward_pass,
    _direct_path_sum_float,
    _draw_durations,
    _forward_pass,
    _path_enumeration_est_lst,
    _verify_float_independent,
    _verify_path1_wins_classification,
    _verify_shared_fixture,
)

LOCATE_TOPIC = "p10c_critical_path_locate"
FLOAT_TOPIC = "p10c_critical_path_float"
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]


# --- Determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [LOCATE_TOPIC, FLOAT_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [LOCATE_TOPIC, FLOAT_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


# --- Real grader accept/reject -- Skill 1 (mcq) -----------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_locate_real_grader_accepts_canonical_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        question = make(LOCATE_TOPIC, seed=seed, difficulty=difficulty)
        options = mcq_options(question)
        correct_idx = options.index(question.answer_text)
        assert grade_mcq(question, str(correct_idx)).correct
        for idx, opt in enumerate(options):
            if opt != question.answer_text:
                assert not grade_mcq(question, str(idx)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_locate_real_grader_rejects_ungradable_input(difficulty):
    question = make(LOCATE_TOPIC, seed=0, difficulty=difficulty)
    assert not grade_mcq(question, "banana").correct
    assert not grade_mcq(question, "99").correct  # out of range


# --- Real grader accept/reject -- Skill 2 (number) --------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_float_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(FLOAT_TOPIC, seed=seed, difficulty=difficulty)
        wrong = question.answer + 1
        assert not grade_number(question, str(wrong)).correct


# --- Answer-kind sanity ------------------------------------------------------


@pytest.mark.parametrize(
    "difficulty,valid",
    [
        ("easy", {"Yes", "No"}),
        ("medium", {"C", "E"}),
        ("hard", {"2", "3", "4", "5"}),
    ],
)
def test_locate_answer_is_a_valid_kind(difficulty, valid):
    for seed in SEEDS:
        question = make(LOCATE_TOPIC, seed=seed, difficulty=difficulty)
        assert question.answer_text in valid


@pytest.mark.parametrize("difficulty", TIERS)
def test_float_answer_always_a_positive_integer(difficulty):
    for seed in SEEDS:
        question = make(FLOAT_TOPIC, seed=seed, difficulty=difficulty)
        assert "." not in question.answer_text
        assert int(question.answer_text) > 0


# --- Shared fixture: independent check (path enumeration vs recursion) -----


def test_shared_fixture_agrees_both_routes_many_seeds():
    """Explicit path enumeration and forward/backward-pass recursion must
    agree on EST/LST for every activity, over many independently-drawn
    duration sets (both branches of which path wins)."""
    import random

    rng = random.Random(12345)
    checked = 0
    for _ in range(200):
        durations = {label: rng.randint(2, 12) for label in ACTIVITIES}
        _verify_shared_fixture(durations)  # raises on any mismatch
        checked += 1
    assert checked == 200


def test_shared_fixture_independent_check_fires_on_corrupted():
    """Both routes recompute from the SAME durations dict, so they always
    agree with each other for any valid input (this is an identity of the
    two formulas, proved above by sweeping 200 independent draws) -- there is
    no duration corruption that breaks agreement between them. To exercise
    the underlying `check_equivalent` firing on a genuine mismatch directly
    (the same primitive `_verify_shared_fixture` is built from), call it with
    a hand-crafted wrong value, mirroring `_verify_float_independent`'s and
    `_verify_path1_wins_classification`'s corrupted-input tests above."""
    durations = {"A": 3, "B": 12, "C": 6, "D": 10, "E": 11, "F": 11}
    est, eft = _forward_pass(durations)
    completion_time = max(eft.values())
    lst, lft = _backward_pass(durations, completion_time)
    est_pe, lst_pe, completion_pe = _path_enumeration_est_lst(durations)
    assert completion_pe == completion_time
    assert est_pe == est
    assert lst_pe == lst
    from mathgen.core import check_equivalent

    with pytest.raises(VerificationError):
        check_equivalent(lst_pe["A"], lst["A"] + 1, "deliberately wrong LST(A) must not verify")


def test_locate_classification_independent_check_fires_on_corrupted():
    _verify_path1_wins_classification(10, 5, True)  # true positive, no raise
    with pytest.raises(VerificationError):
        _verify_path1_wins_classification(10, 5, False)
    with pytest.raises(VerificationError):
        _verify_path1_wins_classification(5, 10, True)


def test_float_independent_check_fires_on_corrupted():
    durations = {"A": 3, "B": 12, "C": 6, "D": 10, "E": 11, "F": 11}
    est, eft = _forward_pass(durations)
    completion_time = max(eft.values())
    true_answer = _direct_path_sum_float(durations, est, completion_time, "A", ("C", "D"))
    _verify_float_independent(durations, est, completion_time, "A", ("C", "D"), true_answer)  # no raise
    with pytest.raises(VerificationError):
        _verify_float_independent(durations, est, completion_time, "A", ("C", "D"), true_answer + 1)


# --- Shared-float-value property (Skill 2) ----------------------------------


def test_shared_float_value_across_a_losing_chain():
    """Every activity on the same losing chain shares ONE float value --
    verified over many independently-drawn duration sets for both chains."""
    import random

    rng = random.Random(999)
    both_seen = {False: 0, True: 0}
    for _ in range(300):
        path1_wins = rng.choice([True, False])
        durations = _draw_durations(rng, path1_wins)
        est, eft = _forward_pass(durations)
        completion_time = max(eft.values())
        lst, lft = _backward_pass(durations, completion_time)
        if path1_wins:
            chain = ("B", "E")
        else:
            chain = ("A", "C", "D")
        floats = {label: lst[label] - est[label] for label in chain}
        assert len(set(floats.values())) == 1
        both_seen[path1_wins] += 1
    assert both_seen[True] > 0 and both_seen[False] > 0


def test_degenerate_lst_equals_float_for_activity_a():
    """The generator substitutes LFT(A) as the 'reports-lst-not-float' distractor for A specifically because of
    this."""
    import random

    rng = random.Random(42)
    for _ in range(100):
        durations = {label: rng.randint(2, 12) for label in ACTIVITIES}
        est, eft = _forward_pass(durations)
        completion_time = max(eft.values())
        lst, lft = _backward_pass(durations, completion_time)
        assert est["A"] == 0
        assert lst["A"] - est["A"] == lst["A"]


# Balance sweeps ----


def test_locate_easy_yes_no_balance_400_seeds():
    counts = {"Yes": 0, "No": 0}
    for seed in range(400):
        q = make(LOCATE_TOPIC, seed=seed, difficulty="easy")
        counts[q.answer_text] += 1
    total = sum(counts.values())
    assert total == 400
    # A fair coin's 400-draw share should not be wildly skewed; generous
    # bounds (six SDs on a fair coin at n=400) to avoid flaking.
    assert 0.35 <= counts["Yes"] / total <= 0.65
    assert 0.35 <= counts["No"] / total <= 0.65


def test_locate_medium_ce_balance_400_seeds():
    counts = {"C": 0, "E": 0}
    for seed in range(400):
        q = make(LOCATE_TOPIC, seed=seed, difficulty="medium")
        counts[q.answer_text] += 1
    total = sum(counts.values())
    assert total == 400
    assert 0.35 <= counts["C"] / total <= 0.65
    assert 0.35 <= counts["E"] / total <= 0.65


def test_locate_hard_3_vs_4_balance_400_seeds():
    counts = {"3": 0, "4": 0}
    for seed in range(400):
        q = make(LOCATE_TOPIC, seed=seed, difficulty="hard")
        counts[q.answer_text] += 1
    total = sum(counts.values())
    assert total == 400
    assert 0.35 <= counts["3"] / total <= 0.65
    assert 0.35 <= counts["4"] / total <= 0.65


# --- Pool variety: >=20 distinct rendered question TEXT per tier / 60 seeds


@pytest.mark.parametrize("topic", [LOCATE_TOPIC, FLOAT_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20


# --- Both skills load through the API ---------------------------------------


def test_both_skills_load_through_api():
    domains, skills = load_skills()
    skills_dict = {skill.id: skill for skill in skills}

    locate_skill = skills_dict.get("senior-use-ests-and-lsts-to-locate-the-critical")
    assert locate_skill is not None
    assert locate_skill.prerequisites == ["simple-interest"]
    assert locate_skill.answer == "mcq"
    assert locate_skill.placement_eligible is False
    assert len(locate_skill.sources) == 5  # easy, medium, hard, exam, exam-single

    float_skill = skills_dict.get("senior-calculate-float-times-for-non-critical-activities")
    assert float_skill is not None
    assert float_skill.prerequisites == ["simple-interest"]
    assert float_skill.answer == "number"
    assert float_skill.placement_eligible is False
    assert len(float_skill.sources) == 4  # easy, medium, hard, exam


def test_locate_raises_on_unknown_tier():
    with pytest.raises(ValueError):
        make(LOCATE_TOPIC, seed=1, tier="unknown")


def test_float_raises_on_unknown_tier():
    with pytest.raises(ValueError):
        make(FLOAT_TOPIC, seed=1, tier="unknown")


# Oracle-style independent parse of the rendered question --------------- Parses ``question.text``
# fresh (not the generator's own data structures) and recomputes the answer from scratch, the same
# discipline as tests/test_p6b_project_network_oracle.py.

_ACTIVITY_RE = re.compile(r"Activity\s+(\w+):\s+duration\s+(\d+)\s+days?,\s+(.+?)\.(?:\s|$)")


def _parse_network(text):
    """Independent parse: {label: (duration, [predecessors])}."""
    network = {}
    for match in _ACTIVITY_RE.finditer(text):
        label, duration, tail = match.group(1), int(match.group(2)), match.group(3).strip()
        if tail.lower() == "no predecessors":
            preds = []
        else:
            preds = [p.strip() for p in tail.split(":", 1)[1].split(",")]
        network[label] = (duration, preds)
    assert network, f"no activities parsed from: {text}"
    return network


def _oracle_cpm(network):
    """Fresh forward/backward pass over the parsed network -- does not import
    anything from mathgen.generators.p10c_critical_path_build."""
    durations = {label: d for label, (d, _preds) in network.items()}
    preds = {label: p for label, (_d, p) in network.items()}
    succ = {label: [] for label in network}
    for label, plist in preds.items():
        for p in plist:
            succ[p].append(label)

    est, eft = {}, {}
    remaining = set(network)
    while remaining:
        progressed = False
        for label in list(remaining):
            if all(p in eft for p in preds[label]):
                est[label] = 0 if not preds[label] else max(eft[p] for p in preds[label])
                eft[label] = est[label] + durations[label]
                remaining.discard(label)
                progressed = True
        assert progressed, "oracle CPM made no progress -- cyclic or malformed network"

    completion_time = max(eft.values())
    lst, lft = {}, {}
    remaining = set(network)
    while remaining:
        progressed = False
        for label in list(remaining):
            if all(s in lst for s in succ[label]):
                lft[label] = completion_time if not succ[label] else min(lst[s] for s in succ[label])
                lst[label] = lft[label] - durations[label]
                remaining.discard(label)
                progressed = True
        assert progressed, "oracle CPM (backward) made no progress"

    return est, eft, lst, lft, completion_time


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_locate_matches_generator(difficulty):
    for seed in range(60):
        q = make(LOCATE_TOPIC, seed=seed, difficulty=difficulty)
        network = _parse_network(q.text)
        est, eft, lst, lft, completion_time = _oracle_cpm(network)
        # Critical activities are those whose EST == LST (zero float).
        critical = {label for label in network if est[label] == lst[label]}
        if difficulty == "easy":
            expected = "Yes" if "D" in critical else "No"
        elif difficulty == "medium":
            expected = "C" if "C" in critical else "E"
        else:
            expected = str(len(critical))
        assert q.answer_text == expected, f"seed {seed} {difficulty}: oracle {expected}, generator {q.answer_text}"


@pytest.mark.parametrize("difficulty,activity", [("easy", "E"), ("medium", "A"), ("hard", "C")])
def test_oracle_float_matches_generator(difficulty, activity):
    for seed in range(60):
        q = make(FLOAT_TOPIC, seed=seed, difficulty=difficulty)
        network = _parse_network(q.text)
        est, eft, lst, lft, completion_time = _oracle_cpm(network)
        expected = lst[activity] - est[activity]
        assert int(q.answer_text) == expected, f"seed {seed} {difficulty}: oracle {expected}, generator {q.answer_text}"
        assert int(q.answer_text) not in {int(d) for d in q.distractors}
