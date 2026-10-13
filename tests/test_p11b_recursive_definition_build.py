"""Focused contract tests for the recursive-definition skills.

The reverse-inference direction of the shipping
senior-use-recursion-to-generate-an-arithmetic-sequence /
senior-use-recursion-to-generate-a-geometric-sequence skills
(``p3c_recursion_build.py``): here t1 and the rule's constant are never
given, only the terms, and the learner states the constant as a labelled
`pair` (``answer: pair``, not ``ordered_list``)."""

from pathlib import Path
import random
import re

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p11b_recursive_definition_build import (
    _verify_arithmetic_definition,
    _verify_geometric_definition,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]

ARITH_ID = "senior-recognise-and-use-the-recursive-definition-of-an"
GEOM_ID = "senior-recognise-and-use-the-recursive-definition-of-a"
ARITH_TOPIC = "arithmetic_recursive_definition_state"
GEOM_TOPIC = "geometric_recursive_definition_state"
TOPICS = [ARITH_TOPIC, GEOM_TOPIC]


# --- Determinism, shape, distractor hygiene -------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        q1 = make(topic, seed=seed, difficulty=difficulty)
        q2 = make(topic, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_labelled_pair_of_plain_ints(topic, difficulty):
    label = "d" if topic == ARITH_TOPIC else "r"
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        assert set(q.answer) == {"first_term", label}
        assert isinstance(q.answer["first_term"], int)
        assert isinstance(q.answer[label], int)
        # No decimals/fractions anywhere in the canonical text.
        assert re.fullmatch(rf"first_term = -?\d+, {label} = -?\d+", q.answer_text), q.answer_text


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_non_empty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert q.distractors, f"seed {seed}: no distractors"
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors)), q.distractors


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_never_states_t1_or_constant_in_question_text(topic, difficulty):
    """The question must never leak t1= or d=/r= -- only the bare terms."""
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert "t1 =" not in q.text
        assert "d =" not in q.text
        assert "r =" not in q.text


@pytest.mark.parametrize("topic,label", [(ARITH_TOPIC, "d"), (GEOM_TOPIC, "r")])
def test_distractors_formatted_identically_to_the_answer(topic, label):
    """No option is identifiable by its shape -- every option (answer and
    every distractor) is 'first_term = <int>, {label} = <int>'."""
    pattern = re.compile(rf"^first_term = -?\d+, {label} = -?\d+$")
    for seed in SEEDS:
        for tier in TIERS:
            q = make(topic, seed=seed, difficulty=tier)
            assert pattern.match(q.answer_text), q.answer_text
            for d in q.distractors:
                assert pattern.match(d), d


# --- Real grader: accept canonical, reject every distractor and a deliberate wrong answer


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"seed {seed}: canonical {q.answer_text!r} not accepted"
        for i, distractor in enumerate(q.distractors):
            result = grade_pair(q, distractor)
            assert not result.correct, f"seed {seed}: distractor {i} {distractor!r} incorrectly accepted"


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_labelled_pair_order_independence(topic, difficulty):
    """'d = X, first_term = Y' (labels swapped) is ALSO accepted -- the
    order-independent labelled-pair grading path this design depends on
    (mathgen/grading/typed.py::_labelled_pair_verdict)."""
    label = "d" if topic == ARITH_TOPIC else "r"
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        first_term = q.answer["first_term"]
        const = q.answer[label]
        swapped = f"{label} = {const}, first_term = {first_term}"
        result = grade_pair(q, swapped)
        assert result.correct, f"seed {seed}: swapped-label answer {swapped!r} not accepted"


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_deliberately_wrong_answer_is_rejected(topic, difficulty):
    label = "d" if topic == ARITH_TOPIC else "r"
    for seed in range(20):
        q = make(topic, seed=seed, difficulty=difficulty)
        wrong = f"first_term = {q.answer['first_term'] + 1}, {label} = {q.answer[label]}"
        result = grade_pair(q, wrong)
        assert not result.correct, f"seed {seed}: deliberately wrong {wrong!r} accepted"


# --- Independent check fires on corrupted answers -------------------------


def test_arithmetic_independent_check_fires_on_corrupted_first_pair():
    with pytest.raises(VerificationError):
        _verify_arithmetic_definition(1, 2, [1, 3, 6])  # should be [1, 3, 5]


def test_arithmetic_independent_check_fires_on_wrong_first_term():
    with pytest.raises(VerificationError):
        _verify_arithmetic_definition(5, 2, [1, 3, 5])  # terms[0] != a1


def test_geometric_independent_check_fires_on_corrupted_last_pair():
    with pytest.raises(VerificationError):
        _verify_geometric_definition(2, 3, [2, 6, 19])  # should be [2, 6, 18]


def test_geometric_independent_check_fires_on_inexact_ratio():
    with pytest.raises(VerificationError):
        _verify_geometric_definition(7, 2, [7, 15, 28])  # 15 not exactly divisible by 7


# --- Arithmetic-specific tier construction --------------------------------

_ARITH_TERMS_RE = re.compile(r"(-?\d+(?:,\s*-?\d+)+)\.\s+State ")


def test_arithmetic_easy_has_three_terms_and_positive_d():
    for seed in SEEDS:
        q = make(ARITH_TOPIC, seed=seed, difficulty="easy")
        match = _ARITH_TERMS_RE.search(q.text)
        assert match, q.text
        terms = [t.strip() for t in match.group(1).split(",")]
        assert len(terms) == 3, terms
        assert q.answer["d"] >= 1


def test_arithmetic_medium_has_four_terms_and_allows_negative_d():
    saw_negative = False
    for seed in SEEDS:
        q = make(ARITH_TOPIC, seed=seed, difficulty="medium")
        assert q.answer["first_term"] != 0
        if q.answer["d"] < 0:
            saw_negative = True
    assert saw_negative, "never saw a negative d in medium over 40 seeds"


def test_arithmetic_hard_embeds_a_real_world_context():
    contexts = ("temperature", "altitude", "bank account balance", "drone")
    for seed in SEEDS:
        q = make(ARITH_TOPIC, seed=seed, difficulty="hard")
        assert any(c in q.text for c in contexts), q.text
        assert q.answer["first_term"] != 0


def test_arithmetic_uses_magnitude_only_distractor_only_when_d_negative():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(ARITH_TOPIC, seed=seed, difficulty=tier)
            magnitude_only = f"first_term = {q.answer['first_term']}, d = {abs(q.answer['d'])}"
            if q.answer["d"] < 0:
                assert magnitude_only in q.distractors, (tier, seed, q.distractors)
            else:
                assert magnitude_only not in q.distractors or magnitude_only == q.answer_text


def test_arithmetic_off_by_one_and_adds_instead_always_present():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(ARITH_TOPIC, seed=seed, difficulty=tier)
            a1, d = q.answer["first_term"], q.answer["d"]
            terms1 = a1 + d
            off_by_one = f"first_term = {terms1}, d = {d}"
            adds_instead = f"first_term = {a1}, d = {a1 + terms1}"
            assert off_by_one in q.distractors, (tier, seed, q.distractors)
            assert adds_instead in q.distractors, (tier, seed, q.distractors)


# --- Geometric-specific tier construction ---------------------------------


def test_geometric_easy_has_three_terms_and_positive_growth():
    for seed in SEEDS:
        q = make(GEOM_TOPIC, seed=seed, difficulty="easy")
        assert q.answer["r"] in (2, 3, 4)
        assert q.answer["first_term"] > 0


def test_geometric_medium_allows_negative_alternating_ratio():
    saw_negative = False
    for seed in SEEDS:
        q = make(GEOM_TOPIC, seed=seed, difficulty="medium")
        if q.answer["r"] < 0:
            saw_negative = True
    assert saw_negative, "never saw a negative r in medium over 40 seeds"


def test_geometric_hard_embeds_a_real_world_context_matching_sign():
    growth_contexts = ("bacteria", "investment", "viral video")
    alternating_contexts = ("pendulum", "signal")
    for seed in SEEDS:
        q = make(GEOM_TOPIC, seed=seed, difficulty="hard")
        r = q.answer["r"]
        if r > 0:
            assert any(c in q.text for c in growth_contexts), q.text
        else:
            assert any(c in q.text for c in alternating_contexts), q.text


def test_geometric_uses_magnitude_only_distractor_only_when_r_negative():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(GEOM_TOPIC, seed=seed, difficulty=tier)
            magnitude_only = f"first_term = {q.answer['first_term']}, r = {abs(q.answer['r'])}"
            if q.answer["r"] < 0:
                assert magnitude_only in q.distractors, (tier, seed, q.distractors)
            else:
                assert magnitude_only not in q.distractors or magnitude_only == q.answer_text


def test_geometric_off_by_one_always_present():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(GEOM_TOPIC, seed=seed, difficulty=tier)
            a1, r = q.answer["first_term"], q.answer["r"]
            off_by_one = f"first_term = {a1 * r}, r = {r}"
            assert off_by_one in q.distractors, (tier, seed, q.distractors)


def test_geometric_adds_instead_of_multiplies_collision_resample_never_exhausts():
    """The adds-instead-of-multiplies candidate must always end up distinct
    from the answer -- the resample loop (up to 20 attempts) must never be
    exhausted across a real sweep."""
    for tier in TIERS:
        for seed in SEEDS:
            q = make(GEOM_TOPIC, seed=seed, difficulty=tier)  # would raise if exhausted
            a1, r = q.answer["first_term"], q.answer["r"]
            assert a1 * (r - 1) != r


def test_geometric_adds_instead_of_multiplies_collision_rate_measured():
    """Measured (not assumed) pre-resample collision rate for the adds-instead-of-multiplies candidate, over the
    actual draw ranges each tier uses."""

    def collision_rate(r_choices, a1_lo, a1_hi, n=20000, seed=12345):
        rng = random.Random(seed)
        hits = 0
        for _ in range(n):
            r = rng.choice(r_choices)
            a1 = rng.randint(a1_lo, a1_hi)
            if a1 * (r - 1) == r:
                hits += 1
        return hits / n

    easy_rate = collision_rate([2, 3, 4], 1, 9)
    other_rate = collision_rate([2, 3, 4, -2, -3, -4], 1, 9)
    assert 0 <= easy_rate < 0.10
    assert 0 <= other_rate < 0.10


# --- Skill contract loading tests -----------------------------------------


def test_both_skills_load_with_answer_pair_and_their_prerequisites():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    arith = by_id[ARITH_ID]
    geom = by_id[GEOM_ID]
    assert arith.answer == "pair"
    assert geom.answer == "pair"
    assert arith.prerequisites == ["generate-sequence-from-rule", "common-difference"]
    assert geom.prerequisites == ["generate-sequence-from-rule", "ratio-rate-proportion"]


def test_both_skills_are_active_and_not_placement_eligible():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    for skill_id in (ARITH_ID, GEOM_ID):
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert skill.placement_eligible is False


def test_both_skills_have_sources_covering_all_difficulties():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    for skill_id in (ARITH_ID, GEOM_ID):
        # an optional exam source is covered by that skill's tests/test_exd_x_* file
        difficulties = {src.difficulty for src in by_id[skill_id].sources} - {"exam"}
        assert difficulties == {"easy", "medium", "hard"}


# --- Variety: at least 20 distinct rendered questions per tier over 60 seeds


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{topic} [{difficulty}]: only {len(texts)} distinct questions over 60 seeds"
