"""Tests for:
- ucv1-chinese-remainder-two-congruences (number-theory)
- ucv1-inclusion-exclusion-two-sets (discrete-mathematics)
"""

import math
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent
from mathgen.generators.p11e_number_discrete_build import (
    _crt_combine,
    _verify_crt,
    _verify_neither,
    _verify_union_by_sets,
)

CRT_TOPIC = "chinese_remainder_two_congruences"
IE_TOPIC = "inclusion_exclusion_two_sets"
TOPICS = [CRT_TOPIC, IE_TOPIC]
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)
VARIETY_SEEDS = range(60)

_SKILL_IDS = {
    CRT_TOPIC: "ucv1-chinese-remainder-two-congruences",
    IE_TOPIC: "ucv1-inclusion-exclusion-two-sets",
}

_CRT_RE = re.compile(r"x ≡ (-?\d+) \(mod (\d+)\).*?x ≡ (-?\d+) \(mod (\d+)\)")


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, topic, difficulty, seed):
    """Generate through the skill's own `sources` entry, not a bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[_SKILL_IDS[topic]]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------

# --- Answer/distractor formatting: plain integer, no comma, no decimal ------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_are_plain_integers_no_comma(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert isinstance(q.answer, int)
        assert "." not in q.answer_text
        assert "," not in q.answer_text
        for d in q.distractors:
            assert "," not in d


# --- Answers stay well under the four-digit magnitude the conventions cap --


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_stays_under_1000(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert 0 <= q.answer < 1000, (topic, difficulty, seed, q.answer)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        wrong = str(q.answer + 1)  # an integer answer graded EXACTLY: +1 always clears it
        assert not grade_number(q, wrong).correct, (topic, difficulty, seed, q.text, wrong)


# --- Independent check fires on a corrupted answer ---------------------


def test_verify_crt_independent_check_fires_on_corrupted_answer():
    x = _crt_combine(2, 3, 4, 5)  # true answer 14
    with pytest.raises(VerificationError):
        _verify_crt(x + 1, 2, 3, 4, 5)


def test_verify_union_by_sets_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_union_by_sets(a_only=5, b_only=3, inter=2, union=999)


def test_verify_neither_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_neither(total=50, union=30, neither=999)


def test_check_equivalent_itself_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "must differ")


# --- CRT: coprimality is guaranteed on every draw, every tier ----------------

_CRT_MODULUS_BOUNDS = {"easy": (2, 6), "medium": (4, 12), "hard": (4, 12)}


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_crt_moduli_are_always_coprime_and_in_range(skills_dict, difficulty):
    lo, hi = _CRT_MODULUS_BOUNDS[difficulty]
    for seed in VARIETY_SEEDS:
        q = _make_from_sources(skills_dict, CRT_TOPIC, difficulty, seed)
        m = _CRT_RE.search(q.text)
        assert m, (difficulty, seed, q.text)
        _, m1_s, _, m2_s = m.groups()
        m1, m2 = int(m1_s), int(m2_s)
        assert lo <= m1 <= hi and lo <= m2 <= hi, (difficulty, seed, m1, m2)
        assert math.gcd(m1, m2) == 1, (difficulty, seed, m1, m2, "non-coprime system was never excluded")


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_crt_remainders_are_canonical_non_negative(skills_dict, difficulty):
    for seed in VARIETY_SEEDS:
        q = _make_from_sources(skills_dict, CRT_TOPIC, difficulty, seed)
        m = _CRT_RE.search(q.text)
        r1_s, m1_s, r2_s, m2_s = m.groups()
        r1, m1, r2, m2 = int(r1_s), int(m1_s), int(r2_s), int(m2_s)
        assert 0 <= r1 < m1, (difficulty, seed, r1, m1)
        assert 0 <= r2 < m2, (difficulty, seed, r2, m2)


def test_crt_hard_tier_first_remainder_is_always_negative(skills_dict):
    for seed in VARIETY_SEEDS:
        q = _make_from_sources(skills_dict, CRT_TOPIC, "hard", seed)
        m = _CRT_RE.search(q.text)
        r1_s, m1_s, r2_s, m2_s = m.groups()
        r1_raw, m1 = int(r1_s), int(m1_s)
        assert r1_raw < 0, (seed, q.text)
        assert 0 <= (r1_raw % m1) < m1  # reduces to a genuine canonical residue


def test_crt_hard_tier_sign_flip_never_coincides_with_the_answer(skills_dict):
    """The sign-flip misconception (reading the negative remainder's digits
    as positive instead of reducing them) is guaranteed distinct from the
    true answer by construction (resampled whenever 2*r1 == m1); confirm it
    over a real sweep by recomputing it here, via a SECOND, independently
    re-typed CRT combination (not imported from the generator's own
    `_crt_combine` call site inside `_crt_question`, though it does reuse
    the shared `_crt_combine` helper itself -- the point under test is the
    resample guard, not the arithmetic route)."""
    for seed in VARIETY_SEEDS:
        q = _make_from_sources(skills_dict, CRT_TOPIC, "hard", seed)
        m = _CRT_RE.search(q.text)
        r1_s, m1_s, r2_s, m2_s = m.groups()
        r1_raw, m1, r2, m2 = int(r1_s), int(m1_s), int(r2_s), int(m2_s)
        r1_true = r1_raw % m1
        sign_flip_value = m1 - r1_true
        x_sign_flip = _crt_combine(sign_flip_value, m1, r2, m2)
        assert x_sign_flip != int(q.answer_text), (seed, q.text)


# --- CRT: the independent brute-force check agrees on every rendered seed --


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_crt_brute_force_finds_exactly_one_solution_matching_the_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, CRT_TOPIC, difficulty, seed)
        m = _CRT_RE.search(q.text)
        r1_s, m1_s, r2_s, m2_s = m.groups()
        r1_raw, m1, r2, m2 = int(r1_s), int(m1_s), int(r2_s), int(m2_s)
        solutions = [x for x in range(m1 * m2) if x % m1 == r1_raw % m1 and x % m2 == r2 % m2]
        assert len(solutions) == 1, (difficulty, seed, q.text)
        assert solutions[0] == int(q.answer_text), (difficulty, seed, q.text)


# --- Inclusion-exclusion: every intermediate/final quantity is non-negative -


def test_ie_hard_tier_ignores_neither_distractor_never_negative(skills_dict):
    """'s own required assertion: `ignores-neither` can go negative by formula (|A| + |B| - T), and the generator
    resamples whenever it would -- confirmed over a real 60-seed sweep, every distractor value."""
    for seed in VARIETY_SEEDS:
        q = _make_from_sources(skills_dict, IE_TOPIC, "hard", seed)
        for d in q.distractors:
            assert int(d) >= 0, (seed, q.text, d)


def test_ie_hard_tier_no_negative_quantity_appears_in_the_question(skills_dict):
    for seed in VARIETY_SEEDS:
        q = _make_from_sources(skills_dict, IE_TOPIC, "hard", seed)
        assert "-" not in q.text, (seed, q.text)
        for n in re.findall(r"\d+", q.text):
            assert int(n) >= 0


# --- Distractor-count / pool-size requirement: >= 20 distinct prompts per
#     tier over 60 seeds -------------------------------------------------

# --- Skill loading: prerequisites, answer kind, placement, stages -----------


@pytest.mark.parametrize("topic", TOPICS)
def test_skill_loads_with_expected_shape(skills_dict, topic):
    skill = skills_dict[_SKILL_IDS[topic]]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    stages = {stage for src in skill.sources for stage in src.stages}
    assert stages, f"{topic}: no stages declared across sources"


def test_crt_skill_prerequisites():
    _, skills = load_skills()
    skill = {s.id: s for s in skills}[_SKILL_IDS[CRT_TOPIC]]
    assert skill.prerequisites == ["modular-arithmetic-congruence", "euclidean-algorithm-gcd"]


def test_ie_skill_prerequisites():
    _, skills = load_skills()
    skill = {s.id: s for s in skills}[_SKILL_IDS[IE_TOPIC]]
    assert skill.prerequisites == ["finite-set-operations"]
