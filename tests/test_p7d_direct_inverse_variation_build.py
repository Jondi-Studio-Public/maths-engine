"""Tests `p7d_examine_direct_proportion` and `p7d_examine_inverse_proportion`: distractor counts and hard-tier
answer checks."""

import functools

"""Direct and inverse variation.

- senior-examine-examples-of-direct-proportion-and-linearly-related (ACMMM002)
- senior-examine-examples-of-inverse-proportion (ACMMM012)

The hard tier of both topics avoids echoing its givens in the answer, and easy/medium add a "no-change" distractor to easy/medium
(distractor-count consistency fix).
"""
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7d_direct_inverse_variation_build import (
    _verify_direct,
    _verify_direct_hard,
    _verify_inverse,
    _verify_inverse_hard,
)

SEEDS = range(40)

TOPIC_DIRECT = "p7d_examine_direct_proportion"
TOPIC_INVERSE = "p7d_examine_inverse_proportion"
SKILL_DIRECT = "senior-examine-examples-of-direct-proportion-and-linearly-related"
SKILL_INVERSE = "senior-examine-examples-of-inverse-proportion"

DIFFICULTIES = ["easy", "medium", "hard"]


# `load_skills()` rescans every skill YAML from disk and costs ~560 ms. These
# helpers are called once per SEED, so an uncached lookup spent ~290 s of the
# suite reloading the same library. Cached: `make()` still runs per seed and
# every assertion is unchanged -- only the repeated disk scan is removed.
@functools.lru_cache(maxsize=None)
def _skill(skill_id):
    _domains, skills_list = load_skills()
    by_id = {s.id: s for s in skills_list}
    return by_id[skill_id]


def _make_via_sources(skill_id, difficulty, seed):
    skill = _skill(skill_id)
    source = next(s for s in skill.sources if s.difficulty == difficulty)
    params = source.params or {}
    return make(source.topic, seed=seed, difficulty=source.difficulty, **params)


# --- Determinism --------------------------------------------------------

# --- Distractor shape -----------------------------------------------------


@pytest.mark.parametrize("skill_id", [SKILL_DIRECT, SKILL_INVERSE])
def test_distractor_count_matches_construction(skill_id):
    """Every tier ships exactly three distractors: the two named cross-relation misconceptions ("as-if-X" for the
    other two relations) plus a generic "no-change" (forgot the value changes) candidate -- brought
    easy/medium into line with the hard tier and with the sibling quadratic-variation skill's convention,
    which all three tiers now match."""
    for difficulty in DIFFICULTIES:
        for seed in SEEDS[:10]:
            q = _make_via_sources(skill_id, difficulty, seed)
            assert len(q.distractors) == 3, (
                f"{skill_id}/{difficulty} seed {seed}: expected 3 distractors, got {q.distractors}"
            )


# --- Hard tier: nothing printed in the question equals the answer ---------
#
# The fix composes two scale steps and never prints the combined factor. This test is the direct
# regression guard: the answer must never equal a number printed in the question, over every seed --
# an "echo the question" learner gets nothing. The one designed exception is the "no-change"
# distractor (deliberately the printed "before" value F1/P1, a legitimate "forgot the value changes"
# wrong answer) -- so exactly one distractor, never the answer, may echo a printed number, and it
# must be the stated "before" force/pressure specifically (not some other printed number by
# coincidence).

_BEFORE_VALUE_RE = re.compile(r"force is (\d+) N|pressure is (\d+) kPa")


@pytest.mark.parametrize("skill_id", [SKILL_DIRECT, SKILL_INVERSE])
def test_hard_tier_answer_never_echoes_a_printed_number(skill_id):
    for seed in range(60):
        q = _make_via_sources(skill_id, "hard", seed)
        printed = {int(tok) for tok in re.findall(r"\d+", q.text)}
        assert int(q.answer_text) not in printed, (
            f"{skill_id} hard seed {seed}: answer {q.answer_text} equals a number printed in the question: {q.text!r}"
        )
        echoing = [d for d in q.distractors if int(d) in printed]
        assert len(echoing) == 1, (
            f"{skill_id} hard seed {seed}: expected exactly one distractor (the deliberate "
            f"no-change candidate) to echo a printed number, found {echoing} in {q.text!r}"
        )
        m = _BEFORE_VALUE_RE.search(q.text)
        assert m, f"{skill_id} hard seed {seed}: could not parse the stated before-value: {q.text!r}"
        before_value = int(m.group(1) or m.group(2))
        assert int(echoing[0]) == before_value, (
            f"{skill_id} hard seed {seed}: the echoing distractor {echoing[0]!r} is not the "
            f"stated before-value {before_value} in {q.text!r}"
        )


# --- Every candidate is a positive integer, formatted alike ---------------


@pytest.mark.parametrize("skill_id", [SKILL_DIRECT, SKILL_INVERSE])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_every_distractor_is_a_plain_positive_integer(skill_id, difficulty):
    for seed in SEEDS:
        q = _make_via_sources(skill_id, difficulty, seed)
        assert isinstance(q.answer, int)
        assert q.answer > 0
        assert q.answer_text == str(q.answer)
        assert q.answer_text.isdigit()
        for d in q.distractors:
            assert d.isdigit(), f"seed {seed}: distractor {d!r} is not a plain positive integer string"


# --- Real grader accept/reject -------------------------------------------


@pytest.mark.parametrize("skill_id", [SKILL_DIRECT, SKILL_INVERSE])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_rejects_distractors(skill_id, difficulty):
    for seed in SEEDS:
        q = _make_via_sources(skill_id, difficulty, seed)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed {seed}: canonical answer {q.answer_text!r} rejected"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed {seed}: distractor {d!r} was accepted"


@pytest.mark.parametrize("skill_id", [SKILL_DIRECT, SKILL_INVERSE])
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_deliberately_wrong_answer(skill_id, difficulty):
    for seed in SEEDS:
        q = _make_via_sources(skill_id, difficulty, seed)
        wrong = str(int(q.answer_text) + 1000003)  # far outside any tolerance
        result = grade_number(q, wrong)
        assert not result.correct, f"seed {seed}: wrong answer {wrong!r} accepted"


# --- At least 20 distinct rendered questions per tier over 60 seeds -------

# --- Independent check fires on corrupted answers, every tier -------------


def test_direct_independent_check_fires_on_corrupted_answer():
    _verify_direct(3, 15, 15, 75, "must not raise")  # y=kx: k=5, y2=5*15=75, correct
    with pytest.raises(VerificationError):
        _verify_direct(3, 15, 15, 76, "corrupted y2")  # off by one


def test_direct_hard_independent_check_fires_on_corrupted_answer():
    # x1=7, f1=3, t1=2, t2=3: F2 = 3*2*3 = 18, genuinely correct.
    _verify_direct_hard(7, 3, 2, 3, 18, "must not raise")
    with pytest.raises(VerificationError):
        _verify_direct_hard(7, 3, 2, 3, 19, "corrupted F2")


def test_inverse_independent_check_fires_on_corrupted_answer():
    _verify_inverse(2, 10, 4, 5, "must not raise")  # x*y=20 both sides
    with pytest.raises(VerificationError):
        _verify_inverse(2, 10, 4, 6, "corrupted y2")


def test_inverse_hard_independent_check_fires_on_corrupted_answer():
    # v1=2, p1=12, t1=2, t2=3: P2 = 12/(2*3) = 2, genuinely correct.
    _verify_inverse_hard(2, 12, 2, 3, 2, "must not raise")
    with pytest.raises(VerificationError):
        _verify_inverse_hard(2, 12, 2, 3, 3, "corrupted P2")


# --- Nonsensical preset raises -------------------------------------------


def test_direct_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make(TOPIC_DIRECT, seed=0, variant="not_a_real_variant")


def test_inverse_raises_on_nonsensical_preset():
    with pytest.raises(ValueError):
        make(TOPIC_INVERSE, seed=0, variant="not_a_real_variant")


# --- Skill loading ---------------------------------------------------------


@pytest.mark.parametrize("skill_id", [SKILL_DIRECT, SKILL_INVERSE])
def test_skill_loads_with_correct_properties(skill_id):
    skill = _skill(skill_id)
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["function-notation"]


@pytest.mark.parametrize("skill_id,topic", [(SKILL_DIRECT, TOPIC_DIRECT), (SKILL_INVERSE, TOPIC_INVERSE)])
def test_skill_sources_cover_correct_stages(skill_id, topic):
    skill = _skill(skill_id)
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == topic
    by_difficulty = {s.difficulty: s for s in skill.sources}
    assert set(by_difficulty["easy"].stages) == {0, 1, 2}
    assert set(by_difficulty["medium"].stages) == {3, 4}
    assert set(by_difficulty["hard"].stages) == {5, 6, "maintenance"}


# The formula and relation type are always stated --


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_direct_question_always_states_the_formula(difficulty):
    for seed in SEEDS[:10]:
        q = _make_via_sources(SKILL_DIRECT, difficulty, seed)
        assert "y = kx" in q.text or "F = kx" in q.text


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_inverse_question_always_states_the_formula(difficulty):
    for seed in SEEDS[:10]:
        q = _make_via_sources(SKILL_INVERSE, difficulty, seed)
        assert "y = k/x" in q.text or "P = k/V" in q.text
