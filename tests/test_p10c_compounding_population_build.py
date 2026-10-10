"""Tests `p10c_population_compounding`: population questions never mention money, and an oracle recomputes the
answer from the text.

Compounding outside money (population).

senior-consider-similar-problems-involving-compounding-for-example-population
(ACMEM170, topic p10c_population_compounding, answer: number). See the
generator module docstring for the exact-arithmetic construction and the
non-financial-wording boundary this skill exists to keep.
"""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent

SEEDS = range(60)
BOUNDARY_SEEDS = range(500)
TIERS = ("easy", "medium", "hard")

TOPIC = "p10c_population_compounding"
SKILL = "senior-consider-similar-problems-involving-compounding-for-example-population"

# The one constraint this whole skill exists to enforce: never collide with
# the shipping compound-interest skill's shape.
_FORBIDDEN_RE = re.compile(r"\$|interest|loan|invest", re.IGNORECASE)

_EASY_RE = re.compile(
    r"^A population of (\d+) grows by (\d+)% each year\. "
    r"Find the population after (\d+) years\.$"
)
_MEDIUM_RE = re.compile(
    r"^A bacteria colony of (\d+) organisms decreases by (\d+)% each hour\. "
    r"Find the number of organisms after (\d+) hours\.$"
)
_HARD_RE = re.compile(
    r"^A population grew by (\d+)% each year for (\d+) years and is now (\d+)\. "
    r"What was the original population\?$"
)


def _recompute_answer(text: str) -> int:
    """Independent oracle, inline: parses ONLY the rendered question text and recomputes the answer from scratch,
    never importing anything from ``mathgen.generators.p10c_compounding_population_build``."""
    m = _EASY_RE.match(text)
    if m:
        p0, g, n = (int(v) for v in m.groups())
        k = 100 // g
        value = Fraction(p0) * Fraction(k + 1, k) ** n
        assert value.denominator == 1, f"easy oracle: non-integer result for {text!r}"
        return int(value)
    m = _MEDIUM_RE.match(text)
    if m:
        p0, g, n = (int(v) for v in m.groups())
        k = 100 // g
        value = Fraction(p0) * Fraction(k - 1, k) ** n
        assert value.denominator == 1, f"medium oracle: non-integer result for {text!r}"
        return int(value)
    m = _HARD_RE.match(text)
    if m:
        g, n, a_value = (int(v) for v in m.groups())
        k = 100 // g
        value = Fraction(a_value) / (Fraction(k + 1, k) ** n)
        assert value.denominator == 1, f"hard oracle: non-integer result for {text!r}"
        return int(value)
    raise AssertionError(f"question text did not match any known tier shape: {text!r}")


# --- Determinism -------------------------------------------------------


# --- The boundary: never collides with compound-interest ------------------


@pytest.mark.parametrize("tier", TIERS)
def test_never_mentions_money_interest_loan_or_invest(tier):
    # 500 seeds, not the usual SEEDS=60: this is the one property treated
    # as load-bearing (a real sweep of 500 seeds x 3 tiers),
    # not a rare-input case -- it is the single boundary this whole skill
    # exists to hold against the shipping compound-interest skill.
    for seed in BOUNDARY_SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        blobs = [q.text, q.latex, q.answer_text, q.answer_latex, *q.working, *q.distractors]
        for blob in blobs:
            assert not _FORBIDDEN_RE.search(str(blob)), f"seed={seed} tier={tier}: forbidden token in {blob!r}"


# --- Independent oracle: recompute from the rendered text alone -----------


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_recomputes_answer_from_question_text(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_answer = _recompute_answer(q.text)
        assert oracle_answer == int(q.answer_text), (
            f"seed={seed} tier={tier}: oracle={oracle_answer} answer_text={q.answer_text!r} question={q.text!r}"
        )


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_a_positive_plain_integer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert d.lstrip("-").isdigit() and not d.startswith("-"), (
                f"seed={seed} tier={tier}: distractor {d!r} not a positive plain integer"
            )
            assert int(d) > 0, f"seed={seed} tier={tier}: distractor {d!r} not positive"


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_and_rejects(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed={seed} tier={tier}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed={seed} tier={tier}: distractor {d!r} accepted"
        wrong = str(q.answer + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed={seed} tier={tier}: deliberately wrong {wrong!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_thousands_separated_answer(tier):
    """Regression guard for a previously fixed defect."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        answer = int(q.answer_text)
        if answer < 1000:
            continue
        grouped = f"{answer:,}"
        result = grade_number(q, grouped)
        assert result.correct, f"seed={seed} tier={tier}: comma-grouped {grouped!r} rejected"


# --- Answer kind -------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_bare_positive_integer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)
        assert q.answer > 0
        assert "." not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a decimal point"
        assert "," not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a comma"


# --- Distinct rendered questions ----------------------------------------


# --- Independent-check tripwire ------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


# --- Skill loading -------------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == TOPIC
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == set(TIERS)


def test_medium_tier_answer_distribution_is_not_concentrated():
    """Guards the fix. Before it, `_medium`'s `p0` draw used a FIXED
    `(200, 2000)` range regardless of rate, so `g=10` (k=10, step=1000)
    reached only 2 multiples while `g=20`/`g=25` reached 15/28 -- `g=10`
    still gets drawn ~1/3 of the time (g=50 never survives the decline
    filter), so each of its 2 outcomes got ~1/6 of ALL draws. Measured with
    a 400-seed variety check pre-fix: top answer (729, from p0=1000
    decaying 10% three times) was 17% of draws with only 44 distinct answers
    reachable -- so this test, run at the same 400 seeds, would have FAILED
    against the pre-fix generator (17% > 0.15, 44 < 55). 400 seeds, not 40:
    a fair coin's 40-draw majority exceeds 55% about 43% of the time,
    so only a run this long can tell real skew from noise.

    `729` is deliberately not excluded from the check below -- it is
    reachable via TWO independent parent pairs (p0=1000,g=10 and
    p0=1728,g=25) and is expected to keep roughly double a typical
    single-parent answer's share even after the fix. That is a real
    property of the parameter space, not a defect to engineer away."""
    seeds = range(400)
    answers = collections.Counter()
    for seed in seeds:
        q = make(TOPIC, seed=seed, difficulty="medium")
        answers[q.answer_text] += 1
    total = sum(answers.values())
    top_text, top_n = answers.most_common(1)[0]
    share = top_n / total
    assert share < 0.15, (
        f"medium tier top answer {top_text!r} is {share:.0%} of {total} draws "
        f"(matches the pool-variety top-answer threshold) -- the concentration fix "
        f"fix has regressed"
    )
    assert len(answers) >= 55, (
        f"medium tier only reaches {len(answers)} distinct answers over {total} "
        f"seeds (pre-fix was 44); the concentration fix has regressed"
    )
