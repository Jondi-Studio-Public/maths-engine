"""Pv0a_geometric_threshold (senior-use-geometric-sequences-in-contexts-involving-geometric-growth, ACMMM076)."""

import functools
import math
import re
from fractions import Fraction

import mathgen.generators.pv0a_geometric_threshold_build  # noqa: F401 -- registers the topic
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check, check_equivalent

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
TOPIC = "pv0a_geometric_threshold"
SKILL_ID = "senior-use-geometric-sequences-in-contexts-involving-geometric-growth"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# =============================================================================
# Determinism and shape checks shared across tiers
# =============================================================================


def test_deterministic_across_seeds():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            a = make(TOPIC, seed=seed, difficulty=difficulty)
            b = make(TOPIC, seed=seed, difficulty=difficulty)
            assert a.text == b.text
            assert a.answer_text == b.answer_text
            assert a.distractors == b.distractors


def test_answer_is_a_plain_whole_number():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text.isdigit(), (difficulty, seed, q.answer_text)
            assert isinstance(q.answer, int), (difficulty, seed, q.answer)


def test_distractors_distinct_nonempty_never_equal_answer():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert len(q.distractors) >= 1, (difficulty, seed)
            assert len(set(q.distractors)) == len(q.distractors), (difficulty, seed)
            assert q.answer_text not in q.distractors, (difficulty, seed)


def test_at_least_two_distractors_on_most_draws():
    for difficulty in DIFFICULTIES:
        counts = [len(make(TOPIC, seed=s, difficulty=difficulty).distractors) for s in range(200)]
        at_least_two = sum(1 for c in counts if c >= 2)
        assert at_least_two / 200 >= 0.85, (difficulty, at_least_two)


def test_at_least_20_distinct_questions_per_tier_over_60_seeds():
    for difficulty in DIFFICULTIES:
        texts = {make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}
        assert len(texts) >= 20, (difficulty, len(texts))


def test_money_prompts_never_state_a_rate_above_12_percent():
    """On the hard tier a money prompt states its RATE first and, only in the growth-money template, a second,
    unrelated relative-threshold percentage afterwards (e.g. "...grows by 8% every year... first exceed 325%
    of the amount originally invested?") -- that second number is a ratio, not a rate, and is correctly
    allowed to exceed 12%."""
    for difficulty in DIFFICULTIES:
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if "$" not in q.text:
                continue  # a non-money (population/colony/mg) prompt: no cap
            pcts = [int(p) for p in re.findall(r"(\d+)%", q.text)]
            rates = pcts[:1] if difficulty == "hard" else pcts
            for r in rates:
                assert r <= 12, (difficulty, seed, rates, q.text)


def test_skill_loads_with_expected_shape():
    skill = _skills_dict()[SKILL_ID]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == set(DIFFICULTIES)
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]


# =============================================================================
# Real-grader accept/reject
# =============================================================================


def test_number_grader_rejects_a_deliberately_wrong_value():
    for difficulty in DIFFICULTIES:
        q = _make_from_sources(difficulty, 0)
        wrong = str(int(q.answer_text) + 97)
        assert not grade_number(q, wrong).correct, (difficulty, q.text, wrong)


# =============================================================================
# Independent in-generator check fires on a corrupted answer
# =============================================================================


def test_check_equivalent_fires_on_a_corrupted_route():
    fired = False
    try:
        check_equivalent(Fraction(1), Fraction(2), "deliberately different")
    except VerificationError:
        fired = True
    assert fired


def test_check_fires_on_a_corrupted_condition():
    fired = False
    try:
        check(False, "deliberately false")
    except VerificationError:
        fired = True
    assert fired


def test_generators_own_runaway_guard_fires_on_a_factor_that_never_crosses():
    """Exercises this module's own in-generator check directly (not a bare
    `mathgen.core` call on literals): a growth factor of exactly 1 never
    crosses a threshold above the first value, so `_first_crossing`'s bound
    (`check(k < 300, ...)`) must fire."""
    from mathgen.generators.pv0a_geometric_threshold_build import _first_crossing

    fired = False
    try:
        _first_crossing(100, Fraction(1), 200, growth=True)
    except VerificationError:
        fired = True
    assert fired


# =============================================================================
# Boundary/plausibility property: the key period is never on the
# log-space boundary -- proven RED first against an unconstrained draw.
# =============================================================================

_MONEY_RE = re.compile(r"\$(\d+)")
_PCT_RE = re.compile(r"(\d+)%")


def _parse_easy(text, answer):
    amounts = [int(a) for a in _MONEY_RE.findall(text)]
    V, T = amounts[0], amounts[1]
    r = int(_PCT_RE.search(text).group(1))
    return V, T, 1 + r / 100


def test_boundary_property_is_proven_red_first():
    """Reproduces the log-space margin check with an UNCONSTRAINED threshold
    (no [0.05, 0.95] margin enforced) and shows it fails on some draws --
    proof that the real generator's margin construction is doing real work,
    not merely restating an already-true property."""
    import random

    failures = 0
    for seed in range(300):
        rng = random.Random(f"unconstrained:{seed}")
        k_target = rng.randint(3, 25)
        rate_pct = rng.choice((2, 3, 4, 5, 6, 8, 10, 12))
        factor = 1 + rate_pct / 100
        V = rng.randint(2, 20) * 500
        # An UNCONSTRAINED f, drawn over the full (0, 1) range with no margin
        # excluded -- unlike the real generator's [0.15, 0.85].
        f = rng.uniform(0.0, 1.0)
        exponent = k_target - 1 + f
        exact = V * (factor**exponent)
        T = round(exact)
        if T == V:
            continue
        margin = (math.log(T / V) / math.log(factor)) % 1.0
        if not (0.05 <= margin <= 0.95):
            failures += 1
    assert failures > 0, "the unconstrained draw never landed on the boundary -- test is not RED"


def test_boundary_property_holds_on_the_real_generator():
    for difficulty in DIFFICULTIES:
        for seed in range(150):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if difficulty == "easy":
                V, T, factor = _parse_easy(q.text, q.answer_text)
            elif difficulty == "medium":
                amounts = [int(a) for a in _MONEY_RE.findall(q.text)]
                V, T = amounts[0], amounts[1]
                r = int(_PCT_RE.search(q.text).group(1))
                factor = 1 - r / 100
            elif difficulty == "hard":
                pcts = [int(p) for p in _PCT_RE.findall(q.text)]
                r, pct_threshold = pcts[0], pcts[1]
                growth = "exceed" in q.text
                factor = (1 + r / 100) if growth else (1 - r / 100)
                V, T = 100, pct_threshold
            else:
                continue  # exam's boundary is checked in-generator across two phases;
                # its text does not carry a single (V, T, factor) triple
            margin = (math.log(T / V) / math.log(factor)) % 1.0
            assert 0.05 <= margin <= 0.95, (difficulty, seed, q.text, margin)
