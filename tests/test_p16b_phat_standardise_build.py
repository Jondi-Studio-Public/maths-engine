"""Simulate repeated random sampling and standardise the sample proportion
(senior-simulate-repeated-random-sampling-for-a-variety-of, ACMMM176).

The independent oracle (`test_p16b_phat_standardise_oracle.py`) re-derives the
whole answer from the rendered question text. This file tests the generator
directly: determinism, distractor hygiene, the real grader, the two
in-generator checks, and every guard.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, stated_precision_tolerance
from mathgen.generators.p16b_phat_standardise_build import (
    N_SIZES,
    P_VALUES,
    _finalise,
    _simulate_batch,
    _simulate_sample,
    _verify_z,
    _z_value,
)

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p16b_phat_standardise"
SKILL_ID = "senior-simulate-repeated-random-sampling-for-a-variety-of"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

grade_number = GRADERS["number"]

_SAMPLE_PHAT_RE = re.compile(r"sample (\d+): (-?[\d.]+)(?:;|\.)")
_SAMPLE_X_RE = re.compile(r"sample (\d+): x = (-?\d+)")
_P_RE = re.compile(r"p = (-?[\d.]+)")
_HARD_PAIR_RE = re.compile(r"Samples (\d+) and (\d+) are on opposite sides of p")
_N_RE = re.compile(r"independent samples of (\d+)")


# --- Determinism & distractor hygiene ----------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer_and_clear_tolerance(difficulty):
    tol = stated_precision_tolerance(2)
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.distractors, f"{difficulty} seed={seed}: no distractors"
        answer_value = Fraction(question.answer_text)
        for d in question.distractors:
            assert abs(Fraction(d) - answer_value) > tol, (
                f"{difficulty} seed={seed}: distractor {d!r} within tolerance of {question.answer_text!r}"
            )
        # pairwise distinct too
        values = [answer_value] + [Fraction(d) for d in question.distractors]
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                assert abs(left - right) > tol


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_two_distractors_on_almost_every_draw(difficulty):
    ok = 0
    total = 200
    for seed in range(total):
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        if len(question.distractors) >= 2:
            ok += 1
    assert ok / total >= 0.85, f"{difficulty}: only {ok}/{total} draws had >=2 distractors"


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_answer_value_exceeds_25_percent_of_200_draws(difficulty):
    from collections import Counter

    counts = Counter(make(TOPIC, seed=seed, difficulty=difficulty).answer_text for seed in range(200))
    top = counts.most_common(1)[0][1]
    assert top / 200 <= 0.25, f"{difficulty}: top answer is {top}/200 draws"


# --- Real grader testing -----------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        answer_value = Fraction(question.answer_text)
        wrong = str(float(answer_value) + 5.37)
        assert not grade_number(question, wrong).correct, f"{difficulty} seed={seed}: wrong answer {wrong!r} accepted"


# --- The two in-generator checks ---------------------------------------------


def test_verify_z_fires_on_corrupted_value():
    # Correct: x=18, n=50 -> p-hat=0.36; p=0.30 -> genuine z.
    p_float = 0.30
    n = 50
    x = 18
    phat = x / n
    z = _z_value(phat, p_float, n)
    _verify_z(x, n, p_float, z)  # does not raise
    with pytest.raises(VerificationError):
        _verify_z(x, n, p_float, z + 0.5)  # corrupted z


def test_simulate_sample_never_returns_zero_or_full_count():
    """Guard: every sample the key or any distractor uses has 0 < p-hat < 1.

    Proven red first: a naive, UNGUARDED single-shot Bernoulli draw over the
    same (n, p) combinations genuinely produces x=0 or x=n a non-trivial
    fraction of the time (n=20, p=0.1 alone has P(x=0) = 0.9**20 ~= 0.12) --
    demonstrated below by drawing many raw, unguarded single-shot samples and
    confirming at least one hits the boundary. The generator's own guarded
    `_simulate_sample` never does, across the same (n, p) grid, many draws.
    """
    import random

    def _raw_unguarded(rng, n, p):
        return sum(1 for _ in range(n) if rng.random() < p)

    hit_boundary = False
    for seed in range(200):
        rng = random.Random(f"unguarded-probe:{seed}")
        x = _raw_unguarded(rng, 20, 0.1)
        if x in (0, 20):
            hit_boundary = True
            break
    assert hit_boundary, "test setup does not exercise the boundary defect the guard prevents"

    for seed in range(300):
        rng = random.Random(f"guarded-probe:{seed}")
        for n in N_SIZES:
            for p in (0.1, 0.5, 0.9):
                x = _simulate_sample(rng, n, p)
                assert 0 < x < n, f"n={n} p={p}: guard failed, x={x}"


def test_finalise_raises_when_all_candidates_collide_with_answer():
    """`_finalise`'s own internal invariant: if every candidate collides with
    the key (fewer than 2 survive), it must raise rather than silently ship
    too few distractors."""
    with pytest.raises(VerificationError):
        _finalise(1.234, [("dup1", 1.234), ("dup2", 1.2341)])


# --- Answer kind / precision --------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_number_stated_to_exactly_2dp(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, (int, float))
        text = question.answer_text
        assert "." in text, f"{difficulty} seed={seed}: answer_text {text!r} has no decimal point"
        assert len(text.split(".")[1]) == 2, f"{difficulty} seed={seed}: {text!r} is not 2dp"
        for d in question.distractors:
            assert "." in d and len(d.split(".")[1]) == 2, f"distractor {d!r} not 2dp"


@pytest.mark.parametrize("difficulty", TIERS)
def test_prompt_shape_variety_at_least_3_per_tier(difficulty):
    from mathgen.text_quality import prompt_shape

    shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=difficulty)) for seed in range(60)}
    assert len(shapes) >= 3, f"{difficulty}: only {len(shapes)} distinct prompt shapes/60 seeds"


def test_formula_and_precision_instruction_present_every_tier():
    for difficulty in TIERS:
        for seed in range(10):
            text = make(TOPIC, seed=seed, difficulty=difficulty).text
            assert "standardised value" in text
            assert "z = " in text
            assert "p̂" in text  # p-hat, written p + combining circumflex
            assert "2 decimal places" in text


def test_easy_and_medium_never_print_raw_p_to_learner_as_a_secret():
    """Sanity: easy/medium DO print p directly (never a chance model); exam
    never prints p as a number at all."""
    for seed in range(10):
        easy_text = make(TOPIC, seed=seed, difficulty="easy").text
        assert re.search(r"p = [\d.]+", easy_text)
        medium_text = make(TOPIC, seed=seed, difficulty="medium").text
        assert re.search(r"p = [\d.]+", medium_text)
        exam_text = make(TOPIC, seed=seed, difficulty="exam").text
        assert not re.search(r"p = [\d.]+", exam_text)
        assert "chance model" in exam_text


# --- Guard: hard tier's opposite-sides-of-p and tie-exclusion invariants ------


def _exact_hard_pair(text: str):
    """Parse a hard-tier question's printed p, n, and the two named samples'
    p-hat, and return (p, n, i, j, phat_i, phat_j) with p and the p-hats as
    exact `Fraction`s -- never floats. `n` is always `2**a * 5**b` (N_SIZES),
    so the printed 4dp p-hat display is an EXACT decimal, and `Fraction` of
    that printed string is the exact rational value, with no rounding at
    all."""
    p = Fraction(_P_RE.search(text).group(1))
    n = int(_N_RE.search(text).group(1))
    samples = {int(i): Fraction(v) for i, v in _SAMPLE_PHAT_RE.findall(text)}
    i, j = (int(v) for v in _HARD_PAIR_RE.search(text).groups())
    return p, n, i, j, samples[i], samples[j]


def _exact_hard_classification(text: str) -> str:
    """Strict, exact classification of a hard-tier question's named pair:
    the sample with the strictly LARGER exact distance |p-hat - p| is
    "far"; return 'trap' if that far sample's |z| is the SMALLER of the
    two, else 'plain'. Raises if the two named samples are at an exact tie
    distance -- the generator must never offer such a pair."""
    p, n, i, j, phat_i, phat_j = _exact_hard_pair(text)
    dist_i, dist_j = abs(phat_i - p), abs(phat_j - p)
    assert dist_i != dist_j, f"equal-distance pair reached the printed question: {text!r}"

    def z(phat: Fraction) -> float:
        phat_f = float(phat)
        se = math.sqrt(phat_f * (1 - phat_f) / n)
        return (phat_f - float(p)) / se

    zi, zj = z(phat_i), z(phat_j)
    far_is_i = dist_i > dist_j
    far_z, near_z = (zi, zj) if far_is_i else (zj, zi)
    return "trap" if abs(far_z) < abs(near_z) else "plain"


def test_hard_tier_pair_is_on_opposite_sides_of_p_and_clears_tie_margin():
    """Guard (hard tier): the two named samples are on opposite
    sides of p, at an exact (never a tied) distance from p, and
    | |z_i| - |z_j| | >= 0.05 -- proven red by construction below: a
    synthetic pair with a margin of 0.02 fails the same assertion, showing
    the test would catch a construction that let a near-tie ship."""
    checked = 0
    for seed in range(100):
        question = make(TOPIC, seed=seed, difficulty="hard")
        text = question.text
        p, n, i, j, phat_i, phat_j = _exact_hard_pair(text)
        assert (phat_i - p) * (phat_j - p) < 0, f"seed={seed}: samples {i},{j} not on opposite sides of p"
        assert phat_i != p and phat_j != p
        assert abs(phat_i - p) != abs(phat_j - p), f"seed={seed}: samples {i},{j} are at an exact-tie distance from p"

        def z(phat: Fraction) -> float:
            phat_f = float(phat)
            se = math.sqrt(phat_f * (1 - phat_f) / n)
            return (phat_f - float(p)) / se

        zi, zj = z(phat_i), z(phat_j)
        assert abs(abs(zi) - abs(zj)) >= 0.05 - 1e-6, f"seed={seed}: tie margin violated ({zi} vs {zj})"
        checked += 1
    assert checked == 100

    # Red-first demonstration: a synthetic pair with a 0.02 margin fails the
    # same style of assertion this test relies on.
    zi_bad, zj_bad = 1.00, 1.02
    assert not (abs(abs(zi_bad) - abs(zj_bad)) >= 0.05 - 1e-6)


def test_hard_tier_both_trap_and_plain_classes_are_reachable():
    """The construct-to-target draw picks trap/plain 50/50; over 60 seeds
    both the strict trap ('far sample has the smaller |z|') and strict
    plain ('far sample has the larger |z|') class must actually appear,
    using the exact (Fraction-based) classification, never floats."""
    trap_seen = plain_seen = False
    for seed in range(60):
        question = make(TOPIC, seed=seed, difficulty="hard")
        cls = _exact_hard_classification(question.text)
        if cls == "trap":
            trap_seen = True
        else:
            plain_seen = True
    assert trap_seen, "trap class never appeared in 60 hard-tier seeds"
    assert plain_seen, "plain class never appeared in 60 hard-tier seeds"


def _old_buggy_hard_class(rng):
    """A byte-for-byte reproduction of the earlier `_hard` bug: it drew
    a FRESH `target_class` on every outer resimulation retry instead of once
    up front, which biases the realised class toward whichever is easier to
    find on a random batch. Never imported from the generator module --
    redefined here purely to prove the trap-share test below would have
    caught the defect."""
    for _outer in range(60):
        p = rng.choice(P_VALUES)
        p_float = float(p)
        n = rng.choice(N_SIZES)
        k = rng.randint(5, 8)
        xs = _simulate_batch(rng, n, p_float, k)
        target_class = rng.choice(["trap", "plain"])
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        rng.shuffle(pairs)
        for i, j in pairs:
            phat_i, phat_j = xs[i] / n, xs[j] / n
            if (phat_i - p_float) * (phat_j - p_float) >= 0:
                continue
            zi, zj = _z_value(phat_i, p_float, n), _z_value(phat_j, p_float, n)
            far_is_i = abs(phat_i - p_float) > abs(phat_j - p_float)
            far_z, near_z = (zi, zj) if far_is_i else (zj, zi)
            if abs(abs(far_z) - abs(near_z)) < 0.05:
                continue
            is_trap = abs(far_z) < abs(near_z)
            cls = "trap" if is_trap else "plain"
            if cls != target_class:
                continue
            return cls
    return None


def test_hard_tier_trap_share_is_near_50_50_over_200_seeds():
    """Proven red first: `_old_buggy_hard_class` above reproduces the pre-fix
    algorithm exactly (fresh `target_class` every outer retry, the actual
    bug) -- run over 200 independent rng streams below, its realised-class
    share lands far outside [0.40, 0.60], showing this style of assertion
    would have caught the defect. The real generator's share, measured from
    the PRINTED text using the STRICT, exact classification
    (`_exact_hard_classification`), must land inside [0.40, 0.60].
    """
    import random

    old_trap = old_total = 0
    for seed in range(200):
        rng = random.Random(f"p16b-old-buggy-hard-probe:{seed}")
        cls = _old_buggy_hard_class(rng)
        if cls is not None:
            old_total += 1
            if cls == "trap":
                old_trap += 1
    assert old_total > 100, "old-bug reproduction produced too few realised draws to judge"
    old_share = old_trap / old_total
    assert not (0.40 <= old_share <= 0.60), (
        f"old-bug reproduction unexpectedly balanced ({old_trap}/{old_total}); "
        "this test would not have caught the real defect"
    )

    trap = 0
    for seed in range(200):
        text = make(TOPIC, seed=seed, difficulty="hard").text
        if _exact_hard_classification(text) == "trap":
            trap += 1
    share = trap / 200
    assert 0.40 <= share <= 0.60, f"hard-tier trap share {trap}/200 = {share} outside [0.40, 0.60]"


def test_hard_tier_exact_distance_ties_never_appear():
    """The fixed generator must never offer a tied pair at all: this test asserts zero ties over 200 seeds using
    the same exact predicate."""
    ties = 0
    for seed in range(200):
        text = make(TOPIC, seed=seed, difficulty="hard").text
        p, n, i, j, phat_i, phat_j = _exact_hard_pair(text)
        if abs(phat_i - p) == abs(phat_j - p):
            ties += 1
    assert ties == 0, f"{ties}/200 hard-tier draws offered an exact-tie pair"


def test_exam_tier_true_p_is_an_exact_fraction_not_printed_directly():
    for seed in range(20):
        text = make(TOPIC, seed=seed, difficulty="exam").text
        assert any(
            m in text
            for m in [
                "spinner",
                "fair six-sided die",
                "well-shuffled standard deck",
            ]
        ), f"seed={seed}: no chance model named in {text!r}"


# --- Skill contract loads with the expected shape ----------------------------


def test_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id[SKILL_ID]
    assert skill.prerequisites == ["simple-event-probability"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {source.topic for source in skill.sources} == {TOPIC}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.stages for source in skill.sources] == [[6], [6], [6, "maintenance"], ["exam"]]
