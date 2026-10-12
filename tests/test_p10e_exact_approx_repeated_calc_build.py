"""Focused tests for f10-exact-approximate-repeated-calculation."""

import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, round_half_up, stated_precision_tolerance
from mathgen.generators.p10e_exact_approx_repeated_calc_build import (
    _verify_approx_final,
    _verify_exact_final,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
COMBO_SEEDS = range(200)
TOPIC = "p10e_exact_approx_repeated_calc"

_TEXT_RE = re.compile(
    r"Start with (\S+)\. .+, in that order\. Work out the result two ways: "
    r"\(1\) using full precision throughout(?: \(using the true, unrounded value of pi "
    r"where it appears\))?, rounding only the FINAL result to (\d) decimal places?; "
    r"\(2\) rounding the running value to \d decimal places? after every single step above\. "
    r"What is the discrepancy \(the positive difference\) between the two final results, "
    r"to \d decimal places?\?"
)


def _v0_and_d(question):
    m = _TEXT_RE.match(question.text)
    assert m, f"unexpected question text shape: {question.text!r}"
    return Fraction(m.group(1)), int(m.group(2))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        _, d = _v0_and_d(q)
        wrong = q.answer + 5.0  # clears any stated precision tolerance by a wide margin
        assert not grade_number(q, f"{wrong:.{d}f}").correct


def test_medium_answer_carries_exactly_the_stated_precision():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        v0, d = _v0_and_d(q)
        assert d == 2
        assert len(q.answer_text.split(".")[1]) == d


def test_easy_tier_d_varies_over_1_and_2():
    """Unlike medium (fixed at 2 dp), the easy tier's stated rounding precision now varies -- part of the fix for
    the tier's own answer almost always being `0.01` (see
    `test_easy_tier_answer_distribution_is_no_longer_a_coin_flip` below). Both `answer_text` and the question
    text itself must show exactly the stated `d` places, whichever `d` was drawn."""
    seen_d = set()
    for seed in COMBO_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        v0, d = _v0_and_d(q)
        seen_d.add(d)
        assert len(q.answer_text.split(".")[1]) == d
    assert seen_d == {1, 2}


def test_answer_and_every_distractor_carry_exactly_the_stated_precision_every_tier():
    """`_stated_precision_string` must never collapse a trailing zero (e.g.
    "0.10" -> "0.1"): `grade_number` derives its comparison tolerance from
    the number of decimal places `answer_text` itself displays, so an
    under-stated literal would grade with a tolerance ten times too loose.
    200 seeds per tier, both `answer_text` and every distractor."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            _, d = _v0_and_d(q)
            assert "." in q.answer_text
            assert len(q.answer_text.split(".")[1]) == d, (
                f"{difficulty} seed {seed}: answer_text {q.answer_text!r} does not show {d} places"
            )
            for distractor in q.distractors:
                assert "." in distractor
                assert len(distractor.split(".")[1]) == d, (
                    f"{difficulty} seed {seed}: distractor {distractor!r} does not show {d} places"
                )


def test_pi_parenthetical_appears_only_when_pi_is_used():
    for seed in range(60):
        for difficulty in ("easy", "medium"):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "pi" not in q.text.lower()
            assert "unrounded value of pi" not in q.text
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert "unrounded value of pi where it appears" in q.text


def test_hard_tier_uses_pi_exactly_once_and_d_varies():
    seen_d = set()
    seen_n_ops = set()
    for seed in COMBO_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        v0, d = _v0_and_d(q)
        seen_d.add(d)
        assert q.text.lower().count("multiply by pi") == 1
        seen_n_ops.add(q.text.count("; then ") + 1)
    assert seen_d == {2, 3}
    assert seen_n_ops == {3, 4}


def test_hard_tier_pi_position_is_not_always_first_or_last():
    positions = set()
    for seed in COMBO_SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        ops_text = q.text.split("in that order")[0]
        clauses = ops_text.split("Start with")[1].split(". ", 1)[1].split("; then ")
        idx = next(i for i, c in enumerate(clauses) if "pi" in c)
        if idx == 0:
            positions.add("first")
        elif idx == len(clauses) - 1:
            positions.add("last")
        else:
            positions.add("middle")
    assert "middle" in positions, "pi never landed anywhere but the ends across 200 hard seeds"


def test_easy_tier_answer_distribution_is_no_longer_a_coin_flip():
    """The easy tier used to answer `0.01` on 91% of 400 seeds -- a learner who typed that without doing either
    calculation scored 9 times in 10. This is the acceptance test, re-derived
    directly rather than merely re-asserting the old (now fixed) narrowness: at 400 seeds, no fewer than 6
    distinct answers and no single answer above ~35% of draws."""
    from collections import Counter

    answers = Counter(make(TOPIC, seed=seed, difficulty="easy").answer_text for seed in range(400))
    assert len(answers) >= 6, f"only {len(answers)} distinct answers over 400 seeds: {answers}"
    top_text, top_n = answers.most_common(1)[0]
    top_share = top_n / 400
    assert top_share <= 0.35, f"{top_text!r} is {top_share:.0%} of 400 draws: {answers}"


def test_medium_tier_answer_distribution_is_no_longer_a_coin_flip():
    """Same acceptance bar as the easy tier's own test above (>= 6 distinct answers, no answer over ~35% share,
    400 seeds), re-derived directly. Measured after the fix: 32 distinct answers, top `0.01` at 30%."""
    from collections import Counter

    answers = Counter(make(TOPIC, seed=seed, difficulty="medium").answer_text for seed in range(400))
    assert len(answers) >= 6, f"only {len(answers)} distinct answers over 400 seeds: {answers}"
    top_text, top_n = answers.most_common(1)[0]
    top_share = top_n / 400
    assert top_share <= 0.35, f"{top_text!r} is {top_share:.0%} of 400 draws: {answers}"


def test_every_distractor_clears_stated_precision_tolerance_every_seed_every_tier():
    for difficulty in ("easy", "medium", "hard"):
        for seed in COMBO_SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            _, d = _v0_and_d(q)
            tol = float(stated_precision_tolerance(d))
            answer_val = float(q.answer_text)
            for distractor in q.distractors:
                gap = abs(answer_val - float(distractor))
                assert gap > tol, f"seed {seed} {difficulty}: distractor {distractor!r} gap {gap} <= {tol}"


def test_hard_tier_always_shows_exactly_three_distractors():
    """`pick_distractors`' universal n=3 cap means only 3 of the 4 legitimate hard-tier misconceptions ever
    render; this test only asserts the count, not which candidate survives -- see the separate margin sweep
    for that."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert len(q.distractors) == 3


def test_every_candidate_clears_at_least_one_grid_unit_from_the_answer():
    """By construction every candidate (answer and every misconception
    value) sits on the exact `10**-d` grid (each has been through
    `round_half_up(..., d)`), so the smallest possible nonzero gap between
    any two of them is exactly one grid unit -- exactly 2x
    `stated_precision_tolerance(d)` -- never a near-miss. Verified directly
    over 200 seeds per tier, both for the rendered distractors and by
    reconstructing the full (pre-`pick_distractors`-cap) candidate list.

    The easy tier's real call shape now varies `d` and passes `easy_wide_mul=True` (see
    `p10e_exact_approx_repeated_calc`) rather than the old fixed `(n_ops=2, d=2, easy_wide_mul=False)` -- both
    `d` values it can draw are covered here directly, not just the pre-fix combination, so this test keeps
    covering what the generator actually builds.

    The medium tier's real call shape now passes `medium_wide_mul=True` (see
    `p10e_exact_approx_repeated_calc`) rather than the old fixed `(n_ops=3, d=2, easy_wide_mul=False,
    medium_wide_mul=False)` -- updated here for the same reason as the easy row above."""
    from mathgen.generators.p10e_exact_approx_repeated_calc_build import _make_seed
    import random

    tiers = [
        ("easy", 2, False, 1, True, False),
        ("easy", 2, False, 2, True, False),
        ("medium", 3, False, 2, False, True),
    ]
    min_ratio = None
    for tier, n_ops, has_pi, d, easy_wide_mul, medium_wide_mul in tiers:
        for seed in range(200):
            rng = random.Random(f"{TOPIC}:{tier}:{d}:{seed}")
            _, _, _, _, discrepancy, candidates = _make_seed(
                rng, n_ops, has_pi, d, easy_wide_mul=easy_wide_mul, medium_wide_mul=medium_wide_mul
            )
            tol = float(stated_precision_tolerance(d))
            for c in candidates:
                ratio = abs(float(c) - float(discrepancy)) / tol
                min_ratio = ratio if min_ratio is None else min(min_ratio, ratio)
    for seed in range(200):
        rng = random.Random(f"{TOPIC}:hard:{seed}")
        n_ops = rng.choice([3, 4])
        d = rng.choice([2, 3])
        _, _, _, _, discrepancy, candidates = _make_seed(rng, n_ops, True, d)
        tol = float(stated_precision_tolerance(d))
        for c in candidates:
            ratio = abs(float(c) - float(discrepancy)) / tol
            min_ratio = min(min_ratio, ratio)
    assert min_ratio >= 2.0 - 1e-9, f"a candidate sits closer than one grid unit: ratio {min_ratio}"


def test_independent_exact_check_fires_on_corrupted_answer():
    _verify_exact_final(Fraction(2), [("mul", Fraction(3, 2))], 2, False, Fraction(3))  # not raise
    with pytest.raises(VerificationError):
        _verify_exact_final(Fraction(2), [("mul", Fraction(3, 2))], 2, False, Fraction(4))


def test_independent_approx_check_fires_on_corrupted_answer():
    v0 = Fraction(2)
    ops = [("mul", Fraction(3, 2))]
    expected = round_half_up(v0 * Fraction(3, 2), 2)
    _verify_approx_final(v0, ops, 2, expected)  # must not raise
    with pytest.raises(VerificationError):
        _verify_approx_final(v0, ops, 2, expected + 1)


def test_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["f10-exact-approximate-repeated-calculation"]

    assert skill.prerequisites == ["f10-measurement-error-interpretation"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources} == {TOPIC, "exd_x_n3_unit_price_rounding"}
    assert [source.stages for source in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in skill.misconceptions} == {
        "reports-approximate-value",
        "reports-exact-value",
        "rounds-pi-early",
        "skips-one-rounding-step",
    }
