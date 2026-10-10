"""Tests `pv0a_annuity_duration` (ACMGM099): the duration of an annuity modelled by a recurrence relation.

Covers ``mathgen/generators/pv0a_annuity_duration_build.py``'s one topic,
``pv0a_annuity_duration``.
"""

import functools
import random
from fractions import Fraction

import mathgen.generators.pv0a_annuity_duration_build as gen  # noqa: F401 -- registers the topic
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
TOPIC = "pv0a_annuity_duration"
SKILL_ID = "senior-use-a-recurrence-relation-to-model-an-annuity"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# =============================================================================
# Determinism, shape, distractors
# =============================================================================


def test_deterministic_across_seeds():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            a = make(TOPIC, seed=seed, difficulty=difficulty)
            b = make(TOPIC, seed=seed, difficulty=difficulty)
            assert a.text == b.text
            assert a.answer_text == b.answer_text
            assert a.distractors == b.distractors


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


def test_answer_is_always_a_plain_whole_number_of_withdrawals():
    for difficulty in DIFFICULTIES:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text.isdigit(), (difficulty, seed, q.answer_text)
            assert isinstance(q.answer, int)
            for d in q.distractors:
                assert d.lstrip("-").isdigit() or d.isdigit(), (difficulty, seed, d)


def test_every_prompt_states_the_answer_form():
    for difficulty in DIFFICULTIES:
        for seed in range(20):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "whole number of withdrawals" in q.text, (difficulty, seed, q.text)


def test_skill_loads_with_expected_shape():
    skill = _skills_dict()[SKILL_ID]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == set(DIFFICULTIES)
    exam_src = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


# =============================================================================
# Real-grader accept/reject
# =============================================================================


def test_number_grader_rejects_a_deliberately_wrong_answer():
    for difficulty in DIFFICULTIES:
        for seed in range(10):
            q = _make_from_sources(difficulty, seed)
            wrong = str(int(q.answer_text) + 97)
            assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, wrong)


# =============================================================================
# The in-generator check fires on a corrupted answer
# =============================================================================


def test_check_equivalent_fires_on_a_corrupted_route():
    fired = False
    try:
        check_equivalent(5, 6, "deliberately different")
    except VerificationError:
        fired = True
    assert fired


# ============================================================================= The construction
# margin -- proven RED first against an unconstrained draw.
# =============================================================================


def test_construction_margins_hold_and_readings_agree():
    """For a broad sample of (r, d, n) draws that the real solver accepts,
    re-run the recurrence from scratch on the returned opening balance and
    confirm: reading (a)/(c)'s failure margin (`A(n)(1+r) <= d - m`), reading
    (b)'s success margin (`A(n-1) >= d + m`), and that reading (b)'s own
    independent loop produces the identical count."""
    from mathgen.generators.pv0a_annuity_duration_build import (
        _iterate_annuity,
        _iterate_annuity_reading_b,
        _margin,
        _solve_a0_for_target,
    )

    checked = 0
    rng = random.Random(12345)
    for _ in range(200):
        hundredths = rng.randint(20, 60)
        r = Fraction(hundredths, 10000)
        d = Fraction(rng.randrange(400, 6001, 50))
        n = rng.randint(6, 40)
        solved = _solve_a0_for_target(r, d, n)
        if solved is None:
            continue
        a0, n_actual, _ = solved
        assert n_actual == n
        n_chk, b_after, prev_b, fail_grown = _iterate_annuity(a0, r, d)
        m = _margin(d)
        assert n_chk == n
        assert prev_b is not None
        assert prev_b - d >= m, (a0, r, d, n, "reading (b) success margin violated")
        assert d - fail_grown >= m, (a0, r, d, n, "reading (a)/(c) failure margin violated")
        n_b, _ = _iterate_annuity_reading_b(a0, r, d)
        assert n_b == n, (a0, r, d, n, n_b, "reading (a) and reading (b) disagree")
        checked += 1
    assert checked >= 100, checked


def test_unconstrained_draw_can_violate_the_margin_a_real_solve_never_does():
    """The RED-first control: solving for A(0) at the EDGE of the safe window
    (`frac = 1`, i.e. the residual placed exactly at `max_residual`, the
    boundary the real solver always stays strictly inside of) can leave the
    rounded-to-the-dollar balance failing the margin -- demonstrating the
    margin is a real, checkable property and not automatically satisfied by
    any solve. The real solver (tested above) never returns such a case."""
    from mathgen.generators.pv0a_annuity_duration_build import _iterate_annuity, _margin
    from mathgen.core import VerificationError

    def _solve_at_exact_boundary(r, d, n):
        m = _margin(d)
        growth = (1 + r) ** n
        max_residual = (d - m) / (1 + r)
        a0 = (max_residual + d * (growth - 1) / r) / growth
        from mathgen.core import round_half_up

        return round_half_up(a0, 0)

    found_a_violation = False
    for hundredths in range(20, 61):
        r = Fraction(hundredths, 10000)
        for d_int in range(400, 6001, 200):
            d = Fraction(d_int)
            for n in (6, 15, 30, 40):
                a0 = _solve_at_exact_boundary(r, d, n)
                if a0 * r >= d or a0 <= 0:
                    continue
                try:
                    n_chk, b_after, prev_b, fail_grown = _iterate_annuity(a0, r, d)
                except VerificationError:
                    continue
                if n_chk != n or prev_b is None:
                    continue
                m = _margin(d)
                if d - fail_grown < m:
                    found_a_violation = True
                    break
            if found_a_violation:
                break
        if found_a_violation:
            break
    assert found_a_violation, "expected the boundary-placed draw to violate the margin at least once"


# Every exam template now states the change as happening strictly AFTER the kth withdrawal
# throughout, and never uses either phrase.
# =============================================================================

import re

_AMBIGUOUS_PHRASING_RE = re.compile(r"onward|From the")


def test_exam_prompts_never_use_the_ambiguous_onward_wording():
    """Red-first proof: the pre-fix exam templates matched this pattern via "From the {k}th withdrawal onward" /
    "From the {k}th withdrawal onward the rate changes"; every current exam prompt over 200 seeds must be
    clear of it."""
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        m = _AMBIGUOUS_PHRASING_RE.search(q.text)
        assert m is None, (seed, m.group(0), q.text)
