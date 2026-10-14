"""This file holds the
generator-specific proofs: the in-generator checks firing on a corrupted
value, the real-grader battery, and the answer's own-kind checks.

The independent ORACLE lives in `tests/test_p18a_line_graph_claims_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

import mathgen.generators.p18a_line_graph_claims_build as gen

SKILL_ID = "f10-line-graph-change-over-time"
PREREQS = ["calculate-mean", "calculate-median", "calculate-mode", "calculate-range"]
TOPIC = "p18a_line_graph_claims"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
SWEEP_SEEDS = range(60)


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources(SKILL_ID)


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        options = mcq_options(q)
        assert len(options) == 4
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, correct_index).correct
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, index).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_index_and_junk(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        options = mcq_options(q)
        wrong_index = next(i for i, o in enumerate(options) if o != q.answer_text)
        assert not grade_mcq(q, wrong_index).correct
        assert not grade_mcq(q, "not a number").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_option_shape_is_supported_or_not_supported_colon_fact(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for option in (q.answer_text, *q.distractors):
            assert option == "Supported" or option.startswith("Not supported: "), (difficulty, seed, option)
        not_supported = [o for o in (q.answer_text, *q.distractors) if o != "Supported"]
        assert len(not_supported) == 3
        assert "Supported" in (q.answer_text, *q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_supported_share_is_25_to_35_percent_of_200_draws(difficulty):
    counts = Counter(_make(difficulty, seed).answer_text == "Supported" for seed in range(200))
    share = counts[True] / 200
    assert 0.25 <= share <= 0.35, f"{difficulty}: Supported share {share:.1%}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_exactly_three_distractors_every_draw(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert len(q.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_double_space_or_immediate_word_repeat(difficulty):
    import re

    for seed in range(100):
        q = _make(difficulty, seed)
        assert "  " not in q.text, (difficulty, seed, q.text)
        assert not re.search(r"\b(\w+)\s+\1\b", q.text, re.I), (difficulty, seed, q.text)


def test_monotonic_claim_check_fires_on_corrupted_break_count(monkeypatch):
    """`_monotonic_claim`'s own independent check recomputes the number of
    consecutive pairs whose direction disagrees with the claimed one and
    fails unless it is exactly 1 (the "single breaking pair" invariant the
    hard/exam-monotonic tier depends on). Monkeypatch `_monotonic_series` to
    inject a SECOND mismatch beyond the intended `break_idx` and confirm the
    real generator function raises `VerificationError` -- proving the check
    is load-bearing, not merely present."""
    import random

    real_series = gen._monotonic_series

    def _corrupted_series(rng, q, n, want_supported):
        values, diffs, direction, break_idx = real_series(rng, q, n, False)
        for i in range(len(diffs)):
            if i != break_idx and gen._diff_word(diffs[i]) == direction:
                diffs[i] = -diffs[i] if diffs[i] != 0 else 1
                values[i + 1] = values[i] + diffs[i]
                break
        return values, diffs, direction, break_idx

    monkeypatch.setattr(gen, "_monotonic_series", _corrupted_series)
    labels = [str(2000 + i) for i in range(5)]
    hit = False
    for seed in range(80):
        rng = random.Random(seed)
        try:
            gen._monotonic_claim(rng, gen._QUANTITIES[0], labels, "year")
        except VerificationError:
            hit = True
            break
    assert hit, "corrupted break count never raised VerificationError over 80 seeds"


@pytest.mark.parametrize("difficulty", TIERS)
def test_data_sequence_is_drawn_as_a_line_graph_not_listed(difficulty):
    for seed in range(20):
        q = _make(difficulty, seed)
        fig = q.figure
        assert fig["kind"] == "cartesian" and fig["equal_scale"] is False
        assert len(fig["points"]) >= 4 and len(fig["segments"]) == len(fig["points"]) - 1
        # the points are joined in time order and every value sits on a gridline
        xs = [p["at"][0] for p in fig["points"]]
        assert xs == sorted(xs)
        for p in fig["points"]:
            assert p["at"][1] % fig["y_grid"] == 0
        assert "(" not in q.text.split("A claim is made")[0], q.text


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "statistics-probability"
    assert SKILL.status == "active"
    assert SKILL.needs_figure is True and SKILL.capability_gates == ["figure_stimulus"]
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "mcq"
    assert SKILL.prerequisites == PREREQS
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert SOURCES["exam"].stages == ["exam"]
    assert SOURCES["hard"].stages == [5, 6, "maintenance"]
    assert SOURCES["easy"].stages == [0, 1, 2]
    assert SOURCES["medium"].stages == [3, 4]


@pytest.mark.parametrize("prereq", PREREQS)
def test_prerequisite_skills_still_load(prereq):
    _domains, skills = load_skills()
    match = next(s for s in skills if s.id == prereq)
    assert match.status == "active"


def test_hard_tier_break_option_names_a_specific_pair():
    for seed in range(60):
        q = _make("hard", seed)
        not_supported = [o for o in (q.answer_text, *q.distractors) if o != "Supported"]
        for opt in not_supported:
            assert " between " in opt, (seed, opt)


def test_exam_tier_reaches_both_variants():
    saw_overall = False
    saw_monotonic = False
    for seed in range(120):
        q = _make("exam", seed)
        if "Overall, from" in q.text:
            saw_overall = True
        if "every " in q.text or "stayed the same throughout" in q.text:
            saw_monotonic = True
    assert saw_overall, "exam tier never drew the overall-change variant in 120 seeds"
    assert saw_monotonic, "exam tier never drew the monotonic-over-longer-sequence variant in 120 seeds"


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_tier_draws_the_graph_in_time_order(difficulty):
    """A line graph is drawn in time order; the lower tiers use 4-5 points and the exam tier a longer graph."""
    for seed in range(60):
        q = _make(difficulty, seed)
        xs = [p["at"][0] for p in q.figure["points"]]
        assert xs == sorted(xs) and len(set(xs)) == len(xs), (difficulty, seed)
        assert len(xs) >= (7 if difficulty == "exam" else 4), (difficulty, seed, len(xs))
        assert len(xs) <= (8 if difficulty == "exam" else 5), (difficulty, seed, len(xs))
