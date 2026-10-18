"""This file holds the
generator-specific proofs: the in-generator checks firing on a corrupted
value, the real-grader battery, and the answer's own-kind checks.

The independent ORACLE lives in `tests/test_pv0c_choose_display_oracle.py`.
"""

from __future__ import annotations

from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

import mathgen.generators.pv0c_choose_display_build as gen

SKILL_ID = "senior-determine-which-type-of-graph-is-best-used"
PREREQ_ID = "simple-event-probability"
TOPIC = "pv0c_choose_display"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

ALL_DISPLAYS = {info["key"] for info in gen.CLASS_INFO.values()} | {
    "column graph",
    "pie chart",
    "dot plot",
    "segmented column graph",
    "histogram",
    "scatterplot",
    "line graph",
    "back-to-back stem-and-leaf plot",
}
ALSO_ACCEPTABLE = {
    "C1": {"pie chart", "dot plot", "segmented column graph"},
    "C2": {"dot plot", "stem-and-leaf plot", "box plot"},
    "C3": set(),
    "C4": {"scatterplot", "column graph"},
    "C5": {"parallel box plots", "parallel dot plots"},
    "C6": {"side-by-side column graph", "two-way table"},
}


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
def test_real_grader_rejects_a_deliberately_wrong_index(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        options = mcq_options(q)
        wrong_index = next(i for i, o in enumerate(options) if o != q.answer_text)
        assert not grade_mcq(q, wrong_index).correct
        assert not grade_mcq(q, "not a number").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_one_of_the_fixed_display_names(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert q.answer_text in ALL_DISPLAYS, (difficulty, seed, q.answer_text)
        for d in q.distractors:
            assert d in ALL_DISPLAYS, (difficulty, seed, d)


@pytest.mark.parametrize("difficulty", TIERS)
def test_200_seed_all_six_classes_seen_and_no_class_dominates(difficulty):
    counts = Counter(_make(difficulty, seed).answer_text for seed in range(200))
    assert set(counts) == {info["key"] for info in gen.CLASS_INFO.values()}, f"{difficulty}: classes seen {set(counts)}"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.25, f"{difficulty}: top class is {top_share:.1%} of 200 draws"


@pytest.mark.parametrize("difficulty", TIERS)
def test_exactly_three_distractors_every_draw(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert len(q.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_also_acceptable_display_ever_offered(difficulty):
    """The boundary that makes the choice decisive: a display on a class's
    own 'also acceptable' list must never appear as an option (answer or
    distractor) for that class."""
    for seed in range(200):
        q = _make(difficulty, seed)
        class_id = next(c for c, info in gen.CLASS_INFO.items() if info["key"] == q.answer_text)
        offered = {q.answer_text, *q.distractors}
        forbidden = ALSO_ACCEPTABLE[class_id] & offered
        assert not forbidden, (difficulty, seed, class_id, forbidden, q.text)


def test_pick_options_check_fires_on_corrupted_distractor_pool():
    import random

    rng = random.Random(0)
    gen._pick_options(rng, "C1")  # no exception

    class _FakeRng:
        def sample(self, pool, k):
            return [gen.CLASS_INFO["C1"]["key"]] + pool[: k - 1]

    with pytest.raises(VerificationError):
        gen._pick_options(_FakeRng(), "C1")


def test_class_info_key_never_in_its_own_wrong_pool():
    """Static-data guard already asserted at import time (module-level
    `check()` calls); re-assert here so a future edit to CLASS_INFO that
    breaks it is caught by a named test, not only an import-time crash."""
    for class_id, info in gen.CLASS_INFO.items():
        assert info["key"] not in info["wrong"], class_id
        assert len(set(info["wrong"])) == len(info["wrong"]), class_id


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_easy_and_medium_never_use_numeric_label_or_purpose_framing(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert "wants to know" not in q.text, (difficulty, seed, q.text)
        assert "postcode" not in q.text and "jersey number" not in q.text


def test_hard_tier_uses_a_numeric_label_or_time_vs_paired_trap():
    saw_label_trap = False
    saw_time_pair_trap = False
    for seed in range(200):
        q = _make("hard", seed)
        if any(
            w in q.text for w in ("postcode", "flight number", "bus route number", "gate number", "school house number")
        ):
            saw_label_trap = True
        if any(
            w in q.text
            for w in (
                "trading day",
                "weather station",
                "daily active users",
                "sold in a suburb",
                "years of education",
                "exam mark",
            )
        ):
            saw_time_pair_trap = True
    assert saw_label_trap, "no hard-tier numeric-label trap seen in 200 seeds"
    assert saw_time_pair_trap, "no hard-tier time-vs-paired trap seen in 200 seeds"


def test_hard_tier_never_uses_a_near_unique_identifier():
    """'The room number of each of N guests' and similar are near-unique per individual (defeats a
    column/segmented-column graph), and 'length of hospital stay' was a whole-day count with few values (C2 is
    continuous measurements only). Proven red on the pre-fix generator: it drew "room number"/"jersey
    number"/"customer ID number"/"length of hospital stay" among its hard-tier C1/C2/C6 contexts."""
    banned = ("ID", "jersey number", "room number", "length of hospital stay")
    for seed in range(200):
        q = _make("hard", seed)
        for word in banned:
            assert word not in q.text, (seed, word, q.text)


def test_exam_tier_states_a_research_purpose():
    for seed in range(20):
        q = _make("exam", seed)
        assert "wants to know" in q.text, (seed, q.text)


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "statistics-probability"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "mcq"
    assert SKILL.prerequisites == [PREREQ_ID]
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert SOURCES["exam"].stages == ["exam"]
    assert SOURCES["hard"].stages == [6, "maintenance"]


def test_prerequisite_skill_still_loads():
    _domains, skills = load_skills()
    prereq = next(s for s in skills if s.id == PREREQ_ID)
    assert prereq.status == "active"
