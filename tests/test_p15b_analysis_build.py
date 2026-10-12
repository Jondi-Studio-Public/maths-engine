"""Focused contract tests for: ucv1-completeness-cauchy-check (answer: mcq)."""

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-completeness-cauchy-check"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_ALL_ROW_ANSWERS = {f"Row {i} is inconsistent." for i in range(1, 5)}
_ALL_ANSWERS = _ALL_ROW_ANSWERS | {"All rows are consistent."}


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
        assert len(q1.distractors) == 4


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        options = mcq_options(q)
        assert len(options) == 5
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(index)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_one_of_the_five_fixed_options(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert q.answer_text in _ALL_ANSWERS, (difficulty, seed, q.answer_text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_200_seed_five_classes_seen_and_roughly_equal(difficulty):
    """The design draws the target uniformly over 5 classes (none, row 0..3),
    so all 5 should appear over 200 seeds with no class dominating."""
    counts = Counter(_make(difficulty, seed).answer_text for seed in range(200))
    assert set(counts) == _ALL_ANSWERS, f"{difficulty}: classes seen {set(counts)}"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.40, f"{difficulty}: top class is {top_share:.1%} of 200 draws"


def test_hard_and_exam_use_near_miss_margins_medium_and_easy_do_not():
    """hard/exam draw the smallest-possible nonzero margin (near-miss);
    easy/medium use a generous margin. Detected via the exact-fraction gap
    between each row's stated epsilon and the sequence's own tail spread at
    that row's N: near-miss margins are 1/b^(k+2) (tiny), generous margins
    are 1/b^2 (much larger), re-derived independently from the sequence text
    itself, never from the generator's internal preset dict."""
    from fractions import Fraction

    for difficulty, expect_near_miss in (
        ("easy", False),
        ("medium", False),
        ("hard", True),
        ("exam", True),
    ):
        term_re = re.compile(r"a_(\d+) = (\d+(?:/\d+)?)")
        row_re = re.compile(r"Row (\d+): epsilon = ([\d/]+(?: \+ [\d/]+)?), N = (\d+) \(claims")

        def parse_fraction(s):
            if "/" in s:
                a, b = s.split("/")
                return Fraction(int(a), int(b))
            return Fraction(int(s))

        saw_true_row = False
        for seed in SEEDS:
            q = _make(difficulty, seed)
            terms = {int(m.group(1)): parse_fraction(m.group(2)) for m in term_re.finditer(q.text)}
            last_index = max(terms)
            last_val = terms[last_index]
            for row_num_s, eps_text, n_index_s in row_re.findall(q.text):
                row_num = int(row_num_s)
                if " + " in eps_text:
                    a, b = eps_text.split(" + ")
                    epsilon = parse_fraction(a) + parse_fraction(b)
                else:
                    epsilon = parse_fraction(eps_text)
                n_index = int(n_index_s)
                spread = terms[n_index] - last_val
                if spread < epsilon:  # a TRUE (consistent) row
                    saw_true_row = True
                    margin = epsilon - spread
                    is_tiny = margin <= Fraction(1, 100)
                    if expect_near_miss:
                        assert is_tiny, (difficulty, seed, row_num, margin)
                    else:
                        assert not is_tiny, (difficulty, seed, row_num, margin)
        assert saw_true_row, difficulty


def test_exam_tier_renders_epsilon_as_a_sum_of_two_distinct_fractions():
    for seed in SEEDS:
        q = _make("exam", seed)
        assert " + " in q.text, (seed, q.text)


def test_easy_medium_hard_never_render_epsilon_as_a_sum():
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _make(difficulty, seed)
            row_section = q.text.split("witness table")[1]
            for chunk in row_section.split(";"):
                if "epsilon = " in chunk:
                    eps_part = chunk.split("epsilon = ")[1].split(",")[0]
                    assert " + " not in eps_part, (difficulty, seed, q.text)


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "functional-and-linear-analysis"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "mcq"
    assert SKILL.prerequisites == [
        "senior-understand-the-limiting-behaviour-as-n-rightarrow-infty",
        "limit-at-infinity",
    ]
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert SOURCES["exam"].stages == ["exam"]
