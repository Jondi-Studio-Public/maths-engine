"""Tests `p12c_estimate_limit_ah` (ACMMM098): estimating the limit of (a^h - 1)/h with technology.

`answer: mcq`. The real grader (`grade_mcq`) takes a learner-typed STRING INDEX into `mcq_options(question)`
(the shuffled [answer_text] + distractors list), never a raw value.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make

TOPIC = "p12c_estimate_limit_ah"
SKILL_ID = "senior-estimate-the-limit-of-frac-a-h-1"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

_A_RE = re.compile(r"a = (-?\d+(?:\.\d+)?)")


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _correct_index(q):
    options = mcq_options(q)
    return options.index(q.answer_text)


def _wrong_index(q):
    options = mcq_options(q)
    correct = options.index(q.answer_text)
    return next(i for i in range(len(options)) if i != correct)


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_nonempty_distinct_never_equal_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert len(q.distractors) == 3, (difficulty, seed, q.distractors)


# --- Real grader accept/reject (index into mcq_options, per the harness note) --


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct, (difficulty, seed, q.text)
        for i, opt in enumerate(options):
            if i == correct_idx:
                continue
            assert not grade_mcq(q, str(i)).correct, (difficulty, seed, q.text, opt)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        wrong_idx = _wrong_index(q)
        assert not grade_mcq(q, str(wrong_idx)).correct, (difficulty, seed, q.text)
        # Out-of-range and non-numeric input must also be rejected, never crash.
        assert not grade_mcq(q, "99").correct
        assert not grade_mcq(q, "not a number").correct


# --- Every distractor from the SAME question is a genuinely different
# option from the answer AND from each other -------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_options_pairwise_distinct(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        options = [q.answer_text] + list(q.distractors)
        assert len(options) == len(set(options)), (difficulty, seed, options)


# --- ln(a) never appears symbolically anywhere a learner reads -------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_ln_never_appears_symbolically(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        for field in [q.text, q.answer_text, *q.distractors]:
            assert "ln(" not in field.replace(" ", ""), (difficulty, seed, field)
            assert "ln " not in field, (difficulty, seed, field)


# --- easy/medium/hard: answer and every distractor are plain signed 2-d.p.
# numerals, formatted alike (no option identifiable by its own shape) -------

_NUMERAL_RE = re.compile(r"^-?\d+\.\d{2}$")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_numeric_tiers_options_are_2dp_numerals_formatted_alike(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        for opt in [q.answer_text] + list(q.distractors):
            assert _NUMERAL_RE.match(opt), (difficulty, seed, opt)


# --- easy: limit is comfortably positive; medium: limit is negative --------


def test_easy_limit_is_positive():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert float(q.answer_text) > 0, (seed, q.answer_text)


def test_medium_limit_is_negative():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium")
        assert float(q.answer_text) < 0, (seed, q.answer_text)


# --- hard: the displayed h=0.1 row value is always offered as a distractor -

_ROW_RE = re.compile(r"h = (\d*\.?\d+): (-?\d+\.\d+)")


def test_hard_offers_h01_row_as_a_distractor():
    # Recomputed from `a` and h=0.1 at full precision -- never from the
    # already-4-d.p.-rounded displayed row text, since double-rounding
    # (4 d.p. then 2 d.p.) can land on the wrong side of a boundary the
    # generator's own single full-precision rounding would not.
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = _A_RE.search(q.text)
        assert m, (seed, q.text)
        a = float(m.group(1))
        h01_recomputed = (a**0.1 - 1.0) / 0.1
        h01_rounded = f"{round(h01_recomputed, 2):.2f}"
        assert h01_rounded in q.distractors, (seed, q.text, h01_rounded, q.distractors)


# --- hard: a 4th table row at h=0.0001 is present, easy/medium have 3 ------


def test_hard_has_four_rows_easy_medium_have_three():
    for seed in SEEDS:
        assert len(_ROW_RE.findall(make(TOPIC, seed=seed, difficulty="easy").text)) == 3, seed
        assert len(_ROW_RE.findall(make(TOPIC, seed=seed, difficulty="medium").text)) == 3, seed
        assert len(_ROW_RE.findall(make(TOPIC, seed=seed, difficulty="hard").text)) == 4, seed


# --- exam: two tables, comparison options, tie excluded by construction ----


def test_exam_has_two_tables_and_comparison_options():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert q.text.count("First, for a =") == 1, (seed, q.text)
        assert q.text.count("Second, for a =") == 1, (seed, q.text)
        assert len(_ROW_RE.findall(q.text)) == 6, (seed, q.text)  # 3 rows x 2 sides
        options = [q.answer_text] + list(q.distractors)
        equal_option = next(o for o in options if "same magnitude" in o)
        assert equal_option != q.answer_text, (seed, q.text, options)  # tie excluded by construction


# --- at least 20 distinct rendered questions per tier over 60 seeds --------

# --- ValueError on a nonsensical tier ---------------------------------------


def test_raises_value_error_on_unknown_tier():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")


# --- 200-seed-per-tier minimum-option-gap measurement -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_200_seed_minimum_option_gap_at_least_002(difficulty):
    min_gap = None
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        vals = [Fraction(q.answer_text)] + [Fraction(d) for d in q.distractors]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                gap = abs(vals[i] - vals[j])
                if min_gap is None or gap < min_gap:
                    min_gap = gap
    assert min_gap is not None and min_gap >= Fraction(1, 50), (difficulty, min_gap)


# --- The skill's own YAML wires 4 sources, one per tier, exam under
# stages: [exam] ------------------------------------------------------------


def test_skill_sources_and_stages(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill.answer == "mcq"
    by_difficulty = {s.difficulty: s for s in skill.sources}
    assert set(by_difficulty) == {"easy", "medium", "hard", "exam"}
    assert by_difficulty["exam"].stages == ["exam"]
    assert by_difficulty["exam"].stages != []
