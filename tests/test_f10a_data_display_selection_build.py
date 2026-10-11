"""Generator-specific proofs for `f10-data-display-selection`: the real-grader battery, option hygiene, the
misconception tagging of every wrong option, tier shape and the skill's own properties.

The independent ORACLE is in `tests/test_f10a_data_display_selection_oracle.py`.
"""

from __future__ import annotations

from collections import Counter

import pytest

import mathgen.generators.f10a_data_display_selection_build as gen
from mathgen import make
from mathgen.core import VerificationError
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills

SKILL_ID = "f10-data-display-selection"
TOPIC = "f10a_data_display_selection"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)
ALL_DISPLAYS = {gen.COLUMN, gen.PIE, gen.HIST, gen.LINE, gen.SCATTER, gen.BACK}


def _skill():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill()
MISCONCEPTION_IDS = {m["id"] if isinstance(m, dict) else m.id for m in SKILL.misconceptions}


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_same_seed_same_question(difficulty):
    for seed in SEEDS:
        a, b = _make(difficulty, seed), _make(difficulty, seed)
        assert (a.text, a.answer_text, a.distractors) == (b.text, b.answer_text, b.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_key_and_rejects_every_other_option(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4
        for index, option in enumerate(options):
            assert grade_mcq(q, str(index)).correct == (option == q.answer_text)
        assert not grade_mcq(q, "not a number").correct
        assert not grade_mcq(q, "9").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_option_is_a_known_display_and_every_distractor_is_tagged(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert q.answer_text in ALL_DISPLAYS
        assert len(q.distractors) == 3
        assert set(q.option_notes) == set(q.distractors)
        for note in q.option_notes.values():
            assert note in MISCONCEPTION_IDS, (difficulty, seed, note)


def test_every_skill_misconception_is_used_by_the_generator():
    used = {m for rows in gen.WRONG.values() for _d, m in rows}
    assert used == MISCONCEPTION_IDS


def test_no_class_lists_its_own_key_as_a_wrong_display():
    for cls, rows in gen.WRONG.items():
        assert gen.KEY[cls] not in {d for d, _m in rows}, cls
        assert len(rows) >= 3, cls


def test_defensible_rivals_are_never_offered():
    # Pie chart would also compare category sizes (A, H1); column graph would also show a share (B), a continuous
    # shape grouped into bars (C, H2) or a yearly trend (D, H3); histogram would also be drawn per group (F).
    banned = {
        "A": {gen.PIE},
        "H1": {gen.PIE},
        "B": {gen.COLUMN},
        "C": {gen.COLUMN},
        "H2": {gen.COLUMN},
        "D": {gen.COLUMN, gen.SCATTER},
        "H3": {gen.COLUMN},
        "F": {gen.HIST},
    }
    for cls, rivals in banned.items():
        assert not rivals & {d for d, _m in gen.WRONG[cls]}, cls


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_display_is_the_key_more_than_a_third_of_draws_and_position_is_not_fixed(difficulty):
    keys = Counter(_make(difficulty, seed).answer_text for seed in range(240))
    assert max(keys.values()) / 240 <= 0.34, keys
    positions = Counter()
    for seed in range(240):
        q = _make(difficulty, seed)
        positions[mcq_options(q).index(q.answer_text)] += 1
    assert max(positions.values()) / 240 <= 0.40, positions


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_is_not_given_away_by_option_length(difficulty):
    longest = shortest = 0
    for seed in range(240):
        q = _make(difficulty, seed)
        lengths = [len(o) for o in (q.answer_text, *q.distractors)]
        k = len(q.answer_text)
        longest += k == max(lengths) and lengths.count(k) == 1
        shortest += k == min(lengths) and lengths.count(k) == 1
    assert longest / 240 <= 0.34, (difficulty, longest)
    assert shortest / 240 <= 0.34, (difficulty, shortest)


def test_easy_names_the_variable_and_medium_hard_do_not():
    for seed in range(60):
        assert "The variable is" in _make("easy", seed).text or "There are two numerical" in _make("easy", seed).text
        assert "The variable is" not in _make("medium", seed).text
        assert "There are two numerical variables, both" not in _make("medium", seed).text


def test_hard_tier_reaches_all_four_traps():
    seen = set()
    for seed in range(120):
        t = _make("hard", seed).text
        if "bib" in t or "postcode" in t or "room number" in t:
            seen.add("label")
        if "not when each call" in t or "whatever day" in t or "not how they changed" in t:
            seen.add("order-ignored")
        if "two numerical variables: the year" in t:
            seen.add("year")
        if "whole number" in t or "(1 to 40)" in t or "(0 to 30)" in t:
            seen.add("ordered-integer")
    assert seen == {"label", "order-ignored", "year", "ordered-integer"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_text_only_and_clean(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert q.figure is None
        assert "  " not in q.text and "**" not in q.text
        assert q.text.rstrip().endswith("?")


def test_construction_check_fires_when_a_rival_is_listed_as_the_key(monkeypatch):
    bad = {k: list(v) for k, v in gen.WRONG.items()}
    bad["A"] = [(gen.COLUMN, "line-graph-for-categories")] + bad["A"][1:]
    monkeypatch.setattr(gen, "WRONG", bad)
    hit = False
    for seed in range(200):
        try:
            make(TOPIC, seed=seed, difficulty="easy")
        except (VerificationError, ValueError):
            hit = True
            break
    assert hit


def test_skill_properties():
    assert SKILL.domain == "statistics-probability"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "mcq"
    assert {s.difficulty for s in SKILL.sources} == set(TIERS)
    assert "AC9M9ST04" in SKILL.tags
    for prereq in SKILL.prerequisites:
        _d, skills = load_skills()
        assert next(s for s in skills if s.id == prereq).status == "active"
