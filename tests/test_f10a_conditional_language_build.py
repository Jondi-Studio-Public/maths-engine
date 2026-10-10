"""Generator-specific proofs for `f10-conditional-probability-language`: the real-grader battery, option hygiene,
misconception tagging, template coverage per tier and the skill's own properties.

The independent ORACLE is in `tests/test_f10a_conditional_language_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.f10a_conditional_language_build as gen
from mathgen import make
from mathgen.core import VerificationError
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills

SKILL_ID = "f10-conditional-probability-language"
TOPIC = "f10a_conditional_language"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)


def _skill():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill()
MISCONCEPTION_IDS = {m["id"] if isinstance(m, dict) else m.id for m in SKILL.misconceptions}


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _template(text):
    s = text.split("Read this sentence:")[1]
    if "given that" in s:
        return "given"
    if "Knowing that" in s:
        return "knowing"
    if '"If ' in s:
        return "ifthen"
    return "of"


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
def test_every_distractor_is_tagged_with_a_declared_misconception(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert len(q.distractors) == 3
        assert set(q.option_notes) == set(q.distractors)
        assert set(q.option_notes.values()) <= MISCONCEPTION_IDS


def test_every_declared_misconception_appears_across_draws():
    seen = set()
    for seed in range(60):
        seen |= set(_make("easy", seed).option_notes.values())
    assert seen == MISCONCEPTION_IDS


def test_template_coverage_per_tier():
    for seed in range(120):
        assert _template(_make("easy", seed).text) in ("given", "knowing")
        assert _template(_make("medium", seed).text) in ("ifthen", "of")
    assert {_template(_make("hard", s).text) for s in range(120)} == {"given", "knowing", "ifthen", "of"}


def test_numbers_only_where_the_tier_says():
    for seed in range(60):
        assert not re.search(r"\d", _make("easy", seed).text)
        assert "survey" not in _make("medium", seed).text
        assert "do both" in _make("hard", seed).text
        t = _make("hard", seed).text
        sentence = t.split("Read this sentence:")[1]
        if _template(t) == "of":
            assert " also " in sentence


@pytest.mark.parametrize("difficulty", TIERS)
def test_target_is_not_predictable_from_sentence_position_across_the_tier(difficulty):
    # Whether the target is named first in the sentence must not be constant: in medium/hard it varies by template,
    # and within a template the events are drawn in random order.
    first = Counter()
    for seed in range(200):
        q = _make(difficulty, seed)
        key_target = q.answer_text.split("the probability of ")[1].split(", given ")[0]
        sentence = q.text.split("Read this sentence:")[1]
        verb_of_target = {
            v for e in [e for c in gen.CONTEXTS for e in c["events"]] if e[0] == key_target for v in e[1:]
        }
        pos = min(sentence.find(v) for v in verb_of_target if sentence.find(v) >= 0)
        cond_name = q.answer_text.split(", given ")[1]
        verb_of_cond = {v for e in [e for c in gen.CONTEXTS for e in c["events"]] if e[0] == cond_name for v in e[1:]}
        pos_c = min(sentence.find(v) for v in verb_of_cond if sentence.find(v) >= 0)
        first[pos < pos_c] += 1
    if difficulty == "easy":
        assert 0.3 <= first[True] / 200 <= 0.7, first
    elif difficulty == "medium":
        assert first[True] == 0  # "if" and "of" always name the condition first
    else:
        assert 0.15 <= first[True] / 200 <= 0.6, first


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_position_and_option_length_do_not_give_it_away(difficulty):
    positions = Counter()
    longest = 0
    for seed in range(240):
        q = _make(difficulty, seed)
        positions[mcq_options(q).index(q.answer_text)] += 1
        lengths = [len(o) for o in (q.answer_text, *q.distractors)]
        longest += len(q.answer_text) == max(lengths) and lengths.count(max(lengths)) == 1
    assert max(positions.values()) / 240 <= 0.40, positions
    assert longest / 240 <= 0.5, longest


@pytest.mark.parametrize("difficulty", TIERS)
def test_text_only_and_clean(difficulty):
    for seed in range(60):
        q = _make(difficulty, seed)
        assert q.figure is None
        assert "  " not in q.text and "**" not in q.text
        assert not re.search(r"\b(\w+)\s+\1\b", q.text, re.I), q.text


def test_marker_check_fires_when_the_sentence_names_the_wrong_condition(monkeypatch):
    real = gen._sentence

    def swapped(template, ctx, cond, target, c, b, hard, pct):
        return real(template, ctx, target, cond, c, b, hard, pct)

    monkeypatch.setattr(gen, "_sentence", swapped)
    with pytest.raises(VerificationError):
        for seed in range(30):
            make(TOPIC, seed=seed, difficulty="easy")


def test_skill_properties():
    assert SKILL.domain == "statistics-probability"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "mcq"
    assert {s.difficulty for s in SKILL.sources} == set(TIERS)
    assert "AC9M10P01" in SKILL.tags
    _d, skills = load_skills()
    for prereq in SKILL.prerequisites:
        assert next(s for s in skills if s.id == prereq).status == "active"
