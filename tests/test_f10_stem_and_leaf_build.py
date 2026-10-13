"""Generator-specific proofs for `f10-stem-and-leaf-plots` (`answer: number`).

The independent oracle (it reads only the question text) is `tests/test_f10_stem_and_leaf_oracle.py`.
"""

from __future__ import annotations

import random
import re
from collections import Counter

import pytest

import mathgen.generators.f10_stem_and_leaf_build as gen
from mathgen import make
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills

SKILL_ID = "f10-stem-and-leaf-plots"
TOPIC = "f10_stem_and_leaf"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(80)


def _skill():
    _domains, skills = load_skills()
    return next(s for s in skills if s.id == SKILL_ID)


SKILL = _skill()
SOURCES = {src.difficulty: src for src in SKILL.sources}


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def test_skill_shape():
    assert SKILL.status == "active"
    assert SKILL.answer == "number"
    assert [s.difficulty for s in SKILL.sources] == TIERS
    assert "AC9M7ST02" in SKILL.tags


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism(difficulty):
    for seed in range(15):
        assert _make(difficulty, seed).to_json() == _make(difficulty, seed).to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_the_answer_and_rejects_every_distractor(difficulty):
    miscon = {m.id for m in SKILL.misconceptions}
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert grade_number(q, q.answer_text).correct, (difficulty, seed)
        assert 2 <= len(q.distractors) <= 3, (difficulty, seed)
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not grade_number(q, d).correct, (difficulty, seed, d, q.answer_text)
            assert q.option_notes[d] in miscon, (difficulty, seed, q.option_notes[d])
        assert set(q.option_notes) == set(q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_neighbouring_numbers_are_rejected(difficulty):
    for seed in range(40):
        q = _make(difficulty, seed)
        for bump in (-1, 1, 0.5, -0.5):
            assert not grade_number(q, str(q.answer + bump)).correct, (difficulty, seed, bump)


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_declared_misconception_is_used_somewhere_or_is_in_the_catalogue_of_the_skill(difficulty):
    used = set()
    for seed in range(300):
        used |= set(_make(difficulty, seed).option_notes.values())
    declared = {m.id for m in SKILL.misconceptions}
    assert used <= declared
    assert len(used) >= 4


def test_every_declared_misconception_is_reachable():
    used = set()
    for difficulty in TIERS:
        for seed in range(400):
            used |= set(_make(difficulty, seed).option_notes.values())
    assert {m.id for m in SKILL.misconceptions} == used


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_pool_is_varied(difficulty):
    pool = Counter(_make(difficulty, s).answer_text for s in range(300))
    assert len(pool) >= 25
    assert max(pool.values()) <= 0.2 * 300


@pytest.mark.parametrize("difficulty", TIERS)
def test_text_is_one_clean_line_and_does_not_state_the_answer(difficulty):
    for seed in range(80):
        q = _make(difficulty, seed)
        assert "\n" not in q.text and "  " not in q.text, (difficulty, seed, q.text)
        assert "Key:" in q.text and " | " in q.text
        intro, plot_and_ask = q.text.split("Plot, rows written as ", 1)
        prose = intro + plot_and_ask.split(": ", 1)[1].split(". ", 1)[1]
        assert not re.search(r"\b(\w+)\s+\1\b", prose, re.I), (difficulty, seed, q.text)
        assert not re.search(r"\d", intro.split("Key:")[0].replace("out of 100", "")), (difficulty, seed, q.text)


def _rows(q):
    body = q.text.split("rows written as ", 1)[1].split(": ", 1)[1].split(". ", 1)[0]
    return body.split("; ")


def test_easy_asks_all_four_things():
    asks = Counter()
    for seed in range(120):
        ask = _make("easy", seed).text.rsplit(". ", 1)[1]
        asks[
            "count"
            if "are shown in the plot" in ask
            else "largest"
            if "largest" in ask
            else "smallest"
            if "smallest" in ask
            else "stem"
        ] += 1
    assert set(asks) == {"count", "largest", "smallest", "stem"}
    assert min(asks.values()) >= 15


def test_easy_key_never_names_the_value_asked_for():
    for seed in range(120):
        q = _make("easy", seed)
        m = re.search(r"Key: (\d+) \| (\d) means (\d+)\.", q.text)
        assert int(m.group(1)) * 10 + int(m.group(2)) == int(m.group(3))
        if "largest" in q.text or "smallest" in q.text:
            assert int(m.group(3)) != q.answer, q.text


def test_medium_covers_each_statistic_and_both_orders():
    kinds, ordered = Counter(), Counter()
    for seed in range(200):
        q = _make("medium", seed)
        kinds[re.search(r"What is the (median|range|mode)", q.text).group(1)] += 1
        ordered["not in order" in q.text] += 1
    assert min(kinds.values()) >= 25 and set(kinds) == {"median", "range", "mode"}
    assert ordered[True] >= 60 and ordered[False] >= 60


def test_medium_even_count_medians_are_halves_and_both_parities_occur():
    halves = whole = 0
    for seed in range(200):
        q = _make("medium", seed)
        if "median" in q.text:
            if q.answer != int(q.answer):
                halves += 1
                assert q.answer * 2 == int(q.answer * 2)
            else:
                whole += 1
    assert halves >= 15 and whole >= 15


def test_medium_mode_is_unique_in_every_question():
    for seed in range(200):
        q = _make("medium", seed)
        if "mode" not in q.text:
            continue
        body = _rows(q)
        values = [int(r.split(" | ")[0]) * 10 + int(x) for r in body for x in r.split(" | ")[1].split() if x.isdigit()]
        counts = Counter(values).most_common(2)
        assert counts[0][1] >= 2 and (len(counts) == 1 or counts[0][1] > counts[1][1])


def test_hard_has_both_back_to_back_and_hidden_leaf_and_both_hidden_routes():
    kinds = Counter()
    for seed in range(200):
        t = _make("hard", seed).text
        if "back-to-back" in t:
            kinds["median" if "median" in t else "range"] += 1
        else:
            kinds["hidden-median" if "The median is" in t else "hidden-range"] += 1
    assert set(kinds) == {"median", "range", "hidden-median", "hidden-range"}
    assert min(kinds.values()) >= 20


def test_hard_back_to_back_key_matches_the_plot():
    for seed in range(120):
        q = _make("hard", seed)
        m = re.search(r"Key: (\d) \| (\d+) \| (\d) means (\d+) for .+? and (\d+) for ", q.text)
        if not m:
            continue
        la, s, lb, va, vb = map(int, m.groups())
        assert va == 10 * s + la and vb == 10 * s + lb
        row = next(r for r in _rows(q) if f" | {s} | " in r)
        left, _stem, right = row.split(" | ")
        assert str(la) in left.split() and str(lb) in right.split()


def test_hard_hidden_leaf_is_one_question_mark_and_an_even_median_is_a_half():
    for seed in range(120):
        q = _make("hard", seed)
        if "back-to-back" in q.text:
            continue
        assert "".join(_rows(q)).count("?") == 1
        if "The median is" in q.text:
            assert re.search(r"The median is \d+\.5\.", q.text)


def test_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        gen.f10_stem_and_leaf(random.Random(0), tier="exam")
