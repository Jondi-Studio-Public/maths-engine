"""Contract tests for f10-symmetry-construction (topic f10_symmetry_lines_rotation, answer kind pair)."""

import math
import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.generators.f10_symmetry_build import (
    EASY_KEYS,
    EXTERIOR_SIDE_COUNTS,
    HARD_FEATURE_KEYS,
    MEDIUM_KEYS,
    SHAPES,
    _symmetries,
)
from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills

TOPIC = "f10_symmetry_lines_rotation"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(80)


def test_symmetry_finder_agrees_with_well_known_values_it_was_not_tuned_to():
    # Independent, hand-known results (the finder is also run against the table at import time).
    assert _symmetries([(0, 0), (1, 0), (1, 1), (0, 1)]) == (4, 4)  # unit square
    assert _symmetries([(0, 0), (2, 0), (2, 1), (0, 1)]) == (2, 2)  # 2 by 1 rectangle
    assert _symmetries([(0, 0), (2, 0), (3, 1), (1, 1)]) == (0, 2)  # slanted parallelogram
    assert _symmetries([(0, 0), (4, 0), (0, 3)]) == (0, 1)  # 3-4-5 triangle
    assert _symmetries([(0, 0), (4, 0), (2, 5)]) == (1, 1)  # isosceles
    for n in range(3, 21):
        pts = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
        assert _symmetries(pts) == (n, n)


def test_table_values_are_the_textbook_values():
    textbook = {
        "regular-3": (3, 3),
        "regular-4": (4, 4),
        "isosceles-triangle": (1, 1),
        "scalene-triangle": (0, 1),
        "rectangle": (2, 2),
        "parallelogram": (0, 2),
        "rhombus": (2, 2),
        "kite": (1, 1),
        "isosceles-trapezium": (1, 1),
        "right-trapezium": (0, 1),
        "plus-sign": (4, 4),
        "five-pointed-star": (5, 5),
    }
    for key, expected in textbook.items():
        assert (SHAPES[key].lines, SHAPES[key].order) == expected
        assert _symmetries(SHAPES[key].vertices) == expected


def test_tier_pools_use_only_known_shapes():
    for keys in (EASY_KEYS, MEDIUM_KEYS, HARD_FEATURE_KEYS):
        assert set(keys) <= set(SHAPES)
    assert all(360 % n == 0 for n in EXTERIOR_SIDE_COUNTS)


@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_and_every_distractor_clears_the_real_grader(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.to_json() == make(TOPIC, seed=seed, difficulty=difficulty).to_json()
        assert set(q.answer) == {"lines", "order"}
        assert grade_pair(q, q.answer_text).correct
        assert grade_pair(q, f"order = {q.answer['order']}, lines = {q.answer['lines']}").correct
        assert len(q.distractors) >= 2
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert not grade_pair(q, d).correct, (q.text, d)
        assert set(q.option_notes) == set(q.distractors)


def _shape_from_text(text, difficulty):
    """The shape whose name (or, in the hard tier, feature phrase) appears in the text; longest phrase wins."""
    best = None
    for shape in SHAPES.values():
        phrase = shape.features if difficulty == "hard" else shape.name
        if phrase in text and (best is None or len(phrase) > best[0]):
            best = (len(phrase), shape)
    return best[1] if best else None


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_matches_independent_recomputation_from_the_text(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        exterior = re.search(r"exterior angle of a regular polygon is (\d+)", q.text)
        if exterior:
            n = 360 // int(exterior.group(1))
            assert (q.answer["lines"], q.answer["order"]) == (n, n)
            continue
        shape = _shape_from_text(q.text, difficulty)
        assert shape is not None, q.text
        assert (q.answer["lines"], q.answer["order"]) == _symmetries(shape.vertices)


def test_hard_tier_never_names_the_shape_and_has_both_modes():
    modes = set()
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert not any(shape.name in q.text for shape in SHAPES.values())
        modes.add("exterior" if "exterior angle" in q.text else "features")
    assert modes == {"exterior", "features"}


def test_medium_lines_and_order_often_differ():
    differing = sum(
        1
        for s in SEEDS
        if make(TOPIC, seed=s, difficulty="medium").answer["lines"]
        != make(TOPIC, seed=s, difficulty="medium").answer["order"]
    )
    assert differing >= 20


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_pool_is_not_dominated_by_one_answer(difficulty):
    pool = Counter(make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(300))
    assert len(pool) >= 7
    assert max(pool.values()) / 300 <= 0.35


def test_skill_is_active_pair_and_every_note_is_declared():
    _, skills = load_skills()
    skill = next(x for x in skills if x.id == "f10-symmetry-construction")
    assert skill.status == "active" and skill.answer == "pair"
    ids = {m["id"] if isinstance(m, dict) else m.id for m in skill.misconceptions}
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert set(q.option_notes.values()) <= ids
