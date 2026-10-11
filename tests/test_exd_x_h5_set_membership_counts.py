"""Tests for the exam source of `senior-determine-and-use-the-formulas-for-finding-the`
(topic `exd_x_h5_set_membership_counts`)."""

from __future__ import annotations

import re

import sympy

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h5_set_membership_counts"
SKILL_TOPIC = "p11c_set_union_cardinality"
_GROUP = {"exactly one": 1, "exactly two": 2, "all three": 3}


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard_shapes(n=60):
    return {prompt_shape(make(SKILL_TOPIC, seed=s, difficulty="hard", mode="three_set_unknown")) for s in range(n)}


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_prompt_variety_and_new_shapes():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3
    assert shapes - _hard_shapes()


def test_determinism():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def _group(phrase_text):
    for key, val in _GROUP.items():
        if phrase_text.startswith(key):
            return val
    raise AssertionError(phrase_text)


def _oracle(text):
    """Re-derive the asked-for count from the prompt alone, solving the two linear relations with sympy."""
    nums = [int(v) for v in re.findall(r"\d+", text.split("How many")[0].split("Find the number")[0])]
    n_total = nums[0]
    a, b, c = nums[1:4]
    e0, known_value = nums[4], nums[5]
    known = re.search(r"belong to (exactly one|exactly two|all three)", text.split("Find the")[0].split("How many")[0])
    known_group = _group(known.group(1))
    ask_part = re.split(r"How many students belong to |Find the number of students who belong to ", text)[1]
    asked_group = _group(ask_part)
    e1, e2, e3 = sympy.symbols("e1 e2 e3")
    syms = {1: e1, 2: e2, 3: e3}
    sol = sympy.solve(
        [
            sympy.Eq(e1 + e2 + e3, n_total - e0),
            sympy.Eq(e1 + 2 * e2 + 3 * e3, a + b + c),
            sympy.Eq(syms[known_group], known_value),
        ],
        [e1, e2, e3],
        dict=True,
    )
    assert len(sol) == 1
    return int(sol[0][syms[asked_group]])


def test_key_matches_independent_oracle_over_300_seeds():
    bad = [(s, _exam(s).text) for s in range(300) if _oracle(_exam(s).text) != _exam(s).answer]
    assert bad == []


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["number"]
    for seed in range(120):
        q = _exam(seed)
        assert grade(q, q.answer_text).correct
        assert len(q.distractors) == 3
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
