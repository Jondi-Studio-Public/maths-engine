"""Tests for the exam source of `senior-find-all-solutions-of-mathrm-f-left-mathrm` (topic `exd_x_f5_trigquad`):
quadratics in sin(x) or cos(x)."""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f5_trigquad"
_GRADER = GRADERS["mcq"]


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed):
    return make("trig_equation_all_solutions", seed=seed, difficulty_tier="hard")


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert exam_shapes - hard_shapes


def test_deterministic():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nope")


def _solutions(text):
    """Every whole degree in [0, 360) satisfying the equation printed in the prompt, found by substitution."""
    eq = re.search(r"solutions of (.*?) for 0 <= x < 360", text).group(1)
    eq = re.sub(r"(sin|cos)\^2\(x\)", r"(\1(x))**2", eq)
    eq = re.sub(r"(\d)(sin|cos|\()", r"\1*\2", eq)
    lhs, rhs = eq.split(" = ")
    found = []
    for d in range(360):
        env = {"sin": lambda v: math.sin(math.radians(v)), "cos": lambda v: math.cos(math.radians(v)), "x": d}
        if abs(eval(lhs, {"__builtins__": {}}, env) - eval(rhs, {"__builtins__": {}}, env)) < 1e-9:
            found.append(d)
    return found


def test_key_matches_independent_derivation_over_320_seeds():
    bad = []
    for seed in range(320):
        q = _exam(seed)
        listed = [int(v) for v in re.findall(r"(\d+)°", q.answer_text)]
        if _solutions(q.text) != listed:
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor_over_120_seeds():
    bad = []
    for seed in range(120):
        q = _exam(seed)
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4
        for idx, option in enumerate(options):
            ok = _GRADER(q, idx).correct
            if ok != (option == q.answer_text):
                bad.append((seed, idx, option, q.answer_text))
    assert bad == [], bad[:3]
