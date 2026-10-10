"""Exam source for total change when the rate's constant is recovered from the change over a first interval.

The oracle re-reads the exponent, the first-interval change and the two limits from the prompt and solves for k with sympy.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import Question
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_c4_total_change"
HARD_TOPIC = "calculate_total_change_by_integrating_instantaneous_or_marginal"
GRADER = GRADERS["expression"]
_HARD_KW = {}


def _make(seed: int) -> Question:
    return make(TOPIC, seed=seed, difficulty="exam")


def _hard(seed: int) -> Question:
    return make(HARD_TOPIC, seed=seed, difficulty="hard", **_HARD_KW)


def test_no_text_defects():
    for seed in range(60):
        q = _make(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_make(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert shapes - hard_shapes, (shapes, hard_shapes)


def test_determinism():
    for seed in (0, 1, 7, 99, 12345):
        assert _make(seed).to_json() == _make(seed).to_json()


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")


from sympy import Symbol, integrate, solve, sqrt, sympify

_T = Symbol("t", positive=True)
_K = Symbol("k", positive=True)


def _agrees(q: Question) -> bool:
    t = q.text
    if "k·sqrt(t)" in t:
        rate = _K * sqrt(_T)
    else:
        m = re.search(r"k·t(?:\^(\d))?", t)
        rate = _K * _T ** int(m.group(1) or 1)
    lo, hi = (int(v) for v in re.search(r"between t = (\d+) and t = (\d+)", t).groups())
    first = int(re.search(r"(?:is|by) (\d+)[.]", t).group(1))
    k = solve(integrate(rate, (_T, 0, lo)) - first, _K)[0]
    return sympify(q.answer_text) == integrate(rate.subs(_K, k), (_T, lo, hi))


def test_prompts_show_both_forms_of_wording():
    starts = {q.text.split()[0] for q in (_make(s) for s in range(60))}
    assert starts == {"The", "For"}, starts


def test_independent_oracle_matches_key_over_400_seeds():
    bad = []
    for seed in range(400):
        q = _make(seed)
        if not _agrees(q):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:5]


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for seed in range(150):
        q = _make(seed)
        assert len(q.distractors) >= 1, (seed, q.text)
        if not GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:5]
