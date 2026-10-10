"""Hypergeometric-probability at-least-one hard tier."""

from __future__ import annotations

import re
from fractions import Fraction
from math import comb

from mathgen.grading import GRADERS
from mathgen import make
from mathgen.text_quality import prompt_shape

_PROMPT = re.compile(
    r"A population has (\d+) objects, (\d+) of them successes\. "
    r"If (\d+) objects are drawn without replacement, find the probability that at least one"
)


def _make(seed, **kw):
    return make("hypergeometric_probability", seed=seed, difficulty="hard", **kw)


def test_at_least_key_matches_recomputation_from_prompt():
    for seed in range(60):
        q = _make(seed, kind="at_least")
        n_pop, succ, draws = map(int, _PROMPT.search(q.text).groups())
        assert 10 <= n_pop <= 45 and 2 <= succ <= n_pop - 3 and 2 <= draws <= n_pop - succ
        by_sum = sum(
            Fraction(comb(succ, x) * comb(n_pop - succ, draws - x), comb(n_pop, draws))
            for x in range(1, min(draws, succ) + 1)
        )
        assert 0 < by_sum < 1
        assert Fraction(q.answer_text) == by_sum, seed


def test_default_path_unchanged_and_new_kind_differs():
    for seed in range(20):
        default = _make(seed)
        assert default.text == _make(seed, kind=None).text
        assert "P(X=" in default.text
        assert _make(seed, kind="at_least").text != default.text


def test_at_least_distractors_clear_the_real_grader():
    grader = GRADERS["fraction"]
    for seed in range(60):
        q = _make(seed, kind="at_least")
        assert grader(q, q.answer_text).correct
        assert len(q.distractors) == 3
        for distractor in q.distractors:
            assert not grader(q, distractor).correct, (seed, distractor, q.answer_text)


def test_at_least_prompt_shape_differs_from_easy_and_medium():
    hard = prompt_shape(_make(3, kind="at_least"))
    for diff in ("easy", "medium"):
        other = make("hypergeometric_probability", seed=3, difficulty=diff)
        assert prompt_shape(other) != hard
