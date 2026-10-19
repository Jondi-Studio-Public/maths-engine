"""Normality-of-p-hat hard tier states failures, not successes."""

from __future__ import annotations

import re

from mathgen.grading import GRADERS, mcq_options
from mathgen import make
from mathgen.generators.p8c_normality_approx_build import CATEGORY_P_FAILS, CATEGORY_Q_FAILS
from mathgen.text_quality import prompt_shape

TOPIC = "p8c_normality_approx_phat"
_PAT = re.compile(r"n = (\d+) finds x = (\d+) failures, and p-hat is the proportion of successes")


def _key_from_text(text: str) -> str:
    """Own parser: successes = n - x, failures = x."""
    m = _PAT.search(text)
    assert m, text
    n, x = int(m.group(1)), int(m.group(2))
    assert x >= 2 and n - x >= 2
    successes, failures = n - x, x
    if successes < 10 <= failures:
        return CATEGORY_P_FAILS
    if failures < 10 <= successes:
        return CATEGORY_Q_FAILS
    raise AssertionError(f"symmetric draw: {text}")


def _hard(seed):
    return make(TOPIC, seed=seed, difficulty="hard", variant="from_failures")


def test_key_matches_rendered_text_and_both_categories_reachable():
    seen = set()
    for seed in range(1000):
        q = _hard(seed)
        assert q.answer_text == _key_from_text(q.text), seed
        seen.add(q.answer_text)
    assert seen == {CATEGORY_P_FAILS, CATEGORY_Q_FAILS}


def test_default_path_unchanged_and_new_text_differs():
    for seed in range(20):
        a = make(TOPIC, seed=seed, difficulty="hard")
        b = make(TOPIC, seed=seed, difficulty="hard", variant="borderline")
        assert a.text == b.text and a.answer_text == b.answer_text
        assert "failures" not in a.text
        assert a.text != _hard(seed).text


def test_distractors_clear_the_real_grader():
    grader = GRADERS["mcq"]
    for seed in range(60):
        q = _hard(seed)
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4
        for i, option in enumerate(options):
            assert grader(q, i).correct == (option == q.answer_text), (seed, option)


def test_prompt_shape_differs_from_easy_and_medium():
    hard = prompt_shape(_hard(3))
    assert hard != prompt_shape(make(TOPIC, seed=3, difficulty="easy"))
    assert hard != prompt_shape(make(TOPIC, seed=3, difficulty="medium"))
