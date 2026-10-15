"""Tests the exam source of `p8c_normality_approx_phat` (`exd_x_s3_normality`): the key is re-derived from the
rendered prompt alone."""

from __future__ import annotations

import re

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s3_normality"
HARD = "p8c_normality_approx_phat"


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(text: str) -> str:
    total, share = map(int, re.search(r"survey of (\d+) \w+ found that (\d+)%", text).groups())
    q = int(re.search(r"Of this group, (\d+)%", text).group(1))
    n_sub = total * share // 100
    x = n_sub * q // 100
    assert n_sub * q % 100 == 0 and total * share % 100 == 0
    p_ok, q_ok = x >= 10, n_sub - x >= 10
    if p_ok and q_ok:
        return "Valid"
    if p_ok:
        return "Not valid -- n×(1-p-hat) is less than 10"
    if q_ok:
        return "Not valid -- n×p-hat is less than 10"
    return "Not valid -- both"


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        if not q.answer_text.startswith(_oracle(q.text)):
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_all_four_categories_occur():
    assert len({_make(s).answer_text for s in range(100)}) == 4


def test_grader_accepts_key_and_rejects_every_distractor_over_100_seeds():
    grader = GRADERS["mcq"]
    for seed in range(100):
        q = _make(seed)
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4
        for i, option in enumerate(options):
            assert grader(q, i).correct == (option == q.answer_text), (seed, option)
