"""Tests the exam source of `probability_likelihood_compare` (`exd_x_s3_likelihood`): the key is re-derived from the
rendered prompt alone by enumerating every ordered pair of labelled counters."""

from __future__ import annotations

import itertools
import re
from fractions import Fraction

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s3_likelihood"
HARD = "probability_likelihood_compare"


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(text: str) -> str:
    r, b = map(int, re.search(r"holds (\d+) red and (\d+) blue", text).groups())
    replace = "without replacement" not in text
    event = re.search(r"Event A is that (.*?)\. Event B", text).group(1)
    num_b, den_b = map(int, re.search(r"probability (\d+)/(\d+)", text).groups())
    counters = list("R" * r + "B" * b)
    idx = range(len(counters))
    pairs = itertools.product(idx, repeat=2) if replace else itertools.permutations(idx, 2)
    good = total = 0
    for i, j in pairs:
        c1, c2 = counters[i], counters[j]
        total += 1
        if event == "both counters are red":
            good += c1 == c2 == "R"
        elif event == "the two counters are the same colour":
            good += c1 == c2
        elif event == "at least one counter is red":
            good += "R" in (c1, c2)
        else:
            assert event == "exactly one counter is red", event
            good += (c1 == "R") != (c2 == "R")
    p_a, p_b = Fraction(good, total), Fraction(num_b, den_b)
    return "A" if p_a > p_b else "B" if p_a < p_b else "equal"


_WORD = {"A": "event A is more likely", "B": "event B is more likely", "equal": "A and B are equally likely"}


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        if q.answer_text != _WORD[_oracle(q.text)]:
            bad.append((seed, q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_all_three_outcomes_and_all_events_occur():
    assert {_make(s).answer_text for s in range(100)} == set(_WORD.values())
    events = {re.search(r"Event A is that (.*?)\.", _make(s).text).group(1) for s in range(100)}
    assert len(events) == 4


def test_grader_accepts_key_and_rejects_every_distractor_over_100_seeds():
    grader = GRADERS["mcq"]
    for seed in range(100):
        q = _make(seed)
        options = mcq_options(q)
        assert len(options) == 3 and len(set(options)) == 3
        for i, option in enumerate(options):
            assert grader(q, i).correct == (option == q.answer_text), (seed, option)
