"""Tests the rendered text of the p16a questions for lowercase sentence starts and stray repeat-count phrasing.

Render quality."""

from __future__ import annotations

import re

import pytest

from mathgen import make

TOPICS_AND_TIERS = [
    ("p16a_group_centre_difference", ["easy", "medium", "hard", "exam"]),
    ("p16a_trend_value", ["easy", "medium", "hard", "exam"]),
    ("p16a_conditional_independence", ["easy", "medium", "hard", "exam"]),
    ("p16a_estimated_distribution", ["easy", "medium", "hard", "exam"]),
    ("p16a_relative_frequency_estimate", ["easy", "medium", "hard", "exam"]),
    ("p16a_histogram_probability", ["easy", "medium", "hard", "exam"]),
]

SEEDS = range(30)

# A sentence boundary is the very start of the text, or ". " (a period then
# a space). The character right after it must never be a lowercase letter --
# this is what "the morning shift recorded ..." opening a sentence, or a
# joined clause like "of the students who ..." left uncapitalised after its
# own period, both look like.
_SENTENCE_START_RE = re.compile(r"(?:^|\. )([a-z])")

_WAS_USED_NON_DEVICE_RE = re.compile(r"\b(?:calls?|packets?|seeds?|surveys?|records?)\b[^.]*\bwas used \d+ times")


@pytest.mark.parametrize("topic,tiers", TOPICS_AND_TIERS)
def test_no_lowercase_sentence_start(topic, tiers):
    for tier in tiers:
        for seed in SEEDS:
            q = make(topic, seed=seed, difficulty=tier)
            m = _SENTENCE_START_RE.search(q.text)
            assert not m, f"{topic}/{tier} seed {seed}: lowercase sentence start {m.group(0)!r}. {q.text}"


@pytest.mark.parametrize("topic,tiers", TOPICS_AND_TIERS)
def test_no_was_used_n_times_on_a_non_device(topic, tiers):
    for tier in tiers:
        for seed in SEEDS:
            q = make(topic, seed=seed, difficulty=tier)
            m = _WAS_USED_NON_DEVICE_RE.search(q.text)
            assert not m, f"{topic}/{tier} seed {seed}: {m.group(0)!r}. {q.text}"
