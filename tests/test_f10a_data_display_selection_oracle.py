"""Independent oracle for `f10-data-display-selection`.

It does not import the generator. From the question text alone it works out which display the stated purpose calls
for, using its own keyword rules on the purpose sentence, and checks that the key is that display and no distractor is.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "f10a_data_display_selection"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("F10A_DDS_ORACLE_SEEDS", "200"))

# Ordered rules over the "The aim is to ..." sentence. First match wins.
RULES = [
    (r"value by value", "back-to-back stem-and-leaf plot"),
    (r"shape of|are distributed|are spread, whatever", "histogram"),
    (r"what (?:share|part) of", "pie chart"),
    (r"compare (?:how many|the number)", "column graph"),
    (r"whether .* tend", "scatterplot"),
    (r"changed|trend|moved|rising", "line graph"),
]


def expected_display(text: str) -> str:
    aim = re.search(r"The aim is to ([^?]*?\.)", text).group(1)
    for pattern, display in RULES:
        if re.search(pattern, aim):
            return display
    raise AssertionError(f"no rule matches: {aim}")


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_the_display_the_purpose_calls_for(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert q.answer_text == expected_display(q.text), (tier, seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_no_distractor_is_the_expected_display(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert expected_display(q.text) not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_pie_chart_is_the_key_only_for_a_share_of_a_whole(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert (q.answer_text == "pie chart") == bool(re.search(r"what (?:share|part) of", q.text))


@pytest.mark.parametrize("tier", TIERS)
def test_pie_chart_distractor_only_for_non_share_aims(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        if "pie chart" in q.distractors:
            assert q.option_notes["pie chart"] == "pie-chart-for-numerical-data"
            assert not re.search(r"what (?:share|part) of", q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_options_are_never_the_two_banned_rivals(tier):
    # A dot plot or box plot would also suit a distribution; neither is ever offered.
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        for option in (q.answer_text, *q.distractors):
            assert "dot plot" not in option and "box" not in option and "segmented" not in option
