"""Independent oracle for `f10-conditional-probability-language`.

It does not import the generator. From the finished question text it parses the sentence by its marker word, turns
the two verb phrases into event names with its own small lexicon, builds the expected "the probability of <target>,
given <condition>" description, and checks the key equals it while every distractor differs.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "f10a_conditional_language"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("F10A_CPL_ORACLE_SEEDS", "250"))

# verb phrase (third person singular or base) -> event name
LEXICON = {
    "swims": "swimming", "swim": "swimming",
    "plays netball": "playing netball", "play netball": "playing netball",
    "walks to school": "walking to school", "walk to school": "walking to school",
    "owns a bike": "owning a bike", "own a bike": "owning a bike",
    "learns the piano": "learning the piano", "learn the piano": "learning the piano",
    "studies French": "studying French", "study French": "studying French",
    "pays by card": "paying by card", "pay by card": "paying by card",
    "buys fruit": "buying fruit", "buy fruit": "buying fruit",
    "uses a loyalty card": "using a loyalty card", "use a loyalty card": "using a loyalty card",
    "shops online": "shopping online", "shop online": "shopping online",
    "brings their own bag": "bringing their own bag", "bring their own bag": "bringing their own bag",
    "owns a dog": "owning a dog", "own a dog": "owning a dog",
    "owns a cat": "owning a cat", "own a cat": "owning a cat",
    "recycles": "recycling", "recycle": "recycling",
    "has solar panels": "having solar panels", "have solar panels": "having solar panels",
    "grows vegetables": "growing vegetables", "grow vegetables": "growing vegetables",
    "borrows a novel": "borrowing a novel", "borrow a novel": "borrowing a novel",
    "borrows a DVD": "borrowing a DVD", "borrow a DVD": "borrowing a DVD",
    "uses the study room": "using the study room", "use the study room": "using the study room",
    "attends story time": "attending story time", "attend story time": "attending story time",
}  # fmt: skip
_PHRASES = sorted(LEXICON, key=len, reverse=True)
_PHRASE_RE = "(" + "|".join(re.escape(p) for p in _PHRASES) + ")"

GIVEN = re.compile(
    r'"What is the probability that an? randomly chosen [a-z ]+? '
    + _PHRASE_RE
    + r", given that the [a-z ]+? "
    + _PHRASE_RE
    + r'\?"'
)
KNOWING = re.compile(
    r'"Knowing that an? randomly chosen [a-z ]+? '
    + _PHRASE_RE
    + r", what is the probability that the [a-z ]+? "
    + _PHRASE_RE
    + r'\?"'
)
IFTHEN = re.compile(
    r'"If an? randomly chosen [a-z ]+? '
    + _PHRASE_RE
    + r", then the probability that the [a-z ]+? "
    + _PHRASE_RE
    + r' is [^"]+\."'
)
OF = re.compile(r'"Of the \d+ [a-z ]+? who ' + _PHRASE_RE + r", \d+ (?:also )?" + _PHRASE_RE + r'\."')


def condition_and_target(text: str):
    """(condition, target) event names, read from the sentence by its marker."""
    if m := GIVEN.search(text):
        return LEXICON[m.group(2)], LEXICON[m.group(1)]
    if m := KNOWING.search(text):
        return LEXICON[m.group(1)], LEXICON[m.group(2)]
    if m := IFTHEN.search(text):
        return LEXICON[m.group(1)], LEXICON[m.group(2)]
    if m := OF.search(text):
        return LEXICON[m.group(1)], LEXICON[m.group(2)]
    raise AssertionError(f"no template matches: {text}")


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_the_target_given_the_condition(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        cond, target = condition_and_target(q.text)
        assert cond != target
        assert q.answer_text == f"the probability of {target}, given {cond}", (tier, seed, q.text)
        assert f"the probability of {cond}, given {target}" in q.distractors
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_exactly_one_distractor_of_each_expected_misconception(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        cond, target = condition_and_target(q.text)
        notes = q.option_notes
        assert notes[f"the probability of {cond}, given {target}"] == "swaps-condition-and-target"
        assert notes[f"the probability of {target} and {cond} together"] == "reads-conditional-as-both-events"
        third = [d for d in q.distractors if "together" not in d and "given" not in d]
        assert len(third) == 1
        assert notes[third[0]] in ("drops-the-condition", "reports-the-condition-as-the-target")
        assert third[0] in (f"the probability of {target}", f"the probability of {cond}")


def test_hard_counts_line_is_consistent_with_the_sentence():
    tier = "hard"
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        total, c, t, b = map(
            int, re.search(r"survey of (\d+) .*?, (\d+) .*?, (\d+) .* and (\d+) do both", q.text).groups()
        )
        assert b < c and b < t and c != t and c + t - b <= total
        m = OF.search(q.text)
        if m:
            nums = re.findall(r"\d+", q.text.split("Read this sentence:")[1])
            assert (int(nums[0]), int(nums[1])) == (c, b)
        m = IFTHEN.search(q.text)
        if m:
            assert f"is {b}/{c}." in q.text
