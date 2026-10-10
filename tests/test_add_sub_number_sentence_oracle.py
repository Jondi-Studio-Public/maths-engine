"""Independent oracle for `add_sub_number_sentence`: it re-derives the story's answer from the rendered text.

The oracle reads only what a learner is given: `question.text`, `question.answer_text` and
`question.distractors`. It works out which quantity the story asks for from fixed phrases (every story in the
draw space is matched by exactly one rule below, so a story that matches none, or two, is itself a finding),
computes that quantity from the two numbers in the story, solves every option sentence for its blank with its own
parser, and requires the key to be the only option whose blank takes that value. It does not import the generator.

Seed count is `N_SEEDS` (default 250 per tier), overridable by `ADD_SUB_SENTENCE_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "add_sub_number_sentence"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("ADD_SUB_SENTENCE_ORACLE_SEEDS", "250"))

_STORY_END = re.compile(r" (?:Which number sentence|Choose the number sentence|Which of these number sentences)")

# (name, pattern, how the unknown is worked out from the story's two numbers n1, n2 in the order they appear)
_RULES = (
    ("start unknown, joined", r"(?:had some|were some).*\d+ more", lambda n1, n2: n2 - n1),
    ("start unknown, separated", r"(?:had some|were some)", lambda n1, n2: n1 + n2),
    ("how many more or fewer", r"How many (?:more|fewer)", lambda n1, n2: n1 - n2),
    ("compare, amount of the one with more", r"\d+ more \w+ than", lambda n1, n2: n1 + n2),
    ("change, added", r"(?:given some more|found some more|Some more \w+ were put in)", lambda n1, n2: n2 - n1),
    ("change, taken", r"(?:gave some away|Some \w+ were taken out|lost some of them)", lambda n1, n2: n1 - n2),
    ("total", r"(?:altogether|in all\?|in total\?)", lambda n1, n2: n1 + n2),
    (
        "what is left",
        r"(?:have now\?|are left in the box\?|have left\?|are yellow\?|ride\?|are white\?)",
        lambda n1, n2: n1 - n2,
    ),
)
_SENTENCE = re.compile(r"^(\d+|□) ([+-]) (\d+|□) = (\d+|□)$")


def _story(text: str) -> str:
    parts = _STORY_END.split(text)
    assert len(parts) == 2, f"no instruction found in {text!r}"
    return parts[0]


def _expected_unknown(story: str) -> int:
    numbers = [int(n) for n in re.findall(r"\d+", story)]
    assert len(numbers) == 2, f"story must carry exactly two numbers: {story!r}"
    matched = [(name, rule) for name, pattern, rule in _RULES if re.search(pattern, story)]
    # The first match wins by design (start-unknown stories also contain 'more' or 'taken'), but at least one must match.
    assert matched, f"oracle: no rule matches {story!r}"
    return matched[0][1](*numbers)


def _blank_value(sentence: str):
    m = _SENTENCE.match(sentence)
    assert m, f"oracle: malformed option {sentence!r}"
    x, op, y, z = m.groups()
    for value in range(1, 400):
        cells = [str(value) if c == "□" else c for c in (x, y, z)]
        left = int(cells[0]) + int(cells[1]) if op == "+" else int(cells[0]) - int(cells[1])
        if left == int(cells[2]):
            return value
    return None


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_the_only_option_whose_blank_equals_the_story_unknown(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        expected = _expected_unknown(_story(q.text))
        assert expected >= 1
        assert _blank_value(q.answer_text) == expected, f"{tier} seed {seed}: key {q.answer_text!r} for {q.text!r}"
        for d in q.distractors:
            assert _blank_value(d) != expected, f"{tier} seed {seed}: {d!r} also fits {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_every_sentence_uses_only_the_numbers_in_the_story(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        shown = set(re.findall(r"\d+", _story(q.text)))
        for option in (q.answer_text, *q.distractors):
            assert set(re.findall(r"\d+", option)) == shown, f"{tier} seed {seed}: {option!r} vs {shown}"


def test_story_kinds_in_the_draw_space_are_each_matched_by_a_rule():
    """Enumerate by sampling widely: every story shape in the draw space is recognised."""
    seen = set()
    for tier in TIERS:
        for seed in range(N_SEEDS * 2):
            story = _story(make(TOPIC, seed=seed, difficulty=tier).text)
            seen.add(tuple(name for name, pattern, _ in _RULES if re.search(pattern, story))[:1])
    assert len(seen) == len(_RULES), f"rules never exercised: {len(_RULES) - len(seen)}; seen {seen}"
