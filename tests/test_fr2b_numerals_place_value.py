"""Tests `f10-numerals-words-and-place-value` (topic `fr2b_numerals_place_value`).

The oracle reads the number words, the numeral or the stated change back out of the question text and rebuilds the
key with its own words parser and place arithmetic.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.fr2b_numerals_place_value_build  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "fr2b_numerals_place_value"
SKILL_ID = "f10-numerals-words-and-place-value"
TIERS = ("easy", "medium", "hard")
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen".split()
    )
}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split())}
_WORDS_RE = re.compile(r"(?:[a-z]+(?:-[a-z]+)?(?:,? and| thousand| hundred|,)? ?)+")


def words_to_number(words: str) -> int:
    """An independent parser for Australian number words below one million."""
    total, current = 0, 0
    for token in re.findall(r"[a-z]+", words.lower().replace("-", " ")):
        if token in _UNITS:
            current += _UNITS[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token == "hundred":
            current *= 100
        elif token == "thousand":
            total, current = total + current * 1000, 0
    return total + current


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def test_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [(TOPIC, t) for t in TIERS]
    assert all(s.params == {"tier": s.difficulty} for s in SKILL.sources)


def test_easy_key_is_the_numeral_for_the_words():
    zeros = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        words = re.search(r"(?:Write|says|number) (.+?)(?: people| in numerals| using digits|\.)", q.text).group(1)
        assert words_to_number(words) == q.answer, (seed, q.text)
        assert len(str(q.answer)) == 5 and "0" in str(q.answer)[1:], q.answer
        zeros[str(q.answer)[2:].count("0")] += 1
    assert len(zeros) >= 2, zeros


def test_medium_key_is_the_value_of_the_digit():
    places = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        numeral = re.search(r"\d{3} \d{3}", q.text).group(0).replace(" ", "")
        digit = re.search(r"digit (\d)", q.text).group(1)
        assert numeral.count(digit) == 1, q.text
        position = len(numeral) - 1 - numeral.index(digit)
        assert q.answer == int(digit) * 10**position, (seed, q.text)
        places[position] += 1
    assert set(places) == {1, 2, 3, 4, 5}, places


def test_hard_key_regroups_across_the_changed_place():
    steps = {"ten": 10, "one hundred": 100, "one thousand": 1000, "ten thousand": 10000}
    regrouped = 0
    for seed in range(150):
        q = _q("hard", seed)
        m = re.search(r"(ten thousand|one thousand|one hundred|ten) (more|less)", q.text)
        step, sign = steps[m.group(1)], 1 if m.group(2) == "more" else -1
        start = words_to_number(re.search(r"(?:than|Start with) (.+?)(?:\?|\.)", q.text).group(1))
        assert q.answer == start + sign * step, (seed, q.text)
        digit = (start // step) % 10
        assert digit == (9 if sign == 1 else 0), (seed, q.text)
        regrouped += 1
    assert regrouped == 150


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    tags = set()
    for seed in range(120):
        q = _q(tier, seed)
        assert grade(SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert grade(SKILL, q, f"{q.answer:,}".replace(",", " ")).correct
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(SKILL, q, text).correct, (seed, text, q.answer_text)
            assert "." not in text
            assert q.option_notes[text] in CATALOGUE
            tags.add(q.option_notes[text])
    assert len(tags) >= 3, tags


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_tiers_ask_different_things():
    shapes = {tier: {_q(tier, s).text.split()[0] for s in range(60)} for tier in TIERS}
    assert not shapes["medium"] & {"Write", "A"}, shapes
    assert len({_q("easy", s).text for s in range(60)}) >= 55
    assert len({_q("medium", s).answer for s in range(60)}) >= 25
