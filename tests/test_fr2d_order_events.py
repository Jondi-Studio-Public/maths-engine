"""Tests for `f10-order-events-independent-dependent`: orderings and independence are re-derived from the stem."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.engine import grade
from tests.fr2d_kit import SEEDS, check_determinism, check_options, check_text, check_tiers, question, skill

SKILL = "f10-order-events-independent-dependent"

_DIE = {
    "an even number": 3,
    "a number greater than 4": 2,
    "a 6": 1,
    "a number less than 5": 4,
    "a number that is not 1": 5,
    "a multiple of 3": 2,
    "a 7": 0,
    "a number less than 7": 6,
}
_CARDS = {
    "a multiple of 5": 4,
    "a multiple of 4": 5,
    "greater than 15": 5,
    "less than 4": 3,
    "an even number": 10,
    "a prime number": 8,
}


def _events(stem: str) -> dict[str, str]:
    body = re.search(r"Events?: (.*?)\. List", stem) or re.search(r"four events\. (.*?)\. List", stem)
    return {item[0]: item[3:] for item in body.group(1).split("; ")}


def _chance(text: str) -> Fraction:
    if m := re.match(r"drawing a \w+ counter from a bag of (\d+) counters, (\d+) of which (?:is|are)", text):
        return Fraction(int(m.group(2)), int(m.group(1)))
    if m := re.match(r"rolling (.+) on a fair six-sided die", text):
        return Fraction(_DIE[m.group(1)], 6)
    if m := re.match(r"spinning a spinner with (\d+) equal sections and stopping on one of its (\d+) ", text):
        return Fraction(int(m.group(2)), int(m.group(1)))
    if text.startswith("tossing a fair coin"):
        return Fraction(1, 2)
    if m := re.match(r"choosing a card at random from cards numbered 1 to 20 and it is (.+)", text):
        return Fraction(_CARDS[m.group(1)], 20)
    if m := re.match(r"choosing a student at random from a class of (\d+), (\d+) of whom", text):
        return Fraction(int(m.group(2)), int(m.group(1)))
    raise AssertionError(text)


def oracle_order(stem: str) -> list[str]:
    events = _events(stem)
    if stem.startswith("A bag holds"):
        by_colour = {c: int(n) for n, c in re.findall(r"(\d+) (\w+)", stem.split(". ")[0])}
        probs = {k: Fraction(by_colour[re.search(r"is (\w+)$", v).group(1)]) for k, v in events.items()}
    else:
        probs = {k: _chance(v) for k, v in events.items()}
    assert len(set(probs.values())) == len(probs), "tied events"
    return sorted(probs, key=lambda k: probs[k])


def _independent(pair: str) -> bool:
    if "puts it back" in pair or "fair coin is tossed twice" in pair or "and a fair coin is tossed" in pair:
        return True
    if "Two spinners" in pair:
        return True
    if "keeps it" in pair or "not put back" in pair:
        return False
    if "rolled once" in pair or "chosen at random from 1 to 10" in pair:
        return False
    raise AssertionError(pair)


def oracle_pairs(stem: str) -> list[str]:
    pairs = re.findall(r"Pair ([A-D]): (.*?)\.(?= Pair| Which)", stem)
    assert len(pairs) == 4
    return [letter for letter, text in pairs if _independent(text)]


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_ordering_matches_the_events(tier):
    for seed in SEEDS:
        q = question(SKILL, tier, seed)
        assert q.answer_kind == "ordered_list"
        assert q.answer == oracle_order(q.text), (seed, q.text)
        assert q.answer_text == ", ".join(oracle_order(q.text))


def test_independence_matches_the_mechanism():
    kinds = set()
    for seed in SEEDS:
        q = question(SKILL, "hard", seed)
        assert q.answer == oracle_pairs(q.text), (seed, q.text)
        kinds.add(len(q.answer))
    assert kinds == {1, 2, 3}


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_options_text_and_determinism(tier):
    check_options(SKILL, tier, min_options=2)
    check_text(SKILL, tier)
    check_determinism(SKILL, tier)


def test_tiers_differ():
    check_tiers(SKILL)


def test_a_reply_in_the_wrong_order_is_marked_wrong():
    sk = skill(SKILL)
    for seed in SEEDS:
        q = question(SKILL, "medium", seed)
        reply = ", ".join(reversed(q.answer))
        assert not grade(sk, q, reply).correct
        assert grade(sk, q, " , ".join(q.answer)).correct


def test_medium_events_cannot_be_ordered_by_favourable_count_alone():
    for seed in SEEDS:
        q = question(SKILL, "medium", seed)
        events = _events(q.text)
        probs = {k: _chance(v) for k, v in events.items()}
        assert len({v.denominator for v in probs.values()}) >= 2
        assert all(abs(a - b) >= Fraction(1, 20) for a in probs.values() for b in probs.values() if a != b)
