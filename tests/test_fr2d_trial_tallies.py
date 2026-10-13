"""Tests for `f10-trial-tallies-compare-outcomes`: every key is re-derived from the tallies printed in the stem."""

from __future__ import annotations

import re

import pytest

from tests.fr2d_kit import SEEDS, check_determinism, check_options, check_text, check_tiers, question

SKILL = "f10-trial-tallies-compare-outcomes"
GATE = "||||/"


def _count(tally: str) -> int:
    return sum(5 if g == GATE else len(g) for g in tally.split())


def _table(stem: str) -> dict[str, int]:
    body = re.search(r"The tally of the results: (.*?)\. ", stem).group(1)
    out = {}
    for item in body.split("; "):
        cat, tally = item.split(": ")
        out[cat] = _count(tally)
    return out


def _numbers(stem: str, label: str) -> dict[str, int]:
    body = re.search(rf"{label}: (.*?)\. ", stem).group(1)
    return {cat: int(n) for cat, n in (item.rsplit(" ", 1) for item in body.split(", "))}


def oracle_easy(stem: str) -> int:
    t = _table(stem)
    if "most often come up" in stem or "most common result came up" in stem:
        return max(t.values())
    if "fewer times" in stem or "gap between" in stem:
        return max(t.values()) - min(t.values())
    ask = stem.split("results: ")[1].split(". ", 1)[1]
    if m := re.search(r"How many more times did (.+) come up than (.+)\?", ask):
        return t[m.group(1)] - t[m.group(2)]
    if m := re.search(r"(.+) came up more often than (.+)\. How many more times\?", ask):
        return t[m.group(1)] - t[m.group(2)]
    if m := re.search(r"How many times did (.+) come up\?", ask):
        return t[m.group(1)]
    if m := re.search(r"How many (.+) results were there\?", ask):
        return t[m.group(1)]
    raise AssertionError(stem)


def oracle_medium(stem: str) -> int:
    t = _table(stem)
    n = int(re.search(r"(\d+) times", stem).group(1))
    assert sum(t.values()) == n
    share = n // len(t)
    word, x = re.search(r"How many (more|fewer) times than that(?: count)? did (.+) come up\?", stem).groups()
    diff = t[x] - share
    assert (diff > 0) == (word == "more")
    return abs(diff)


def oracle_hard(stem: str) -> int:
    first = _numbers(stem, r"After \d+ \w+s")
    total = _numbers(stem, r"After \d+ \w+s altogether")
    extra = {c: total[c] - first[c] for c in first}
    assert all(v > 0 for v in extra.values())
    if m := re.search(r"How many of the extra \d+ \w+s were (.+)\?", stem):
        return extra[m.group(1)]
    if m := re.search(r"how many more times did (.+) come up than (.+)\?", stem):
        return extra[m.group(1)] - extra[m.group(2)]
    t = sum(total.values())
    assert "equally likely" in stem
    return max(total.values()) - t // len(total)


ORACLES = {"easy": oracle_easy, "medium": oracle_medium, "hard": oracle_hard}


@pytest.mark.parametrize("tier", ORACLES)
def test_key_matches_the_printed_tallies(tier):
    for seed in SEEDS:
        q = question(SKILL, tier, seed)
        assert q.answer_text == str(ORACLES[tier](q.text)), (seed, q.text, q.answer_text)
        assert q.answer_kind == "number"


@pytest.mark.parametrize("tier", ORACLES)
def test_options_text_and_determinism(tier):
    check_options(SKILL, tier, min_options=3)
    check_text(SKILL, tier)
    check_determinism(SKILL, tier)


def test_tiers_differ_and_use_no_fractions():
    check_tiers(SKILL)
    for tier in ORACLES:
        for seed in SEEDS:
            q = question(SKILL, tier, seed)
            words = q.text.replace(GATE, "")  # the diagonal of a tally group is not a fraction slash
            assert "/" not in words and "fraction" not in words and "%" not in words


def test_hard_tier_asks_more_than_one_kind_of_question():
    asked = set()
    for seed in SEEDS:
        stem = question(SKILL, "hard", seed).text
        asked.add("extra" if "of the extra" in stem else "gap" if "only, how many more" in stem else "share")
    assert asked == {"extra", "gap", "share"}


def test_a_wrong_reading_of_a_tally_is_a_named_distractor():
    seen = set()
    for seed in SEEDS:
        q = question(SKILL, "easy", seed)
        seen |= set(q.option_notes.values())
    assert {"fr2d-tally-group-as-one", "fr2d-adds-instead-of-compares"} <= seen
