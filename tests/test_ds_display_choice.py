"""Tests for `f10-display-choice-then-read` (AC9M9ST04, `answer: number`).

Each key is re-derived by an oracle that reads only the rendered question text, never the generator's variables.
Every distractor must grade wrong through `mathgen.engine.grade`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

import mathgen.generators.ds_display_choice_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

SKILL_ID = "f10-display-choice-then-read"
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
TIERS = ("easy", "medium", "hard")
SEEDS = range(120)
MONTHS = "January February March April May June July August September October November December".split()


def one(tier: str, seed: int):
    src = next(s for s in SKILL.sources if s.difficulty == tier)
    return make(src.topic, seed=seed, difficulty=tier, **src.params)


def ints(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def month_span(text: str) -> int:
    a, b = re.search(r"from (\w+) to (\w+)", text).groups()
    return MONTHS.index(b) - MONTHS.index(a) + 1


def ceil_to(value: int, step: int) -> int:
    return -(-value // step) * step


# -- oracles ------------------------------------------------------------------------------------------------------


def oracle_easy(text: str) -> int:
    if "shared among the answers" in text:
        listing = text.split(": ", 1)[1].split(". The aim")[0]
        return len(listing.split(", "))
    if "change over time" in text:
        return month_span(text) * (2 if text.startswith("Two ") else 1)
    if "side by side" in text:
        answers = text.split("The answers were ")[1].split(". The aim")[0].split(", ")
        groups = 3 if re.match(r"[^.]*?, [^.]*? and ", text.split(" were each asked")[0]) else 2
        return len(answers) * groups
    if "related" in text:
        return int(re.match(r"For each of (\d+)", text).group(1))
    raise AssertionError(text)


def oracle_medium(text: str) -> int:
    if "angle" in text and "%" not in text:
        counts = [int(c) for c in re.findall(r"(\d+) [a-z ]+(?:,|$)", text.split(": ", 1)[1].split(". The aim")[0])]
        total = int(re.match(r"A survey of (\d+)", text).group(1))
        assert sum(counts) == total
        return 360 * max(counts) // total
    if "angle" in text:
        pct = int(re.search(r"(\d+)% gave", text).group(1))
        return 360 * pct // 100
    if "columns" in text:
        lo, hi = map(int, re.search(r"range from (\d+) to (\d+)", text).groups())
        w = int(re.search(r"classes of width (\d+)", text).group(1))
        return hi // w - lo // w + 1
    if "counting axis" in text:
        counts = [int(c) for c in re.findall(r"(\d+) [a-z ]+(?:,|\. The aim)", text.split("counts: ", 1)[1])]
        step = int(re.search(r"steps of (\d+)", text).group(1))
        return ceil_to(max(counts), step)
    if "segments" in text:
        return month_span(text) - 1
    raise AssertionError(text)


def oracle_hard(text: str) -> int:
    if "stems" in text:
        nums = ints(text.split("recorded. ", 1)[1].split(". The aim")[0])
        # group names carry no digits except "Class 9A"; drop that digit
        nums = [n for n in nums if n >= 10]
        return max(nums) // 10 - min(nums) // 10 + 1
    if "shirt numbers" in text:
        return len(text.split("numbers: ", 1)[1].split(". The")[0].split(", "))
    if "distribution of this measured" in text:
        nums = ints(text.split(": ", 1)[1].split(". The aim")[0])
        w = int(re.search(r"width (\d+)", text).group(1))
        return max(nums) // w - min(nums) // w + 1
    if "others" in text:
        shares = [
            int(p) for p in re.findall(r"(\d+)% [a-z ]+", text.split("the answers were ")[1].split(", and everyone")[0])
        ]
        return 360 * (100 - sum(shares)) // 100
    if "vertical axis" in text:
        v0 = int(re.search(r"was (\d+)\. In the months", text).group(1))
        changes = [int(c.replace("−", "-")) for c in re.findall(r"[A-Z][a-z]{2} ([+−]\d+)", text)]
        series = [v0]
        for c in changes:
            series.append(series[-1] + c)
        step = int(re.search(r"steps of (\d+)", text).group(1))
        return ceil_to(max(series), step)
    raise AssertionError(text)


ORACLES = {"easy": oracle_easy, "medium": oracle_medium, "hard": oracle_hard}


# -- wiring -------------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert SKILL.status == "active" and SKILL.answer == "number"
    assert [s.difficulty for s in SKILL.sources] == list(TIERS)
    assert all(s.topic.startswith("ds_display_choice_") for s in SKILL.sources)
    assert {"AC9M9ST04", "acara-f10-v9"} <= set(SKILL.tags)
    listed = {m.id for m in SKILL.misconceptions}
    used = {tag for t in TIERS for s in SEEDS for tag in one(t, s).option_notes.values()}
    assert used <= listed and used, used - listed
    assert listed <= set(CATALOGUE)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in range(10):
        assert one(tier, seed).to_dict() == one(tier, seed).to_dict()


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_text_oracle_and_distractors_grade_wrong(tier):
    for seed in SEEDS:
        q = one(tier, seed)
        assert int(q.answer_text) == ORACLES[tier](q.text), (q.text, q.answer_text)
        assert grade(SKILL, q, q.answer_text).correct, (q.text, q.answer_text)
        assert len(set(q.distractors)) == 3
        for d in q.distractors:
            assert not grade(SKILL, q, d).correct, (q.text, q.answer_text, d)
            assert q.option_notes.get(d) in CATALOGUE, (q.text, d)
        assert find_text_defects(q) == [], (q.text, find_text_defects(q))


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_three_prompt_shapes_and_varied_answers(tier):
    shapes = {prompt_shape(one(tier, s)) for s in range(200)}
    assert len(shapes) >= 3, len(shapes)
    answers = Counter(one(tier, s).answer_text for s in range(200))
    assert len(answers) >= 5


def test_display_is_never_named():
    banned = re.compile(r"pie chart|histogram|line graph|column graph|scatterplot|stem-and-leaf|bar graph", re.I)
    for tier in TIERS:
        for seed in SEEDS:
            assert not banned.search(one(tier, seed).text), one(tier, seed).text


def test_hard_label_and_measure_variants_both_occur_and_differ():
    seen = Counter()
    for seed in range(200):
        q = one("hard", seed)
        if "shirt numbers" in q.text:
            seen["labels"] += 1
        elif "distribution of this measured" in q.text:
            seen["measures"] += 1
    assert seen["labels"] >= 5 and seen["measures"] >= 5


def test_histogram_boundary_trap_occurs():
    boundary = 0
    for seed in range(300):
        q = one("medium", seed)
        if "columns" in q.text:
            lo, hi = map(int, re.search(r"range from (\d+) to (\d+)", q.text).groups())
            w = int(re.search(r"classes of width (\d+)", q.text).group(1))
            boundary += hi % w == 0
    assert boundary >= 3


def test_unique_largest_sector():
    for seed in SEEDS:
        q = one("medium", seed)
        if "most common answer" in q.text:
            counts = [
                int(c) for c in re.findall(r"(\d+) [a-z ]+(?:,|$)", q.text.split(": ", 1)[1].split(". The aim")[0])
            ]
            assert counts.count(max(counts)) == 1
