"""`f10-tessellation-vertex-angles`: topic `fr2a_tessellation_vertex` (easy, medium, hard)."""

from __future__ import annotations

import re
from fractions import Fraction

import fr2a_support as sup
import pytest

SKILL = "f10-tessellation-vertex-angles"
TOPIC = "fr2a_tessellation_vertex"
SEEDS = range(150)

NAMES = {
    "equilateral triangle": 3,
    "square": 4,
    "regular pentagon": 5,
    "regular hexagon": 6,
    "regular heptagon": 7,
    "regular octagon": 8,
    "regular nonagon": 9,
    "regular decagon": 10,
    "regular dodecagon": 12,
}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def interior(n):
    """Interior angle by the exterior-angle rule (the generator's first route is the angle sum)."""
    return 180 - Fraction(360, n)


def polygons_in(text):
    """(count, sides) for every polygon the stem names, in the order named (a, an or no word counts as one)."""
    found = []
    names = "|".join(sorted(NAMES, key=len, reverse=True))
    pattern = rf"(?:\b(a|an|one|two|three|four|five|six) )?\b({names})s?\b|regular polygon with (\d+) sides"
    for m in re.finditer(pattern, text.lower()):
        found.append((WORDS.get(m[1], 1), int(m[3]) if m[3] else NAMES[m[2]]))
    return found


def test_wiring():
    sup.assert_wired(SKILL, TOPIC, None, "number", False)


def test_contract_all_tiers():
    sup.assert_single_answer_contract(SKILL, TOPIC, SEEDS, reader=False)


def test_easy_questions_are_each_of_three_kinds():
    kinds = set()
    for seed in SEEDS:
        question = sup.q(TOPIC, "easy", seed)
        text = question.text
        if "interior angle" in text:
            kinds.add("interior")
            ((_, n),) = polygons_in(text)[:1]
            assert int(question.answer_text) == interior(n)
        elif "Three tiles" in text or "three tile corners" in text:
            kinds.add("missing")
            a, b = (int(v) for v in re.findall(r"(\d+)°", text)[:2])
            assert int(question.answer_text) == 360 - a - b and 60 <= int(question.answer_text) < 180
        else:
            kinds.add("count")
            ((_, n),) = polygons_in(text)[:1]
            assert int(question.answer_text) * interior(n) == 360
    assert kinds == {"interior", "missing", "count"}


def test_medium_the_third_polygon_completes_the_point():
    for seed in SEEDS:
        question = sup.q(TOPIC, "medium", seed)
        known = polygons_in(question.text)
        sides = [n for count, n in known for _ in range(count)]
        assert len(sides) == 2, (seed, question.text)
        n = int(question.answer_text)
        assert sum(interior(k) for k in sides) + interior(n) == 360, (seed, question.text)


def test_hard_identical_polygons_and_gap_questions():
    kinds = set()
    for seed in SEEDS:
        question = sup.q(TOPIC, "hard", seed)
        text = question.text
        if "identical" in text or "all the same" in text:
            kinds.add("identical")
            m = re.search(r"(one|two|three|four|five|six) (?:more )?(?:identical )?regular polygons", text)
            copies = WORDS[m[1]]
            known = polygons_in(text[: m.start()])
            used = sum(c * interior(n) for c, n in known)
            n = int(question.answer_text)
            assert used + copies * interior(n) == 360, (seed, text)
            assert copies >= 2
        else:
            kinds.add("gap")
            copies = WORDS[text.split()[0].lower()]
            ((_, n),) = polygons_in(text)[:1]
            gap = int(question.answer_text)
            assert gap == 360 - copies * interior(n) and gap > 0 and copies * interior(n) + interior(n) > 360, (
                seed,
                text,
            )
    assert kinds == {"identical", "gap"}


def test_tiers_differ_in_shape():
    first = {t: {re.sub(r"\d+", "N", sup.q(TOPIC, t, s).text.split("?")[0]) for s in SEEDS} for t in sup.TIERS}
    assert first["easy"].isdisjoint(first["medium"]) and first["medium"].isdisjoint(first["hard"])


def test_answer_variety():
    for tier in sup.TIERS:
        assert len({sup.q(TOPIC, tier, s).text for s in SEEDS}) >= (20 if tier != "easy" else 12), tier
