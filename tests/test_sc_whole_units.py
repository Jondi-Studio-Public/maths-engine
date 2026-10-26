"""Tests for `whole-units-needed-round-up-down`: the practice tiers (`sc_whole_units`) and the exam chain
(`sc_whole_units_chain`).

The oracle reads the numbers and the situation back out of the rendered text with its own regular expressions and
recomputes the count, the cost or the leftover with plain integer arithmetic (and a brute-force search for the
hard tier), never with the generator's helpers.
"""

from __future__ import annotations

import itertools
import random
import re
from fractions import Fraction

import pytest

import r4e_support as t
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from r4e_support import sup

SKILL = "whole-units-needed-round-up-down"
CHAIN = "sc_whole_units_chain"
TOPIC = "sc_whole_units"


def ints(text):
    return [int(v) for v in re.findall(r"\d+", text)]


def nearest(total, size):
    return int(Fraction(total, size) + Fraction(1, 2))  # no ties occur, and the value is positive


# -- reading a story back out ---------------------------------------------------------------------------------------

_STORIES = (
    # (pattern, total from the groups, size index, rounds up)
    (
        r"paint (\d+) identical walls, each (\d+) m wide and (\d+) m high, with (\d+) coats\. One tin covers (\d+)",
        lambda g: g[0] * g[1] * g[2] * g[3],
        4,
        True,
    ),
    (r"lawn measures (\d+) m by (\d+) m\. Rolls of turf each cover (\d+)", lambda g: g[0] * g[1], 2, True),
    (r"year groups of (\d+), (\d+) and (\d+) students .* seats (\d+) students", lambda g: g[0] + g[1] + g[2], 3, True),
    (r"delivers (\d+) bags of mulch and (\d+) bags of soil .* carries (\d+) bags", lambda g: g[0] + g[1], 2, True),
    (
        r"tiling (\d+) identical rooms, each (\d+) m by (\d+) m\. A pack of tiles covers (\d+)",
        lambda g: g[0] * g[1] * g[2],
        3,
        True,
    ),
    (r"camp has (\d+) boys and (\d+) girls\. They are put into teams of (\d+)", lambda g: g[0] + g[1], 2, False),
    (
        r"fills a tank at (\d+) litres per minute and runs for (\d+) minutes\..* hold (\d+) litres",
        lambda g: g[0] * g[1],
        2,
        False,
    ),
    (
        r"makes (\d+) trays of cookies with (\d+) cookies on each tray\. .* packed (\d+) to a box",
        lambda g: g[0] * g[1],
        2,
        False,
    ),
    (
        r"earns \$(\d+) an hour for (\d+) hours .* \$(\d+) bonus\. Show tickets cost \$(\d+)",
        lambda g: g[0] * g[1] + g[2],
        3,
        False,
    ),
    (
        r"shelf is (\d+) centimetres long and already holds (\d+) books that are each (\d+) centimetres thick\. "
        r"More books, each (\d+) centimetres thick",
        lambda g: g[0] - g[1] * g[2],
        3,
        False,
    ),
)


def read_story(text):
    for pattern, total, size_at, up in _STORIES:
        m = re.search(pattern, text)
        if m:
            g = [int(v) for v in m.groups()]
            return total(g), g[size_at], up
    raise AssertionError(f"unrecognised story: {text}")


def ask_of(text):
    tail = re.sub(r"Give your answer.*$", "", text)
    if re.search(r"left over|unused|empty|not in a full team|left unboxed|dollars does Sam have left", tail):
        return "left"
    if re.search(r"total (?:cost|hire cost|entry fee|value|takings)|total cost", tail):
        return "cost"
    return "count"


def price_of(text):
    m = re.search(r"Each [a-z ]+? (?:costs|sells for|pays an entry fee of) \$(\d+)", text)
    return int(m.group(1)) if m else None


def price_in_chain(q):
    """The unit price of part (c); for the ticket story it is the ticket price, the pack size."""
    price = price_of(q.parts[2].prompt)
    return price if price is not None else read_story(q.text)[1]


def whole(total, size, up):
    return -(-total // size) if up else total // size


# -- easy ---------------------------------------------------------------------------------------------------------------

_EASY_UP = re.compile(r"are needed|must be bought|does she need|trips are needed|tables are needed")


def easy_oracle(q):
    total, size = ints(q.text)[:2]
    up = bool(_EASY_UP.search(q.text))
    return str(whole(total, size, up))


# -- medium -------------------------------------------------------------------------------------------------------------


def medium_oracle(q):
    total, size, up = read_story(q.text)
    ask = ask_of(q.text)
    n = whole(total, size, up)
    if ask == "count":
        return str(n)
    if ask == "cost":
        return str(n * price_of(q.text))
    return str(n * size - total if up else total - n * size)


# -- hard ---------------------------------------------------------------------------------------------------------------


def hard_oracle(q):
    text = q.text
    nums = ints(re.sub(r"Give your answer.*$", "", text))
    if "at least" in text or "least total cost" in text or "cheapest" in text:
        need, s1, p1, s2, p2 = nums
        best = min(
            a * p1 + b * p2
            for a, b in itertools.product(range(need // s1 + 2), range(need // s2 + 2))
            if a * s1 + b * s2 >= need
        )
        return str(best)
    if re.search(r"Each (?:cake|kit|roll|box) (?:needs|uses)", text):
        a, b, A, B = nums
        return str(min(A // a, B // b))
    if "gap" in text or "between each pair" in text or "between neighbouring" in text:
        length, w, g = nums
        return str(max(m for m in range(1, 300) if m * w + (m - 1) * g <= length))
    # cutting: the order of the three numbers depends on the wording
    if text.startswith("A tailor"):
        need, p, board = nums
    else:
        p, board, need = nums
    per = board // p
    boards = 0
    while per * boards < need:
        boards += 1
    return str(boards)


# -- the tiers ----------------------------------------------------------------------------------------------------------


def test_easy():
    t.assert_tier(SKILL, "easy", seeds=range(100), oracle=easy_oracle)


def test_medium():
    t.assert_tier(SKILL, "medium", seeds=range(100), oracle=medium_oracle)


def test_hard():
    t.assert_tier(SKILL, "hard", seeds=range(100), oracle=hard_oracle)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=20, top=0.12)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_prompt_variety(tier):
    from mathgen.text_quality import prompt_shape

    shapes = {prompt_shape(t.question(SKILL, tier, s)) for s in range(120)}
    assert len(shapes) >= 3, (tier, len(shapes))


def test_nearest_whole_number_is_never_the_key():
    """The common wrong value is the quotient rounded to the nearest whole number: it must differ from the key."""
    for seed in range(120):
        q = t.question(SKILL, "easy", seed)
        total, size = ints(q.text)[:2]
        assert nearest(total, size) != int(q.answer_text), (seed, q.text)
        q = t.question(SKILL, "medium", seed)
        total, size, up = read_story(q.text)
        if ask_of(q.text) == "count":
            assert nearest(total, size) != int(q.answer_text), (seed, q.text)


def test_direction_is_balanced_so_the_topic_does_not_decide_it():
    ups = {"easy": 0, "medium": 0}
    for seed in range(120):
        ups["easy"] += bool(_EASY_UP.search(t.question(SKILL, "easy", seed).text))
        ups["medium"] += read_story(t.question(SKILL, "medium", seed).text)[2]
    for tier, count in ups.items():
        assert 45 <= count <= 80, (tier, count)


def test_every_tagged_wrong_answer_is_catalogued_and_listed_in_the_skill_or_catalogue():
    listed = {m["id"] for m in sup.skill_yaml(SKILL)["misconceptions"]}
    assert listed <= set(CATALOGUE)
    seen = set()
    for tier in ("easy", "medium", "hard"):
        for seed in range(100):
            q = t.question(SKILL, tier, seed)
            seen |= set(q.option_notes.values())
    for seed in range(100):
        for part in sup.exam_question(CHAIN, seed).parts:
            seen |= set(part.option_notes.values())
    assert seen <= set(CATALOGUE)
    assert all(i.startswith("wu-") or i == "near-miss-value" for i in seen)
    assert {"wu-rounds-to-nearest", "wu-rounds-down-when-covering", "wu-rounds-up-when-only-full"} <= seen


def test_stems_say_nothing_about_rounding_and_state_the_precision():
    for tier in ("easy", "medium", "hard"):
        for seed in range(120):
            q = t.question(SKILL, tier, seed)
            assert "round" not in q.text.lower(), (tier, seed, q.text)
            assert "whole number" in q.text, (tier, seed, q.text)
            assert "*" not in q.text and "**" not in q.text, (tier, seed)
            assert not re.search(r"(?<![\w.$])1 [a-z]{3,}s\b", q.text), (tier, seed, q.text)
    for seed in range(80):
        q = sup.exam_question(CHAIN, seed)
        assert "round" not in q.text.lower() and "*" not in q.text, (seed, q.text)


def test_hard_tier_covers_all_four_traps():
    kinds = set()
    for seed in range(120):
        text = t.question(SKILL, "hard", seed).text
        if "least" in text and "cost" in text or "cheapest" in text or "lowest possible" in text:
            kinds.add("packs")
        elif re.search(r"Each (?:cake|kit|roll|box) (?:needs|uses)", text):
            kinds.add("limit")
        elif "gap" in text or "between each pair" in text or "between neighbouring" in text:
            kinds.add("gap")
        else:
            kinds.add("cut")
    assert kinds == {"packs", "limit", "gap", "cut"}


def test_cheapest_combination_beats_each_single_pack_size():
    """The packs question is only a search problem when a mix beats buying one size alone."""
    for seed in range(100):
        q = t.question(SKILL, "hard", seed)
        if not ("least" in q.text and "cost" in q.text or "lowest possible" in q.text):
            continue
        need, s1, p1, s2, p2 = ints(re.sub(r"Give your answer.*$", "", q.text))
        only_large = -(-need // s2) * p2
        only_small = -(-need // s1) * p1
        assert int(q.answer_text) < min(only_large, only_small), (seed, q.text)


def test_unknown_tier_raises():
    from mathgen.generators.sc_whole_units_build import sc_whole_units, sc_whole_units_chain

    with pytest.raises(ValueError):
        sc_whole_units(random.Random(0), tier="impossible")
    with pytest.raises(ValueError):
        sc_whole_units_chain(random.Random(0), tier="impossible")


# -- the exam chain -----------------------------------------------------------------------------------------------------


def test_registered():
    sources = {s["difficulty"]: s for s in sup.skill_yaml(SKILL)["sources"]}
    assert set(sources) == {"easy", "medium", "hard", "exam"}
    sup.assert_registered(SKILL, CHAIN)


def test_chain_shape():
    sup.assert_chain_shape(CHAIN, ["number"] * 4, [1, 2, 2, 2], [0, 1, 1, 1])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN)
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=100, top=0.05)


def test_chain_matches_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(CHAIN, seed)
        total, size, up = read_story(q.text)
        n = whole(total, size, up)
        price = price_in_chain(q)
        left = n * size - total if up else total - n * size
        got = [p.answer_text for p in q.parts]
        assert got == [str(total), str(n), str(n * price), str(left)], (seed, q.text, got)
        assert nearest(total, size) != n, (seed, q.text)


def test_chain_follows_through_from_wrong_earlier_answers():
    """A wrong count in (b) carried correctly into (c) and (d) still earns those marks."""
    sk = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.exam_question(CHAIN, seed)
        total, size, up = read_story(q.text)
        price = price_in_chain(q)
        canon = {p.label: p.answer_text for p in q.parts}
        n = whole(total, size, up) + 2
        reply = {
            "a": canon["a"],
            "b": str(n),
            "c": str(n * price),
            "d": str(n * size - total if up else total - n * size),
        }
        if reply["d"] in (reply["b"], canon["d"]):
            continue  # the same number written twice is not following through
        result = grade(sk, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)
        assert result.parts["d"].marks == q.parts[3].marks, (seed, reply)


def test_chain_direction_is_balanced():
    ups = sum(read_story(sup.exam_question(CHAIN, s).text)[2] for s in range(200))
    assert 70 <= ups <= 130, ups


def _family(text):
    for i, (pattern, *_rest) in enumerate(_STORIES):
        if re.search(pattern, text):
            return i
    raise AssertionError(text)


def test_every_story_family_is_audited_against_the_oracle():
    """Each of the ten story families is hit many times, and its key is recomputed from the rendered numbers."""
    from collections import Counter

    hits = Counter()
    for seed in range(300):
        q = t.question(SKILL, "medium", seed)
        hits[_family(q.text)] += 1
        assert medium_oracle(q) == q.answer_text, (seed, q.text)
    assert set(hits) == set(range(len(_STORIES))) and min(hits.values()) >= 3, hits


def test_shelf_story_counts_books_one_at_a_time():
    """The shelf keys by stacking books until one no longer fits, not by the formula under test."""
    for seed in range(300):
        q = t.question(SKILL, "medium", seed)
        m = re.search(
            r"shelf is (\d+) centimetres long and already holds (\d+) books that are each (\d+) centimetres", q.text
        )
        if not m:
            continue
        length, k, w = (int(v) for v in m.groups())
        s = int(re.search(r"More books, each (\d+) centimetres", q.text).group(1))
        free, books = length - k * w, 0
        while free >= s:
            free, books = free - s, books + 1
        ask = ask_of(q.text)
        if ask == "count":
            assert q.answer_text == str(books), (seed, q.text)
        elif ask == "left":
            assert q.answer_text == str(free), (seed, q.text)
