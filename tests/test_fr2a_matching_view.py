"""`f10-pick-the-matching-view`: topic `fr2a_matching_view` (easy, medium, hard) and the exam chain."""

from __future__ import annotations

import re

import fr2a_support as sup
import pytest

import mathgen.generators.fr2a_matching_view_build as mv

SKILL = "f10-pick-the-matching-view"
TOPIC = "fr2a_matching_view"
EXAM = "exd_x_fr2a_matching_view_chain"
SEEDS = range(80)


def panel_x(question):
    """Left edge of each lettered panel and the common panel width, from the labels under the panels."""
    at = {p["label"]: p["at"] for p in question.figure["points"] if p["label"] in "ABCD" and len(p["label"]) == 1}
    width = round(at["B"][0] - at["A"][0] - 2)
    return {k: round(v[0] - width / 2) for k, v in at.items()}, width


def plan_of(question):
    nums = {
        (int(p["at"][0] - 0.5), int(p["at"][1] - 0.5)): int(p["label"])
        for p in question.figure["points"]
        if p["label"].isdigit()
    }
    x0 = min(k[0] for k in nums)
    y0 = min(k[1] for k in nums)
    w = 1 + max(k[0] for k in nums) - x0
    d = 1 + max(k[1] for k in nums) - y0
    return [[nums[(x0 + x, y0 + y)] for x in range(w)] for y in range(d)]


def drawn_heights(question, letter):
    cells = mv._cells(question.figure)
    xs, width = panel_x(question)
    heights = tuple(mv._column_height(cells, xs[letter] + j, 0) for j in range(width))
    return heights


def trim(heights):
    return tuple(h for h in heights if h)


def test_wiring():
    sup.assert_wired(SKILL, TOPIC, EXAM, "mcq", True)


def test_contract_all_tiers():
    sup.assert_single_answer_contract(SKILL, TOPIC, range(40), minimum=3)


def test_the_elevation_rules_agree_two_ways():
    grid = [[1, 3, 0], [2, 0, 4], [0, 2, 2]]
    assert mv.front(grid) == mv.front_by_layers(grid) == (2, 3, 4)
    assert mv.right(grid) == mv.right_by_layers(grid) == (3, 4, 2)


@pytest.mark.parametrize("tier,side", [("easy", "front"), ("medium", "right")])
def test_exactly_one_panel_is_the_elevation_and_it_is_the_key(tier, side):
    for seed in SEEDS:
        question = sup.q(TOPIC, tier, seed)
        grid = plan_of(question)
        want = trim(mv.front(grid) if side == "front" else mv.right(grid))
        assert (side == "front") == ("front elevation" in question.text)
        matches = [L for L in "ABCD" if trim(drawn_heights(question, L)) == want]
        assert len(matches) == 1, (tier, seed, matches)
        assert question.answer_text == f"Option {matches[0]}"
        # the wanted view needs the hidden stacks: it is not just the nearest stacks
        nearest = tuple(grid[0]) if side == "front" else tuple(row[-1] for row in grid)
        assert trim(want) != trim(nearest)


def test_hard_one_plan_matches_both_elevations():
    for seed in SEEDS:
        question = sup.q(TOPIC, "hard", seed)
        pos = {p["label"]: p["at"] for p in question.figure["points"]}
        xs, width = panel_x(question)
        cells = mv._cells(question.figure)
        top = round(pos["Front elevation"][1] + 0.6)
        f = trim(
            tuple(mv._column_height(cells, round(pos["Front elevation"][0] - width / 2) + j, top) for j in range(width))
        )
        r = trim(
            tuple(
                mv._column_height(cells, round(pos["Right-side elevation"][0] - width / 2) + j, top)
                for j in range(width)
            )
        )
        nums = {
            (int(p["at"][0] - 0.5), int(p["at"][1] - 0.5)): int(p["label"])
            for p in question.figure["points"]
            if p["label"].isdigit()
        }
        fits = []
        for letter in "ABCD":
            mine = {k: v for k, v in nums.items() if xs[letter] <= k[0] < xs[letter] + width}
            w = 1 + max(k[0] for k in mine) - xs[letter]
            d = 1 + max(k[1] for k in mine)
            grid = [[mine[(xs[letter] + x, y)] for x in range(w)] for y in range(d)]
            if mv.front(grid) == f and mv.right(grid) == r:
                fits.append(letter)
        assert len(fits) == 1 and question.answer_text == f"Option {fits[0]}", (seed, fits)


def test_each_wrong_panel_is_tagged_with_a_different_slip():
    for tier in sup.TIERS:
        for seed in range(40):
            question = sup.q(TOPIC, tier, seed)
            notes = [question.option_notes[d] for d in question.distractors]
            assert len(set(notes)) == len(notes) == 3, (tier, seed, notes)


def test_tiers_differ_in_shape():
    asks = {
        t: {
            re.sub(r"\b[A-Z][a-z]+\b", "X", sup.q(TOPIC, t, s).text[sup.q(TOPIC, t, s).text.index("Which") :])
            for s in range(30)
        }
        for t in sup.TIERS
    }
    assert any("front elevation" in a for a in asks["easy"]) and not any("right-side" in a for a in asks["easy"])
    assert all("right-side" in a for a in asks["medium"])
    assert all("plan view" in a for a in asks["hard"])


def test_the_option_letters_are_spread():
    from collections import Counter

    counts = Counter(sup.q(TOPIC, "easy", s).answer_text for s in range(200))
    assert len(counts) == 4 and min(counts.values()) >= 30


# ------------------------------------------------------------------------------------------ exam


def test_exam_contract():
    sup.assert_exam_contract(SKILL, EXAM, range(60))


def test_exam_counts_rederived_from_the_plan():
    for seed in range(100):
        question = sup.exam(EXAM, seed)
        grid = plan_of(question)
        want = trim(mv.front(grid))
        matches = [L for L in "ABCD" if trim(drawn_heights(question, L)) == want]
        got = sup.canonical(question)
        assert got["a"] == f"Option {matches[0]}" and len(matches) == 1
        shown = sum(max(row) for row in grid)
        total = sum(map(sum, grid))
        assert (int(got["b"]), int(got["c"]), int(got["d"])) == (shown, total, total - shown)
        assert total > shown


def test_exam_follow_through_from_wrong_counts():
    for seed in range(40):
        question = sup.exam(EXAM, seed)
        b, c = int(question.parts[1].answer_text), int(question.parts[2].answer_text)
        sup.assert_follow_through(SKILL, question, {"b": str(b + 3), "c": str(c + 5)}, {"d": str(c + 5 - b - 3)})
