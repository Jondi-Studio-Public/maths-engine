"""`f10-grid-map-familiar-place`: topic `fr2a_grid_map` (easy, medium, hard)."""

from __future__ import annotations

import re

import fr2a_support as sup

SKILL = "f10-grid-map-familiar-place"
TOPIC = "fr2a_grid_map"
SEEDS = range(120)
STEP = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}


def places(question):
    """Landmark name -> (column, row), read from where each name is drawn (columns and rows count from 1)."""
    out = {}
    for p in question.figure["points"]:
        label = p["label"]
        if len(label) == 1 and label.isalpha() or label.isdigit():
            continue
        out[label] = (int(p["at"][0]) + 1, int(p["at"][1]) + 1)
    return out


def grid_size(question):
    return int(question.figure["x_range"][1]), int(question.figure["y_range"][1])


def test_wiring():
    sup.assert_wired(SKILL, TOPIC, None, "pair", True)


def test_contract_all_tiers():
    sup.assert_single_answer_contract(SKILL, TOPIC, SEEDS)


def test_stems_stay_short():
    for tier in sup.TIERS:
        for seed in SEEDS:
            assert len(sup.q(TOPIC, tier, seed).text.split()) <= 36, (tier, seed)
    assert max(len(sup.q(TOPIC, t, s).text.split()) for t in ("easy", "medium") for s in SEEDS) <= 36


def test_axes_are_labelled_and_landmarks_sit_inside_one_square_each():
    for tier in sup.TIERS:
        for seed in range(30):
            question = sup.q(TOPIC, tier, seed)
            cols, rows = grid_size(question)
            letters = sorted(
                p["label"] for p in question.figure["points"] if len(p["label"]) == 1 and p["label"].isalpha()
            )
            assert letters == list("ABCDEFG"[:cols])
            numbers = sorted(int(p["label"]) for p in question.figure["points"] if p["label"].isdigit())
            assert numbers == list(range(1, rows + 1))
            spots = list(places(question).values())
            assert len(set(spots)) == len(spots) >= 4
            assert all(1 <= c <= cols and 1 <= r <= rows for c, r in spots)
            assert "north is at the top" in question.figure["title"]


def test_easy_reads_the_square_of_the_named_landmark():
    for seed in SEEDS:
        question = sup.q(TOPIC, "easy", seed)
        pos = places(question)
        name = next(n for n in pos if f"the {n}" in question.text)
        assert question.answer == {"column": pos[name][0], "row": pos[name][1]}


def test_medium_follows_two_moves_from_the_landmark():
    for seed in SEEDS:
        question = sup.q(TOPIC, "medium", seed)
        pos = places(question)
        name = next(n for n in pos if f"the {n}" in question.text)
        col, row = pos[name]
        moves = re.findall(r"(\d+) squares? (east|west|north|south)", question.text)
        assert len(moves) == 2 and {STEP[w][0] != 0 for _, w in moves} == {True, False}
        for n, word in moves:
            col, row = col + STEP[word][0] * int(n), row + STEP[word][1] * int(n)
        cols, rows = grid_size(question)
        assert 1 <= col <= cols and 1 <= row <= rows
        assert question.answer == {"column": col, "row": row}


def test_hard_legs_partly_cancel_and_the_net_is_the_answer():
    for seed in SEEDS:
        question = sup.q(TOPIC, "hard", seed)
        pos = places(question)
        order = sorted((question.text.find(f"the {n}"), n) for n in pos if f"the {n}" in question.text)
        a, b, c = (pos[n] for _, n in order[:3])
        leg1, leg2 = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
        assert leg1[0] * leg2[0] < 0 or leg1[1] * leg2[1] < 0, "the legs never go back on each other"
        assert question.answer == {"east": c[0] - a[0], "north": c[1] - a[1]}
        # adding the sizes of the moves (the slip) is a wrong answer
        added = {"east": abs(leg1[0]) + abs(leg2[0]), "north": abs(leg1[1]) + abs(leg2[1])}
        assert added != question.answer


def test_tiers_differ_in_shape():
    shapes = {
        t: {re.sub(r"[A-Z][a-z]+( [a-z]+)?|\d+", "X", sup.q(TOPIC, t, s).text.split("?")[0])[:40] for s in range(40)}
        for t in sup.TIERS
    }
    assert shapes["easy"].isdisjoint(shapes["medium"]) and shapes["medium"].isdisjoint(shapes["hard"])
    sizes = {t: grid_size(sup.q(TOPIC, t, 0)) for t in sup.TIERS}
    assert sizes["easy"][0] < sizes["medium"][0] < sizes["hard"][0]


def test_themes_vary():
    themes = {sup.q(TOPIC, "easy", s).figure["title"] for s in SEEDS}
    assert len(themes) == 4
