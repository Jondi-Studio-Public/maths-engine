"""`f10-three-d-coordinates-position-distance`: topic `fr2a_box_coordinates` and the exam chain."""

from __future__ import annotations

import math
import re

import fr2a_support as sup
import pytest

SKILL = "f10-three-d-coordinates-position-distance"
TOPIC = "fr2a_box_coordinates"
EXAM = "exd_x_fr2a_box_chain"
SEEDS = range(80)
# corner name -> (x, y, z) multiples of the edge lengths, written out here rather than imported
CORNER = {
    "O": (0, 0, 0),
    "P": (1, 0, 0),
    "Q": (1, 1, 0),
    "R": (0, 1, 0),
    "S": (0, 0, 1),
    "T": (1, 0, 1),
    "U": (1, 1, 1),
    "V": (0, 1, 1),
}


def edge_lengths(question):
    """Lengths read from the drawn edges OP, OR and OS."""
    found = {}
    for seg in question.figure["segments"]:
        m = re.fullmatch(r"(\d+) m", seg.get("label", ""))
        if m:
            found[frozenset((seg["from"], seg["to"]))] = int(m[1])
    return found[frozenset("OP")], found[frozenset("OR")], found[frozenset("OS")]


def base_of(text):
    m = re.search(r"Corner O is at the point \((\d+), (\d+), (\d+)\)", text)
    return tuple(int(v) for v in m.groups()) if m else (0, 0, 0)


def corner(name, dims, base=(0, 0, 0)):
    return tuple(b + f * d for b, f, d in zip(base, CORNER[name], dims))


def test_wiring():
    sup.assert_wired(SKILL, TOPIC, EXAM, "pair", True)


def test_contract_all_tiers():
    sup.assert_single_answer_contract(SKILL, TOPIC, SEEDS)


def test_the_drawn_edges_run_the_right_way_and_dashed_edges_hide_one_corner():
    question = sup.q(TOPIC, "easy", 3)
    segs = {frozenset((s["from"], s["to"])): s for s in question.figure["segments"]}
    assert len(segs) == 12
    dashed = {k for k, s in segs.items() if s.get("dashed")}
    assert dashed == {frozenset("RO"), frozenset("RQ"), frozenset("RV")}
    pos = {p["id"]: p["at"] for p in question.figure["points"]}
    assert pos["P"][0] > pos["O"][0] and abs(pos["P"][1] - pos["O"][1]) < 1e-9  # x runs right
    assert pos["S"][1] > pos["O"][1] and abs(pos["S"][0] - pos["O"][0]) < 1e-9  # z runs up
    assert pos["R"][0] > pos["O"][0] and pos["R"][1] > pos["O"][1]  # y runs back and up


def test_easy_names_a_corner_from_the_lengths():
    for seed in SEEDS:
        question = sup.q(TOPIC, "easy", seed)
        dims = edge_lengths(question)
        assert len(set(dims)) == 3
        name = re.search(r"coordinates of ([A-V]) as", question.text)[1]
        assert name in "QTUV"
        x, y, z = corner(name, dims)
        assert question.answer == {"x": x, "y": y, "z": z}


def test_medium_midpoint_of_the_joining_line():
    for seed in SEEDS:
        question = sup.q(TOPIC, "medium", seed)
        dims = edge_lengths(question)
        u, w = re.search(r"joining ([A-V]) and ([A-V])", question.text).groups()
        mid = [(a + b) / 2 for a, b in zip(corner(u, dims), corner(w, dims))]
        assert [question.answer[k] for k in "xyz"] == mid
        assert all(d % 2 == 0 for d in dims)


def test_hard_pair_floor_and_straight_line():
    for seed in SEEDS:
        question = sup.q(TOPIC, "hard", seed)
        dims, base = edge_lengths(question), base_of(question.text)
        assert base != (0, 0, 0)
        u, w = re.search(r"Corners ([A-V]) and ([A-V])", question.text).groups()
        pu, pw = corner(u, dims, base), corner(w, dims, base)
        assert question.answer == {
            "floor": math.hypot(pu[0] - pw[0], pu[1] - pw[1]),
            "straight": math.dist(pu, pw),
        }
        assert sorted(CORNER[u]) == sorted(1 - c for c in CORNER[w]) and all(
            a != b for a, b in zip(CORNER[u], CORNER[w])
        )


def test_tiers_differ_in_shape():
    kinds = {t: {tuple(sorted(sup.q(TOPIC, t, s).answer)) for s in SEEDS} for t in sup.TIERS}
    assert kinds["easy"] == kinds["medium"] == {("x", "y", "z")} and kinds["hard"] == {("floor", "straight")}
    assert all("survey grid" in sup.q(TOPIC, "hard", s).text for s in SEEDS)
    assert not any("survey grid" in sup.q(TOPIC, "easy", s).text for s in SEEDS)


# ------------------------------------------------------------------------------------------ exam


def test_exam_contract():
    sup.assert_exam_contract(SKILL, EXAM, range(80))


def test_exam_parts_rederived_from_the_drawn_lengths():
    for seed in range(120):
        question = sup.exam(EXAM, seed)
        a, b, c = edge_lengths(question)
        got = sup.canonical(question)
        assert got["a"] == f"x = {a}, y = {b}, z = {c}"
        assert got["b"] == f"x = {a // 2}, y = {b // 2}, z = {c // 2}"
        assert float(got["c"]) == round(math.hypot(a, b) + 1e-12, 1)
        assert float(got["d"]) == round(math.sqrt(a * a + b * b + c * c) + 1e-12, 1)
        assert a % 2 == b % 2 == c % 2 == 0


def test_exam_follow_through_from_a_wrong_floor_diagonal():
    for seed in range(40):
        question = sup.exam(EXAM, seed)
        c = edge_lengths(question)[2]
        wrong = float(question.parts[2].answer_text) + 2
        expect = round(math.sqrt(wrong**2 + c * c) + 1e-12, 1)
        sup.assert_follow_through(SKILL, question, {"c": f"{wrong:.1f}"}, {"d": f"{expect:.1f}"})
