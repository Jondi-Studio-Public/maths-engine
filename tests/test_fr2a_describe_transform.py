"""`f10-describe-transformation-from-points`: topic `fr2a_describe_transform` and the exam chain."""

from __future__ import annotations

import math
import re

import fr2a_support as sup
import pytest

SKILL = "f10-describe-transformation-from-points"
TOPIC = "fr2a_describe_transform"
EXAM = "exd_x_fr2a_describe_chain"
SEEDS = range(70)


def drawn(question):
    """(original corners, image corners) read from the labels of the figure."""
    obj, img = {}, {}
    for p in question.figure["points"]:
        label = p["label"]
        (img if "'" in label else obj)[label[0]] = tuple(int(c) for c in p["at"])
    return [obj[n] for n in "ABC"], [img[n] for n in "ABC"]


def test_wiring():
    sup.assert_wired(SKILL, TOPIC, EXAM, "pair", True)


def test_contract_all_tiers():
    sup.assert_single_answer_contract(SKILL, TOPIC, SEEDS)


def test_easy_every_corner_moves_by_the_answer():
    for seed in SEEDS:
        question = sup.q(TOPIC, "easy", seed)
        obj, img = drawn(question)
        right, up = question.answer["right"], question.answer["up"]
        assert all((i[0] - o[0], i[1] - o[1]) == (right, up) for o, i in zip(obj, img)), (seed, question.text)
        assert right != 0 and up != 0


def test_medium_reflection_and_half_turn_fit_every_corner():
    kinds = set()
    for seed in SEEDS:
        question = sup.q(TOPIC, "medium", seed)
        obj, img = drawn(question)
        if "k = _" in question.text:
            axis = 0 if "vertical" in question.text else 1
            k, d = question.answer["k"], question.answer["dist"]
            kinds.add("mirror")
            assert all((o[axis] + i[axis]) / 2 == k and o[1 - axis] == i[1 - axis] for o, i in zip(obj, img))
            assert d == abs(obj[0][axis] - k) and d > 0
        else:
            kinds.add("turn")
            cx, cy = question.answer["x"], question.answer["y"]
            assert all(((o[0] + i[0]) / 2, (o[1] + i[1]) / 2) == (cx, cy) for o, i in zip(obj, img))
    assert kinds == {"mirror", "turn"}


def test_hard_undoing_the_other_move_gives_the_translation():
    seen = set()
    for seed in SEEDS:
        question = sup.q(TOPIC, "hard", seed)
        obj, img = drawn(question)
        right, up = question.answer["right"], question.answer["up"]
        text = question.text
        turn_first = (
            text.index("translated") > text.index("reflected")
            if "reflected" in text
            else text.index("translated") > text.index("rotated")
        )
        flip = (
            (lambda p: (p[0], -p[1]))
            if "x-axis" in text
            else (lambda p: (-p[0], p[1]))
            if "y-axis" in text
            else (lambda p: (-p[0], -p[1]))
        )
        for o, i in zip(obj, img):
            if turn_first:
                f = flip(o)
                assert (i[0] - f[0], i[1] - f[1]) == (right, up), (seed, text)
            else:
                f = flip(i)
                assert (f[0] - o[0], f[1] - o[1]) == (right, up), (seed, text)
        seen.add(turn_first)
        # leaving the other move out gives a different pair
        assert (img[0][0] - obj[0][0], img[0][1] - obj[0][1]) != (right, up)
    assert seen == {True, False}


def test_tiers_differ_in_shape():
    texts = {t: {re.sub(r"\d+", "N", sup.q(TOPIC, t, s).text) for s in SEEDS} for t in sup.TIERS}
    assert not (texts["easy"] & texts["medium"]) and not (texts["medium"] & texts["hard"])
    assert all("first image is not drawn" in sup.q(TOPIC, "hard", s).text for s in SEEDS)


# ------------------------------------------------------------------------------------------ exam


def test_exam_contract():
    sup.assert_exam_contract(SKILL, EXAM, range(70))


def test_exam_move_and_details_are_consistent_with_the_picture():
    names = set()
    for seed in range(120):
        question = sup.exam(EXAM, seed)
        obj, img = drawn(question)
        key = question.parts[0].answer_text
        names.add(key)
        diffs = {(i[0] - o[0], i[1] - o[1]) for o, i in zip(obj, img)}
        mids = {((i[0] + o[0]) / 2, (i[1] + o[1]) / 2) for o, i in zip(obj, img)}
        pair = question.parts[1].answer
        if key == "a translation":
            assert len(diffs) == 1 and pair == {"right": next(iter(diffs))[0], "up": next(iter(diffs))[1]}
        elif key == "a reflection in a vertical line":
            assert {m[0] for m in mids} == {pair["k"]} and all(o[1] == i[1] for o, i in zip(obj, img))
            assert pair["dist"] == abs(obj[0][0] - pair["k"])
        else:
            assert key == "a rotation of 180° (a half turn)" and len(mids) == 1
            assert pair == {"x": int(next(iter(mids))[0]), "y": int(next(iter(mids))[1])}
        sides = sorted(math.dist(img[a], img[b]) for a, b in ((0, 1), (1, 2), (2, 0)))
        assert float(question.parts[2].answer_text) == round(sides[2])
        assert float(question.parts[3].answer_text) == round(sum(sides))
    assert len(names) == 3


def test_exam_follow_through_from_a_wrong_longest_side():
    for seed in range(40):
        question = sup.exam(EXAM, seed)
        hyp = float(question.parts[2].answer_text)
        legs = float(question.parts[3].answer_text) - hyp
        sup.assert_follow_through(SKILL, question, {"c": str(int(hyp + 2))}, {"d": str(int(hyp + 2 + legs))})
