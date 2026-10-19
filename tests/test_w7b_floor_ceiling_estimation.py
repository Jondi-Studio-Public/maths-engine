"""Tests `floor-ceiling-estimation` (topic `w7b_floor_ceiling_estimation`) and its exam chain.

The oracles read the quantities back out of the question TEXT and re-derive the key with `math.ceil`, `math.floor` and
Fraction arithmetic, never from the generator's helpers.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

import mathgen.generators.w7b_floor_ceiling_estimation_build  # noqa: F401  (registers the topics)
from tests import w7b_support as sup

TOPIC = "w7b_floor_ceiling_estimation"
EXAM = "exd_x_w7b_floor_ceiling_estimation"
SKILL = sup.skill_of("floor-ceiling-estimation")


def _ints(text: str):
    return [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text)]


def _up(x: int, place: int) -> int:
    return math.ceil(Fraction(x, place)) * place


def _down(x: int, place: int) -> int:
    return math.floor(Fraction(x, place)) * place


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")
    assert "fce-rounds-down" in {m.id for m in SKILL.misconceptions}
    steps = [SKILL.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_easy_keys_by_independent_arithmetic():
    seen = set()
    for seed in range(250):
        q = sup.single(TOPIC, "easy", seed)
        t, c = _ints(q.text)[:2]
        if "left over" in q.text:
            assert int(q.answer) == t - c * (t // c) == t % c, (seed, q.text)
            seen.add("leftover")
        elif re.search(r"(cut|bows|bottles|pavers|cable runs)", q.text):
            assert int(q.answer) == math.floor(Fraction(t, c)), (seed, q.text)
            assert int(q.answer) * c <= t < (int(q.answer) + 1) * c
            seen.add("floor")
        else:
            assert int(q.answer) == math.ceil(Fraction(t, c)), (seed, q.text)
            seen.add("exact" if t % c == 0 else "ceiling")
    assert seen == {"leftover", "floor", "ceiling", "exact"}


def test_medium_keys():
    seen = set()
    for seed in range(300):
        q = sup.single(TOPIC, "medium", seed)
        text = q.text
        nums = _ints(text)
        if "classes" in text:
            g, e, x, c = nums
            assert int(q.answer) == math.ceil(Fraction(g * e + x, c)), (seed, text)
            seen.add("extras")
        elif "works out" in text:
            a, b, wrong = nums[:3]
            assert int(q.answer) == wrong - _up(a, 10) * _up(b, 10), (seed, text)
            assert int(q.answer) > 0
            seen.add("check")
        elif "upper estimate" in text or "lower estimate" in text:
            a, b = nums[:2]
            upper = "upper" in text
            pa = 100 if "nearest hundred" in text else 10
            f = _up if upper else _down
            assert int(q.answer) == f(a, pa) * f(b, 10), (seed, text)
            assert (int(q.answer) >= a * b) if upper else (int(q.answer) <= a * b)
            seen.add("estimate-up" if upper else "estimate-down")
        else:
            t, c = nums[:2]
            assert int(q.answer) == math.ceil(Fraction(t, c)) * c - t, (seed, text)
            seen.add("shortfall")
    assert seen == {"extras", "estimate-up", "estimate-down", "check", "shortfall"}


def test_hard_keys():
    seen = set()
    for seed in range(300):
        q = sup.single(TOPIC, "hard", seed)
        text = q.text
        if "crates" in text:
            cap, count, total, w = _ints(text)[:4]
            per_trip = min(count, cap // w)
            assert per_trip == cap // w < count
            assert int(q.answer) == math.ceil(Fraction(total, per_trip)), (seed, text)
            seen.add("limits")
        elif "fence" in text or "circular" in text:
            length = _ints(text)[0]
            spacing = Fraction(re.search(r"no more than ([\d.]+) m", text).group(1))
            gaps = math.ceil(Fraction(length) / spacing)
            assert int(q.answer) == (gaps if "circular" in text else gaps + 1), (seed, text)
            seen.add("loop" if "circular" in text else "fence")
        elif "Coaches seat" in text:
            people, big, small = _ints(text)[:3]
            full, rest = divmod(people, big)
            assert int(q.answer) == full + math.ceil(Fraction(rest, small)), (seed, text)
            seen.add("vehicles")
        else:
            lo, hi, c = _ints(text)[:3]
            if "certain there are enough" in text:
                assert int(q.answer) == math.ceil(Fraction(hi, c)), (seed, text)
                seen.add("enough")
            else:
                assert int(q.answer) == math.floor(Fraction(lo, c)), (seed, text)
                seen.add("certain-full")
    assert seen == {"limits", "loop", "fence", "vehicles", "enough", "certain-full"}


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 1, 2, 2], ["number"] * 4)


def test_exam_keys_by_independent_arithmetic():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        t, c, p = _ints(q.stem)[:3]
        a, b, cc, d = q.parts
        boxes = math.ceil(Fraction(t, c))
        assert int(a.answer) == boxes
        assert int(b.answer) == boxes * c - t
        est = _up(boxes, 10) * _up(p, 10)
        assert int(cc.answer) == est and est > boxes * p
        assert int(d.answer) == est - boxes * p


def test_exam_follow_through_from_a_wrong_box_count():
    def wrong(q, canon):
        t, c, p = _ints(q.stem)[:3]
        boxes = math.ceil(Fraction(t, c)) + 1
        return {"a": str(boxes), "b": str(boxes * c - t)}

    assert sup.check_follow_through(SKILL, EXAM, "b", wrong) >= 30

    def wrong_estimate(q, canon):
        t, c, p = _ints(q.stem)[:3]
        boxes = math.ceil(Fraction(t, c))
        est = _up(boxes, 10) * _up(p, 10) + 50
        return {"c": str(est), "d": str(est - boxes * p)}

    assert sup.check_follow_through(SKILL, EXAM, "d", wrong_estimate) >= 30


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier, 120)
        assert len(spread) >= 20, (tier, len(spread))
        assert max(spread.values()) <= 0.2 * 120, (tier, spread.most_common(3))
