"""Pins the fixes from the exam-tier review: a pyramid face on the right edge, no free follow-through marks,
no doubling that repeats the amount, one format per part's options, and no two parts with the same answer."""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills

_SKILLS = {s.id: s for s in load_skills()[1]}
SEEDS = range(300)


def _places(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


# -- 1. pyramid: the named edge is the one the face stands on, so the run is half of the OTHER edge ------------------

PYRAMID = "exd_x_fu_pyramid_volume"


def test_pyramid_face_edge_and_height_agree_with_the_stem():
    rectangular = 0
    for seed in SEEDS:
        q = make(PYRAMID, seed=seed)
        slant = int(re.search(r"slant height of (\d+) cm", q.stem).group(1))
        square = re.search(r"square base of side (\d+) cm", q.stem)
        if square:
            edge = other = int(square.group(1))
        else:
            rectangular += 1
            b, w = (int(n) for n in re.search(r"base (\d+) cm by (\d+) cm", q.stem).groups())
            word = re.search(r"standing on an? (longer|shorter) edge", q.stem).group(1)
            edge, other = (max(b, w), min(b, w)) if word == "longer" else (min(b, w), max(b, w))
            assert edge != other
        run = other / 2  # the face stands on `edge`, so it rises over half of the perpendicular edge
        assert run < slant, (seed, q.stem)
        h2 = slant**2 - run**2
        assert math.isqrt(int(h2)) ** 2 == h2, (seed, q.stem)
        assert q.parts[0].answer_text == str(math.isqrt(int(h2))), (seed, q.stem)
    assert rectangular > 50


# -- 2. metric batches: a degenerate (a) earns no follow-through ---------------------------------------------------


@pytest.mark.parametrize("kind", ["length", "capacity", "mass"])
def test_metric_degenerate_reply_earns_no_follow_through(kind):
    skill = _SKILLS[f"convert-metric-{kind}"]
    carried = 0
    for seed in SEEDS:
        q = make(f"exd_x_m1_metric_{kind}", seed=seed, difficulty="exam")
        for reply in ({"a": "1", "b": "0", "c": "0"}, {"a": "0", "b": "0", "c": "0"}):
            assert grade(skill, q, reply).marks == 0, (kind, seed, reply)
        # A genuine slip in (a) that still holds at least one portion keeps its follow-through into (b).
        wrong_a = int(q.parts[0].answer_text) * 10
        size = int(re.findall(r"(\d+) (?:mL|mm|km|kg|cm|L|g|m)\b", q.stem)[1])
        result = grade(skill, q, {"a": str(wrong_a), "b": str(wrong_a // size)})
        carried += result.parts["b"].follow_through is True
    assert carried > 100


# -- 3. evaluate-powers: never a factor of 2, so (b) is not (a) again -----------------------------------------------


def test_powers_growth_is_never_doubling_and_b_differs_from_a():
    skill = _SKILLS["evaluate-powers"]
    for seed in SEEDS:
        q = make("exd_x_n2_powers_growth", seed=seed, difficulty="exam")
        factor = int(re.search(r"multiplied by (\d+)", q.stem).group(1))
        assert factor >= 3, (seed, q.stem)
        assert q.parts[1].answer_text != q.parts[0].answer_text
        assert grade(skill, q, {"a": "1", "b": "1", "c": "1"}).marks == 0, seed


# -- 4. every option shares the key's format ------------------------------------------------------------------------

FORMAT_TOPICS = (
    "exd_x_pr4_decimal_estimate_budget",
    "exd_x_pr4_decimal_notation_lengths",
    "exd_x_n2_fraction_remainders",
)


@pytest.mark.parametrize("topic", FORMAT_TOPICS)
def test_number_options_share_the_keys_decimal_point_and_places(topic):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            assert len(p.distractors) >= 2, (topic, seed, p.label)
            for d in p.distractors:
                d = str(d)
                assert ("." in d) == ("." in p.answer_text), (topic, seed, p.label, p.answer_text, d)
                assert _places(d) == _places(p.answer_text), (topic, seed, p.label, p.answer_text, d)


# -- 5. no two parts share an answer --------------------------------------------------------------------------------


@pytest.mark.parametrize("topic", ["exd_x_m6_chord_lengths", "exd_x_m3_scale_reading_chain"])
def test_no_two_parts_share_an_answer(topic):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        values = [float(p.answer_text) for p in q.parts]
        assert len(set(values)) == len(values), (topic, seed, values)


# -- 6. the stem names the flights in words, not as "Flight 1" ------------------------------------------------------


def test_two_flight_stem_has_no_numbered_flights():
    for seed in range(30):
        q = make("exd_x_m1_two_flight_journey", seed=seed, difficulty="exam")
        text = " ".join([q.stem, *(p.prompt for p in q.parts)])
        assert not re.search(r"Flight \d", text)
        assert "The first flight leaves" in q.stem and "The second flight leaves" in q.stem
