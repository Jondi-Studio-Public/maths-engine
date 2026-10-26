"""``grid_latlong_typed`` (senior-latitude-longitude-typed-coordinates, ACMEM159): the learner TYPES a position.

Every answer is re-derived here from the labelled lines of the figure and the wording, by a route that shares no code
with the generator, then marked by the real ``pair`` grader (the key accepted, every wrong option and the usual slips
refused).
"""

from __future__ import annotations

import functools
import re

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills

SKILL = "senior-latitude-longitude-typed-coordinates"
TOPIC = "grid_latlong_typed"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(120)


@functools.lru_cache(maxsize=None)
def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _deg(text):
    n = int(text.split("°")[0])
    return -n if text.endswith(("S", "W")) else n


def _places(q):
    """{letter: (latitude, longitude)} from the labels alone: lines up the side, lines along the top, letters on them."""
    labels = q.figure["labels"]
    lat = {lb["at"][1]: _deg(lb["text"]) for lb in labels if len(lb["text"]) > 1 and lb["at"][0] == 0}
    lon = {lb["at"][0]: _deg(lb["text"]) for lb in labels if len(lb["text"]) > 1 and lb["at"][0] != 0}
    x0 = next(iter(lon))
    return {
        lb["text"]: (lat[0] + 10 * lb["at"][1], lon[x0] + 10 * (lb["at"][0] - x0))
        for lb in labels
        if len(lb["text"]) == 1
    }


def _expected(q):
    places, text = _places(q), q.text
    if "halfway" in text or "midway" in text or "midpoint" in text:
        (a, b) = places["A"], places["B"]
        assert (a[0] + b[0]) % 2 == 0 and (a[1] + b[1]) % 2 == 0
        return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    if re.search(r"\d+ degrees (north|south)", text):
        n, ns = re.search(r"(\d+) degrees (north|south)", text).groups()
        e, ew = re.search(r"(\d+) degrees (east|west)", text).groups()
        la, lo = places["A"]
        return (la + int(n) * (1 if ns == "north" else -1), lo + int(e) * (1 if ew == "east" else -1))
    if "level with" in text:
        f, s = re.search(r"level with (\w) .* in line with (\w)", text).groups()
    elif "latitude of" in text:
        f, s = re.search(r"latitude of (\w) and the longitude of (\w)", text).groups()
    elif "line of latitude as place" in text:
        f, s = re.search(r"latitude as place (\w) and the same line of longitude as place (\w)", text).groups()
    else:
        p = re.search(r"place ([ABC])", text).group(1)
        return places[p]
    return (places[f][0], places[s][1])


def _pair(q):
    return tuple(q.answer.values())


@pytest.mark.parametrize("tier", TIERS)
def test_the_key_is_what_the_figure_and_wording_give(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert _pair(q) == _expected(q)
        assert q.answer_kind == "pair"
        assert q.answer_text == f"({_pair(q)[0]}, {_pair(q)[1]})"
        assert not figures.validate(q.figure)


@pytest.mark.parametrize("tier", TIERS)
def test_the_registered_reader_agrees(tier):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in SEEDS:
        q = _q(tier, seed)
        assert reader(q) == q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_the_real_grader_takes_the_key_and_refuses_every_wrong_option(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade("pair", q, q.answer_text).correct
        la, lo = _pair(q)
        assert grade("pair", q, f"{la}, {lo}").correct
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        for d in q.distractors:
            assert not grade("pair", q, d).correct, (q.answer_text, d)
        if la != lo:
            assert not grade("pair", q, f"({lo}, {la})").correct


@pytest.mark.parametrize("tier", TIERS)
def test_the_stem_states_the_sign_convention_and_the_scale(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert "north and east positive, south and west negative" in q.text
        assert "(latitude, longitude)" in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_there_are_several_prompt_shapes_and_many_answers(tier):
    shapes = {re.sub(r"\b[ABC]\b|\d+", "#", _q(tier, s).text.split("Answer as")[0]) for s in SEEDS}
    assert len(shapes) >= 3
    assert len({_q(tier, s).answer_text for s in SEEDS}) >= 25


def test_medium_takes_one_coordinate_from_each_place_and_signs_vary():
    signs = set()
    for seed in SEEDS:
        q = _q("medium", seed)
        la, lo = _pair(q)
        places = _places(q)
        assert la in {p[0] for p in places.values()} and lo in {p[1] for p in places.values()}
        assert (la, lo) not in places.values()
        signs.add((la > 0, lo > 0))
    assert len(signs) >= 3


def test_hard_has_both_forms_and_every_move_crosses_zero():
    forms = set()
    for seed in SEEDS:
        q = _q("hard", seed)
        if "degrees" in q.text.split("Answer as")[0].replace("10 degrees", ""):
            forms.add("move")
            la, lo = _places(q)["A"]
            n, ns = re.search(r"(\d+) degrees (north|south)", q.text).groups()
            e, ew = re.search(r"(\d+) degrees (east|west)", q.text).groups()
            end = _pair(q)
            assert la * end[0] < 0 or lo * end[1] < 0
            assert abs(end[0]) <= 90 and abs(end[1]) <= 180
            assert {"north", "south"} >= {ns} and {"east", "west"} >= {ew}
        else:
            forms.add("midpoint")
    assert forms == {"move", "midpoint"}


def test_every_slip_is_offered_and_catalogued():
    skill = next(s for s in load_skills()[1] if s.id == SKILL)
    declared = {m.id for m in skill.misconceptions}
    seen = set()
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(tier, seed)
            seen |= set(q.option_notes.values())
    assert seen, "the tagged slips are readable on the question"
    assert seen <= declared
    assert declared <= set(CATALOGUE)
    assert declared == seen


@pytest.mark.parametrize("tier", ["", "exam", "expert"])
def test_an_unknown_tier_is_refused(tier):
    with pytest.raises(ValueError, match="unknown tier"):
        make(TOPIC, seed=1, difficulty="easy", tier=tier)


def test_the_skill_is_typed_pair_and_gated_on_figures():
    skill = next(s for s in load_skills()[1] if s.id == SKILL)
    assert skill.answer == "pair" and skill.needs_figure
    assert [s.difficulty for s in skill.sources] == TIERS
    assert "figure_stimulus" in skill.capability_gates
    assert "figure_stimulus" in load_capability_gates()
