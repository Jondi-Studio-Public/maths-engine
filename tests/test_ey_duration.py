"""Comparing how long events take from a picture: the ``durations`` layout and the skill ``ey-compare-duration``.

The skill is generated through its own sources at every tier and 60 seeds: the figure is valid and does not show
the key, the registered reader re-derives the key from the figure and the prompt, and the real grader accepts the
key and rejects every distractor. Then the traps each tier promises, the refusal of bad specs, and the wiring.
"""

from __future__ import annotations

import copy
import functools

import pytest

from mathgen import figures, make, svg
from mathgen.capabilities import load_capability_gates
from mathgen.figure_kinds import compare as cmp
from mathgen.grading import grade, mcq_options
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills
from scripts import skill_contract

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)
SKILL = "ey-compare-duration"
TOPIC = "compare_duration_picture"


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


@functools.lru_cache(maxsize=None)
def _q(tier, seed):
    spec = next(s for s in _skills()[SKILL].sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=tier, **(spec.params or {}))


def _is_right(q, text):
    return grade("mcq", q, str(mcq_options(q).index(text))).correct


def _names(fig):
    return [e["label"] for e in fig["events"]]


# ------------------------------------------------------------------ the skill file


def test_skill_wiring():
    skill = _skills()[SKILL]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.read_aloud is True and skill.learner_band == "early"
    assert skill.answer == "mcq" and skill.domain == "measurement"
    assert "ac9m1m01" in skill.tags
    assert [(s.topic, s.difficulty) for s in skill.sources] == [(TOPIC, t) for t in TIERS]
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
    assert len(skill.misconceptions) >= 3 and all(m.description for m in skill.misconceptions)
    for pre in skill.prerequisites:
        assert _skills()[pre].status == "active"


@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(family):
    skill_contract.FAMILIES[family](SKILL)


# ------------------------------------------------------------------ every question


@pytest.mark.parametrize("tier", TIERS)
def test_every_question_draws_a_valid_figure_that_does_not_show_the_key(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.figure["kind"] == "compare_picture" and q.figure["layout"] == "durations"
        assert figures.validate(q.figure) == [], (tier, seed)
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert svg.figure_svg(q.figure).startswith("<svg")
        assert len(q.text.split()) <= READ_ALOUD_MAX_WORDS, q.text


@pytest.mark.parametrize("tier", TIERS)
def test_registered_reader_re_derives_the_key(tier):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in SEEDS:
        q = _q(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_the_key_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert _is_right(q, q.answer_text), (tier, seed)
        assert len(q.distractors) >= 2 and q.answer_text not in q.distractors
        for d in q.distractors:
            assert not _is_right(q, d), (tier, seed, d, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_not_always_the_same_answer(tier):
    assert len({_q(tier, seed).answer_text for seed in SEEDS}) >= 3


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")


# ------------------------------------------------------------------ the traps each tier promises


def test_equality_is_a_key_only_under_a_prompt_that_asks_for_it():
    same = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(tier, seed)
            if q.answer_text == "They take the same time":
                same += 1
                assert "Is one longer" in q.text or "are they the same" in q.text
                assert len({e["blocks"] for e in q.figure["events"]}) == 1
            else:
                assert len({e["blocks"] for e in q.figure["events"]}) == len(q.figure["events"])
    assert same >= 1


def test_easy_is_two_events_from_one_start_line_and_the_bigger_card_is_often_not_the_key():
    off = 0
    for seed in SEEDS:
        fig = _q("easy", seed).figure
        assert len(fig["events"]) == 2 and fig["same_start"] is True
        assert {e["start"] for e in fig["events"]} == {0}
        sizes = {e["label"]: e["size"] for e in fig["events"]}
        q = _q("easy", seed)
        if len(set(sizes.values())) == 2 and q.answer_text != "They take the same time":
            off += max(sizes, key=sizes.get) != q.answer_text.removeprefix("The ")
    assert off >= 8


def test_medium_key_is_never_the_top_event_or_the_one_with_the_biggest_card():
    for seed in SEEDS:
        q = _q("medium", seed)
        fig = q.figure
        assert len(fig["events"]) == 3 and fig["same_start"] is True
        key = q.answer_text.removeprefix("The ")
        sizes = {e["label"]: e["size"] for e in fig["events"]}
        top = max(sizes.values())
        biggest = [k for k, v in sizes.items() if v == top]
        assert key != fig["events"][0]["label"]
        assert not (len(biggest) == 1 and biggest[0] == key)
        assert f"The {fig['events'][0]['label']}" in q.distractors


def test_hard_events_start_apart_and_the_last_to_finish_is_never_the_key_of_a_longest_or_shortest_question():
    asked_order = asked_extreme = 0
    for seed in SEEDS:
        q = _q("hard", seed)
        fig = q.figure
        assert fig["same_start"] is False and len(fig["events"]) == 3
        assert len({e["start"] for e in fig["events"]}) >= 2
        ends = {e["label"]: e["start"] + e["blocks"] for e in fig["events"]}
        if "list" in q.text or "in order" in q.text:
            asked_order += 1
            assert q.answer_text.count(", ") == 2 and len(q.distractors) == 3
            by_end = ", ".join(f"The {k}" for k in sorted(ends, key=ends.get))
            assert by_end != q.answer_text and by_end in q.distractors
            for d in q.distractors:
                assert sorted(d.split(", ")) == sorted(q.answer_text.split(", "))
        else:
            asked_extreme += 1
            more = any(w in q.text for w in ("longest", "most"))
            far = (max if more else min)(ends, key=ends.get)
            assert q.answer_text != f"The {far}" and f"The {far}" in q.distractors
    assert asked_order >= 5 and asked_extreme >= 5


def test_longer_names_have_more_boxes_so_the_picture_and_life_agree():
    from mathgen.generators.ey_duration_build import _GROUPS

    rank = {n: i for i, g in enumerate(_GROUPS) for n in g}
    for tier in TIERS:
        for seed in SEEDS:
            ev = _q(tier, seed).figure["events"]
            for a in ev:
                for b in ev:
                    if rank[a["label"]] < rank[b["label"]]:
                        assert a["blocks"] < b["blocks"]


# ------------------------------------------------------------------ the figure kind refuses bad specs


def _durations():
    return cmp.compare_durations(
        events=[
            {"label": "clap", "size": 2, "start": 1, "blocks": 2},
            {"label": "sing", "size": 1, "start": 3, "blocks": 5},
        ],
        unit="sand",
        same_start=False,
        alt="Two events",
    )


def _bad(mutate):
    spec = copy.deepcopy(_durations())
    mutate(spec)
    return figures.validate(spec)


def test_a_good_duration_spec_validates_and_reads_back():
    fig = _durations()
    assert figures.validate(fig) == []
    assert cmp.read_durations(fig) == {"clap": 2, "sing": 5}
    for unit in cmp.DURATION_UNITS:
        assert svg.figure_svg(fig | {"unit": unit}).startswith("<svg")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["events"].pop(),
        lambda s: s["events"].extend(dict(s["events"][0], label=f"x{i}") for i in range(3)),
        lambda s: s["events"][1].update(label="clap"),
        lambda s: s["events"][1].update(label=""),
        lambda s: s["events"][1].update(label="a very long label"),
        lambda s: s["events"][1].update(size=4),
        lambda s: s["events"][1].update(blocks=0),
        lambda s: s["events"][1].update(blocks=2.5),
        lambda s: s["events"][1].update(start=-1),
        lambda s: s["events"][1].update(start=9),
        lambda s: s.update(unit="brick"),
        lambda s: s.update(same_start=True),
        lambda s: s["events"][1].update(start=1),
        lambda s: s.update(same_start="yes"),
    ],
)
def test_bad_duration_specs_are_refused(mutate):
    assert _bad(mutate)
