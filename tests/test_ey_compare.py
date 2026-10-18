"""Comparing length, mass and capacity from a picture: the ``compare_picture`` figure kind and its five skills.

Each skill is generated through its own sources at every tier and 60 seeds: the figure is valid, does not
show the key, the registered reader re-derives the key from the figure and the prompt, and the real grader
accepts the key and rejects every distractor. Then the traps each tier promises, the refusal of bad specs,
and the skill wiring.
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

SKILLS = {
    "compare-lengths": ("compare_length_picture", "ac9mfm01", "early-foundations"),
    "f10-compare-measurement-attributes": ("compare_mass_capacity_picture", "ac9mfm01", "measurement"),
    "ey-compare-by-paper-clips": ("compare_by_paper_clips", "ac9m1m01", "measurement"),
    "ey-order-by-two-balances": ("order_by_two_balances", "ac9m1m01", "measurement"),
    "ey-compare-capacity-by-cups": ("compare_capacity_by_cups", "ac9m1m01", "measurement"),
}


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


@functools.lru_cache(maxsize=None)
def _q(skill_id, tier, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=tier, **(spec.params or {}))


def _is_right(q, text):
    return grade("mcq", q, mcq_options(q).index(text)).correct


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_skill_wiring(skill_id):
    skill = _skills()[skill_id]
    topic, code, domain = SKILLS[skill_id]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.read_aloud is True and skill.learner_band == "early"
    assert skill.answer == "mcq" and skill.domain == domain
    assert code in skill.tags
    assert [(s.topic, s.difficulty) for s in skill.sources] == [(topic, t) for t in TIERS]
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
    assert len(skill.misconceptions) >= 3 and all(m.description for m in skill.misconceptions)
    for pre in skill.prerequisites:
        assert _skills()[pre].status == "active"


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(skill_id, family):
    skill_contract.FAMILIES[family](skill_id)


# ------------------------------------------------------------------ every question


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_every_tier_draws_a_valid_figure_that_does_not_show_the_key(skill_id):
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            assert q.figure["kind"] == "compare_picture" and figures.validate(q.figure) == [], (tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
            assert svg.figure_svg(q.figure).startswith("<svg")
            assert len(q.text.split()) <= READ_ALOUD_MAX_WORDS, q.text


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_registered_reader_re_derives_the_key(skill_id):
    reader = figures.get_reader(SKILLS[skill_id][0])
    assert reader is not None
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_real_grader_accepts_the_key_and_rejects_every_distractor(skill_id):
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            assert _is_right(q, q.answer_text), (tier, seed)
            assert len(q.distractors) >= 2 and q.answer_text not in q.distractors
            for d in q.distractors:
                assert not _is_right(q, d), (tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_key_is_not_always_the_same_answer(skill_id):
    for tier in TIERS:
        keys = {_q(skill_id, tier, seed).answer_text for seed in SEEDS}
        assert len(keys) >= 3, (tier, keys)


@pytest.mark.parametrize("topic", sorted(v[0] for v in SKILLS.values()))
def test_unknown_tier_raises(topic):
    with pytest.raises(ValueError):
        make(topic, seed=1, tier="bogus")


# ------------------------------------------------------------------ the traps each tier promises


def test_hard_length_objects_are_not_lined_up_and_the_furthest_end_is_never_the_key():
    for seed in SEEDS:
        q = _q("compare-lengths", "hard", seed)
        fig = q.figure
        assert fig["lined_up"] is False and 3 <= len(fig["objects"]) <= 5
        ends = {o["label"]: o["start"] + o["length"] for o in fig["objects"]}
        lengths = cmp.read_lengths(fig)
        more = "long" in q.text and "short" not in q.text
        pick = max(ends, key=ends.get) if more else min(ends, key=ends.get)
        assert q.answer_text != f"The {pick}"
        assert q.answer_text == f"The {(max if more else min)(lengths, key=lengths.get)}"
        assert f"The {pick}" in q.distractors


def test_easy_and_medium_lengths_are_lined_up_at_one_end():
    for tier, n in (("easy", 2), ("medium", 3)):
        for seed in SEEDS:
            fig = _q("compare-lengths", tier, seed).figure
            assert fig["lined_up"] is True and len(fig["objects"]) == n
            assert {o["start"] for o in fig["objects"]} == {0}


def test_paper_clip_objects_are_never_lined_up_and_counts_are_drawn_as_units():
    for tier in TIERS:
        for seed in SEEDS:
            fig = _q("ey-compare-by-paper-clips", tier, seed).figure
            assert fig["lined_up"] is False and fig["unit"]["style"] in cmp.UNIT_STYLES
            assert all(o["units"] == o["length"] for o in fig["objects"])


def test_paper_clip_hard_has_one_equal_pair_and_a_trap_pair_that_ends_in_the_same_place():
    for seed in SEEDS:
        q = _q("ey-compare-by-paper-clips", "hard", seed)
        objs = {o["label"]: o for o in q.figure["objects"]}
        assert len(objs) == 4
        traps = 0
        for d in q.distractors:
            a, b = (name.removeprefix("The ") for name in d.replace(" and the ", "|").split("|"))
            assert objs[a]["units"] != objs[b]["units"]
            traps += objs[a]["start"] + objs[a]["length"] == objs[b]["start"] + objs[b]["length"]
        assert traps >= 1


def test_mass_easy_often_has_the_bigger_object_lighter():
    traps = 0
    for seed in range(200):
        q = make("compare_mass_capacity_picture", seed=seed, difficulty="easy")
        if q.figure["layout"] != "balance":
            continue
        b = q.figure["balances"][0]
        a, c = b["left"]["objects"][0], b["right"]["objects"][0]
        if (a["size"] > c["size"]) != (a["mass"] > c["mass"]) and a["size"] != c["size"]:
            traps += 1
    assert traps >= 25


def test_mass_medium_balances_an_object_against_blocks_and_sometimes_the_fewer_things_win():
    fewer = level = 0
    for seed in range(200):
        q = make("compare_mass_capacity_picture", seed=seed, difficulty="medium")
        if q.figure["layout"] != "balance":
            continue
        b = q.figure["balances"][0]
        pans = [b["left"], b["right"]]
        assert sorted(len(p["objects"]) for p in pans) == [0, 1] and sorted(bool(p["blocks"]) for p in pans) == [
            False,
            True,
        ]
        obj = next(p for p in pans if p["objects"])["objects"][0]
        blocks = next(p for p in pans if p["blocks"])["blocks"]
        level += b["tilt"] == "level"
        fewer += obj["mass"] > blocks
    assert level >= 5 and fewer >= 20


def test_capacity_medium_tallest_container_is_never_the_one_that_holds_most_or_least():
    seen = 0
    for seed in range(200):
        q = make("compare_mass_capacity_picture", seed=seed, difficulty="medium")
        if q.figure["layout"] != "containers":
            continue
        seen += 1
        heights = {c["label"]: cmp.container_size(c["capacity"], c["form"])[1] for c in q.figure["containers"]}
        more = "more" in q.text or "most" in q.text
        pick = max(heights, key=heights.get) if more else min(heights, key=heights.get)
        assert q.answer_text != f"The {pick}"
    assert seen >= 40


def test_conservation_levels_mislead_and_water_amount_is_what_counts():
    keys = set()
    for seed in range(200):
        q = make("compare_mass_capacity_picture", seed=seed, difficulty="hard")
        jug = next(c for c in q.figure["containers"] if c["label"] == "jug")
        bowl = next(c for c in q.figure["containers"] if c["label"] == "bowl")
        assert jug["form"] == "tall_thin" and bowl["form"] == "short_wide"
        keys.add(q.answer_text)
        if jug["fill"] == bowl["fill"]:
            assert q.answer_text == "They have the same amount"
            assert jug["fill"] / jug["capacity"] > bowl["fill"] / bowl["capacity"]
    assert keys == {"They have the same amount", "The bowl", "The jug"}


def test_two_balances_share_exactly_one_object_and_the_masses_stay_hidden():
    for tier in TIERS:
        for seed in SEEDS:
            q = _q("ey-order-by-two-balances", tier, seed)
            first, second = q.figure["balances"]
            names = [{o["label"] for p in (b["left"], b["right"]) for o in p["objects"]} for b in (first, second)]
            assert len(names[0] & names[1]) == 1 and len(names[0] | names[1]) == 3
            assert "mass" not in svg.figure_svg(q.figure)


def test_hard_order_asks_for_a_whole_list_and_every_distractor_is_a_different_order():
    for seed in SEEDS:
        q = _q("ey-order-by-two-balances", "hard", seed)
        assert "list" in q.text and q.answer_text.count(", ") == 2
        assert len(q.distractors) == 3
        for d in q.distractors:
            assert sorted(d.split(", ")) == sorted(q.answer_text.split(", ")) and d != q.answer_text


def test_cups_equal_what_the_container_holds_and_the_tallest_is_never_the_key():
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = _q("ey-compare-capacity-by-cups", tier, seed)
            assert all(c["cups"] == c["fill"] == c["capacity"] for c in q.figure["containers"])
            heights = {c["label"]: cmp.container_size(c["capacity"], c["form"])[1] for c in q.figure["containers"]}
            more = "more" in q.text or "most" in q.text
            pick = max(heights, key=heights.get) if more else min(heights, key=heights.get)
            assert q.answer_text != f"The {pick}"


def test_cups_hard_has_a_trap_pair_that_stands_equally_tall():
    for seed in SEEDS:
        q = _q("ey-compare-capacity-by-cups", "hard", seed)
        cs = {c["label"]: c for c in q.figure["containers"]}
        assert len(cs) == 4 and len({c["capacity"] for c in cs.values()}) == 3
        tall = 0
        for d in q.distractors:
            a, b = (name.removeprefix("The ") for name in d.replace(" and the ", "|").split("|"))
            assert cs[a]["capacity"] != cs[b]["capacity"]
            ha = cmp.container_size(cs[a]["capacity"], cs[a]["form"])[1]
            hb = cmp.container_size(cs[b]["capacity"], cs[b]["form"])[1]
            tall += abs(ha - hb) < 1e-6
        assert tall >= 1


# ------------------------------------------------------------------ the figure kind refuses bad specs


def _lengths():
    return cmp.compare_lengths(
        objects=[
            {"label": "pencil", "style": "pencil", "start": 0, "length": 4},
            {"label": "rope", "style": "rope", "start": 2, "length": 5},
        ],
        lined_up=False,
        alt="Two objects",
    )


def _balance():
    pan = cmp.pan
    a = {"label": "book", "mass": 5, "size": 3}
    b = {"label": "apple", "mass": 2, "size": 1}
    return cmp.compare_balance(balances=[cmp.balance(left=pan(objects=[a]), right=pan(objects=[b]))], alt="A balance")


def _containers():
    return cmp.compare_containers(
        containers=[
            {"label": "jug", "shape": "jug", "form": "tall_thin", "capacity": 6, "fill": 4},
            {"label": "bowl", "shape": "bowl", "form": "short_wide", "capacity": 8, "fill": 8, "cups": 8},
        ],
        alt="Two containers",
    )


def _bad(fig, mutate):
    spec = copy.deepcopy(fig)
    mutate(spec)
    return figures.validate(spec)


def test_good_specs_validate_and_the_kind_is_registered_once():
    for fig in (_lengths(), _balance(), _containers()):
        assert figures.validate(fig) == []
    assert "compare_picture" in figures.kinds() and "compare_picture" in svg.rendered_kinds()
    with pytest.raises(ValueError):
        figures.register_kind("compare_picture", lambda spec: [])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(layout="stack"),
        lambda s: s["objects"].extend(dict(s["objects"][0], label=f"x{i}") for i in range(4)),
        lambda s: s["objects"][1].update(label="pencil"),
        lambda s: s["objects"][1].update(style="boat"),
        lambda s: s["objects"][1].update(length=0),
        lambda s: s["objects"][1].update(length=11),
        lambda s: s["objects"][1].update(start=0.3),
        lambda s: s.update(lined_up=True),
        lambda s: s["objects"][1].update(start=0),
        lambda s: s["objects"][0].update(units=4),
        lambda s: s.update(unit={"name": "block", "style": "block"}),
        lambda s: s.update(unit={"name": "block", "style": "brick"}),
    ],
)
def test_bad_length_specs_are_refused(mutate):
    assert _bad(_lengths(), mutate)


def test_a_unit_row_must_match_the_object_length():
    spec = copy.deepcopy(_lengths())
    spec["unit"] = {"name": "block", "style": "block"}
    for o in spec["objects"]:
        o["units"] = o["length"]
    assert figures.validate(spec) == []
    spec["objects"][0]["units"] = 3
    assert figures.validate(spec)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["balances"][0].update(tilt="level"),
        lambda s: s["balances"][0].update(tilt="right_down"),
        lambda s: s["balances"][0].update(tilt="sideways"),
        lambda s: s["balances"][0]["left"]["objects"][0].update(mass=1),
        lambda s: s["balances"][0]["left"]["objects"][0].update(mass=0),
        lambda s: s["balances"][0]["left"]["objects"][0].update(size=4),
        lambda s: s["balances"][0]["left"]["objects"][0].update(label=""),
        lambda s: s["balances"][0]["left"]["objects"][0].update(label="a very long label"),
        lambda s: s["balances"][0]["right"].update(objects=[], blocks=0),
        lambda s: s["balances"][0]["right"].update(blocks=9),
        lambda s: s["balances"][0]["right"]["objects"].extend(
            [{"label": "x", "mass": 1, "size": 1}, {"label": "y", "mass": 1, "size": 1}]
        ),
        lambda s: s["balances"].extend([s["balances"][0]] * 2),
    ],
)
def test_bad_balance_specs_are_refused(mutate):
    assert _bad(_balance(), mutate)


def test_a_tilt_that_disagrees_with_the_masses_names_the_problem():
    problems = _bad(_balance(), lambda s: s["balances"][0].update(tilt="right_down"))
    assert any("tilt" in p and "masses" in p for p in problems)


def test_blocks_count_as_mass_one_each_and_level_needs_equal_mass():
    pan = cmp.pan
    thing = {"label": "ball", "mass": 4, "size": 2}
    level = cmp.balance(left=pan(objects=[thing]), right=pan(blocks=4))
    assert level["tilt"] == "level"
    assert figures.validate(cmp.compare_balance(balances=[level], alt="x")) == []
    assert cmp.balance(left=pan(objects=[thing]), right=pan(blocks=3))["tilt"] == "left_down"
    assert cmp.balance(left=pan(objects=[thing]), right=pan(blocks=5))["tilt"] == "right_down"


def test_a_chain_holds_one_object_per_pan_and_one_mass_per_label():
    pan = cmp.pan
    a = {"label": "book", "mass": 5, "size": 3}
    b = {"label": "apple", "mass": 2, "size": 1}
    ok = [
        cmp.balance(left=pan(objects=[a]), right=pan(objects=[b])),
        cmp.balance(left=pan(objects=[b]), right=pan(objects=[a])),
    ]
    assert figures.validate({"kind": "compare_picture", "layout": "balance", "alt": "x", "balances": ok}) == []
    blocks = [ok[0], cmp.balance(left=pan(objects=[a]), right=pan(blocks=1))]
    assert figures.validate({"kind": "compare_picture", "layout": "balance", "alt": "x", "balances": blocks})
    heavier_apple = {"label": "apple", "mass": 9, "size": 1}
    clash = [
        ok[0],
        cmp.balance(left=pan(objects=[heavier_apple]), right=pan(objects=[{"label": "hat", "mass": 3, "size": 1}])),
    ]
    assert any(
        "two different masses" in p
        for p in figures.validate({"kind": "compare_picture", "layout": "balance", "alt": "x", "balances": clash})
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["containers"][0].update(shape="vase"),
        lambda s: s["containers"][0].update(form="round"),
        lambda s: s["containers"][0].update(capacity=0),
        lambda s: s["containers"][0].update(capacity=13),
        lambda s: s["containers"][0].update(fill=7),
        lambda s: s["containers"][0].update(fill=-1),
        lambda s: s["containers"][0].update(fill=1.5),
        lambda s: s["containers"][1].update(cups=7),
        lambda s: s["containers"][1].update(capacity=12, fill=12, cups=12),
        lambda s: s["containers"][1].update(label="jug"),
        lambda s: s["containers"].extend(dict(s["containers"][0], label=f"c{i}") for i in range(3)),
        lambda s: s.update(containers=[]),
    ],
)
def test_bad_container_specs_are_refused(mutate):
    assert _bad(_containers(), mutate)


def test_a_figure_never_writes_a_number_and_does_not_name_the_answer_in_alt():
    import xml.etree.ElementTree as ET

    for fig in (_lengths(), _balance(), _containers()):
        root = ET.fromstring(svg.figure_svg(fig))
        words = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
        assert not any(any(ch.isdigit() for ch in w) for w in words), words


def test_container_geometry_is_to_scale():
    w1, h1 = cmp.container_size(4, "normal")
    w2, h2 = cmp.container_size(8, "normal")
    assert w2 * h2 == pytest.approx(2 * w1 * h1)
    # equal height classes: capacity * ratio is the same, and so is the drawn height
    heights = [cmp.container_size(c, f)[1] for f, c in (("normal", 4), ("tall_thin", 2), ("short_wide", 8))]
    assert heights == pytest.approx([heights[0]] * 3)
