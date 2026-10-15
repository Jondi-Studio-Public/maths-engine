"""The ``shape_scene`` figure kind and the three early-years shape skills built on it.

Every skill is generated through its own sources, at every tier and 60 seeds: the figure is valid, does not
show the key, the registered reader re-derives the key from the figure and the prompt alone, and the real
grader accepts the key and rejects every distractor. Also: bad specs are refused, the facts table, the
read-aloud cap and the skill wiring.
"""

from __future__ import annotations

import functools

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import mcq_options
from mathgen.figure_kinds import scene
from mathgen.grading import grade
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills
from mathgen.svg import figure_svg

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

SKILLS = {
    "name-basic-shapes": "ey_shape_recognise",
    "f10-sort-classify-familiar-shapes": "ey_shape_classify",
    "f10-shape-sides-spatial-terms": "ey_shape_sides",
}
CODES = {
    "name-basic-shapes": "ac9mfsp01",
    "f10-sort-classify-familiar-shapes": "ac9m1sp01",
    "f10-shape-sides-spatial-terms": "ac9m2sp01",
}


@functools.lru_cache(maxsize=1)
def _skills():
    return {s.id: s for s in load_skills()[1]}


@functools.lru_cache(maxsize=None)
def _q(topic, tier, seed):
    return make(topic, seed=seed, difficulty=tier)


def _all():
    return [(sid, topic, tier) for sid, topic in SKILLS.items() for tier in TIERS]


def _is_right(q, text) -> bool:
    return grade("mcq", q, str(mcq_options(q).index(text))).correct


# ------------------------------------------------------------------ the figure kind


def _item(shape="square", **kw):
    return scene.scene_item(shape, **kw)


def test_a_scene_of_every_shape_validates_and_renders():
    items = [_item(s, style=st) for s, st in zip(scene.FLAT + scene.SOLIDS, ["plain", "shaded", "dotted"] * 6)]
    spec = scene.shape_scene(items=items[:16], alt="shapes", layout="grid", cols=4)
    assert figures.validate(spec) == []
    assert figure_svg(spec).startswith("<svg")


@pytest.mark.parametrize("name", sorted(scene.OBJECTS))
def test_every_object_validates_and_renders(name):
    spec = scene.shape_scene(object=name, alt=f"a picture of {name}")
    assert figures.validate(spec) == []
    assert "<svg" in figure_svg(spec)
    assert scene.object_part_shape(spec, scene.OBJECTS[name]["parts"][0][0]) == scene.OBJECTS[name]["parts"][0][1]


def _bad_specs():
    ok = _item()
    base = dict(kind="shape_scene", alt="x", layout="row", items=[ok], object=None)
    yield "no items", {**base, "items": []}
    yield "17 items", {**base, "items": [ok] * 17}
    yield "unknown shape", {**base, "items": [{**ok, "shape": "blob"}]}
    yield "bad size", {**base, "items": [{**ok, "size": "huge"}]}
    yield "bad style", {**base, "items": [{**ok, "style": "red"}]}
    yield "turn not a multiple of 15", {**base, "items": [{**ok, "turn": 10}]}
    yield "turn out of range", {**base, "items": [{**ok, "turn": 360}]}
    yield "turn as bool", {**base, "items": [{**ok, "turn": True}]}
    yield "solid turned", {**base, "items": [{**_item("cube"), "turn": 30}]}
    yield "long label", {**base, "items": [{**ok, "label": "ABC"}]}
    yield "missing key", {**base, "items": [{"shape": "square"}]}
    yield "bad layout", {**base, "layout": "pile"}
    yield "grid without cols", {**base, "layout": "grid"}
    yield "grid cols too wide", {**base, "layout": "grid", "cols": 9}
    yield "cols on a row", {**base, "cols": 3}
    yield "items and object", {**base, "object": {"name": "house", "parts": []}}
    yield "unknown object", {**base, "items": [], "object": {"name": "castle", "parts": []}}
    yield "object parts wrong", {**base, "items": [], "object": {"name": "house", "parts": []}}
    yield "no alt", {**base, "alt": ""}


@pytest.mark.parametrize("why,spec", list(_bad_specs()), ids=lambda v: v if isinstance(v, str) else "")
def test_bad_specs_are_refused(why, spec):
    assert figures.validate(spec), why
    with pytest.raises(ValueError):
        figures.check(spec)


def test_sixteen_items_is_the_cap_and_is_accepted():
    spec = scene.shape_scene(items=[_item()] * 16, alt="x")
    assert figures.validate(spec) == []


def test_item_labels_are_the_only_drawn_text_and_feed_the_leak_check():
    spec = scene.shape_scene(items=[_item(label="A"), _item("circle", label="B")], alt="two shapes")
    assert figures.labels(spec) == ["two shapes", "A", "B"]
    with pytest.raises(AssertionError):
        figures.assert_no_leak(spec, ["Shape A"[-1:]])


def test_styles_are_told_apart_without_colour():
    plain, shaded, dotted = (
        figure_svg(scene.shape_scene(items=[_item("circle", style=s)], alt="x")) for s in scene.STYLES
    )
    assert "stroke-dasharray" in dotted and "stroke-dasharray" not in plain
    assert plain != shaded != dotted != plain


def test_a_turn_is_drawn_as_a_rotation():
    spec = scene.shape_scene(items=[_item(turn=45)], alt="x")
    assert "rotate(45)" in figure_svg(spec)


def test_fact_table_matches_what_children_are_taught():
    f = scene.FACTS
    assert [f[s]["sides"] for s in ("triangle", "square", "rectangle", "pentagon", "hexagon")] == [3, 4, 4, 5, 6]
    assert f["circle"]["corners"] == 0 and f["oval"]["corners"] == 0
    assert f["kite"]["sides"] == 4 and not f["kite"]["parallel"]
    assert f["rhombus"]["parallel"] and f["square"]["parallel"] and f["hexagon"]["parallel"]
    assert not f["pentagon"]["parallel"] and not f["triangle"]["parallel"]
    assert f["semicircle"]["curved"] and not f["semicircle"]["straight_only"]
    for solid in scene.SOLIDS:
        assert f[solid]["sides"] is None


# ------------------------------------------------------------------ the skills


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_skill_wiring(skill_id):
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.capability_gates == ["figure_stimulus"] and skill.needs_figure is True
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.learner_band == "early" and skill.read_aloud is True
    assert CODES[skill_id] in skill.tags
    assert skill.answer == "mcq"
    assert [s.difficulty for s in skill.sources] == TIERS
    assert {s.topic for s in skill.sources} == {SKILLS[skill_id]}
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
    assert len(skill.misconceptions) >= 4 and all(m.description for m in skill.misconceptions)
    for pre in skill.prerequisites:
        assert _skills()[pre].status == "active"


def test_the_prerequisite_chain_runs_from_naming_to_sides():
    assert _skills()["f10-sort-classify-familiar-shapes"].prerequisites == ["name-basic-shapes"]
    assert _skills()["f10-shape-sides-spatial-terms"].prerequisites == ["f10-sort-classify-familiar-shapes"]


@pytest.mark.parametrize("skill_id,topic,tier", _all())
def test_every_figure_validates_renders_and_does_not_show_the_key(skill_id, topic, tier):
    for seed in SEEDS:
        q = _q(topic, tier, seed)
        assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
        figure_svg(q.figure)
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])


@pytest.mark.parametrize("skill_id,topic,tier", _all())
def test_registered_reader_re_derives_the_key_from_the_figure(skill_id, topic, tier):
    reader = figures.get_reader(topic)
    assert reader is not None
    for seed in SEEDS:
        q = _q(topic, tier, seed)
        assert reader(q) == q.answer_text, (tier, seed, q.text)


@pytest.mark.parametrize("skill_id,topic,tier", _all())
def test_the_real_grader_accepts_the_key_and_rejects_every_distractor(skill_id, topic, tier):
    for seed in SEEDS:
        q = _q(topic, tier, seed)
        assert _is_right(q, q.answer_text), (tier, seed)
        assert len(q.distractors) >= 2 and q.answer_text not in q.distractors
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert not _is_right(q, d), (tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id,topic,tier", _all())
def test_every_prompt_is_short_enough_to_read_aloud(skill_id, topic, tier):
    for seed in SEEDS:
        assert len(_q(topic, tier, seed).text.split()) <= READ_ALOUD_MAX_WORDS


@pytest.mark.parametrize("topic", sorted(SKILLS.values()))
def test_deterministic_and_unknown_tier_refused(topic):
    for tier in TIERS:
        for seed in range(15):
            assert make(topic, seed=seed, difficulty=tier).to_json() == _q(topic, tier, seed).to_json()
    with pytest.raises((ValueError, TypeError)):
        make(topic, seed=1, tier="bogus")


@pytest.mark.parametrize("topic", sorted(SKILLS.values()))
def test_each_tier_has_variety(topic):
    for tier in TIERS:
        assert len({(_q(topic, tier, s).text, str(_q(topic, tier, s).figure)) for s in SEEDS}) >= 20


# ------------------------------------------------------------------ tier shape: what each question really asks


def test_naming_shows_one_upright_basic_shape():
    for seed in SEEDS:
        fig = _q("ey_shape_recognise", "easy", seed).figure
        (item,) = fig["items"]
        assert item["shape"] in ("circle", "triangle", "square", "rectangle") and item["turn"] == 0


def test_counting_puts_a_turned_target_and_a_near_neighbour_in_the_scene():
    turned = 0
    for seed in SEEDS:
        q = _q("ey_shape_recognise", "medium", seed)
        target = next(s for s in scene.FLAT if scene.PLURAL[s] in q.text)
        items = q.figure["items"]
        mine = [i for i in items if i["shape"] == target]
        assert 1 <= len(mine) <= 4 and len(items) > len(mine) + 1
        turned += any(i["turn"] for i in mine) or target == "circle"
    assert turned == len(SEEDS)


def test_hard_recognition_covers_every_object_and_the_reverse_question():
    objects, reverse = set(), 0
    for seed in range(120):
        q = (
            _q("ey_shape_recognise", "hard", seed)
            if seed < 60
            else make("ey_shape_recognise", seed=seed, difficulty="hard")
        )
        if q.figure["object"]:
            objects.add(q.figure["object"]["name"])
        else:
            reverse += 1
    assert objects == set(scene.OBJECTS) - {"ice cream"} and reverse > 0


def test_the_object_answer_is_the_listed_part_shape():
    for seed in SEEDS:
        q = _q("ey_shape_recognise", "hard", seed)
        if q.figure["object"]:
            assert q.answer_text in {scene.DISPLAY[p["shape"]] for p in q.figure["object"]["parts"]}


def test_classify_easy_counts_the_right_group_and_never_the_other_one():
    for seed in SEEDS:
        q = _q("ey_shape_classify", "easy", seed)
        solids = sum(1 for i in q.figure["items"] if i["shape"] in scene.SOLIDS)
        flat = len(q.figure["items"]) - solids
        assert int(q.answer_text) == (solids if "solid" in q.text else flat)
        assert str(flat if "solid" in q.text else solids) in q.distractors or solids == flat


def _holds(items, at, reason) -> bool:
    """Does ``reason`` really hold of items[at], judged from the drawn items alone?"""
    it, f = items[at], scene.FACTS[items[at]["shape"]]
    rank = {s: i for i, s in enumerate(scene.SIZES)}
    others = [o for i, o in enumerate(items) if i != at]
    if reason == "it has a curved side":
        return bool(f["curved"])
    if reason == "it has corners":
        return bool(f["corners"])
    if reason.endswith(" sides"):
        return f["sides"] == int(reason.split()[2])
    if reason == "it is bigger":
        return all(rank[it["size"]] > rank[o["size"]] for o in others)
    if reason == "it is turned":
        return it["turn"] != 0
    if reason == "it is shaded":
        return it["style"] == "shaded"
    raise AssertionError(reason)


def test_odd_one_out_no_reason_offered_is_true_of_the_item_it_names():
    for seed in SEEDS:
        q = _q("ey_shape_classify", "hard", seed)
        items = q.figure["items"]
        letters = [i["label"] for i in items]
        for d in q.distractors:
            letter, reason = d.split(", ", 1)
            assert not _holds(items, letters.index(letter), reason), (seed, d)


def test_odd_one_out_key_item_is_the_unique_outlier_and_nothing_else_singles_one_out():
    for seed in SEEDS:
        q = _q("ey_shape_classify", "hard", seed)
        items = q.figure["items"]
        letters = [i["label"] for i in items]
        letter, reason = q.answer_text.split(", ", 1)
        at = letters.index(letter)
        assert [i for i in range(4) if _holds(items, i, reason)] == [at], (seed, q.answer_text)
        # size, style and turn are the same for all four, so none of them points at another item
        assert (
            len({i["size"] for i in items}) == len({i["style"] for i in items}) == len({i["turn"] for i in items}) == 1
        )
        # the three that match are alike on the stated attribute, and the offered options include both kinds of miss
        assert any(d.startswith(letter + ",") for d in q.distractors)
        assert any(d.endswith(reason) for d in q.distractors)


def test_a_scene_never_asks_for_x_while_drawing_another_shape_that_is_also_x():
    asked_total = 0
    for seed in SEEDS:
        q = _q("ey_shape_recognise", "medium", seed)
        target = next(s for s in scene.FLAT if scene.PLURAL[s] in q.text)
        asked_total += 1
        for i in q.figure["items"]:
            assert i["shape"] == target or not scene.is_also(i["shape"], target), (seed, q.text, i["shape"])
    assert asked_total == len(SEEDS)
    assert (
        scene.is_also("square", "rectangle") and scene.is_also("square", "rhombus") and scene.is_also("circle", "oval")
    )
    assert not scene.is_also("rectangle", "square") and not scene.is_also("oval", "circle")


def test_a_name_question_never_offers_two_right_names():
    for tier in ("easy", "hard"):
        for seed in SEEDS:
            q = _q("ey_shape_recognise", tier, seed)
            shown = (
                q.figure["items"][0]["shape"]
                if q.figure["items"] and q.figure["items"][0]["shape"] in scene.FLAT
                else None
            )
            if q.figure["object"]:
                shown = q.answer_text.lower()
            if shown is None or shown not in scene.FLAT:
                continue
            for d in q.distractors:
                assert not scene.is_also(shown, d.lower()), (tier, seed, shown, d)


def test_odd_one_out_uses_all_three_kinds_of_rule():
    reasons = {_q("ey_shape_classify", "hard", s).answer_text.split(", ", 1)[1] for s in SEEDS}
    assert "it has a curved side" in reasons and "it has corners" in reasons
    assert any("sides" in r and "curved" not in r for r in reasons)


def test_sides_easy_counts_a_polygon_and_never_a_curved_shape():
    for seed in SEEDS:
        q = _q("ey_shape_sides", "easy", seed)
        (item,) = q.figure["items"]
        assert scene.FACTS[item["shape"]]["sides"] == int(q.answer_text)
        assert "4" in q.distractors or q.answer_text == "4"


def test_sides_medium_has_four_distinct_labelled_shapes_and_one_match():
    for seed in SEEDS:
        q = _q("ey_shape_sides", "medium", seed)
        items = q.figure["items"]
        assert len({i["shape"] for i in items}) == 4
        assert sorted(q.distractors + [q.answer_text]) == [f"Shape {c}" for c in "ABCD"]


def test_sides_medium_covers_curved_straight_and_parallel_and_the_semicircle_decoy():
    words, semi = set(), 0
    for seed in SEEDS:
        q = _q("ey_shape_sides", "medium", seed)
        words |= {w for w in ("curved", "straight", "parallel") if w in q.text}
        semi += any(i["shape"] == "semicircle" for i in q.figure["items"])
    assert words == {"curved", "straight", "parallel"} and semi > 10


def test_sides_hard_mixes_shape_kinds_and_a_turned_one_among_the_matches():
    for seed in SEEDS:
        q = _q("ey_shape_sides", "hard", seed)
        items = q.figure["items"]
        assert 7 <= len(items) <= 10 and 2 <= int(q.answer_text) <= 5
        if "parallel" in q.text:
            match = [i for i in items if scene.FACTS[i["shape"]]["parallel"]]
        else:
            n = int("".join(c for c in q.text if c.isdigit()))
            match = [i for i in items if scene.FACTS[i["shape"]]["sides"] == n]
        assert len(match) == int(q.answer_text) and any(i["turn"] for i in match)


# ------------------------------------------------------------------ review fixes

_SAME_OUTLINE = {("circle", "sphere"), ("rectangle", "rectangular prism"), ("square", "cube"), ("triangle", "cone")}


def test_every_question_has_four_distinct_options_over_300_seeds():
    for topic in SKILLS.values():
        for tier in TIERS:
            for seed in range(300):
                q = make(topic, seed=seed, difficulty=tier)
                options = mcq_options(q)
                assert len(options) == 4 and len(set(options)) == 4, (topic, tier, seed, options)


def test_no_question_offers_a_flat_name_beside_the_solid_with_that_outline_or_a_cube_beside_a_prism():
    for tier in ("easy", "hard"):
        for seed in range(300):
            q = make("ey_shape_recognise", seed=seed, difficulty=tier)
            names = {o.lower() for o in mcq_options(q)}
            for flat, solid in _SAME_OUTLINE:
                assert not {flat, solid} <= names, (tier, seed, names)
            assert not {"cube", "rectangular prism"} <= names


def test_the_ice_cream_is_never_asked_and_the_ball_is_drawn_as_a_solid():
    seen_ball = False
    for seed in range(300):
        q = make("ey_shape_recognise", seed=seed, difficulty="hard")
        if q.figure["object"]:
            assert q.figure["object"]["name"] != "ice cream"
            if q.figure["object"]["name"] == "ball":
                seen_ball = True
                assert q.answer_text == "Sphere"
                assert "M-88 0A88 26.4" in figure_svg(q.figure)  # the equator ellipse that makes it a sphere
    assert seen_ball


def test_the_object_working_text_reads_naturally():
    for seed in range(300):
        q = make("ey_shape_recognise", seed=seed, difficulty="hard")
        for line in q.working:
            assert "of a box" not in line and "The ball of" not in line and "The can of" not in line, line


def test_parallel_prompts_say_at_least_one_pair_and_the_facts_agree():
    for seed in SEEDS:
        for tier in ("medium", "hard"):
            q = _q("ey_shape_sides", tier, seed)
            if "parallel" in q.text and tier == "hard":
                assert "at least one pair" in q.text
    assert scene.FACTS["hexagon"]["parallel"] and scene.FACTS["rectangle"]["parallel"]
    assert any("at least one pair" in _q("ey_shape_sides", "medium", s).text for s in range(120))


def test_the_semicircle_is_left_out_of_every_sides_and_corners_scene():
    for seed in range(120):
        hard = _q("ey_shape_sides", "hard", seed)
        assert all(i["shape"] != "semicircle" for i in hard.figure["items"])
        easy = _q("ey_shape_sides", "easy", seed)
        assert all(i["shape"] != "semicircle" for i in easy.figure["items"])
        q = _q("ey_shape_classify", "medium", seed)
        if "corner" in q.text:
            assert all(i["shape"] != "semicircle" for i in q.figure["items"])
