"""Scale-drawing skills drawn on a square grid: ``grid_scale_drawing_build``.

Four skills, three tiers each, every figure a ``grid_drawing``. Each tier is generated through the skill's own
sources, so what is tested is what a learner is served.
"""

import functools
import re

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.generators import grid_scale_drawing_build as gsd
from mathgen.grading.typed import grade
from mathgen.skills import load_skills

SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

# skill id -> topic
SKILLS = {
    "senior-find-actual-measurements-from-scale-drawings-such-as": "grid_scale_actual",
    "senior-obtain-measurements-from-scale-drawings-such-as-maps": "grid_scale_map",
    "senior-estimate-and-compare-quantities-materials-and-costs-using": "grid_scale_materials",
    "senior-understand-and-apply-drawing-conventions-of-scale-drawings": "grid_scale_conventions",
}
CASES = [(sid, tier) for sid in SKILLS for tier in TIERS]


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


@functools.lru_cache(maxsize=None)
def _gen(skill_id, tier, seed):
    src = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("skill_id", SKILLS)
def test_skill_is_active_behind_the_figure_gate(skill_id):
    skill = _skills()[skill_id]
    assert skill.answer == "number"
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert [s.difficulty for s in skill.sources] == TIERS
    assert all(s.topic == SKILLS[skill_id] for s in skill.sources)
    assert len(skill.misconceptions) >= 4


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_figure_validates_does_not_leak_and_reader_agrees_with_key(skill_id, tier):
    reader = figures.get_reader(SKILLS[skill_id])
    assert reader is not None
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert q.figure["kind"] == "grid_drawing" and q.figure["grid"]["type"] == "square"
        assert figures.validate(q.figure) == [], (seed, figures.validate(q.figure))
        figures.assert_no_leak(q.figure, [q.answer_text])
        assert not any(ch.isdigit() for ch in q.figure["alt"])
        assert reader(q) == q.answer_text, (skill_id, tier, seed, q.text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_key_grades_correct_and_every_distractor_wrong(skill_id, tier):
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        assert grade("number", q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        for d in q.distractors:
            assert not grade("number", q, d).correct, (skill_id, tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_deterministic_varied_and_names_its_unit(skill_id, tier):
    seen = set()
    for seed in SEEDS:
        q = _gen(skill_id, tier, seed)
        again = make(q.topic, seed=seed, difficulty=tier)
        assert q.to_json() == again.to_json()
        seen.add((q.text, q.answer_text, repr(q.figure)))
    assert len(seen) >= 50, (skill_id, tier, len(seen))


def test_a_labelled_dimension_is_never_the_answer():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen("senior-understand-and-apply-drawing-conventions-of-scale-drawings", tier, seed)
            shown = [lb["text"] for lb in q.figure.get("labels", [])] + [q.figure.get("scale", {}).get("text", "")]
            assert all(q.answer_text not in re.findall(r"\d+(?:\.\d+)?", t) for t in shown), (tier, seed)


def test_actual_measurements_text_names_the_unit():
    for seed in SEEDS:
        assert "in metres" in _gen(SKILL_ACTUAL, "easy", seed).text
        assert "in metres" in _gen(SKILL_ACTUAL, "medium", seed).text
        assert "in square metres" in _gen(SKILL_ACTUAL, "hard", seed).text
        assert "in kilometres" in _gen(SKILL_MAP, "easy", seed).text
        assert "minutes" in _gen(SKILL_MAP, "hard", seed).text
        assert "in centimetres" in _gen(SKILL_CONV, "hard", seed).text


SKILL_ACTUAL, SKILL_MAP, SKILL_MATERIALS, SKILL_CONV = SKILLS


# ------------------------------------------------------------------ the tiers' reasoning really differs


def test_actual_hard_area_scales_by_the_square_and_names_the_linear_slip():
    for seed in SEEDS:
        q = _gen(SKILL_ACTUAL, "hard", seed)
        poly = gsd._drawn_polygon(q.figure)
        cells, k = gsd._area(poly), gsd._metres_per_cm(q.figure)
        assert float(cells * k * k) == q.answer
        assert str(gsd._text(cells * k)) in q.distractors or cells * k < 1  # the linear-factor slip is offered


def test_actual_medium_is_a_perimeter_of_a_shape_with_more_than_four_sides():
    for seed in SEEDS:
        q = _gen(SKILL_ACTUAL, "medium", seed)
        poly = gsd._drawn_polygon(q.figure)
        assert len(poly) in (6, 8)
        assert gsd._perimeter(poly) * gsd._metres_per_cm(q.figure) == q.answer


def test_actual_tiers_use_the_ratio_form_of_the_scale_sometimes():
    ratios = {_gen(SKILL_ACTUAL, "hard", s).figure["scale"]["text"].startswith("1 :") for s in SEEDS}
    assert ratios == {True, False}


def test_map_medium_road_doubles_back_and_hard_has_more_legs_and_a_speed():
    for seed in SEEDS:
        for tier, legs in (("medium", 3), ("hard", None)):
            q = _gen(SKILL_MAP, tier, seed)
            road = next(ln for ln in q.figure["lines"] if ln["style"] == "road")["points"]
            assert len(road) - 1 >= 3 and (legs is None or len(road) - 1 == legs)
            straight = abs(road[-1][0] - road[0][0]) + abs(road[-1][1] - road[0][1])
            assert straight < gsd._length([tuple(p) for p in road])  # the straight-across slip differs
        assert "km/h" in _gen(SKILL_MAP, "hard", seed).text
        assert "km/h" not in _gen(SKILL_MAP, "medium", seed).text
        assert (
            len([ln for ln in _gen(SKILL_MAP, "easy", seed).figure["lines"] if ln["style"] == "road"][0]["points"]) == 2
        )


def test_materials_round_up_and_hard_compares_two_options():
    for seed in SEEDS:
        for tier in TIERS:
            q = _gen(SKILL_MATERIALS, tier, seed)
            if tier == "hard":
                assert q.text.count("Option") == 2
                continue
            poly = gsd._drawn_polygon(q.figure)
            k = gsd._metres_per_cm(q.figure)
            m = re.search(r"cover (\d+) square metres", q.text) or re.search(r"(\d+) metres long", q.text)
            measure = gsd._area(poly) * k * k if "cover" in m.group(0) else gsd._perimeter(poly) * k
            assert measure % int(m.group(1)) != 0  # the pack count really is a rounding-up question
            assert q.answer == -(-measure // int(m.group(1)))


def test_conventions_tiers_ask_different_things():
    easy = {_gen(SKILL_CONV, "easy", s).text for s in SEEDS}
    assert any("ratio 1 : n" in t for t in easy) and any("How many metres" in t for t in easy)
    for seed in SEEDS:
        assert "scale" not in _gen(SKILL_CONV, "medium", seed).figure  # the scale is not written: it must be found
        assert "scale" not in _gen(SKILL_CONV, "hard", seed).figure
        assert "fence" in _gen(SKILL_CONV, "hard", seed).text
    medium = {("1 : n" in _gen(SKILL_CONV, "medium", s).text) for s in SEEDS}
    assert medium == {True, False}


def test_bad_tier_is_refused():
    for topic in SKILLS.values():
        with pytest.raises(Exception):
            make(topic, seed=1, difficulty="easy", tier="nope")


def test_map_symbols_are_spaced_and_the_river_keeps_off_the_road():
    for tier in TIERS:
        for seed in SEEDS:
            fig = _gen(SKILL_MAP, tier, seed).figure
            at = [tuple(s["at"]) for s in fig["symbols"]]
            for i, a in enumerate(at):
                for b in at[i + 1 :]:
                    assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2, (tier, seed, a, b)
            road = next(ln for ln in fig["lines"] if ln["style"] == "road")["points"]
            river = next(ln for ln in fig["lines"] if ln["style"] == "river")["points"]
            assert river[0][1] != river[1][1] and river[1][1] != river[2][1]  # slanted: never along a grid line
            assert road and all(p not in at[2:] for p in road)


def test_distractors_share_the_shape_of_the_answer():
    for skill_id, tier in CASES:
        for seed in SEEDS:
            q = _gen(skill_id, tier, seed)
            decimal = "." in q.answer_text
            assert all(("." in d) == decimal for d in q.distractors), (skill_id, tier, seed)
