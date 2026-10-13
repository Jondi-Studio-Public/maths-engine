"""Tests for the measurement figure kinds (clock, clock_set, protractor, unit_strip, scale) and the
seven skills that draw them.

Every tier of every source is generated through the skill's own sources; the figure must validate,
must not leak its answer, the registered reader must re-derive the key from the figure alone, and
every distractor must be graded wrong by the real grader.
"""

import collections
import copy
import functools

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.figure_kinds import measure
from mathgen.grading.typed import grade, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape
from scripts.measure_tiers import classify_skill

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

# skill id -> (topic, answer kind)
SKILLS = {
    "tell-time-to-hour": ("clock_hours", "mcq"),
    "f10-clock-half-quarter-hour": ("clock_half_quarter", "mcq"),
    "f10-read-time-to-the-minute": ("clock_read_minute", "mcq"),
    "f10-measure-angle-protractor": ("protractor_measure_angle", "number"),
    "f10-measure-uniform-informal-units": ("informal_units_measure", "mcq"),
    "f10-read-measurement-scale": ("scale_read_value", "number"),
    "f10-compare-scale-readings": ("scale_compare_readings", "number"),
}
TIERED = [s for s in SKILLS if s != "tell-time-to-hour"]


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


def _gen(skill_id, tier, seed):
    sources = _skills()[skill_id].sources
    spec = next(s for s in sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _tiers(skill_id):
    return ["easy"] if skill_id == "tell-time-to-hour" else TIERS


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", SKILLS)
def test_skill_loads_with_the_figure_stimulus_shape(skill_id):
    skill = _skills()[skill_id]
    assert skill.answer == SKILLS[skill_id][1]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.mental is None
    assert [s.difficulty for s in skill.sources] == _tiers(skill_id)
    assert skill.misconceptions


@pytest.mark.parametrize("skill_id", SKILLS)
def test_every_tier_draws_a_valid_figure_that_does_not_leak(skill_id):
    for tier in _tiers(skill_id):
        for seed in SEEDS:
            q = _gen(skill_id, tier, seed)
            assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
            assert q.figure["alt"] and q.answer_text not in q.figure["alt"]


@pytest.mark.parametrize("skill_id", SKILLS)
def test_deterministic_and_unknown_tier_raises(skill_id):
    topic = SKILLS[skill_id][0]
    for tier in _tiers(skill_id):
        for seed in range(10):
            assert _gen(skill_id, tier, seed).to_json() == _gen(skill_id, tier, seed).to_json()
    if skill_id != "tell-time-to-hour":
        with pytest.raises(ValueError):
            make(topic, seed=1, tier="bogus")


# ------------------------------------------------------------------ readers and graders


@pytest.mark.parametrize("skill_id", SKILLS)
def test_registered_reader_equals_the_key(skill_id):
    topic = SKILLS[skill_id][0]
    reader = figures.get_reader(topic)
    assert reader is not None
    for tier in _tiers(skill_id):
        for seed in range(400):
            q = _gen(skill_id, tier, seed)
            assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("skill_id", SKILLS)
def test_the_real_grader_accepts_the_key_and_rejects_every_distractor(skill_id):
    kind = SKILLS[skill_id][1]
    for tier in _tiers(skill_id):
        for seed in SEEDS:
            q = _gen(skill_id, tier, seed)
            assert len(q.distractors) == 3 and q.answer_text not in q.distractors
            assert len(set(q.distractors)) == 3
            if kind == "mcq":
                options = mcq_options(q)
                assert sorted(options) == sorted([q.answer_text, *q.distractors])
                for idx, opt in enumerate(options):
                    assert grade("mcq", q, str(idx)).correct == (opt == q.answer_text), (tier, seed, opt)
            else:
                assert grade("number", q, q.answer_text).correct, (tier, seed)
                for d in q.distractors:
                    assert not grade("number", q, d).correct, (tier, seed, d, q.answer_text)
                assert ("." in q.answer_text) == all("." in d for d in q.distractors)


def test_the_readers_do_not_peek_at_the_answer():
    for topic, skill_id in [
        ("protractor_measure_angle", "f10-measure-angle-protractor"),
        ("clock_read_minute", "f10-read-time-to-the-minute"),
    ]:
        q = _gen(skill_id, "medium", 3)
        key = q.answer_text
        q.answer_text, q.answer, q.distractors = "999", 999, []
        assert figures.get_reader(topic)(q) == key


@pytest.mark.parametrize("skill_id", [s for s in SKILLS if SKILLS[s][1] == "mcq"])
def test_the_key_is_not_stuck_in_one_position(skill_id):
    for tier in _tiers(skill_id):
        pos = collections.Counter()
        n = 200
        for seed in range(n):
            q = _gen(skill_id, tier, seed)
            pos[mcq_options(q).index(q.answer_text)] += 1
        assert max(pos.values()) / n < 0.42, (tier, dict(pos))


# ------------------------------------------------------------------ variety and escalation


@pytest.mark.parametrize("skill_id", TIERED)
def test_each_tier_has_three_shapes_and_twenty_distinct_questions(skill_id):
    topic = SKILLS[skill_id][0]
    for tier in TIERS:
        qs = [make(topic, seed=s, difficulty=tier) for s in range(60)]
        assert len({prompt_shape(q) for q in qs}) >= 3, tier
        assert len({q.text for q in qs}) >= 20, tier


@pytest.mark.parametrize("skill_id", TIERED)
def test_every_tier_adds_a_prompt_shape(skill_id):
    verdict, adds = classify_skill(_skills()[skill_id], 60)
    assert verdict == "escalates" and adds[1:] == [True, True], adds


def test_tell_time_to_hour_has_twenty_distinct_prompts():
    assert len({_gen("tell-time-to-hour", "easy", s).text for s in range(60)}) >= 20


def test_prompts_never_state_the_answer():
    for skill_id in ("f10-measure-angle-protractor", "f10-read-measurement-scale", "f10-compare-scale-readings"):
        for tier in TIERS:
            for seed in SEEDS:
                q = _gen(skill_id, tier, seed)
                assert q.answer_text not in q.text.split(), (skill_id, tier, seed)


# ------------------------------------------------------------------ tier shape


def test_clock_tiers_escalate_in_the_times_and_the_figure():
    easy = {_gen("f10-clock-half-quarter-hour", "easy", s).figure["minute"] for s in SEEDS}
    medium = {_gen("f10-clock-half-quarter-hour", "medium", s).figure["minute"] for s in SEEDS}
    assert easy == {0, 30} and {15, 45} <= medium
    hard = _gen("f10-clock-half-quarter-hour", "hard", 0).figure
    assert hard["kind"] == "clock_set" and len(hard["clocks"]) == 4
    # a quarter-to clock draws the short hand three quarters of the way to the next numeral
    spec = next(
        q.figure for s in SEEDS if (q := _gen("f10-clock-half-quarter-hour", "medium", s)).figure["minute"] == 45
    )
    assert spec["hour_angle"] % 30 == 22.5


def test_read_minute_tiers_escalate_from_five_minutes_to_elapsed_time():
    assert all(_gen("f10-read-time-to-the-minute", "easy", s).figure["minute"] % 5 == 0 for s in SEEDS)
    assert all(_gen("f10-read-time-to-the-minute", "medium", s).figure["minute"] % 5 != 0 for s in SEEDS)
    assert all(_gen("f10-read-time-to-the-minute", "hard", s).figure["kind"] == "clock_set" for s in SEEDS)


def test_protractor_tiers_escalate_from_outer_zero_to_turned_and_reflex():
    easy = [_gen("f10-measure-angle-protractor", "easy", s).figure for s in SEEDS]
    assert all(f["zero_on"] == "outer" and f["rotation"] == 0 and f["angle"] % 10 == 0 for f in easy)
    medium = [_gen("f10-measure-angle-protractor", "medium", s).figure for s in SEEDS]
    assert {f["zero_on"] for f in medium} == {"outer", "inner"} and all(f["rotation"] == 0 for f in medium)
    hard = [_gen("f10-measure-angle-protractor", "hard", s).figure for s in SEEDS]
    assert all(f["rotation"] % 15 == 0 and f["rotation"] != 0 for f in hard) and any(f["reflex"] for f in hard)
    for tier in TIERS:  # the wrong-scale misreading 180 - x is always on offer for a non-reflex angle
        for s in SEEDS:
            q = _gen("f10-measure-angle-protractor", tier, s)
            if not q.figure["reflex"]:
                assert str(180 - int(q.answer_text)) in q.distractors


def test_informal_units_tiers_cover_count_part_unit_and_fairness():
    easy = [_gen("f10-measure-uniform-informal-units", "easy", s).figure for s in SEEDS]
    assert all(len(f["rows"]) == 1 and measure.classify_unit_row(f["rows"][0]) == "fair" for f in easy)
    medium = [_gen("f10-measure-uniform-informal-units", "medium", s).figure for s in SEEDS]
    assert any(len(f["rows"]) == 2 for f in medium) and any(f["rows"][0]["length"] % 1 == 0.5 for f in medium)
    kinds = collections.Counter(
        measure.classify_unit_row(_gen("f10-measure-uniform-informal-units", "hard", s).figure["rows"][0])
        for s in range(200)
    )
    assert set(kinds) == {"fair", "gaps", "overlaps", "unequal"} and min(kinds.values()) > 25
    q = _gen("f10-measure-uniform-informal-units", "hard", 1)
    assert len(q.distractors) == 3  # all four statements are offered


def test_scale_tiers_use_the_intended_instruments():
    medium = [_gen("f10-read-measurement-scale", "medium", s).figure for s in SEEDS]
    assert any(f["variant"] == "dial" for f in medium) and any(f["min"] < 0 for f in medium)
    hard = [_gen("f10-read-measurement-scale", "hard", s) for s in SEEDS]
    assert any(f.figure["extra_ticks"] for f in hard) and any("millimetres" in f.text for f in hard)
    cmp_medium = [_gen("f10-compare-scale-readings", "medium", s).figure for s in SEEDS]
    assert all(f["variant"] == "ruler" and f["object"]["start"] > 0 for f in cmp_medium)
    cmp_hard = [_gen("f10-compare-scale-readings", "hard", s).figure for s in SEEDS]
    assert any(f["variant"] == "ruler" and f["min"] > 0 for f in cmp_hard)
    assert any(
        f["variant"] == "thermometer" and min(m["value"] for m in f["marks"]) < 0 < max(m["value"] for m in f["marks"])
        for f in cmp_hard
    )


# ------------------------------------------------------------------ the validators


def _clock(**kw):
    return {**measure.clock(hour=3, minute=45, alt="a clock"), **kw}


def test_clock_hour_hand_must_be_in_its_true_position():
    assert measure.clock(hour=3, minute=45, alt="a clock")["hour_angle"] == 112.5
    assert figures.validate(_clock(hour_angle=120)) != []
    assert figures.validate(_clock(minute_angle=260)) != []
    assert measure.read_clock(_clock()) == (3, 45)
    with pytest.raises(ValueError):
        measure.read_clock(_clock(hour_angle=135))  # hour hand at a numeral while the minute hand says 45


def test_clock_rejects_unreadable_faces():
    assert figures.validate(_clock(hour=7, minute=37, hour_angle=218.5, minute_angle=222, minute_ticks=False)) != []
    assert figures.validate(_clock(numerals="none", minute_ticks=False)) != []
    with pytest.raises(ValueError, match="hide each other"):
        measure.clock(hour=6, minute=33, alt="a clock")  # hands 1.5 degrees apart
    assert figures.validate(_clock(numerals="roman")) != [] and figures.validate(_clock(hour=13)) != []
    assert figures.validate({**measure.clock(hour=12, minute=0, alt="a clock")}) == []


def test_clock_set_needs_distinct_captions():
    a = measure.clock(hour=3, minute=0, alt="a clock", caption="A")
    b = measure.clock(hour=4, minute=0, alt="a clock", caption="B")
    assert figures.validate(measure.clock_set(clocks=[a, b], alt="two clocks")) == []
    assert figures.validate({"kind": "clock_set", "alt": "x", "clocks": [a, dict(a)]}) != []
    assert figures.validate({"kind": "clock_set", "alt": "x", "clocks": [a]}) != []
    assert figures.validate({"kind": "clock_set", "alt": "x", "clocks": [a, {**b, "caption": ""}]}) != []


def _protractor(**kw):
    return {**measure.protractor(angle=65, rotation=0, zero_on="outer", alt="a protractor"), **kw}


def test_protractor_validator_and_reader():
    spec = measure.protractor(angle=65, rotation=30, zero_on="inner", alt="a protractor")
    assert spec["base_direction"] == 210 and spec["second_direction"] == 145
    assert measure.read_protractor(spec) == 65 and measure.read_protractor_scale(spec) == 65
    assert (
        measure.read_protractor(measure.protractor(angle=65, rotation=0, zero_on="outer", reflex=True, alt="x")) == 295
    )
    for bad in (
        {"angle": 90},
        {"angle": 67},
        {"angle": 5},
        {"angle": 175},
        {"zero_on": "middle"},
        {"rotation": 360},
        {"second_direction": 100},  # not the stated angle
        {"base_direction": 20},  # not along the baseline
        {"names": {"vertex": "ab"}},
    ):
        assert figures.validate(_protractor(**bad)) != [], bad


def test_the_other_scale_reads_180_minus_the_angle():
    spec = measure.protractor(angle=65, rotation=0, zero_on="outer", alt="x")
    local = spec["second_direction"] - spec["rotation"]
    assert (local, 180 - local) == (65, 115)


def _strip(rows):
    return {"kind": "unit_strip", "alt": "a strip", "unit": "cube", "unit_style": "cube", "rows": rows}


def _row(length, units, label="pencil"):
    return {"label": label, "length": length, "units": [{"start": s, "length": w} for s, w in units]}


def test_unit_strip_validator_and_classifier():
    fair = _row(4, [(0, 1), (1, 1), (2, 1), (3, 1)])
    assert figures.validate(_strip([fair])) == []
    assert measure.classify_unit_row(fair) == "fair" and measure.read_unit_row(fair) == 4
    half = _row(3.5, [(0, 1), (1, 1), (2, 1), (3, 1)])
    assert figures.validate(_strip([half])) == [] and measure.read_unit_row(half) == 3.5
    gaps = _row(4.5, [(0, 1), (1.5, 1), (2.5, 1), (3.5, 1)])
    assert figures.validate(_strip([gaps])) == [] and measure.classify_unit_row(gaps) == "gaps"
    over = _row(3.5, [(0, 1), (0.5, 1), (1.5, 1), (2.5, 1)])
    assert figures.validate(_strip([over])) == [] and measure.classify_unit_row(over) == "overlaps"
    unequal = _row(4.5, [(0, 1), (1, 1.5), (2.5, 1), (3.5, 1)])
    assert figures.validate(_strip([unequal])) == [] and measure.classify_unit_row(unequal) == "unequal"
    with pytest.raises(ValueError):
        measure.read_unit_row(gaps)
    for bad in (
        _row(4, [(0, 1), (1, 1), (2, 1)]),  # stops short of the end
        _row(4, [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),  # a unit wholly beyond the end
        _row(3.75, [(0, 1), (1, 1), (2, 1), (3, 1)]),  # not a multiple of 0.5
        _row(3.25, [(0, 1), (1, 1), (2, 1), (3, 1)]),
        _row(4, [(0.5, 1), (1.5, 1), (2.5, 1), (3.5, 1)]),  # first unit not at 0
        _row(5, [(0, 1), (1.5, 1), (2.25, 1), (3, 1), (4, 1)]),  # a gap and an overlap in one row
        _row(4, [(0, 1), (1, 1), (2, 1), (3, 1)], label=""),
    ):
        assert figures.validate(_strip([bad])) != [], bad
    assert figures.validate(_strip([fair, fair])) != []  # duplicate labels
    assert figures.validate({**_strip([fair]), "unit_style": "star"}) != []


def _ruler(**kw):
    base = measure.scale(
        variant="ruler", unit="cm", min_=0, max_=10, major=1, minor=0.5, alt="a ruler", object_={"start": 2, "end": 7.5}
    )
    return {**base, **kw}


def test_scale_validator_requires_values_on_ticks_or_marked_halves():
    assert figures.validate(_ruler()) == []
    assert measure.scale_readings(_ruler()) == [2, 7.5]
    assert figures.validate(_ruler(object={"start": 2, "end": 7.25})) != []  # between ticks
    assert figures.validate(_ruler(object={"start": 2, "end": 7.25}, extra_ticks=[7.25])) == []  # a marked half
    assert figures.validate(_ruler(object={"start": 2, "end": 11})) != []  # off the scale
    assert figures.validate(_ruler(object={"start": 5, "end": 3})) != []
    assert figures.validate(_ruler(extra_ticks=[2.5])) != []  # that is a minor tick already
    assert figures.validate(_ruler(minor=0.1, max=10)) != []  # 100 intervals
    assert figures.validate(_ruler(major=1, minor=0.3)) != []
    assert figures.validate(_ruler(level=3)) != []
    assert figures.validate(_ruler(marks=[{"value": 3, "label": "A"}])) != []  # object and marks together
    offset = measure.scale(
        variant="ruler", unit="cm", min_=3, max_=13, major=1, minor=0.5, alt="a ruler", object_={"start": 4, "end": 9}
    )
    assert figures.validate(offset) == []


def test_scale_variants():
    thermo = measure.scale(variant="thermometer", unit="°C", min_=-10, max_=40, major=10, minor=2, alt="x", level=-4)
    assert figures.validate(thermo) == [] and measure.scale_readings(thermo) == [-4]
    assert figures.validate({**thermo, "level": -3}) != []
    assert figures.validate({**thermo, "level": None, "marks": []}) != []
    two = measure.scale(
        variant="thermometer", unit="°C", min_=-10, max_=40, major=10, minor=2, alt="x",
        marks=[{"value": -4, "label": "A"}, {"value": 8, "label": "B"}],
    )  # fmt: skip
    assert figures.validate(two) == [] and measure.scale_readings(two) == [-4, 8]
    assert figures.validate({**two, "marks": two["marks"] + [{"value": 2, "label": "C"}]}) != []
    assert figures.validate({**two, "marks": [two["marks"][0], {**two["marks"][1], "label": "A"}]}) != []
    jug = measure.scale(variant="jug", unit="mL", min_=0, max_=500, major=100, minor=50, alt="x", level=350)
    assert figures.validate(jug) == [] and figures.validate({**jug, "min": 100}) != []
    dial = measure.scale(
        variant="dial", unit="kg", min_=0, max_=5, major=1, minor=0.2, alt="x", level=2.4, start_angle=-135, sweep=270
    )
    assert figures.validate(dial) == []
    assert figures.validate({**dial, "sweep": 100}) != [] and figures.validate({**dial, "start_angle": None}) != []
    assert figures.validate({**jug, "sweep": 270}) != []


def test_kinds_report_every_string_they_draw():
    assert "A" in figures.labels(measure.clock(hour=3, minute=0, alt="a", caption="A"))
    fig = measure.protractor(
        angle=65, rotation=0, zero_on="outer", alt="a", names={"vertex": "O", "base": "A", "second": "B"}
    )
    assert {"O", "A", "B"} <= set(figures.labels(fig))
    strip = _strip([_row(2, [(0, 1), (1, 1)])])
    assert {"cube", "pencil"} <= set(figures.labels(strip))
    assert "cm" in figures.labels(_ruler())


def test_specs_are_plain_json():
    import json

    for spec in (_clock(), _protractor(), _ruler(), _strip([_row(2, [(0, 1), (1, 1)])])):
        assert json.loads(json.dumps(copy.deepcopy(spec))) == spec
