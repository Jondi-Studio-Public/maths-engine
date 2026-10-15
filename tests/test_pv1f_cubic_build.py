"""Tests `pv1f_cubic_y_intercept` (ACMMM017), an active figure skill, through its own sources.

``pv1f_cubic_y_intercept`` (``answer: number``, ACMMM017). The skill is ``status: active``
(visual), so the loader's own generation checks skip it: every tier of every source is
generated here through the skill's own sources.
"""

import collections
import functools

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1f_cubic_build as gen
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape
from scripts.measure_tiers import classify_skill

SKILL_ID = "senior-recognise-features-of-the-graphs-of-y-x-acmmm017"
TOPIC = "pv1f_cubic_y_intercept"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@functools.lru_cache(maxsize=1)
def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _gen(tier, seed):
    spec = next(s for s in _skill().sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


# ------------------------------------------------------------------ the skill file


def test_skill_loads_with_the_figure_stimulus_shape():
    skill = _skill()
    assert skill.answer == "number" and skill.domain == "functions-sequences"
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.mental is None
    assert [s.difficulty for s in skill.sources] == TIERS
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert skill.misconceptions


def test_every_tier_draws_a_valid_figure_with_no_leak():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen(tier, seed)
            assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
            assert all(len(p.get("label", "")) <= 1 for p in q.figure["points"])
            assert all(len(ln.get("label", "")) <= 1 for ln in q.figure["lines"])


def test_deterministic_at_every_tier():
    for tier in TIERS:
        for seed in SEEDS:
            assert _gen(tier, seed).to_json() == _gen(tier, seed).to_json()


# ------------------------------------------------------------------ the real grader


@pytest.mark.parametrize("tier", TIERS)
def test_number_grader_accepts_the_key_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = _gen(tier, seed)
        y0 = q.answer
        assert isinstance(y0, int) and 0 < abs(y0) <= 8 and q.answer_text == str(y0)
        assert len(q.distractors) == 3, (tier, seed, q.distractors)
        for d in q.distractors:
            assert int(d) != 0 and abs(int(d)) <= 9
        for wrong in (y0 + 1, y0 - 1, -y0):
            assert not grade_number(q, str(wrong)).correct
        assert not grade_number(q, "").correct


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration_and_distractor_distinctness(tier):
    counts = collections.Counter()
    n = 200
    for seed in range(n):
        q = _gen(tier, seed)
        counts[q.answer_text] += 1
        values = [q.answer] + [int(d) for d in q.distractors]
        assert len(set(values)) == len(values), (tier, seed, values)
    assert max(counts.values()) / n < 0.25, (tier, counts.most_common(3))
    assert len(counts) >= 10, (tier, len(counts))


# ------------------------------------------------------------------ the reader


@pytest.mark.parametrize("tier", TIERS)
def test_registered_reader_equals_the_key_1000_seeds(tier):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in range(1000):
        q = _gen(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


def test_reader_does_not_peek_at_the_answer():
    q = _gen("exam", 4)
    key = q.answer_text
    q.answer_text, q.answer, q.distractors = "99", 99, []
    assert figures.get_reader(TOPIC)(q) == key


# ------------------------------------------------------------------ the in-generator checks fire


def test_in_generator_intercept_check_fires_when_the_expanded_constant_is_wrong(monkeypatch):
    monkeypatch.setattr(gen, "_y_intercept", lambda spec: gen._value(spec, 0) + 1)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, tier="easy")


def test_figure_reading_check_fires_on_a_wrong_key():
    q = _gen("hard", 3)
    gen._read_from_figure(q.figure, q.text)
    shifted = {
        **q.figure,
        "curves": [{"pieces": [[[x, y + 1] for x, y in piece] for piece in q.figure["curves"][0]["pieces"]]}],
    }
    assert gen._read_from_figure(shifted, q.text) != q.answer_text


def test_reader_refuses_a_curve_that_misses_the_axis_on_a_grid_line():
    q = _gen("easy", 2)
    half = {
        **q.figure,
        "curves": [{"pieces": [[[x, y + 0.5] for x, y in piece] for piece in q.figure["curves"][0]["pieces"]]}],
    }
    with pytest.raises(VerificationError):
        gen._read_from_figure(half, q.text)


# ------------------------------------------------------------------ boundaries


def test_every_spec_has_a_readable_axis_crossing_inside_the_window():
    for y0, (shifts, roots) in gen._SPECS_BY_Y0.items():
        for spec in shifts + roots:
            assert gen._y_intercept(spec) == y0 and abs(y0) <= 8 and y0 != 0
            assert abs(gen._slope_at_axis(spec)) <= 6, spec
            assert abs(gen._value(spec, 0)) <= 8


def test_hard_line_never_shares_the_cubics_intercept_and_exam_curves_stay_apart():
    for seed in range(120):
        q = _gen("hard", seed)
        (ln,) = q.figure["lines"]
        m = ln["through"][0][1]
        assert ln["label"] == "L" and ln["through"][1][1] == m
        assert m != q.answer and abs(m - q.answer) >= 2
        assert str(m) in q.distractors  # L's own height is always the first lure offered
    for seed in range(120):
        q = _gen("exam", seed)
        assert len(q.figure["curves"]) == 2 and len(q.figure["points"]) == 1
        (pt,) = q.figure["points"]
        assert pt["label"] == "P" and abs(pt["at"][0]) >= 2
        ys = []
        for curve in q.figure["curves"]:
            hits = {y for piece in curve["pieces"] for x, y in piece if x == 0}
            assert len(hits) == 1
            ys.append(hits.pop())
        assert abs(ys[0] - ys[1]) >= 2
        other = next(int(y) for y in ys if int(y) != q.answer)
        assert str(other) in q.distractors  # the other curve's intercept is always offered


def test_medium_curve_has_three_visible_x_intercepts():
    for seed in range(60):
        q = _gen("medium", seed)
        crossings = 0
        for piece in q.figure["curves"][0]["pieces"]:
            for (x0, y0), (x1, y1) in zip(piece, piece[1:]):
                if y0 == 0 and y1 != 0:
                    crossings += 1
                elif y0 * y1 < 0:
                    crossings += 1
        assert crossings == 3, (seed, crossings)


# ------------------------------------------------------------------ variety and escalation


@pytest.mark.parametrize("tier", TIERS)
def test_each_tier_has_three_shapes_and_twenty_distinct_questions(tier):
    qs = [make(TOPIC, seed=s, difficulty=tier) for s in range(60)]
    assert len({prompt_shape(q) for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 20


def test_every_tier_adds_a_prompt_shape():
    verdict, adds = classify_skill(_skill(), 60)
    assert verdict == "escalates" and adds == [False, True, True, True], adds


@pytest.mark.parametrize("tier", TIERS)
def test_prompts_state_no_value_and_no_python_notation(tier):
    for seed in SEEDS:
        q = _gen(tier, seed)
        assert "**" not in q.text and "*" not in q.text
        assert q.answer_text not in q.text.replace("1 unit", "").split()
        assert not any(ch.isdigit() for ch in q.text.replace("1 unit", "").replace("^3", "").replace("x = 0", "")), (
            q.text
        )
