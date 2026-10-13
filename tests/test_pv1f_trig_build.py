"""Tests `pv1f_trig_graph_id` (ACMMM036), an active figure skill whose every tier is generated through its own
sources.

``pv1f_trig_graph_id`` (``answer: mcq``, ACMMM036). The skill is ``status: active``
(visual): every tier of every source is generated here through the skill's own sources.
"""

import collections
import functools

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1f_trig_build as gen
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape
from scripts.measure_tiers import classify_skill

SKILL_ID = "senior-recognise-the-graphs-of-y-sin-x-y"
TOPIC = "pv1f_trig_graph_id"
TIERS = ["easy", "medium", "hard", "exam"]
OPTIONS = ["y = cos x", "y = sin x", "y = tan x"]
CURVES = {"easy": 1, "medium": 1, "hard": 2, "exam": 3}
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
    assert skill.answer == "mcq" and skill.domain == "trigonometry"
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.mental is None
    assert [s.difficulty for s in skill.sources] == TIERS
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert skill.misconceptions


def test_every_tier_draws_a_valid_pi_tick_figure_with_no_leak():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen(tier, seed)
            fig = q.figure
            assert fig is not None and figures.validate(fig) == [], (tier, seed)
            figures.assert_no_leak(fig, [q.answer_text, *q.distractors])
            assert fig["x_tick_format"] == "pi" and fig["equal_scale"] is False
            assert fig["y_range"] == [-3, 3] and fig["y_grid"] == 1
            assert len(fig["curves"]) == CURVES[tier]
            assert all(len(p.get("label", "")) <= 1 for p in fig["points"])
            assert fig["alt"].split()[-4:] == ["on", "a", "coordinate", "grid"]
            assert not any(w in fig["alt"] for w in ("sin", "cos", "tan"))


def test_deterministic_at_every_tier():
    for tier in TIERS:
        for seed in SEEDS:
            assert _gen(tier, seed).to_json() == _gen(tier, seed).to_json()


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")


# ------------------------------------------------------------------ the real grader


@pytest.mark.parametrize("tier", TIERS)
def test_mcq_grader_accepts_the_key_and_rejects_every_other_option(tier):
    for seed in SEEDS:
        q = _gen(tier, seed)
        options = mcq_options(q)
        assert sorted(options) == OPTIONS, (tier, seed, options)
        assert len(q.distractors) == 2 and q.answer_text not in q.distractors
        for idx, opt in enumerate(options):
            assert grade_mcq(q, str(idx)).correct == (opt == q.answer_text), (tier, seed, idx)
        assert q.answer_latex.startswith("y=\\")


@pytest.mark.parametrize("tier", TIERS)
def test_class_draw_and_key_position_are_balanced_at_400_seeds(tier):
    pos, cls = collections.Counter(), collections.Counter()
    n = 400
    for seed in range(n):
        q = _gen(tier, seed)
        pos[mcq_options(q).index(q.answer_text)] += 1
        cls[q.answer_text] += 1
    assert len(cls) == 3 and len(pos) == 3
    assert max(pos.values()) / n < 0.42 and min(pos.values()) / n > 0.25, (tier, dict(pos))
    assert max(cls.values()) / n < 0.42 and min(cls.values()) / n > 0.25, (tier, dict(cls))


# ------------------------------------------------------------------ the reader


@pytest.mark.parametrize("tier", TIERS)
def test_registered_reader_equals_the_key_1000_seeds(tier):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in range(1000):
        q = _gen(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


def test_pure_figure_reading_does_not_use_the_answer():
    q = _gen("exam", 4)
    key = q.answer_text
    q.answer_text, q.answer, q.distractors = "y = zzz", "y = zzz", []
    assert gen._read_from_figure(q.figure, q.text) == key


def test_in_generator_check_fires_when_the_window_hides_the_feature(monkeypatch):
    monkeypatch.setattr(gen, "_shows_distinguishing_feature", lambda *a: False)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, tier="easy")


def test_in_generator_check_fires_when_the_figure_is_read_as_another_function(monkeypatch):
    monkeypatch.setattr(gen, "_classify", lambda curve: "sin")
    with pytest.raises(VerificationError):
        for seed in range(30):
            make(TOPIC, seed=seed, tier="easy")


def test_classifier_refuses_a_curve_that_is_none_of_the_three():
    q = _gen("easy", 3)
    bent = {"pieces": [[[x, y * 0.9] for x, y in piece] for piece in q.figure["curves"][0]["pieces"]]}
    with pytest.raises(VerificationError):
        gen._classify(bent)


# ------------------------------------------------------------------ boundaries per tier


def test_every_window_contains_the_origin_and_shows_the_distinguishing_feature():
    for tier in TIERS:
        for seed in range(80):
            q = _gen(tier, seed)
            lo, hi = q.figure["x_range"]
            assert lo < 0 < hi, (tier, seed)
            for curve in q.figure["curves"]:
                ys = [y for piece in curve["pieces"] for _x, y in piece]
                x0 = {x: y for piece in curve["pieces"] for x, y in piece}
                if max(ys) > 2.4:  # tan: leaves the grid through the top and bottom
                    assert min(ys) < -2.4
                elif abs(x0.get(0, 9) - 1) < 1e-9:  # cos: a maximum at x = 0
                    assert max(ys) <= 1.0000001
                else:  # sin: a zero at x = 0, rising
                    assert abs(x0[0]) < 1e-9 and x0[0.025] > 0


def test_medium_window_is_narrower_than_easy_and_easy_spans_minus_2pi_to_2pi():
    easy_widths = {tuple(_gen("easy", s).figure["x_range"]) for s in range(40)}
    medium_widths = {tuple(_gen("medium", s).figure["x_range"]) for s in range(80)}
    assert easy_widths == {(-2, 2)}
    assert all(hi - lo < 4 for lo, hi in medium_widths) and len(medium_widths) >= 2


def test_locator_point_lies_on_exactly_one_curve_with_clearance():
    for tier in ("hard", "exam"):
        for seed in range(150):
            q = _gen(tier, seed)
            (p,) = q.figure["points"]
            assert p["label"] == "P"
            window = q.figure["x_range"]
            near = [gen._norm_distance(c, p["at"], window) for c in q.figure["curves"]]
            assert sorted(d < 1e-9 for d in near).count(True) == 1, (tier, seed, near)
            assert min(d for d in near if d >= 1e-9) >= gen._MIN_SEPARATION - 1e-9
            u, y = p["at"]
            assert (u * 4) == int(u * 4) and y == int(y)  # a grid intersection (x_grid 1/4)


def test_hard_draws_two_different_functions_and_exam_all_three():
    for seed in range(60):
        h = _gen("hard", seed)
        exam = _gen("exam", seed)
        assert len({gen._classify(c) for c in h.figure["curves"]}) == 2
        assert {gen._classify(c) for c in exam.figure["curves"]} == {"sin", "cos", "tan"}


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
def test_prompts_never_name_the_answer_or_a_parameter(tier):
    for seed in SEEDS:
        q = _gen(tier, seed)
        assert "amplitude" not in q.text and "period" not in q.text
        assert "**" not in q.text and "*" not in q.text
        assert not any(ch.isdigit() for ch in q.text)
