"""Regression tests for the 2026-10-19 bug hunt, integration pass: one display pipeline, worked solutions that
used to be empty, and a handful of leftover wording and distractor defects (worker integ)."""

import re
from fractions import Fraction

import pytest

from mathgen import engine
from mathgen.core import _display_text, _learner_text, display_question

_ERRS, _SKILLS = engine.load_skills()
SKILLS = {s.id: s for s in _SKILLS}


def _make(skill_id, seed, source):
    return engine.make_for_skill(SKILLS[skill_id], seed, source=source)


def _units(q):
    """The question itself, or each of its parts."""
    return list(getattr(q, "parts", None) or [q])


# --- the display pipeline is one idempotent function --------------------------------------------------------

_PIPELINE_SAMPLES = [
    "y - (-51) = -1(x - 3), so y = -x - 48.",
    "Tie -- the same price",
    "a 8 cm side and 16 cm2 of tiles",
    "1 days, 9/-1 and x - -23",
    "20 x 12 and 3 - 1i, 0x + 5",
    "so 12 = 12",
    "Part 1(a) of 1(b)",
]


def test_display_text_is_the_learner_text_pipeline():
    assert _display_text is _learner_text


@pytest.mark.parametrize("text", _PIPELINE_SAMPLES)
@pytest.mark.parametrize("working", [False, True])
def test_display_pipeline_is_idempotent(text, working):
    once = _learner_text(text, working=working)
    assert _learner_text(once, working=working) == once


@pytest.mark.parametrize(
    "raw, shown",
    [
        ("y - (-51) = -1(x - 3), so y = -x - 48.", "y - (-51) = -(x - 3), so y = -x - 48."),
        ("(d) f = 1(15/2)² + 4(-5/4)²", "(d) f = (15/2)² + 4(-5/4)²"),
        ("3 + 1(x - 1) = 0", "3 + (x - 1) = 0"),
        ("Part 1(a) and 1(b)", "Part 1(a) and 1(b)"),
        ("a 21(x + 1) term", "a 21(x + 1) term"),
    ],
)
def test_coefficient_one_before_a_bracket(raw, shown):
    assert _learner_text(raw) == shown


# --- worked solutions that used to be empty -----------------------------------------------------------------

# (skill id, sources) -- every tier of each skill must now explain itself.
_NOW_EXPLAINED = [
    ("senior-determine-displacement-given-velocity-in-linear-motion-problems", range(4)),
    ("senior-determine-f-left-x-right-given-f-x", range(4)),
    ("senior-determine-positions-given-acceleration-and-initial-values-of", range(4)),
    ("senior-find-and-use-the-inverse-trigonometric-functions-arcsine", range(4)),
    ("senior-recognise-and-use-linearity-properties-of-the-derivative", range(4)),
    ("senior-use-the-leibniz-notation-delta-x-and-delta", [3]),
    ("senior-use-the-notation-frac-delta-y-delta-x", range(4)),
    ("senior-use-the-notation-int-f-left-x-right", range(4)),
    ("vector-cross-product", range(4)),
    ("vector-magnitude", range(3)),
    ("partial-derivative-x", range(3)),
    ("partial-derivative-y", range(3)),
    ("solve-ode-separable-poly", range(3)),
    ("solve-ode-exponential", range(3)),
    ("solve-simultaneous-equations", range(3)),
    ("matrix-solve-system", range(3)),
    ("matrix-determinant-3x3", range(2)),
    ("moment-of-a-force", [0]),
    ("senior-review-the-rules-p-left-overline-a-right", range(3)),
    ("senior-when-a-secant-meeting-the-circle-at-a", range(3)),
    ("rectangle-area-perimeter-mixed-units", range(3)),
    ("senior-review-metric-units-of-length-their-abbreviations-conversions", range(3)),
]
_EXPLAINED_CASES = [(sid, src) for sid, srcs in _NOW_EXPLAINED for src in srcs]


@pytest.mark.parametrize("skill_id, source", _EXPLAINED_CASES)
def test_every_tier_has_a_worked_solution_and_a_graded_key(skill_id, source):
    skill = SKILLS[skill_id]
    for seed in range(12):
        q = _make(skill_id, seed, source)
        shown = display_question(q)
        for unit in _units(q):
            assert unit.working, f"{skill_id} src{source} seed{seed} has no working"
        lines = list(shown["working"]) + [w for p in shown.get("parts", ()) for w in p["working"]]
        joined = "\n".join(lines)
        assert re.search(r"\bNone\b|\*\*|\{|\}|\.\.\.", joined) is None, joined
        # A coefficient of 1 is never written out in front of a bracket.
        assert re.search(r"(?<![\w.^/)\]])[-−]?1\((?=[^()]*[\d+/^])", joined) is None, joined
        if not getattr(q, "parts", None):
            assert engine.grade(skill, q, q.answer_text).correct
            for d in q.distractors:
                assert not engine.grade(skill, q, d).correct


# --- working that states only what the stem gives ------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_conditional_probability_working_uses_the_stem(seed):
    skill = "senior-use-the-notation-p-a-vert-b"
    hard = display_question(_make(skill, seed, 2))
    assert "Group:" not in "".join(hard["working"])
    assert "÷" in "".join(hard["working"])
    medium = display_question(_make(skill, seed, 1))
    assert "Sport-likers =" in medium["working"][0]


@pytest.mark.parametrize("source", [1, 2])
@pytest.mark.parametrize("seed", range(30))
def test_pascal_working_matches_the_position_asked(source, seed):
    shown = display_question(_make("senior-use-pascal-s-triangle-and-its-properties", seed, source))
    text = " ".join(shown["working"])
    row = re.search(r"value at position (\d+) in row (\d+)", shown["text"])
    assert row is not None
    assert f"position {row.group(1)} in row {row.group(2)}" in text or "missing value" in text.lower()
    assert shown["answer_text"] in text


@pytest.mark.parametrize("seed", range(30))
def test_choose_model_working_names_the_model_in_words(seed):
    shown = display_question(_make("f10-choose-model-from-data", seed, 2))
    joined = " ".join(shown["working"])
    assert "exp (" not in joined and "lin (" not in joined and "quad (" not in joined


@pytest.mark.parametrize("source", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", range(80))
def test_time_series_never_predicts_nothing(source, seed):
    q = _make("time-series-features", seed, source)
    for unit in _units(q):
        if "predict" in unit.text.lower():
            assert str(unit.answer_text).strip() not in ("0", "0.0")


# --- stems and options --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id, source",
    [
        ("prism-capacity-three-units", 3),
        ("senior-calculate-seasonal-indices-by-using-the-average-percentage", 3),
        ("senior-investigate-the-effect-of-outliers-on-the-mean", 3),
    ],
)
@pytest.mark.parametrize("seed", range(20))
def test_exam_stems_do_not_talk_about_themselves(skill_id, source, seed):
    assert "exam question" not in display_question(_make(skill_id, seed, source))["text"].lower()


@pytest.mark.parametrize("seed", range(20))
def test_outcome_classification_asks_for_the_most_specific_description(seed):
    text = _make("senior-review-the-concepts-and-language-of-outcomes-sample", seed, 0).text
    assert "most specific description" in text


@pytest.mark.parametrize("source", range(3))
@pytest.mark.parametrize("seed", range(40))
def test_target_population_options_cite_no_unstated_sample_size(source, seed):
    q = _make("senior-identify-the-target-population-to-be-surveyed", seed, source)
    for option in [q.answer_text, *q.distractors]:
        assert not re.search(r"the \d+ people who were actually surveyed", option)


@pytest.mark.parametrize("seed", range(30))
def test_unit_choice_does_not_tag_a_static_length_with_the_activity_verb(seed):
    for skill in ("senior-identify-the-appropriate-units-for-different-activities-such",):
        for source in range(len(SKILLS[skill].sources)):
            assert "(during " not in _make(skill, seed, source).text


@pytest.mark.parametrize("seed", range(30))
def test_land_chain_treatment_covers_the_whole_block(seed):
    q = _make("senior-convert-between-metric-units-of-area-and-other", seed, 3)
    assert "whole block" in q.text


# --- distractors --------------------------------------------------------------------------------------------


def _as_fraction(text):
    try:
        return Fraction(str(text).strip().replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


@pytest.mark.parametrize("source", range(4))
@pytest.mark.parametrize("seed", range(60))
def test_a_variance_is_never_offered_as_negative(source, seed):
    q = _make("finite-distribution-variance", seed, source)
    for unit in _units(q):
        for d in unit.distractors:
            value = _as_fraction(d)
            assert value is None or value >= 0, (unit.text, d)


@pytest.mark.parametrize("source", range(3))
@pytest.mark.parametrize("seed", range(60))
def test_an_area_is_never_offered_as_negative(source, seed):
    q = _make("senior-calculate-the-area-between-curves-in-simple-cases", seed, source)
    for d in q.distractors:
        value = _as_fraction(d)
        assert value is None or value >= 0, (q.text, d)


@pytest.mark.parametrize("source", range(3))
@pytest.mark.parametrize("seed", range(40))
def test_probability_rules_offer_only_possible_probabilities(source, seed):
    q = _make("senior-review-the-rules-p-left-overline-a-right", seed, source)
    assert len(q.distractors) >= 2
    for d in q.distractors:
        value = _as_fraction(d)
        assert value is not None and 0 <= value <= 1


@pytest.mark.parametrize("seed", range(40))
def test_triangular_prism_direct_offers_more_than_one_distractor(seed):
    q = _make("f10-triangular-prism-volume", seed, 0)
    assert len(q.distractors) >= 2


# --- grading hazards of the new wording are absent ----------------------------------------------------------


@pytest.mark.parametrize("source", [0, 1, 2])
@pytest.mark.parametrize("seed", range(25))
def test_spread_working_checks_every_option(source, seed):
    shown = display_question(_make("senior-use-informal-ways-of-describing-spread-such-as", seed, source))
    joined = " ".join(shown["working"])
    assert "Checking each option against the rule..." not in joined
    for label in "ABCD":
        assert f"Data set {label}:" in joined


@pytest.mark.parametrize("source", [0, 1, 3])
@pytest.mark.parametrize("seed", range(25))
def test_group_centre_working_computes_each_group(source, seed):
    shown = display_question(
        _make("senior-implement-the-statistical-investigation-process-to-answer-questions", seed, source)
    )
    joined = " ".join(shown["working"])
    assert "computed from the stated data" not in joined
    assert "values add to" in joined or "sorted values" in joined
