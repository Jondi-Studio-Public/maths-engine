"""Tests for `scatterplot-describe-association` (ACMGM??? bivariate association): direction, form, strength, outliers."""

from __future__ import annotations

import pytest

from mathgen import figures, make
from mathgen.generators._sr2a_kit import corr, curvature_gain, strength_band
from tests.sr2a_support import (
    TIERS,
    check_figure,
    check_follow_through,
    check_question_grading,
    check_tagged,
    distinct_texts,
    questions,
    top_share,
)

TOPIC = "sr2a_scatter_assoc"
CHAIN = "sr2a_scatter_assoc_chain"


def _xy(q):
    pts = q.figure["points"]
    return [p["at"][0] for p in pts], [p["at"][1] for p in pts], [p.get("label") for p in pts]


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)
        check_figure(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert len({repr(make(TOPIC, seed=s, difficulty=tier).figure) for s in range(1, 41)}) >= 38
    assert distinct_texts(TOPIC, tier) >= 4
    assert top_share(TOPIC, tier) <= (0.4 if tier == "hard" else 0.3)


def test_easy_matches_the_correlation_and_keeps_clear_of_band_edges():
    for q in questions(TOPIC, "easy", range(1, 61)):
        xs, ys, _ = _xy(q)
        r = corr(xs, ys)
        assert q.answer_text == f"{'positive' if r > 0 else 'negative'}, {strength_band(r)}"
        assert all(abs(abs(r) - e) > 0.04 for e in (0.5, 0.75)), r


def test_medium_form_is_clear():
    seen = set()
    for q in questions(TOPIC, "medium", range(1, 81)):
        xs, ys, _ = _xy(q)
        gain = curvature_gain(xs, ys)
        word = q.answer_text.split(", ")[1]
        seen.add(word)
        assert (gain >= 0.045) == (word == "non-linear"), (q.seed, gain)
        assert gain <= 0.03 or gain >= 0.06, (q.seed, gain)
    assert seen == {"linear", "non-linear"}


def test_hard_outlier_effect_is_computed_from_the_data():
    effects = set()
    for q in questions(TOPIC, "hard", range(1, 81)):
        xs, ys, labels = _xy(q)
        i = labels.index("P")
        r = corr(xs, ys)
        r2 = corr([x for k, x in enumerate(xs) if k != i], [y for k, y in enumerate(ys) if k != i])
        effect = "stronger" if abs(r2) > abs(r) else "weaker"
        assert q.answer_text == f"{'positive' if r > 0 else 'negative'}, {strength_band(r)}, {effect}"
        assert abs(abs(r2) - abs(r)) > 0.05
        effects.add(effect)
    assert effects == {"stronger", "weaker"}


def test_exam_chain_values_and_follow_through():
    honoured = 0
    for q in questions(CHAIN, "exam"):
        assert [p.answer_kind for p in q.parts] == ["ordered_list", "number", "number", "mcq"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        assert not figures.validate(q.figure)
        assert figures.get_reader(CHAIN)(q) == "; ".join(f"({p.label}) {p.answer_text}" for p in q.parts)
        b, c = int(q.parts[1].answer_text), int(q.parts[2].answer_text)
        assert b + c == 100
    assert honoured > 10
