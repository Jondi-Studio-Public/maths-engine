"""Generator-specific proofs for `senior-use-an-appropriately-percentaged-two-way-frequency-table`
(`answer: table`).

The independent oracle (it reads only the question text) is
`tests/test_senior_percentaged_two_way_table_oracle.py`.
"""

from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.senior_percentaged_two_way_table_build as gen
from tests.senior_table_support import (
    TIERS,
    assert_each_blank_is_graded,
    assert_grader_behaviour,
    assert_masked_prompt,
    assert_skill_shape,
    assert_variety,
    load_skill,
    make_tier,
    used_notes,
)

SKILL_ID = "senior-use-an-appropriately-percentaged-two-way-frequency-table"
TOPIC = "senior_percentaged_two_way_table"
SKILL = load_skill(SKILL_ID)
SEEDS = range(50)


def _layout(q):
    columns = q.answer["columns"]
    n_cols = sum(1 for c in columns if c.endswith("(%)"))
    return len(q.answer["rows"]) - 1, n_cols, n_cols + 2  # rows, categories across, first % column


def _filled(q):
    rows = [list(r) for r in q.answer["rows"]]
    for key, value in q.answer["blanks"].items():
        r, c = map(int, key.split(","))
        rows[r][c] = value
    return rows


def test_skill_shape():
    assert_skill_shape(SKILL, TOPIC, "ACMGM050", used_notes(SKILL))


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in range(15):
        assert make_tier(SKILL, tier, seed).to_json() == make_tier(SKILL, tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_counts_add_up_and_percentages_are_of_one_consistent_total(tier):
    for seed in SEEDS:
        q = make_tier(SKILL, tier, seed)
        n_rows, n_cols, first = _layout(q)
        rows = _filled(q)
        for r in range(n_rows):
            assert sum(rows[r][1 : 1 + n_cols]) == rows[r][1 + n_cols]
        for c in range(1, n_cols + 2):
            assert sum(rows[r][c] for r in range(n_rows)) == rows[n_rows][c]
        # the percentages are of the totals of one direction only: either every row or every column sums to 100
        pct = [[Fraction(rows[r][first + c]) for c in range(n_cols)] for r in range(n_rows)]
        row_sums_100 = all(abs(sum(pct[r]) - 100) <= Fraction(n_cols, 20) for r in range(n_rows))
        col_sums_100 = all(
            abs(sum(pct[r][c] for r in range(n_rows)) - 100) <= Fraction(n_rows, 20) for c in range(n_cols)
        )
        assert row_sums_100 != col_sums_100, (tier, seed)


def test_shapes_by_tier():
    shapes = {t: Counter() for t in TIERS}
    for seed in range(120):
        for t in TIERS:
            n_rows, n_cols, _ = _layout(make_tier(SKILL, t, seed))
            shapes[t][(n_rows, n_cols)] += 1
    assert set(shapes["easy"]) == {(2, 2)}
    assert set(shapes["medium"]) == {(2, 3), (3, 2)}
    assert set(shapes["hard"]) == {(3, 3)}
    assert set(shapes["exam"]) == {(3, 3), (2, 3), (3, 2)}


@pytest.mark.parametrize("tier", ["medium", "hard", "exam"])
def test_no_percentage_is_left_in_view_to_reveal_the_total(tier):
    for seed in SEEDS:
        q = make_tier(SKILL, tier, seed)
        n_rows, n_cols, first = _layout(q)
        for r in range(n_rows):
            for c in range(n_cols):
                assert f"{r},{first + c}" in q.answer["blanks"]
        # and the percentage totals are not shown at all
        assert q.answer["rows"][n_rows][first:] == ["-"] * n_cols
        assert not any(c.startswith("Total (%)") for c in q.answer["columns"])


def test_easy_leaves_two_or_three_percentages_to_fill():
    for seed in SEEDS:
        assert 2 <= len(make_tier(SKILL, "easy", seed).answer["blanks"]) <= 3


def test_exam_missing_counts_are_recoverable_from_the_totals():
    for seed in range(120):
        q = make_tier(SKILL, "exam", seed)
        n_rows, n_cols, first = _layout(q)
        missing = [tuple(map(int, k.split(","))) for k in q.answer["blanks"] if int(k.split(",")[1]) < first]
        assert 2 <= len(missing) <= min(n_rows, n_cols)
        assert len({r for r, _c in missing}) == len(missing)
        assert len({c for _r, c in missing}) == len(missing)
        assert all(1 <= c <= n_cols for _r, c in missing)


def test_the_groups_sit_on_either_axis():
    sides = Counter()
    for seed in range(200):
        q = make_tier(SKILL, "hard", seed)
        stem = q.text.split("Table (")[0]
        row_var = q.answer["columns"][0].split(" / ")[0]
        sides["rows" if f"depends on the {row_var} " in stem or f"depends on the {row_var}." in stem else "cols"] += 1
    assert sides["rows"] >= 50 and sides["cols"] >= 50


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_the_precision(tier):
    for seed in SEEDS:
        assert "1 decimal place" in make_tier(SKILL, tier, seed).text


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_each_form_and_rejects_every_distractor(tier):
    assert_grader_behaviour(SKILL, tier, SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_changing_or_dropping_any_one_blank_is_rejected(tier):
    assert_each_blank_is_graded(SKILL, tier, range(15))


@pytest.mark.parametrize("tier", TIERS)
def test_masked_prompt_hides_every_blank(tier):
    assert_masked_prompt(SKILL, tier, range(40))


@pytest.mark.parametrize("tier", TIERS)
def test_answers_vary(tier):
    qs = [make_tier(SKILL, tier, s) for s in range(200)]
    assert len({q.text for q in qs}) >= 190
    if tier != "easy":
        assert_variety(SKILL, tier)


def test_wrong_total_distractor_is_offered_on_every_question():
    for tier in TIERS:
        for seed in SEEDS:
            q = make_tier(SKILL, tier, seed)
            assert "percentages-of-the-wrong-total" in q.option_notes.values(), (tier, seed)


def test_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        gen.senior_percentaged_two_way_table(random.Random(0), tier="extreme")
