"""Generator-specific proofs for `senior-calculate-the-dividend-paid-on-a-portfolio-of` (`answer: table`).

The independent oracle (it reads only the question text) is `tests/test_senior_dividend_portfolio_table_oracle.py`.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

import pytest

import mathgen.generators.senior_dividend_portfolio_table_build as gen
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

SKILL_ID = "senior-calculate-the-dividend-paid-on-a-portfolio-of"
TOPIC = "senior_dividend_portfolio_table"
SKILL = load_skill(SKILL_ID)
SEEDS = range(50)


def _filled(q):
    rows = [list(r) for r in q.answer["rows"]]
    for key, value in q.answer["blanks"].items():
        r, c = map(int, key.split(","))
        rows[r][c] = value
    return q.answer["columns"], rows


def _col(columns, name):
    return columns.index(name)


def test_skill_shape():
    assert_skill_shape(SKILL, TOPIC, "ACMGM008", used_notes(SKILL))


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in range(15):
        assert make_tier(SKILL, tier, seed).to_json() == make_tier(SKILL, tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_completed_table_is_consistent(tier):
    for seed in SEEDS:
        columns, rows = _filled(make_tier(SKILL, tier, seed))
        body, total = rows[:-1], rows[-1]
        paid = _col(columns, "Dividend paid ($)")
        assert sum(Fraction(r[paid]) for r in body) == Fraction(total[paid])
        if "Dividend per share ($)" in columns:
            dps = _col(columns, "Dividend per share ($)")
            for r in body:
                assert Fraction(r[1]) * Fraction(r[dps]) == Fraction(r[paid])
        if "P/E ratio" in columns:
            pe, price, eps = (_col(columns, c) for c in ("P/E ratio", "Share price ($)", "Earnings per share ($)"))
            for r in body:
                assert abs(Fraction(r[price]) / Fraction(r[eps]) - Fraction(r[pe])) <= Fraction(1, 20)
        if "Dividend yield (%)" in columns:
            y, dps, price = (
                _col(columns, c) for c in ("Dividend yield (%)", "Dividend per share ($)", "Share price ($)")
            )
            for r in body:
                assert abs(Fraction(r[dps]) / Fraction(r[price]) * 100 - Fraction(r[y])) <= Fraction(1, 20)


def test_table_shape_and_blanks_by_tier():
    for seed in SEEDS:
        easy, medium, hard, exam = (make_tier(SKILL, t, seed) for t in TIERS)
        for q, rows in ((easy, 4), (medium, 5), (hard, 5), (exam, 4)):
            assert len(q.answer["rows"]) == rows
            n_rows = rows - 1
            assert f"{n_rows},{len(q.answer['columns']) - 1}" in q.answer["blanks"] or tier_total_blank(q)
        assert 3 <= len(easy.answer["blanks"]) <= 4
        assert 4 <= len(medium.answer["blanks"]) <= 5
        assert 7 <= len(hard.answer["blanks"]) <= 9
        assert len(exam.answer["blanks"]) == 10


def tier_total_blank(q):
    n_rows = len(q.answer["rows"]) - 1
    return any(k.startswith(f"{n_rows},") for k in q.answer["blanks"])


def test_medium_mixes_dollar_and_percentage_dividends():
    for seed in SEEDS:
        q = make_tier(SKILL, "medium", seed)
        declared = [r[3] for r in q.answer["rows"][:-1]]
        assert any(d.endswith("% of share price") for d in declared)
        assert any(d.endswith("per share") for d in declared)


def test_hard_blanks_cover_each_kind_of_cell():
    for seed in SEEDS:
        q = make_tier(SKILL, "hard", seed)
        columns = q.answer["columns"]
        blank_columns = {columns[int(k.split(",")[1])] for k in q.answer["blanks"]}
        assert {"Dividend paid ($)", "Dividend yield (%)", "P/E ratio"} <= blank_columns


def test_exam_rows_leave_out_different_cells_and_include_a_backward_step():
    kinds = set()
    for seed in range(120):
        q = make_tier(SKILL, "exam", seed)
        columns = q.answer["columns"]
        per_row = []
        for r in range(3):
            cols = {columns[int(k.split(",")[1])] for k in q.answer["blanks"] if k.startswith(f"{r},")}
            assert len(cols) == 3
            per_row.append(frozenset(cols))
        assert len(set(per_row)) >= 2
        kinds |= set(per_row)
    assert {c for k in kinds for c in k} >= {"Share price ($)", "Shares held", "Dividend per share ($)"}


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_the_precision(tier):
    for seed in SEEDS:
        text = make_tier(SKILL, tier, seed).text
        assert "to the cent" in text
        if tier in ("hard", "exam"):
            assert "one decimal place" in text


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_each_form_and_rejects_every_distractor(tier):
    assert_grader_behaviour(SKILL, tier, SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_changing_or_dropping_any_one_blank_is_rejected(tier):
    assert_each_blank_is_graded(SKILL, tier, range(20))


@pytest.mark.parametrize("tier", TIERS)
def test_masked_prompt_hides_every_blank(tier):
    assert_masked_prompt(SKILL, tier, range(40))


@pytest.mark.parametrize("tier", TIERS)
def test_answers_and_blank_positions_vary(tier):
    qs = assert_variety(SKILL, tier)
    patterns = {frozenset(q.answer["blanks"]) for q in qs}
    if tier == "medium":
        assert len(patterns) >= 4  # only five cells can be blank
    if tier in ("hard", "exam"):
        assert len(patterns) >= 10


def test_no_reference_to_a_real_company():
    names = {q.answer["rows"][i][0] for q in (make_tier(SKILL, "easy", s) for s in SEEDS) for i in range(3)}
    assert len(names) >= 8
    assert not any(re.search(r"BHP|Woolworths|Telstra|Commonwealth|Qantas", n) for n in names)


def test_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        gen.senior_dividend_portfolio_table(random.Random(0), tier="extreme")
