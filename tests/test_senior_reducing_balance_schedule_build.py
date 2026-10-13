"""Generator-specific proofs for `senior-use-a-recurrence-relation-to-model-a-reducing` (`answer: table`).

The independent oracle (it reads only the question text) is `tests/test_senior_reducing_balance_schedule_oracle.py`.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

import mathgen.generators.senior_reducing_balance_schedule_build as gen
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

SKILL_ID = "senior-use-a-recurrence-relation-to-model-a-reducing"
TOPIC = "senior_reducing_balance_schedule"
SKILL = load_skill(SKILL_ID)
SEEDS = range(50)


def _filled(q):
    rows = [list(r) for r in q.answer["rows"]]
    for key, value in q.answer["blanks"].items():
        r, c = map(int, key.split(","))
        rows[r][c] = value
    return rows


def _cents(text):
    return int(Fraction(text) * 100)


def _blank_rows(q):
    return sorted({int(k.split(",")[0]) for k in q.answer["blanks"]})


def test_skill_shape():
    assert_skill_shape(SKILL, TOPIC, "ACMGM097", used_notes(SKILL))


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in range(15):
        assert make_tier(SKILL, tier, seed).to_json() == make_tier(SKILL, tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_each_row_follows_the_recurrence_and_rows_chain(tier):
    for seed in SEEDS:
        rows = _filled(make_tier(SKILL, tier, seed))
        body = [r for r in rows if r[0] != "Total"]
        for r in body:
            opening, interest, repay, closing = (_cents(x) for x in r[1:5])
            assert opening + interest - repay == closing
            assert closing > repay > 0 and interest > 0
        for prev, nxt in zip(body, body[1:]):
            assert prev[4] == nxt[1]
        assert len({r[3] for r in body}) == 1, "equal instalments"
        if rows[-1][0] == "Total":
            assert _cents(rows[-1][2]) == sum(_cents(r[2]) for r in body)
            assert _cents(rows[-1][3]) == sum(_cents(r[3]) for r in body)


def test_blanks_are_a_block_at_the_foot_never_readable_from_a_given_cell():
    for tier in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = make_tier(SKILL, tier, seed)
            data_rows = [i for i, r in enumerate(q.answer["rows"]) if r[0] != "Total"]
            blank_rows = [i for i in data_rows if any(k.startswith(f"{i},") for k in q.answer["blanks"])]
            first, last = min(blank_rows), max(blank_rows)
            assert last == data_rows[-1]
            assert blank_rows == list(range(first, last + 1))
            # every period from the first blank one to the last has its interest and closing balance blank
            for r in blank_rows:
                assert f"{r},2" in q.answer["blanks"] and f"{r},4" in q.answer["blanks"]
            # the opening balance that starts the block is given, so the block cannot be read backwards
            assert f"{first},1" not in q.answer["blanks"]
            assert f"{first},3" not in q.answer["blanks"]


def test_tier_structure():
    for seed in SEEDS:
        easy, medium, hard, exam = (make_tier(SKILL, t, seed) for t in TIERS)
        assert len(easy.answer["rows"]) in (3, 4, 5) and len(easy.answer["blanks"]) == 2
        assert len(medium.answer["rows"]) == 4 and len(medium.answer["blanks"]) in (5, 8)
        assert len(hard.answer["rows"]) == 6 and "Total" == hard.answer["rows"][-1][0]
        assert 8 <= len(hard.answer["blanks"]) <= 14
        assert len(exam.answer["rows"]) == 4 and len(exam.answer["blanks"]) == 10
        first_row_blanks = {k for k in exam.answer["blanks"] if k.startswith("0,")}
        assert first_row_blanks == {"0,3"}, "only the first repayment is left out of the first row"


def test_hard_uses_an_annual_rate_and_names_the_compounding():
    seen = set()
    for seed in range(60):
        text = make_tier(SKILL, "hard", seed).text
        assert "per annum" in text
        seen.add("monthly" if "added monthly" in text else "quarterly")
    assert seen == {"monthly", "quarterly"}


def test_medium_gives_the_recurrence_and_exam_states_neither_rate_nor_repayment():
    for seed in SEEDS:
        assert "A(n+1) = 1." in make_tier(SKILL, "medium", seed).text
        exam = make_tier(SKILL, "exam", seed).text
        assert "%" not in exam.split("Table (")[0]
        assert "$" not in exam.split("Table (")[0]


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
def test_prompt_states_the_rounding_and_precision(tier):
    for seed in SEEDS:
        text = make_tier(SKILL, tier, seed).text
        assert "nearest cent" in text and "to the cent" in text


@pytest.mark.parametrize("tier", TIERS)
def test_answers_vary(tier):
    qs = assert_variety(SKILL, tier)
    assert len({q.text for q in qs}) >= 190


def test_unknown_tier_is_an_error():
    with pytest.raises(ValueError):
        gen.senior_reducing_balance_schedule(random.Random(0), tier="extreme")
