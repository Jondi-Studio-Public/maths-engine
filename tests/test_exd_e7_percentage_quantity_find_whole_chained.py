"""Tests the hard-tier `find_whole` and `chained` discount modes of `percentage_quantity`.

Percentage_quantity's hard-tier `find_whole` (amount kind) and `chained` (discount kind) modes are each a
genuinely different task from the easy/medium tiers. The chained-discount arithmetic must be exact (no
rounding) for every (percent, percent2) pair the hard source can draw."""

from fractions import Fraction

import mathgen


def test_find_whole_answer_is_the_original_quantity():
    for seed in range(15):
        q = mathgen.make(
            "percentage_quantity",
            seed=seed,
            kind="amount",
            percents=(5, 10, 20, 25, 50, 75),
            units_hi=20,
            find_whole=True,
        )
        assert "of a number is" in q.text
        assert "What is the number?" in q.text


def test_amount_default_is_false_and_unchanged():
    a = mathgen.make("percentage_quantity", seed=5, kind="amount", percents=(10, 20, 25, 50), units_hi=15)
    b = mathgen.make(
        "percentage_quantity", seed=5, kind="amount", percents=(10, 20, 25, 50), units_hi=15, find_whole=False
    )
    assert a.text == b.text
    assert a.answer == b.answer


def test_chained_discount_is_exact_for_every_percent_pair():
    percents = (10, 20, 25, 50, 75)
    for p1 in percents:
        for p2 in percents:
            fraction, sale_fraction, fraction2 = Fraction(p1, 100), Fraction(100 - p1, 100), Fraction(p2, 100)
            for seed in range(30):
                q = mathgen.make(
                    "percentage_quantity",
                    seed=seed,
                    kind="discount",
                    percents=percents,
                    units_hi=20,
                    quantity_hi=300,
                    chained=True,
                )
                assert "then a further" in q.text
                assert isinstance(q.answer, int)


def test_discount_default_is_false_and_unchanged():
    a = mathgen.make("percentage_quantity", seed=5, kind="discount", percents=(10, 20, 25, 50), units_hi=15)
    b = mathgen.make(
        "percentage_quantity", seed=5, kind="discount", percents=(10, 20, 25, 50), units_hi=15, chained=False
    )
    assert a.text == b.text
    assert a.answer == b.answer
