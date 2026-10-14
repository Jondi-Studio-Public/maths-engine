"""Stem wording for the two number-skills exam sources: takings for sales, no singular verb for a plural food."""

from __future__ import annotations

import pytest

from mathgen import make


@pytest.mark.parametrize("seed", range(200))
def test_a_stall_that_sells_does_not_ask_for_a_cost(seed):
    q = make("exd_x_n2_leading_digit_error", seed=seed)
    if " sells " in q.stem:
        assert "total cost" not in q.parts[0].prompt


@pytest.mark.parametrize("seed", range(200))
def test_shop_stem_has_no_plural_food_with_singular_verb_or_dangling_sentence(seed):
    q = make("exd_x_n2_rates_unit_cost", seed=seed)
    assert " comes in " not in q.stem
    assert "wants the better value" not in q.stem
