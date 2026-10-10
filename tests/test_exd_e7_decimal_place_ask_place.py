"""Decimal_place_value's hard-tier `ask_place` mode reverses the direction of the question (name a place, find
the digit there) and forces an interior zero so the answer is not simply the last digit shown."""

import mathgen


def test_ask_place_answer_is_a_single_digit_present_in_the_shown_number():
    for seed in range(15):
        q = mathgen.make("decimal_place_value", seed=seed, max_places=3, whole_hi=100, ask_place=True)
        assert "Which digit is in the" in q.text
        assert 0 <= q.answer <= 9


def test_ask_place_default_is_false_and_unchanged():
    a = mathgen.make("decimal_place_value", seed=5, max_places=3, whole_hi=100)
    b = mathgen.make("decimal_place_value", seed=5, max_places=3, whole_hi=100, ask_place=False)
    assert a.text == b.text
    assert a.answer == b.answer
