"""Fraction_of_quantity's hard-tier `find_whole` mode gives the part and asks for the whole -- the reverse of
what easy/medium ask."""

import mathgen


def test_find_whole_answer_is_the_original_quantity():
    for seed in range(15):
        q = mathgen.make("fraction_of_quantity", seed=seed, max_den=12, quantity_hi=144, find_whole=True)
        assert "of a number is" in q.text
        assert "What is the number?" in q.text


def test_find_whole_default_is_false_and_unchanged():
    a = mathgen.make("fraction_of_quantity", seed=5, max_den=12, quantity_hi=144)
    b = mathgen.make("fraction_of_quantity", seed=5, max_den=12, quantity_hi=144, find_whole=False)
    assert a.text == b.text
    assert a.answer == b.answer
