"""`1 success` is a count of one with a singular noun, not a plural defect."""

from mathgen.text_quality import NOT_PLURAL_NOUNS, _singular_count_plural


def test_success_is_not_flagged_as_a_plural():
    assert "success" in NOT_PLURAL_NOUNS
    assert _singular_count_plural("The trial has 1 success in 10 attempts.") is None


def test_real_plurals_are_still_flagged():
    assert _singular_count_plural("There is 1 apples in the bag.") == "1 apples"
