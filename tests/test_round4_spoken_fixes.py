"""Round 4 spoken-answer fixes (D7 pair/list count, order and negation; D8 relative tolerance; D13 ordinals)."""

from types import SimpleNamespace

import pytest

from mathgen import engine
from mathgen.grading import spoken


def _skill(skill_id):
    return next(s for s in engine.load_skills()[1] if s.id == skill_id)


def q(answer, kind=None):
    return SimpleNamespace(answer=answer, answer_kind=kind)


def test_pair_requires_order_count_and_no_negation():
    pair = q({"share_p": 9, "share_q": 15})
    assert spoken.check_answer(pair, "nine fifteen")
    assert spoken.check_answer(pair, "nine and fifteen")
    for bad in ("fifteen nine", "not nine not fifteen", "nine", "nine fifteen twenty", "nine fifteen nine"):
        assert not spoken.check_answer(pair, bad), bad
    counting = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    assert not spoken.check_answer(pair, counting)


def test_pair_on_real_skill():
    skill = _skill("divide-quantity-in-ratio")
    question = engine.make_for_skill(skill, 0, source=0)
    values = list(question.answer.values())
    words = " ".join(str(v) for v in values)
    assert spoken.check_answer(question, words)
    assert not spoken.check_answer(question, " ".join(str(v) for v in reversed(values)))
    assert not spoken.check_answer(question, f"not {values[0]} not {values[1]}")


def test_unordered_list_free_order_but_counted_and_negation():
    s = q([3, 7, 12], "set")
    assert spoken.check_answer(s, "twelve three seven")
    assert not spoken.check_answer(s, "three seven")
    assert not spoken.check_answer(s, "three seven twelve twelve")
    assert not spoken.check_answer(s, "three seven not twelve")


def test_ordered_list_requires_order():
    o = q([3, 7, 12], "ordered_list")
    assert spoken.check_answer(o, "three seven twelve")
    assert not spoken.check_answer(o, "twelve seven three")


@pytest.mark.parametrize(
    "key,reply,expected",
    [
        (0.03, "zero point zero three", True),
        (0.03, "point zero three", True),
        (0.03, "zero point zero two", False),
        (0.03, "zero", False),
        (0.03, "zero point zero eight", False),
        (0.03, "zero point zero one", False),
        (0.03, "three hundredths", True),
        (12, "twelve point zero five", True),
        (12, "eleven point nine five", True),
        (12, "twelve point one", False),
        (12, "twelve", True),
        (0.3, "zero point three", True),
        (0.3, "zero point three four", True),
        (12.34, "twelve point three four", True),
        (12.34, "twelve point four", False),
        (0.0004, "zero point zero zero zero four", True),
        (0.0004, "zero point one", False),
        (0.5, "zero point five", True),
        (0.5, "zero point five four", True),
    ],
)
def test_tolerance_relative_to_displayed_places(key, reply, expected):
    assert spoken.check_answer(q(key), reply) is expected


def test_fraction_key_tolerance():
    third = q({"p": 1, "q": 3})
    assert spoken.check_answer(third, "one third")
    assert not spoken.check_answer(third, "zero")
    two_thirds = q({"p": 2, "q": 3})
    assert spoken.check_answer(two_thirds, "zero point six seven")
    assert spoken.check_answer(two_thirds, "zero point seven")


def test_f10_decimal_notation_small_key():
    skill = _skill("f10-decimal-notation-tenths-hundredths")
    question = engine.make_for_skill(skill, 21, source=0)
    assert question.answer == 0.03
    assert not spoken.check_answer(question, "zero")
    assert not spoken.check_answer(question, "zero point zero eight")
    assert spoken.check_answer(question, "zero point zero three")


@pytest.mark.parametrize(
    "reply,p,qq",
    [
        ("one fiftieth", 1, 50),
        ("one twentieth", 1, 20),
        ("one hundredth", 1, 100),
        ("a hundredth", 1, 100),
        ("three hundredths", 3, 100),
        ("seven thousandths", 7, 1000),
        ("one thirtieth", 1, 30),
        ("three quarters", 3, 4),
        ("seven twelfths", 7, 12),
    ],
)
def test_ordinal_denominators(reply, p, qq):
    from fractions import Fraction

    assert spoken.extract_values(reply) == [Fraction(p, qq)]
    assert spoken.check_answer(q({"p": p, "q": qq}), reply)
