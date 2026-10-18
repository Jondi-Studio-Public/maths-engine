"""Tests for the multi-part exam source of senior-solve-practical-problems-requiring-basic-number-operations: a
group purchase with two prices, a percentage discount, a percentage addition and an equal split among the adults.
Keys are re-derived from the rendered stem with exact fractions."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import mathgen.generators.exd_x_r2i_group_purchase  # noqa: F401  (registers the topic)
from tests.exd_x_r2i_support import exam, follow_through_check, standard_checks

TOPIC = "exd_x_r2i_group_purchase"
SKILL = "senior-solve-practical-problems-requiring-basic-number-operations"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 1, 1, 2]
FT_LABELS = {"b", "c", "d"}


def _money(x: Fraction) -> str:
    return str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _facts(q):
    n1, p1, n2, p2 = re.search(
        r": (\d+) adult \w+ at \$(\d+\.\d\d) each and (\d+) child \w+ at \$(\d+\.\d\d) each", q.stem
    ).groups()
    discount = int(re.search(r"A (\d+)% group discount", q.stem).group(1))
    addition = int(re.search(r"and then (?:a )?(\d+)% (?:GST|service charge) is added", q.stem).group(1))
    adults = int(re.search(r"among the (\d+) adults", q.stem).group(1))
    assert adults == int(n1)
    subtotal = int(n1) * Fraction(p1) + int(n2) * Fraction(p2)
    after_discount = subtotal * (100 - discount) / 100
    total = after_discount * (100 + addition) / 100
    return subtotal, discount, after_discount, addition, total, adults


def oracle(q):
    subtotal, discount, after_discount, addition, total, adults = _facts(q)
    return {
        "a": _money(subtotal),
        "b": _money(after_discount),
        "c": _money(total),
        "d": _money(total / adults),
    }


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, FT_LABELS)


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_oracle_agrees_with_every_part():
    CHECKS["oracle_agrees"]()


def test_composed_reply_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_are_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_children_do_not_share_the_cost():
    for s in range(120):
        q = exam(TOPIC, s)
        subtotal, discount, after_discount, addition, total, adults = _facts(q)
        assert "the children pay nothing" in q.stem
        assert q.parts[3].prompt.startswith("How much does each adult pay")
        assert total * 100 == int(total * 100) and after_discount * 100 == int(after_discount * 100)


def test_wrong_subtotal_carries_through_discount_addition_and_share():
    def chain(q):
        subtotal, discount, after_discount, addition, total, adults = _facts(q)
        wrong_a = subtotal + 10
        wrong_b = wrong_a * (100 - discount) / 100
        wrong_c = wrong_b * (100 + addition) / 100
        replies = {
            "a": _money(wrong_a),
            "b": _money(wrong_b),
            "c": _money(wrong_c),
            "d": _money(Fraction(_money(wrong_c)) / adults),
        }
        return "a", replies, sum(MARKS) - MARKS[0]

    follow_through_check(TOPIC, SKILL, chain)


def test_wrong_discounted_cost_carries_into_the_rest():
    def chain(q):
        subtotal, discount, after_discount, addition, total, adults = _facts(q)
        wrong_b = subtotal * (100 + discount) / 100  # the discount applied the wrong way
        wrong_c = Fraction(_money(wrong_b)) * (100 + addition) / 100
        replies = {
            "a": _money(subtotal),
            "b": _money(wrong_b),
            "c": _money(wrong_c),
            "d": _money(Fraction(_money(wrong_c)) / adults),
        }
        return "b", replies, sum(MARKS) - MARKS[1]

    follow_through_check(TOPIC, SKILL, chain)
