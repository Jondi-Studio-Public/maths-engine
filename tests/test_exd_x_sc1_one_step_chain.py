"""Exam source of `one-step-equations`: three linked one-step equations in a story."""

from __future__ import annotations

import re

from mathgen.engine import grade
from tests import exd_x_sc1_support as sup

SKILL_ID = "one-step-equations"
TOPIC = "exd_x_sc1_one_step_chain"
_GROUP = re.compile(r"(\d+) [A-Za-z ]+? (?:cost|costs|pays) \$(\d+)")


def facts(q):
    """(count, total, budget, share) read from the rendered stem."""
    n, total = (int(v) for v in _GROUP.search(q.stem).groups())
    amounts = sup.money(q.stem)
    assert len(amounts) == 3 and amounts[0] == total
    return n, amounts[0], amounts[1], amounts[2]


def oracle(q):
    n, total, budget, share = facts(q)
    assert total % n == 0
    price = total // n
    assert budget % price == 0
    count = budget // price
    return {"a": price, "b": count, "c": share * count}


def test_wiring_shapes_and_text():
    sup.check_wiring(SKILL_ID, TOPIC)
    sup.check_shapes(SKILL_ID)
    sup.check_no_text_defects(SKILL_ID)


def test_shape_of_the_question():
    for seed in range(60):
        q = sup.exam(SKILL_ID, seed)
        assert [p.marks for p in q.parts] == [1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = sup.exam(SKILL_ID, seed)
        assert {p.label: int(p.answer_text) for p in q.parts} == oracle(q), (seed, q.text)


def test_grading_and_tags():
    sup.check_grading(SKILL_ID)
    sup.check_tags(SKILL_ID)


def test_follow_through_from_a_wrong_price():
    """A wrong price used correctly gives the learner's own count, and a wrong count their own total."""
    checked = 0
    for seed in range(150):
        q = sup.exam(SKILL_ID, seed)
        n, _total, budget, share = facts(q)
        price = oracle(q)["a"]
        # a nearby price that divides the budget, so the count is whole
        options = [d for d in range(1, budget + 1) if budget % d == 0 and d != price and abs(d - price) <= price]
        if not options:
            continue
        bad = min(options, key=lambda d: abs(d - price))
        count = budget // bad
        total = share * count
        if len({bad, count, total}) < 3:
            continue
        result = grade(sup.skill(SKILL_ID), q, {"a": str(bad), "b": str(count), "c": str(total)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["c"].marks == 2, (seed, bad, count, total)
        checked += 1
    assert checked >= 50
