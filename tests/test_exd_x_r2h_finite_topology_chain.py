"""Tests for the multi-part exam source `exd_x_r2h_finite_topology_chain` (skill `ucv1-topology-basis-membership`).

The oracle reads the universe, the basis and the subset out of the rendered text, lists every union of basis members
as the open sets, and finds the interior and the closure from the open sets directly.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks
from itertools import chain, combinations

TOPIC = "exd_x_r2h_finite_topology_chain"
SKILL_ID = "ucv1-topology-basis-membership"
HARD_TOPIC = "p15c_topology_basis_membership"
LABELS = list("abcd")
MARKS = [2, 1, 1, 2]
FT_LABELS = ["d"]


def _set(text):
    return frozenset(int(v) for v in text.split(","))


def parse(q):
    universe = _set(re.search(r"X = \{([\d, ]+)\}", q.text).group(1))
    basis = [_set(body) for body in re.findall(r"B\d+ = \{([\d, ]+)\}", q.text)]
    a_set = _set(re.search(r"\bA = \{([\d, ]+)\}", q.text).group(1))
    return universe, basis, a_set


def oracle(q):
    universe, basis, a_set = parse(q)
    combos = chain.from_iterable(combinations(basis, k) for k in range(len(basis) + 1))
    opens = {frozenset().union(*combo) for combo in combos}
    assert universe in opens
    interior = frozenset().union(*[o for o in opens if o <= a_set])
    closure = frozenset(x for x in universe if all(o & a_set for o in opens if x in o))
    return {
        "a": str(len(opens)),
        "b": str(len(interior)),
        "c": str(len(closure)),
        "d": str(len(closure) - len(interior)),
    }


def test_follow_through_from_wrong_closure():
    def wrong_chain(q):
        key_b = int(q.parts[1].answer_text)
        wrong = int(q.parts[2].distractors[0])
        replies = {
            "a": q.parts[0].answer_text,
            "b": q.parts[1].answer_text,
            "c": str(wrong),
            "d": str(wrong - key_b),
        }
        return "c", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_interior():
    def wrong_chain(q):
        key_c = int(q.parts[2].answer_text)
        wrong = int(q.parts[1].distractors[0])
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(wrong),
            "c": q.parts[2].answer_text,
            "d": str(key_c - wrong),
        }
        return "b", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        assert grade(skill, q, {"a": "1009", "b": "211", "c": "307", "d": "401"}).marks == 0


CHECKS = standard_checks(TOPIC, SKILL_ID, oracle, LABELS, MARKS, FT_LABELS, HARD_TOPIC)


def test_wired():
    CHECKS["wired"]()


def test_shape():
    CHECKS["shape"]()


def test_prompt_shapes():
    CHECKS["shapes"]()


def test_oracle_agrees_over_300_seeds():
    CHECKS["oracle_agrees"]()


def test_composed_answer_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()
