"""Tests for the multi-part exam source of ucv1-compactness-finite-check: essential members, smallest subcover,
discards and the number of subcovers of a finite open cover. Keys are re-derived from the rendered stem."""

from __future__ import annotations

import re
from itertools import combinations

import mathgen.generators.exd_x_r2g_subcover_count  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import follow_through_check, standard_checks

TOPIC = "exd_x_r2g_subcover_count"
SKILL = "ucv1-compactness-finite-check"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 1, 2]


def _parse(q):
    x = set(re.search(r"X = \{([^}]*)\}", q.stem).group(1).split(", "))
    sets = [set(m.split(", ")) for m in re.findall(r"U\d+ = \{([^}]*)\}", q.stem)]
    return x, sets


def _facts(q):
    x, sets = _parse(q)
    n = len(sets)
    essential = sum(1 for i, s in enumerate(sets) if s - set().union(*(t for j, t in enumerate(sets) if j != i)))
    covers = [c for r in range(1, n + 1) for c in combinations(range(n), r) if set().union(*(sets[i] for i in c)) == x]
    return n, essential, min(len(c) for c in covers), len(covers)


def oracle(q):
    n, essential, smallest, count = _facts(q)
    return {"a": str(essential), "b": str(smallest), "c": str(n - smallest), "d": str(count)}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"c"}, "p15b_compactness_subcover_check")


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_keys_match_independent_derivation():
    CHECKS["oracle_agrees"]()


def test_composed_answer_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_rejected_and_tagged():
    CHECKS["distractors_rejected"]()


def test_follow_through_from_wrong_smallest_subcover():
    """A smallest size one too large in (b) is carried correctly into (c)."""

    def wrong_chain(q):
        n, essential, smallest, count = _facts(q)
        wrong = smallest + 1
        if n - wrong == wrong:
            return None
        replies = {"a": str(essential), "b": str(wrong), "c": str(n - wrong), "d": str(count)}
        return "b", replies, 5

    follow_through_check(TOPIC, SKILL, wrong_chain)


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable("exd_x_r2g_subcover_count", "ucv1-compactness-finite-check")


def test_smallest_subcover_size_is_spread():
    """Part (b), the size of a smallest subcover, is 3, 4 or 5 in a spread (it was 3 on about three seeds in four)."""
    from tests.exd_junk_probe import assert_answer_spread

    assert_answer_spread("exd_x_r2g_subcover_count", "b", at_least=3)
