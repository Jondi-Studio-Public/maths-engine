"""Tests for the multi-part exam source of ucv1-definition-quantifier-check: elements without an image, surplus
pairs, the repair cost and the number of admissible functions. Keys are re-derived from the rendered stem."""

from __future__ import annotations

import re
from itertools import product

import mathgen.generators.exd_x_r2g_relation_repair  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import follow_through_check, standard_checks

TOPIC = "exd_x_r2g_relation_repair"
SKILL = "ucv1-definition-quantifier-check"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 2, 2]


def _facts(q):
    a_set = [int(x) for x in re.search(r"A = \{([^}]*)\}", q.stem).group(1).split(", ")]
    b_set = re.search(r"B = \{([^}]*)\}", q.stem).group(1).split(", ")
    pairs = [(int(x), y) for x, y in re.findall(r"\((\d+), ([a-z])\)", q.stem)]
    images = {x: {y for (u, y) in pairs if u == x} for x in a_set}
    none = sum(1 for v in images.values() if not v)
    surplus = sum(len(v) - 1 for v in images.values() if len(v) > 1)
    # count the admissible functions by direct enumeration of choices per element
    choices = [sorted(images[x]) if images[x] else b_set for x in a_set]
    total = 1
    for c in choices:
        total *= len(c)
    return none, surplus, total, choices


def oracle(q):
    none, surplus, total, _choices = _facts(q)
    return {"a": str(none), "b": str(surplus), "c": str(none + surplus), "d": str(total)}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"c", "d"}, "p15a_definition_quantifier_check")


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


def test_enumerated_function_count_for_small_cases():
    """Enumerate the admissible functions explicitly where the count is small enough."""
    from tests.exd_x_r2g_support import exam

    checked = 0
    for s in range(300):
        none, surplus, total, choices = _facts(exam(TOPIC, s))
        if total <= 20000:
            assert sum(1 for _ in product(*choices)) == total
            checked += 1
    assert checked >= 30


def test_follow_through_from_wrong_no_image_count():
    """A count of elements without an image that is one too high is carried into (c) and (d)."""

    def wrong_chain(q):
        none, surplus, total, _choices = _facts(q)
        b_size = len(re.search(r"B = \{([^}]*)\}", q.stem).group(1).split(", "))
        images_product = total // b_size**none  # the factor from elements that R gives several images
        wrong = none + 1
        replies = {
            "a": str(wrong),
            "b": str(surplus),
            "c": str(wrong + surplus),
            "d": str(b_size**wrong * images_product),
        }
        return "a", replies, 2 + 2 + 2

    follow_through_check(TOPIC, SKILL, wrong_chain)
