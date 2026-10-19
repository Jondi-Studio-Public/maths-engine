"""Independent oracle for f10-polygon-classification. Deliberately does NOT import anything from
``mathgen.generators.p14b_polygon_classification_build`` -- it parses
``question.text`` with its own regexes over the literal stated facts (side
lengths, angle measures, the parallel-side clause) and re-derives the class
from first principles via its OWN independent decision path, then compares
against ``question.answer_text``.

This is the "enumeration matrix" oracle shape for a classification skill: it sweeps enough seeds per tier to
enumerate every class the tier's pool can produce (not a five-seed sample), and for every row checks BOTH that
the oracle's own re-derived class matches the rendered answer AND that every other pool member is offered as a
distractor -- the matrix a defect like ``causality_correlation_classify``'s mismatched pairing would have
shown immediately.
"""

import re

import pytest

from mathgen import make

TOPIC = "p14b_polygon_classification"
SWEEP = range(300)
MEDIUM_SWEEP = range(1000)  # The medium-tier ambiguity fix needs its own >= 1000-seed proof

EASY_POOL = {"scalene", "isosceles", "equilateral"}
MEDIUM_POOL = {
    "scalene, not right-angled",
    "isosceles, not right-angled",
    "scalene right-angled",
    "isosceles right-angled",
}
HARD_POOL = {"parallelogram", "rhombus", "trapezium"}
EXAM_POOL = {"parallelogram", "rectangle", "rhombus", "square"}

# -- Independent parsing (own regexes, own decision path) --------------------

_THREE_UNITS_RE = re.compile(r"(\d+), (\d+) and (\d+) units")
_THREE_DEGREES_RE = re.compile(r"(\d+), (\d+) and (\d+) degrees")
# The medium stem states the side facts in words (no list of lengths, which could not agree with the angles).
_EQUAL_PAIR_RE = re.compile(r"two of its sides are each (\d+) units long and the third side is a different length")
_ALL_DIFFERENT_CLAUSE = "its three sides all have different lengths"


def _medium_is_isosceles(text: str) -> bool:
    """Read the side-equality fact from the stated clause, then require the stated angles to agree with it
    (a triangle has two equal sides exactly when it has two equal angles), so the fact is checked, not trusted."""
    pair, different = _EQUAL_PAIR_RE.search(text), _ALL_DIFFERENT_CLAUSE in text
    assert pair is not None or different, f"could not parse medium-tier sides: {text!r}"
    assert not (pair is not None and different), f"ambiguous medium-tier sides clause: {text!r}"
    am = _THREE_DEGREES_RE.search(text)
    assert am, f"could not parse medium-tier angles: {text!r}"
    angles = [int(am.group(i)) for i in (1, 2, 3)]
    has_equal_angles = len(set(angles)) < 3
    assert (pair is not None) == has_equal_angles, f"stated sides and angles disagree: {text!r}"
    return pair is not None


_ALL_SIDES_EQUAL_RE = re.compile(r"all four sides are equal, each (\d+) units")
_OPPOSITE_PAIRS_RE = re.compile(
    r"opposite sides are equal in pairs: (\d+) units and \d+ units, (\d+) units and \d+ units"
)
_TRAPEZIUM_SIDES_RE = re.compile(r"side lengths are (\d+), (\d+), (\d+) and (\d+) units, with no two pairs")
_ALL_ANGLES_RIGHT_CLAUSE = "all four interior angles are right angles"
_NOT_ALL_ANGLES_RIGHT_RE = re.compile(r"one interior angle measures (\d+) degrees, so not every angle is a right angle")
_BOTH_PAIRS_PARALLEL_CLAUSE = "both pairs of opposite sides are parallel"
_ONE_PAIR_PARALLEL_CLAUSE = "only one pair of opposite sides is parallel"


def _oracle_easy(text: str) -> str:
    m = _THREE_UNITS_RE.search(text)
    assert m, f"could not parse easy-tier sides: {text!r}"
    sides = [int(m.group(i)) for i in (1, 2, 3)]
    pairs_equal = (sides[0] == sides[1]) + (sides[1] == sides[2]) + (sides[0] == sides[2])
    if pairs_equal == 3:
        return "equilateral"
    if pairs_equal >= 1:
        return "isosceles"
    return "scalene"


def _oracle_medium(text: str) -> str:
    am = _THREE_DEGREES_RE.search(text)
    assert am, f"could not parse medium-tier angles: {text!r}"
    angles = [int(am.group(i)) for i in (1, 2, 3)]
    assert sum(angles) == 180, f"medium-tier angles do not sum to 180: {angles} in {text!r}"
    is_isosceles = _medium_is_isosceles(text)
    is_right = 90 in angles
    if is_isosceles and is_right:
        return "isosceles right-angled"
    if is_isosceles:
        return "isosceles, not right-angled"
    if is_right:
        return "scalene right-angled"
    return "scalene, not right-angled"


# Per-label truth predicate, independent of the answer -------
#
# Each offered medium-tier label names a full (side-equality, right-angle) cell, never a single fact
# alone -- this is the fix itself. Given a label string and the triangle's two
# independently-recomputed facts, decide whether the label is a TRUE description, entirely
# separately from which one construction/the oracle's `_oracle_medium` picked as "the" answer.


def _is_true_description_medium(label: str, is_isosceles: bool, is_right: bool) -> bool:
    if label == "isosceles right-angled":
        return is_isosceles and is_right
    if label == "scalene right-angled":
        return (not is_isosceles) and is_right
    if label == "isosceles, not right-angled":
        return is_isosceles and not is_right
    if label == "scalene, not right-angled":
        return (not is_isosceles) and not is_right
    raise AssertionError(f"unrecognised medium-tier label: {label!r}")


def _facts_from_medium_text(text: str):
    am = _THREE_DEGREES_RE.search(text)
    assert am, f"could not parse medium-tier angles: {text!r}"
    angles = [int(am.group(i)) for i in (1, 2, 3)]
    return _medium_is_isosceles(text), 90 in angles


def test_medium_exactly_one_offered_option_is_true_over_1000_seeds():
    """This is what the original defect violated: seed-1-shaped draws (a scalene triangle with a stated right
    angle) had both the key and a distractor true simultaneously."""
    challenged = []
    for seed in MEDIUM_SWEEP:
        q = make(TOPIC, seed=seed, difficulty="medium")
        is_isosceles, is_right = _facts_from_medium_text(q.text)
        offered = [q.answer_text] + list(q.distractors)
        true_count = sum(_is_true_description_medium(opt, is_isosceles, is_right) for opt in offered)
        if true_count != 1:
            challenged.append((seed, true_count, offered, is_isosceles, is_right, q.text))
    assert not challenged, (
        f"medium: {len(challenged)} draws over {len(MEDIUM_SWEEP)} did not have exactly one true "
        f"offered option: {challenged[:5]}"
    )


def test_medium_old_pool_would_have_failed_this_assertion():
    """Proves the new assertion is not vacuous: replayed against the OLD (pre-fix) one-fact label set on the
    exact defect scenario (sides 21, 9, 23; angles 67, 23, 90 -- a scalene triangle with a stated right
    angle), the old pool's naive one-fact labels give two true options, which the new assertion would have
    caught."""
    sides = [21, 9, 23]
    angles = [67, 23, 90]
    is_isosceles = sides[0] == sides[1] or sides[1] == sides[2] or sides[0] == sides[2]
    is_right = 90 in angles
    assert not is_isosceles and is_right

    def old_is_true(label):
        # The pre-fix pool's naive, one-fact-per-label semantics: "scalene"
        # true whenever the sides are scalene, full stop; "right-angled"
        # true whenever a right angle is stated, full stop.
        if label == "right-angled":
            return is_right
        if label == "scalene":
            return not is_isosceles
        if label == "isosceles":
            return is_isosceles
        if label == "isosceles right-angled":
            return is_isosceles and is_right
        raise AssertionError(label)

    old_offered = ["right-angled", "scalene", "isosceles", "isosceles right-angled"]
    old_true_count = sum(old_is_true(opt) for opt in old_offered)
    assert old_true_count == 2, (
        f"expected the old pool to be ambiguous (2 true options) on the defect scenario, got {old_true_count}"
    )


def _oracle_hard(text: str) -> str:
    both_pairs = _BOTH_PAIRS_PARALLEL_CLAUSE in text
    one_pair = _ONE_PAIR_PARALLEL_CLAUSE in text
    assert both_pairs != one_pair, f"ambiguous/missing parallel clause: {text!r}"
    if one_pair:
        m = _TRAPEZIUM_SIDES_RE.search(text)
        assert m, f"could not parse trapezium sides: {text!r}"
        return "trapezium"
    m_rhombus = _ALL_SIDES_EQUAL_RE.search(text)
    m_para = _OPPOSITE_PAIRS_RE.search(text)
    assert (m_rhombus is not None) != (m_para is not None), f"ambiguous sides clause: {text!r}"
    return "rhombus" if m_rhombus else "parallelogram"


def _oracle_exam(text: str) -> str:
    m_all_equal = _ALL_SIDES_EQUAL_RE.search(text)
    m_pairs = _OPPOSITE_PAIRS_RE.search(text)
    assert (m_all_equal is not None) != (m_pairs is not None), f"ambiguous sides clause: {text!r}"
    all_equal = m_all_equal is not None

    all_right = _ALL_ANGLES_RIGHT_CLAUSE in text
    not_all_right = _NOT_ALL_ANGLES_RIGHT_RE.search(text) is not None
    assert all_right != not_all_right, f"ambiguous angle clause: {text!r}"

    if all_equal and all_right:
        return "square"
    if all_equal:
        return "rhombus"
    if all_right:
        return "rectangle"
    return "parallelogram"


_ORACLES = {"easy": _oracle_easy, "medium": _oracle_medium, "hard": _oracle_hard, "exam": _oracle_exam}
_POOLS = {"easy": EASY_POOL, "medium": MEDIUM_POOL, "hard": HARD_POOL, "exam": EXAM_POOL}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_oracle_rederives_every_row_in_the_sweep(difficulty):
    """Every rendered row's class, re-derived independently from the
    rendered text, matches ``answer_text`` -- the artefact the learner
    receives. Also confirms every other pool member is offered as a
    distractor on every row (the enumeration-matrix check for a
    classification skill: no row ever offers a proper subset of the
    plausible wrong classes)."""
    oracle = _ORACLES[difficulty]
    pool = _POOLS[difficulty]
    seen_classes = set()
    challenged = []
    for seed in SWEEP:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        try:
            derived = oracle(q.text)
        except AssertionError as exc:
            challenged.append((seed, str(exc)))
            continue
        seen_classes.add(derived)
        if derived != q.answer_text:
            challenged.append((seed, f"oracle says {derived!r}, answer_text is {q.answer_text!r}: {q.text!r}"))
            continue
        expected_distractors = pool - {derived}
        if set(q.distractors) != expected_distractors:
            challenged.append(
                (seed, f"distractors {set(q.distractors)!r} != expected {expected_distractors!r}: {q.text!r}")
            )
    assert not challenged, f"{difficulty}: {len(challenged)} challenged rows over {len(SWEEP)}: {challenged[:5]}"
    assert seen_classes == pool, f"{difficulty}: sweep only reached {seen_classes}, expected the full pool {pool}"
