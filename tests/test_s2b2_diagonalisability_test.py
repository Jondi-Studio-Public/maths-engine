"""Tests `uc2-diagonalisability-test` (topic `s2b2_diagonalisability_test`) and its exam chain.

The oracle reads the matrix (and the eigenvalue, or the parameter k) back out of the question TEXT and re-derives each
key with SymPy: eigenvalue multiplicities, null spaces and `is_diagonalizable`.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.s2b2_diagonalisability_build  # noqa: F401  (registers the topics)
from mathgen import make

from tests.s2b2_support import (
    TIERS,
    check_exam_shape,
    check_follow_through_marks,
    check_follow_through_strings,
    check_single_tier,
    check_wiring,
    distinct_questions,
    load_skill,
    matrices,
)

TOPIC = "s2b2_diagonalisability_test"
EXAM = "exd_x_s2b2_diagonalisability_test"
SKILL = load_skill("uc2-diagonalisability-test")
K = sympy.Symbol("k")


def test_wiring():
    check_wiring(SKILL, TOPIC, EXAM, "number", ["ucv1-eigenvectors", "ucv1-null-space"])


def test_easy_key_is_the_geometric_multiplicity():
    seen = set()
    for seed in range(80):
        q = make(TOPIC, seed=seed, difficulty="easy")
        a = matrices(q.text)[0]
        lam = int(re.search(r"λ = (-?\d+)", q.text).group(1))
        geometric = len((a - lam * sympy.eye(a.rows)).nullspace())
        assert int(q.answer) == geometric, (seed, q.text)
        # The root is repeated (otherwise the question would be trivial), and the eigenspace is smaller or equal.
        algebraic = a.eigenvals()[lam]
        assert 2 <= algebraic and 1 <= geometric <= algebraic
        seen.add((algebraic, geometric))
    assert len(seen) >= 3


def _diagonalisable_ks(a):
    return [k for k in range(-15, 16) if a.subs(K, k).is_diagonalizable()]


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_parameter_key_is_the_unique_value(tier):
    negated = 0
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=tier)
        a = matrices(q.text, symbols="k")[0]
        good = _diagonalisable_ks(a)
        if re.search(r"not diagonalisable|cannot be diagonalised", q.text):
            bad = [k for k in range(-15, 16) if k not in good]
            assert bad == [int(q.answer)], (seed, q.text, bad)
            negated += 1
        else:
            assert good == [int(q.answer)], (seed, q.text, good)
    assert (negated > 0) == (tier == "hard")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SKILL, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    assert distinct_questions(TOPIC, tier) >= 50


def test_exam_shape_marks_and_follow_through():
    check_exam_shape(SKILL, EXAM, ("number",) * 4, (3, 1, 2, 1), 2)
    check_follow_through_strings(EXAM)
    check_follow_through_marks(SKILL, EXAM)


def test_exam_keys_match_independent_derivation():
    for seed in range(60):
        q = make(EXAM, seed=seed, difficulty="exam")
        a = matrices(q.stem)[0]
        repeated = [lam for lam, m in a.eigenvals().items() if m == 2]
        assert len(repeated) == 1
        lam = repeated[0]
        assert int(q.parts[0].answer) == lam
        assert int(q.parts[1].answer) == a.trace() - 2 * lam
        rank = (a - lam * sympy.eye(3)).rank()
        assert int(q.parts[2].answer) == rank
        assert int(q.parts[3].answer) == 3 - rank
