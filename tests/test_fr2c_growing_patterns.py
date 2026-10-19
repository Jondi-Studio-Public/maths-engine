"""`f10-growing-visual-patterns`: topic `fr2c_growing_patterns` (easy, medium, hard) and the exam chain.

The oracle counts the matchsticks of Figures 1 to 3 from the drawn segments, finds the constant difference, and works every
answer from that.
"""

from __future__ import annotations

import re

import pytest

from mathgen import figures, make
from mathgen.generators.fr2c_growing_patterns_build import _drawn_counts
from tests.fr2c_checks import check_follow_through, check_parts, check_single, skill

SKILL = "f10-growing-visual-patterns"
TOPIC = "fr2c_growing_patterns"
EXAM = "fr2c_growing_patterns_exam"
TIERS = ("easy", "medium", "hard")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _rule(q):
    c1, c2, c3 = _drawn_counts(q.figure)
    d = c2 - c1
    assert c3 - c2 == d
    return d, c1 - d


def test_skill_wiring():
    sk = skill(SKILL)
    assert sk.answer == "number" and sk.needs_figure is True and sk.capability_gates == ["figure_stimulus"]
    assert [(s.topic, s.difficulty) for s in sk.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_valid_and_the_reader_agrees(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert q.figure["kind"] == "cartesian" and not figures.validate(q.figure)
        assert figures.get_reader(TOPIC)(q) == q.answer_text, (tier, seed)


def test_all_four_patterns_appear_with_their_rules():
    rules = set()
    for seed in range(80):
        rules.add(_rule(_q("easy", seed)))
    assert rules == {(3, 1), (2, 1), (5, 2), (4, 1)}


def test_easy_is_the_next_figure():
    for seed in range(80):
        q = _q("easy", seed)
        d, k = _rule(q)
        n = int(re.search(r"in Figure (\d+)\?", q.text).group(1))
        assert n in (4, 5, 6) and int(q.answer_text) == d * n + k


def test_medium_is_a_far_figure_and_hard_is_the_inverse():
    for seed in range(80):
        q = _q("medium", seed)
        d, k = _rule(q)
        n = int(re.search(r"Figure (\d+)\?", q.text).group(1))
        assert n >= 8 and int(q.answer_text) == d * n + k
        h = _q("hard", seed)
        d, k = _rule(h)
        stock = int(re.search(r"has (\d+) ", h.text).group(1))
        n = int(h.answer_text)
        assert d * n + k <= stock < d * (n + 1) + k, h.text  # the largest figure that can be built
        assert d * n + k != stock  # sticks are left over


def test_exam_chain_shape():
    check_parts(
        SKILL,
        EXAM,
        marks=[1, 2, 1, 2],
        kinds=["number", "expression", "number", "number"],
        follow=[False, False, True, True],
    )


def test_exam_answers_recomputed_from_the_figure():
    for seed in range(60):
        q = make(EXAM, seed=seed, difficulty="exam")
        d, k = _rule(q)
        p = {x.label: x for x in q.parts}
        assert int(p["a"].answer_text) == 5 * d + k
        far = int(re.search(r"Figure (\d+)\.", p["c"].prompt).group(1))
        assert int(p["c"].answer_text) == d * far + k
        exact = int(re.search(r"exactly (\d+) ", p["d"].prompt).group(1))
        assert int(p["d"].answer_text) * d + k == exact


def test_exam_follow_through_from_a_wrong_rule():
    def replace(q, parts):
        far = int(re.search(r"Figure (\d+)\.", parts["c"].prompt).group(1))
        wrong = parts["b"].distractors[0]
        from sympy import Symbol
        from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

        n = Symbol("n")
        expr = parse_expr(wrong, transformations=standard_transformations + (implicit_multiplication_application,))
        return wrong, str(expr.subs(n, far))

    check_follow_through(SKILL, EXAM, "b", replace, "c", seeds=range(40))
