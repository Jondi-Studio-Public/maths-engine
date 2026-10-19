"""Tests the pigeonhole-and-binomial-identities skill: keys re-derived by exhaustive search or from the stem, the exam chain."""

from __future__ import annotations

import itertools
import math
import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3i_support as sup

SKILL = "pigeonhole-and-binomial-identities"
TOPIC = "r3i_pigeonhole"
EXAM = "r3i_exam_pigeonhole"
N = sympy.Symbol("n")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_CATEGORY_SIZES = {
    "birth month": 12,
    "day of the week of their birth": 7,
    "suit": 4,
    "score": 6,
    "last digit": 10,
    "remainder when divided by 5": 5,
    "remainder when divided by 8": 8,
    "remainder when divided by 3": 3,
}


def _read(text):
    return parse_expr(text.replace("^", "**"), local_dict={"n": N}, transformations=_TRANSFORMS)


def _stem_expression(text):
    """'C(n + 1, 2) - C(n, 2)' or '(n + 2)! ÷ (n - 1)!' as a SymPy expression in n."""
    text = text.replace("÷", "/")
    text = re.sub(r"\(([^()]+)\)!", r"factorial(\1)", text)
    text = text.replace("C(", "binomial(")
    return parse_expr(text, local_dict={"n": N}, transformations=_TRANSFORMS)


def _agree(stem_expr, key):
    for n in (5, 6, 8, 11, 14):
        assert sympy.simplify(sympy.expand_func(stem_expr.subs(N, n)) - key.subs(N, n)) == 0, (n, stem_expr, key)


def _least_forcing(size, bad):
    """One more than the largest subset of {1..size} with no 'bad' pair, by exhaustive search."""
    best = 0
    for r in range(size + 1):
        for subset in itertools.combinations(range(1, size + 1), r):
            if not any(bad(a, b) for a, b in itertools.combinations(subset, 2)):
                best = max(best, r)
                break
    return best + 1


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "expression", ["combinations", "senior-use-factorial-notation"])


def test_easy_keys():
    seen = set()
    for seed in range(120):
        q = sup.tier_question(TOPIC, "easy", seed)
        boxes = re.match(r"(\d+) \w+ are placed in (\d+)", q.text)
        if boxes:
            seen.add("boxes")
            n, k = int(boxes.group(1)), int(boxes.group(2))
            # the fullest box has at least the average, so m is the least whole number with k*m >= n
            assert int(q.answer_text) == math.ceil(n / k), (seed, q.text, q.answer_text)
        else:
            seen.add("repeat")
            category = re.search(r"have the same (.+)\?", q.text).group(1)
            assert int(q.answer_text) == _CATEGORY_SIZES[category] + 1
    assert seen == {"boxes", "repeat"}


def test_medium_keys():
    seen = set()
    for seed in range(200):
        q = sup.tier_question(TOPIC, "medium", seed)
        text = q.text
        if text.startswith("A bag holds"):
            seen.add("colours")
            counts = [int(c) for c in re.findall(r"(\d+) (?:red|blue|green|black)", text)]
            m = int(re.search(r"certain of (\d+)", text).group(1))
            safe = max(sum(c) for c in itertools.product(*[range(min(cap, m - 1) + 1) for cap in counts]))
            assert int(q.answer_text) == safe + 1, (seed, text)
        elif text.startswith("How many"):
            seen.add("same kind")
            m = int(re.search(r"certain that (\d+) of them", text).group(1))
            category = re.search(r"have the same (.+)\?", text).group(1)
            assert int(q.answer_text) == _CATEGORY_SIZES[category] * (m - 1) + 1, (seed, text)
        else:
            seen.add("expression")
            stem = re.match(r"Simplify (.+) as an expression in n\.", text).group(1)
            _agree(_stem_expression(stem), _read(q.answer_text))
    assert seen == {"colours", "same kind", "expression"}


def test_hard_keys():
    seen = set()
    for seed in range(200):
        q = sup.tier_question(TOPIC, "hard", seed)
        text = q.text
        if text.startswith("Numbers are chosen"):
            size = int(re.search(r"whole numbers 1 to (\d+)\.", text).group(1))
            if "consecutive" in text:
                seen.add("consecutive")
                expected = _least_forcing(size, lambda a, b: abs(a - b) == 1)
            else:
                seen.add("sum")
                expected = _least_forcing(size, lambda a, b: a + b == size + 1)
            assert int(q.answer_text) == expected, (seed, text)
        elif text.startswith("Use Pascal"):
            seen.add("numeric")
            a, b = (int(v) for v in re.search(r"C\((\d+), (\d+)\)", text).groups())
            assert int(q.answer_text) == math.comb(a, b) + math.comb(a, b + 1)
        else:
            seen.add("identity")
            stem = re.match(r"Simplify (.+) as an expression in n\.", text).group(1)
            _agree(_stem_expression(stem), _read(q.answer_text))
    assert seen == {"consecutive", "sum", "numeric", "identity"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8 if tier == "easy" else 12, top=0.3 if tier == "easy" else 0.2)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(40)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(80)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(80)]
    assert not any("Simplify" in t for t in easy)
    assert any("Simplify" in t and "!" in t for t in medium) and any("bag" in t for t in medium)
    assert (
        any("consecutive" in t for t in hard)
        and any("Pascal" in t for t in hard)
        and any("C(n, 2) + C(n, 3)" in t for t in hard)
    )
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number", "number", "expression", "number"], [1, 2, 2, 1], [False, True, False, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.1)


def test_exam_keys():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        total, k = (int(v) for v in re.search(r"There are (\d+) students.*one of (\d+) houses", q.text).groups())
        a = int(q.parts[0].answer_text)
        assert a == math.ceil(total / k) and k * (a - 1) < total
        b = int(q.parts[1].answer_text)
        assert b == k * a + 1
        assert _read(q.parts[2].answer_text) == N
        assert int(q.parts[3].answer_text) == math.comb(b + 1, 2) - math.comb(b, 2)


def test_exam_follow_through_from_wrong_early_parts():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        k = int(re.search(r"one of (\d+) houses", q.text).group(1))
        for wrong_a in (1, 2, 3, 4, 5, 9, 10, 11):
            if wrong_a == int(canon["a"]):
                continue
            wrong_b = k * wrong_a + 1
            replies = {**canon, "a": str(wrong_a), "b": str(wrong_b), "d": str(wrong_b)}
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            if result.parts["b"].marks == q.parts[1].marks and result.parts["d"].marks == q.parts[3].marks:
                break
        else:
            raise AssertionError(f"no usable wrong reply for seed {seed}")
