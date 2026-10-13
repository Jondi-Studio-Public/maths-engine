"""Tests the skill `am-gm-inequality`.

The oracle never uses AM-GM: it reads the function (or the constraint) back out of the rendered question and finds the
least or greatest value, and the x where it occurs, by calculus with SymPy (set the derivative to zero and take the
positive root). The generator builds its answers from the equality case, so the two routes are independent.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from tests.r4f_support import (
    canonical,
    check_answer_space_single,
    check_exam_clean,
    check_exam_oracle,
    check_exam_shape,
    check_single_clean,
    check_tiers_differ,
    check_wired,
    exam_questions,
    parse,
)

SKILL_ID = "am-gm-inequality"
TOPIC = "r4f_am_gm"
EXAM = "exd_x_r4f_am_gm"
x = sympy.Symbol("x", positive=True)


def frac(value) -> Fraction:
    value = sympy.nsimplify(value)
    return Fraction(int(value.p), int(value.q))


def optimum(expr):
    """(x*, f(x*)) of a function of x > 0 with a single stationary point, by calculus; checks it is the extreme."""
    roots = [r for r in sympy.solve(sympy.diff(expr, x), x) if r.is_real and r > 0]
    assert len(roots) == 1, (expr, roots)
    root = roots[0]
    second = sympy.diff(expr, x, 2).subs(x, root)
    assert second != 0
    return frac(root), frac(expr.subs(x, root)), second > 0


def sym(text: str):
    return parse(text.replace("²", "**2"), x=x)


def product_optimum(p, q, s):
    """Greatest xy with p x + q y = s, x, y > 0: (x*, xy at x*)."""
    expr = x * (s - p * x) / q
    xs, best, is_min = optimum(expr)
    assert not is_min
    return xs, best


def single_oracle(q):
    """(value, x*) for a single-answer question from its text alone."""
    text = q.text
    if m := re.search(r"Find the minimum value of f\(x\) = (.+?) for x > 0", text):
        xs, best, is_min = optimum(sym(m.group(1)))
        assert is_min
        return best
    if m := re.search(r"satisfy (\d*)x \+ (\d*)y = (\d+)\. Find the greatest possible value of the product", text):
        return product_optimum(int(m.group(1) or 1), int(m.group(2) or 1), int(m.group(3)))[1]
    if m := re.search(r"f\(x\) = (.+?) has a least value\. Find the value of x", text):
        return optimum(sym(m.group(1)))[0]
    if m := re.search(r"satisfy (\d*)x \+ (\d*)y = (\d+), and the product xy is greatest", text):
        return product_optimum(int(m.group(1) or 1), int(m.group(2) or 1), int(m.group(3)))[0]
    if m := re.search(r"f\(x\) = \(x \+ (\d+)\)\(x \+ (\d+)\)/x has a least value", text):
        return optimum(sym(f"(x + {m.group(1)})*(x + {m.group(2)})/x"))[0]
    if m := re.search(r"least value of f\(x\) = (\(x \+ \d+\)\(x \+ \d+\)/x) for x > 0", text):
        return optimum(sym(m.group(1)))[1]
    if m := re.search(r"area (\d+) m²", text):
        a = int(m.group(1))
        return optimum(2 * x + a / x)[1]
    if m := re.search(r"C = (.+?) \+ (\d+) dollars per hour", text):
        return optimum(sym(m.group(1)) + int(m.group(2)))[1]
    if m := re.search(r"S = x² \+ (\d+)/x square centimetres", text):
        return optimum(x**2 + int(m.group(1)) / x)[1]
    if m := re.search(r"2x \+ 3y = (\d+) metres", text):
        return product_optimum(2, 3, int(m.group(1)))[1]
    raise AssertionError(text)


def test_wired():
    check_wired(SKILL_ID, TOPIC, EXAM)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_single_tiers_grade_cleanly(tier):
    check_single_clean(SKILL_ID, TOPIC, tier, n=60)


def test_tiers_differ_in_shape():
    check_tiers_differ(TOPIC)


@pytest.mark.parametrize(("tier", "floor"), [("easy", 25), ("medium", 12), ("hard", 25)])
def test_answer_space(tier, floor):
    check_answer_space_single(TOPIC, tier, n=300, floor=floor)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_key_rederived_by_calculus(tier):
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert Fraction(q.answer_text) == single_oracle(q), (tier, seed, q.text)


def test_every_hard_word_problem_appears():
    seen = set()
    for seed in range(120):
        text = make(TOPIC, seed=seed, difficulty="hard").text
        seen.add(
            "paddock"
            if "paddock" in text
            else "box"
            if "open-top" in text
            else "pen"
            if "divided" in text
            else "expand"
            if "(x +" in text
            else "cost"
        )
    assert seen == {"paddock", "box", "pen", "expand", "cost"}


def test_medium_x_may_be_a_half():
    answers = {make(TOPIC, seed=s, difficulty="medium").answer_text for s in range(300)}
    assert any("." in a for a in answers) and any("." not in a for a in answers)


def exam_oracle(q):
    text = q.stem
    prompts = {p.label: p.prompt for p in q.parts}
    if "paddock" in text:
        area = int(re.search(r"area (\d+) m²", text).group(1))
        w1 = int(re.search(r"when w = (\d+)", prompts["a"]).group(1))
        price = int(re.search(r"\$(\d+) per metre", prompts["d"]).group(1))
        ws, low, _ = optimum(2 * x + area / x)
        return {"a": 2 * w1 + Fraction(area, w1), "b": low, "c": ws, "d": price * low}
    if "machine" in text:
        expr_text, fixed = re.search(r"C = (.+?) \+ (\d+) dollars per hour", text).groups()
        expr = sym(expr_text)
        x1 = int(re.search(r"when x = (\d+)", prompts["a"]).group(1))
        xs, low, _ = optimum(expr)
        return {"a": frac(expr.subs(x, x1)) + int(fixed), "b": low, "c": low + int(fixed), "d": xs}
    if "open-top" in text:
        s_coef = int(re.search(r"S = x² \+ (\d+)/x", text).group(1))
        volume = int(re.search(r"volume of (\d+) cm³", text).group(1))
        x1 = int(re.search(r"when x = (\d+)", prompts["a"]).group(1))
        expr = x**2 + s_coef / x
        xs, low, _ = optimum(expr)
        assert s_coef == 4 * volume
        return {"a": frac(expr.subs(x, x1)), "b": xs, "c": low, "d": Fraction(volume) / xs**2}
    fence = int(re.search(r"2x \+ 3y = (\d+) metres", text).group(1))
    x1 = int(re.search(r"when x = (\d+)", prompts["a"]).group(1))
    xs, best = product_optimum(2, 3, fence)
    return {"a": Fraction(fence - 2 * x1, 3), "b": best, "c": xs, "d": (fence - 2 * xs) / 3}


def test_exam_clean_and_graded():
    check_exam_clean(SKILL_ID, EXAM, n=60)


def test_exam_key_rederived_by_calculus():
    check_exam_oracle(EXAM, exam_oracle, n=240)


def test_exam_has_all_four_problems():
    kinds = set()
    for q in exam_questions(EXAM, 60):
        t = q.stem
        kinds.add("paddock" if "paddock" in t else "machine" if "machine" in t else "box" if "open-top" in t else "pen")
    assert kinds == {"paddock", "machine", "box", "pen"}


def test_exam_answer_space():
    check_exam_shape(EXAM, floor_distinct=120, n=300)


def test_exam_parts_shape():
    from mathgen.engine import grade
    from tests.r4f_support import skill

    sk = skill(SKILL_ID)
    done = 0
    for q in exam_questions(EXAM, 80):
        assert all(p.answer_kind == "number" for p in q.parts)
        assert sum(p.follow_through is not None for p in q.parts) >= 1
        # a wrong bound carried correctly into the dependent parts earns them as follow-through
        b_wrong = int(q.parts[1].distractors[0]) if "paddock" in q.stem or "machine" in q.stem else None
        if b_wrong is None or b_wrong <= 0:
            continue
        if "paddock" in q.stem:
            price = int(re.search(r"\$(\d+) per metre", q.parts[3].prompt).group(1))
            reply = canonical(q) | {"b": str(b_wrong), "c": str(Fraction(b_wrong, 4)), "d": str(price * b_wrong)}
            if Fraction(b_wrong, 4).denominator != 1:
                continue
            got = grade(sk, q, reply)
            assert got.parts["b"].marks == 0
            for label in "cd":
                assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, q.text
            done += 1
    assert done >= 3
