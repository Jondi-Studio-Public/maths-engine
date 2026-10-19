"""Tests the five equation-foundation skills (`w5_meaning_of_equality`, `w5_inverse_operations`, `w5_solution_set`,
`w5_identity_no_solution`, `w5_check_substitution`) and their exam chains.

The oracles read the sentence, equation or machine back out of the question TEXT and judge every option with SymPy or
plain integer arithmetic (never with the generators' helpers): a key must be true, correct or equivalent, and every
distractor must be false, wrong or inequivalent.
"""

from __future__ import annotations

import re
from fractions import Fraction as Fr

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen.generators.w5_equations_foundations_build  # noqa: F401  (registers the topics)
from tests import w5_support as sup

X, N, K = sympy.symbols("x n k", real=True)
_TR = standard_transformations + (implicit_multiplication_application, convert_xor)

TOPICS = {
    "w5_meaning_of_equality": ("meaning-of-equality", "exd_x_w5_meaning_of_equality"),
    "w5_inverse_operations": ("inverse-operations", "exd_x_w5_inverse_operations"),
    "w5_solution_set": ("solution-and-solution-set", "exd_x_w5_solution_set"),
    "w5_identity_no_solution": ("identity-and-no-solution", "exd_x_w5_identity_no_solution"),
    "w5_check_substitution": ("check-solution-by-substitution", "exd_x_w5_check_substitution"),
}
SKILLS = {t: sup.skill_of(sid) for t, (sid, _) in TOPICS.items()}
EXAMS = {t: ex for t, (_, ex) in TOPICS.items()}


def _e(text: str):
    """A SymPy expression from learner text (x for times, a unicode minus or division sign accepted)."""
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
    return parse_expr(text, local_dict={"x": X, "n": N, "k": K}, transformations=_TR)


def _sides(sentence: str):
    left, right = sentence.split(" = ")
    return _e(left), _e(right)


_TOKEN = re.compile(r"[-+\d()xnk^×÷]+")


def _eq_in(text: str) -> str:
    """The longest run of equation tokens in `text` that contains an equals sign, e.g. '4x + 6 = 34'."""
    runs, run = [], []
    for tok in text.split(" "):
        bare = tok.rstrip(".,?:")
        if bare == "=" or _TOKEN.fullmatch(bare):
            run.append(bare)
            if bare != tok:
                runs.append(run)
                run = []
        else:
            runs.append(run)
            run = []
    runs.append(run)
    runs = [r for r in runs if "=" in r]
    return " ".join(max(runs, key=len))


def _stem(q) -> str:
    """The shared stem of an exam chain (the text before part (a))."""
    return q.text.split(" (a)")[0]


def _true(sentence: str) -> bool:
    left, right = _sides(sentence)
    return sympy.simplify(left - right) == 0


# -- wiring, keys and distractors, pools ------------------------------------------------------------------------------


@pytest.mark.parametrize("topic", list(TOPICS))
def test_wiring(topic):
    skill = SKILLS[topic]
    sup.check_wiring(skill, topic, EXAMS[topic], "mcq")
    steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
@pytest.mark.parametrize("topic", list(TOPICS))
def test_key_accepted_and_distractors_rejected(topic, tier):
    sup.check_single_tier(SKILLS[topic], topic, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
@pytest.mark.parametrize("topic", list(TOPICS))
def test_four_plus_options_and_varied_keys(topic, tier):
    keys = set()
    for seed in range(60):
        q = sup.single(topic, tier, seed)
        opts = [q.answer_text, *q.distractors]
        assert len(set(opts)) == len(opts) >= 5, (seed, opts)
        assert not any(o.strip().lstrip("-").isdigit() for o in opts), (seed, opts)
        keys.add(q.answer_text)
    assert len(keys) >= 12, (topic, tier, len(keys))


@pytest.mark.parametrize("topic", list(TOPICS))
def test_pools_and_shapes(topic):
    sup.check_pools(topic, min_prompts=20)


@pytest.mark.parametrize("topic", list(TOPICS))
def test_key_position_is_not_a_tell(topic):
    """The key is not systematically the longest or the shortest option."""
    longest = shortest = 0
    for tier in sup.TIERS:
        for seed in range(60):
            q = sup.single(topic, tier, seed)
            lens = [len(q.answer_text)] + [len(d) for d in q.distractors]
            longest += lens[0] == max(lens) and lens.count(lens[0]) == 1
            shortest += lens[0] == min(lens) and lens.count(lens[0]) == 1
    assert longest < 0.45 * 180 and shortest < 0.45 * 180, (topic, longest, shortest)


@pytest.mark.parametrize("topic", list(TOPICS))
def test_exam_chain_grades_and_distractors(topic):
    q = sup.exam(EXAMS[topic], 0)
    marks = [p.marks for p in q.parts]
    kinds = [{p.answer_kind} for p in q.parts]
    kinds = [{"number", "fraction"} if k <= {"number", "fraction"} else k for k in kinds]
    sup.check_exam_chain_kinds(SKILLS[topic], EXAMS[topic], marks, kinds, seeds=40)


# -- meaning of equality ------------------------------------------------------------------------------------------------

_SENT = re.compile(r"\d+ [+×] \d+ = \d+ [+×] \d+")
_EQN = re.compile(r"((?:\d+|n)(?: [+×] (?:\d+|n))* = (?:\d+|n)(?: [+×] (?:\d+|n))*)")


def test_equality_easy_and_story():
    for seed in range(80):
        q = sup.single("w5_meaning_of_equality", "easy", seed)
        if q.answer_text.startswith(("Yes", "No:")):
            sentence = _SENT.search(q.text).group(0)
            left, right = (_e(s) for s in sentence.split(" = "))
            if left == right:
                assert q.answer_text == f"Yes: both sides are equal to {left}"
                assert all(d.startswith("No") for d in q.distractors)
            else:
                assert q.answer_text == f"No: the left side is {left} and the right side is {right}"
                for d in q.distractors:
                    assert d.startswith("Yes") or d != q.answer_text
                    m = re.fullmatch(r"No: the left side is (-?\d+) and the right side is (-?\d+)", d)
                    if m:
                        assert (int(m.group(1)), int(m.group(2))) != (left, right)
            continue
        asks_false = "false" in q.text or "NOT correct" in q.text
        assert _true(q.answer_text) is (not asks_false), (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert _true(d) is asks_false, (seed, q.text, d)


def test_equality_medium_missing_number():
    for seed in range(80):
        q = sup.single("w5_meaning_of_equality", "medium", seed)
        sentence = _EQN.search(q.text).group(1)
        left, right = _sides(sentence)
        want = sympy.solve(left - right, N)
        assert len(want) == 1
        assert q.answer_text == f"n = {want[0]}", (seed, q.text, q.answer_text)
        for d in q.distractors:
            assert int(d.removeprefix("n = ")) != want[0]


def test_equality_hard():
    for seed in range(80):
        q = sup.single("w5_meaning_of_equality", "hard", seed)
        if "expression" in q.text or "Which one does not" in q.text:
            a, b = re.search(r"(\d+) × (\d+)", q.text).groups()
            target = int(a) * int(b)
            assert _e(q.answer_text) != target, (seed, q.text, q.answer_text)
            for d in q.distractors:
                assert _e(d) == target, (seed, q.text, d)
        else:
            base = _SENT.search(q.text).group(0)
            assert _true(base)
            assert _true(q.answer_text), (seed, q.answer_text)
            for d in q.distractors:
                assert not _true(d), (seed, d)


def test_equality_exam():
    for seed in range(60):
        q = sup.exam("exd_x_w5_meaning_of_equality", seed)
        a, b, c = (int(v) for v in re.search(r"has (\d+) \+ (\d+) on its left side and n \+ (\d+)", q.text).groups())
        pa, pb, pc = q.parts
        assert int(pa.answer_text) == a + b - c
        assert int(pb.answer_text) == a + b
        k = int(re.search(r"adds (\d+) to the left side only", pc.prompt).group(1))
        assert pc.answer_text == f"Add {k} to the right side"


# -- inverse operations -------------------------------------------------------------------------------------------------


def _step(word: str):
    """A function of one number from 'multiply by 3', 'add 5', 'subtract the number from 15'..."""
    word = word.lower()
    if m := re.fullmatch(r"add (\d+)", word):
        return lambda v, k=int(m.group(1)): v + k
    if m := re.fullmatch(r"subtract (\d+)", word):
        return lambda v, k=int(m.group(1)): v - k
    if m := re.fullmatch(r"multiply by (\d+)", word):
        return lambda v, k=int(m.group(1)): v * k
    if m := re.fullmatch(r"divide by (\d+)", word):
        return lambda v, k=int(m.group(1)): v / k
    if m := re.fullmatch(r"subtract the number from (\d+)", word):
        return lambda v, k=int(m.group(1)): k - v
    raise AssertionError(word)


def _chain(text: str):
    fs = [_step(w.strip().rstrip(".")) for w in text.split(", then ")]

    def run(v):
        for f in fs:
            v = f(v)
        return v

    return run


def _returns_input(forward, back) -> bool:
    return all(back(forward(Fr(v))) == v for v in (3, 7, 12, 25))


def test_inverse_easy():
    for seed in range(80):
        q = sup.single("w5_inverse_operations", "easy", seed)
        if m := re.search(
            r"(?:undoes|reverse) (adding|subtracting|multiplying by|dividing by) (\d+)|(Adding|Subtracting|Multiplying by|Dividing by) (\d+) is undone",
            q.text,
        ):
            word = (m.group(1) or m.group(3)).lower()
            k = m.group(2) or m.group(4)
            fwd = {
                "adding": f"add {k}",
                "subtracting": f"subtract {k}",
                "multiplying by": f"multiply by {k}",
                "dividing by": f"divide by {k}",
            }[word]
            f = _step(fwd)
            assert _returns_input(f, _step(q.answer_text.lower())), (seed, q.text, q.answer_text)
            for d in q.distractors:
                assert not _returns_input(f, _step(d.lower())), (seed, q.text, d)
        else:
            m = re.search(
                r"(\d+) is added to a number n and the result is (\d+)|(\d+) is subtracted from a number n|has (\d+) subtracted|multiplied by (\d+) and|divided by (\d+) and",
                q.text,
            )
            out = int(re.search(r"result is (\d+)", q.text).group(1))
            k = int(next(g for g in m.groups() if g))
            t = q.text
            n = (
                Fr(out - k if "added" in t else out + k if "subtracted" in t else out)
                if ("added" in t or "subtracted" in t)
                else (Fr(out, k) if "multiplied" in t else Fr(out * k))
            )
            for text, good in [(q.answer_text, True), *[(d, False) for d in q.distractors]]:
                val = _eval_calc(text)
                assert (val == n) is good, (seed, t, text, val, n)


def _eval_calc(text: str):
    if m := re.fullmatch(r"Add (\d+) to (\d+)", text):
        return int(m.group(1)) + int(m.group(2))
    if m := re.fullmatch(r"Subtract (\d+) from (\d+)", text):
        return int(m.group(2)) - int(m.group(1))
    if m := re.fullmatch(r"Multiply (\d+) by (\d+)", text):
        return int(m.group(1)) * int(m.group(2))
    if m := re.fullmatch(r"Divide (\d+) by (\d+)", text):
        return Fr(int(m.group(1)), int(m.group(2)))
    raise AssertionError(text)


def test_inverse_medium_and_hard():
    for tier in ("medium", "hard"):
        for seed in range(80):
            q = sup.single("w5_inverse_operations", tier, seed)
            if "own inverse" in q.text or "itself" in q.text or "twice" in q.text:

                def apply(rule: str, v):
                    return _e(rule.split("→")[1].replace("n", f"({v})").replace("÷", "/")).subs({})

                for text, good in [(q.answer_text, True), *[(d, False) for d in q.distractors]]:
                    twice = all(apply(text, apply(text, v)) == v for v in (3, 7, 11))
                    assert twice is good, (seed, text)
                continue
            if m := re.search(r"(\d+)x ([+-]) (\d+) = (\d+)", q.text):
                a_, sign, b_, c_ = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
                x_true = Fr(c_ - b_ if sign == "+" else c_ + b_, a_)
                assert _chain(q.answer_text)(Fr(c_)) == x_true, (seed, q.text, q.answer_text)
                for d in q.distractors:
                    assert _chain(d)(Fr(c_)) != x_true, (seed, q.text, d)
                continue
            steps_text = re.search(
                r"(?:in order(?: to x)?|machine that does|through these steps in order|steps to x in order): (.+?)\.(?: |$)",
                q.text,
            ).group(1)
            fwd = _chain(steps_text)
            if q.answer_text.startswith("x = "):
                out = int(re.search(r"output(?: of a machine)?(?: is)? (\d+)", q.text).group(1))
                val = Fr(sympy.Rational(q.answer_text.removeprefix("x = ")))
                assert fwd(val) == out, (seed, q.text, q.answer_text)
                for d in q.distractors:
                    assert fwd(Fr(sympy.Rational(d.removeprefix("x = ")))) != out
            else:
                assert _returns_input(fwd, _chain(q.answer_text)), (seed, q.text, q.answer_text)
                for d in q.distractors:
                    assert not _returns_input(fwd, _chain(d)), (seed, q.text, d)


def test_inverse_exam():
    for seed in range(60):
        q = sup.exam("exd_x_w5_inverse_operations", seed)
        steps_text = re.search(r"in order to an input x: (.+)\.", _stem(q) + " ").group(1)
        fwd = _chain(steps_text)
        pa, pb, pc = q.parts
        assert _returns_input(fwd, _chain(pa.answer_text))
        out_b = int(re.search(r"output is (\d+)", pb.prompt).group(1))
        assert fwd(Fr(int(pb.answer_text))) == out_b
        out_c = int(re.search(r"output is (\d+)", pc.prompt).group(1))
        assert fwd(Fr(int(pc.answer_text))) == out_c
        for p, out in ((pb, out_b), (pc, out_c)):
            for d in p.distractors:
                assert fwd(Fr(sympy.Rational(d))) != out


# -- solutions and solution sets --------------------------------------------------------------------------------------


def _solves(equation: str, name: str, value) -> bool:
    left, right = _sides(equation)
    sym = {"n": N, "x": X}[name]
    return sympy.simplify((left - right).subs(sym, value)) == 0


def test_solution_set_easy_and_medium():
    for tier in ("easy", "medium"):
        for seed in range(80):
            q = sup.single("w5_solution_set", tier, seed)
            if q.answer_text.startswith("n = "):
                eq = _eq_in(q.text)
                v = sympy.Rational(q.answer_text.removeprefix("n = "))
                assert _solves(eq, "n", v), (seed, q.text, q.answer_text)
                for d in q.distractors:
                    assert not _solves(eq, "n", sympy.Rational(d.removeprefix("n = "))), (seed, q.text, d)
            elif q.answer_text.startswith("{"):
                eq = _eq_in(q.text)
                found = {v for v in range(-90, 91) if _solves(eq, "x", v)}
                got = {int(t) for t in q.answer_text.strip("{}").split(", ")}
                assert got == found, (seed, q.text, q.answer_text, found)
                for d in q.distractors:
                    assert {int(t) for t in d.strip("{}").split(", ")} != found
            else:
                nval = int(re.search(r"n = (-?\d+)", q.text).group(1))
                assert _solves(q.answer_text, "n", nval), (seed, q.text, q.answer_text)
                for d in q.distractors:
                    assert not _solves(d, "n", nval), (seed, q.text, d)


def test_solution_set_hard():
    for seed in range(80):
        q = sup.single("w5_solution_set", "hard", seed)
        if q.answer_text.startswith("{"):
            eq = _eq_in(q.text)
            lo = 1 if "positive whole" in q.text else 0 if "whole number (0" in q.text else -90
            found = {v for v in range(lo, 91) if _solves(eq, "x", v)}
            assert {int(t) for t in q.answer_text.strip("{}").split(", ")} == found, (seed, q.text, q.answer_text)
            for d in q.distractors:
                assert {int(t) for t in d.strip("{}").split(", ")} != found
        else:
            nums = [
                int(v)
                for v in re.findall(
                    r"-?\d+",
                    re.search(r"(?:solutions x = (-?\d+) and x = (-?\d+)|\{(-?\d+), (-?\d+)\})", q.text).group(0),
                )
            ]
            want = set(nums)

            def roots(eq):
                left, right = _sides(eq)
                return set(sympy.solve(left - right, X))

            assert roots(q.answer_text) == want, (seed, q.text, q.answer_text)
            for d in q.distractors:
                assert roots(d) != want, (seed, q.text, d)


def test_solution_set_exam():
    for seed in range(60):
        q = sup.exam("exd_x_w5_solution_set", seed)
        eq = _eq_in(_stem(q))
        left, right = _sides(eq)
        (s,) = sympy.solve(left - right, N)
        pa, pb, pc = q.parts
        assert pa.answer_text == f"n = {s}"
        for d in pa.distractors:
            assert not _solves(eq, "n", sympy.Rational(d.removeprefix("n = ")))
        assert int(pb.answer_text) == left.subs(N, s) == right.subs(N, s)
        assert sympy.solve(_sides(pc.answer_text)[0] - _sides(pc.answer_text)[1], N) == [s]
        for d in pc.distractors:
            assert sympy.solve(_sides(d)[0] - _sides(d)[1], N) != [s]


# -- identities and no solution ---------------------------------------------------------------------------------------


def _describe(left, right) -> str:
    sol = sympy.solveset(sympy.expand(left - right), X, sympy.S.Reals)
    if sol == sympy.S.Reals:
        return "Every number is a solution"
    if sol == sympy.S.EmptySet:
        return "No solution"
    (v,) = tuple(sol)
    return f"Exactly one solution, x = {v}"


def test_identity_classification():
    for tier in ("easy", "medium", "hard"):
        for seed in range(80):
            q = sup.single("w5_identity_no_solution", tier, seed)
            if "ends with" in q.text:
                shown = _eq_in(q.text)
                left, right = _sides(shown)
                if left == X:
                    want = f"Exactly one solution, x = {right}"
                else:
                    want = "Every number is a solution" if left == right else "No solution"
            elif " k" in q.text and "k =" not in q.text and "kx" in q.text:
                eq = _eq_in(q.text)
                mode = "none" if "no solution" in q.text else "all"
                left, right = _sides(eq)
                kv = int(q.answer_text.removeprefix("k = "))
                assert _describe(left, right.subs(K, kv)) == (
                    "No solution" if mode == "none" else "Every number is a solution"
                )
                for d in q.distractors:
                    got = _describe(left, right.subs(K, int(d.removeprefix("k = "))))
                    assert got.startswith("Exactly one"), (seed, q.text, d)
                continue
            else:
                eq = _eq_in(q.text.replace("Solve ", ""))
                left, right = _sides(eq)
                want = _describe(left, right)
            assert q.answer_text == want, (seed, q.text, q.answer_text, want)
            for d in q.distractors:
                assert d != want
                if d.startswith("Exactly one solution"):
                    pass


def test_identity_exam():
    for seed in range(60):
        q = sup.exam("exd_x_w5_identity_no_solution", seed)
        pa, pb, pc = q.parts
        m = re.search(r"equation (\d+)\(x \+ (\d+)\)(?: ([+-]) (\d+))? = \d+x \+ k", _stem(q))
        a, p = int(m.group(1)), int(m.group(2))
        q0 = 0 if m.group(3) is None else (int(m.group(4)) if m.group(3) == "+" else -int(m.group(4)))
        kk = int(re.search(r"k = (-?\d+)", pa.prompt).group(1))
        left = sympy.expand(a * (X + p) + q0)
        r = int(left.subs(X, 0))
        assert _describe(left, a * X + kk) == "No solution"
        assert pa.answer_text == f"No solution, because {r} = {kk} is false"
        for d in pa.distractors:  # every wrong option states a verdict that differs from "no solution"
            assert d.split(",")[0] != "No solution", d
        assert int(pb.answer_text) == r
        assert _describe(left, a * X + int(pb.answer_text)) == "Every number is a solution"
        assert pc.answer_text == "No solution"


# -- checking by substitution -----------------------------------------------------------------------------------------


def test_check_easy_medium_hard():
    for tier in sup.TIERS:
        for seed in range(80):
            q = sup.single("w5_check_substitution", tier, seed)
            if "mistake" not in q.text and ("correct working" in q.text or "Which line shows" in q.text):
                if m := re.search(r"left side of (.+?) when x = (-?\d+)", q.text):
                    expr_text, v = m.group(1), int(m.group(2))
                else:
                    m = re.search(r"Substitute x = (-?\d+) into (.+?)\. Which line", q.text)
                    expr_text, v = m.group(2), int(m.group(1))
                expr = _e(expr_text)
                value = expr.subs(X, v)
                a = int(sympy.Poly(expr, X).coeff_monomial(X))
                b = int(sympy.Poly(expr, X).coeff_monomial(1))
                sign, mag = ("+", b) if b > 0 else ("-", -b)
                assert q.answer_text == f"{a} × {v} {sign} {mag} = {value}", (seed, q.text, q.answer_text)
                for d in q.distractors:
                    assert d != q.answer_text
                    lhs, rhs = d.split(" = ")
                    assert _e(lhs) == int(rhs), (seed, d)
            elif "mistake" in q.text:
                m = re.search(
                    r"(?:the expression |works out )(\d+)\(x \+ (\d+)\) ([+-]) (\d+)(?: for| and)|(\d+)\(x \+ (\d+)\) ([+-]) (\d+)",
                    q.text,
                )
                a, p, sgn, qq = next(g for g in (m.groups()[:4], m.groups()[4:]) if g[0])
                a, p, qq = int(a), int(p), int(qq) * (1 if sgn == "+" else -1)
                v = (
                    int(re.search(r"x = (-?\d+)", q.text).group(1))
                    if "x = " in q.text
                    else int(re.search(r"for x = (-?\d+)", q.text).group(1))
                )
                shown = int(re.search(r"(?:gets|equals|as) (-?\d+)", q.text).group(1))
                true_value = a * (v + p) + qq
                assert shown != true_value
                effects = {
                    "She multiplied only the x inside the bracket by the number outside it": a * v + p + qq,
                    "She treated the negative value of x as if it were positive": a * (abs(v) + p) + qq,
                    "She did not multiply the bracket by the number outside it": (v + p) + qq,
                    "She changed the sign of the last number in the expression": a * (v + p) - qq,
                    "She made no mistake": true_value,
                }
                assert effects[q.answer_text] == shown, (seed, q.text, q.answer_text)
                for d in q.distractors:
                    assert effects[d] != shown, (seed, q.text, d)
            else:
                m = re.search(r"x = (-?\d+)", q.text)
                v = int(m.group(1))
                eq = _eq_in(
                    q.text.split(" is a solution of ")[-1]
                    if " is a solution of " in q.text
                    else q.text.split(" into ")[-1]
                )
                left, right = _sides(eq)
                L, R = left.subs(X, v), right.subs(X, v)
                good = f"Left side {L}, right side {R}: x = {v} {'is' if L == R else 'is not'} a solution"
                assert q.answer_text == good, (seed, q.text, q.answer_text)
                for d in q.distractors:
                    assert d != good


def test_check_exam():
    for seed in range(60):
        q = sup.exam("exd_x_w5_check_substitution", seed)
        v = int(re.search(r"whether x = (\d+) is a solution", _stem(q)).group(1))
        eq = _eq_in(_stem(q).split("solution of")[1])
        left, right = _sides(eq)
        pa, pb, pc, pd = q.parts
        L, R = left.subs(X, v), right.subs(X, v)
        assert int(pa.answer_text) == L and int(pb.answer_text) == R
        (s,) = sympy.solve(left - right, X)
        if L == R:  # the proposed value is the solution
            assert v == s
            assert pc.answer_text == f"Yes: both sides equal {L}"
            assert pd.answer_text == f"x = {v} is the only solution"
            for d in pc.distractors:  # the verdict is yes with the right value, so each wrong option differs
                assert d != pc.answer_text and (d.startswith("No") or f"equal {L}" not in d), d
            for d in pd.distractors:
                assert d.startswith(("Every number", "There is no solution", f"x = {v} and x = ")), d
        else:
            assert pc.answer_text == f"No: the sides are not equal ({L} and {R})"
            assert pd.answer_text == f"x = {s}"
            for d in pd.distractors:
                assert sympy.simplify((left - right).subs(X, sympy.Rational(d.removeprefix("x = ")))) != 0
