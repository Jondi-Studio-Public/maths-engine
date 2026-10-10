"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p10a_antiderivative_ic_build` -- the
derivative expression and the initial condition (a, b) are parsed back out of
the rendered prose with a regular expression, then the antiderivative is
re-derived with `sympy.integrate()` (a different implementation from the
generator's own hand-rolled term-by-term power-rule loop) and the constant
solved by direct substitution, never by calling into the generator.
"""

import re
from fractions import Fraction

import pytest
from sympy import Eq, integrate, solve, symbols, sympify

from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

TOPIC = "p10a_antiderivative_given_fprime"

_x, _C = symbols("x C")

_HEAD_RE = re.compile(r"^f'\(x\) = (?P<fprime>.+?)\. If f\((?P<a>-?\d+)\) = (?P<b>-?\d+), find f\(x\)\.$")


def _parse_poly(prose: str):
    """A generator-rendered polynomial ("5·x^2 + 6", "-8·x - 5", "4·x - 8")
    back into a sympy expression in x. `·` is the generator's own stand-in
    for `*`; `^` is its stand-in for `**`. Both are exactly reversible."""
    py_text = prose.replace("^", "**").replace("·", "*")
    return sympify(py_text, locals={"x": _x})


def _oracle(question):
    """Re-derive the answer from the rendered text alone: parse f'(x), a, b;
    integrate f'(x) with sympy.integrate() (never the generator's own
    term-by-term power-rule construction); solve b = integral(a) + C for C
    via sympy.solve (never by directly subtracting, though the two routes
    agree); return the final F(x) with C substituted in."""
    head = _HEAD_RE.search(question.text)
    assert head is not None, f"could not parse question text: {question.text!r}"
    f_prime_expr = _parse_poly(head.group("fprime"))
    a = int(head.group("a"))
    b = int(head.group("b"))

    g_expr = integrate(f_prime_expr, _x)
    equation = Eq(g_expr.subs(_x, a) + _C, b)
    c_solutions = solve(equation, _C)
    assert len(c_solutions) == 1, f"expected exactly one solution for C, got {c_solutions}"
    f_expr = g_expr + c_solutions[0]
    return f_expr, a, b


def _parse_typed(text: str):
    return sympify(text.replace("^", "**"))


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_f, a, b = _oracle(q)
        rendered = _parse_typed(q.answer_text)
        assert (oracle_f - rendered).expand() == 0, (
            f"seed={seed}/{tier}: oracle {oracle_f} != rendered answer {rendered}\ntext: {q.text}"
        )
        # cross-check the initial condition directly against the parsed text
        assert oracle_f.subs(_x, a) == b, f"seed={seed}/{tier}: oracle F(a) != stated b"
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_f, _a, _b = _oracle(q)
        for d in q.distractors:
            candidate = _parse_typed(d)
            assert (oracle_f - candidate).expand() != 0, (
                f"seed={seed}/{tier}: distractor {d!r} equals the oracle's own answer"
            )


def test_oracle_hard_tier_ignored_negative_base_sign_would_be_rejected():
    """The `ignored-negative-base-sign` misconception (hard tier only) uses
    |a| in place of the stated negative a when solving for C. Recompute it
    purely from the parsed text -- f'(x), a and b -- and confirm it differs
    from the oracle's own independently-derived answer. Exercises the
    misconception's value even on seeds where pick_distractors's n=3 cap
    disclosure)."""
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        head = _HEAD_RE.search(q.text)
        f_prime_expr = _parse_poly(head.group("fprime"))
        a = int(head.group("a"))
        b = int(head.group("b"))
        assert a < 0, f"seed={seed}: hard tier a is not negative: {a}"

        oracle_f, _a, _b = _oracle(q)
        g_expr = integrate(f_prime_expr, _x)
        # solve using |a| instead of the stated (negative) a
        equation = Eq(g_expr.subs(_x, abs(a)) + _C, b)
        wrong_c = solve(equation, _C)[0]
        wrong_f = g_expr + wrong_c
        assert (wrong_f - oracle_f).expand() != 0, (
            f"seed={seed}: |a| variant {wrong_f} equals the oracle answer {oracle_f}"
        )
        checked += 1
    assert checked == len(SEEDS)
