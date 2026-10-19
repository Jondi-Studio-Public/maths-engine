"""Exam-tier source for ``nth-term-rule``: recover a linear nth-term rule from two relations.

The practice tiers give the rule's raw material directly (consecutive terms, or two terms at known
positions). Here the learner is told only how terms relate to each other, for example "term 7 is 3
times term 2" together with the value of one more term (or the sum of two terms), and must turn the
statements into a pair of linear equations in the first term ``a`` and the common difference ``d``
before writing the rule. The ratio statement is homogeneous, so it fixes ``d/a`` and the second
statement fixes the scale.

Built backwards: ``a`` and ``d`` are drawn first, then a ratio statement that is true of that
sequence is searched for, then the second statement is read off the same sequence. The pair of
equations is re-solved with sympy and must have exactly the drawn solution.
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, expand, linsolve, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    phrase,
    pick_distractors,
    symbolically_equal,
)

x = Symbol("x")
_A, _D = symbols("A D")

_VALUE_WORDINGS = (
    "In an arithmetic sequence, term {r} is {k} times term {s}, and term {u} is {V}. "
    "Find the nth-term rule (use x for the term number).",
    "An arithmetic sequence has term {r} equal to {k} times term {s}. Its term {u} is {V}. "
    "Find the nth-term rule (use x for the term number).",
)
_SUM_WORDINGS = (
    "In an arithmetic sequence, term {r} is {k} times term {s}, and the sum of terms {p} and {q} is {S}. "
    "Find the nth-term rule (use x for the term number).",
    "The sum of terms {p} and {q} of an arithmetic sequence is {S}, and term {r} is {k} times term {s}. "
    "Find the nth-term rule (use x for the term number).",
)


def _term(a, d, n):
    return a + (n - 1) * d


def _rule(a, d):
    return expand(Rational(a) + Rational(d) * (x - 1))


def _solve(eq1, eq2):
    """Unique (A, D) solution of two linear equations, or None."""
    sol = linsolve([eq1, eq2], _A, _D)
    if len(sol) != 1:
        return None
    a, d = next(iter(sol))
    if not (a.is_number and d.is_number):
        return None
    return a, d


def _ratio_eq(r, s, k):
    return _term(_A, _D, r) - k * _term(_A, _D, s)


def _draw(rng: random.Random) -> dict:
    for _ in range(500):
        a = rng.randint(-6, 9)
        d = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
        ratios = [
            (r, s, k)
            for s in range(2, 6)
            for r in range(s + 1, 10)
            for k in (2, 3, 4)
            if _term(a, d, s) != 0 and _term(a, d, r) == k * _term(a, d, s)
        ]
        if not ratios:
            continue
        r, s, k = rng.choice(ratios)
        if rng.random() < 0.5:
            u = rng.choice([n for n in range(1, 10) if n not in (r, s)])
            second = ("value", dict(u=u, V=_term(a, d, u)), _term(_A, _D, u) - _term(a, d, u))
        else:
            p, q = sorted(rng.sample(range(1, 10), 2))
            total = _term(a, d, p) + _term(a, d, q)
            second = ("sum", dict(p=p, q=q, S=total), _term(_A, _D, p) + _term(_A, _D, q) - total)
        sol = _solve(_ratio_eq(r, s, k), second[2])
        if sol is None or sol != (a, d):
            continue
        return dict(a=a, d=d, r=r, s=s, k=k, kind=second[0], slots=second[1], second=second[2])
    check(False, "could not draw an exam-tier nth-term question")


def _build(rng: random.Random) -> dict | None:
    q = _draw(rng)
    a, d, r, s, k = q["a"], q["d"], q["r"], q["s"], q["k"]
    wordings = _VALUE_WORDINGS if q["kind"] == "value" else _SUM_WORDINGS
    text = phrase(rng, wordings, r=r, s=s, k=k, **q["slots"])

    rule = _rule(a, d)
    # Second route: rebuild the rule from the sequence's own terms and compare.
    check_equivalent(rule, expand(_term(a, d, 1) + d * (x - 1)))
    check(_term(a, d, r) == k * _term(a, d, s), "ratio statement is not true of the sequence")

    # Misconceptions: (1) read "k times" as "k more than" the other term; (2) forgot the -1 offset
    # (d*x + a); (3) swapped first term and common difference; (4) d*x + d (used d as the first term).
    more_than = _solve((_term(_A, _D, r)) - (_term(_A, _D, s) + k), q["second"])
    candidates = []
    # Keep only a whole-number misread, so no distractor is given away by a fraction.
    if more_than is not None and all(v.is_integer for v in more_than):
        candidates.append(_rule(*more_than))
    candidates += [expand(d * x + a), expand(a * x + d), expand(d * x + d)]
    wrong = []
    for c in candidates:
        if symbolically_equal(rule, c) or any(symbolically_equal(w, c) for w in wrong):
            continue
        wrong.append(c)
    if len(wrong) < 3:
        return None
    wrong = wrong[:3]
    check_distinct(rule, *wrong)

    ans_str = str(rule)
    return dict(
        text=text,
        answer=ans_str,
        answer_text=ans_str,
        answer_latex=latex_answer(rule),
        working=[
            f"Let the first term be a and the common difference d. Term {r} = {k} × term {s} gives "
            f"{join_terms([(1, 'a'), (r - 1, 'd')])} = {k}({join_terms([(1, 'a'), (s - 1, 'd')])}). "
            f"The other statement gives a second linear equation. "
            f"Solving gives a = {a} and d = {d}, so the rule is {ans_str}."
        ],
        distractors=pick_distractors(ans_str, [str(w) for w in wrong]),
    )


@generator(
    "exd_x_f1_nth_term",
    level=8,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f1_nth_term(rng: random.Random, tier: str = "exam") -> dict:
    """Linear nth-term rule recovered from a ratio relation between two terms plus one more fact."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(50):
        built = _build(rng)
        if built is not None:
            return built
    check(False, "no draw with three distinct distractors")
