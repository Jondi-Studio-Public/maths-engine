"""Generic recipe asks: parts about any subject that declares the properties an ask needs.

Ask names and the properties they need follow ``subject-vocabulary.md`` (the grid's operations table). Each ask
reads the subject's hooks and checks the answer a second way from ``subject.expr`` (a SymPy expression in
``X``) with SymPy, so a subject cannot disagree with its own function. Subjects are thin grid cells; the asking,
the grading kinds and the self-checks live here once.

Hooks, by ask (a subject needs the property AND the hooks; until a SymPy-expression base ``Subject`` exists the hooks
are a convention, documented here):

| Ask | Needs | Hooks |
|---|---|---|
| ``implied_domain`` | ``restricted_domain`` | ``domain()`` (SymPy ``Interval``), ``domain_working()``, ``domain_wrong()`` as ``[(text, tag)]``, ``fn_text()`` |
| ``stationary_point`` | ``has_stationary_point`` | ``stationary_x()``, ``stationary_working()``, ``stationary_wrong()`` as ``[(value, tag)]``, ``fn_text()``, ``domain()`` |
| ``solve_equal_constant`` | ``has_surd_roots`` | ``level_text()``, ``level_value()``, ``solutions()``, ``solution_text()``, ``solution_value()``, ``level_working()``, ``solution_wrong()`` as ``[(text, tag)]``, ``fn_text()``, ``domain()`` |

``stationary_point`` asks for the x-coordinate (a whole-number answer) of a maximum.
"""

from __future__ import annotations

import sympy as sp

from ..answers import Distractor, exact_value
from ..core import check, phrase, pick_distractors
from ..grading.inequality import read_solution_set, set_text
from ..recipes import ask
from .recipe_subject_log_quadratic import X


@ask(None, "implied_domain", needs=("restricted_domain",), default_marks=2)
def implied_domain(s, rng, ctx) -> dict:
    domain = s.domain()
    text = set_text(domain, "x")
    read = read_solution_set(text)
    check(read is not None and read[0] == domain, "the domain text does not read back as the domain")
    # second route: the function is defined (real) just inside each end and not at or beyond the ends
    inside = [domain.start + (domain.end - domain.start) * sp.Rational(i, 10) for i in (1, 5, 9)]
    outside = [domain.start, domain.end, domain.start - 1, domain.end + 1]
    for x in inside:
        check(sp.im(sp.N(s.expr.subs(X, x))) == 0 and sp.N(s.expr.subs(X, x)).is_finite, f"f({x}) is undefined")
    for x in outside:
        v = sp.N(s.expr.subs(X, x))
        check(not v.is_real or not v.is_finite, f"f({x}) is defined outside the domain")
    cands = [Distractor(t, tag, value=t) for t, tag in s.domain_wrong()]
    prompt = phrase(
        rng,
        (
            "Find the implied domain of {f}.",
            "State the maximal domain of {f}.",
            "Find the set of real x for which {f}(x) is defined, as an inequality.",
        ),
        f=s.fn_text(),
    )
    return dict(
        prompt=prompt,
        answer_kind="inequality",
        answer=text,
        answer_text=text,
        answer_latex="plain:" + text,
        working=[*s.domain_working(), f"So {text}."],
        distractors=pick_distractors(text, cands, 3),
    )


@ask(None, "stationary_point", needs=("has_stationary_point",), default_marks=1)
def stationary_point(s, rng, ctx) -> dict:
    value = s.stationary_x()
    found = sp.solve(sp.diff(s.expr, X), X)
    check(found == [value], f"stationary points {found} disagree with {value}")
    domain = s.domain()
    check(domain.start < value < domain.end, "the stationary point lies outside the domain")
    check(sp.diff(s.expr, X, 2).subs(X, value) < 0, "the stationary point is not a maximum")
    cands = [Distractor(str(v), tag, value=exact_value(v)) for v, tag in s.stationary_wrong()]
    prompt = phrase(
        rng,
        (
            "Find the x-coordinate of the local maximum of {f}.",
            "Find the value of x at which {f} has its stationary point.",
            "At what value of x does the graph of {f} have a local maximum?",
        ),
        f=s.fn_text(),
    )
    return dict(
        prompt=prompt,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=str(value),
        sensible_range=(int(domain.start), int(domain.end)),
        working=s.stationary_working(),
        distractors=pick_distractors(str(value), cands, 3),
    )


@ask(None, "solve_equal_constant", needs=("has_surd_roots",), default_marks=2)
def solve_equal_constant(s, rng, ctx) -> dict:
    domain = s.domain()
    for root in s.solutions():
        check(sp.simplify(s.expr.subs(X, root) - s.level_value()) == 0, "a solution does not satisfy f(x) = c")
        check(domain.start < root < domain.end, "a solution lies outside the domain")
    text = s.solution_text()
    cands = [Distractor(t, tag) for t, tag in s.solution_wrong()]
    prompt = phrase(
        rng,
        (
            "Solve {f}(x) = {rhs} for x. Give your answer in the form x = p ± √q.",
            "Find the exact values of x for which {f}(x) = {rhs}, in the form x = p ± √q.",
            "Solve the equation {f}(x) = {rhs}, giving exact solutions in the form x = p ± √q.",
        ),
        f=s.fn_text(),
        rhs=s.level_text(),
    )
    return dict(
        prompt=prompt,
        answer_kind="expression",
        answer=s.solution_value(),
        answer_text=text,
        answer_latex="plain:" + text,
        working=s.level_working(),
        distractors=pick_distractors(text, cands, 3),
    )
