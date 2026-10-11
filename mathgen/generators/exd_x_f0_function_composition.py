"""Exam-tier source for `function-notation`: function notation used as an operation, not a substitution.

Two kinds of question, both built backwards from chosen coefficients so the key is exact:

  composition -- two named functions, one linear and one quadratic, and a request for f(g(n)) or g(f(n)).
                 The order of application is part of the question.
  reverse     -- "Given that f(px + q) = <polynomial in x>, find f(m)". The rule for f is never shown; the
                 learner works out which x makes the argument equal m (or recovers f) before evaluating.
                 The polynomial is the expansion of A(px+q)^2 + B(px+q) + C, so f is recovered uniquely.
"""

from __future__ import annotations

import random

from sympy import Poly, Symbol, expand

from ..core import check, generator, join_terms, pick_distractors

_x = Symbol("x")


def _poly_text(coeffs: dict[int, int]) -> str:
    """Render {degree: coefficient} (degrees 2, 1, 0) as '3x^2 - x + 4'."""
    names = {2: "x^2", 1: "x", 0: ""}
    return join_terms([(coeffs.get(d, 0), names[d]) for d in (2, 1, 0)])


def _signed(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([-1, 1]) * rng.randint(lo, hi)


def _pack(text: str, label: str, value: int, working: list[str], wrong: list[int]) -> dict:
    answer_text = f"{label} = {value}"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=value,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [f"{label} = {w}" for w in wrong]),
    )


def _composition(rng: random.Random) -> dict:
    for _ in range(200):
        quad = {2: rng.randint(1, 3), 1: rng.randint(-5, 5), 0: rng.randint(-6, 6)}
        lin = {1: _signed(rng, 1, 4), 0: rng.randint(-6, 6)}
        n = rng.randint(-3, 4)
        quad_is_f = rng.random() < 0.5
        f_c, g_c = (quad, lin) if quad_is_f else (lin, quad)

        def f(v):
            return sum(c * v**d for d, c in f_c.items())

        def g(v):
            return sum(c * v**d for d, c in g_c.items())

        outer_first = rng.random() < 0.5  # True: ask f(g(n)); False: ask g(f(n))
        value = f(g(n)) if outer_first else g(f(n))
        swapped = g(f(n)) if outer_first else f(g(n))
        if len({value, swapped, f(n) * g(n), f(n) + g(n)}) == 4 and abs(value) < 400:
            break
    else:
        check(False, "no usable composition")
    # Independent check with sympy substitution.
    fe = sum(c * _x**d for d, c in f_c.items())
    ge = sum(c * _x**d for d, c in g_c.items())
    expect = fe.subs(_x, ge.subs(_x, n)) if outer_first else ge.subs(_x, fe.subs(_x, n))
    check(expect == value, "composition disagrees with sympy substitution")
    call = f"f(g({n}))" if outer_first else f"g(f({n}))"
    text = f"Let f(x) = {_poly_text(f_c)} and g(x) = {_poly_text(g_c)}. Find {call}."
    inner = g(n) if outer_first else f(n)
    working = [
        f"Work from the inside: {'g' if outer_first else 'f'}({n}) = {inner}.",
        f"Then {call} = {'f' if outer_first else 'g'}({inner}) = {value}.",
    ]
    wrong = [swapped, f(n) * g(n), f(n) + g(n)]
    return _pack(text, call, value, working, wrong)


def _reverse(rng: random.Random) -> dict:
    for _ in range(200):
        a = rng.randint(1, 3)
        b = rng.randint(-5, 5)
        c = rng.randint(-6, 6)
        p = _signed(rng, 1, 3)
        q = rng.randint(-5, 5)
        x0 = rng.randint(-3, 4)
        m = p * x0 + q
        f_of_m = a * m * m + b * m + c
        rhs = expand(a * (p * _x + q) ** 2 + b * (p * _x + q) + c)
        coeffs = {d: int(Poly(rhs, _x).coeff_monomial(_x**d)) for d in (2, 1, 0)}
        rhs_at = lambda v: sum(co * v**d for d, co in coeffs.items())  # noqa: E731
        # the polynomial read at the argument itself; the inner rule applied once more; x instead of f(m)
        candidates = [rhs_at(m), rhs_at(p * m + q), x0]
        if m != 0 and len({f_of_m, *candidates}) == 1 + len(candidates) and abs(f_of_m) < 400 and p != 1:
            break
    else:
        check(False, "no usable reverse-function question")
    # Independent check: the stated polynomial evaluated at the x that makes the argument equal m.
    check((m - q) % p == 0 and rhs_at((m - q) // p) == f_of_m, "reverse substitution disagrees with the key")
    text = f"Given that f({join_terms([(p, 'x'), (q, '')])}) = {_poly_text(coeffs)}, find f({m})."
    working = [
        f"f({m}) needs {join_terms([(p, 'x'), (q, '')])} = {m}, so x = {x0}.",
        f"Then f({m}) = {_poly_text(coeffs).replace('x', f'({x0})')} = {f_of_m}.",
    ]
    return _pack(text, f"f({m})", f_of_m, working, candidates)


@generator(
    "exd_x_f0_function_composition",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_f0_function_composition(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "this topic only builds the exam tier")
    return rng.choice([_composition, _reverse])(rng)
