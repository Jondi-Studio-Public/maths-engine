"""Generates `salg_binomial_expand` questions (easy, medium, hard) and the exam chain `salg_exam_binomial_expand`.

Skill ``senior-expand-x-y-n-for-small-positive-integers-n`` (``senior-mathematics``, answer ``expression``, graded
with ``answer_form="expanded"`` so an unexpanded power such as ``(x + 2)^4`` is not accepted): expand a power of a
binomial for a small positive integer n, reading the coefficients from Pascal's triangle.

* easy: ``(x + y)^n`` for n = 3 to 6, in letters that change from question to question. The work is the row of
  Pascal's triangle and the falling and rising powers.
* medium: ``(x + b)^n`` with b a non-zero integer other than 1, n = 3 to 5. Every term after the first carries a power
  of b, and a negative b alternates the signs.
* hard: ``(ax + b)^n`` or ``(ax + by)^n`` with a of 2 or 3 and a negative or positive b, n = 4 or 5 (3 or 4 for two
  letters), so a is raised to a power as well as b.
* exam (``salg_exam_binomial_expand``): (a) expand ``(x + a)^3``; (b) hence expand ``(x + a)^4`` by multiplying (a)
  by ``x + a`` (follows through from (a)); (c) the coefficient of x² in (b); (d) the sum of the coefficients, found
  by substituting x = 1 into the expansion in (b) (follows through from (b)).

Every expansion is built from the binomial theorem term by term and confirmed against SymPy's ``expand``. Each
distractor is the expansion a learner gets by making one named error, and is confirmed to differ from the key.
"""

from __future__ import annotations

from math import comb

import sympy as sp

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, latex_answer, phrase, pick_distractors
from ._salg_kit import int_dist, shown, sup, typed

_LETTERS = (("x", "y"), ("a", "b"), ("p", "q"), ("m", "k"), ("u", "v"), ("s", "t"))
_FORM = "expanded"


def _mono(letter: str, e: int) -> str:
    return "" if e == 0 else letter if e == 1 else f"{letter}^{e}"


def _term_var(first: str, second: str | None, i: int, j: int) -> str:
    """The variable part of the term in `first^i second^j` (`second` is None when the second term is a number)."""
    return _mono(first, i) + (_mono(second, j) if second else "")


def _build(n: int, coeffs, first: str, second: str | None) -> tuple[str, sp.Expr]:
    """The expansion text and SymPy expression for the terms `coeffs[k] * first^(n-k) * second^k`."""
    text = join_terms([(coeffs[k], _term_var(first, second, n - k, k)) for k in range(n + 1)])
    syms = {first: sp.Symbol(first)}
    if second:
        syms[second] = sp.Symbol(second)
    expr = sum(
        coeffs[k] * sp.Symbol(first) ** (n - k) * (sp.Symbol(second) ** k if second else 1) for k in range(n + 1)
    )
    return text, sp.expand(expr)


def _true_coeffs(n: int, a: int, b: int) -> list[int]:
    return [comb(n, k) * a ** (n - k) * b**k for k in range(n + 1)]


def _wrong_expansions(n: int, a: int, b: int, *, with_pascal: bool = True) -> list[tuple[str, list[int]]]:
    """(error id, coefficient list) for the slips that matter at this (n, a, b)."""
    out: list[tuple[str, list[int]]] = []
    ks = range(n + 1)
    out.append(("binom-pascal-coefficients-omitted", [a ** (n - k) * b**k for k in ks]))
    out.append(("binom-middle-terms-dropped", [a**n if k == 0 else b**n if k == n else 0 for k in ks]))
    row = [comb(n - 1, k) for k in range(n)] + [1]
    out.append(("binom-coefficients-from-previous-row", [row[k] * a ** (n - k) * b**k for k in ks]))
    if abs(a) != 1 or abs(b) != 1:
        out.append(
            (
                "binom-base-not-raised",
                [comb(n, k) * (a if n - k > 0 else 1) * (b if k > 0 else 1) for k in ks],
            )
        )
        out.append(
            (
                "binom-base-times-exponent",
                [comb(n, k) * (a * (n - k) if n - k > 0 else 1) * (b * k if k > 0 else 1) for k in ks],
            )
        )
    if b < 0:
        out.append(("binom-sign-not-alternating", [comb(n, k) * a ** (n - k) * (-(abs(b) ** k) if k else 1) for k in ks]))
        out.append(("binom-negative-sign-dropped", [comb(n, k) * a ** (n - k) * abs(b) ** k for k in ks]))
    return out


def _distractors(n, a, b, first, second, true_expr, true_text, count=3) -> list[Distractor]:
    cands = []
    seen = [true_expr]
    for tag, coeffs in _wrong_expansions(n, a, b):
        text, expr = _build(n, coeffs, first, second)
        if any(sp.expand(expr - other) == 0 for other in seen):
            continue
        seen.append(expr)
        cands.append(Distractor(text, tag, typed(expr)))
    picked = pick_distractors(true_text, cands, count)
    check(len(picked) >= 2, f"too few expansion distractors for n={n}, a={a}, b={b}")
    return picked


def _base_text(a: int, b: int, first: str, second: str | None) -> str:
    return "(" + join_terms([(a, first), (b, second or "")]) + ")"


def _working(n, a, b, first, second, coeffs, text):
    row = ", ".join(str(comb(n, k)) for k in range(n + 1))
    steps = [f"Row {n} of Pascal's triangle is {row}."]
    if abs(a) != 1 or abs(b) != 1:
        steps.append(
            "Each term is the coefficient times a power of "
            + (f"{join_terms([(a, first)])}" if a != 1 else first)
            + " times a power of "
            + (f"({b})" if second is None else join_terms([(b, second)]))
            + ", with the powers of the first falling from "
            + f"{n} to 0 and the powers of the second rising from 0 to {n}."
        )
    else:
        steps.append(f"The powers of {first} fall from {n} to 0 while the powers of the other letter rise from 0 to {n}.")
    steps.append(f"{_base_text(a, b, first, second)}{sup(n)} = {text}.")
    return steps


_STEMS = ("Expand {e}.", "Expand and simplify {e}.", "Write {e} as a sum of terms.", "Find the expansion of {e}.")


def _question(rng, n, a, b, first, second):
    coeffs = _true_coeffs(n, a, b)
    text, expr = _build(n, coeffs, first, second)
    # Independent route: SymPy expands the power.
    f, s = sp.Symbol(first), (sp.Symbol(second) if second else 1)
    check(sp.expand((a * f + b * s) ** n - expr) == 0, "binomial expansion differs from SymPy")
    dist = _distractors(n, a, b, first, second, expr, text)
    stem = phrase(rng, _STEMS, e=f"{_base_text(a, b, first, second)}{sup(n)}")
    return dict(
        text=stem,
        answer_kind="expression",
        answer=typed(expr),
        answer_text=text,
        answer_latex=latex_answer(expr),
        answer_form=_FORM,
        working=_working(n, a, b, first, second, coeffs, text),
        distractors=dist,
    )


def _easy(rng):
    n = rng.choice([3, 4, 5, 6])
    first, second = rng.choice(_LETTERS)
    return _question(rng, n, 1, 1, first, second)


def _medium(rng):
    n = rng.choice([3, 4, 5])
    b = rng.choice([-4, -3, -2, 2, 3, 4, 5])
    return _question(rng, n, 1, b, rng.choice("xtmpuz"), None)


def _hard(rng):
    a = rng.choice([2, 3])
    b = rng.choice([-3, -2, -1, 1, 2, 3])
    if rng.random() < 0.5:
        return _question(rng, rng.choice([4, 5]), a, b, "x", None)
    return _question(rng, rng.choice([3, 4]), a, b, *rng.choice(_LETTERS))


@generator(
    "salg_binomial_expand",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def salg_binomial_expand(rng, tier="easy"):
    """Expand a small power of a binomial; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("salg_exam_binomial_expand", level=11, presets={"exam": dict(tier="exam")})
def salg_exam_binomial_expand(rng, tier="exam"):
    """Exam chain: expand (x + a)^3, hence (x + a)^4, a coefficient, and the sum of the coefficients."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a = rng.choice([-3, -2, 2, 3, 4])
    x = sp.Symbol("x")
    base = f"(x {'+' if a > 0 else '-'} {abs(a)})"

    c3, c4 = _true_coeffs(3, 1, a), _true_coeffs(4, 1, a)
    text3, e3 = _build(3, c3, "x", None)
    text4, e4 = _build(4, c4, "x", None)
    check(sp.expand((x + a) ** 3 - e3) == 0 and sp.expand((x + a) ** 4 - e4) == 0, "expansion differs from SymPy")
    check(sp.expand((x + a) * e3 - e4) == 0, "(x + a) times the cube differs from the fourth power")
    coef2 = comb(4, 2) * a**2
    check(sp.Poly(e4, x).coeff_monomial(x**2) == coef2, "coefficient of x^2 differs from SymPy")
    total = (1 + a) ** 4
    check(e4.subs(x, 1) == total, "sum of coefficients differs from SymPy")

    # (a) the cube: the same slips as any single expansion.
    a_dist = _distractors(3, 1, a, "x", None, e3, text3)
    # (b) the fourth power.
    b_dist = _distractors(4, 1, a, "x", None, e4, text4)
    # (c) the coefficient of x^2: C(4,2) a^2.
    c_dist = pick_distractors(
        str(coef2),
        [
            int_dist(comb(4, 2), "binom-base-not-raised"),
            int_dist(comb(4, 2) * a, "binom-power-of-base-missed"),
            int_dist(comb(4, 1) * a**3, "binom-wrong-term-picked"),
            int_dist(comb(4, 3) * a, "binom-wrong-term-picked"),
            int_dist(4 * a * a, "binom-pascal-coefficient-misread"),
        ],
        3,
    )
    # (d) the sum of the coefficients: (1 + a)^4.
    d_dist = pick_distractors(
        str(total),
        [
            int_dist(1 + a**4, "binom-middle-terms-dropped"),
            int_dist(4 * (1 + a), "binom-base-times-exponent"),
            int_dist(a**4, "binom-constant-term-only"),
            int_dist(sum(abs(c) for c in c4), "binom-negative-sign-dropped"),
            int_dist(16, "binom-pascal-coefficients-only"),
        ],
        3,
    )
    for label, ds in (("a", a_dist), ("b", b_dist), ("c", c_dist), ("d", d_dist)):
        check(len(ds) >= 2, f"too few distractors in part {label}")

    sign = "+" if a > 0 else "-"
    ft_b = f"(x {sign} {abs(a)})*part_a"
    return dict(
        text=f"Consider powers of the binomial x {sign} {abs(a)}.",
        parts=[
            dict(
                prompt=f"Expand {base}{sup(3)}.",
                answer_kind="expression",
                answer=typed(e3),
                answer_text=text3,
                answer_latex=latex_answer(e3),
                answer_form=_FORM,
                marks=2,
                working=[f"Row 3 of Pascal's triangle is 1, 3, 3, 1.", f"{base}{sup(3)} = {text3}."],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Hence expand {base}{sup(4)}.",
                answer_kind="expression",
                answer=typed(e4),
                answer_text=text4,
                answer_latex=latex_answer(e4),
                answer_form=_FORM,
                marks=2,
                follow_through=ft_b,
                working=[f"{base}{sup(4)} = {base}{base}{sup(3)}, so multiply your answer to (a) by {base}.", f"{base}{sup(4)} = {text4}."],
                distractors=b_dist,
            ),
            dict(
                prompt=f"State the coefficient of x{sup(2)} in the expansion of {base}{sup(4)}.",
                answer_kind="number",
                answer=coef2,
                answer_text=str(coef2),
                answer_latex=str(coef2),
                marks=1,
                sensible_range=(0, None),
                working=[f"The x{sup(2)} term is C(4, 2) × x{sup(2)} × ({a}){sup(2)} = {coef2}x{sup(2)}."],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Substitute x = 1 into your expansion in (b) to find the sum of its coefficients.",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=str(total),
                marks=2,
                sensible_range=(0, None),
                follow_through="Subs(part_b, (x,), (1,))",
                working=[f"At x = 1 the expansion equals (1 {sign} {abs(a)}){sup(4)} = {total}."],
                distractors=d_dist,
            ),
        ],
    )
