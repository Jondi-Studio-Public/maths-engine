"""Generates `salg_binomial_coefficient` questions (easy, medium, hard) and the exam chain
`salg_exam_binomial_coefficient`.

Skill ``senior-recognise-the-numbers-as-binomial-coefficients`` (``senior-mathematics``, answer ``number``): the
number C(n, r) is the coefficient of ``x^r y^(n-r)`` in the expansion of ``(x + y)^n``, so a single coefficient can be
found without expanding the whole power.

* easy: the coefficient of ``x^r y^(n-r)`` in ``(x + y)^n`` for n = 4 to 8 and 2 <= r <= n - 2, or the value of C(n, r) said to be that
  coefficient. The answer is C(n, r) itself.
* medium: the coefficient of ``x^r`` in ``(x + b)^n`` or in ``(1 + bx)^n`` for a small integer b of either sign, so the
  binomial coefficient is multiplied by a power of b whose exponent has to be worked out.
* hard: the coefficient of ``x^r`` in ``(ax + b)^n`` (two numbers raised to powers), or the term independent of x in
  ``(ax + b/x)^n``, where the term has to be found first.
* exam (``salg_exam_binomial_coefficient``): for ``(px + q)^n``, (a) C(n, r); (b) the coefficient of x^r, using (a)
  (follows through); (c) the coefficient of x^(r+1); (d) the coefficient of x^(r+1) in ``(px + q)^(n+1)``, which is
  ``p`` times (b) plus ``q`` times (c) (follows through from both).

Each coefficient is computed from ``math.comb`` and confirmed by reading it off SymPy's expansion of the whole power.
"""

from __future__ import annotations

from math import comb

import sympy as sp

from ..core import check, generator, paren_negative, phrase, pick_distractors
from ._salg_kit import int_dist, sup

_X = sp.Symbol("x")


def _coeff(expr, power: int) -> int:
    """The coefficient of x^power in `expr`, by SymPy (Laurent terms included)."""
    return int(sp.expand(expr).coeff(_X, power))


def _mono(letter: str, e: int) -> str:
    return letter if e == 1 else f"{letter}{sup(e)}"


def _dists(answer: int, cands: list[tuple[int, str]], count: int = 3):
    out = pick_distractors(str(answer), [int_dist(v, tag) for v, tag in cands if v != answer and v != 0], count)
    check(len(out) >= 2, f"too few distractors for {answer}: {cands}")
    return out


# -- easy ----------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Find the coefficient of {term} in the expansion of (x + y){n}.",
    "What is the coefficient of {term} when (x + y){n} is expanded?",
    "Evaluate C({n_plain}, {r}), the coefficient of {term} in the expansion of (x + y){n}.",
)


def _easy(rng):
    n = rng.randint(4, 8)
    r = rng.randint(2, n - 2)
    answer = comb(n, r)
    y = sp.Symbol("y")
    check(int(sp.Poly(sp.expand((_X + y) ** n), _X, y).coeff_monomial(_X**r * y ** (n - r))) == answer, "SymPy differs")
    cands = [
        (comb(n, r + 1) if r + 1 <= n else 0, "binom-wrong-term-picked"),
        (comb(n, r - 1), "binom-wrong-term-picked"),
        (n * r, "binom-permutation-for-combination"),
        (comb(n, r) * r if r > 1 else 0, "binom-permutation-for-combination"),
        (2**n, "binom-row-sum-given"),
    ]
    if n > 3:
        cands.insert(2, (comb(n + 1, r), "binom-wrong-term-picked"))
    perm = 1
    for i in range(r):
        perm *= n - i
    if perm <= 20 * answer:
        cands.insert(2, (perm, "binom-permutation-for-combination"))
    term = _mono("x", r) + _mono("y", n - r)
    stem = phrase(rng, _EASY_STEMS, term=term, n=sup(n), n_plain=n, r=r)
    return dict(
        text=stem,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"The coefficient of x^{r} y^{n - r} in (x + y)^{n} is C({n}, {r}) = {answer}."],
        distractors=_dists(answer, cands),
    )


# -- medium ----------------------------------------------------------------------------------------------------------


def _medium(rng):
    n = rng.randint(4, 7)
    r = rng.randint(1, n - 1)
    b = rng.choice([-3, -2, 2, 3])
    shape = rng.choice(["const", "scaled"])
    if shape == "const":
        expr, k = (_X + b) ** n, n - r  # x^r comes with b^(n-r)
        base = f"(x {'+' if b > 0 else '-'} {abs(b)})"
    else:
        expr, k = (1 + b * _X) ** n, r  # x^r comes with b^r
        base = f"(1 {'+' if b > 0 else '-'} {abs(b)}x)"
    answer = comb(n, r) * b**k
    check(_coeff(expr, r) == answer, "coefficient differs from SymPy")
    other = n - r if shape == "scaled" else r  # the exponent a learner uses by swapping
    cands = [
        (comb(n, r), "binom-numbers-omitted"),
        (comb(n, r) * b, "binom-power-of-base-missed"),
        (comb(n, r) * b**other, "binom-powers-swapped"),
        (
            comb(n, r + 1) * b ** (k - 1) if shape == "const" else comb(n, r - 1) * b ** (r - 1),
            "binom-wrong-term-picked",
        ),
    ]
    if b < 0:
        cands.append((comb(n, r) * abs(b) ** k, "binom-negative-sign-dropped"))
        cands.append((-answer, "binom-negative-sign-dropped"))
    stem = phrase(
        rng,
        (
            "Find the coefficient of x{r} in the expansion of {base}{n}.",
            "What is the coefficient of x{r} when {base}{n} is expanded?",
        ),
        r="" if r == 1 else sup(r),
        base=base,
        n=sup(n),
    )
    return dict(
        text=stem,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"The term in x^{r} is C({n}, {r}) × {paren_negative(b)}^{k}, so the coefficient is {comb(n, r)} × {b**k} = {answer}."
        ],
        distractors=_dists(answer, cands),
    )


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_linear(rng):
    n = rng.randint(4, 6)
    r = rng.randint(1, n - 1)
    a, b = rng.choice([2, 3]), rng.choice([-3, -2, -1, 1, 2, 3])
    answer = comb(n, r) * a**r * b ** (n - r)
    check(_coeff((a * _X + b) ** n, r) == answer, "coefficient differs from SymPy")
    cands = [
        (comb(n, r) * a * b, "binom-power-of-base-missed"),
        (comb(n, r) * b ** (n - r), "binom-first-number-not-raised"),
        (comb(n, r) * a**r, "binom-second-number-not-raised"),
        (comb(n, r) * a ** (n - r) * b**r, "binom-powers-swapped"),
        (comb(n, r + 1) * a ** (r + 1) * b ** (n - r - 1), "binom-wrong-term-picked"),
        (comb(n, r) * a**r * abs(b) ** (n - r), "binom-negative-sign-dropped") if b < 0 else (0, ""),
    ]
    cands = [c for c in cands if c[1]]
    base = f"({a}x {'+' if b > 0 else '-'} {abs(b)})"
    stem = phrase(
        rng,
        (
            "Find the coefficient of x{r} in the expansion of {base}{n}.",
            "What is the coefficient of x{r} in {base}{n}?",
        ),
        r="" if r == 1 else sup(r),
        base=base,
        n=sup(n),
    )
    return dict(
        text=stem,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"The term in x^{r} is C({n}, {r}) × {a}^{r} × {paren_negative(b)}^{n - r} = {answer}."],
        distractors=_dists(answer, cands),
    )


def _hard_constant(rng):
    n = rng.choice([4, 6])
    k = n // 2
    a, b = rng.choice([2, 3]), rng.choice([-3, -2, 2, 3])
    answer = comb(n, k) * a**k * b**k
    check(_coeff((a * _X + sp.Rational(b) / _X) ** n, 0) == answer, "constant term differs from SymPy")
    cands = [
        (comb(n, k), "binom-numbers-omitted"),
        (comb(n, k) * a * b, "binom-power-of-base-missed"),
        (comb(n, k + 1) * a ** (k + 1) * b ** (k - 1), "binom-wrong-term-picked"),
        (comb(n, k - 1) * a ** (k - 1) * b ** (k + 1), "binom-wrong-term-picked"),
        (comb(n, k) * a**n, "binom-first-number-only"),
    ]
    if b < 0:
        cands.append((comb(n, k) * a**k * abs(b) ** k, "binom-negative-sign-dropped"))
    base = f"({a}x {'+' if b > 0 else '-'} {abs(b)}/x)"
    stem = phrase(
        rng,
        (
            "Find the term independent of x in the expansion of {base}{n}.",
            "Find the constant term in the expansion of {base}{n}.",
        ),
        base=base,
        n=sup(n),
    )
    return dict(
        text=stem,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"The term with {k} factors of {a}x and {k} factors of {b}/x has the powers of x cancelling.",
            f"C({n}, {k}) × {a}^{k} × {paren_negative(b)}^{k} = {answer}.",
        ],
        distractors=_dists(answer, cands),
    )


@generator(
    "salg_binomial_coefficient",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def salg_binomial_coefficient(rng, tier="easy"):
    """One coefficient of a binomial expansion; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_linear, _hard_linear, _hard_constant])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("salg_exam_binomial_coefficient", level=11, presets={"exam": dict(tier="exam")})
def salg_exam_binomial_coefficient(rng, tier="exam"):
    """Exam chain: C(n, r), the coefficient of x^r, of x^(r+1), and of x^(r+1) in the next power up."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = rng.choice([5, 6])
    r = rng.choice([2, 3])
    p = rng.choice([2, 3])
    q = rng.choice([v for v in (-4, -3, -2, 2, 3, 4) if abs(v) != p])
    base = f"({p}x {'+' if q > 0 else '-'} {abs(q)})"
    a = comb(n, r)
    b = a * p**r * q ** (n - r)
    c = comb(n, r + 1) * p ** (r + 1) * q ** (n - r - 1)
    d = p * b + q * c
    expr = (p * _X + q) ** n
    check(_coeff(expr, r) == b and _coeff(expr, r + 1) == c, "coefficients differ from SymPy")
    check(_coeff((p * _X + q) ** (n + 1), r + 1) == d, "coefficient in the next power differs from SymPy")
    check(d == comb(n + 1, r + 1) * p ** (r + 1) * q ** (n - r), "Pascal identity differs")

    a_dist = _dists(
        a,
        [
            (comb(n, r + 1), "binom-wrong-term-picked"),
            (comb(n, r - 1), "binom-wrong-term-picked"),
            (comb(n + 1, r), "binom-wrong-term-picked"),
            (n * r, "binom-permutation-for-combination"),
        ],
    )
    b_dist = _dists(
        b,
        [
            (a, "binom-numbers-omitted"),
            (a * p * q, "binom-power-of-base-missed"),
            (a * p ** (n - r) * q**r, "binom-powers-swapped"),
            (a * p**r, "binom-second-number-not-raised"),
            (a * q ** (n - r), "binom-first-number-not-raised"),
        ],
    )
    c_dist = _dists(
        c,
        [
            (comb(n, r + 1), "binom-numbers-omitted"),
            (comb(n, r + 1) * p * q, "binom-power-of-base-missed"),
            (comb(n, r + 1) * p ** (n - r - 1) * q ** (r + 1), "binom-powers-swapped"),
            (comb(n, r + 1) * p ** (r + 1), "binom-second-number-not-raised"),
        ],
    )
    d_dist = _dists(
        d,
        [
            (b + c, "binom-multipliers-omitted"),
            (q * b + p * c, "binom-multipliers-swapped"),
            (p * b, "binom-one-product-only"),
            (q * c, "binom-one-product-only"),
        ],
    )
    return dict(
        text=f"Consider the expansion of {base}{sup(n)} in powers of x.",
        parts=[
            dict(
                prompt=f"Write down the value of C({n}, {r}).",
                answer_kind="number",
                answer=a,
                answer_text=str(a),
                answer_latex=str(a),
                marks=1,
                sensible_range=(0, None),
                working=[f"C({n}, {r}) = {a}."],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Hence find the coefficient of x{sup(r)} in the expansion of {base}{sup(n)}.",
                answer_kind="number",
                answer=b,
                answer_text=str(b),
                answer_latex=str(b),
                marks=2,
                follow_through=f"part_a*{p}**{r}*({q})**{n - r}",
                working=[
                    f"The x{sup(r)} term is C({n}, {r}) × {p}^{r} × {paren_negative(q)}^{n - r}, so the coefficient is {a} × {p**r} × {q ** (n - r)} = {b}."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=f"Find the coefficient of x{sup(r + 1)} in the expansion of {base}{sup(n)}.",
                answer_kind="number",
                answer=c,
                answer_text=str(c),
                answer_latex=str(c),
                marks=2,
                working=[
                    f"The x{sup(r + 1)} term is C({n}, {r + 1}) × {p}^{r + 1} × {paren_negative(q)}^{n - r - 1} = {c}."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Hence find the coefficient of x{sup(r + 1)} in the expansion of {base}{sup(n + 1)}.",
                answer_kind="number",
                answer=d,
                answer_text=str(d),
                answer_latex=str(d),
                marks=2,
                follow_through=f"{p}*part_b + ({q})*part_c",
                working=[
                    f"{base}{sup(n + 1)} = {base}{base}{sup(n)}, so the x{sup(r + 1)} term comes from {p}x times the x{sup(r)} term and {q} times the x{sup(r + 1)} term.",
                    f"{p} × {b} + ({q}) × {c} = {d}.",
                ],
                distractors=d_dist,
            ),
        ],
    )
