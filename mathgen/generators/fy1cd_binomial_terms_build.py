"""Generates `ucv1_binomial_terms` questions and the exam chain `exd_x_fy_binomial_terms`.

``ucv1-binomial-theorem-terms`` (``discrete-mathematics``, answer ``number``): the general term of a binomial
expansion, used to pick out one coefficient. The general term of (a xᵖ + b x⁻ᑫ)ⁿ is the (r + 1)th,
C(n, r) aⁿ⁻ʳ bʳ x^(p(n - r) - q r), and r is found by matching the power of x.

* easy: the coefficient of xᵏ in (a + bx)ⁿ for n up to 7, where the power of x is the index r itself.
* medium: the coefficient of a stated power of x in (a xᵖ + b/xᵠ)ⁿ, where r has to be found from p(n - r) - q r.
* hard: the term independent of x in (a xᵖ + b/xᵠ)ⁿ with n up to 12 (so r comes from p(n - r) = q r), or the
  coefficient of x² or x³ in a product (1 + ax)ⁿ (1 + bx)ᵐ, which has more than one contributing pair of terms.
* exam (``exd_x_fy_binomial_terms``): the term independent of x in (a xᵖ + b/xᵠ)ⁿ. (a) r, (b) C(n, r) (following through
  from (a)), (c) the term itself (from (a) and (b)), (d) the coefficient of the next term (from (a) and (c)).

Each coefficient is re-derived by multiplying the bracket out term by term (no binomial coefficients) and reading it off.
"""

from __future__ import annotations

from math import comb, factorial, gcd

from ..core import attempts, check, generator, pick_distractors
from ._fy1cd_kit import F, SUP, int_dist, say


def xpow(t: int) -> tuple[str, str]:
    """x to the power t: plain text and TeX."""
    if t == 1:
        return "x", "x"
    if 2 <= t <= 6:
        return f"x{SUP[t]}", f"x^{{{t}}}"
    if t > 0:
        return f"x^{t}", f"x^{{{t}}}"
    return f"x^({t})", f"x^{{{t}}}"


def _binomial_formula(a: int, p: int, b: int, q: int, n: int) -> F:
    """(a x^p + b/x^q)^n as a formula."""
    head = "" if a == 1 else str(a)
    x1, x1t = xpow(p)
    mag = abs(b)
    sign = "-" if b < 0 else "+"
    denom, denom_t = xpow(q)
    plain = f"({head}{x1} {sign} {mag}/{denom})^{n}"
    tex = rf"\left({head}{x1t}{sign}\frac{{{mag}}}{{{denom_t}}}\right)^{{{n}}}"
    return F(plain, tex)


def _linear_formula(a: int, b: int, n: int, var: str = "x") -> F:
    mag = abs(b)
    body = var if mag == 1 else f"{mag}{var}"
    plain = f"({a} {'-' if b < 0 else '+'} {body})^{n}"
    return F(plain, rf"\left({a}{'-' if b < 0 else '+'}{body}\right)^{{{n}}}")


def _multiply(f: dict[int, int], g: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, c in f.items():
        for j, d in g.items():
            out[i + j] = out.get(i + j, 0) + c * d
    return out


def _power(base: dict[int, int], n: int) -> dict[int, int]:
    """base**n by repeated multiplication: a Laurent polynomial as {power of x: coefficient}."""
    out = {0: 1}
    for _ in range(n):
        out = _multiply(out, base)
    return out


def _coef_of_power(poly: dict[int, int], t: int) -> int:
    return poly.get(t, 0)


def term(n: int, r: int, a: int, b: int) -> int:
    return comb(n, r) * a ** (n - r) * b**r


def exponent(n: int, r: int, p: int, q: int) -> int:
    return p * (n - r) - q * r


class _Retry(Exception):
    """A draw whose numbers leave too few distinct wrong answers (a = 1 makes several errors coincide)."""


def _need(ds) -> None:
    if len(ds) < 2:
        raise _Retry


def _retrying(build, rng):
    for _ in attempts(100, "a draw with enough distinct wrong answers"):
        try:
            return build(rng)
        except _Retry:
            continue


def _dists(answer, candidates, n: int = 4):
    return pick_distractors(str(answer), [int_dist(v, t) for v, t in candidates], n)


# -- easy -------------------------------------------------------------------------------------------------------


def _easy(rng):
    n = rng.randint(4, 7)
    k = rng.randint(2, n - 1)
    a = rng.randint(1, 3)
    b = rng.choice([-3, -2, -1, 1, 2, 3])
    answer = term(n, k, a, b)
    expr = _power({0: a, 1: b}, n)
    check(_coef_of_power(expr, k) == answer, "the expanded power differs")
    xk = xpow(k)
    stem = say("Find the coefficient of ", F(xk[0], xk[1]), " in the expansion of ", _linear_formula(a, b, n), ".")
    cands = [
        (comb(n, k) * b**k, "binom-term-forgot-a-power"),
        (comb(n, k) * a**k * b ** (n - k), "binom-term-powers-swapped"),
        (term(n, k - 1, a, b), "binom-term-off-by-one"),
        (term(n, k + 1, a, b) if k + 1 <= n else term(n, k - 2, a, b), "binom-term-off-by-one"),
        (comb(n, k) * a ** (n - k) * abs(b) ** k, "binom-term-sign-lost"),
        (comb(n, k) * a ** (n - k), "binom-term-forgot-b-power"),
    ]
    ds = _dists(answer, cands)
    _need(ds)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"The term in x^{k} is C({n}, {k}) × {a}^{n - k} × ({b})^{k}.",
            f"{comb(n, k)} × {a ** (n - k)} × {b**k} = {answer}.",
        ],
        distractors=ds,
    )


# -- medium -----------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(500, "a power of x that is not 0, 1 or the extreme"):
        p = rng.choice([1, 2, 3])
        q = rng.choice([1, 2])
        n = rng.randint(5, 9)
        a = rng.choice([1, 1, 2, 3])
        b = rng.choice([-3, -2, -1, 1, 2, 3])
        r = rng.randint(1, n - 1)
        t = exponent(n, r, p, q)
        if t != 0 and len({exponent(n, i, p, q) for i in range(n + 1)}) == n + 1:
            break
    answer = term(n, r, a, b)
    expr = _power({p: a, -q: b}, n)
    check(_coef_of_power(expr, t) == answer, "the expanded power differs")
    xt = xpow(t)
    stem = say(
        "Find the coefficient of ",
        F(xt[0], xt[1]),
        " in the expansion of ",
        _binomial_formula(a, p, b, q, n),
        ".",
    )
    cands = [
        (term(n, r - 1, a, b), "binom-term-off-by-one"),
        (term(n, r + 1, a, b), "binom-term-off-by-one"),
        (comb(n, r) * b**r, "binom-term-forgot-a-power"),
        (comb(n, r) * a**r * b ** (n - r), "binom-term-powers-swapped"),
        (comb(n, r) * a ** (n - r) * abs(b) ** r, "binom-term-sign-lost"),
        (term(n, n - r, a, b), "binom-count-from-wrong-end"),
        (comb(n, r), "binom-term-coefficients-dropped"),
    ]
    ds = _dists(answer, cands)
    _need(ds)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"The (r + 1)th term has x to the power {p}({n} - r) - {q}r = {p * n} - {p + q}r.",
            f"Setting this equal to {t} gives r = {r}.",
            f"The coefficient is C({n}, {r}) × {a}^{n - r} × ({b})^{r} = {answer}.",
        ],
        distractors=ds,
    )


# -- hard -------------------------------------------------------------------------------------------------------

_INDEPENDENT = [
    (p, q, n)
    for p in (1, 2, 3)
    for q in (1, 2, 3)
    for n in range(6, 13)
    if gcd(p, q) == 1 and (p * n) % (p + q) == 0 and 0 < (p * n) // (p + q) < n
]


def _hard_independent(rng):
    p, q, n = rng.choice(_INDEPENDENT)
    a = rng.choice([1, 1, 2])
    b = rng.choice([-2, -1, 1, 2])
    r = p * n // (p + q)
    answer = term(n, r, a, b)
    expr = _power({p: a, -q: b}, n)
    check(_coef_of_power(expr, 0) == answer, "the expanded independent term differs")
    stem = say("Find the term independent of x in the expansion of ", _binomial_formula(a, p, b, q, n), ".")
    cands = [
        (term(n, r - 1, a, b), "binom-term-off-by-one"),
        (term(n, r + 1, a, b), "binom-term-off-by-one"),
        (comb(n, r), "binom-term-coefficients-dropped"),
        (term(n, n // 2, a, b), "binom-independent-term-middle"),
        (term(n, n - r, a, b), "binom-term-powers-swapped"),
    ]
    ds = _dists(answer, cands)
    _need(ds)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"The (r + 1)th term has x to the power {p * n} - {p + q}r, which is 0 when r = {r}.",
            f"The term is C({n}, {r}) × {a}^{n - r} × ({b})^{r} = {answer}.",
        ],
        distractors=ds,
    )


def _hard_product(rng):
    n, m = rng.randint(4, 6), rng.randint(3, 5)
    a, b = rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2])
    t = rng.choice([2, 3])
    expr = _multiply(_power({0: 1, 1: a}, n), _power({0: 1, 1: b}, m))
    answer = sum(comb(n, i) * a**i * comb(m, t - i) * b ** (t - i) for i in range(t + 1) if t - i <= m)
    check(_coef_of_power(expr, t) == answer, "the expanded power differs")
    xt = xpow(t)
    stem = say(
        "Find the coefficient of ",
        F(xt[0], xt[1]),
        " in the expansion of ",
        _linear_formula(1, a, n),
        " ",
        _linear_formula(1, b, m),
        ".",
    )
    alone = comb(n, t) * a**t + comb(m, t) * b**t
    cands = [
        (alone, "binom-product-ignores-cross-terms"),
        (comb(n, t) * a**t * comb(m, t) * b**t, "binom-product-multiplies-coefficients"),
        (
            sum(comb(n, i) * a**i * comb(m, t - i) * b ** (t - i) for i in range(1, t) if t - i <= m),
            "binom-product-drops-end-terms",
        ),
        (answer - comb(n, t) * a**t, "binom-product-drops-end-terms"),
    ]
    ds = _dists(answer, cands)
    _need(ds)
    pairs = [f"C({n}, {i})({a})^{i} × C({m}, {t - i})({b})^{t - i}" for i in range(t + 1) if 0 <= t - i <= m and i <= n]
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"x^{t} comes from every pair of terms whose powers add to {t}: " + " + ".join(pairs) + ".",
            f"Adding them gives {answer}.",
        ],
        distractors=ds,
    )


@generator(
    "ucv1_binomial_terms",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_binomial_terms(rng, tier="easy"):
    """Coefficients from the binomial theorem; the tiers are described in the module docstring."""
    if tier == "easy":
        return _retrying(_easy, rng)
    if tier == "medium":
        return _retrying(_medium, rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retrying(rng.choice([_hard_independent, _hard_independent, _hard_product]), rng)


# -- exam -------------------------------------------------------------------------------------------------------

_EXAM_SHAPES = [
    (p, q, n)
    for p in (1, 2, 3, 4)
    for q in (1, 2, 3, 4)
    for n in range(6, 13)
    if gcd(p, q) == 1 and (p * n) % (p + q) == 0 and 0 < (p * n) // (p + q) < n - 1
]


@generator("exd_x_fy_binomial_terms", level=18, presets={"exam": dict(tier="exam")})
def exd_x_fy_binomial_terms(rng, tier="exam"):
    """Exam chain: r, C(n, r), the term independent of x, and the coefficient of the next term."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _retrying(_exam, rng)


def _exam(rng):
    p, q, n = rng.choice(_EXAM_SHAPES)
    a = rng.choice([1, 2, 2, 3, 3, 4])
    b = rng.choice([-3, -2, -1, 1, 2, 3])
    r = p * n // (p + q)
    c = comb(n, r)
    indep = term(n, r, a, b)
    if abs(term(n, r + 1, a, b)) > 10**8 or abs(indep) > 10**8:
        raise _Retry
    nxt = term(n, r + 1, a, b)
    e_next = exponent(n, r + 1, p, q)
    check(e_next == -(p + q), "the next power is not -(p + q)")
    expr = _power({p: a, -q: b}, n)
    check(_coef_of_power(expr, 0) == indep, "the expanded independent term differs")
    check(_coef_of_power(expr, e_next) == nxt, "the expanded next coefficient differs")

    a_cands = [
        (r + 1, "binom-term-number-confusion"),
        (n - r, "binom-count-from-wrong-end"),
        (n // 2, "binom-independent-term-middle"),
        (r - 1, "binom-term-off-by-one"),
    ]
    b_cands = [
        (comb(n, r + 1), "binom-term-off-by-one"),
        (comb(n, r - 1), "binom-term-off-by-one"),
        (n * r, "binom-coefficient-as-product"),
        (comb(n, n // 2), "binom-independent-term-middle"),
        (factorial(n) // factorial(r), "binom-coefficient-forgot-denominator"),
        (comb(n + 1, r), "binom-term-off-by-one"),
    ]
    c_cands = [
        (c, "binom-term-coefficients-dropped"),
        (c * b**r, "binom-term-forgot-a-power"),
        (c * a ** (n - r), "binom-term-forgot-b-power"),
        (term(n, n - r, a, b), "binom-term-powers-swapped"),
        (c * a ** (n - r) * abs(b) ** r, "binom-term-sign-lost"),
    ]
    d_cands = [
        (term(n, r + 2, a, b), "binom-term-off-by-one"),
        (term(n, r - 1, a, b), "binom-term-off-by-one"),
        (-nxt, "binom-term-sign-flipped"),
        (indep, "binom-next-term-not-updated"),
        (comb(n, r + 1), "binom-term-coefficients-dropped"),
    ]
    a_dist, b_dist = _dists(r, a_cands), _dists(c, b_cands)
    c_dist, d_dist = _dists(indep, c_cands), _dists(nxt, d_cands)

    # A later option that a learner's own earlier (wrong) answers lead to by following through is right for them, so
    # it is not offered as a wrong one.
    a_values = [r] + [int(d.text) for d in a_dist]
    b_values = [c] + [int(d.text) for d in b_dist]

    def ft_b(av):
        return comb(n, av) if 0 <= av <= n else None

    def ft_c(av, bv):
        return bv * a ** (n - av) * b**av if 0 <= av <= n else None

    def ft_d(av, cv):
        return (
            cv * (n - av) * b // (a * (av + 1)) if 0 <= av <= n and (cv * (n - av) * b) % (a * (av + 1)) == 0 else None
        )

    b_dist = [d for d in b_dist if int(d.text) not in {ft_b(av) for av in a_values} - {c}]
    own_c = {ft_c(av, bv) for av in a_values for bv in b_values} - {indep}
    c_dist = [d for d in c_dist if int(d.text) not in own_c]
    c_values = [indep] + [int(d.text) for d in c_dist]
    own_d = {ft_d(av, cv) for av in a_values for cv in c_values} - {nxt}
    d_dist = [d for d in d_dist if int(d.text) not in own_d]
    for ds in (a_dist, b_dist, c_dist, d_dist):
        _need(ds)

    form = _binomial_formula(a, p, b, q, n)
    stem = say("Consider the expansion of ", form, ", written in descending powers of x.")
    xe = xpow(e_next)
    return dict(
        text=stem[0],
        latex=stem[1],
        parts=[
            dict(
                prompt="The term independent of x is the (r + 1)th term. Find r.",
                prompt_latex=r"\text{The term independent of }x\text{ is the }(r+1)\text{th term. Find }r\text{.}",
                answer_kind="number",
                answer=r,
                answer_text=str(r),
                answer_latex=str(r),
                marks=2,
                sensible_range=(0, n),
                working=[f"The (r + 1)th term has x to the power {p * n} - {p + q}r, which is 0 for r = {r}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence find the binomial coefficient in that term.",
                answer_kind="number",
                answer=c,
                answer_text=str(c),
                answer_latex=str(c),
                marks=1,
                follow_through=f"factorial({n})/(factorial(part_a)*factorial({n}-part_a))",
                sensible_range=(0, None),
                working=[f"C({n}, {r}) = {c}."],
                distractors=b_dist,
            ),
            dict(
                prompt="Hence find the term independent of x.",
                answer_kind="number",
                answer=indep,
                answer_text=str(indep),
                answer_latex=str(indep),
                marks=2,
                follow_through=f"part_b*({a})**({n}-part_a)*({b})**part_a",
                working=[f"The term is {c} × {a}^{n - r} × ({b})^{r} = {indep}."],
                distractors=c_dist,
            ),
            dict(
                prompt=f"The next term in the expansion contains x to the power {e_next}. Find its coefficient.",
                prompt_latex=rf"\text{{The next term in the expansion contains }}x^{{{e_next}}}\text{{. Find its coefficient.}}",
                answer_kind="number",
                answer=nxt,
                answer_text=str(nxt),
                answer_latex=str(nxt),
                marks=3,
                follow_through=f"part_c*({n}-part_a)/(part_a+1)*({b})/({a})",
                working=[
                    f"Each term is ({n} - r)/(r + 1) × ({b}/{a}) times the one before, so the next coefficient is {indep} × {n - r}/{r + 1} × {b}/{a} = {nxt}."
                ],
                distractors=d_dist,
            ),
        ],
    )
