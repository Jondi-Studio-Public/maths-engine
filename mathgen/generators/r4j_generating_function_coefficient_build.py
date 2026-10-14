"""Generates `uc2-generating-function-coefficient` questions (easy, medium, hard) and the exam chain
`exd_x_r4j_generating_function_coefficient`.

``uc2-generating-function-coefficient`` (``discrete-mathematics``, answer ``number``): the coefficient of x^k in a
generating function written in closed form. The learner decides how to expand it; no formula is named.

* easy: a product of two geometric series, 1/((1 - ax)(1 - bx)): the coefficient is a convolution.
* medium: a shifted, scaled negative binomial series, x^s / (1 - ax)^r.
* hard: a bounded count. Either ((1 - x^(u+1)) / (1 - x))^3, the number of ways to write n as an ordered sum of three
  parts each at most u, or (1 - x^a)^2 / (1 - x)^4; the numerator's terms cut the unrestricted count.
* exam (``exd_x_r4j_generating_function_coefficient``): sharing k identical tickets among three friends, two of them
  capped. (a) the unrestricted count, (b) and (c) the counts that break one cap, (d) the capped count by inclusion and
  exclusion, which follows through from (a) to (c).

Every answer is computed twice: from a closed form in binomial coefficients, and by expanding the series term by term
(multiplying and dividing truncated integer series). The tests add a third route with SymPy.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._r4j_kit import comb, keep_distinct, neg_binomial, say

_TAGS = {
    "added": "gf-series-coefficients-added",
    "multiplied": "gf-series-coefficients-multiplied",
    "terms": "gf-convolution-term-count",
    "scale": "gf-scale-factor-dropped",
    "scale_linear": "gf-scale-factor-not-powered",
    "shift": "gf-shift-ignored",
    "power": "gf-negbinomial-power-off-by-one",
    "choose": "gf-negbinomial-choose-swapped",
    "numerator": "gf-bounded-numerator-ignored",
    "truncated": "gf-correction-terms-truncated",
    "cutoff": "gf-cutoff-off-by-one",
    "sign": "gf-inclusion-exclusion-sign",
    "denominator": "gf-wrong-denominator-power",
    "index": "gf-coefficient-index-off-by-one",
}


# -- series arithmetic (the second route) -------------------------------------------------------------------------


def _times_geometric(series, a, p, k):
    """The series times 1/(1 - a x^p), truncated at degree k."""
    out = list(series)
    for i in range(p, k + 1):
        out[i] += a * out[i - p]
    return out


def _times_poly(series, poly, k):
    """The series times a polynomial (a list of coefficients, constant first), truncated at degree k."""
    out = [0] * (k + 1)
    for i, c in enumerate(series):
        if c == 0:
            continue
        for j, d in enumerate(poly):
            if i + j <= k:
                out[i + j] += c * d
    return out


def _one(k):
    return [1] + [0] * k


def _ft(a):
    """`(1 − x)` or `(1 − 2x)`."""
    return "(1 − x)" if a == 1 else f"(1 − {a}x)"


def _ft_tex(a):
    return "(1-x)" if a == 1 else f"(1-{a}x)"


def _xpow(k):
    """(`x^6`, `x^{6}`) and the bare `x` for k = 1."""
    return ("x", "x") if k == 1 else (f"x^{k}", f"x^{{{k}}}")


_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _ask(rng, k, gf):
    """One of four phrasings of "find the coefficient of x^k in this generating function"."""
    form = rng.randrange(4)
    if form == 0:
        return say("Find the coefficient of ", _xpow(k), " in the expansion of ", gf, " as a power series in x.")
    if form == 1:
        return say("Expand ", gf, " as a power series in x. What is the coefficient of ", _xpow(k), "?")
    if form == 2:
        return say(
            "The generating function G(x) = ", gf, " has the series G(x) = Σ gₙxⁿ. Find g", str(k).translate(_SUB), "."
        )
    return say("Write ", gf, " as a power series in x and give the term in ", _xpow(k), ", without its power of x.")


def _out(text, latex, value, distractors, working):
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        latex=latex,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        working=working,
        distractors=distractors,
    )


def _cands(value, items):
    return keep_distinct(value, [(v, tag, str(v)) for v, tag in items if v > 0])


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    a, b = rng.choice([(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4)])
    if rng.random() < 0.5:
        a, b = b, a
    k = rng.randint(4, 7)
    series = _times_geometric(_times_geometric(_one(k), a, 1, k), b, 1, k)
    value = series[k]
    check(value * (a - b) == a ** (k + 1) - b ** (k + 1), "convolution and closed form disagree")
    gf = (f"1/({_ft(a)}{_ft(b)})", rf"\frac{{1}}{{{_ft_tex(a)}{_ft_tex(b)}}}")
    text, latex = _ask(rng, k, gf)
    items = [
        (a**k + b**k, _TAGS["added"]),
        (a**k * b**k, _TAGS["multiplied"]),
        ((a**k - b**k) // (a - b), _TAGS["terms"]),
        ((a ** (k + 2) - b ** (k + 2)) // (a - b), _TAGS["terms"]),
    ]
    return _out(
        text,
        latex,
        value,
        _cands(value, items),
        [
            f"1/{_ft(a)} = 1 + {_c(a)}x + {_c(a**2)}x² + …, and 1/{_ft(b)} = 1 + {_c(b)}x + {_c(b**2)}x² + ….",
            f"The coefficient of x^{k} in the product is the sum over i from 0 to {k} of {a}^i · {b}^({k} − i).",
            f"That sum is {value}.",
        ],
    )


def _c(n):
    return "" if n == 1 else str(n)


# -- medium --------------------------------------------------------------------------------------------------------


def _medium(rng):
    a = rng.choice([2, 3])
    r = rng.choice([3, 4])
    s = rng.choice([1, 2, 3])
    m = rng.randint(3, 5)
    k = s + m
    series = _one(k)
    for _ in range(r):
        series = _times_geometric(series, a, 1, k)
    value = series[k - s]  # the factor x^s moves every coefficient up by s
    check(value == a**m * neg_binomial(m, r), "series and closed form disagree")
    xs = "x" if s == 1 else f"x^{s}"
    xs_tex = "x" if s == 1 else f"x^{{{s}}}"
    gf = (f"{xs}/(1 − {a}x)^{r}", rf"\frac{{{xs_tex}}}{{(1-{a}x)^{{{r}}}}}")
    text, latex = _ask(rng, k, gf)
    items = [
        (neg_binomial(m, r), _TAGS["scale"]),
        (a**k * neg_binomial(k, r), _TAGS["shift"]),
        (a**m * neg_binomial(m, r + 1), _TAGS["power"]),
        (a**m * comb(m + r - 1, r), _TAGS["choose"]),
        (a * m * neg_binomial(m, r), _TAGS["scale_linear"]),
    ]
    return _out(
        text,
        latex,
        value,
        _cands(value, items),
        [
            f"Multiplying by {xs} raises every power of x by {s}, so the coefficient of x^{k} is the coefficient of x^{m} in 1/(1 − {a}x)^{r}.",
            f"1/(1 − y)^{r} has coefficient C({m} + {r} − 1, {r} − 1) of y^{m}; with y = {a}x there is also a factor {a}^{m}.",
            f"{a}^{m} · C({m + r - 1}, {r - 1}) = {value}.",
        ],
    )


# -- hard ----------------------------------------------------------------------------------------------------------


def _hard_bounded_cube(rng):
    u = rng.randint(4, 6)
    n = rng.randint(2 * u + 2, 3 * u - 1)
    closed = comb(n + 2, 2) - 3 * comb(n - u - 1 + 2, 2) + 3 * comb(n - 2 * u - 2 + 2, 2) - comb(n - 3 * u - 3 + 2, 2)
    k = n
    direct = _times_poly(_times_poly(_one(k), [1] * (u + 1), k), [1] * (u + 1), k)
    direct = _times_poly(direct, [1] * (u + 1), k)[k]
    numer = _times_poly(_one(k), [1] + [0] * u + [-1], k)  # (1 - x^(u+1)), cubed below
    numer = _times_poly(_times_poly(numer, [1] + [0] * u + [-1], k), [1] + [0] * u + [-1], k)
    quotient = numer
    for _ in range(3):
        quotient = _times_geometric(quotient, 1, 1, k)
    check(closed == direct == quotient[k], "three routes to the bounded count disagree")
    gf = (f"((1 − x^{u + 1})/(1 − x))^3", rf"\left(\frac{{1-x^{{{u + 1}}}}}{{1-x}}\right)^{{3}}")
    if rng.random() < 0.34:
        text, latex = say(
            f"How many ordered triples (p, q, r) of whole numbers, each at most {u}, have p + q + r = {n}?"
        )
    else:
        text, latex = _ask(rng, k, gf)

    items = [
        (comb(n + 2, 2), _TAGS["numerator"]),
        (comb(n + 2, 2) - 3 * comb(n - u - 1 + 2, 2), _TAGS["truncated"]),
        (
            comb(n + 2, 2) - 3 * comb(n - u + 2, 2) + 3 * comb(n - 2 * u + 2, 2) - comb(n - 3 * u + 2, 2),
            _TAGS["cutoff"],
        ),
        (comb(n + 2, 2) + 3 * comb(n - u - 1 + 2, 2) + 3 * comb(n - 2 * u - 2 + 2, 2), _TAGS["sign"]),
    ]
    return _out(
        text,
        latex,
        closed,
        _cands(closed, items),
        [
            f"(1 − x^{u + 1})/(1 − x) = 1 + x + … + x^{u}, so the coefficient counts ordered triples of parts from 0 to {u} with sum {n}.",
            f"Expand (1 − x^{u + 1})³ = 1 − 3x^{u + 1} + 3x^{2 * u + 2} − x^{3 * u + 3} and use 1/(1 − x)³ = Σ C(j + 2, 2) x^j.",
            f"C({n + 2}, 2) − 3·C({n - u + 1}, 2) + 3·C({n - 2 * u}, 2) = {closed}.",
        ],
    )


def _hard_squared_numerator(rng):
    a = rng.randint(3, 5)
    k = rng.randint(2 * a, 2 * a + 3)
    closed = neg_binomial(k, 4) - 2 * neg_binomial(k - a, 4) + neg_binomial(k - 2 * a, 4)
    quotient = _times_poly(_one(k), [1] + [0] * (a - 1) + [-1], k)
    quotient = _times_poly(quotient, [1] + [0] * (a - 1) + [-1], k)
    for _ in range(4):
        quotient = _times_geometric(quotient, 1, 1, k)
    check(closed == quotient[k], "series and closed form disagree")
    gf = (f"(1 − x^{a})^2/(1 − x)^4", rf"\frac{{(1-x^{{{a}}})^{{2}}}}{{(1-x)^{{4}}}}")
    if rng.random() < 0.34:
        text, latex = say(
            f"How many solutions in whole numbers does w + x + y + z = {k} have if w ≤ {a - 1} and x ≤ {a - 1}?"
        )
    else:
        text, latex = _ask(rng, k, gf)
    items = [
        (neg_binomial(k, 4), _TAGS["numerator"]),
        (neg_binomial(k, 4) - 2 * neg_binomial(k - a, 4), _TAGS["truncated"]),
        (neg_binomial(k, 4) + 2 * neg_binomial(k - a, 4) + neg_binomial(k - 2 * a, 4), _TAGS["sign"]),
        (neg_binomial(k, 4) - 2 * neg_binomial(k - a + 1, 4) + neg_binomial(k - 2 * a + 2, 4), _TAGS["cutoff"]),
        (neg_binomial(k, 3) - 2 * neg_binomial(k - a, 3) + neg_binomial(k - 2 * a, 3), _TAGS["denominator"]),
    ]
    return _out(
        text,
        latex,
        closed,
        _cands(closed, items),
        [
            f"(1 − x^{a})² = 1 − 2x^{a} + x^{2 * a}, and 1/(1 − x)⁴ = Σ C(j + 3, 3) x^j.",
            f"The coefficient of x^{k} is C({k + 3}, 3) − 2·C({k - a + 3}, 3) + C({k - 2 * a + 3}, 3).",
            f"That is {closed}.",
        ],
    )


def _hard(rng):
    return _hard_bounded_cube(rng) if rng.random() < 0.5 else _hard_squared_numerator(rng)


@generator(
    "r4j_generating_function_coefficient",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4j_generating_function_coefficient(rng, tier="easy"):
    """The coefficient of x^k in a closed-form generating function; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator("exd_x_r4j_generating_function_coefficient", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4j_generating_function_coefficient(rng, tier="exam"):
    """Exam chain: the unrestricted count, the two counts that break a cap, then the capped count by inclusion-exclusion."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "caps a < b with room for both"):
        a, b = rng.randint(3, 5), rng.randint(4, 6)
        if a < b:
            break
    k = a + b + rng.randint(1, 3)
    total, over_a, over_b, over_both = (
        neg_binomial(k, 3),
        neg_binomial(k - a, 3),
        neg_binomial(k - b, 3),
        neg_binomial(k - a - b, 3),
    )
    capped = total - over_a - over_b + over_both
    # second route: expand (1 - x^a)(1 - x^b) / (1 - x)^3 term by term
    series = _times_poly(_one(k), [1] + [0] * (a - 1) + [-1], k)
    series = _times_poly(series, [1] + [0] * (b - 1) + [-1], k)
    for _ in range(3):
        series = _times_geometric(series, 1, 1, k)
    check(series[k] == capped, "series and inclusion-exclusion disagree")
    check(capped > 0 and over_both > 0, "degenerate chain")

    f_tex = rf"F(x)=\frac{{(1-x^{{{a}}})(1-x^{{{b}}})}}{{(1-x)^{{3}}}}"
    f_pair = (f"F(x) = (1 − x^{a})(1 − x^{b})/(1 − x)^3", f_tex)
    scene = rng.randrange(3)
    if scene == 0:
        stem, stem_tex = say(
            f"Priya shares {k} identical tickets among three friends: Ana, Ben and Cai. Ana can take at most {a - 1} tickets, "
            f"Ben at most {b - 1}, and Cai any number. The number of ways is the coefficient of ",
            _xpow(k),
            " in ",
            f_pair,
            ".",
        )
    elif scene == 1:
        stem, stem_tex = say(
            f"A coach hands out {k} identical medals to three players, Dev, Eli and Fay. Dev may receive at most {a - 1}, Eli at "
            f"most {b - 1}, and Fay as many as are left. With ",
            f_pair,
            f", the number of ways is the coefficient of ",
            _xpow(k),
            " in F(x).",
        )
    else:
        stem, stem_tex = say(
            f"{k} identical pencils go into three boxes X, Y and Z. Box X holds at most {a - 1} pencils and box Y at most "
            f"{b - 1}; box Z has no limit. Let ",
            f_pair,
            ". The number of ways to place the pencils is the coefficient of ",
            _xpow(k),
            " in F(x).",
        )

    def part(prompt_parts, value, marks, distractors, working, follow=None):
        prompt, prompt_tex = say(*prompt_parts)
        d = dict(
            prompt=prompt,
            prompt_latex=prompt_tex,
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            marks=marks,
            working=working,
            distractors=distractors,
            sensible_range=(0, None),
        )
        if follow:
            d["follow_through"] = follow
        return d

    def shifted(q):  # distractors for "the coefficient of x^k in x^s / (1 - x)^3", where q = k - s
        return [
            (neg_binomial(q, 4), _TAGS["power"]),
            (neg_binomial(k, 3), _TAGS["shift"]),
            (comb(q + 2, 3), _TAGS["choose"]),
            (neg_binomial(q - 1, 3), _TAGS["index"]),
        ]

    parts = [
        part(
            ["Find the coefficient of ", _xpow(k), " in ", ("1/(1 − x)^3", r"\frac{1}{(1-x)^{3}}"), "."],
            total,
            2,
            _cands(
                total,
                [
                    (neg_binomial(k, 4), _TAGS["power"]),
                    (k + 1, _TAGS["denominator"]),
                    (comb(k + 2, 3), _TAGS["choose"]),
                ],
            ),
            [f"1/(1 − x)³ = Σ C(j + 2, 2) x^j, so the coefficient is C({k + 2}, 2) = {total}."],
        ),
        part(
            [
                "Find the coefficient of ",
                _xpow(k),
                " in ",
                (f"x^{a}/(1 − x)^3", rf"\frac{{x^{{{a}}}}}{{(1-x)^{{3}}}}"),
                ".",
            ],
            over_a,
            2,
            _cands(over_a, shifted(k - a)),
            [f"The factor x^{a} moves the coefficient of x^{k - a} to x^{k}: C({k - a + 2}, 2) = {over_a}."],
        ),
        part(
            [
                "Find the coefficient of ",
                _xpow(k),
                " in ",
                (f"x^{b}/(1 − x)^3", rf"\frac{{x^{{{b}}}}}{{(1-x)^{{3}}}}"),
                ".",
            ],
            over_b,
            2,
            _cands(over_b, shifted(k - b)),
            [f"Likewise C({k - b + 2}, 2) = {over_b}."],
        ),
        part(
            ["Hence find the coefficient of ", _xpow(k), " in F(x): the number of ways."],
            capped,
            4,
            _cands(
                capped,
                [
                    (total - over_a - over_b, _TAGS["truncated"]),
                    (total - over_a - over_b - over_both, _TAGS["sign"]),
                    (total - over_a, _TAGS["truncated"]),
                    (total, _TAGS["numerator"]),
                ],
            ),
            [
                f"(1 − x^{a})(1 − x^{b}) = 1 − x^{a} − x^{b} + x^{a + b}, so the answer is (a) − (b) − (c) + the coefficient of x^{k} in x^{a + b}/(1 − x)³.",
                f"That last term is C({k - a - b + 2}, 2) = {over_both}, giving {total} − {over_a} − {over_b} + {over_both} = {capped}.",
            ],
            follow=f"part_a - part_b - part_c + {over_both}",
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, latex=stem_tex, parts=parts)
