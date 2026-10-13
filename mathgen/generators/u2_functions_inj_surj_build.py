"""Generates `u2_functions_injective_surjective` questions and the exam chain `exd_x_u2_functions_injective_surjective`.

``ucv1-functions-injective-surjective`` (``discrete-mathematics``, answer ``number``): one-to-one, onto and bijective
functions, counted between small finite sets and decided for a quadratic rule on the real line.

* easy: count the injective functions between finite sets, or the bijections of a set that obey two fixed values.
* medium: count the surjective functions (inclusion-exclusion over the targets that are missed), or the functions that
  are not injective (a complement).
* hard: a quadratic rule with its turning point somewhere other than the origin: the least `a` for which it is
  injective on `[a, infinity)` (or the greatest `a` on `(-infinity, a]`), the least value it must take for the codomain
  `[m, infinity)` to be hit, and the number of multipliers `a` for which `x -> a x + b` is a bijection of the integers
  modulo `n`.
* exam: one quadratic, restricted to the domain on which it is injective: (a) where the domain starts, (b) the smallest
  codomain that makes it onto, (c) the inverse at a value, (d) the number of integers in the domain with `f(x)` at most
  a stated bound.

Every count is re-derived by listing the functions (a brute-force enumeration, with the sets kept tiny) or, for the
quadratic and the modular rule, by evaluating the rule on a grid of integers.
"""

from __future__ import annotations

import itertools
from math import comb, factorial, gcd, isqrt

from ..answers import Distractor
from ..core import attempts, check, generator
from ._u2_kit import int_d, keep, number_part, number_q, redraws

TOPIC = "u2_functions_injective_surjective"
EXAM_TOPIC = "exd_x_u2_functions_injective_surjective"


def falling(n, m):
    out = 1
    for k in range(m):
        out *= n - k
    return out


def surjections(m, n):
    """Functions from an m-set onto an n-set: inclusion-exclusion over the targets that are missed."""
    return sum((-1) ** k * comb(n, k) * (n - k) ** m for k in range(n + 1))


def _enumerate(m, n, keep_fn):
    return sum(1 for f in itertools.product(range(n), repeat=m) if keep_fn(f, n))


def _brute_injective(m, n):
    return _enumerate(m, n, lambda f, n: len(set(f)) == m)


def _brute_surjective(m, n):
    return _enumerate(m, n, lambda f, n: len(set(f)) == n)


def _set(n):
    """The set {1, 2, ..., n} written out, with an ellipsis once it is long."""
    if n <= 5:
        return "{" + ", ".join(str(k) for k in range(1, n + 1)) + "}"
    return "{1, 2, ..., " + str(n) + "}"


# -- easy -----------------------------------------------------------------------------------------------------------

_NAMES = ["A", "B"]


def _easy_injective(rng):
    m = rng.choice([3, 4])
    n = rng.choice([5, 6, 7]) if m == 3 else rng.choice([6, 7, 8])
    value = falling(n, m)
    if n <= 6:
        check(value == _brute_injective(m, n), "brute-force injective count differs")
    cands = [
        int_d(n**m, "inj-count-all-functions"),
        int_d(comb(n, m), "inj-count-unordered"),
        int_d(m**n, "inj-count-roles-swapped"),
        int_d(factorial(n), "inj-count-unordered"),
    ]
    text = f"Let A = {_set(m)} and B = {_set(n)}. How many one-to-one (injective) functions f : A → B are there?"
    working = [
        f"The image of 1 can be any of the {n} elements of B, the image of 2 any of the {n - 1} that remain, and so on.",
        f"There are {' × '.join(str(n - k) for k in range(m))} = {value} such functions.",
    ]
    return number_q(text, value, working, keep(str(value), cands, 4))


def _easy_bijection_fixed(rng):
    n = rng.choice([5, 6, 7])
    a, b, c = rng.sample(range(1, n + 1), 3)
    d = rng.choice([v for v in range(1, n + 1) if v not in (b, c)])
    value = factorial(n - 2)
    cands = [
        int_d(factorial(n), "bij-count-ignores-conditions"),
        int_d(factorial(n - 1), "bij-count-one-condition"),
        int_d(factorial(n - 2) * 2, "bij-count-one-condition"),
        int_d(factorial(n - 3), "bij-count-one-condition"),
    ]
    # The two given values must differ for such a bijection to exist.
    return (
        None
        if b == d
        else number_q(
            f"Let X = {_set(n)}. How many bijections g : X → X satisfy g({a}) = {b} and g({c}) = {d}?",
            value,
            [
                f"Two elements of X are already sent to {b} and {d}, so {n - 2} elements and {n - 2} targets remain.",
                f"They can be matched in {n - 2}! = {value} ways.",
            ],
            keep(str(value), cands, 4),
        )
    )


def _easy(rng):
    if rng.random() < 0.55:
        return _easy_injective(rng)
    for _ in attempts(100, "a bijection question"):
        q = _easy_bijection_fixed(rng)
        if q is not None:
            return q
    raise AssertionError  # pragma: no cover


# -- medium ---------------------------------------------------------------------------------------------------------


def _stirling2(m, n):
    """Stirling numbers of the second kind by the recurrence S(m, n) = n S(m-1, n) + S(m-1, n-1)."""
    table = [[0] * (n + 1) for _ in range(m + 1)]
    table[0][0] = 1
    for i in range(1, m + 1):
        for j in range(1, min(i, n) + 1):
            table[i][j] = j * table[i - 1][j] + table[i - 1][j - 1]
    return table[m][n]


def _medium_surjective(rng, m, n):
    value = surjections(m, n)
    check(value == factorial(n) * _stirling2(m, n), "the Stirling-number count differs")
    if m <= 6:
        check(value == _brute_surjective(m, n), "brute-force surjective count differs")
    cands = [
        int_d(n**m - n * (n - 1) ** m, "surj-count-no-add-back"),
        int_d(n**m - n, "surj-count-no-correction"),
        int_d(factorial(n) * comb(m, n), "surj-count-choose-preimages"),
        int_d(n**m, "inj-count-all-functions"),
    ]
    text = (
        f"How many functions f : {_set(m)} → {_set(n)} are onto (surjective), "
        "so that every element of the codomain is the image of at least one element?"
    )
    terms = " ".join(("+" if k % 2 == 0 else "-") + f" {comb(n, k)}·{n - k}^{m}" if k else f"{n}^{m}" for k in range(n))
    working = [
        f"Count all {n}^{m} = {n**m} functions, then remove those that miss a target by inclusion-exclusion.",
        f"The number onto is {terms} = {value}.",
    ]
    return number_q(text, value, working, keep(str(value), cands, 4))


def _medium_not_injective(rng, m, n):
    inj = falling(n, m)
    value = n**m - inj
    cands = [
        int_d(inj, "noninj-count-wrong-complement"),
        int_d(n**m, "inj-count-all-functions"),
        int_d(n**m - comb(n, m), "inj-count-unordered"),
        int_d(n**m - surjections(m, n) if m >= n else n**m - n, "noninj-count-wrong-complement"),
    ]
    text = (
        f"How many functions f : {_set(m)} → {_set(n)} are not one-to-one? "
        "(A function that is not one-to-one sends at least two elements to the same value.)"
    )
    working = [
        f"There are {n}^{m} = {n**m} functions in all, and {inj} of them are one-to-one.",
        f"So {n**m} - {inj} = {value} are not one-to-one.",
    ]
    return number_q(text, value, working, keep(str(value), cands, 4))


_SURJECTIVE_SIZES = [(m, n) for m in range(4, 8) for n in range(2, m)]
_NOT_INJECTIVE_SIZES = [(m, n) for m in range(3, 7) for n in range(max(m, 4), 9)]


def _medium(rng):
    if rng.randrange(len(_SURJECTIVE_SIZES) + len(_NOT_INJECTIVE_SIZES)) < len(_SURJECTIVE_SIZES):
        return _medium_surjective(rng, *rng.choice(_SURJECTIVE_SIZES))
    return _medium_not_injective(rng, *rng.choice(_NOT_INJECTIVE_SIZES))


# -- hard -----------------------------------------------------------------------------------------------------------


def _rule(alpha, h, gamma):
    """Text of alpha x^2 - 2 alpha h x + gamma, with a clean sign for each term."""
    lead = {1: "x²", 2: "2x²", 3: "3x²"}[alpha]
    mid = 2 * alpha * h
    out = lead
    out += f" - {mid}x" if mid > 0 else f" + {-mid}x"
    if gamma:
        out += f" + {gamma}" if gamma > 0 else f" - {-gamma}"
    return out


def _quad(rng, alpha_choices=(1, 2, 3)):
    alpha = rng.choice(alpha_choices)
    h = rng.choice([v for v in range(-5, 7) if v not in (0, 1, -1)])
    gamma = rng.randint(-9, 12)
    return alpha, h, gamma


def _check_quad(alpha, h, gamma):
    def f(x):
        return alpha * x * x - 2 * alpha * h * x + gamma

    # One-to-one from h upward and from h downward, not across h; minimum value at h on a grid.
    check(all(f(h + k) < f(h + k + 1) for k in range(8)), "f is not increasing right of h")
    check(f(h - 1) == f(h + 1), "f is not symmetric about h")
    check(min(f(x) for x in range(h - 12, h + 13)) == f(h), "minimum is not at h")
    return f


def _xh(h) -> str:
    """x - h as a learner writes it: x - 5, x + 3, x."""
    return "x" if h == 0 else (f"x - {h}" if h > 0 else f"x + {-h}")


def _vertex(alpha, h, k) -> str:
    """alpha(x - h)^2 + k written with proper signs: 3(x - 5)² - 66, -2(x + 1)² + 4."""
    lead = {1: "", -1: "-"}.get(alpha, str(alpha))
    inner = "x" if h == 0 else (f"x - {h}" if h > 0 else f"x + {-h}")
    square = f"{lead}x²" if h == 0 else f"{lead}({inner})²"
    return square + ("" if k == 0 else f" + {k}" if k > 0 else f" - {-k}")


def _hard_injective_threshold(rng):
    alpha, h, gamma = _quad(rng)
    _check_quad(alpha, h, gamma)
    right = rng.random() < 0.6
    cands = [
        int_d(-h, "inj-vertex-sign"),
        int_d(2 * h, "inj-vertex-missing-two"),
        int_d(gamma - alpha * h * h, "inj-vertex-value-not-position"),
        int_d(-2 * h, "inj-vertex-missing-two"),
    ]
    rule = _rule(alpha, h, gamma)
    if right:
        text = (
            f"The function f(x) = {rule} is defined for all real x ≥ a. "
            "Find the smallest value of a for which f is one-to-one."
        )
        extra = f"f is increasing for x ≥ {h}, so the smallest a is {h}."
    else:
        text = (
            f"The function f(x) = {rule} is defined for all real x ≤ a. "
            "Find the largest value of a for which f is one-to-one."
        )
        extra = f"f is decreasing for x ≤ {h}, so the largest a is {h}."
    working = [
        f"f(x) = {_vertex(alpha, h, gamma - alpha * h * h)}, so the turning point is at x = {h}.",
        extra,
    ]
    return number_q(text, h, working, keep(str(h), cands, 4))


def _hard_range_minimum(rng):
    alpha, h, gamma = _quad(rng)
    _check_quad(alpha, h, gamma)
    m = gamma - alpha * h * h
    cands = [
        int_d(gamma, "range-min-intercept"),
        int_d(gamma + alpha * h * h, "range-min-sign"),
        int_d(h, "inj-vertex-value-not-position"),
        int_d(-m, "range-min-sign"),
    ]
    text = (
        f"The function f : ℝ → [m, ∞), f(x) = {_rule(alpha, h, gamma)}, is onto (every value in [m, ∞) is "
        "taken). Find the largest value of m for which this is true."
    )
    working = [
        f"Completing the square gives f(x) = {_vertex(alpha, h, m)}.",
        f"The least value of f is {m}, at x = {h}, so m = {m}.",
    ]
    return number_q(text, m, working, keep(str(m), cands, 4))


def _brute_units(n):
    return sum(1 for a in range(n) if gcd(a, n) == 1)


def _prime_factors(n):
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0:
                n //= p
        p += 1
    if n > 1:
        out.append(n)
    return out


def _hard_modular(rng):
    n = rng.choice([12, 15, 18, 20, 21, 24, 28, 30, 36])
    b = rng.randint(1, n - 1)
    value = _brute_units(n)
    # Independent route: the map is onto exactly when it hits every residue.
    check(
        value == sum(1 for a in range(n) if len({(a * x + b) % n for x in range(n)}) == n),
        "bijection count by enumeration differs",
    )
    primes = _prime_factors(n)
    cands = [
        int_d(n - n // primes[0], "unit-count-single-prime"),
        int_d(n - 1, "unit-count-nonzero"),
        int_d(n // 2, "unit-count-single-prime"),
        int_d(n - n // primes[-1], "unit-count-single-prime"),
    ]
    text = (
        f"Each integer a with 0 ≤ a ≤ {n - 1} defines a function f_a : ℤ_{n} → ℤ_{n} by "
        f"f_a(x) = ax + {b} (mod {n}). For how many values of a is f_a a bijection?"
    )
    working = [
        f"On a finite set one-to-one and onto are equivalent. f_a is one-to-one exactly when ax = ay (mod {n}) forces x = y, which holds when gcd(a, {n}) = 1.",
        f"The number of a from 0 to {n - 1} that are coprime to {n} is {value}.",
    ]
    return number_q(text, value, working, keep(str(value), cands, 4))


def _hard(rng):
    return rng.choice([_hard_injective_threshold, _hard_range_minimum, _hard_modular, _hard_injective_threshold])(rng)


@generator(
    TOPIC,
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraws
def u2_functions_injective_surjective(rng, tier="easy"):
    """Counting and deciding one-to-one, onto and bijective functions; the tiers are in the module docstring."""
    if tier == "easy":
        question = _easy(rng)
    elif tier == "medium":
        question = _medium(rng)
    else:
        check(tier == "hard", f"unknown tier {tier!r}")
        question = _hard(rng)
    # A stem that writes a set with braces ("{1, 2, ..., 6}") is plain prose to the prompt converter, which would
    # turn it into italic maths letters, so it is shown as plain text.
    if "{" in question["text"]:
        question["latex"] = "plain:" + question["text"]
    return question


# -- exam -----------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_u2_functions_injective_surjective(rng, tier="exam"):
    """One quadratic on its injective domain: where it starts, its range, an inverse value, a count of integers."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "an exam quadratic with non-zero answers"):
        alpha = rng.choice([1, 2, 3])
        h = rng.choice([2, 3, 4, 5, 6, -2, -3, -4])
        gamma = rng.randint(-8, 10)
        f = _check_quad(alpha, h, gamma)
        m = f(h)
        t = rng.randint(2, 4)
        y0 = f(h + t)
        k = rng.randint(2, 5)
        y1 = f(h + k)
        if 0 not in (m, h + t, y0, y1, gamma):
            break
    rule = _rule(alpha, h, gamma)

    # Independent checks on a grid: inverse is the right branch; count by listing the integers.
    check([x for x in range(h, h + 40) if f(x) == y0] == [h + t], "inverse value is not unique on the domain")
    listed = [x for x in range(h, h + 40) if f(x) <= y1]
    check(len(listed) == k + 1, "count of integers differs")
    check(isqrt((y1 - m) // alpha) == k and (y1 - m) % alpha == 0, "bound is not alpha k^2 above the minimum")

    ft_b = f"{alpha}*part_a**2 - {2 * alpha * h}*part_a + {gamma}".replace("+ -", "- ")
    a_cands = [
        int_d(-h, "inj-vertex-sign"),
        int_d(2 * h, "inj-vertex-missing-two"),
        int_d(m, "inj-vertex-value-not-position"),
    ]
    b_cands = [
        int_d(gamma, "range-min-intercept"),
        int_d(gamma + alpha * h * h, "range-min-sign"),
        int_d(h, "inj-vertex-value-not-position"),
    ]
    c_cands = [
        int_d(h - t, "inverse-wrong-branch"),
        int_d(t, "inverse-value-not-solved"),
        int_d(y0, "inverse-value-not-solved"),
        int_d(h + t * t, "inverse-forgot-square-root"),
    ]
    d_cands = [
        int_d(k, "count-fencepost"),
        int_d(2 * k + 1, "count-both-sides"),
        int_d(k + 2, "count-fencepost"),
    ]
    return dict(
        text=(
            f"Let f(x) = {rule}, defined for real x ≥ a, where a is the least value for which f is one-to-one. "
            "Its codomain is [m, ∞), where m is chosen so that f maps its domain onto [m, ∞)."
        ),
        parts=[
            number_part(
                "Find a.",
                h,
                2,
                [f"f(x) = {_vertex(alpha, h, m)}; f is increasing from the turning point x = {h}, so a = {h}."],
                keep(str(h), a_cands),
            ),
            number_part(
                "Find m.",
                m,
                2,
                [f"On the domain, the least value of f is f({h}) = {m}, so m = {m}."],
                keep(str(m), b_cands),
                follow_through=ft_b,
            ),
            number_part(
                f"The function f has an inverse on this domain. Find f⁻¹({y0}).",
                h + t,
                2,
                [f"{_vertex(alpha, h, m)} = {y0} gives ({_xh(h)})² = {(y0 - m) // alpha}, and x ≥ {h} so x = {h + t}."],
                keep(str(h + t), c_cands),
            ),
            number_part(
                f"How many integers x in the domain satisfy f(x) ≤ {y1}?",
                k + 1,
                3,
                [
                    f"f(x) ≤ {y1} means ({_xh(h)})² ≤ {(y1 - m) // alpha}, so {h} ≤ x ≤ {h + k}.",
                    f"The integers from {h} to {h + k} number {k + 1}.",
                ],
                keep(str(k + 1), d_cands),
                follow_through=f"floor(sqrt(({y1} - part_b)/{alpha})) + 1",
            ),
        ],
    )
