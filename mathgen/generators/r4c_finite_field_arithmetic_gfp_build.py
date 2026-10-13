"""Generates `r4c_finite_field_arithmetic_gfp` questions (easy, medium, hard) and the exam chain
`exd_x_r4c_finite_field_arithmetic_gfp`.

``uc2-finite-field-arithmetic-gfp`` (``abstract-algebra``, answer ``number``): arithmetic in the field GF(p) = Z_p for a
prime p. Every answer is the residue in {0, 1, ..., p - 1}; the stem says so. The primes are large enough (31 to 97)
that inverses cannot be found by guessing.

* easy: solve ax + b = c in GF(p), which needs the inverse of a.
* medium: evaluate a/b + c/d - e in GF(p), which needs two inverses.
* hard: solve a quadratic ax^2 + bx + c = 0 with a not 1 and give the smaller of its two roots (as residues), which needs
  the discriminant, a square root modulo p and the inverse of 2a.
* exam: for f(x) = ax^2 + bx + c over GF(p): (a) the inverse u of a, (b) the discriminant, (c) the integer v in the Bezout
  identity au + pv = 1 (follows from (a)), (d) the set of roots.

The tests redo every answer by searching all p residues.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._r4c_kit import number_part, number_question, set_part

_PRIMES = (31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)


def _quad(a, b, c):
    """ax² + bx + c written without a coefficient of 1 on x."""
    return f"{a}x² + {'' if b == 1 else b}x + {c}"


def _inv(a, p):
    return pow(a, -1, p)


def _easy(rng):
    p = rng.choice(_PRIMES)
    a = rng.randrange(3, p - 1)
    x = rng.randrange(2, p)
    b = rng.randrange(1, p)
    c = (a * x + b) % p
    key = x
    check((a * key + b) % p == c, "the solution does not satisfy the equation")
    inv_a = _inv(a, p)
    stem = rng.choice(
        [
            f"In the field GF({p}) = Z_{p}, solve {a}x + {b} = {c}. Give x as a residue from 0 to {p - 1}.",
            f"Find the residue x from 0 to {p - 1} with {a}x + {b} ≡ {c} (mod {p}).",
            f"Over the finite field of {p} elements, the equation {a}x + {b} = {c} has exactly one solution. "
            f"What is it, as a residue from 0 to {p - 1}?",
        ]
    )
    return number_question(
        stem,
        key,
        [
            f"Subtract {b}: {a}x = {(c - b) % p}.",
            f"The inverse of {a} modulo {p} is {inv_a}, so x = {(c - b) % p} · {inv_a} ≡ {key}.",
        ],
        [
            ((c - b) * a % p, "field-multiplies-by-a-instead-of-its-inverse"),
            ((c + b) * inv_a % p, "field-moves-constant-with-wrong-sign"),
            (
                (c - b) // a if (c - b) % a == 0 and (c - b) // a != key else (c - b) % p // a,
                "field-integer-division-for-inverse",
            ),
            ((-key) % p, "field-negative-of-the-solution"),
            ((c - b) * (p - inv_a) % p, "field-inverse-sign-slip"),
            ((key + 1) % p, None),
        ],
        lo=0,
        hi=p - 1,
    )


def _medium(rng):
    p = rng.choice(_PRIMES)
    a, c, e = rng.randrange(1, p), rng.randrange(1, p), rng.randrange(1, p)
    b, d = rng.randrange(2, p), rng.randrange(2, p)
    key = (a * _inv(b, p) + c * _inv(d, p) - e) % p
    return _medium_question(rng, p, a, b, c, d, e, key)


def _medium_question(rng, p, a, b, c, d, e, key):
    stem = rng.choice(
        [
            f"In the field GF({p}) = Z_{p}, evaluate {a}/{b} + {c}/{d} - {e}. "
            f"Give the value as a residue from 0 to {p - 1}.",
            f"Compute {a}/{b} + {c}/{d} - {e} in GF({p}) and write the result as a residue from 0 to {p - 1}.",
            f"Working modulo {p}, where every non-zero residue has an inverse, what is {a}/{b} + {c}/{d} - {e}? "
            f"Give a residue from 0 to {p - 1}.",
        ]
    )
    ib, id_ = _inv(b, p), _inv(d, p)
    return number_question(
        stem,
        key,
        [
            f"{a}/{b} means {a} · {b}⁻¹ with {b}⁻¹ ≡ {ib} (mod {p}); likewise {c}/{d} = {c} · {id_}.",
            f"({a} · {ib} + {c} · {id_} - {e}) mod {p} = {key}.",
        ],
        [
            (
                ((a + c) * _inv((b + d) % p, p) - e) % p if (b + d) % p else None,
                "field-adds-numerators-and-denominators",
            ),
            ((a * (p - ib) + c * (p - id_) - e) % p, "field-inverse-sign-slip"),
            ((a * ib + c * id_ + e) % p, "field-subtraction-sign-slip"),
            ((a * b + c * d - e) % p, "field-multiplies-by-denominator-instead-of-inverse"),
            ((a * ib - c * id_ - e) % p, None),
            ((key + 1) % p, None),
        ],
        lo=0,
        hi=p - 1,
    )


def _quadratic(rng, p):
    """(a, b, c, r1, r2) with a x^2 + b x + c = a (x - r1)(x - r2) over GF(p), r1 < r2."""
    for _ in attempts(500, "two distinct roots and a leading coefficient above 1"):
        a = rng.randrange(2, p)
        r1, r2 = sorted(rng.sample(range(1, p), 2))
        b = (-a * (r1 + r2)) % p
        c = (a * r1 * r2) % p
        if b and c:
            return a, b, c, r1, r2
    raise AssertionError  # pragma: no cover


def _sqrt_all(n, p):
    return [s for s in range(p) if s * s % p == n % p]


def _hard(rng):
    p = rng.choice(_PRIMES)
    a, b, c, r1, r2 = _quadratic(rng, p)
    disc = (b * b - 4 * a * c) % p
    s = _sqrt_all(disc, p)
    check(len(s) == 2, "the discriminant should have two square roots")
    check(all((a * r * r + b * r + c) % p == 0 for r in (r1, r2)), "a root does not satisfy the equation")
    inv2a = _inv(2 * a % p, p)
    other = (p - r1) % p
    stem = rng.choice(
        [
            f"In the field GF({p}) = Z_{p}, the equation {_quad(a, b, c)} = 0 has two distinct roots. "
            f"Find the smaller of the two, as a residue from 0 to {p - 1}.",
            f"The quadratic {_quad(a, b, c)} has two roots in GF({p}). Give the smaller root, as a residue from "
            f"0 to {p - 1}.",
            f"Solve {_quad(a, b, c)} = 0 over Z_{p}. It has two solutions among the residues 0 to {p - 1}; which is the smaller one?",
        ]
    )
    return number_question(
        stem,
        r1,
        [
            f"The discriminant is {b}² - 4·{a}·{c} ≡ {disc} (mod {p}), whose square roots are {s[0]} and {s[1]}.",
            f"The roots are (-{b} ± s) · (2·{a})⁻¹ with (2·{a})⁻¹ ≡ {inv2a}: they are {r1} and {r2}.",
        ],
        [
            (r2, "quadratic-larger-root-given"),
            ((-b * inv2a) % p, "quadratic-formula-without-the-square-root"),
            (min(((-b + t) * _inv(2, p)) % p for t in s), "quadratic-formula-ignores-leading-coefficient"),
            (min(((b + t) * inv2a) % p for t in s), "quadratic-formula-sign-slip-on-b"),
            (min(((-b + t) * _inv(a, p)) % p for t in s), "quadratic-formula-divides-by-a-not-2a"),
            (other, None),
        ],
        lo=0,
        hi=p - 1,
    )


@generator(
    "r4c_finite_field_arithmetic_gfp",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4c_finite_field_arithmetic_gfp(rng, tier="easy"):
    """Arithmetic in GF(p); the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r4c_finite_field_arithmetic_gfp", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r4c_finite_field_arithmetic_gfp(rng, tier="exam"):
    """Exam chain: inverse of the leading coefficient, discriminant, sum of the roots, the roots."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p = rng.choice(_PRIMES)
    a, b, c, r1, r2 = _quadratic(rng, p)
    inv_a = _inv(a, p)
    disc = (b * b - 4 * a * c) % p
    bezout = (1 - a * inv_a) // p
    check(a * inv_a + p * bezout == 1, "the Bezout identity fails")
    check((a * r1 * r1 + b * r1 + c) % p == 0 and (a * r2 * r2 + b * r2 + c) % p == 0, "a root fails the equation")
    inv2a = _inv(2 * a % p, p)
    s = _sqrt_all(disc, p)
    return dict(
        text=rng.choice(
            [
                f"Work in the field GF({p}) = Z_{p}, and let f(x) = {_quad(a, b, c)}. "
                f"Unless a part says otherwise, give residues from 0 to {p - 1}.",
                f"Let f(x) = {_quad(a, b, c)} be a polynomial over the field GF({p}) = Z_{p}. "
                f"Unless a part says otherwise, answers are residues from 0 to {p - 1}.",
                f"All arithmetic is in GF({p}), the integers modulo the prime {p}, and f(x) = {_quad(a, b, c)}. "
                f"Unless a part says otherwise, write residues from 0 to {p - 1}.",
            ]
        ),
        parts=[
            number_part(
                f"Find the inverse of {a} in GF({p}).",
                inv_a,
                2,
                [f"Extended Euclid on {a} and {p} gives {a} · {inv_a} ≡ 1 (mod {p})."],
                [
                    ((p - inv_a) % p, "field-inverse-sign-slip"),
                    (p // a, "field-integer-division-for-inverse"),
                    ((p - a) % p, None),
                    ((inv_a + 1) % p, None),
                ],
                sensible_range=(0, p),
            ),
            number_part(
                "Find the discriminant b² - 4ac of f.",
                disc,
                1,
                [f"{b}² - 4·{a}·{c} = {b * b - 4 * a * c}, which is {disc} modulo {p}."],
                [
                    ((b * b + 4 * a * c) % p, "field-discriminant-sign-slip"),
                    ((b * b - 2 * a * c) % p, "field-discriminant-wrong-multiplier"),
                    ((b * b) % p, None),
                    ((disc + 1) % p, None),
                ],
                sensible_range=(0, p),
            ),
            number_part(
                f"The inverse u of {a} satisfies {a}u + {p}v = 1 for an integer v (the Bézout identity), taking u as your answer to the first part. Find the integer v.",
                bezout,
                1,
                [f"{a} · {inv_a} = {a * inv_a} = 1 + {-bezout} · {p}, so v = {bezout}."],
                [
                    (-bezout, "bezout-cofactor-sign-slip"),
                    (bezout - 1, None),
                    (bezout + 1, None),
                    (inv_a, "bezout-cofactor-is-the-inverse"),
                ],
                follow_through=f"(1 - {a} * part_a) / {p}",
                sensible_range=(-1000, 1),
            ),
            set_part(
                "Find all roots of f in GF(%d), as a set of residues, for example the set with elements 3 and 17." % p,
                [r1, r2],
                3,
                [
                    f"The discriminant {disc} has square roots {s[0]} and {s[1]}.",
                    f"x = (-{b} ± s)(2·{a})⁻¹ with (2·{a})⁻¹ ≡ {inv2a}, which gives {r1} and {r2}.",
                ],
                [
                    ([(-b * inv2a) % p], "quadratic-formula-without-the-square-root"),
                    (sorted(((b + t) * inv2a) % p for t in s), "quadratic-formula-sign-slip-on-b"),
                    (sorted(((-b + t) * _inv(2, p)) % p for t in s), "quadratic-formula-ignores-leading-coefficient"),
                    (sorted(((-b + t) * _inv(a, p)) % p for t in s), "quadratic-formula-divides-by-a-not-2a"),
                ],
            ),
        ],
    )
