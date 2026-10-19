"""Generates `r4c_irreducibility_test_over_zp` questions (easy, medium, hard) and the exam chain
`exd_x_r4c_irreducibility_test_over_zp`.

``uc2-irreducibility-test-over-zp`` (``abstract-algebra``, answer ``ordered_list``): factor a polynomial over Z_p into
irreducibles and report the degrees of the factors, smallest first and repeated for a repeated factor, written
(d1, d2, ...). A polynomial of degree 2 or 3 is irreducible exactly when it has no root; from degree 4 a root test is not
enough (x^4 + x^2 + 1 over Z_2 has no root and is a square), so the factors of degree 2 and above have to be searched
for as well.

* easy: degree 3 or 4 over Z_3, Z_5 or Z_7, with at least one linear factor (found by testing the p residues).
* medium: degree 4 or 5 over Z_2, Z_3 or Z_5, any factor pattern.
* hard: degree 6 over Z_2 or Z_3, any factor pattern (no roots does not mean irreducible).
* exam: a degree 5 polynomial with a linear factor over Z_3 or Z_5: (a) how many linear factors (with multiplicity),
  (b) the degree of what is left (follows from (a)), (c) the degrees of all the factors, (d) the number of distinct irreducible
  factors, which is the number of maximal ideals of Z_p[x] / ⟨f⟩.

Each polynomial is built as a product of irreducibles listed by trial division; the tests factor it again with SymPy.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._r4c_kit import irreducibles, number_part, ordered_part, ordered_question, peval, pmul, poly_text

_EASY = {3: [(1, 2), (1, 1, 1)], 4: [(1, 3), (1, 1, 2), (1, 1, 1, 1)]}
_MEDIUM = {4: [(2, 2), (4,), (1, 3), (1, 1, 2)], 5: [(5,), (2, 3), (1, 4), (1, 2, 2)]}
_HARD = [(6,), (2, 4), (3, 3), (2, 2, 2), (1, 5), (1, 2, 3), (1, 1, 2, 2), (1, 1, 1, 3)]


def _build(rng, p, pattern):
    """(f, the chosen irreducible factors) for a pattern of degrees."""
    factors = [rng.choice(irreducibles(d, p)) for d in pattern]
    f = (1,)
    for g in factors:
        f = pmul(f, g, p)
    check(len(f) - 1 == sum(pattern), "the product has the wrong degree")
    return f, factors


def _tuple_text(values):
    return "(" + ", ".join(map(str, values)) + ")"


def _candidates(pattern, p, f):
    """Wrong degree lists, each a named error."""
    d = sum(pattern)
    roots = sum(1 for x in range(p) if peval(f, x, p) == 0)
    out = [
        ([d], "irreducible-assumed-without-checking") if len(pattern) > 1 else (None, None),
        ([d], "no-root-taken-as-irreducible") if roots == 0 and len(pattern) > 1 else (None, None),
        ([1, d - 1], "factor-search-stops-after-one-linear-factor")
        if pattern != (1, d - 1) and d > 2
        else (None, None),
        (sorted(set(pattern)), "repeated-factor-counted-once") if len(set(pattern)) < len(pattern) else (None, None),
        ([1] * d, "every-factor-taken-as-linear") if pattern != (1,) * d else (None, None),
        (sorted(x for g in pattern for x in ([1, 1] if g == 2 else [g])), "quadratic-factor-split-into-linears")
        if 2 in pattern
        else (None, None),
        (
            list(reversed(pattern)) if tuple(reversed(pattern)) != tuple(pattern) else None,
            "factor-degrees-in-descending-order",
        ),
        (
            sorted([pattern[0] + pattern[1]] + list(pattern[2:])) if len(pattern) > 2 else None,
            "factors-merged-into-one",
        ),
        (sorted(list(pattern[:-1]) + [1, pattern[-1] - 1]) if pattern[-1] > 2 else None, None),
    ]
    return [(v, t) for v, t in out if v is not None]


def _question(rng, p, pattern, note):
    f, factors = _build(rng, p, pattern)
    stem = rng.choice(
        [
            f"Let f = {poly_text(f)} in Z_{p}[x], with coefficients taken modulo {p}. "
            f"Factor f into irreducible polynomials over Z_{p}. List the degrees of the irreducible factors in increasing "
            "order, repeating a degree once for each time a factor occurs, as (d1, d2, ...).",
            f"Over Z_{p}, the polynomial f = {poly_text(f)} is a product of irreducible polynomials. Give the degrees of the factors, smallest "
            "first, repeating a degree for a repeated factor, as (d1, d2, ...).",
            f"Factor f = {poly_text(f)} completely into irreducibles in Z_{p}[x] (coefficients modulo {p}). "
            "Report the degree of each factor, in increasing order and with repeats, as (d1, d2, ...).",
        ]
    )
    roots = [x for x in range(p) if peval(f, x, p) == 0]
    return (
        ordered_question(
            stem,
            list(pattern),
            [
                f"Substitute x = {', '.join(str(v) for v in range(p))}: f has {len(roots)} root{'s' if len(roots) != 1 else ''} in Z_{p}"
                + (f" ({', '.join(map(str, roots))})" if roots else "")
                + ", each giving a linear factor.",
                note,
                f"The irreducible factors are {', '.join(poly_text(g) for g in sorted(factors, key=lambda g: (len(g), g)))}, of degrees {_tuple_text(pattern)}.",
            ],
            _candidates(pattern, p, f),
        ),
        f,
        factors,
    )


def _easy(rng):
    p = rng.choice([3, 5, 7])
    degree = rng.choice([3, 4])
    pattern = tuple(sorted(rng.choice(_EASY[degree])))
    return _question(
        rng, p, pattern, "Divide out each linear factor, then test what is left for roots and for a quadratic factor."
    )[0]


def _medium(rng):
    p = rng.choice([2, 3, 5])
    degree = rng.choice([4, 5])
    pattern = tuple(sorted(rng.choice(_MEDIUM[degree])))
    return _question(
        rng,
        p,
        pattern,
        "A polynomial of degree 4 or 5 with no root can still have a quadratic factor, so test the irreducible quadratics too.",
    )[0]


def _hard(rng):
    p = rng.choice([2, 3])
    pattern = tuple(sorted(rng.choice(_HARD)))
    return _question(
        rng, p, pattern, "Having no root does not make f irreducible: look for quadratic and cubic factors as well."
    )[0]


@generator(
    "r4c_irreducibility_test_over_zp",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4c_irreducibility_test_over_zp(rng, tier="easy"):
    """Factorisations over Z_p; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r4c_irreducibility_test_over_zp", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r4c_irreducibility_test_over_zp(rng, tier="exam"):
    """Exam chain: linear factors, the rest, the full degree list and the number of distinct irreducible factors."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p = rng.choice([3, 5])
    pattern = tuple(sorted(rng.choice([(1, 4), (1, 1, 3), (1, 2, 2), (1, 1, 1, 2), (1, 1, 1, 1, 1)])))
    for _ in attempts(200, "factors whose distinct count the pattern fixes"):
        f, factors = _build(rng, p, pattern)
        break
    linear = pattern.count(1)
    rest = 5 - linear
    distinct = len(set(factors))
    return dict(
        text=rng.choice(
            [
                f"Let f = {poly_text(f)} in Z_{p}[x], with coefficients taken modulo {p}.",
                f"Consider the polynomial f = {poly_text(f)} over the field Z_{p}, with coefficients taken modulo {p}.",
                f"Over Z_{p} (coefficients modulo {p}), let f = {poly_text(f)}.",
            ]
        ),
        parts=[
            number_part(
                f"How many linear factors does f have over Z_{p}, counting a repeated factor once for each time it occurs?",
                linear,
                2,
                [
                    f"Test x = 0, 1, ..., {p - 1}, and divide out each root as often as it divides: {linear} linear factors in all."
                ],
                [
                    (sum(1 for x in range(p) if peval(f, x, p) == 0), "linear-factors-counted-without-multiplicity"),
                    (linear + 1, None),
                    (max(linear - 1, 0), None),
                    (5 - linear, None),
                ],
                sensible_range=(-1, 6),
            ),
            number_part(
                f"After the linear factors are divided out, what is the degree of what is left?",
                rest,
                1,
                [f"deg f - {linear} = 5 - {linear} = {rest}."],
                [
                    (linear, "factor-degree-left-is-the-linear-count"),
                    (rest + 1, None),
                    (max(rest - 1, 0), None),
                    (5, "factor-degree-left-is-the-whole-degree"),
                ],
                follow_through="5 - part_a",
                sensible_range=(-1, 6),
            ),
            ordered_part(
                f"List the degrees of all the irreducible factors of f over Z_{p} in increasing order, repeating a degree once for each time a factor occurs, as (d1, d2, ...).",
                list(pattern),
                3,
                [f"The factors have degrees {_tuple_text(pattern)}."],
                [
                    (sorted(set(pattern)), "repeated-factor-counted-once"),
                    ([5], "irreducible-assumed-without-checking"),
                    ([1] * 5, "every-factor-taken-as-linear"),
                    (sorted([1] + [4]) if pattern != (1, 4) else None, "factor-search-stops-after-one-linear-factor"),
                    (
                        sorted(x for g in pattern for x in ([1, 1] if g == 2 else [g])),
                        "quadratic-factor-split-into-linears",
                    ),
                ],
            ),
            number_part(
                f"How many maximal ideals does the ring Z_{p}[x] / ⟨f⟩ have?",
                distinct,
                2,
                [f"Its maximal ideals correspond to the distinct irreducible factors of f, and f has {distinct}."],
                [
                    (len(pattern), "maximal-ideals-counted-with-multiplicity"),
                    (1, "maximal-ideals-assumed-one"),
                    (distinct + 1, None),
                    (max(distinct - 1, 1), None),
                ],
                sensible_range=(0, 6),
            ),
        ],
    )
