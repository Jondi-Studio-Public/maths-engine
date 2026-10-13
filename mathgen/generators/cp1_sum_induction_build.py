"""Generates `cp1_sum_induction_proof` items: prove a closed form for a sum by induction.

Topic ``cp1_sum_induction_proof`` (open-response proof, ``level=14``). Every item states a result of the form

    first term + second term + ... + last term = closed form        for every positive integer n

and asks for the inductive step as a chain of equal expressions: start from the assumed closed form for n = k plus
the (k + 1)th term, finish at the closed form for n = k + 1. The base case and the closing sentence of an
induction are not a chain of expressions; they are named in the marking dict for whoever marks the rest.

Tiers climb from sums of squares and odd numbers (a polynomial closed form) through cubes and products of
consecutive integers, to telescoping fractions and factorial sums, to sums with a power of 2 or an alternating sign.

The generator never runs a full CAS chain check. Every closed form is confirmed by direct summation for n = 1..12
and every reference chain numerically at fixed irregular points, with the `math` module only.
"""

from __future__ import annotations

import functools

import re

import sympy as sp

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, T

_n, _k, _j = sp.symbols("n k j")
_BASE_N = range(1, 13)


def _s(expr) -> str:
    return str(expr)


def _family(
    key,
    tier,
    term,
    closed,
    pieces,
    last,
    *,
    target=None,
    lead_note="",
    sym_kind="integer",
    jump_ok=False,
    extra_symbols=None,
    domain="",
    next_term=None,
    target_text=None,
):
    """One sum family. `term` is the general term in j, `closed` the closed form in n (both sympy);
    `pieces` the first two displayed terms and `last` the displayed last term (sympy-style strings in n)."""
    # The claim really holds: direct summation for n = 1..12.
    for count in _BASE_N:
        total = sum(term.subs(_j, v) for v in range(1, count + 1))
        gap = sp.cancel(sp.together(total - closed.subs(_n, count)))
        if gap != 0:
            gap = sp.simplify(gap)
        check(gap == 0, f"{key}: closed form is wrong at n = {count}")
    closed_k = sp.factor(closed.subs(_n, _k))
    closed_k1 = sp.factor(closed.subs(_n, _k + 1)) if target is None else sp.sympify(target)
    if next_term is None:
        next_term = re.sub(r"\bn\b", "(k+1)", last)
        if not (next_term.startswith("(") and next_term.endswith(")")) and re.search(r"[+-]", next_term):
            next_term = f"({next_term})"
    start = f"{_s(closed_k)} + {next_term}"
    ref = [start]
    numer, denom = sp.fraction(sp.together(sp.sympify(start)))
    middle = f"({_s(sp.expand(numer))})/({_s(denom)})" if denom != 1 else _s(sp.expand(numer))
    final = target_text if target_text is not None else _s(closed_k1)
    if middle.replace(" ", "") not in (start.replace(" ", ""), final.replace(" ", "")):
        ref.append(middle)
    ref.append(final)
    claim_lhs = []
    claim_lhs += [T(""), M(pieces[0])]
    for p in pieces[1:]:
        claim_lhs += [T(" - ") if p.startswith("-") else T(" + "), M(p.lstrip("-"))]
    claim_lhs += [T(" + ... + "), M(last), T(" = "), M(_s(sp.factor(closed)) if target is None else _s(closed))]
    lead = (
        [T("Prove by induction that ")]
        + claim_lhs
        + [
            T(
                " for every positive integer n. Check the base case, then in the inductive step assume the result "
                "for n = k and add the (k + 1)th term."
            )
            + T(lead_note)
        ]
    )
    return dict(
        key=key,
        tier=tier,
        kind="induction",
        symbols={"k": sym_kind, **(extra_symbols or {})},
        ref=ref,
        lead=lead,
        domain=domain,
        jump_ok=jump_ok,
    )


def _build_pools() -> dict[str, list[dict]]:
    n, j = _n, _j
    pools: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": [], "exam": []}
    add = lambda tier, fam: pools[tier].append(fam)  # noqa: E731

    for c in (1, 2, 3, 4, 5):
        coef = "" if c == 1 else f"{c}*"
        add(
            "easy",
            _family(
                f"squares_{c}",
                "easy",
                c * j**2,
                c * n * (n + 1) * (2 * n + 1) / 6,
                (f"{coef}1**2", f"{coef}2**2"),
                f"{coef}n**2",
            ),
        )
    add("easy", _family("odd", "easy", 2 * j - 1, n**2, ("1", "3"), "(2*n-1)"))
    add("easy", _family("consecutive_pairs", "easy", j * (j + 1), n * (n + 1) * (n + 2) / 3, ("1*2", "2*3"), "n*(n+1)"))

    for c in (1, 2, 3, 4):
        coef = "" if c == 1 else f"{c}*"
        add(
            "medium",
            _family(
                f"cubes_{c}",
                "medium",
                c * j**3,
                c * n**2 * (n + 1) ** 2 / 4,
                (f"{coef}1**3", f"{coef}2**3"),
                f"{coef}n**3",
            ),
        )
    add(
        "medium",
        _family(
            "triples",
            "medium",
            j * (j + 1) * (j + 2),
            n * (n + 1) * (n + 2) * (n + 3) / 4,
            ("1*2*3", "2*3*4"),
            "n*(n+1)*(n+2)",
        ),
    )
    add(
        "medium",
        _family(
            "odd_squares", "medium", (2 * j - 1) ** 2, n * (2 * n - 1) * (2 * n + 1) / 3, ("1**2", "3**2"), "(2*n-1)**2"
        ),
    )
    add("medium", _family("gap_two", "medium", j * (j + 2), n * (n + 1) * (2 * n + 7) / 6, ("1*3", "2*4"), "n*(n+2)"))
    add(
        "medium",
        _family(
            "linear_quadratic", "medium", j * (2 * j + 1), n * (n + 1) * (4 * n + 5) / 6, ("1*3", "2*5"), "n*(2*n+1)"
        ),
    )
    add("medium", _family("three_minus_one", "medium", j * (3 * j - 1), n**2 * (n + 1), ("1*2", "2*5"), "n*(3*n-1)"))
    add("medium", _family("hex_numbers", "medium", 3 * j**2 - 3 * j + 1, n**3, ("1", "7"), "(3*n**2-3*n+1)"))

    add("hard", _family("telescope_1", "hard", 1 / (j * (j + 1)), n / (n + 1), ("1/(1*2)", "1/(2*3)"), "1/(n*(n+1))"))
    add(
        "hard",
        _family(
            "telescope_odd",
            "hard",
            1 / ((2 * j - 1) * (2 * j + 1)),
            n / (2 * n + 1),
            ("1/(1*3)", "1/(3*5)"),
            "1/((2*n-1)*(2*n+1))",
        ),
    )
    add(
        "hard",
        _family(
            "telescope_gap2",
            "hard",
            1 / (j * (j + 2)),
            n * (3 * n + 5) / (4 * (n + 1) * (n + 2)),
            ("1/(1*3)", "1/(2*4)"),
            "1/(n*(n+2))",
        ),
    )
    add(
        "hard",
        _family(
            "telescope_triple",
            "hard",
            1 / (j * (j + 1) * (j + 2)),
            n * (n + 3) / (4 * (n + 1) * (n + 2)),
            ("1/(1*2*3)", "1/(2*3*4)"),
            "1/(n*(n+1)*(n+2))",
        ),
    )
    add(
        "hard",
        _family(
            "factorial",
            "hard",
            j * sp.factorial(j),
            sp.factorial(n + 1) - 1,
            ("1*factorial(1)", "2*factorial(2)"),
            "n*factorial(n)",
            target="factorial(k+2) - 1",
        ),
    )
    add(
        "hard",
        _family(
            "doubling",
            "hard",
            j * 2 ** (j - 1),
            (n - 1) * 2**n + 1,
            ("1*1", "2*2"),
            "n*2**(n-1)",
            target="k*2**(k+1) + 1",
        ),
    )

    for c in (2, 3):
        add(
            "hard",
            _family(
                f"telescope_scaled_{c}",
                "hard",
                c / (j * (j + 1)),
                c * n / (n + 1),
                (f"{c}/(1*2)", f"{c}/(2*3)"),
                f"{c}/(n*(n+1))",
            ),
        )
    add(
        "hard",
        _family(
            "telescope_gap3",
            "hard",
            1 / (j * (j + 3)),
            n * (11 * n**2 + 48 * n + 49) / (18 * (n + 1) * (n + 2) * (n + 3)),
            ("1/(1*4)", "1/(2*5)"),
            "1/(n*(n+3))",
        ),
    )
    add(
        "hard",
        _family(
            "telescope_thirds",
            "hard",
            1 / ((3 * j - 2) * (3 * j + 1)),
            n / (3 * n + 1),
            ("1/(1*4)", "1/(4*7)"),
            "1/((3*n-2)*(3*n+1))",
        ),
    )
    add(
        "hard",
        _family(
            "triple_growth",
            "hard",
            j * 3 ** (j - 1),
            ((2 * n - 1) * 3**n + 1) / 4,
            ("1*1", "2*3"),
            "n*3**(n-1)",
            target="((2*k+1)*3**(k+1) + 1)/4",
        ),
    )

    add(
        "exam",
        _family(
            "doubling_shift",
            "exam",
            j * 2**j,
            (n - 1) * 2 ** (n + 1) + 2,
            ("1*2", "2*4"),
            "n*2**n",
            target="k*2**(k+2) + 2",
        ),
    )
    add(
        "exam",
        _family(
            "quads",
            "exam",
            j * (j + 1) * (j + 2) * (j + 3),
            n * (n + 1) * (n + 2) * (n + 3) * (n + 4) / 5,
            ("1*2*3*4", "2*3*4*5"),
            "n*(n+1)*(n+2)*(n+3)",
        ),
    )
    add(
        "exam",
        _family(
            "telescope_quad",
            "exam",
            1 / ((j + 1) * (j + 2)),
            n / (2 * (n + 2)),
            ("1/(2*3)", "1/(3*4)"),
            "1/((n+1)*(n+2))",
        ),
    )
    add(
        "exam",
        _family(
            "alternating",
            "exam",
            (-1) ** (j + 1) * j**2,
            (-1) ** (n + 1) * n * (n + 1) / 2,
            ("1**2", "-2**2"),
            "(-1)**(n+1)*n**2",
            target="(-1)**(k+2)*(k+1)*(k+2)/2",
        ),
    )
    add(
        "exam",
        _family(
            "weighted_squares",
            "exam",
            j * (j + 1) ** 2,
            n * (n + 1) * (n + 2) * (3 * n + 5) / 12,
            ("1*2**2", "2*3**2"),
            "n*(n+1)**2",
        ),
    )
    add(
        "exam",
        _family(
            "fourth_powers",
            "exam",
            j**4,
            n * (n + 1) * (2 * n + 1) * (3 * n**2 + 3 * n - 1) / 30,
            ("1**4", "2**4"),
            "n**4",
        ),
    )
    add("exam", _family("odd_cubes", "exam", (2 * j - 1) ** 3, n**2 * (2 * n**2 - 1), ("1**3", "3**3"), "(2*n-1)**3"))
    add(
        "exam",
        _family(
            "alternating_odd",
            "exam",
            (-1) ** (j + 1) * (2 * j - 1),
            (-1) ** (n + 1) * n,
            ("1", "-3"),
            "(-1)**(n+1)*(2*n-1)",
            target="(-1)**(k+2)*(k+1)",
        ),
    )
    add(
        "exam",
        _family(
            "telescope_scaled_triple",
            "exam",
            4 / (j * (j + 1) * (j + 2)),
            n * (n + 3) / ((n + 1) * (n + 2)),
            ("4/(1*2*3)", "4/(2*3*4)"),
            "4/(n*(n+1)*(n+2))",
        ),
    )
    add(
        "exam",
        _family(
            "pairs_times_odd",
            "exam",
            j * (j + 1) * (2 * j + 1),
            n * (n + 1) ** 2 * (n + 2) / 2,
            ("1*2*3", "2*3*5"),
            "n*(n+1)*(2*n+1)",
        ),
    )
    add(
        "exam",
        _family(
            "square_times_next",
            "exam",
            j**2 * (j + 1),
            n * (n + 1) * (n + 2) * (3 * n + 1) / 12,
            ("1**2*2", "2**2*3"),
            "n**2*(n+1)",
        ),
    )
    return pools


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_sum_induction_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_sum_induction_proof(rng, variant="easy"):
    """`senior-prove-results-for-sums-such-as-1-4`. Every question carries a marking dict (`Question.answer`):
    the inductive step as a chain of equal expressions, the reference chain, the key steps and the traps."""
    if variant not in pools():
        raise ValueError(f"unknown variant: {variant}")
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
