"""Generates `cp1_divisibility_proof` items: prove that an expression in n is always a multiple of a given number,
by induction.

Topic ``cp1_divisibility_proof`` (open-response proof, ``level=14``). Every item asks for

    f(n) is divisible by D for every positive integer n

and the chain a symbolic check can follow is the heart of the inductive step: start from f(k + 1) and finish at an
expression built from f(k) plus a multiple of D, so that divisibility of f(k + 1) is visible. The base case and the
closing sentence are named in the marking dict for whoever marks the rest.

Tiers: a^n - 1 and a^n - b^n; a cubic in n and a^n - b^n with a shifted exponent; the textbook shapes such as
3^(2n + 4) - 2^(2n); and two-term mixes that need a second observation to finish.

The generator never runs a full CAS chain check. Every claim is confirmed by direct arithmetic for n = 1..12 and every
reference chain numerically at fixed irregular points, with the `math` module only.
"""

from __future__ import annotations

import functools

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, T


def _holds(f, divisor: int) -> bool:
    return all(f(n) % divisor == 0 for n in range(1, 13))


def _family(key, tier, f_n: str, f_k: str, f_k1: str, divisor: int, py, ref_mid: list[str], target: str, note=""):
    """`py` is a Python callable n -> int for the arithmetic check; `ref_mid` the lines between f(k + 1) and the target."""
    check(_holds(py, divisor), f"{key}: {f_n} is not always divisible by {divisor}")
    lead = [
        T("Prove by induction that "),
        M(f_n),
        T(f" is divisible by {divisor} for every positive integer n. Check the base case, then in the inductive step "),
        T("assume that "),
        M(f_k),
        T(f" is divisible by {divisor} and rewrite "),
        M(f_k1),
        T(" in terms of it." + note),
    ]
    return dict(
        key=key,
        tier=tier,
        kind="induction",
        symbols={"k": "integer"},
        ref=[f_k1, *ref_mid, target],
        lead=lead,
        jump_ok=len(ref_mid) >= 2,
    )


def _power_minus_one(a: int) -> dict:
    d = a - 1
    return _family(
        f"pow_minus_one_{a}",
        "easy",
        f"{a}**n - 1",
        f"{a}**k - 1",
        f"{a}**(k+1) - 1",
        d,
        lambda n: a**n - 1,
        [f"{a}*{a}**k - 1"],
        f"{a}*({a}**k - 1) + {d}",
    )


def _power_difference(a: int, b: int) -> dict:
    d = a - b
    return _family(
        f"pow_diff_{a}_{b}",
        "medium",
        f"{a}**n - {b}**n",
        f"{a}**k - {b}**k",
        f"{a}**(k+1) - {b}**(k+1)",
        d,
        lambda n: a**n - b**n,
        [f"{a}*{a}**k - {b}*{b}**k", f"{a}*({a}**k - {b}**k) + {a}*{b}**k - {b}*{b}**k"],
        f"{a}*({a}**k - {b}**k) + {d}*{b}**k",
    )


def _shifted(a: int, b: int, p: int, q: int, tier: str) -> dict:
    d = a**p - b**p
    f_n = f"{a}**({p}*n+{q}) - {b}**({p}*n)"
    f_k = f"{a}**({p}*k+{q}) - {b}**({p}*k)"
    return _family(
        f"shifted_{a}_{b}_{p}_{q}",
        tier,
        f_n,
        f_k,
        f"{a}**({p}*(k+1)+{q}) - {b}**({p}*(k+1))",
        d,
        lambda n: a ** (p * n + q) - b ** (p * n),
        [
            f"{a**p}*{a}**({p}*k+{q}) - {b**p}*{b}**({p}*k)",
            f"{a**p}*({a}**({p}*k+{q}) - {b}**({p}*k)) + {a**p}*{b}**({p}*k) - {b**p}*{b}**({p}*k)",
        ],
        f"{a**p}*({a}**({p}*k+{q}) - {b}**({p}*k)) + {d}*{b}**({p}*k)",
    )


def _mixed_powers(a: int, b: int) -> dict:
    d = b * b - a
    return _family(
        f"mixed_{a}_{b}",
        "exam",
        f"{a}**(n+1) + {b}**(2*n-1)",
        f"{a}**(k+1) + {b}**(2*k-1)",
        f"{a}**(k+2) + {b}**(2*k+1)",
        d,
        lambda n: a ** (n + 1) + b ** (2 * n - 1),
        [
            f"{a}*{a}**(k+1) + {b * b}*{b}**(2*k-1)",
            f"{a}*({a}**(k+1) + {b}**(2*k-1)) + {b * b}*{b}**(2*k-1) - {a}*{b}**(2*k-1)",
        ],
        f"{a}*({a}**(k+1) + {b}**(2*k-1)) + {d}*{b}**(2*k-1)",
    )


def _build_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": [], "exam": []}
    for a in (3, 4, 5, 6, 7, 8, 9, 10):
        pools["easy"].append(_power_minus_one(a))
    for a, b in ((5, 2), (7, 2), (7, 4), (8, 3), (9, 4), (10, 3), (11, 4), (9, 2)):
        pools["medium"].append(_power_difference(a, b))
    pools["medium"].append(
        _family(
            "cube_minus_n",
            "medium",
            "n**3 - n",
            "k**3 - k",
            "(k+1)**3 - (k+1)",
            6,
            lambda n: n**3 - n,
            ["k**3 + 3*k**2 + 3*k + 1 - k - 1"],
            "(k**3 - k) + 3*k*(k+1)",
            " Use that k(k + 1) is even.",
        )
    )
    pools["medium"].append(
        _family(
            "cube_plus_2n",
            "medium",
            "n**3 + 2*n",
            "k**3 + 2*k",
            "(k+1)**3 + 2*(k+1)",
            3,
            lambda n: n**3 + 2 * n,
            ["k**3 + 3*k**2 + 3*k + 1 + 2*k + 2"],
            "(k**3 + 2*k) + 3*(k**2 + k + 1)",
        )
    )
    for a, b, p, q in (
        (4, 3, 2, 3),
        (4, 3, 2, 6),
        (5, 3, 2, 4),
        (5, 4, 2, 6),
        (5, 2, 2, 6),
        (4, 2, 2, 3),
        (6, 4, 2, 5),
        (5, 2, 1, 2),
        (5, 2, 1, 4),
        (7, 2, 1, 4),
        (8, 3, 1, 4),
    ):
        pools["hard"].append(_shifted(a, b, p, q, "hard"))
    pools["hard"].append(
        _family(
            "cube_plus_11n",
            "hard",
            "n**3 + 11*n",
            "k**3 + 11*k",
            "(k+1)**3 + 11*(k+1)",
            6,
            lambda n: n**3 + 11 * n,
            ["k**3 + 3*k**2 + 3*k + 1 + 11*k + 11"],
            "(k**3 + 11*k) + 3*k*(k+1) + 12",
            " Use that k(k + 1) is even.",
        )
    )
    pools["exam"].append(_shifted(3, 2, 2, 4, "exam"))
    for a, b in ((2, 3), (5, 3), (3, 4), (4, 5), (6, 4), (11, 12)):
        pools["exam"].append(_mixed_powers(a, b))
    pools["exam"].append(
        _family(
            "fifth_minus_n",
            "exam",
            "n**5 - n",
            "k**5 - k",
            "(k+1)**5 - (k+1)",
            5,
            lambda n: n**5 - n,
            ["k**5 + 5*k**4 + 10*k**3 + 10*k**2 + 5*k + 1 - k - 1"],
            "(k**5 - k) + 5*(k**4 + 2*k**3 + 2*k**2 + k)",
        )
    )
    pools["exam"].append(
        _family(
            "n_times_square_plus_5",
            "exam",
            "n*(n**2 + 5)",
            "k*(k**2 + 5)",
            "(k+1)*((k+1)**2 + 5)",
            6,
            lambda n: n * (n * n + 5),
            ["k**3 + 3*k**2 + 8*k + 6"],
            "k*(k**2 + 5) + 3*k*(k+1) + 6",
            " Use that k(k + 1) is even.",
        )
    )
    pools["exam"].append(
        _family(
            "three_cubes",
            "exam",
            "n**3 + (n+1)**3 + (n+2)**3",
            "k**3 + (k+1)**3 + (k+2)**3",
            "(k+1)**3 + (k+2)**3 + (k+3)**3",
            9,
            lambda n: n**3 + (n + 1) ** 3 + (n + 2) ** 3,
            ["(k**3 + (k+1)**3 + (k+2)**3) - k**3 + (k+3)**3"],
            "(k**3 + (k+1)**3 + (k+2)**3) + 9*(k**2 + 3*k + 3)",
        )
    )
    pools["exam"].append(
        _family(
            "seven_five",
            "exam",
            "2*7**n + 3*5**n - 5",
            "2*7**k + 3*5**k - 5",
            "2*7**(k+1) + 3*5**(k+1) - 5",
            24,
            lambda n: 2 * 7**n + 3 * 5**n - 5,
            ["14*7**k + 15*5**k - 5"],
            "7*(2*7**k + 3*5**k - 5) - 6*(5**k - 5)",
            " The leftover 6(5^k - 5) needs one more observation.",
        )
    )
    return pools


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_divisibility_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_divisibility_proof(rng, variant="easy"):
    """`senior-prove-divisibility-results-such-as-3-2n-4`. Every question carries a marking dict (`Question.answer`):
    the inductive step as a chain of equal expressions, the reference chain, the key steps and the traps."""
    if variant not in pools():
        raise ValueError(f"unknown variant: {variant}")
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
