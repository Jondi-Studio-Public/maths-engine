"""Generates `cp1_geometric_sum_proof` items: establish the sum of the first n terms of a geometric sequence,
S_n = t_1 (r^n - 1)/(r - 1), or use it.

Topic ``cp1_geometric_sum_proof`` (open-response proof, ``level=14``). PROVE items are chains of equal expressions:
the telescoping product that produces the formula, the inductive step for a stated sum, and consequences of the
formula. APPLY items are numeric chains that use the formula on a stated sequence.

Tiers: the product (r - 1)(1 + r + ... + r^(m-1)) and induction steps with numbers; induction and recurrences with a
general first term and ratio, and negative ratios; S_2n and S_n - S_(n-1); and S_3n, the gap between S_n and the
limit, and a sum of squares of the terms.

The generator never runs a full CAS chain check. Every claim is confirmed by direct summation, and every reference
chain numerically at fixed irregular points, with the `math` module only.
"""

from __future__ import annotations

import functools
from fractions import Fraction

import sympy as sp

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, T
from .cp1_sum_induction_build import _family as sum_family

_n, _j = sp.symbols("n j")
_AR = {"a": "real", "r": "real"}


def _identity(key, tier, ref, lead, *, symbols, domain, jump_ok=False):
    return dict(
        key=key,
        tier=tier,
        kind="identity",
        symbols=symbols,
        ref=ref,
        lead=lead,
        domain=domain,
        jump_ok=jump_ok,
        given_step="states the geometric sum S_n = t_1 + t_1 r + ... + t_1 r^(n-1) or the formula being used",
    )


def _telescope(m: int, t1: int) -> dict:
    series = "+".join(["1"] + [("r" if i == 1 else f"r**{i}") for i in range(1, m)])
    lead_coeff = "" if t1 == 1 else f"{t1}*"
    start = f"{lead_coeff}(r-1)*({series})"
    shifted = "+".join([("r" if i == 1 else f"r**{i}") for i in range(1, m + 1)])
    target = f"{t1}*(r**{m}-1)" if t1 != 1 else f"r**{m}-1"
    ref = [start, f"{t1}*(({shifted}) - ({series}))" if t1 != 1 else f"({shifted}) - ({series})", target]
    return _identity(
        f"telescope_{m}_{t1}",
        "easy",
        ref,
        [
            T(
                "Multiplying a geometric sum by (r - 1) makes most of it cancel, which is where the formula for S_n comes from. "
                "Show that "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"r": "real"},
        domain="Here r is a real number.",
        jump_ok=False,
    )


def _numeric_induction(t1: int, r: int, tier: str) -> dict:
    term = t1 * r ** (_j - 1)
    closed = t1 * (r**_n - 1) / (r - 1)
    pieces = (f"{t1}", f"{t1 * r}")
    base = f"({r})" if r < 0 else f"{r}"
    den = "" if r == 2 else f"/{r - 1}"
    return sum_family(
        f"induction_{t1}_{r}",
        tier,
        term,
        closed,
        pieces,
        f"{t1}*{base}**(n-1)",
        next_term=f"({t1}*{base}**k)",
        target_text=f"{t1}*({base}**(k+1)-1){den}",
    )


def _general_induction() -> dict:
    a, r = sp.symbols("a r")
    term = a * r ** (_j - 1)
    closed = a * (r**_n - 1) / (r - 1)
    return sum_family(
        "induction_general",
        "medium",
        term,
        closed,
        ("a", "a*r"),
        "a*r**(n-1)",
        extra_symbols=_AR,
        domain="Here r is a real number with r ≠ 1.",
        next_term="(a*r**k)",
        target_text="a*(r**(k+1)-1)/(r-1)",
    )


def _recurrence() -> dict:
    start = "a + r*(a*(r**n-1)/(r-1))"
    target = "a*(r**(n+1)-1)/(r-1)"
    return _identity(
        "recurrence",
        "medium",
        kit.auto_ref(start, target),
        [
            T(
                "Let S_n = a(r^n - 1)/(r - 1) be the sum of the first n terms of a geometric sequence with first term a and ratio r. "
                "Show that S_(n+1) = a + rS_n, that is "
            ),
            M(target),
            T(" = "),
            M(start),
            T(". Work from the right-hand side to the left-hand side."),
        ],
        symbols={"n": "integer", **_AR},
        domain="Here r ≠ 1 and n is a positive integer.",
        jump_ok=True,
    )


def _term_from_sums() -> dict:
    start = "a*(r**n-1)/(r-1) - a*(r**(n-1)-1)/(r-1)"
    target = "a*r**(n-1)"
    return _identity(
        "term_from_sums",
        "medium",
        kit.auto_ref(start, target),
        [
            T(
                "Let S_m = a(r^m - 1)/(r - 1). Show that S_n - S_(n-1) equals the n-th term of the geometric sequence, a r^(n-1), that is "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"n": "integer", **_AR},
        domain="Here r ≠ 1 and n is an integer with n ≥ 2.",
    )


def _double_sum() -> dict:
    start = "a*(r**(2*n)-1)/(r-1)"
    target = "a*(r**n-1)/(r-1)*(1+r**n)"
    return _identity(
        "double_sum",
        "hard",
        [start, "a*(r**n-1)*(r**n+1)/(r-1)", target],
        [
            T("Let S_m = a(r^m - 1)/(r - 1). Show that S_(2n) = S_n(1 + r^n), that is "),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"n": "integer", **_AR},
        domain="Here r ≠ 1 and n is a positive integer.",
    )


def _ratio_of_sums(b: int) -> dict:
    start = f"({b}**(2*n)-1)/({b}**n-1)"
    target = f"{b}**n+1"
    return _identity(
        f"ratio_of_sums_{b}",
        "hard",
        [start, f"({b}**n-1)*({b}**n+1)/({b}**n-1)", target],
        [
            T(
                f"For the geometric sequence with first term 1 and ratio {b}, S_m = ({b}^m - 1)/({b} - 1). Show that S_(2n)/S_n = "
            ),
            M(target),
            T(" by simplifying "),
            M(start),
            T("."),
        ],
        symbols={"n": "integer"},
        domain="Here n is a positive integer.",
    )


def _triple_sum() -> dict:
    start = "a*(r**(3*n)-1)/(r-1)"
    target = "a*(r**n-1)/(r-1)*(1+r**n+r**(2*n))"
    return _identity(
        "triple_sum",
        "exam",
        [start, "a*(r**n-1)*(r**(2*n)+r**n+1)/(r-1)", target],
        [
            T("Let S_m = a(r^m - 1)/(r - 1). Show that S_(3n) = S_n(1 + r^n + r^(2n)), that is "),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"n": "integer", **_AR},
        domain="Here r ≠ 1 and n is a positive integer.",
    )


def _gap_to_limit() -> dict:
    start = "a/(1-r) - a*(1-r**n)/(1-r)"
    target = "a*r**n/(1-r)"
    return _identity(
        "gap_to_limit",
        "exam",
        kit.auto_ref(start, target),
        [
            T(
                "The sum of the first n terms of a geometric sequence can be written S_n = a(1 - r^n)/(1 - r), and its limit as n grows "
                "(when |r| < 1) is a/(1 - r). Show that the gap between them is a r^n/(1 - r), that is "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"n": "integer", **_AR},
        domain="Here r ≠ 1 and n is a positive integer.",
    )


def _squares_induction() -> dict:
    a, r = sp.symbols("a r")
    term = a**2 * r ** (2 * _j - 2)
    closed = a**2 * (r ** (2 * _n) - 1) / (r**2 - 1)
    return sum_family(
        "squares_of_terms",
        "exam",
        term,
        closed,
        ("a**2", "(a*r)**2"),
        "(a*r**(n-1))**2",
        extra_symbols=_AR,
        domain="Here r ≠ 1 and r ≠ -1.",
        next_term="(a*r**k)**2",
        lead_note=" These are the squares of the terms of a geometric sequence with first term a and ratio r.",
    )


def _shifted_start() -> dict:
    start = "a*r*(r**n-1)/(r-1)"
    target = "a*(r**(n+1)-r)/(r-1)"
    return _identity(
        "shifted_start",
        "exam",
        [start, "(a*r*r**n-a*r)/(r-1)", target],
        [
            T(
                "The sum a r + a r^2 + ... + a r^n of n terms of a geometric sequence starting at a r is a r(r^n - 1)/(r - 1). Show that this equals "
            ),
            M(target),
            T(", that is "),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        symbols={"n": "integer", **_AR},
        domain="Here r ≠ 1 and n is a positive integer.",
    )


def _apply_sum(t1: int, r: int, n: int) -> dict:
    terms = [t1, t1 * r, t1 * r * r]
    total = t1 * (r**n - 1) // (r - 1)
    check(total == sum(t1 * r**i for i in range(n)), "apply_sum: the formula disagrees with direct addition")
    ref = [f"{t1}*({r}**{n}-1)/({r}-1)", f"{t1}*({r**n}-1)/({r}-1)", f"{t1}*{r**n - 1}/{r - 1}", f"{total}"]
    return dict(
        key=f"apply_sum_{t1}_{r}_{n}",
        tier="apply_easy",
        lead=[
            T(f"Use S_n = t_1(r^n - 1)/(r - 1) to find the sum of the first {n} terms of the sequence "),
            T(f"{terms[0]}, {terms[1]}, {terms[2]}, ..."),
            T("."),
        ],
        ref=ref,
        wrong_final=f"{total + 4}",
        answer_text=f"S_{n} = {total}",
        given_step="states the sum formula and substitutes t_1, r and n",
    )


def _apply_fractional(t1: int, num: int, den: int, n: int) -> dict:
    r = Fraction(num, den)
    terms = [Fraction(t1), t1 * r, t1 * r * r]
    total = t1 * (1 - r**n) / (1 - r)
    check(total == sum(t1 * r**i for i in range(n)), "apply_fractional: the formula disagrees with direct addition")
    rs = f"{num}/{den}"
    p = r**n
    ref = [
        f"{t1}*(1-({rs})**{n})/(1-{rs})",
        f"{t1}*(1-{p.numerator}/{p.denominator})/(1-{rs})",
        f"{t1}*({(1 - p).numerator}/{(1 - p).denominator})/({(1 - r).numerator}/{(1 - r).denominator})",
        f"{total.numerator}" if total.denominator == 1 else f"{total.numerator}/{total.denominator}",
    ]
    shown = ", ".join(str(t.numerator) if t.denominator == 1 else f"{t.numerator}/{t.denominator}" for t in terms)
    value = ref[-1]
    return dict(
        key=f"apply_fractional_{t1}_{num}_{den}_{n}",
        tier="apply_exam",
        lead=[
            T(
                f"Use S_n = t_1(1 - r^n)/(1 - r) to find the exact sum of the first {n} terms of the sequence {shown}, ..."
            ),
            T("."),
        ],
        ref=ref,
        wrong_final=f"{total.numerator + 1}/{total.denominator}",
        answer_text=f"S_{n} = {value}",
        given_step="states the sum formula, identifies the ratio, and evaluates the powers of the fraction exactly",
    )


def _claim(
    key, tier, start, target, setup, *, domain="Here r ≠ 1 and n is a positive integer.", ref=None, jump_ok=False
):
    """An identity family whose prompt is `setup`, then the claim `start = target` in typed form."""
    return _identity(
        key,
        tier,
        ref or kit.auto_ref(start, target),
        [T(setup), M(start), T(" = "), M(target), T(".")],
        symbols={"n": "integer", **_AR},
        domain=domain,
        jump_ok=jump_ok,
    )


_SA = "Let S_m = a(r^m - 1)/(r - 1) be the sum of the first m terms of a geometric sequence with first term a and ratio r. "


def _more_hard() -> list[dict]:
    return (
        [
            _claim(
                "later_minus_earlier",
                "hard",
                "a*(r**(2*n)-1)/(r-1)-a*(r**n-1)/(r-1)",
                "r**n*(a*(r**n-1)/(r-1))",
                _SA + "Show that S_(2n) - S_n = r^n S_n, that is ",
            ),
            _claim(
                "middle_term_square",
                "hard",
                "(a*r**n)**2",
                "(a*r**(n-1))*(a*r**(n+1))",
                "The terms of a geometric sequence are t_k = a r^(k-1). Show that the square of any term is the product of its neighbours, t_(n+1)^2 = t_n t_(n+2), that is ",
                domain="Here n is a positive integer.",
            ),
            _claim(
                "scaled_back",
                "hard",
                "(a*(r**n-1)/(r-1))*(r-1)+a",
                "a*r**n",
                _SA + "Show that (r - 1)S_n + a = a r^n, that is ",
            ),
            _claim(
                "third_term_start",
                "hard",
                "a*r**2*(r**n-1)/(r-1)",
                "a*(r**(n+2)-r**2)/(r-1)",
                "The sum a r^2 + a r^3 + ... + a r^(n+1) of n terms of a geometric sequence starting at a r^2 is a r^2(r^n - 1)/(r - 1). Show that this equals a(r^(n+2) - r^2)/(r - 1), that is ",
            ),
            _claim(
                "recurrence_sum",
                "hard",
                "a+r*(a*(r**n-1)/(r-1))",
                "a*(r**(n+1)-1)/(r-1)",
                _SA + "Show that S_(n+1) = a + r S_n, that is ",
            ),
        ]
        + [_ratio_of_sums(b) for b in (6, 7)]
        + [_ratio_of_triple(b) for b in (2, 3)]
    )


def _ratio_of_triple(b: int) -> dict:
    start = f"({b}**(3*n)-1)/({b}**n-1)"
    target = f"{b}**(2*n)+{b}**n+1"
    return _identity(
        f"ratio_of_triple_{b}",
        "hard",
        [start, f"({b}**n-1)*({b}**(2*n)+{b}**n+1)/({b}**n-1)", target],
        [
            T(
                f"For the geometric sequence with first term 1 and ratio {b}, S_m = ({b}^m - 1)/({b} - 1). Show that S_(3n)/S_n = "
            ),
            M(target),
            T(" by simplifying "),
            M(start),
            T("."),
        ],
        symbols={"n": "integer"},
        domain="Here n is a positive integer.",
    )


def _more_exam() -> list[dict]:
    return [
        _claim(
            "quadruple_sum",
            "exam",
            "a*(r**(4*n)-1)/(r-1)",
            "a*(r**n-1)/(r-1)*(1+r**n)*(1+r**(2*n))",
            _SA + "Show that S_(4n) = S_n(1 + r^n)(1 + r^(2n)), that is ",
            ref=[
                "a*(r**(4*n)-1)/(r-1)",
                "a*(r**(2*n)-1)*(r**(2*n)+1)/(r-1)",
                "a*(r**n-1)*(r**n+1)*(r**(2*n)+1)/(r-1)",
                "a*(r**n-1)/(r-1)*(1+r**n)*(1+r**(2*n))",
            ],
        ),
        _claim(
            "difference_with_ratio",
            "exam",
            "r*(a*(r**n-1)/(r-1))-a*(r**n-1)/(r-1)",
            "a*(r**n-1)",
            _SA + "Show that r S_n - S_n = a(r^n - 1), that is ",
        ),
        _claim(
            "odd_position_terms",
            "exam",
            "a*r*(r**(2*n)-1)/(r**2-1)",
            "a*(r**(2*n+1)-r)/(r**2-1)",
            "The sum a r + a r^3 + ... + a r^(2n-1) of the odd-numbered terms after the first is a r(r^(2n) - 1)/(r^2 - 1). Show that this equals a(r^(2n+1) - r)/(r^2 - 1), that is ",
            domain="Here r ≠ 1, r ≠ -1 and n is a positive integer.",
        ),
        _claim(
            "alternating_sum",
            "exam",
            "a*((-r)**n-1)/((-r)-1)",
            "a*(1-(-r)**n)/(1+r)",
            "The sum a - a r + a r^2 - ... of n terms is the geometric sum with ratio -r, a((-r)^n - 1)/((-r) - 1). Show that this equals a(1 - (-r)^n)/(1 + r), that is ",
            domain="Here r ≠ -1 and n is a positive integer.",
        ),
        _claim(
            "two_terms_from_sums",
            "exam",
            "a*(r**(n+1)-1)/(r-1)-a*(r**(n-1)-1)/(r-1)",
            "a*r**(n-1)*(1+r)",
            _SA + "Show that S_(n+1) - S_(n-1) is the sum of the n-th and (n+1)th terms, a r^(n-1)(1 + r), that is ",
            domain="Here r ≠ 1 and n is an integer with n ≥ 2.",
        ),
    ]


def _build_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {k: [] for k in ("easy", "medium", "hard", "exam", "apply_easy", "apply_exam")}
    pools["easy"] += [_telescope(m, t1) for m in (3, 4, 5, 6) for t1 in (1, 3)]
    pools["easy"] += [_numeric_induction(t1, r, "easy") for t1, r in ((1, 2), (3, 2), (5, 2), (1, 3), (2, 3))]
    pools["medium"] += [_general_induction(), _recurrence(), _term_from_sums()]
    pools["medium"] += [_numeric_induction(t1, r, "medium") for t1, r in ((3, -2), (1, -3), (2, 4), (4, 5), (2, -2))]
    pools["hard"] += [_double_sum()] + [_ratio_of_sums(b) for b in (2, 3, 4, 5)] + _more_hard()
    pools["exam"] += [_triple_sum(), _gap_to_limit(), _squares_induction(), _shifted_start()] + _more_exam()
    for t1, r, n in ((3, 2, 6), (2, 3, 5), (5, 2, 8), (1, 4, 6), (4, 3, 4), (6, 2, 7), (2, 5, 4), (3, 4, 5)):
        pools["apply_easy"].append(_apply_sum(t1, r, n))
    for t1, num, den, n in ((81, 1, 3, 5), (64, 1, 2, 6), (48, 1, 2, 5), (27, 2, 3, 4), (100, 1, 5, 4), (32, 3, 4, 3)):
        pools["apply_exam"].append(_apply_fractional(t1, num, den, n))
    return pools


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_geometric_sum_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_geometric_sum_proof(rng, variant="easy"):
    """`senior-establish-and-use-the-formula-s-n-t`. Every question carries a marking dict (`Question.answer`):
    the working as a chain of equal expressions, the reference chain, the key steps and the traps."""
    if variant not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"unknown variant: {variant}")
    if variant in ("easy", "exam") and rng.random() < 0.4:
        return kit.apply_item(rng, kit.pick(rng, pools()[f"apply_{variant}"]))
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
