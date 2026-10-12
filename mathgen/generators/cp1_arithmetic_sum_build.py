"""Generates `cp1_arithmetic_sum_proof` items: establish the sum of the first n terms of an arithmetic sequence,
or use it.

Topic ``cp1_arithmetic_sum_proof`` (open-response proof, ``level=14``). PROVE items are chains of equal
expressions: the inductive step for a stated sum, the pairing property behind the formula, and consequences of
S_n = n(2a + (n - 1)d)/2. APPLY items are numeric chains that use the formula on a stated sequence.

Tiers: induction steps for sequences with numbers, then for a general first term and common difference; the pairing
property and S_n - S_(n-1); relations between S_n, S_2n and S_3n; and the term formula recovered from a sum formula.

The generator never runs a full CAS chain check. Every claim is confirmed by direct summation, and every reference
chain numerically at fixed irregular points, with the `math` module only.
"""

from __future__ import annotations

import functools
import math

import sympy as sp

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, T
from .cp1_sum_induction_build import _family as sum_family

_n, _j = sp.symbols("n j")
_SYMS = {"a": "real", "d": "real"}


def _numeric_induction(a: int, d: int) -> dict:
    term = a + (_j - 1) * d
    closed = _n * (2 * a + (_n - 1) * d) / 2
    return sum_family(
        f"induction_{a}_{d}",
        "easy",
        term,
        closed,
        (f"{a}", f"{a + d}"),
        f"({sp.expand(term.subs(_j, _n))})",
        next_term=f"({sp.expand(term.subs(_j, sp.Symbol('k') + 1))})",
    )


def _symbolic_induction() -> dict:
    a, d = sp.symbols("a d")
    term = a + (_j - 1) * d
    closed = _n * (2 * a + (_n - 1) * d) / 2
    fam = sum_family(
        "induction_general",
        "medium",
        term,
        closed,
        ("a", "(a+d)"),
        "(a+(n-1)*d)",
        extra_symbols=_SYMS,
    )
    return fam


def _evens_odds(first: int, step: int, key: str, closed, last: str, pieces) -> dict:
    return sum_family(key, "easy", first + (_j - 1) * step, closed, pieces, last)


def _identity(key, tier, ref, lead_claim, *, symbols=None, domain="Here n is a positive integer.", jump_ok=False):
    return dict(
        key=key,
        tier=tier,
        kind="identity",
        symbols=symbols or {"n": "integer", **_SYMS},
        ref=ref,
        lead=lead_claim,
        domain=domain,
        jump_ok=jump_ok,
        given_step="states the formula S_n = n(2a + (n - 1)d)/2 or the term formula t_n = a + (n - 1)d being used",
    )


def _pairing() -> dict:
    ref = ["a + (a + (n-1)*d)", "2*a + (n-1)*d", "(a + d) + (a + (n-2)*d)"]
    return _identity(
        "pairing",
        "medium",
        ref,
        [
            T(
                "The terms of an arithmetic sequence are t_k = a + (k - 1)d. Show that the first and last of the first n terms "
                "add to the same total as the second and second-last: t_1 + t_n = t_2 + t_(n-1), that is "
            ),
            M(ref[0]),
            T(" = "),
            M(ref[-1]),
            T("."),
        ],
        domain="Here n is an integer with n ≥ 3.",
    )


def _difference_of_sums() -> dict:
    start = "n*(2*a+(n-1)*d)/2 - (n-1)*(2*a+(n-2)*d)/2"
    target = "a + (n-1)*d"
    return _identity(
        "sum_difference",
        "medium",
        kit.auto_ref(start, target),
        [
            T(
                "Let S_m = m(2a + (m - 1)d)/2 be the sum of the first m terms of an arithmetic sequence with first term a and "
                "common difference d. Show that S_n - S_(n-1) equals the n-th term, a + (n - 1)d, that is "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
    )


def _consecutive_sums() -> dict:
    start = "(n+1)*(2*a+n*d)/2 - (n-1)*(2*a+(n-2)*d)/2"
    target = "(a + (n-1)*d) + (a + n*d)"
    return _identity(
        "consecutive_sums",
        "hard",
        kit.auto_ref(start, target),
        [
            T(
                "Let S_m = m(2a + (m - 1)d)/2 be the sum of the first m terms of an arithmetic sequence. Show that "
                "S_(n+1) - S_(n-1) is the sum of the n-th and (n+1)th terms, that is "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        domain="Here n is an integer with n ≥ 2.",
    )


def _double_sum() -> dict:
    start = "2*n*(2*a+(2*n-1)*d)/2 - 2*(n*(2*a+(n-1)*d)/2)"
    target = "n**2*d"
    return _identity(
        "double_sum",
        "hard",
        kit.auto_ref(start, target),
        [
            T(
                "Let S_m = m(2a + (m - 1)d)/2 be the sum of the first m terms of an arithmetic sequence. Show that "
                "S_(2n) - 2S_n = n^2 d, that is "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        jump_ok=True,
    )


def _linear_term_induction() -> dict:
    p, q = sp.symbols("p q")
    term = p * _j + q
    closed = _n * (p * (_n + 1) + 2 * q) / 2
    return sum_family(
        "induction_linear",
        "hard",
        term,
        closed,
        ("(p+q)", "(2*p+q)"),
        "(p*n+q)",
        extra_symbols={"p": "real", "q": "real"},
    )


def _triple_sum() -> dict:
    start = "3*n*(2*a+(3*n-1)*d)/2"
    target = "3*(2*n*(2*a+(2*n-1)*d)/2 - n*(2*a+(n-1)*d)/2)"
    return _identity(
        "triple_sum",
        "exam",
        kit.auto_ref(start, target),
        [
            T(
                "Let S_m = m(2a + (m - 1)d)/2 be the sum of the first m terms of an arithmetic sequence. Show that "
                "S_(3n) = 3(S_(2n) - S_n), that is "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
        jump_ok=True,
    )


def _even_position_terms() -> dict:
    a, d = sp.symbols("a d")
    term = a + (2 * _j - 1) * d
    closed = _n * (a + _n * d)
    return sum_family(
        "even_positions",
        "exam",
        term,
        closed,
        ("(a+d)", "(a+3*d)"),
        "(a+(2*n-1)*d)",
        extra_symbols=_SYMS,
        lead_note=" These are the terms in the even positions of an arithmetic sequence with first term a and common difference d.",
    )


def _quadratic_form() -> dict:
    start = "n*(2*a+(n-1)*d)/2"
    target = "(d/2)*n**2 + (a - d/2)*n"
    return _identity(
        "quadratic_form",
        "exam",
        kit.auto_ref(start, target),
        [
            T(
                "Let S_n = n(2a + (n - 1)d)/2 be the sum of the first n terms of an arithmetic sequence. Show that S_n is a "
                "quadratic in n with no constant term: "
            ),
            M(start),
            T(" = "),
            M(target),
            T("."),
        ],
    )


def _term_from_sum(p: int, q: int) -> dict:
    start = f"{p}*n**2 + {q}*n - ({p}*(n-1)**2 + {q}*(n-1))"
    target = f"{2 * p}*n - {p - q}" if p - q >= 0 else f"{2 * p}*n + {q - p}"
    return _identity(
        f"term_from_sum_{p}_{q}",
        "exam",
        kit.auto_ref(start, target),
        [
            T(
                f"The sum of the first n terms of a sequence is S_n = {p}n^2 + {'' if q == 1 else q}n. Show that the n-th term is "
            ),
            M(target),
            T(", by showing that "),
            M(start),
            T(" equals it."),
        ],
        symbols={"n": "integer"},
        domain="Here n is an integer with n ≥ 2.",
    )


def _apply_sum(a: int, d: int, n: int) -> dict:
    terms = [a + i * d for i in range(3)]
    total = n * (2 * a + (n - 1) * d) // 2
    check(total == sum(a + i * d for i in range(n)), "apply_sum: the formula disagrees with direct addition")
    ref = [f"{n}/2*(2*{a}+({n}-1)*{d})", f"{n}/2*({2 * a}+{(n - 1) * d})", f"{n}/2*{2 * a + (n - 1) * d}", f"{total}"]
    return dict(
        key=f"apply_sum_{a}_{d}_{n}",
        tier="apply_easy",
        lead=[
            T(f"Use S_n = n(2a + (n - 1)d)/2 to find the sum of the first {n} terms of the sequence "),
            T(f"{terms[0]}, {terms[1]}, {terms[2]}, ..."),
            T("."),
        ],
        ref=ref,
        wrong_final=f"{total + 2}",
        answer_text=f"S_{n} = {total}",
        given_step="states the sum formula and substitutes a, d and n",
    )


def _apply_from_ends(a: int, d: int, n: int) -> dict:
    last = a + (n - 1) * d
    total = n * (a + last) // 2
    check(total == sum(a + i * d for i in range(n)), "apply_ends: the formula disagrees with direct addition")
    gap = (last - a) // d
    ref = [
        f"(({last}-{a})/{d}+1)*({a}+{last})/2",
        f"({gap}+1)*({a}+{last})/2",
        f"{n}*{a + last}/2",
        f"{n * (a + last)}/2",
        f"{total}",
    ]
    terms = [a, a + d, a + 2 * d]
    return dict(
        key=f"apply_ends_{a}_{d}_{n}",
        tier="apply_exam",
        lead=[
            T(
                f"Find the sum of the arithmetic sequence {terms[0]}, {terms[1]}, {terms[2]}, ..., {last}. Use the "
                "number of terms n = (l - a)/d + 1 and the sum S_n = n(a + l)/2, where a is the first term, d the "
                "common difference and l the last term."
            )
        ],
        ref=ref,
        wrong_final=f"{total + 3}",
        answer_text=f"sum = {total}",
        given_step="finds the number of terms first, then substitutes into the sum formula",
    )


def _claim(key, tier, start, target, setup, *, domain="Here n is a positive integer.", jump_ok=False, symbols=None):
    """An identity family whose prompt is `setup`, then the claim `start = target` in typed form."""
    return _identity(
        key,
        tier,
        kit.auto_ref(start, target),
        [T(setup), M(start), T(" = "), M(target), T(".")],
        domain=domain,
        jump_ok=jump_ok,
        symbols=symbols,
    )


_S = "Let S_m = m(2a + (m - 1)d)/2 be the sum of the first m terms of an arithmetic sequence with first term a and common difference d. "
_T = "The terms of an arithmetic sequence are t_k = a + (k - 1)d. "


def _more_medium() -> list[dict]:
    return [
        _claim(
            "sum_with_last",
            "medium",
            "n*(2*a+(n-1)*d)/2",
            "n*(a+(a+(n-1)*d))/2",
            _S + "Show that S_n can be written with the first and last terms, that is ",
        ),
        _claim(
            "common_difference",
            "medium",
            "(a+n*d)-(a+(n-1)*d)",
            "d",
            _T + "Show that consecutive terms differ by d, that is ",
        ),
        _claim(
            "next_sum",
            "medium",
            "n*(2*a+(n-1)*d)/2+(a+n*d)",
            "(n+1)*(2*a+n*d)/2",
            _S + "Show that S_n plus the (n+1)th term is S_(n+1), that is ",
        ),
        _claim(
            "middle_term",
            "medium",
            "(a+(n-2)*d)+(a+n*d)",
            "2*(a+(n-1)*d)",
            _T + "Show that each term is the mean of its neighbours: t_(n-1) + t_(n+1) = 2t_n, that is ",
            domain="Here n is an integer with n ≥ 2.",
        ),
        _claim(
            "mean_of_terms",
            "medium",
            "(n*(2*a+(n-1)*d)/2)/n",
            "a+(n-1)*d/2",
            _S + "Show that the mean of the first n terms is a + (n - 1)d/2, that is ",
        ),
    ] + [sum_family_for(a, d) for a, d in ((20, -3), (15, -2), (30, -4))]


def sum_family_for(a: int, d: int) -> dict:
    """An induction step for a sequence with a negative common difference."""
    last = f"({a}{d:+d}*(n-1))".replace("+-", "-")
    return sum_family(
        f"induction_negative_{a}_{abs(d)}",
        "medium",
        a + (_j - 1) * d,
        _n * (2 * a + (_n - 1) * d) / 2,
        (f"{a}", f"({a}{d:+d})"),
        last,
    )


def _more_hard() -> list[dict]:
    return [
        _claim(
            "double_minus_single",
            "hard",
            "2*n*(2*a+(2*n-1)*d)/2-n*(2*a+(n-1)*d)/2",
            "n*(2*a+(3*n-1)*d)/2",
            _S + "Show that S_(2n) - S_n = n(2a + (3n - 1)d)/2, that is ",
        ),
        _claim(
            "second_difference",
            "hard",
            "(n+2)*(2*a+(n+1)*d)/2-2*((n+1)*(2*a+n*d)/2)+n*(2*a+(n-1)*d)/2",
            "d",
            _S + "Show that the second difference S_(n+2) - 2S_(n+1) + S_n is the constant d, that is ",
        ),
        _claim(
            "difference_of_squares",
            "hard",
            "(a+(n-1)*d)**2-(a+(n-2)*d)**2",
            "d*(2*a+(2*n-3)*d)",
            _T + "Show that t_n^2 - t_(n-1)^2 = d(2a + (2n - 3)d), that is ",
            domain="Here n is an integer with n ≥ 2.",
        ),
        _claim(
            "three_terms",
            "hard",
            "(n+3)*(2*a+(n+2)*d)/2-n*(2*a+(n-1)*d)/2",
            "3*(a+(n+1)*d)",
            _S
            + "Show that the three terms after the n-th add to three times the middle one, S_(n+3) - S_n = 3t_(n+2), that is ",
        ),
        _claim(
            "odd_count",
            "hard",
            "(2*n+1)*(2*a+2*n*d)/2",
            "(2*n+1)*(a+n*d)",
            _S + "Show that S_(2n+1) is (2n + 1) times the middle term t_(n+1), that is ",
        ),
        _claim(
            "neighbour_product",
            "hard",
            "(a+n*d)**2-(a+(n-1)*d)*(a+(n+1)*d)",
            "d**2",
            _T + "Show that t_(n+1)^2 - t_n t_(n+2) is d^2 for every n, that is ",
            domain="Here n is an integer with n ≥ 1.",
        ),
    ]


def _build_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {k: [] for k in ("easy", "medium", "hard", "exam", "apply_easy", "apply_exam")}
    for a, d in ((3, 4), (2, 5), (5, 3), (1, 6), (4, 2), (7, 3), (6, 5), (2, 7)):
        pools["easy"].append(_numeric_induction(a, d))
    pools["easy"].append(_evens_odds(2, 2, "evens", _n * (_n + 1), "(2*n)", ("2", "4")))
    pools["easy"].append(_evens_odds(1, 1, "naturals", _n * (_n + 1) / 2, "n", ("1", "2")))
    pools["medium"] += [_symbolic_induction(), _pairing(), _difference_of_sums()] + _more_medium()
    pools["hard"] += [_consecutive_sums(), _double_sum(), _linear_term_induction()] + _more_hard()
    pools["exam"] += [_triple_sum(), _even_position_terms(), _quadratic_form()]
    pools["exam"] += [_term_from_sum(p, q) for p, q in ((3, 2), (2, 5), (4, 1), (5, 3), (2, 1))]
    for a, d, n in ((5, 3, 20), (2, 4, 15), (7, 2, 12), (10, 5, 8), (3, 6, 10), (1, 3, 25)):
        pools["apply_easy"].append(_apply_sum(a, d, n))
    for a, d, n in ((4, 4, 25), (3, 6, 18), (7, 5, 30), (2, 3, 40), (10, 4, 22), (5, 7, 16)):
        pools["apply_exam"].append(_apply_from_ends(a, d, n))
    return pools


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_arithmetic_sum_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_arithmetic_sum_proof(rng, variant="easy"):
    """`senior-establish-and-use-the-formula-for-the-sum`. Every question carries a marking dict (`Question.answer`):
    the working as a chain of equal expressions, the reference chain, the key steps and the traps."""
    if variant not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"unknown variant: {variant}")
    if variant in ("easy", "exam") and rng.random() < 0.4:
        return kit.apply_item(rng, kit.pick(rng, pools()[f"apply_{variant}"]))
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
