"""Generates `cp1_exponential_laws_proof` items: establish an identity between exponential expressions from the laws of
indices, or use the laws to evaluate a power.

Topic ``cp1_exponential_laws_proof`` (open-response proof, ``level=14``). Each PROVE item gives two expressions in x
(and sometimes h) with a numeric or positive symbolic base and asks for a chain of equal expressions between them.
Each APPLY item is a numeric chain that uses the laws to evaluate a power or a product of powers exactly.

Tiers: one law at a time (sum, difference, power, reciprocal); two laws together, changes of base and expanding a
square; factorising in b^x, collecting a shifted family, a difference quotient; and difference-of-squares products,
symbolic bases and ratios that need two laws in the right order.

The generator never runs a full CAS chain check. Every reference chain is confirmed numerically at fixed irregular
points, with the `math` module only.
"""

from __future__ import annotations

import functools

import math
from fractions import Fraction

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import T

_LAWS = "the laws of indices, such as a^(m + n) = a^m × a^n and (a^m)^n = a^(mn)"


def _fam(key, tier, ref, *, symbols=None, domain="Here x and h are real numbers.", extra=""):
    syms = symbols or {"x": "real", "h": "real"}
    lead = [T("Prove that "), kit.M(ref[0]), T(" = "), kit.M(ref[-1]), T(f". You may cite {_LAWS}.")]
    return dict(
        key=key,
        tier=tier,
        kind="identity",
        symbols=syms,
        ref=ref,
        lead=lead,
        domain=domain,
        extra=extra,
        jump_ok=len(ref) >= 4,
        given_step="names the law of indices used in each step",
    )


def _only(*names):
    return {n: ("positive" if n == "a" else "real") for n in names}


def _sum_law(b):
    return _fam(f"sum_{b}", "easy", [f"{b}**(x+h)", f"{b}**x*{b}**h"], symbols=_only("x", "h"))


def _quotient_law(b):
    return _fam(f"quotient_{b}", "easy", [f"{b}**(x-h)", f"{b}**x/{b}**h"], symbols=_only("x", "h"))


def _power_law(b, m):
    return _fam(
        f"power_{b}_{m}",
        "easy",
        [f"{b}**({m}*x)", f"({b}**x)**{m}"],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _step_law(b):
    return _fam(
        f"step_{b}",
        "easy",
        [f"{b}**(x+1)", f"{b}**x*{b}**1", f"{b}*{b}**x"],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _reciprocal_law(b):
    return _fam(
        f"reciprocal_{b}", "easy", [f"{b}**(-x)", f"1/{b}**x"], symbols=_only("x"), domain="Here x is a real number."
    )


def _difference(b):
    return _fam(
        f"difference_{b}",
        "medium",
        [f"{b}**(x+h) - {b}**x", f"{b}**x*{b}**h - {b}**x", f"{b}**x*({b}**h - 1)"],
    )


def _shifted_pair(b, p, q):
    return _fam(
        f"shifted_{b}_{p}_{q}",
        "medium",
        [f"{b}**(x+{p}) - {b}**(x+{q})", f"{b}**{p}*{b}**x - {b}**{q}*{b}**x", f"{b**p - b**q}*{b}**x"],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _change_of_base(big, small, e):
    return _fam(
        f"base_{big}_{small}",
        "medium",
        [f"{big}**x", f"({small}**{e})**x", f"{small}**({e}*x)"],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _mixed_base(big, small, e, k, product):
    """A product or quotient of powers in two related bases, written in the smaller base (several laws in turn)."""
    if product:
        ref = [
            f"{big}**x*{small}**(x+{k})",
            f"({small}**{e})**x*{small}**(x+{k})",
            f"{small}**({e}*x)*{small}**(x+{k})",
            f"{small}**({e}*x+x+{k})",
            f"{small}**({e + 1}*x+{k})",
        ]
    else:
        ref = [
            f"{big}**x/{small}**(x-{k})",
            f"({small}**{e})**x/{small}**(x-{k})",
            f"{small}**({e}*x)/{small}**(x-{k})",
            f"{small}**({e}*x-(x-{k}))",
            f"{small}**({e - 1}*x+{k})",
        ]
    return _fam(
        f"mixed_base_{big}_{small}_{k}_{'p' if product else 'q'}",
        "medium",
        ref,
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _square_expansion(b):
    return _fam(
        f"conjugate_pair_{b}",
        "medium",
        [
            f"({b}**x + {b}**(-x))*({b}**x - {b}**(-x))",
            f"({b}**x)**2 - ({b}**(-x))**2",
            f"{b}**(2*x) - {b}**(-2*x)",
        ],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _quadratic_in_power(b, p, q):
    return _fam(
        f"quadratic_{b}_{p}_{q}",
        "hard",
        [
            f"{b}**(2*x) - {p + q}*{b}**x + {p * q}",
            f"({b}**x)**2 - {p + q}*{b}**x + {p * q}",
            f"({b}**x - {p})*({b}**x - {q})",
        ],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _collect(b, p, q):
    return _fam(
        f"collect_{b}_{p}_{q}",
        "hard",
        [
            f"{b}**(x+{p}) - {b}**(x+{q}) + {b}**x",
            f"{b}**{p}*{b}**x - {b}**{q}*{b}**x + {b}**x",
            f"({b**p} - {b**q} + 1)*{b}**x",
            f"{b**p - b**q + 1}*{b}**x",
        ],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _square_difference(b):
    return _fam(
        f"square_difference_{b}",
        "hard",
        [
            f"({b}**x + {b}**(-x))**2 - ({b}**x - {b}**(-x))**2",
            f"{b}**(2*x) + 2 + {b}**(-2*x) - ({b}**(2*x) - 2 + {b}**(-2*x))",
            "4",
        ],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _cancel_factor(b, power):
    if power == 2:
        ref = [f"({b}**(2*x) - 1)/({b}**x + 1)", f"({b}**x - 1)*({b}**x + 1)/({b}**x + 1)", f"{b}**x - 1"]
    else:
        ref = [
            f"({b}**(3*x) - 1)/({b}**x - 1)",
            f"({b}**x - 1)*({b}**(2*x) + {b}**x + 1)/({b}**x - 1)",
            f"{b}**(2*x) + {b}**x + 1",
        ]
    return _fam(f"cancel_{b}_{power}", "hard", ref, symbols=_only("x"), domain=f"Here x is a positive real number.")


def _difference_quotient_numeric(b):
    return _fam(
        f"quotient_numeric_{b}",
        "exam",
        [f"({b}**(x+h) - {b}**x)/h", f"({b}**x*{b}**h - {b}**x)/h", f"{b}**x*({b}**h - 1)/h"],
        symbols={"x": "real", "h": "nonzero"},
        domain="Here x is real and h is a non-zero real number.",
    )


def _difference_quotient_symbolic():
    return _fam(
        "quotient_symbolic",
        "exam",
        ["(a**(x+h) - a**x)/h", "(a**x*a**h - a**x)/h", "a**x*(a**h - 1)/h"],
        symbols={"a": "positive", "x": "real", "h": "nonzero"},
        domain="Here a > 0, x is real and h is a non-zero real number.",
    )


def _hyperbolic_identity(b):
    return _fam(
        f"hyperbolic_{b}",
        "exam",
        [
            f"(({b}**x + {b}**(-x))/2)**2 - (({b}**x - {b}**(-x))/2)**2",
            f"({b}**(2*x) + 2 + {b}**(-2*x))/4 - ({b}**(2*x) - 2 + {b}**(-2*x))/4",
            "(4)/4",
            "1",
        ],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _growth_ratio(b):
    return _fam(
        f"growth_ratio_{b}",
        "exam",
        [
            f"({b}**(x+1) + {b}**x)/({b}**(x+1) - {b}**x)",
            f"({b}*{b}**x + {b}**x)/({b}*{b}**x - {b}**x)",
            f"{b}**x*({b} + 1)/({b}**x*({b} - 1))",
            f"({b} + 1)/({b} - 1)",
        ],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


def _symbolic_base(kind):
    if kind == "product":
        ref = ["a**(x+h)*a**(x-h)", "a**(x+h+x-h)", "a**(2*x)"]
    elif kind == "ratio":
        ref = ["(a**(x+h))**2/a**(2*x)", "a**(2*x+2*h)/a**(2*x)", "a**(2*x+2*h-2*x)", "a**(2*h)"]
    else:
        ref = ["a**(2*x+h)/a**(x-h)", "a**(2*x+h-x+h)", "a**(x+2*h)"]
    return _fam(
        f"symbolic_{kind}",
        "exam",
        ref,
        symbols={"a": "positive", "x": "real", "h": "real"},
        domain="Here a > 0 and x and h are real numbers.",
    )


def _product_of_powers(a, b):
    big_a, big_b = a * a, b * b
    return _fam(
        f"conjugate_powers_{a}_{b}",
        "exam",
        [
            f"({a}**x + {b}**x)*({a}**x - {b}**x)",
            f"({a}**x)**2 - ({b}**x)**2",
            f"{big_a}**x - {big_b}**x",
        ],
        symbols=_only("x"),
        domain="Here x is a real number.",
    )


# --- APPLY items -----------------------------------------------------------------------------------------------
def _apply_root(c: int, p: int, m: int, n: int) -> dict:
    base = c**p
    e = Fraction(p * m, n)
    check(e.denominator == 1, "apply_root: the exponent must come out whole")
    value = c ** int(e)
    check(math.isclose(base ** (m / n), value, rel_tol=1e-9), "apply_root: the value is wrong")
    ref = [f"{base}**({m}/{n})", f"({c}**{p})**({m}/{n})", f"{c}**({p}*{m}/{n})", f"{c}**{int(e)}", f"{value}"]
    return dict(
        key=f"apply_root_{base}_{m}_{n}",
        tier="apply_easy",
        lead=[T("Use the laws of indices to evaluate "), kit.M(ref[0]), T(" exactly.")],
        ref=ref,
        wrong_final=f"{value + 1}",
        answer_text=f"{kit.disp(ref[0])} = {value}",
        given_step="writes the base as a power of a smaller number and applies (a^m)^n = a^(mn)",
    )


def _apply_mixed(c: int, terms: tuple[tuple[int, str], ...]) -> dict:
    """terms: (power p, exponent text 'm/n' or '-m/n' or 'k') of c**p raised to the exponent; first two multiply, third divides."""
    es, parts = [], []
    for p, ex in terms:
        frac = Fraction(ex)
        e = p * frac
        check(e.denominator == 1, "apply_mixed: every exponent must come out whole")
        es.append(int(e))
        parts.append(f"{c**p}" if ex == "1" else f"{c**p}**({ex})")
    total = es[0] + es[1] - es[2]
    value = Fraction(c) ** total
    value_text = f"{value.numerator}" if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    check(total != 0, "apply_mixed: the answer should not be 1")
    expr = f"{parts[0]}*{parts[1]}/{parts[2]}"
    same_base = f"{c}**({es[0]})*{c}**({es[1]})/{c}**({es[2]})"
    ref = [expr, same_base, f"{c}**({es[0]}+{es[1]}-{es[2]})", f"{c}**({total})", value_text]
    return dict(
        key=f"apply_mixed_{c}_{'_'.join(str(p) for p, _ in terms)}_{total}",
        tier="apply_exam",
        lead=[T("Use the laws of indices to evaluate "), kit.M(expr), T(" exactly.")],
        ref=ref,
        wrong_final=f"{value + 1}" if value.denominator == 1 else f"{value.numerator + 1}/{value.denominator}",
        answer_text=f"{kit.disp(expr)} = {value_text}",
        given_step="writes every power with one base and combines the exponents with the laws of indices",
    )


def _build_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {k: [] for k in ("easy", "medium", "hard", "exam", "apply_easy", "apply_exam")}
    for b in (2, 3, 5):
        pools["easy"] += [_sum_law(b), _quotient_law(b), _step_law(b), _reciprocal_law(b)]
        pools["easy"] += [_power_law(b, m) for m in (2, 3)]
    for b in (2, 3, 5):
        pools["medium"] += [_difference(b), _square_expansion(b)]
    for b in (2, 3):
        pools["medium"] += [_shifted_pair(b, p, q) for p, q in ((2, 1), (3, 1), (3, 2))]
    pools["medium"] += [
        _mixed_base(big, small, e, k, product)
        for big, small, e, k, product in (
            (4, 2, 2, 1, True),
            (9, 3, 2, 2, True),
            (8, 2, 3, 1, True),
            (27, 3, 3, 1, True),
            (25, 5, 2, 1, False),
            (4, 2, 2, 3, False),
            (9, 3, 2, 2, False),
            (8, 2, 3, 1, False),
            (125, 5, 3, 2, False),
        )
    ]
    for b, p, q in ((2, 1, 2), (2, 1, 4), (2, 2, 4), (3, 1, 3), (3, 3, 9), (5, 1, 5), (2, 1, 8)):
        pools["hard"].append(_quadratic_in_power(b, p, q))
    for b in (2, 3, 5):
        pools["hard"] += [_collect(b, p, q) for p, q in ((3, 1), (2, 1), (4, 2))]
    pools["hard"] += [_square_difference(b) for b in (2, 3, 5, 10)]
    pools["hard"] += [_cancel_factor(b, pw) for b in (2, 3, 5) for pw in (2, 3)]
    pools["exam"] += [_difference_quotient_numeric(b) for b in (2, 3, 5)]
    pools["exam"] += [_difference_quotient_symbolic()]
    pools["exam"] += [_hyperbolic_identity(b) for b in (2, 3, 5)]
    pools["exam"] += [_growth_ratio(b) for b in (2, 3, 4, 5)]
    pools["exam"] += [_symbolic_base(k) for k in ("product", "ratio", "quotient")]
    pools["exam"] += [_product_of_powers(a, b) for a, b in ((2, 3), (2, 5), (3, 4), (3, 5))]
    for c, p, m, n in (
        (3, 3, 2, 3),
        (2, 3, 2, 3),
        (2, 4, 3, 4),
        (2, 5, 3, 5),
        (5, 3, 2, 3),
        (3, 4, 3, 4),
        (2, 6, 5, 6),
        (2, 2, 5, 2),
        (3, 2, 3, 2),
        (5, 2, 3, 2),
    ):
        pools["apply_easy"].append(_apply_root(c, p, m, n))
    for c, terms in (
        (2, ((3, "2/3"), (2, "3/2"), (1, "3"))),
        (3, ((2, "3/2"), (1, "-2"), (3, "2/3"))),
        (2, ((4, "3/4"), (5, "-2/5"), (2, "3/2"))),
        (2, ((3, "4/3"), (2, "-1/2"), (6, "1/6"))),
        (5, ((2, "3/2"), (3, "2/3"), (1, "1"))),
        (3, ((4, "1/2"), (3, "4/3"), (2, "1"))),
        (2, ((6, "1/2"), (4, "3/4"), (1, "2"))),
        (3, ((2, "-1/2"), (6, "1/3"), (3, "2/3"))),
    ):
        pools["apply_exam"].append(_apply_mixed(c, terms))
    return pools


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_exponential_laws_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_exponential_laws_proof(rng, variant="easy"):
    """`senior-establish-and-use-the-algebraic-properties-of-exponential`. Every question carries a marking dict
    (`Question.answer`): the working as a chain of equal expressions, the reference chain, the key steps and the traps."""
    if variant not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"unknown variant: {variant}")
    if variant in ("easy", "exam") and rng.random() < 0.4:
        return kit.apply_item(rng, kit.pick(rng, pools()[f"apply_{variant}"]))
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
