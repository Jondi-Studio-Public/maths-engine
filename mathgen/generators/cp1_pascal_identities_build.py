"""Generates `cp1_pascal_identity_proof` items: prove an identity between binomial coefficients, or use one.

Topic ``cp1_pascal_identity_proof`` (open-response proof, ``level=14``). The binomial coefficient C(n, k) is given
to the learner as a polynomial in n (for example C(n, 3) = n(n - 1)(n - 2)/6), so every line of the working is an
ordinary algebraic expression a symbolic check can follow. Two exam items use the factorial form instead,
n!/(r!(n - r)!), with at most three factorials on a line.

Tiers: Pascal's rule and the absorption identity for a small k; the same with a shifted n, ratios of neighbouring
coefficients and differences along a diagonal; Pascal's rule applied twice and a Vandermonde case; the hockey-stick
step by induction, products of coefficients, and the general factorial forms.

Two kinds of item: PROVE (an identity as a chain of equal expressions) and APPLY (a numeric chain that uses an
identity to find one coefficient from others; easy and exam tiers only).

The generator never runs a full CAS chain check. Every claim is confirmed numerically against `math.comb`, and every
reference chain at fixed irregular points, with the `math` module only.
"""

from __future__ import annotations

import functools

import math

import sympy as sp

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, R, T, disp, tex


def _paren(expr) -> str:
    s = str(expr)
    return f"({s})" if isinstance(expr, sp.Add) else s


def cb(top: str, k: int) -> str:
    """C(top, k) as a sympy-style polynomial: top(top-1)...(top-k+1)/k!."""
    if k == 0:
        return "1"
    t = sp.sympify(top)
    factors = "*".join(_paren(t - j) for j in range(k))
    return factors if k == 1 else f"{factors}/{math.factorial(k)}"


def _binom(top: str, k) -> tuple:
    return R(f"C({disp(top)}, {k})", rf"\binom{{{tex(top)}}}{{{k}}}")


def _definitions(pairs: list[tuple[str, int]]) -> list[tuple]:
    """' Here C(n, 2) = n(n - 1)/2 and ...' for each coefficient the statement uses."""
    seen: list[tuple[str, int]] = []
    for pair in pairs:
        if pair not in seen and pair[1] >= 1:
            seen.append(pair)
    parts: list[tuple] = [T(" Use ")]
    for i, (top, k) in enumerate(seen):
        if i:
            parts.append(T(", " if i < len(seen) - 1 else " and "))
        parts += [_binom(top, k), T(" = "), M(cb(top, k))]
    parts.append(T("."))
    return parts


def _holds(key: str, lhs, rhs, lo: int = 6, hi: int = 16) -> None:
    for n in range(lo, hi):
        check(math.isclose(lhs(n), rhs(n), rel_tol=1e-9), f"{key}: the stated identity is false at n = {n}")


_DOMAIN = "Take n to be a positive integer large enough for every coefficient to make sense."


def _identity(key, tier, claim, used, ref, lhs, rhs, *, symbols=None, extra="", jump_ok=False, domain=_DOMAIN):
    _holds(key, lhs, rhs)
    return dict(
        key=key,
        tier=tier,
        kind="identity",
        symbols=symbols or {"n": "integer"},
        ref=ref,
        lead=[T("Prove that ")] + claim + [T(".")] + (_definitions(used) if used else []),
        domain=domain,
        extra=extra,
        jump_ok=jump_ok,
        given_step="states the definition of each coefficient and the identity used to combine them",
    )


def _pascal(tier, a, k):
    top, top1 = ("n" if a == 0 else f"n+{a}"), f"n+{a + 1}"
    c_k, c_k1 = cb(top, k), cb(top, k + 1)
    ref = [
        f"{c_k} + {c_k1}",
        f"({c_k})*(1 + ({top}-{k})/{k + 1})",
        f"({c_k})*({top}+1)/{k + 1}",
        cb(top1, k + 1),
    ]
    return _identity(
        f"pascal_{a}_{k}",
        tier,
        [_binom(top, k), T(" + "), _binom(top, k + 1), T(" = "), _binom(top1, k + 1)],
        [(top, k), (top, k + 1), (top1, k + 1)],
        ref,
        lambda n: math.comb(n + a, k) + math.comb(n + a, k + 1),
        lambda n: math.comb(n + a + 1, k + 1),
        jump_ok=True,
    )


def _absorb(k):
    prod = "*".join(["n"] + [f"(n-{j})" for j in range(1, k)])
    ref = [f"{k}*({cb('n', k)})", f"{prod}/{math.factorial(k - 1)}", f"n*({cb('n-1', k - 1)})"]
    return _identity(
        f"absorb_{k}",
        "easy",
        [T(f"{k} × "), _binom("n", k), T(" = n × "), _binom("n-1", k - 1)],
        [("n", k), ("n-1", k - 1)],
        ref,
        lambda n: k * math.comb(n, k),
        lambda n: n * math.comb(n - 1, k - 1),
    )


def _adjacent_squares(a):
    m, m1 = ("n" if a == 0 else f"n+{a}"), f"n+{a + 1}"
    ref = [
        f"{cb(m, 2)} + {cb(m1, 2)}",
        f"({m})*({m}-1)/2 + ({m}+1)*({m})/2",
        f"({m})*(({m}-1) + ({m}+1))/2",
        f"({m})**2",
    ]
    return _identity(
        f"adjacent_squares_{a}",
        "easy",
        [_binom(m, 2), T(" + "), _binom(m1, 2), T(" = "), M(f"({m})**2")],
        [(m, 2), (m1, 2)],
        ref,
        lambda n: math.comb(n + a, 2) + math.comb(n + a + 1, 2),
        lambda n: (n + a) ** 2,
    )


def _ratio(k):
    ref = [
        f"({cb('n', k + 1)})/({cb('n', k)})",
        f"(({cb('n', k)})*(n-{k})/{k + 1})/({cb('n', k)})",
        f"(n-{k})/{k + 1}",
    ]
    return _identity(
        f"ratio_{k}",
        "medium",
        [_binom("n", k + 1), T(" ÷ "), _binom("n", k), T(" = "), M(f"(n-{k})/{k + 1}")],
        [("n", k), ("n", k + 1)],
        ref,
        lambda n: math.comb(n, k + 1) / math.comb(n, k),
        lambda n: (n - k) / (k + 1),
    )


def _shift(k):
    ref = [f"(n-{k})*({cb('n', k)})", f"(n-{k})*({cb('n', k)})*{k + 1}/{k + 1}", f"{k + 1}*({cb('n', k + 1)})"]
    return _identity(
        f"shift_{k}",
        "medium",
        [T(f"(n - {k}) × "), _binom("n", k), T(f" = {k + 1} × "), _binom("n", k + 1)],
        [("n", k), ("n", k + 1)],
        ref,
        lambda n: (n - k) * math.comb(n, k),
        lambda n: (k + 1) * math.comb(n, k + 1),
    )


def _diagonal(k):
    ref = [
        f"{cb('n+1', k + 1)} - ({cb('n', k + 1)})",
        f"({cb('n', k + 1)}) + ({cb('n', k)}) - ({cb('n', k + 1)})",
        cb("n", k),
    ]
    return _identity(
        f"diagonal_{k}",
        "medium",
        [_binom("n+1", k + 1), T(" - "), _binom("n", k + 1), T(" = "), _binom("n", k)],
        [("n+1", k + 1), ("n", k + 1), ("n", k)],
        ref,
        lambda n: math.comb(n + 1, k + 1) - math.comb(n, k + 1),
        lambda n: math.comb(n, k),
    )


def _double_pascal(k):
    ref = [
        cb("n+2", k + 2),
        f"({cb('n+1', k + 1)}) + ({cb('n+1', k + 2)})",
        f"(({cb('n', k)}) + ({cb('n', k + 1)})) + (({cb('n', k + 1)}) + ({cb('n', k + 2)}))",
        f"{cb('n', k)} + 2*({cb('n', k + 1)}) + {cb('n', k + 2)}",
    ]
    return _identity(
        f"double_pascal_{k}",
        "hard",
        [
            _binom("n+2", k + 2),
            T(" = "),
            _binom("n", k),
            T(" + 2 × "),
            _binom("n", k + 1),
            T(" + "),
            _binom("n", k + 2),
        ],
        [("n+2", k + 2), ("n", k), ("n", k + 1), ("n", k + 2)],
        ref,
        lambda n: math.comb(n + 2, k + 2),
        lambda n: math.comb(n, k) + 2 * math.comb(n, k + 1) + math.comb(n, k + 2),
        jump_ok=True,
    )


def _vandermonde(m):
    top = f"n+{m}"
    half = m * (m - 1) // 2
    ref = [
        cb(top, 2),
        f"(n+{m})*(n+{m - 1})/2",
        f"(n**2 + {2 * m - 1}*n + {m * (m - 1)})/2",
        f"{cb('n', 2)} + {half} + {m}*n",
    ]
    return _identity(
        f"vandermonde_{m}",
        "hard",
        [_binom(top, 2), T(" = "), _binom("n", 2), T(f" + {half} + {m}n")],
        [(top, 2), ("n", 2)],
        ref,
        lambda n: math.comb(n + m, 2),
        lambda n: math.comb(n, 2) + half + m * n,
        jump_ok=True,
    )


def _diagonal_ratio(k):
    ref = [
        f"({cb('n+1', k + 1)})/({cb('n', k)})",
        f"(({cb('n', k)})*(n+1)/{k + 1})/({cb('n', k)})",
        f"(n+1)/{k + 1}",
    ]
    return _identity(
        f"diagonal_ratio_{k}",
        "hard",
        [_binom("n+1", k + 1), T(" ÷ "), _binom("n", k), T(" = "), M(f"(n+1)/{k + 1}")],
        [("n+1", k + 1), ("n", k)],
        ref,
        lambda n: math.comb(n + 1, k + 1) / math.comb(n, k),
        lambda n: (n + 1) / (k + 1),
    )


def _hockey_stick(r):
    for n in range(r + 1, 16):
        check(
            sum(math.comb(j, r) for j in range(r, n + 1)) == math.comb(n + 1, r + 1),
            f"hockey_stick_{r}: the stated sum is false at n = {n}",
        )
    ref = [
        f"{cb('k+1', r + 1)} + {cb('k+1', r)}",
        f"({cb('k+1', r)})*(k-{r - 1})/{r + 1} + {cb('k+1', r)}",
        f"({cb('k+1', r)})*(k+2)/{r + 1}",
        cb("k+2", r + 1),
    ]
    lead = (
        [T("Prove by induction that "), _binom(str(r), r), T(" + "), _binom(str(r + 1), r), T(" + ... + ")]
        + [_binom("n", r), T(" = "), _binom("n+1", r + 1)]
        + [
            T(
                f" for every integer n ≥ {r}. Check the base case n = {r}, then in the inductive step assume the "
                "result for n = k and add the next coefficient, "
            ),
            _binom("k+1", r),
            T("."),
        ]
        + _definitions([("n", r), ("n+1", r + 1), ("k+1", r), ("k+1", r + 1), ("k+2", r + 1)])
    )
    return dict(
        key=f"hockey_stick_{r}",
        tier="exam",
        kind="induction",
        symbols={"k": "integer"},
        ref=ref,
        lead=lead,
        jump_ok=True,
    )


def _product(a, b):
    c = math.comb(a + b, a)
    prod = "*".join(["n"] + [f"(n-{j})" for j in range(1, a + b)])
    ref = [
        f"({cb('n', a)})*({cb(f'n-{a}', b)})",
        f"{prod}/{math.factorial(a) * math.factorial(b)}",
        f"{c}*({cb('n', a + b)})",
    ]
    return _identity(
        f"product_{a}_{b}",
        "exam",
        [_binom("n", a), T(" × "), _binom(f"n-{a}", b), T(f" = {c} × "), _binom("n", a + b)],
        [("n", a), (f"n-{a}", b), ("n", a + b)],
        ref,
        lambda n: math.comb(n, a) * math.comb(n - a, b),
        lambda n: c * math.comb(n, a + b),
    )


def _factorial_pair() -> dict:
    ref = [
        "r*(r-1)*factorial(n)/(factorial(r)*factorial(n-r))",
        "factorial(n)/(factorial(r-2)*factorial(n-r))",
        "n*(n-1)*factorial(n-2)/(factorial(r-2)*factorial(n-r))",
    ]
    return dict(
        key="factorial_pair",
        tier="exam",
        kind="identity",
        symbols={"n": "integer", "r": "integer"},
        probes={"n": [12, 11, 13, 14], "r": [5, 6, 4, 7]},
        ref=ref,
        lead=[
            T("Prove that r(r - 1) × "),
            _binom("n", "r"),
            T(" = n(n - 1) × "),
            _binom("n-2", "r-2"),
            T(", where "),
            _binom("n", "r"),
            T(" = n!/(r!(n - r)!)."),
        ],
        domain="Here n and r are integers with 2 ≤ r ≤ n.",
        extra="Write factorials with an exclamation mark, as in n!.",
        given_step="states the factorial form of the coefficient and the factorial identity (m + 1)! = (m + 1) m!",
    )


def _central() -> dict:
    ref = [
        "factorial(2*n+2)/(factorial(n+1)*factorial(n+1))",
        "(2*n+2)*(2*n+1)*factorial(2*n)/((n+1)*(n+1)*factorial(n)*factorial(n))",
        "2*(2*n+1)/(n+1)*factorial(2*n)/(factorial(n)*factorial(n))",
    ]
    return dict(
        key="central",
        tier="exam",
        kind="identity",
        symbols={"n": "integer"},
        ref=ref,
        lead=[
            T("Prove that "),
            _binom("2*n+2", "n+1"),
            T(" = "),
            M("2*(2*n+1)/(n+1)"),
            T(" × "),
            _binom("2*n", "n"),
            T(", where "),
            _binom("m", "r"),
            T(" = m!/(r!(m - r)!)."),
        ],
        domain="Here n is a positive integer.",
        extra="Write factorials with an exclamation mark, as in n!.",
        given_step="states the factorial form of the coefficient and expands (2n + 2)! twice",
    )


# --- APPLY items: a numeric chain that uses an identity to find one coefficient from others ----------------
def _apply_pascal(n: int, r: int) -> dict:
    c1, c2 = math.comb(n, r), math.comb(n, r + 1)
    total = c1 + c2
    check(total == math.comb(n + 1, r + 1), "apply_pascal: Pascal's rule fails")
    return dict(
        key=f"apply_pascal_{n}_{r}",
        tier="apply_easy",
        lead=[
            T("Given that "),
            _binom(str(n), r),
            T(f" = {c1} and "),
            _binom(str(n), r + 1),
            T(f" = {c2}, use Pascal's rule, "),
            _binom("n", "r"),
            T(" + "),
            _binom("n", "r+1"),
            T(" = "),
            _binom("n+1", "r+1"),
            T(", to find "),
            _binom(str(n + 1), r + 1),
            T("."),
        ],
        ref=[f"{c1} + {c2}", f"{total}"],
        wrong_final=f"{total + 10}",
        answer_text=f"C({n + 1}, {r + 1}) = {total}",
        given_step="names Pascal's rule and adds the two given coefficients",
    )


def _apply_absorb(n: int, r: int) -> dict:
    c = math.comb(n - 1, r - 1)
    value = math.comb(n, r)
    check(n * c % r == 0 and n * c // r == value, "apply_absorb: absorption identity fails")
    ref = [f"{n}*{c}/{r}"]
    if c % r == 0:
        ref.append(f"{n}*{c // r}")
    ref.append(f"{value}")
    return dict(
        key=f"apply_absorb_{n}_{r}",
        tier="apply_exam",
        lead=[
            T("Given that "),
            _binom(str(n - 1), r - 1),
            T(f" = {c}, use r × "),
            _binom("n", "r"),
            T(" = n × "),
            _binom("n-1", "r-1"),
            T(" to find "),
            _binom(str(n), r),
            T("."),
        ],
        ref=ref,
        wrong_final=f"{value + 7}",
        answer_text=f"C({n}, {r}) = {value}",
        given_step="names the absorption identity and rearranges it to give the unknown coefficient",
    )


def _apply_step(n: int, k: int) -> dict:
    c = math.comb(n, k)
    value = math.comb(n, k + 1)
    check(c * (n - k) % (k + 1) == 0 and c * (n - k) // (k + 1) == value, "apply_step: ratio identity fails")
    ref = [f"{c}*({n}-{k})/{k + 1}", f"{c}*{n - k}/{k + 1}", f"{value}"]
    return dict(
        key=f"apply_step_{n}_{k}",
        tier="apply_exam",
        lead=[
            T("Given that "),
            _binom(str(n), k),
            T(f" = {c}, use "),
            _binom("n", "k+1"),
            T(" ÷ "),
            _binom("n", "k"),
            T(" = "),
            M("(n-k)/(k+1)"),
            T(" to find "),
            _binom(str(n), k + 1),
            T("."),
        ],
        ref=ref,
        wrong_final=f"{value + 5}",
        answer_text=f"C({n}, {k + 1}) = {value}",
        given_step="names the ratio identity and multiplies the given coefficient by it",
    )


def _build_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {k: [] for k in ("easy", "medium", "hard", "exam", "apply_easy", "apply_exam")}
    pools["easy"] += [_pascal("easy", 0, k) for k in (1, 2, 3)]
    pools["easy"] += [_absorb(k) for k in (2, 3, 4)]
    pools["easy"] += [_adjacent_squares(a) for a in (0, 1, 2)]
    pools["medium"] += [_pascal("medium", a, k) for a in (1, 2) for k in (2, 3)]
    pools["medium"] += [_ratio(k) for k in (1, 2, 3)]
    pools["medium"] += [_shift(k) for k in (1, 2, 3)]
    pools["medium"] += [_diagonal(k) for k in (1, 2, 3)]
    pools["hard"] += [_double_pascal(k) for k in (1, 2)]
    pools["hard"] += [_vandermonde(m) for m in (2, 3, 4, 5, 6)]
    pools["hard"] += [_diagonal_ratio(k) for k in (2, 3)]
    pools["exam"] += [_hockey_stick(r) for r in (2, 3, 4)]
    pools["exam"] += [_product(a, b) for a, b in ((1, 2), (2, 1), (2, 2), (3, 1), (1, 3), (3, 2))]
    pools["exam"] += [_factorial_pair(), _central()]
    pools["apply_easy"] += [
        _apply_pascal(n, r)
        for n, r in ((6, 2), (7, 2), (7, 3), (8, 3), (8, 4), (9, 3), (9, 4), (10, 3), (10, 4), (11, 4))
    ]
    pools["apply_exam"] += [_apply_absorb(n, r) for n, r in ((12, 5), (11, 4), (13, 5), (14, 6), (10, 4), (15, 4))]
    pools["apply_exam"] += [_apply_step(n, k) for n, k in ((10, 3), (11, 3), (12, 4), (13, 4), (9, 2), (14, 3))]
    return pools


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_pascal_identity_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_pascal_identity_proof(rng, variant="easy"):
    """`senior-derive-and-use-simple-identities-associated-with-pascal`. Every question carries a marking dict
    (`Question.answer`): the working as a chain of equal expressions, the reference chain, the key steps and the traps."""
    if variant not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"unknown variant: {variant}")
    if variant in ("easy", "exam") and rng.random() < 0.4:
        return kit.apply_item(rng, kit.pick(rng, pools()[f"apply_{variant}"]))
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
