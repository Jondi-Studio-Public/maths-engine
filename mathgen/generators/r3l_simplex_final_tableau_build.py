"""Generates `uc2-simplex-final-tableau-reading` questions (easy, medium, hard) and the exam chain
`exd_x_r3l_simplex_final_tableau`.

``uc2-simplex-final-tableau-reading`` (``optimisation``, answer ``fraction``): the final (optimal) simplex tableau of a
maximisation problem max z = c.x subject to A x <= b, x >= 0 is supplied, with the basic variable of each row named.
The learner reads off what the tableau says and uses its slack columns for sensitivity.

* easy: the value of one named variable at the optimum (a basic variable is its row's right-hand side; a non-basic
  variable, decision or slack, is 0).
* medium: the shadow price of a constraint: the entry of the objective row under that constraint's slack variable
  (0 for a constraint that is not binding).
* hard: either the largest increase of a binding resource b_k that keeps the same basis (the minimum of
  RHS / (-entry) over rows with a negative entry in the slack column), or the new maximum of z after a stated increase
  of b_k within that range.
* exam (``exd_x_r3l_simplex_final_tableau``): (a) the values of the decision variables, (b) the shadow price of a
  binding constraint, (c) the largest increase of its right-hand side, (d) the new maximum for a stated increase
  (follows through from (b)).

The instance is a random linear programme solved with the fixed pivot rule in `_r3l_kit`; only runs with an integral
optimum, non-degenerate and with half-integer shadow prices are kept. The sensitivity answers are re-derived by
re-solving the programme from the original data with b_k changed (the set of basic variables is unchanged just inside
the range and changes just outside it; the new optimum equals z + y_k delta).
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import VerificationError, attempts, check, generator, pick_distractors
from ._r3l_kit import basis_of, draw_lp, initial_tableau, num, run_simplex, sub
from ._ucv1_linalg import mat_str

_CONTEXTS = (
    "A Tasmanian furniture maker maximises weekly profit z (in hundreds of dollars)",
    "A Melbourne bakery chain maximises daily profit z (in hundreds of dollars)",
    "A Perth workshop maximises monthly profit z (in thousands of dollars)",
)


def _instance(rng, n, m, *, need_binding=True):
    """(a, b, c, final tableau, basis, binding k) for a nondegenerate run with an integral optimum."""
    for _ in attempts(400, "a readable final tableau"):
        a, b, c, tabs, _pivots = draw_lp(rng, n, m)
        final = tabs[-1]
        if any(final[i][-1].denominator != 1 for i in range(m)):
            continue
        if any(p.denominator not in (1, 2) for p in final[-1][n : n + m]):
            continue
        basis = basis_of(final, n + m)
        binding = [k for k in range(m) if (n + k) not in basis]
        if not need_binding:
            return a, b, c, final, basis, None
        ok = [k for k in binding if (_allowable(final, n, k)[0] or 0) >= 2]
        if ok:
            return a, b, c, final, basis, rng.choice(ok)
    raise AssertionError  # pragma: no cover


def _allowable(final, n, k):
    """(largest increase, largest decrease) of b_k that keeps the basis; None for no limit."""
    col = n + k
    up = [final[i][-1] / -final[i][col] for i in range(len(final) - 1) if final[i][col] < 0]
    down = [final[i][-1] / final[i][col] for i in range(len(final) - 1) if final[i][col] > 0]
    return (min(up) if up else None, min(down) if down else None)


def _resolve(a, b, c, k, delta):
    """Re-solve the original programme with b_k + delta: (final tableau, basis)."""
    b2 = [Fraction(v) for v in b]
    b2[k] += delta
    tabs, _ = run_simplex(initial_tableau(a, b2, c), limit=20)
    return tabs[-1], basis_of(tabs[-1], len(c) + len(b))


def _verify_sensitivity(a, b, c, basis, k, price, allow, z0):
    eps = allow / 1000
    inside, inside_basis = _resolve(a, b, c, k, allow - eps)
    check(set(inside_basis) == set(basis), "basis changes inside the allowable range")
    check(inside[-1][-1] == z0 + price * (allow - eps), "z does not move at the shadow price inside the range")
    _outside, outside_basis = _resolve(a, b, c, k, allow + eps)
    check(set(outside_basis) != set(basis), "basis unchanged outside the allowable range")


def _names(n, m):
    return [sub("x", j + 1) for j in range(n)] + [sub("s", i + 1) for i in range(m)]


def _range_candidates(final, n, m, col, allow):
    """Wrong allowable-increase values: the decrease ratios, the other increase ratios, ratios from decision columns."""
    cands = [
        Distractor(num(w), "r3l-tableau-range-sign-reversed", exact_value(w))
        for w in sorted(final[i][-1] / final[i][col] for i in range(m) if final[i][col] > 0)
    ]
    for i in range(m):
        if final[i][col] < 0 and final[i][-1] / -final[i][col] != allow:
            w = final[i][-1] / -final[i][col]
            cands.append(Distractor(num(w), "r3l-tableau-range-not-minimum", exact_value(w)))
    for i in range(m):
        for j in range(n):
            if final[i][j] < 0:
                w = final[i][-1] / -final[i][j]
                cands.append(Distractor(num(w), "r3l-tableau-range-wrong-column", exact_value(w)))
    low = min(final[i][-1] for i in range(m))
    cands.append(Distractor(num(low), "r3l-tableau-objective-for-variable", exact_value(low)))
    cands.append(Distractor(num(final[-1][-1]), "r3l-tableau-objective-for-variable", exact_value(final[-1][-1])))
    for w in (allow + 1, allow * 2, allow - 1):
        if w > 0:
            cands.append(Distractor(num(w), None, exact_value(w)))
    return cands


def _tableau_stem(rng, n, m, final, basis):
    names = _names(n, m)
    rows = ", ".join(names[b] for b in basis)
    slacks = ", ".join(names[n:])
    return (
        f"{rng.choice(_CONTEXTS)} subject to {m} resource limits; the slack variables {slacks} are the unused amounts of "
        f"resources 1 to {m}. The final simplex tableau is {mat_str(final)}, with columns {', '.join(names)}, RHS; the "
        f"last row is the objective row and the basic variables, row by row, are {rows}."
    )


def _easy(rng):
    n, m = rng.choice([(2, 3), (3, 3), (2, 2)])
    _a, _b, _c, final, basis, _k = _instance(rng, n, m, need_binding=False)
    names = _names(n, m)
    nonbasic = [j for j in range(n + m) if j not in basis]
    v = rng.choice(nonbasic) if rng.random() < 0.4 else rng.choice(basis)
    value = final[basis.index(v)][-1] if v in basis else Fraction(0)
    cands = []
    if v in basis:
        i = basis.index(v)
        for r in range(m):
            if r != i:
                cands.append(Distractor(num(final[r][-1]), "r3l-tableau-wrong-row-rhs", exact_value(final[r][-1])))
    else:
        cands.append(Distractor(num(final[-1][v]), "r3l-tableau-objective-entry-as-value", exact_value(final[-1][v])))
        for r in range(m):
            cands.append(Distractor(num(final[r][-1]), "r3l-tableau-wrong-row-rhs", exact_value(final[r][-1])))
    cands.append(Distractor(num(final[-1][-1]), "r3l-tableau-objective-for-variable", exact_value(final[-1][-1])))
    for j in range(n + m):
        if final[-1][j] != 0:
            cands.append(
                Distractor(num(final[-1][j]), "r3l-tableau-objective-entry-as-value", exact_value(final[-1][j]))
            )
    distractors = pick_distractors(num(value), cands, 4)
    check(len(distractors) >= 3, "too few distractors")
    stem = _tableau_stem(rng, n, m, final, basis)
    why = (
        f"{names[v]} is basic in row {basis.index(v) + 1}, so its value is that row's RHS, {num(value)}."
        if v in basis
        else f"{names[v]} is non-basic, so its value is 0."
    )
    return dict(
        text=f"{stem} What is the value of {names[v]} in the optimal solution?",
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=num(value),
        working=[why],
        distractors=distractors,
    )


def _medium(rng):
    n, m = rng.choice([(2, 3), (3, 3), (2, 2)])
    for _ in attempts(100, "a constraint with the wanted status"):
        a, b, c, final, basis, _k = _instance(rng, n, m, need_binding=False)
        binding = [k for k in range(m) if (n + k) not in basis]
        if binding or rng.random() < 0.3:
            break
    k = rng.choice(binding) if binding and rng.random() < 0.8 else rng.randrange(m)
    price = final[-1][n + k]
    names = _names(n, m)
    cands = []
    if (n + k) in basis:
        slack = final[basis.index(n + k)][-1]
        cands.append(Distractor(num(slack), "r3l-tableau-shadow-price-slack-value", exact_value(slack)))
    cands.append(Distractor(num(final[-1][-1]), "r3l-tableau-objective-for-variable", exact_value(final[-1][-1])))
    for j in range(n + m):
        if j != n + k:
            cands.append(
                Distractor(num(final[-1][j]), "r3l-tableau-shadow-price-wrong-column", exact_value(final[-1][j]))
            )
    cands.append(Distractor(num(-price), "r3l-tableau-shadow-price-sign", exact_value(-price)))
    distractors = pick_distractors(num(price), cands, 4)
    check(len(distractors) >= 3, "too few distractors")
    stem = _tableau_stem(rng, n, m, final, basis)
    # independent route: re-solve with a small extra amount of resource k (a step inside the range)
    step = Fraction(1, 100)
    allow = _allowable(final, n, k)[0]
    if allow is None or allow > step:
        re, _ = _resolve(a, b, c, k, step)
        check(re[-1][-1] - final[-1][-1] == price * step, "the shadow price differs from re-solving")
    return dict(
        text=f"{stem} By how much does the maximum value of z rise for each extra unit of resource {k + 1}, as long as the same variables stay basic?",
        answer_kind="fraction",
        answer=exact_value(price),
        answer_text=num(price),
        working=[
            f"The shadow price of resource {k + 1} is the objective-row entry under {names[n + k]}: {num(price)}."
            + (" The constraint is not binding, so extra resource is worth nothing." if price == 0 else "")
        ],
        distractors=distractors,
    )


def _hard(rng):
    n, m = rng.choice([(2, 3), (3, 3), (2, 3)])
    a, b, c, final, basis, k = _instance(rng, n, m)
    allow, _down = _allowable(final, n, k)
    price = final[-1][n + k]
    z0 = final[-1][-1]
    _verify_sensitivity(a, b, c, basis, k, price, allow, z0)
    names = _names(n, m)
    stem = _tableau_stem(rng, n, m, final, basis)
    col = n + k
    if rng.random() < 0.5:
        value = allow
        distractors = pick_distractors(num(value), _range_candidates(final, n, m, col, allow), 4)
        text = f"{stem} By how much can the supply of resource {k + 1} be increased before the set of basic variables changes?"
        working = [
            f"Under {names[col]} the rows with a negative entry limit an increase: RHS / (−entry) for each. The smallest is {num(value)}."
        ]
    else:
        delta = Fraction(rng.randint(1, max(1, int(allow) - 1)))
        check(delta < allow, "delta must be inside the allowable range")
        value = z0 + price * delta
        re, re_basis = _resolve(a, b, c, k, delta)
        check(set(re_basis) == set(basis) and re[-1][-1] == value, "re-solving disagrees with z0 + price * delta")
        cands = [
            Distractor(num(z0 + delta), "r3l-tableau-price-ignored", exact_value(z0 + delta)),
            Distractor(num(z0 - price * delta), "r3l-tableau-shadow-price-sign", exact_value(z0 - price * delta)),
            Distractor(num(price * delta), "r3l-tableau-increase-not-added", exact_value(price * delta)),
        ]
        for j in range(n + m):
            if j != col and final[-1][j] not in (0, price):
                w = z0 + final[-1][j] * delta
                cands.append(Distractor(num(w), "r3l-tableau-shadow-price-wrong-column", exact_value(w)))
        cands += [
            Distractor(num(z0 + price * (delta + 1)), None, exact_value(z0 + price * (delta + 1))),
            Distractor(num(z0 + (price + 1) * delta), None, exact_value(z0 + (price + 1) * delta)),
        ]
        distractors = pick_distractors(num(value), cands, 4)
        text = (
            f"{stem} Resource {k + 1} is increased by {int(delta)} unit{'s' if delta != 1 else ''}, which is small enough for "
            "the same variables to stay basic. What is the new maximum value of z?"
        )
        working = [
            f"The shadow price is {num(price)}, so z rises by {num(price)}·{int(delta)}: {num(z0)} + {num(price * delta)} = {num(value)}."
        ]
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=num(value),
        working=working,
        distractors=distractors,
    )


@generator(
    "r3l_simplex_final_tableau",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3l_simplex_final_tableau(rng, tier="easy"):
    """Read an optimal simplex tableau; the tiers are described in the module docstring."""
    if tier in ("easy", "medium"):
        # A tableau whose entries repeat can leave fewer than three distinct wrong values: redraw it.
        for _ in attempts(60, "a tableau that leaves three usable distractors"):
            try:
                return (_easy if tier == "easy" else _medium)(rng)
            except VerificationError as err:
                if "too few distractors" not in str(err):
                    raise
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------------


@generator("exd_x_r3l_simplex_final_tableau", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r3l_simplex_final_tableau(rng, tier="exam"):
    """Exam chain: decision variables, shadow price, allowable increase, new optimum (follows through)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n, m = rng.choice([(2, 3), (3, 3)])
    a, b, c, final, basis, k = _instance(rng, n, m)
    allow, _down = _allowable(final, n, k)
    price = final[-1][n + k]
    z0 = final[-1][-1]
    _verify_sensitivity(a, b, c, basis, k, price, allow, z0)
    check(price > 0, "a binding constraint has a positive price")
    delta = Fraction(rng.randint(1, max(1, int(allow) - 1)))
    check(delta < allow, "delta inside the range")
    value = z0 + price * delta
    re, re_basis = _resolve(a, b, c, k, delta)
    check(set(re_basis) == set(basis) and re[-1][-1] == value, "re-solving disagrees")
    names = _names(n, m)
    xs = [int(final[basis.index(j)][-1]) if j in basis else 0 for j in range(n)]
    col = n + k
    x_text = ", ".join(str(v) for v in xs)
    stem = _tableau_stem(rng, n, m, final, basis)

    def vec(values, tag):
        return Distractor(", ".join(map(str, values)), tag, list(values))

    a_cands = []
    rhs_rows = [int(final[i][-1]) for i in range(m)][:n]
    a_cands.append(vec(rhs_rows, "r3l-tableau-wrong-row-rhs"))
    a_cands.append(vec(list(reversed(xs)), "r3l-tableau-wrong-row-rhs"))
    obj_entries = [final[-1][j] for j in range(n)]
    if all(v.denominator == 1 for v in obj_entries):
        a_cands.append(vec([int(v) for v in obj_entries], "r3l-tableau-objective-entry-as-value"))
    a_cands.append(vec([v + (1 if v else 0) for v in xs], "r3l-tableau-wrong-row-rhs"))
    a_cands.append(vec([xs[0] + xs[-1]] + xs[1:], "r3l-tableau-wrong-row-rhs"))
    a_dist = pick_distractors(x_text, a_cands, 3, unordered=False)
    b_cands = [Distractor(num(z0), "r3l-tableau-objective-for-variable", exact_value(z0))]
    b_cands += [
        Distractor(num(final[-1][j]), "r3l-tableau-shadow-price-wrong-column", exact_value(final[-1][j]))
        for j in range(n + m)
        if j != col
    ]
    b_dist = pick_distractors(num(price), b_cands, 3)
    c_dist = pick_distractors(num(allow), _range_candidates(final, n, m, col, allow), 3)
    d_cands = [
        Distractor(num(z0 + delta), "r3l-tableau-price-ignored", exact_value(z0 + delta)),
        Distractor(num(price * delta), "r3l-tableau-increase-not-added", exact_value(price * delta)),
        Distractor(num(z0 - price * delta), "r3l-tableau-shadow-price-sign", exact_value(z0 - price * delta)),
    ]
    d_cands += [
        Distractor(num(z0 + price * (delta + 1)), None, exact_value(z0 + price * (delta + 1))),
        Distractor(num(z0 + (price + 1) * delta), None, exact_value(z0 + (price + 1) * delta)),
    ]
    d_dist = pick_distractors(num(value), d_cands, 3)
    check(
        all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)),
        f"too few distractors {[len(x) for x in (a_dist, b_dist, c_dist, d_dist)]}",
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Write the optimal values of {', '.join(names[:n])}, in that order.",
                answer_kind="ordered_list",
                answer=xs,
                answer_text=x_text,
                marks=2,
                working=[f"A basic variable equals its row's RHS and a non-basic one is 0, giving {x_text}."],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Find the shadow price of resource {k + 1}.",
                answer_kind="fraction",
                answer=exact_value(price),
                answer_text=num(price),
                marks=1,
                sensible_range=(0, None),
                working=[f"It is the objective-row entry under {names[col]}: {num(price)}."],
                distractors=b_dist,
            ),
            dict(
                prompt=f"By how much can the supply of resource {k + 1} be increased before the set of basic variables changes?",
                answer_kind="fraction",
                answer=exact_value(allow),
                answer_text=num(allow),
                marks=3,
                sensible_range=(0, None),
                working=[
                    f"The smallest RHS / (−entry) over rows with a negative entry under {names[col]} is {num(allow)}."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Resource {k + 1} is increased by {int(delta)} unit{'s' if delta != 1 else ''}. Find the new maximum value of z.",
                answer_kind="fraction",
                answer=exact_value(value),
                answer_text=num(value),
                marks=2,
                follow_through=f"{num(z0)} + {int(delta)}*part_b",
                sensible_range=(0, None),
                working=[f"z rises by {num(price)}·{int(delta)}, from {num(z0)} to {num(value)}."],
                distractors=d_dist,
            ),
        ],
    )
