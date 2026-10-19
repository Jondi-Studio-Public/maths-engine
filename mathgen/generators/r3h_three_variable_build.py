"""Generates `r3h_three_variable` questions (easy, medium, hard) and the exam chain `exd_x_r3h_three_variable`.

``three-variable-elimination-triple`` (``linear-systems``, answer ``ordered_list``): solve three linear equations in
three unknowns by elimination and give ``x, y, z`` in that order. Every system is built from its integer solution,
so the answer is exact; it is then re-solved with exact matrix arithmetic and substituted back into all three
equations.

* easy: two of the equations contain only two of the unknowns (the third contains all three), so the first
  eliminations are short.
* medium: all three equations contain all three unknowns, and every elimination needs at least one equation scaled.
* hard: the unknowns appear on both sides of some equations, with larger and negative coefficients, so the
  equations have to be rearranged before they can be eliminated.
* exam (``exd_x_r3h_three_variable``): (a) the equation left by adding equations (1) and (2), which removes ``z``
  (given as the coefficients of ``x`` and ``y`` and the constant), (b) ``x``, (c) ``y`` and (d) ``z`` (each following
  through from the earlier answers).
"""

from __future__ import annotations

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, pick_distractors
from ._r3h_kit import build_with_redraws, need, number_part, prompt_part, say_marked

_R = sympy.Rational
_VARS = ("x", "y", "z")


def _lin(*terms) -> str:
    """``(coefficient, name)`` pairs written as ``3x - y``: zero terms dropped, no ``+ -``."""
    out = ""
    for coef, name in terms:
        if coef == 0:
            continue
        body = name if abs(coef) == 1 else f"{abs(coef)}{name}"
        out += (f"-{body}" if coef < 0 else body) if not out else f" {'-' if coef < 0 else '+'} {body}"
    return out or "0"


def _solve(rows, consts):
    """The exact solution of M v = c (a list of Rationals), or None when M is singular."""
    m = sympy.Matrix(rows)
    if m.det() == 0:
        return None
    return list(m.LUsolve(sympy.Matrix(consts)))


def _equation(row, const) -> str:
    return f"{join_terms(list(zip(row, _VARS)))} = {const}"


def _rearranged(row, const, move: int) -> str:
    """Equation text with the term in column `move` carried across the equals sign."""
    left = [(a, v) for j, (a, v) in enumerate(zip(row, _VARS)) if j != move]
    right = [(-row[move], _VARS[move]), (const, "")]
    return f"{join_terms(left)} = {join_terms(right)}"


def _draw_solution(rng, positive=False):
    for _ in attempts(200, "a solution with distinct, non-zero values"):
        sol = [rng.randint(1, 9) for _ in range(3)] if positive else [rng.randint(-5, 7) for _ in range(3)]
        if 0 not in sol and len(set(sol)) == 3:
            return sol
    raise AssertionError("unreachable")


def _draw_system(rng, tier):
    for _ in attempts(3000, "a three-variable system with small coefficients"):
        sol = _draw_solution(rng)
        if tier == "easy":
            rows = [
                [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(3)],
                [0] + [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(2)],
                [0] + [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(2)],
            ]
            order = rng.sample(range(3), 3)
            rows = [rows[i] for i in order]
        elif tier == "medium":
            rows = [[rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(3)] for _ in range(3)]
        else:
            rows = [[rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]) for _ in range(3)] for _ in range(3)]
        consts = [sum(a * s for a, s in zip(r, sol)) for r in rows]
        found = _solve(rows, consts)
        if found is None or found != sol:
            continue
        det = abs(sympy.Matrix(rows).det())
        if det < 3 or det > (40 if tier != "hard" else 90):
            continue
        # no two equations may be multiples of each other, and no constant should be a giveaway zero
        if 0 in consts or max(abs(c) for c in consts) > 60:
            continue
        if tier == "medium" and any(0 in r for r in rows):
            continue
        return rows, consts, sol
    raise AssertionError("unreachable")


def _list_options(answer, wrongs, n=4):
    """Tagged distractors for an ordered list of integers."""
    text = ", ".join(str(v) for v in answer)
    seen = {tuple(answer)}
    out = []
    for tag, w in wrongs:
        w = tuple(int(v) for v in w)
        if w in seen:
            continue
        seen.add(w)
        out.append(Distractor(", ".join(str(v) for v in w), tag, list(w)))
    return pick_distractors(text, out, n)


def _triple_wrongs(rows, consts, sol):
    x, y, z = sol
    out = [
        ("r3h-elim-values-swapped", (y, x, z)),
        ("r3h-elim-values-swapped", (x, z, y)),
        ("r3h-elim-sign-of-last-variable", (x, y, -z)),
        ("r3h-elim-sign-of-last-variable", (x, -y, z)),
    ]
    # one constant read with the wrong sign, one coefficient with the wrong sign: re-solve and keep integer results
    for i in range(3):
        c2 = list(consts)
        c2[i] = -c2[i]
        found = _solve(rows, c2)
        if found and all(v.is_Integer for v in found):
            out.append(("r3h-elim-constant-sign-slip", tuple(int(v) for v in found)))
    for i in range(3):
        for j in range(3):
            r2 = [list(r) for r in rows]
            r2[i][j] = -r2[i][j]
            found = _solve(r2, consts)
            if found and all(v.is_Integer for v in found):
                out.append(("r3h-elim-coefficient-sign-slip", tuple(int(v) for v in found)))
    return out


def _stem(rng, rows, consts, tier) -> str:
    order = "Give your answer as x, y, z in that order."
    if tier == "hard":
        moves = [rng.randrange(3) for _ in rows]
        lines = [_rearranged(r, c, m) for r, c, m in zip(rows, consts, moves)]
    else:
        lines = [_equation(r, c) for r, c in zip(rows, consts)]
    body = "; ".join(f"({i + 1}) {line}" for i, line in enumerate(lines))
    lead = rng.choice(
        [
            "Solve the system of equations",
            "Use elimination to solve the system",
            "Find the values of x, y and z that satisfy all three equations",
        ]
    )
    return f"{lead}: {body}. {order}"


@generator(
    "r3h_three_variable",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3h_three_variable(rng, tier="easy"):
    for _ in attempts(60, "a system with enough distractors"):
        rows, consts, sol = _draw_system(rng, tier)
        options = _list_options(sol, _triple_wrongs(rows, consts, sol))
        if len(options) >= 2:
            break
    # the answer satisfies every equation as printed
    for r, c in zip(rows, consts):
        check(sum(a * s for a, s in zip(r, sol)) == c, "the solution misses an equation")
    text = _stem(rng, rows, consts, tier)
    return dict(
        text=text,
        answer=list(sol),
        answer_text=", ".join(str(v) for v in sol),
        answer_kind="ordered_list",
        distractors=options,
        working=[
            f"Eliminate one unknown from two pairs of equations, solve the pair for the other two, then back-substitute: x = {sol[0]}, y = {sol[1]}, z = {sol[2]}.",
            "Check: " + "; ".join(f"{sum(a * s for a, s in zip(r, sol))} = {c}" for r, c in zip(rows, consts)) + ".",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_CONTEXTS = (
    "The masses, in kilograms, of the three types of crate in a shipment are x, y and z, and they satisfy",
    "The prices, in dollars, of an adult ticket, a child ticket and a concession ticket are x, y and z, and they satisfy",
    "The concentrations, in grams per litre, of three solutions are x, y and z, and they satisfy",
)


def _draw_exam(rng):
    for _ in attempts(4000, "an exam system where equations (1) and (2) cancel z"):
        sol = _draw_solution(rng, positive=True)
        c = rng.choice([1, 2, 3, 4])
        rows = [
            [rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]), rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]), c],
            [rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]), rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]), -c],
            [rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(3)],
        ]
        consts = [sum(a * s for a, s in zip(r, sol)) for r in rows]
        found = _solve(rows, consts)
        if found != sol or abs(sympy.Matrix(rows).det()) < 3:
            continue
        p, q = rows[0][0] + rows[1][0], rows[0][1] + rows[1][1]
        if 0 in (p, q) or 0 in consts or max(abs(v) for v in consts) > 60:
            continue
        # the second elimination must not be as easy as the first: equation (3) has no zero entry
        if 0 in rows[2]:
            continue
        return rows, consts, sol
    raise AssertionError("unreachable")


def _exam_once(rng):
    rows, consts, sol = _draw_exam(rng)
    x, y, z = sol
    (a1, b1, c1), (a2, b2, c2), (a3, b3, c3) = rows
    k1, k2, k3 = consts
    p, q, r = a1 + a2, b1 + b2, k1 + k2
    check(c1 + c2 == 0, "adding equations (1) and (2) must remove z")
    check(p * x + q * y == r, "the added equation fails the solution")
    lines = [_equation(rw, kc) for rw, kc in zip(rows, consts)]
    context = rng.choice(_CONTEXTS)
    stem, latex = say_marked(f"{context} the system: «(1) {lines[0]}», «(2) {lines[1]}», «(3) {lines[2]}».")
    a_list = {
        "answer": [p, q, r],
        "text": f"{p}, {q}, {r}",
    }
    a_wrongs = [
        ("r3h-elim-subtracted-instead-of-added", (a1 - a2, b1 - b2, k1 - k2)),
        ("r3h-elim-constants-not-combined", (p, q, k1)),
        ("r3h-elim-coefficients-swapped", (q, p, r)),
        ("r3h-elim-constant-sign-slip", (p, q, -r)),
        ("r3h-elim-sign-of-last-variable", (-p, -q, r)),
    ]
    a_options = _list_options(a_list["answer"], a_wrongs)
    need(a_options)
    # x: wrong from a coefficient or constant slip, or a sign slip
    x_wrongs = []
    for i in range(3):
        c2s = list(consts)
        c2s[i] = -c2s[i]
        f = _solve(rows, c2s)
        if f and f[0].is_Integer:
            x_wrongs.append(("r3h-elim-constant-sign-slip", int(f[0])))
    for i in range(3):
        for j in range(3):
            r2 = [list(rw) for rw in rows]
            r2[i][j] = -r2[i][j]
            f = _solve(r2, consts)
            if f and f[0].is_Integer:
                x_wrongs.append(("r3h-elim-coefficient-sign-slip", int(f[0])))
    x_wrongs += [("r3h-elim-sign-of-last-variable", -x), ("r3h-elim-values-swapped", y), ("r3h-elim-values-swapped", z)]
    # y follows from (a): y = (r - p x)/q ; z from (1): z = (k1 - a1 x - b1 y)/c1
    y_ft = f"({r} - ({p})*part_b)/({q})"
    z_ft = f"({k1} - ({a1})*part_b - ({b1})*part_c)/({c1})"
    y_wrongs = [
        ("r3h-elim-sign-of-last-variable", -y),
        ("r3h-elim-back-substitution-sign", _R(r + p * x, q)),
        ("r3h-elim-back-substitution-sign", _R(r - p * x, -q)),
        ("r3h-elim-values-swapped", x),
        ("r3h-elim-values-swapped", z),
    ]
    z_wrongs = [
        ("r3h-elim-sign-of-last-variable", -z),
        ("r3h-elim-back-substitution-sign", _R(k1 + a1 * x + b1 * y, c1)),
        ("r3h-elim-back-substitution-sign", _R(k1 - a1 * x + b1 * y, c1)),
        ("r3h-elim-values-swapped", x),
        ("r3h-elim-values-swapped", y),
    ]
    x_wrongs, y_wrongs, z_wrongs = ([(t, v) for t, v in w if abs(v) <= 25] for w in (x_wrongs, y_wrongs, z_wrongs))
    parts = [
        dict(
            **prompt_part(
                "Add equation (1) and equation (2), which removes «z». Give the resulting equation as three numbers: the coefficient of «x», the coefficient of «y» and the constant, in that order."
            ),
            answer_kind="ordered_list",
            answer=a_list["answer"],
            answer_text=a_list["text"],
            marks=2,
            working=[f"Adding the two equations term by term gives {_lin((p, 'x'), (q, 'y'))} = {r}."],
            distractors=a_options,
        ),
        number_part(
            "Use equation (3) as well to find «x».",
            x,
            0,
            3,
            [f"Eliminate z between (3) and (1) or (2), then solve with {_lin((p, 'x'), (q, 'y'))} = {r}: x = {x}."],
            x_wrongs,
            gap=1,
        ),
        number_part(
            "Hence find «y».",
            y,
            0,
            1,
            [f"{_lin((q, 'y'))} = {r} - ({p})({x}), so y = {y}."],
            y_wrongs,
            follow_through=y_ft,
            gap=1,
        ),
        number_part(
            "Find «z».",
            z,
            0,
            1,
            [f"Substitute into (1): {_lin((c1, 'z'))} = {k1} - ({a1})({x}) - ({b1})({y}), so z = {z}."],
            z_wrongs,
            follow_through=z_ft,
            gap=1,
        ),
    ]
    return dict(text=stem, latex=latex, parts=parts)


@generator("exd_x_r3h_three_variable", level=14, presets={"exam": dict(tier="exam")})
def exd_x_r3h_three_variable(rng, tier="exam"):
    """Exam tier: eliminate z from two equations, then find x, y and z."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_with_redraws(rng, _exam_once)
