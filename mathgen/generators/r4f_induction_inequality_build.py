"""Generates `r4f_induction_inequality` questions and the exam chain `exd_x_r4f_induction_inequality`.

Skill `proof-by-induction-inequality-or-recurrence` (``proof-and-foundations``): the typed steps of an induction
proof of an inequality such as 2^n > n² (true for every n >= 5), or of a closed form for a recurrence.

An inequality statement here is P(n): L(n) > R(n) with L(n) = a^n or n! and R(n) one of c·n^p, c·n + e or b^n. When
L(k + 1) = m(k) L(k) (m = a for a^n, m = k + 1 for n!), the inductive step is

    L(k + 1) = m(k) L(k) > m(k) R(k) >= R(k + 1),

so the *key inequality* is m(k) R(k) >= R(k + 1). Two thresholds matter and they are different numbers:

* n0 -- the smallest n0 such that P(n) holds for every n >= n0 (the base case; P is often true at n = 1 and false
  at the next few values, so n0 is not "the first n where it is true");
* k0 -- the smallest k0 such that the key inequality holds for every k >= k0 (what the step needs).

A statement is used only when k0 <= n0 (so the proof is valid from n0), both are brute-forced to n = 400, and the
margins at n0 are positive.

Tiers (topic ``r4f_induction_inequality``):

* easy   -- n0, from trying small values;
* medium -- k0, the smallest k from which the key inequality holds for every k;
* hard   -- a recurrence u(n + 1) = a u(n) + c1 n + c0 with u(1) given and the claim u(n) = A a^n + p n + r: the
  constant r that makes the inductive step close (p is found on the way), or the integer A fixed by the base case.

The exam chain (two shapes) is four number parts. Inequality shape: n0; L(n0) - R(n0) (follow-through from n0); k0;
the key-inequality margin m(n0) R(n0) - R(n0 + 1) (follow-through from n0). Recurrence shape: u(3); p; r (follow-through
from p); A (follow-through from p and r).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from functools import lru_cache

from ..answers import Distractor
from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4f_ft import follow_through_clean
from .exd_x_r3c_parts import number_part

_SUP = {2: "²", 3: "³", 4: "⁴", 5: "⁵"}
_LIMIT = 400


class _Ineq:
    """P(n): L(n) > R(n)."""

    def __init__(self, lhs, rhs):
        self.lhs = lhs  # ("pow", a) or ("fact",)
        self.rhs = rhs  # ("mono", c, p) | ("lin", c, e) | ("exp", b)

    # ---- values -----------------------------------------------------------------------------------------------
    def L(self, n: int) -> int:
        return self.lhs[1] ** n if self.lhs[0] == "pow" else math.factorial(n)

    def R(self, n: int) -> int:
        kind = self.rhs[0]
        if kind == "mono":
            return self.rhs[1] * n ** self.rhs[2]
        if kind == "lin":
            return self.rhs[1] * n + self.rhs[2]
        return self.rhs[1] ** n

    def mult(self, k: int) -> int:
        """L(k + 1) / L(k)."""
        return self.lhs[1] if self.lhs[0] == "pow" else k + 1

    def holds(self, n: int) -> bool:
        return self.L(n) > self.R(n)

    def step_ok(self, k: int) -> bool:
        return self.mult(k) * self.R(k) >= self.R(k + 1)

    # ---- thresholds, brute force --------------------------------------------------------------------------------
    def n0(self) -> int:
        last_fail = max((n for n in range(1, _LIMIT + 1) if not self.holds(n)), default=0)
        check(last_fail < _LIMIT - 5, "the inequality is not eventually true")
        return last_fail + 1

    def k0(self) -> int:
        last_fail = max((k for k in range(1, _LIMIT + 1) if not self.step_ok(k)), default=0)
        check(last_fail < _LIMIT - 5, "the key inequality is not eventually true")
        return last_fail + 1

    # ---- text -------------------------------------------------------------------------------------------------
    def lhs_text(self, var="n") -> str:
        return f"{self.lhs[1]}^{var}" if self.lhs[0] == "pow" else f"{var}!"

    def rhs_text(self, var="n") -> str:
        kind = self.rhs[0]
        wrap = var  # "(k + 1)" already carries its brackets
        if kind == "mono":
            _, c, p = self.rhs
            return f"{'' if c == 1 else c}{wrap}{_SUP[p]}"
        if kind == "lin":
            _, c, e = self.rhs
            return f"{c}{wrap}" + (f" + {e}" if e else "")
        return f"{self.rhs[1]}^{var}"

    def statement(self) -> str:
        return f"{self.lhs_text()} > {self.rhs_text()}"

    def key_text(self) -> str:
        """The key inequality in k: m(k) R(k) >= R(k + 1)."""
        mult = str(self.lhs[1]) if self.lhs[0] == "pow" else "(k + 1)"
        rhs_k = self.rhs_text("k")
        if self.rhs[0] == "lin" and self.rhs[2]:
            rhs_k = f"({rhs_k})"
        return f"{mult} × {rhs_k} ≥ {self.rhs_text('(k + 1)')}"

    # ---- follow-through strings, in the placeholder x --------------------------------------------------------------
    def L_expr(self, x: str) -> str:
        return f"{self.lhs[1]}**({x})" if self.lhs[0] == "pow" else f"factorial({x})"

    def R_expr(self, x: str) -> str:
        kind = self.rhs[0]
        if kind == "mono":
            return f"{self.rhs[1]}*({x})**{self.rhs[2]}"
        if kind == "lin":
            return f"({self.rhs[1]}*({x}) + {self.rhs[2]})"
        return f"{self.rhs[1]}**({x})"

    def mult_expr(self, x: str) -> str:
        return str(self.lhs[1]) if self.lhs[0] == "pow" else f"(({x}) + 1)"

    def verify(self) -> None:
        n0, k0 = self.n0(), self.k0()
        check(not self.holds(n0 - 1) or n0 == 1, "P holds just below n0")
        for n in range(n0, n0 + 60):
            check(self.holds(n), "P(n) fails after n0")
        for k in range(k0, k0 + 60):
            check(self.step_ok(k), "the key inequality fails after k0")
        # the step really transfers truth: every k >= max(n0, k0) in the sampled range
        for k in range(max(n0, k0), max(n0, k0) + 40):
            check(self.L(k + 1) == self.mult(k) * self.L(k), "L(k + 1) is not m(k) L(k)")


@lru_cache(maxsize=1)
def _statements() -> tuple:
    out = []
    lhs_options = [("pow", 2), ("pow", 3), ("pow", 4), ("pow", 5), ("fact",)]
    for lhs in lhs_options:
        rhs_options = []
        for c in (1, 2, 3, 4, 5):
            for p in (2, 3, 4, 5):
                rhs_options.append(("mono", c, p))
        for c in (2, 3, 4, 5, 6):
            for e in (0, 1, 2, 3, 5):
                rhs_options.append(("lin", c, e))
        for b in (2, 3, 4, 5):
            if lhs[0] == "fact" or b > lhs[1]:
                rhs_options.append(("exp", b))
        for rhs in rhs_options:
            if lhs[0] == "pow" and rhs[0] == "exp":
                continue  # a^n > b^n for b > a is false
            stmt = _Ineq(lhs, rhs)
            try:
                n0, k0 = stmt.n0(), stmt.k0()
            except Exception:  # noqa: BLE001 - a statement that is never true is simply not offered
                continue
            if 3 <= n0 <= 20 and k0 <= n0 and k0 >= 2 and stmt.L(n0) - stmt.R(n0) > 0:
                if stmt.mult(n0) * stmt.R(n0) - stmt.R(n0 + 1) > 0 and n0 != k0:
                    out.append((lhs, rhs))
    return tuple(out)


def _draw_ineq(rng: random.Random) -> _Ineq:
    lhs, rhs = rng.choice(_statements())
    stmt = _Ineq(lhs, rhs)
    stmt.verify()
    return stmt


def _mult_text(stmt: _Ineq) -> str:
    return str(stmt.lhs[1]) if stmt.lhs[0] == "pow" else "(k + 1)"


def _step_stem(stmt: _Ineq) -> str:
    rhs_k = stmt.rhs_text("k")
    if stmt.rhs[0] == "lin" and stmt.rhs[2]:
        rhs_k = f"({rhs_k})"
    return (
        f"P(n): {stmt.statement()}. In the inductive step P(k) is assumed true, so "
        f"{stmt.lhs_text('(k + 1)')} = {_mult_text(stmt)} × {stmt.lhs_text('k')} > {_mult_text(stmt)} × {rhs_k}."
    )


def _int_dists(answer: int, cands) -> list:
    out, seen = [], {answer}
    for value, tag in cands:
        if value in seen or value < 1:
            continue
        seen.add(value)
        out.append(Distractor(str(int(value)), tag, int(value)))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Recurrences u(n + 1) = a u(n) + c1 n + c0 with u(n) = A a^n + p n + r


class _Rec:
    def __init__(self, a, A, p, r):
        self.a, self.A, self.p, self.r = a, A, p, r
        self.c1 = p * (1 - a)
        self.c0 = r * (1 - a) + p
        self.u1 = A * a + p + r

    def closed(self, n: int) -> int:
        return self.A * self.a**n + self.p * n + self.r

    def verify(self) -> None:
        u = Fraction(self.u1)
        for n in range(1, 12):
            check(u == self.closed(n), "the closed form disagrees with the recurrence")
            u = self.a * u + self.c1 * n + self.c0
        # fixed-point route: coefficients of k and constants in u(k + 1) = a u(k) + c1 k + c0
        check(self.p == Fraction(self.c1, 1 - self.a), "p disagrees with the k-coefficient equation")
        check(self.r == Fraction(self.c0 - self.p, 1 - self.a), "r disagrees with the constant equation")
        check(self.A == Fraction(self.u1 - self.p - self.r, self.a), "A disagrees with the base case")

    def u(self, n: int) -> int:
        """u(n) by running the recurrence."""
        value = self.u1
        for i in range(1, n):
            value = self.a * value + self.c1 * i + self.c0
        return value


def _draw_rec(rng: random.Random) -> _Rec:
    for _ in attempts(500, "a recurrence with nonzero coefficients"):
        a = rng.choice((2, 2, 3, 4))
        A = rng.choice((1, 2, 3, 4, 5, -1, -2, -3))
        p = rng.choice((-5, -4, -3, -2, 2, 3, 4, 5, 6))
        r = rng.choice((-6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7))
        c0 = r * (1 - a) + p
        if c0 != 0 and p != r and abs(c0) != abs(p * (1 - a)):
            break
    rec = _Rec(a, A, p, r)
    rec.verify()
    return rec


def _pm(value: int) -> str:
    """' + 5' or ' - 5' for a term added to an expression; nothing for zero."""
    return "" if value == 0 else (f" + {value}" if value > 0 else f" - {-value}")


def _par(value) -> str:
    return f"({value})" if value < 0 else str(value)


def _signed(value: int) -> str:
    return f"+ {value}" if value > 0 else f"- {-value}"


def _forcing(rec: _Rec, var: str) -> tuple[str, str]:
    """(sign, text) of the forcing term c1 var + c0 for ``u(var + 1) = a u(var) <sign> <text>``."""
    c1, c0 = rec.c1, rec.c0
    head = "" if c1 == 0 else (var if c1 == 1 else f"-{var}" if c1 == -1 else f"{c1}{var}")
    text = head + ((f" {_signed(c0)}" if head else str(c0)) if c0 else "")
    if text.startswith("-"):
        return "-", text[1:]
    return "+", text


def _rule(rec: _Rec, var: str) -> str:
    sign, text = _forcing(rec, var)
    return f"u({var} + 1) = {rec.a}u({var}) {sign} {text}"


def _rec_stem(rec: _Rec) -> str:
    return f"A sequence has u(1) = {rec.u1} and {_rule(rec, 'n')}"


def _const_equation(rec: _Rec) -> str:
    tail = f" {_signed(rec.c0)}" if rec.c0 else ""
    return f"p + r = {rec.a}r{tail}"


def _diff_text(stmt: _Ineq) -> str:
    rhs = stmt.rhs_text("n")
    if stmt.rhs[0] == "lin" and stmt.rhs[2]:
        rhs = f"({rhs})"
    return f"{stmt.lhs_text('n')} - {rhs}"


def _hard_rec(rng):
    rec = _draw_rec(rng)
    ask_r = rng.random() < 0.5
    a, p, r, A = rec.a, rec.p, rec.r, rec.A
    stem = (
        f"{_rec_stem(rec)}. It is to be proved by induction that u(n) = A·{a}^n + pn + r for every integer n ≥ 1, "
        "where A, p and r are constants. In the inductive step u(k) is replaced by A·%d^k + pk + r in the recurrence, "
        "and the coefficients of k and the constant terms of u(k + 1) are matched with A·%d^(k + 1) + p(k + 1) + r. "
        % (a, a)
    )
    if ask_r:
        cands = [
            (Fraction(rec.c0, 1 - a), "constant-ignores-the-k-term-carry"),
            (-r, "sign-slip-in-matching-coefficients"),
            (rec.c0, "forgets-to-divide-by-one-minus-a"),
            (Fraction(rec.c0 + p, 1 - a), "sign-slip-in-matching-coefficients"),
            (p, "p-and-r-confused"),
            (rec.c0 - p, "forgets-to-divide-by-one-minus-a"),
            (rec.c0 * (1 - a), "multiplies-instead-of-divides"),
        ]
        answer, kind_text, working = (
            r,
            "Find r.",
            [
                f"Matching the coefficients of k: p = {a}p{_pm(rec.c1)}, so p = {p}. Matching constants: p + r = {a}r{_pm(rec.c0)}, so r = {r}."
            ],
        )
    else:
        cands = [
            (Fraction(rec.u1, a), "base-case-ignores-p-and-r"),
            (rec.u1 - p - r, "forgets-to-divide-by-a"),
            (Fraction(rec.u1 + p + r, a), "sign-slip-in-matching-coefficients"),
            (Fraction(rec.u1 - r, a), "base-case-ignores-p-and-r"),
            (Fraction(rec.u1 - p, a), "base-case-ignores-p-and-r"),
            (rec.u1 * a, "multiplies-instead-of-divides"),
            (Fraction(rec.u1 - p + r, a), "sign-slip-in-matching-coefficients"),
            (Fraction(rec.u1 + p - r, a), "sign-slip-in-matching-coefficients"),
            (-A, "sign-slip-in-matching-coefficients"),
            (rec.u1, "base-case-ignores-p-and-r"),
        ]
        answer, kind_text, working = (
            A,
            "Find the integer A.",
            [
                f"Matching coefficients gives p = {p} and r = {r}; then u(1) = {a}A{_pm(p)}{_pm(r)} = {rec.u1}, so A = {A}."
            ],
        )
    distractors, seen = [], {Fraction(answer)}
    for value, tag in cands:
        value = Fraction(value)
        if value.denominator != 1 or value in seen:
            continue
        seen.add(value)
        distractors.append(Distractor(str(int(value)), tag, int(value)))
    check(len(distractors) >= 2, "too few recurrence distractors")
    return dict(
        text=stem + kind_text,
        answer=answer,
        answer_text=str(answer),
        answer_kind="number",
        working=working,
        distractors=distractors[:4],
    )


# ---------------------------------------------------------------------------------------------------------------
# Single-answer tiers for the inequality


def _easy(rng):
    stmt = _draw_ineq(rng)
    n0, k0 = stmt.n0(), stmt.k0()
    first_true = next(n for n in range(1, n0 + 1) if stmt.holds(n))
    non_strict = next(n for n in range(1, n0 + 1) if all(stmt.L(m) >= stmt.R(m) for m in range(n, n0 + 40)))
    cands = [
        (n0 - 1, "off-by-one-base-case"),
        (first_true, "first-true-value-taken-as-base-case"),
        (non_strict, "strict-inequality-treated-as-non-strict"),
        (n0 + 1, "off-by-one-base-case"),
        (k0, "step-threshold-for-base-case"),
    ]
    distractors = _int_dists(n0, cands)
    check(len(distractors) >= 2, "too few n0 distractors")
    shown = [stmt.L(n) for n in range(1, 3)]
    return dict(
        text=(
            f"P(n): {stmt.statement()}. P(n) is true for every integer n from some point on, and the proof by "
            f"induction starts there. Find the smallest positive integer n0 such that P(n) is true for every integer n ≥ n0."
        ),
        answer=n0,
        answer_text=str(n0),
        answer_kind="number",
        working=[
            f"Trying n = 1, 2, 3, …: P fails at n = {n0 - 1} ({stmt.L(n0 - 1)} against {stmt.R(n0 - 1)}) and holds "
            f"from n = {n0} on ({stmt.L(n0)} > {stmt.R(n0)}), so n0 = {n0}."
        ],
        distractors=distractors[:4],
    )


def _medium(rng):
    stmt = _draw_ineq(rng)
    n0, k0 = stmt.n0(), stmt.k0()
    cands = [
        (k0 - 1, "off-by-one-step-threshold"),
        (n0, "base-case-threshold-for-step"),
        (k0 + 1, "off-by-one-step-threshold"),
        (k0 + 2, "off-by-one-step-threshold"),
        (n0 + 1, "base-case-threshold-for-step"),
    ]
    distractors = _int_dists(k0, cands)
    check(len(distractors) >= 2, "too few k0 distractors")
    return dict(
        text=_step_stem(stmt)
        + f" To reach P(k + 1) the step needs {stmt.key_text()}. Find the smallest positive integer k0 such that this "
        "is true for every integer k ≥ k0.",
        answer=k0,
        answer_text=str(k0),
        answer_kind="number",
        working=[
            f"{stmt.key_text()} fails at k = {k0 - 1} "
            f"({stmt.mult(k0 - 1) * stmt.R(k0 - 1)} < {stmt.R(k0)}) and holds from k = {k0} on "
            f"({stmt.mult(k0) * stmt.R(k0)} ≥ {stmt.R(k0 + 1)}), so k0 = {k0}."
        ],
        distractors=distractors[:4],
    )


@generator(
    "r4f_induction_inequality",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4f_induction_inequality(rng: random.Random, tier: str = "easy") -> dict:
    """A typed step of an induction proof: the base-case threshold, the key-inequality threshold, or a recurrence constant."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard_rec(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------------------------------------------
# Exam chain


def _exam_inequality(rng):
    stmt = _draw_ineq(rng)
    n0, k0 = stmt.n0(), stmt.k0()
    diff0 = stmt.L(n0) - stmt.R(n0)
    margin = stmt.mult(n0) * stmt.R(n0) - stmt.R(n0 + 1)
    part_a = number_part(
        f"Find the smallest positive integer n0 such that P(n) is true for every integer n ≥ n0. This is the base case.",
        n0,
        marks=2,
        sensible_range=(0, None),
        working=[f"P fails at n = {n0 - 1} and holds from n = {n0} on, so n0 = {n0}."],
        wrongs=[
            (n0 - 1, "off-by-one-base-case"),
            (next(n for n in range(1, n0 + 1) if stmt.holds(n)), "first-true-value-taken-as-base-case"),
            (n0 + 1, "off-by-one-base-case"),
            (k0, "step-threshold-for-base-case"),
        ],
    )
    part_b = number_part(
        f"Using your n0, find {_diff_text(stmt)} at n = n0, to show that the base case holds.",
        diff0,
        marks=1,
        sensible_range=(0, None),
        working=[f"At n = {n0}: {stmt.L(n0)} - {stmt.R(n0)} = {diff0} > 0."],
        wrongs=[
            (stmt.L(n0), "right-hand-side-left-out"),
            (stmt.R(n0) - stmt.L(n0), "difference-taken-the-wrong-way"),
            (stmt.L(n0) + stmt.R(n0), "sum-taken-for-the-difference"),
            (stmt.R(n0), "left-hand-side-left-out"),
        ],
        follow_through=f"{stmt.L_expr('part_a')} - {stmt.R_expr('part_a')}",
    )
    part_c = number_part(
        f"In the inductive step P(k) is assumed, and the step needs {stmt.key_text()}. Find the smallest positive integer k0 such that this is true for every integer k ≥ k0.",
        k0,
        marks=2,
        sensible_range=(0, None),
        working=[f"The inequality fails at k = {k0 - 1} and holds from k = {k0} on, so k0 = {k0}."],
        wrongs=[
            (k0 - 1, "off-by-one-step-threshold"),
            (k0 + 1, "off-by-one-step-threshold"),
            (n0, "base-case-threshold-for-step"),
            (k0 + 2, "off-by-one-step-threshold"),
        ],
    )
    part_d = number_part(
        "With k = n0, find the left side minus the right side of that inequality. The inductive step needs this not to be negative.",
        margin,
        marks=1,
        sensible_range=(0, None),
        working=[f"At k = {n0}: {stmt.mult(n0) * stmt.R(n0)} - {stmt.R(n0 + 1)} = {margin}."],
        wrongs=[
            (stmt.R(n0 + 1) - stmt.mult(n0) * stmt.R(n0), "difference-taken-the-wrong-way"),
            (stmt.mult(n0) * stmt.R(n0), "right-hand-side-left-out"),
            (stmt.R(n0) - stmt.R(n0 + 1), "multiplier-dropped"),
            (stmt.mult(n0) * stmt.R(n0) + stmt.R(n0 + 1), "sum-taken-for-the-difference"),
        ],
        follow_through=f"{stmt.mult_expr('part_a')}*{stmt.R_expr('part_a')} - {stmt.R_expr('part_a + 1')}",
    )
    return dict(
        text=(
            f"Let P(n) be the statement {stmt.statement()}. P(n) is to be proved by induction for every integer n from "
            f"some base case on. In the inductive step, {stmt.lhs_text('(k + 1)')} = {_mult_text(stmt)} × {stmt.lhs_text('k')}."
        ),
        parts=[part_a, part_b, part_c, part_d],
    )


def _exam_recurrence(rng):
    rec = _draw_rec(rng)
    a, p, r, A = rec.a, rec.p, rec.r, rec.A
    u3 = rec.u(3)
    u2 = rec.u(2)
    part_a = number_part(
        "Use the recurrence to find u(3).",
        u3,
        marks=1,
        working=[
            f"u(2) = {a} × {_par(rec.u1)}{_pm(rec.c1 + rec.c0)} = {u2}, then u(3) = {a} × {_par(u2)}{_pm(2 * rec.c1 + rec.c0)} = {u3}."
        ],
        wrongs=[
            (u2, "stops-one-term-early"),
            (a * u2 + rec.c1 * 3 + rec.c0, "forcing-term-index-off-by-one"),
            (a * u2, "forcing-term-left-out"),
            (a * u2 + rec.c1 + rec.c0, "forcing-term-index-off-by-one"),
        ],
    )
    part_b = number_part(
        (
            f"It is to be proved by induction that u(n) = A·{a}^n + pn + r for every integer n ≥ 1. In the inductive "
            f"step the coefficients of k in {_rule(rec, 'k')} are matched with those of "
            f"A·{a}^(k + 1) + p(k + 1) + r. Find p."
        ),
        p,
        marks=2,
        working=[f"Coefficients of k: p = {a}p{_pm(rec.c1)}, so p = {_par(rec.c1)} ÷ {_par(1 - a)} = {p}."],
        wrongs=[
            (-p, "sign-slip-in-matching-coefficients"),
            (rec.c1, "forgets-to-divide-by-one-minus-a"),
            (Fraction(rec.c1, 1 + a), "sign-slip-in-matching-coefficients"),
            (Fraction(rec.c1, a), "forgets-to-divide-by-one-minus-a"),
            (rec.c1 * (1 - a), "multiplies-instead-of-divides"),
            (Fraction(rec.c0, 1 - a), "matches-the-constant-equation-instead"),
            (p + 1, "arithmetic-slip"),
            (p - 1, "arithmetic-slip"),
        ],
    )
    part_c = number_part(
        f"Matching the constant terms (using your p) gives {_const_equation(rec)}. Find r.",
        r,
        marks=2,
        working=[f"{_const_equation(rec)}, so r = ({rec.c0} - {_par(p)}) ÷ {_par(1 - a)} = {r}."],
        wrongs=[
            (Fraction(rec.c0, 1 - a), "constant-ignores-the-k-term-carry"),
            (-r, "sign-slip-in-matching-coefficients"),
            (rec.c0 - p, "forgets-to-divide-by-one-minus-a"),
            (rec.c0 * (1 - a), "multiplies-instead-of-divides"),
            (p, "p-and-r-confused"),
        ],
        follow_through=f"({rec.c0} - part_b)/(1 - {a})",
    )
    part_d = number_part(
        f"Use u(1) = {rec.u1} (the base case) with your p and r to find the integer A.",
        A,
        marks=1,
        working=[f"u(1) = {a}A + p + r = {rec.u1}, so A = ({rec.u1} - {_par(p)} - {_par(r)}) ÷ {a} = {A}."],
        wrongs=[
            (Fraction(rec.u1, a), "base-case-ignores-p-and-r"),
            (rec.u1 - p - r, "forgets-to-divide-by-a"),
            (Fraction(rec.u1 - r, a), "base-case-ignores-p-and-r"),
            (Fraction(rec.u1 - p, a), "base-case-ignores-p-and-r"),
            (rec.u1 * a, "multiplies-instead-of-divides"),
            (-A, "sign-slip-in-matching-coefficients"),
            (rec.u1, "base-case-ignores-p-and-r"),
        ],
        follow_through=f"({rec.u1} - part_b - part_c)/{a}",
    )
    return dict(text=f"{_rec_stem(rec)}.", parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r4f_induction_inequality", level=19, presets={"exam": dict(tier="exam")})
@distinct_part_answers
@follow_through_clean
def exd_x_r4f_induction_inequality(rng: random.Random, tier: str = "exam") -> dict:
    """Exam chain: an inequality proof (four parts), or a recurrence closed form (four parts)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _exam_recurrence(rng) if rng.random() < 0.4 else _exam_inequality(rng)
