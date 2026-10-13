"""Generates `r4f_induction_divisibility` questions and the exam chain `exd_x_r4f_induction_divisibility`.

Skill `proof-by-induction-divisibility` (``proof-and-foundations``): the typed steps of an induction proof that a
two-exponential expression such as

    f(n) = 3^(2n + 1) + 2^(n + 2)        is divisible by 7 for every integer n >= 1.

Every statement here has the shape f(n) = T(n) + s S(n) with T(n) = r1^(m1 n + s1) and S(n) = r2^(m2 n + s2), so that
with u = r1^m1 and v = r2^m2

    f(k + 1) = v f(k) + (u - v) T(k)        and        f(k + 1) = u f(k) + s (v - u) S(k),

and the divisor d divides both f(1) and u - v. At least one exponent in each term carries a shift or a multiplier, so
the plain ``a^n - 1`` and ``a^n + b`` statements of ``ucv1-proof-by-induction`` are not repeated here.

Tiers (topic ``r4f_induction_divisibility``):

* easy   -- the base case: f(1) = d q, find the whole number q;
* medium -- the remainder R(k) in f(k + 1) = v f(k) + R(k), as an expression in k (v is stated, so R is unique);
* hard   -- the multiple exhibited: M(k) in f(k + 1) = c f(k) + d M(k), where the stem fixes c as the base of one of
  the two terms (so M is unique), as an expression in k.

The exam chain is four number parts: f(1); the number m in f(k + 1) = v f(k) + m T(k); f(3) from f(2) using that
rearrangement (follow-through from m); and the whole number q with f(3) = d q (follow-through from f(3)).

Every statement is confirmed for n = 1..12 by direct integer arithmetic, and every rearrangement is confirmed at
k = 1..8 with exact integers, so each key is derived twice.
"""

from __future__ import annotations

import random
from functools import lru_cache
from math import gcd

from ..answers import Distractor
from ..core import attempts, check, generator, pick_distractors
from ._exd_distinct_parts import distinct_part_answers
from ._r4f_ft import follow_through_clean
from .exd_x_r3c_parts import number_part


@lru_cache(maxsize=1)
def _configs() -> tuple:
    """Every (r1, m1, s1, r2, m2, s2, sign, d) whose statement is true, built once."""
    out = []
    for r1 in range(2, 10):
        for r2 in range(2, 10):
            if gcd(r1, r2) != 1:
                continue
            for m1 in (1, 2):
                for m2 in (1, 2):
                    u, v = r1**m1, r2**m2
                    if u == v:
                        continue
                    for s1 in range(4):
                        for s2 in range(4):
                            if (m1 == 1 and s1 == 0) or (m2 == 1 and s2 == 0):
                                continue
                            for sign in (1, -1):
                                f1 = r1 ** (m1 + s1) + sign * r2 ** (m2 + s2)
                                if f1 <= 0 or f1 > 3000:
                                    continue
                                g = gcd(abs(u - v), f1)
                                for d in range(3, min(g, 40) + 1):
                                    if g % d == 0 and _true_for(r1, m1, s1, r2, m2, s2, sign, d):
                                        out.append((r1, m1, s1, r2, m2, s2, sign, d))
    return tuple(out)


def _true_for(r1, m1, s1, r2, m2, s2, sign, d) -> bool:
    for n in range(1, 13):
        value = r1 ** (m1 * n + s1) + sign * r2 ** (m2 * n + s2)
        if value <= 0 or value % d:
            return False
    return True


class _Stmt:
    """One statement f(n) = r1^(m1 n + s1) + sign r2^(m2 n + s2), divisible by d."""

    def __init__(self, cfg):
        self.r1, self.m1, self.s1, self.r2, self.m2, self.s2, self.sign, self.d = cfg
        self.u = self.r1**self.m1
        self.v = self.r2**self.m2

    def f(self, n: int) -> int:
        return self.r1 ** (self.m1 * n + self.s1) + self.sign * self.r2 ** (self.m2 * n + self.s2)

    def T(self, n: int) -> int:
        return self.r1 ** (self.m1 * n + self.s1)

    def S(self, n: int) -> int:
        """The second power, unsigned."""
        return self.r2 ** (self.m2 * n + self.s2)

    def verify(self) -> None:
        for n in range(1, 13):
            check(self.f(n) > 0 and self.f(n) % self.d == 0, "the divisibility statement is false")
        for k in range(1, 9):
            check(self.f(k + 1) == self.v * self.f(k) + (self.u - self.v) * self.T(k), "route 1 rearrangement wrong")
            check(
                self.f(k + 1) == self.u * self.f(k) + self.sign * (self.v - self.u) * self.S(k),
                "route 2 rearrangement wrong",
            )
        check((self.u - self.v) % self.d == 0, "the divisor does not divide u - v")

    # ---- text -------------------------------------------------------------------------------------------------
    @staticmethod
    def _exp(m: int, s: int, var: str) -> str:
        head = var if m == 1 else f"{m}{var}"
        return head if s == 0 else f"{head} + {s}"

    def power1(self, var: str = "n") -> str:
        return f"{self.r1}^({self._exp(self.m1, self.s1, var)})"

    def power2(self, var: str = "n") -> str:
        return f"{self.r2}^({self._exp(self.m2, self.s2, var)})"

    def f_text(self) -> str:
        return f"{self.power1()} {'+' if self.sign > 0 else '-'} {self.power2()}"

    def key_text(self, coef: int, which: int) -> str:
        """The answer text of ``coef * power`` in k, parseable as typed: 7*3^(2*k + 1)."""
        r, m, s = (self.r1, self.m1, self.s1) if which == 1 else (self.r2, self.m2, self.s2)
        inner = (f"{m}*k" if m != 1 else "k") + (f" + {s}" if s else "")
        base = f"{r}^({inner})"
        if coef == 1:
            return base
        if coef == -1:
            return f"-{base}"
        return f"{coef}*{base}"

    def key_shown(self, coef: int, which: int) -> str:
        return self.key_text(coef, which).replace("*", "·")


def _draw(rng: random.Random) -> _Stmt:
    stmt = _Stmt(rng.choice(_configs()))
    stmt.verify()
    return stmt


def _int_dists(answer: int, cands) -> list:
    out, seen = [], {answer}
    for value, tag in cands:
        if value in seen:
            continue
        seen.add(value)
        out.append(Distractor(str(int(value)), tag, int(value)))
    return out


def _expr_dists(stmt: _Stmt, answer_pair, cands) -> list:
    """Expression distractors from (coef, which, tag); the key is ``answer_pair`` = (coef, which)."""
    out, seen = [], {answer_pair}
    for coef, which, tag in cands:
        if coef == 0 or (coef, which) in seen:
            continue
        seen.add((coef, which))
        text = stmt.key_text(coef, which)
        out.append(Distractor(text, tag, text))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Single-answer tiers


def _easy(rng):
    stmt = _draw(rng)
    f1 = stmt.f(1)
    q = f1 // stmt.d
    dropped_shift = stmt.u + stmt.sign * stmt.v
    dropped_mult = stmt.r1 ** (1 + stmt.s1) + stmt.sign * stmt.r2 ** (1 + stmt.s2)
    cands = [
        (f1, "forgets-to-divide-by-d"),
        (stmt.f(2) // stmt.d, "base-case-evaluated-at-wrong-n"),
        (stmt.f(3) // stmt.d, "base-case-evaluated-at-wrong-n"),
        (dropped_shift // stmt.d if dropped_shift % stmt.d == 0 else dropped_shift, "exponent-shift-dropped"),
        (dropped_mult // stmt.d if dropped_mult % stmt.d == 0 else dropped_mult, "exponent-multiplier-dropped"),
    ]
    distractors = _int_dists(q, cands)
    check(len(distractors) >= 3, "too few base-case distractors")
    return dict(
        text=(
            f"P(n): f(n) = {stmt.f_text()} is divisible by {stmt.d}, for every integer n ≥ 1. P(n) is to be proved "
            f"by induction. For the base case, f(1) is written as f(1) = {stmt.d}q. Find the whole number q."
        ),
        answer=q,
        answer_text=str(q),
        answer_kind="expression",
        working=[
            f"f(1) = {stmt.r1}^{stmt.m1 + stmt.s1} {'+' if stmt.sign > 0 else '-'} {stmt.r2}^{stmt.m2 + stmt.s2} = {f1} = {stmt.d} × {q}."
        ],
        distractors=distractors[:4],
    )


def _medium(rng):
    stmt = _draw(rng)
    coef = stmt.u - stmt.v
    answer_pair = (coef, 1)
    cands = [
        (-coef, 1, "remainder-sign-reversed"),
        (coef, 2, "remainder-keeps-the-wrong-term"),
        (1, 1, "remainder-drops-the-factor"),
        (stmt.u + stmt.v, 1, "uses-sum-not-difference-of-bases"),
        (stmt.u - stmt.v if stmt.sign < 0 else -(stmt.u - stmt.v), 2, "remainder-keeps-the-wrong-term"),
    ]
    distractors = _expr_dists(stmt, answer_pair, cands)
    check(len(distractors) >= 3, "too few remainder distractors")
    key = stmt.key_text(coef, 1)
    return dict(
        text=(
            f"P(n): f(n) = {stmt.f_text()} is divisible by {stmt.d}, for every integer n ≥ 1. In the inductive step, "
            f"f(k + 1) is written as f(k + 1) = {stmt.v}f(k) + R, where R is an expression in k. "
            "Find R."
        ),
        answer=key,
        answer_text=key,
        answer_kind="expression",
        working=[
            f"f(k + 1) = {stmt.u}·{stmt.power1('k')} {'+' if stmt.sign > 0 else '-'} {stmt.v}·{stmt.power2('k')} "
            f"= {stmt.v}f(k) {'+' if coef > 0 else '-'} {abs(coef)}·{stmt.power1('k')}, so R = {stmt.key_shown(coef, 1)}."
        ],
        distractors=distractors[:4],
    )


def _hard(rng):
    stmt = _draw(rng)
    route = rng.choice((1, 2))
    if route == 1:
        c, coef, which = stmt.v, (stmt.u - stmt.v) // stmt.d, 1
        wrong_other = (stmt.sign * (stmt.v - stmt.u)) // stmt.d
    else:
        c, coef, which = stmt.u, stmt.sign * (stmt.v - stmt.u) // stmt.d, 2
        wrong_other = (stmt.u - stmt.v) // stmt.d
    other = 3 - which
    answer_pair = (coef, which)
    cands = [
        (coef * stmt.d, which, "forgets-to-divide-the-remainder"),
        (-coef, which, "remainder-sign-reversed"),
        (wrong_other, other, "remainder-keeps-the-wrong-term"),
        (coef * stmt.d, other, "remainder-keeps-the-wrong-term"),
        (1, which, "remainder-drops-the-factor"),
    ]
    distractors = _expr_dists(stmt, answer_pair, cands)
    check(len(distractors) >= 3, "too few multiple distractors")
    key = stmt.key_text(coef, which)
    return dict(
        text=(
            f"P(n): f(n) = {stmt.f_text()} is divisible by {stmt.d}, for every integer n ≥ 1. In the inductive step, "
            f"f(k + 1) is written as f(k + 1) = {c}f(k) + {stmt.d}M, where M is an expression in k, so that "
            f"f(k + 1) is seen to be a multiple of {stmt.d} once f(k) is. Find M."
        ),
        answer=key,
        answer_text=key,
        answer_kind="expression",
        working=[
            f"f(k + 1) - {c}f(k) = {stmt.key_shown(coef * stmt.d, which)} = {stmt.d}·({stmt.key_shown(coef, which)}), "
            f"so M = {stmt.key_shown(coef, which)}."
        ],
        distractors=distractors[:4],
    )


@generator(
    "r4f_induction_divisibility",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4f_induction_divisibility(rng: random.Random, tier: str = "easy") -> dict:
    """A typed step of a divisibility induction: base-case quotient, remainder, or the multiple exhibited."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------------------------------------------
# Exam chain


@generator("exd_x_r4f_induction_divisibility", level=19, presets={"exam": dict(tier="exam")})
@distinct_part_answers
@follow_through_clean
def exd_x_r4f_induction_divisibility(rng: random.Random, tier: str = "exam") -> dict:
    """Four number parts: f(1); m in f(k + 1) = v f(k) + m T(k); f(3) from f(2); the quotient f(3) / d."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "a statement whose f(3) stays modest"):
        stmt = _draw(rng)
        if stmt.f(3) <= 150000:
            break
    d, u, v = stmt.d, stmt.u, stmt.v
    f1, f2, f3 = stmt.f(1), stmt.f(2), stmt.f(3)
    m = u - v
    t2 = stmt.T(2)
    check(f3 == v * f2 + m * t2, "f(3) disagrees with the rearrangement")
    check(f3 % d == 0, "f(3) is not a multiple of d")
    q3 = f3 // d
    sign = "+" if stmt.sign > 0 else "-"
    part_a = number_part(
        "Verify P(1): find the value of f(1).",
        f1,
        marks=1,
        working=[
            f"f(1) = {stmt.r1}^{stmt.m1 + stmt.s1} {sign} {stmt.r2}^{stmt.m2 + stmt.s2} = {f1} = {d} × {f1 // d}."
        ],
        wrongs=[
            (f2, "base-case-evaluated-at-wrong-n"),
            (stmt.u + stmt.sign * stmt.v, "exponent-shift-dropped"),
            (stmt.r1 ** (1 + stmt.s1) + stmt.sign * stmt.r2 ** (1 + stmt.s2), "exponent-multiplier-dropped"),
            (f1 // d, "gives-quotient-not-value"),
        ],
    )
    part_b = number_part(
        (
            f"In the inductive step f(k) is assumed divisible by {d}. Write f(k + 1) = {v}f(k) + m × "
            f"{stmt.power1('k')}, where m is a number, and state m."
        ),
        m,
        marks=2,
        working=[
            f"f(k + 1) = {u}·{stmt.power1('k')} {sign} {v}·{stmt.power2('k')} = {v}f(k) + ({u} - {v})·{stmt.power1('k')}, so m = {m}."
        ],
        wrongs=[
            (-m, "remainder-sign-reversed"),
            (u + v, "uses-sum-not-difference-of-bases"),
            (u, "remainder-drops-the-factor"),
            (v, "remainder-keeps-the-wrong-term"),
        ],
    )
    part_c = number_part(
        (
            f"When k = 2, f(2) = {f2} and {stmt.power1('k')} = {t2}. Use your rearrangement to find "
            f"f(3) = {v}f(2) + m × {t2}."
        ),
        f3,
        marks=2,
        sensible_range=(0, None),
        working=[f"f(3) = {v} × {f2} {'+' if m > 0 else '-'} {abs(m)} × {t2} = {f3}."],
        wrongs=[
            (u * f2 + m * t2, "wrong-multiplier-on-f-of-k"),
            (v * f2, "remainder-dropped"),
            (f2 + m * t2, "multiplier-dropped"),
        ],
        follow_through=f"{v}*{f2} + part_b*{t2}",
    )
    part_d = number_part(
        f"Hence f(3) is a multiple of {d}. Write f(3) = {d}q and find the whole number q.",
        q3,
        marks=1,
        sensible_range=(0, None),
        working=[f"q = f(3) ÷ {d} = {f3} ÷ {d} = {q3}."],
        wrongs=[
            (f3, "forgets-to-divide-by-d"),
            (f2 // d, "base-case-evaluated-at-wrong-n"),
            (stmt.f(4) // d, "off-by-one-term-count"),
            (f3 // (d + 1), "divides-by-the-wrong-divisor"),
        ],
        follow_through=f"part_c/{d}",
    )
    return dict(
        text=(
            f"Let P(n) be the statement that f(n) = {stmt.f_text()} is divisible by {d}, for every integer n ≥ 1. "
            "P(n) is to be proved by induction."
        ),
        parts=[part_a, part_b, part_c, part_d],
    )
