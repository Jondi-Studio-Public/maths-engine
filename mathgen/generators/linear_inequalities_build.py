"""Linear inequalities: four skills, each with easy/medium/hard practice and an exam chain.

Topics (all answers are the solved inequality, graded as ``inequality``: the learner's solution set is compared with
the key's, so ``x < 3``, ``3 > x`` and ``(-inf, 3)`` are the same answer):

* ``lin_ineq_one_step``  (``one-step-inequalities``): easy ``x + b < c``; medium ``ax <= c`` with a bound that is
  usually a fraction; hard ``x/a >= c`` or ``a(x + m) > c``.
* ``lin_ineq_negative``  (``inequalities-negative-mult-div``): easy ``-ax < c``; medium ``-x/a >= c`` or
  ``b - ax > c``; hard variables on both sides where the x coefficient ends negative.
* ``lin_ineq_two_step``  (``two-step-inequalities``): easy ``ax + b > c``; medium brackets or a fraction
  coefficient; hard variables on both sides with a bracket, or a double inequality such as ``-3 < 2x + 1 <= 9``
  (sometimes with a negative coefficient, which reverses the order).
* ``lin_ineq_word``      (``inequality-word-problems``): budgets, phone plans, fares, loads, savings; the hard tier
  compares two plans, asks for a test mark that lifts an average, or bounds a fare between two amounts.

Exam topics ``exd_x_ineq_one_step``, ``exd_x_ineq_negative``, ``exd_x_ineq_two_step`` and ``exd_x_ineq_word`` are
3-4 part chains. The later parts ask for the greatest or least integer solution, evaluate a stated expression at
it (follow-through from the integer), count and sum the integer solutions of a double inequality, or, in a word
problem, find the greatest whole number of items, the cost at that number (follow-through) and the range once a
second constraint is added.

Every answer is built from exact ``Fraction`` arithmetic and then re-derived by probing the ORIGINAL statement at
points either side of the bound (a different route from the algebra that produced the bound). Every distractor is a
real slip and is read by the real inequality grader, which must refuse it.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction as Fr
from types import SimpleNamespace

from ..answers import Distractor
from ..core import (
    VerificationError,
    check,
    fmt_signed,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    rational_text,
)
from ..grading import GRADERS

_OPS = ("<", "≤", ">", "≥")
_FLIP = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}
_STRICT_SWAP = {"<": "≤", "≤": "<", ">": "≥", "≥": ">"}
_MAX_DRAWS = 400

# Misconception ids (catalogue block "inequalities and matrix inverses" in mathgen/misconceptions.py).
M_NOT_REVERSED = "ineq-direction-not-reversed"
M_REVERSED_UNPROMPTED = "ineq-reversed-without-need"
M_STRICT = "strict-nonstrict-confused"
M_SIGN = "sign-error-moving-term"
M_NO_DIVIDE = "forgot-to-divide-by-coefficient"
M_IGNORED_CONST = "ignored-constant-term"
M_WRONG_INVERSE = "used-wrong-inverse-operation"
M_ONE_TERM = "operation-on-one-term-only"
M_BOUND_SIGN = "bound-sign-not-changed"
M_COMPOUND_PART = "compound-operation-missed-part"
M_SWAPPED_CONTEXT = "at-least-no-more-than-swapped"
M_ROUNDS_WRONG = "rounds-wrong-way-in-context"
M_STRICT_VALUE = "strict-bound-taken-as-value"
M_OFF_BY_ONE = "integer-solutions-endpoint-miscount"
M_ONE_CONSTRAINT = "compound-bound-omitted"
M_SUBSTITUTION = "substitution-sign-error"


# -- small helpers ---------------------------------------------------------------------------------------------------


def _nz(rng: random.Random, lo: int, hi: int, *avoid: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0 and v not in avoid])


def _rt(value) -> str:
    return rational_text(Fr(value))


def _holds(left, op: str, right) -> bool:
    return {"<": left < right, "≤": left <= right, ">": left > right, "≥": left >= right}[op]


def _dec(value) -> str | None:
    """A bound as a whole number or a terminating decimal of at most 2 places ("25.6"), else None."""
    v = Fr(value)
    if 100 % v.denominator:
        return None
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def _single_text(var: str, op: str, bound, dec: bool = False) -> str:
    if dec:
        b = _dec(bound)
        return "" if b is None else f"{var} {op} {b}"
    return f"{var} {op} {_rt(bound)}"


def _range_text(var: str, lo, lo_op: str, hi, hi_op: str, dec: bool = False) -> str:
    """``lo < x <= hi`` with ``lo_op`` and ``hi_op`` each "<" or "≤"."""
    if dec:
        a, b = _dec(lo), _dec(hi)
        return "" if a is None or b is None else f"{a} {lo_op} {var} {hi_op} {b}"
    return f"{_rt(lo)} {lo_op} {var} {hi_op} {_rt(hi)}"


def _accepts(answer_text: str, reply: str) -> bool:
    question = SimpleNamespace(answer=answer_text, answer_text=answer_text, answer_form=None, text="", parts=None)
    return bool(GRADERS["inequality"](question, reply).correct)


def _tagged(answer_text: str, cands, *, limit: int = 3) -> list[Distractor] | None:
    """Distractors from ``(text, misconception)`` candidates in priority order: no repeats, none the answer's text and
    none the real grader accepts. None when fewer than two survive (the caller redraws)."""
    out, seen = [], {answer_text}
    for text, mid in cands:
        if not text or text in seen or _accepts(answer_text, text):
            continue
        seen.add(text)
        out.append(Distractor(text, mid, value=text))
        if len(out) == limit:
            break
    return out if len(out) >= 2 else None


def _probes(bounds) -> list[Fr]:
    pts = {Fr(b) + d for b in bounds for d in (Fr(-2), Fr(-1), Fr(-1, 7), Fr(0), Fr(1, 7), Fr(1), Fr(2))}
    if len(bounds) == 2:
        pts.add((Fr(bounds[0]) + Fr(bounds[1])) / 2)
    return sorted(pts)


def _verify_single(sat, op: str, bound) -> None:
    """The solved inequality ``x op bound`` and the original statement `sat` agree at points around the bound."""
    for p in _probes([bound]):
        check(bool(sat(p)) == _holds(p, op, Fr(bound)), f"answer disagrees with the statement at x = {p}")


def _verify_range(sat, lo, lo_op: str, hi, hi_op: str) -> None:
    for p in _probes([lo, hi]):
        mine = _holds(Fr(lo), lo_op, p) and _holds(p, hi_op, Fr(hi))
        check(bool(sat(p)) == mine, f"range answer disagrees with the statement at x = {p}")


_VARS = ("x", "x", "x", "x", "n", "y", "t")
_ASKW = "Give your answer as an inequality in {v}; write any bound that is not a whole number as a decimal."
_ASK = "Give your answer as an inequality in {v}; write any bound that is not a whole number as a fraction."
_LEADS = ["Solve {e}.", "Find all values of {v} for which {e}.", "Solve the inequality {e}."]


def _prompt(rng: random.Random, v: str, e: str) -> str:
    return f"{rng.choice(_LEADS).format(e=e, v=v)} {_ASK.format(v=v)}"


def _sgn_inner(v: str, m: int) -> str:
    return join_terms([(1, v), (m, "")])


# -- single-inequality draws: each returns the statement, the key and the tagged slips -------------------------------
# A draw is a dict(expr, sat, op, bound, cands, working) with `cands` a list of (text, misconception).


def _draw_add(rng, v):
    b, x0 = _nz(rng, -15, 15), rng.randint(-12, 15)
    c, op = x0 + b, rng.choice(_OPS)
    lhs = f"{b} + {v}" if b > 0 and rng.random() < 0.25 else join_terms([(1, v), (b, "")])
    S = lambda o, k: _single_text(v, o, k)  # noqa: E731
    return dict(
        expr=f"{lhs} {op} {c}",
        sat=lambda x: _holds(x + b, op, c),
        op=op,
        bound=Fr(x0),
        cands=[
            (S(op, c + b), M_SIGN),
            (S(_FLIP[op], x0), M_REVERSED_UNPROMPTED),
            (S(_STRICT_SWAP[op], x0), M_STRICT),
            (S(op, c), M_IGNORED_CONST),
        ],
        working=[f"Undo the {fmt_signed(b)}: {v} {op} {c} {'-' if b > 0 else '+'} {abs(b)} = {x0}."],
    )


def _draw_mult(rng, v):
    a, op = rng.randint(2, 9), rng.choice(_OPS)
    c = _nz(rng, -45, 45)
    if rng.random() < 0.35:
        c = a * _nz(rng, -9, 9)
    x0 = Fr(c, a)
    S = lambda o, k: _single_text(v, o, k)  # noqa: E731
    return dict(
        expr=f"{a}{v} {op} {c}",
        sat=lambda x: _holds(a * x, op, c),
        op=op,
        bound=x0,
        cands=[
            (S(op, c), M_NO_DIVIDE),
            (S(op, c * a), M_WRONG_INVERSE),
            (S(_FLIP[op], x0), M_REVERSED_UNPROMPTED),
            (S(_STRICT_SWAP[op], x0), M_STRICT),
        ],
        working=[f"Divide both sides by {a} (positive, so the sign stays): {v} {op} {_rt(x0)}."],
    )


def _draw_div(rng, v):
    a, op = rng.randint(2, 9), rng.choice(_OPS)
    c = _nz(rng, -9, 10)
    S = lambda o, k: _single_text(v, o, k)  # noqa: E731
    return dict(
        expr=f"{v}/{a} {op} {c}",
        sat=lambda x: _holds(x / a, op, c),
        op=op,
        bound=Fr(a * c),
        cands=[
            (S(op, Fr(c, a)), M_WRONG_INVERSE),
            (S(op, c), M_NO_DIVIDE),
            (S(_FLIP[op], a * c), M_REVERSED_UNPROMPTED),
            (S(_STRICT_SWAP[op], a * c), M_STRICT),
        ],
        working=[f"Multiply both sides by {a}: {v} {op} {a * c}."],
    )


def _draw_bracket_one_step(rng, v):
    a, m, k, op = rng.randint(2, 7), _nz(rng, -9, 9), _nz(rng, -8, 10), rng.choice(_OPS)
    c = a * k
    S = lambda o, b: _single_text(v, o, b)  # noqa: E731
    return dict(
        expr=f"{a}({_sgn_inner(v, m)}) {op} {c}",
        sat=lambda x: _holds(a * (x + m), op, c),
        op=op,
        bound=Fr(k - m),
        cands=[
            (S(op, k), M_IGNORED_CONST),
            (S(op, k + m), M_SIGN),
            (S(op, Fr(c - m, a)), M_ONE_TERM),
            (S(_FLIP[op], k - m), M_REVERSED_UNPROMPTED),
            (S(_STRICT_SWAP[op], k - m), M_STRICT),
        ],
        working=[f"Divide both sides by {a}: {_sgn_inner(v, m)} {op} {k}.", f"Then {v} {op} {k - m}."],
    )


def _draw_neg_mult(rng, v):
    a, op = rng.randint(1, 9), rng.choice(_OPS)
    x0 = _nz(rng, -10, 10)
    c = -a * x0
    S = lambda o, b: _single_text(v, o, b)  # noqa: E731
    return dict(
        expr=f"{fmt_signed(-a, v)} {op} {c}",
        sat=lambda x: _holds(-a * x, op, c),
        op=_FLIP[op],
        bound=Fr(x0),
        cands=[
            (S(op, x0), M_NOT_REVERSED),
            (S(_FLIP[op], -x0), M_BOUND_SIGN),
            (S(_FLIP[op], c), M_NO_DIVIDE),
            (S(_STRICT_SWAP[_FLIP[op]], x0), M_STRICT),
        ],
        working=[f"Divide both sides by {-a} (negative), so reverse the sign: {v} {_FLIP[op]} {x0}."],
    )


def _draw_neg_div(rng, v):
    a, op = rng.randint(2, 9), rng.choice(_OPS)
    c = _nz(rng, -8, 8)
    S = lambda o, b: _single_text(v, o, b)  # noqa: E731
    return dict(
        expr=f"-{v}/{a} {op} {c}",
        sat=lambda x: _holds(-x / a, op, c),
        op=_FLIP[op],
        bound=Fr(-a * c),
        cands=[
            (S(op, -a * c), M_NOT_REVERSED),
            (S(_FLIP[op], a * c), M_BOUND_SIGN),
            (S(_FLIP[op], Fr(-c, a)), M_WRONG_INVERSE),
            (S(_STRICT_SWAP[_FLIP[op]], -a * c), M_STRICT),
        ],
        working=[f"Multiply both sides by {-a} (negative), so reverse the sign: {v} {_FLIP[op]} {-a * c}."],
    )


def _draw_b_minus_ax(rng, v):
    a, b, op = rng.randint(1, 6), _nz(rng, -12, 12), rng.choice(_OPS)
    c = _nz(rng, -20, 20)
    if rng.random() < 0.4:
        c = b + a * _nz(rng, -8, 8)
    bound = Fr(c - b, -a)
    S = lambda o, k: _single_text(v, o, k)  # noqa: E731
    return dict(
        expr=f"{join_terms([(b, ''), (-a, v)])} {op} {c}",
        sat=lambda x: _holds(b - a * x, op, c),
        op=_FLIP[op],
        bound=bound,
        cands=[
            (S(op, bound), M_NOT_REVERSED),
            (S(_FLIP[op], Fr(c + b, -a)), M_SIGN),
            (S(_FLIP[op], Fr(c - b, a)), M_BOUND_SIGN),
            (S(_STRICT_SWAP[_FLIP[op]], bound), M_STRICT),
        ],
        working=[
            f"Subtract {b}: {fmt_signed(-a, v)} {op} {c - b}.",
            f"Divide by {-a} and reverse: {v} {_FLIP[op]} {_rt(bound)}.",
        ],
    )


def _draw_both_negative(rng, v):
    """px + q op rx + s with p < r, so the x coefficient is negative once the x terms are collected on the left."""
    for _ in range(_MAX_DRAWS):
        p, r = rng.randint(1, 9), rng.randint(1, 9)
        if p >= r:
            continue
        q, s, op = _nz(rng, -15, 15), _nz(rng, -20, 20), rng.choice(_OPS)
        k = p - r
        bound = Fr(s - q, k)
        S = lambda o, b: _single_text(v, o, b)  # noqa: E731
        return dict(
            expr=f"{join_terms([(p, v), (q, '')])} {op} {join_terms([(r, v), (s, '')])}",
            sat=lambda x: _holds(p * x + q, op, r * x + s),
            op=_FLIP[op],
            bound=bound,
            cands=[
                (S(op, bound), M_NOT_REVERSED),
                (S(_FLIP[op], Fr(s + q, k)), M_SIGN),
                (S(_FLIP[op], Fr(s - q, p + r)), M_SIGN),
                (S(_STRICT_SWAP[_FLIP[op]], bound), M_STRICT),
            ],
            working=[
                f"Collect the {v} terms on the left: {fmt_signed(k, v)} {op} {s - q}.",
                f"Divide by {k} (negative) and reverse: {v} {_FLIP[op]} {_rt(bound)}.",
            ],
        )
    raise VerificationError("no both-sides draw with a negative net coefficient")


def _draw_two_step_basic(rng, v, *, fractional: bool):
    a, b, op = rng.randint(2, 9), _nz(rng, -15, 15), rng.choice(_OPS)
    x0 = Fr(rng.randint(-10, 12))
    c = int(a * x0 + b)
    if fractional:
        c += rng.randint(1, a - 1)
    bound = Fr(c - b, a)
    S = lambda o, k: _single_text(v, o, k)  # noqa: E731
    return dict(
        expr=f"{join_terms([(a, v), (b, '')])} {op} {c}",
        sat=lambda x: _holds(a * x + b, op, c),
        op=op,
        bound=bound,
        cands=[
            (S(_FLIP[op], bound), M_REVERSED_UNPROMPTED),
            (S(op, Fr(c, a) - b), M_ONE_TERM),
            (S(op, Fr(c + b, a)), M_SIGN),
            (S(op, c - b), M_NO_DIVIDE),
            (S(_STRICT_SWAP[op], bound), M_STRICT),
        ],
        working=[f"Subtract {b}: {a}{v} {op} {c - b}.", f"Divide by {a}: {v} {op} {_rt(bound)}."],
    )


def _draw_two_step_bracket(rng, v):
    a, m, op = rng.randint(2, 6), _nz(rng, -9, 9), rng.choice(_OPS)
    c = a * rng.randint(-6, 9) + (a * m if rng.random() < 0.5 else 0) + rng.choice([0, 0, 1, 2])
    if c == 0:
        c = a * m + 1
    bound = Fr(c, a) - m
    S = lambda o, k: _single_text(v, o, k)  # noqa: E731
    return dict(
        expr=f"{a}({_sgn_inner(v, m)}) {op} {c}",
        sat=lambda x: _holds(a * (x + m), op, c),
        op=op,
        bound=bound,
        cands=[
            (S(op, Fr(c - m, a)), M_ONE_TERM),
            (S(op, Fr(c, a) + m), M_SIGN),
            (S(_FLIP[op], bound), M_REVERSED_UNPROMPTED),
            (S(op, c - a * m), M_NO_DIVIDE),
            (S(_STRICT_SWAP[op], bound), M_STRICT),
        ],
        working=[f"Divide both sides by {a}: {_sgn_inner(v, m)} {op} {_rt(Fr(c, a))}.", f"Then {v} {op} {_rt(bound)}."],
    )


def _draw_two_step_fraction(rng, v):
    a, b, op = rng.randint(2, 6), _nz(rng, -9, 9), rng.choice(_OPS)
    c = _nz(rng, -8, 8)
    bound = Fr(a * (c - b))
    S = lambda o, k: _single_text(v, o, k)  # noqa: E731
    if rng.random() < 0.5:
        return dict(
            expr=f"{v}/{a} {'+' if b > 0 else '-'} {abs(b)} {op} {c}",
            sat=lambda x: _holds(x / a + b, op, c),
            op=op,
            bound=bound,
            cands=[
                (S(op, a * c - b), M_ONE_TERM),
                (S(op, a * c + b), M_SIGN),
                (S(op, Fr(c - b, a)), M_WRONG_INVERSE),
                (S(_FLIP[op], bound), M_REVERSED_UNPROMPTED),
                (S(_STRICT_SWAP[op], bound), M_STRICT),
            ],
            working=[f"Subtract {b}: {v}/{a} {op} {c - b}.", f"Multiply by {a}: {v} {op} {a * (c - b)}."],
        )
    p = rng.randint(2, 5)
    q = _nz(rng, -9, 9)
    bound = Fr(a * c - q, p)
    return dict(
        expr=f"({join_terms([(p, v), (q, '')])})/{a} {op} {c}",
        sat=lambda x: _holds((p * x + q) / a, op, c),
        op=op,
        bound=bound,
        cands=[
            (S(op, Fr(a * c, p) - q), M_ONE_TERM),
            (S(op, Fr(a * c + q, p)), M_SIGN),
            (S(op, Fr(c - q, p)), M_NO_DIVIDE),
            (S(_FLIP[op], bound), M_REVERSED_UNPROMPTED),
            (S(_STRICT_SWAP[op], bound), M_STRICT),
        ],
        working=[f"Multiply by {a}: {join_terms([(p, v), (q, '')])} {op} {a * c}.", f"Then {v} {op} {_rt(bound)}."],
    )


def _draw_two_step_both(rng, v):
    """a(x + m) op cx + d, any sign of the net coefficient."""
    for _ in range(_MAX_DRAWS):
        a, c, m, d, op = rng.randint(2, 6), rng.randint(1, 8), _nz(rng, -6, 6), _nz(rng, -20, 20), rng.choice(_OPS)
        if a == c:
            continue
        k = a - c
        bound = Fr(d - a * m, k)
        ans_op = op if k > 0 else _FLIP[op]
        S = lambda o, b: _single_text(v, o, b)  # noqa: E731
        return dict(
            expr=f"{a}({_sgn_inner(v, m)}) {op} {join_terms([(c, v), (d, '')])}",
            sat=lambda x: _holds(a * (x + m), op, c * x + d),
            op=ans_op,
            bound=bound,
            cands=[
                (S(ans_op, Fr(d - m, k)), M_ONE_TERM),
                (S(_FLIP[ans_op], bound), M_NOT_REVERSED if k < 0 else M_REVERSED_UNPROMPTED),
                (S(ans_op, Fr(d + a * m, k)), M_SIGN),
                (S(ans_op, Fr(d - a * m, a + c)), M_SIGN),
                (S(_STRICT_SWAP[ans_op], bound), M_STRICT),
            ],
            working=[
                f"Expand: {join_terms([(a, v), (a * m, '')])} {op} {join_terms([(c, v), (d, '')])}.",
                f"Collect: {fmt_signed(k, v)} {op} {d - a * m}; then {v} {ans_op} {_rt(bound)}.",
            ],
        )
    raise VerificationError("no both-sides bracket draw")


def _finish_single(d: dict, v: str, rng) -> dict | None:
    """Verify a single-inequality draw and package it as a question dict, or None to redraw."""
    _verify_single(d["sat"], d["op"], d["bound"])
    ans = _single_text(v, d["op"], d["bound"])
    dis = _tagged(ans, d["cands"])
    if dis is None:
        return None
    return dict(
        text=_prompt(rng, v, d["expr"]),
        answer=ans,
        answer_text=ans,
        answer_latex=latex_answer(ans, plain=True),
        working=d["working"],
        distractors=dis,
    )


def _practice(draws):
    """A generator body that picks one of `draws` per question and retries until it has two usable distractors."""

    def body(rng: random.Random):
        for _ in range(_MAX_DRAWS):
            v = rng.choice(_VARS)
            out = _finish_single(rng.choice(draws)(rng, v), v, rng)
            if out is not None:
                return out
        raise VerificationError("no usable inequality draw")

    return body


# -- compound (double) inequalities ----------------------------------------------------------------------------------


def _draw_compound(rng, v, *, allow_negative: bool = True):
    """``L op1 p x + q op2 U``; a negative p reverses the order of the answer."""
    for _ in range(_MAX_DRAWS):
        p = rng.choice([2, 3, 4, 5, 2, 3] + ([-2, -3, -4] if allow_negative else []))
        q = _nz(rng, -9, 9)
        lo_val = rng.randint(-12, 12)
        hi_val = lo_val + rng.randint(4, 24)
        op1, op2 = rng.choice(("<", "≤")), rng.choice(("<", "≤"))
        a, b = Fr(lo_val - q, p), Fr(hi_val - q, p)
        if p > 0:
            lo, lo_op, hi, hi_op = a, op1, b, op2
            slip_lo, sign_lo, sign_hi = Fr(lo_val, p), Fr(lo_val + q, p), Fr(hi_val + q, p)
        else:
            lo, lo_op, hi, hi_op = b, op2, a, op1
            slip_lo, sign_lo, sign_hi = Fr(hi_val, p), Fr(hi_val + q, p), Fr(lo_val + q, p)
        mid = join_terms([(p, v), (q, "")]) if p > 0 else join_terms([(q, ""), (p, v)])
        if hi - lo < 2 or hi - lo > 14:
            continue
        S = lambda l_, lo_o, h_, hi_o: _range_text(v, l_, lo_o, h_, hi_o)  # noqa: E731
        cands = [
            (S(slip_lo, lo_op, hi, hi_op), M_COMPOUND_PART),
            (S(sign_lo, lo_op, sign_hi, hi_op), M_SIGN),
            (S(lo, _STRICT_SWAP[lo_op], hi, _STRICT_SWAP[hi_op]), M_STRICT),
        ]
        if p < 0:
            cands.insert(0, (S(a, op1, b, op2), M_NOT_REVERSED))
        text_expr = f"{lo_val} {op1} {mid} {op2} {hi_val}"
        return dict(
            expr=text_expr,
            sat=lambda x: _holds(lo_val, op1, p * x + q) and _holds(p * x + q, op2, hi_val),
            lo=lo,
            lo_op=lo_op,
            hi=hi,
            hi_op=hi_op,
            cands=cands,
            working=[
                f"Subtract {q} from every part: {lo_val - q} {op1} {fmt_signed(p, v)} {op2} {hi_val - q}.",
                f"Divide every part by {p}" + (" and reverse both signs." if p < 0 else "."),
                f"{_range_text(v, lo, lo_op, hi, hi_op)}.",
            ],
        )
    raise VerificationError("no compound draw")


def _finish_compound(d: dict, v: str, rng) -> dict | None:
    _verify_range(d["sat"], d["lo"], d["lo_op"], d["hi"], d["hi_op"])
    ans = _range_text(v, d["lo"], d["lo_op"], d["hi"], d["hi_op"])
    dis = _tagged(ans, d["cands"])
    if dis is None:
        return None
    text = f"{rng.choice(['Solve', 'Solve the double inequality'])} {d['expr']}. {_ASK.format(v=v)}"
    return dict(
        text=text,
        answer=ans,
        answer_text=ans,
        answer_latex=latex_answer(ans, plain=True),
        working=d["working"],
        distractors=dis,
    )


# -- practice topics -------------------------------------------------------------------------------------------------

_ONE_STEP = {
    "easy": _practice([_draw_add]),
    "medium": _practice([_draw_mult]),
    "hard": _practice([_draw_div, _draw_bracket_one_step]),
}
_NEGATIVE = {
    "easy": _practice([_draw_neg_mult]),
    "medium": _practice([_draw_neg_div, _draw_b_minus_ax]),
    "hard": _practice([_draw_both_negative]),
}


def _two_step_hard(rng: random.Random):
    for _ in range(_MAX_DRAWS):
        v = rng.choice(_VARS)
        if rng.random() < 0.5:
            out = _finish_compound(_draw_compound(rng, v), v, rng)
        else:
            out = _finish_single(_draw_two_step_both(rng, v), v, rng)
        if out is not None:
            return out
    raise VerificationError("no usable two-step hard draw")


_TWO_STEP = {
    "easy": _practice([lambda r, v: _draw_two_step_basic(r, v, fractional=False)]),
    "medium": _practice(
        [
            lambda r, v: _draw_two_step_basic(r, v, fractional=True),
            _draw_two_step_bracket,
            _draw_two_step_fraction,
            _draw_b_minus_ax,
        ]
    ),
    "hard": _two_step_hard,
}


@generator(
    "lin_ineq_one_step",
    level=7,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def lin_ineq_one_step(rng: random.Random, tier: str = "easy") -> dict:
    return _ONE_STEP[tier](rng)


@generator(
    "lin_ineq_negative",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def lin_ineq_negative(rng: random.Random, tier: str = "easy") -> dict:
    return _NEGATIVE[tier](rng)


@generator(
    "lin_ineq_two_step",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def lin_ineq_two_step(rng: random.Random, tier: str = "easy") -> dict:
    return _TWO_STEP[tier](rng)


def _wsingle(var, op, bound):
    return _single_text(var, op, bound, dec=True)


def _wrange(var, lo, lo_op, hi, hi_op):
    return _range_text(var, lo, lo_op, hi, hi_op, dec=True)


# -- word problems ---------------------------------------------------------------------------------------------------

_LIMIT = {
    "≤": ["no more than", "at most", "not more than", "no greater than"],
    "<": ["less than", "under", "below"],
    "≥": ["at least", "no less than", "not less than", "a minimum of"],
    ">": ["more than", "over", "greater than"],
}
_WHO = ["Priya", "Tom", "Alicia", "Wei", "Sam", "Mia", "Jack", "Zoe", "Omar", "Nina"]

# slope = rate per unit; `const` is the fixed amount added (cost) or taken off (target) before comparing with the limit.
# Each context: stem with {who} {r} {f} {limit} {T}; `var`, `meaning` and the sign of the constant.
_EASY = [
    dict(
        stem="Each notebook costs ${r}. {who} wants the total cost of the notebooks to be {limit} ${T}.",
        var="n",
        meaning="the number of notebooks bought",
        ops="≤<",
        rate=(2, 9),
    ),
    dict(
        stem="Bottles of juice hold {r} L each. For a party, {who} needs the total volume of juice to be {limit} {T} L.",
        var="b",
        meaning="the number of bottles",
        ops="≥>",
        rate=(2, 6),
    ),
    dict(
        stem="Tickets to the school fete cost ${r} each. The school wants the ticket sales to be {limit} ${T}.",
        var="t",
        meaning="the number of tickets sold",
        ops="≥>",
        rate=(3, 12),
    ),
    dict(
        stem="A lift is allowed to carry a total of {limit} {T} kg. Each parcel has a mass of {r} kg.",
        var="p",
        meaning="the number of parcels in the lift",
        ops="≤<",
        rate=(3, 15),
    ),
]
_MEDIUM = [
    dict(
        stem="A mobile plan costs ${f} a month plus ${r} for each gigabyte (GB) of data used. {who} wants the monthly bill to be {limit} ${T}.",
        var="g",
        meaning="the number of gigabytes of data used",
        amount="the monthly bill in dollars",
        ops="≤<",
        sign=1,
        rate=(2, 9),
        fixed=(10, 45),
    ),
    dict(
        stem="A taxi charges a flag fall of ${f} plus ${r} per kilometre. {who} wants the fare to be {limit} ${T}.",
        var="k",
        meaning="the distance travelled in kilometres",
        amount="the fare in dollars",
        ops="≤<",
        sign=1,
        rate=(2, 5),
        fixed=(3, 9),
    ),
    dict(
        stem="A delivery van has a mass of {f} kg when empty and carries crates of {r} kg each. The total mass of the loaded van must be {limit} {T} kg.",
        var="c",
        meaning="the number of crates loaded",
        amount="the total mass of the loaded van in kilograms",
        ops="≤<",
        sign=1,
        rate=(5, 25),
        fixed=(1200, 2400),
    ),
    dict(
        stem="{who} has saved ${f} and adds ${r} each week. {who} wants the total saved to be {limit} ${T}.",
        var="w",
        meaning="the number of weeks of saving",
        amount="the total saved in dollars",
        ops="≥>",
        sign=1,
        rate=(10, 40),
        fixed=(40, 200),
    ),
    dict(
        stem="A netball club sells raffle tickets for ${r} each and has ${f} of costs to cover. It wants the money left after costs to be {limit} ${T}.",
        var="n",
        meaning="the number of tickets sold",
        ops="≥>",
        sign=-1,
        amount="the money left after costs in dollars",
        rate=(2, 8),
        fixed=(20, 90),
    ),
]


def _word_single(rng: random.Random, ctx: dict, *, tier: str):
    """Draw one fixed-plus-rate (or rate-only) word problem: ``s*v + c0 op T`` with slope s > 0."""
    for _ in range(_MAX_DRAWS):
        op = rng.choice(list(ctx["ops"]))
        r = rng.choice([v for v in (2, 4, 5, 10, 20, 25) if ctx["rate"][0] <= v <= ctx["rate"][1]])
        f = rng.randint(*ctx["fixed"]) if "fixed" in ctx else 0
        c0 = f * ctx.get("sign", 1)
        n_target = rng.randint(3, 30)
        T = r * n_target + c0 + (rng.randint(1, r - 1) if rng.random() < 0.65 else 0)
        if T <= 0:
            continue
        bound = Fr(T - c0, r)
        if bound < 1 or 100 % bound.denominator:
            continue
        return op, r, f, c0, T, bound
    raise VerificationError("no word draw")


def _word_cands(v, op, r, c0, T, bound):
    S = lambda o, k: _wsingle(v, o, k)  # noqa: E731
    cands = [
        (S(_FLIP[op], bound), M_SWAPPED_CONTEXT),
        (S(_STRICT_SWAP[op], bound), M_STRICT),
    ]
    if c0:
        cands += [
            (S(op, Fr(T, r)), M_IGNORED_CONST),
            (S(op, Fr(T + c0, r)), M_SIGN),
            (S(op, T - c0), M_NO_DIVIDE),
        ]
    else:
        cands += [(S(op, T), M_NO_DIVIDE), (S(op, T * r), M_WRONG_INVERSE)]
    return cands


def _draw_word_basic(rng, tier):
    ctx = rng.choice(_EASY if tier == "easy" else _MEDIUM)
    who = rng.choice(_WHO)
    op, r, f, c0, T, bound = _word_single(rng, ctx, tier=tier)
    v = ctx["var"]
    limit = rng.choice(_LIMIT[op])
    stem = ctx["stem"].format(who=who, r=r, f=f, T=T, limit=limit)
    sat = lambda x: _holds(r * x + c0, op, T)  # noqa: E731
    return dict(
        text=f"{stem} Let {v} be {ctx['meaning']}. Write an inequality for {v} and solve it. {_ASKW.format(v=v)}",
        sat=sat,
        op=op,
        bound=bound,
        v=v,
        cands=_word_cands(v, op, r, c0, T, bound),
        working=[
            f"{join_terms([(r, v), (c0, '')])} {op} {T}",
            f"{fmt_signed(r, v)} {op} {T - c0}, so {v} {op} {_rt(bound)}.",
        ],
    )


def _draw_word_plans(rng):
    for _ in range(_MAX_DRAWS):
        f1, f2 = rng.randint(10, 40), rng.randint(20, 70)
        r1, r2 = rng.randint(3, 9), rng.randint(1, 7)
        if not (f2 > f1 and r1 > r2):
            continue
        who, ask_b = rng.choice(_WHO), rng.random() < 0.5
        v = "g"
        # statement: B cheaper than A (or A cheaper than B)
        if ask_b:
            sat = lambda x: f2 + r2 * x < f1 + r1 * x  # noqa: E731
            k, rhs, op = r2 - r1, f1 - f2, "<"
            nm = "B"
        else:
            sat = lambda x: f1 + r1 * x < f2 + r2 * x  # noqa: E731
            k, rhs, op = r1 - r2, f2 - f1, "<"
            nm = "A"
        bound = Fr(rhs, k)
        ans_op = op if k > 0 else _FLIP[op]
        if bound <= 0 or 100 % bound.denominator:
            continue
        S = lambda o, b_: _wsingle(v, o, b_)  # noqa: E731
        stem = (
            f"Plan A costs ${f1} a month plus ${r1} per gigabyte (GB) of data. Plan B costs ${f2} a month plus ${r2} per GB. "
            f"{who} wants to know when Plan {nm} costs less than the other plan. Let {v} be the number of gigabytes used in a month. "
            f"Write an inequality for {v} and solve it. {_ASKW.format(v=v)}"
        )
        return dict(
            text=stem,
            sat=sat,
            op=ans_op,
            bound=bound,
            v=v,
            cands=[
                (S(_FLIP[ans_op], bound), M_SWAPPED_CONTEXT if k > 0 else M_NOT_REVERSED),
                (S(ans_op, Fr(rhs, abs(k)) * -1), M_SIGN),
                (S(_STRICT_SWAP[ans_op], bound), M_STRICT),
                (S(ans_op, Fr(rhs, f1 + f2)), M_ONE_TERM),
            ],
            working=[f"Plan {nm} is cheaper when {fmt_signed(k, v)} < {rhs}, so {v} {ans_op} {_rt(bound)}."],
        )
    raise VerificationError("no plans draw")


def _draw_word_average(rng):
    for _ in range(_MAX_DRAWS):
        n_tests = rng.choice([3, 4])
        scores = [rng.randint(52, 92) for _ in range(n_tests)]
        need = rng.randint(55, 98)
        total = sum(scores) + need
        if total % (n_tests + 1):
            continue
        m = total // (n_tests + 1)
        if not 55 <= m <= 90:
            continue
        who, v = rng.choice(_WHO), "x"
        listed = ", ".join(str(s) for s in scores[:-1]) + f" and {scores[-1]}"
        S = lambda o, b_: _wsingle(v, o, b_)  # noqa: E731
        sat = lambda x: _holds((sum(scores) + x) / (n_tests + 1), "≥", m)  # noqa: E731
        bound = Fr(need)
        word = {3: "three", 4: "four"}[n_tests]
        nxt = {3: "fourth", 4: "fifth"}[n_tests]
        return dict(
            text=(
                f"{who} scored {listed} on {word} tests. To pass the unit, {who} needs a mean of at least {m} over {n_tests + 1} tests. "
                f"Let {v} be the mark on the {nxt} test. Write an inequality for {v} and solve it. {_ASKW.format(v=v)}"
            ),
            sat=sat,
            op="≥",
            bound=bound,
            v=v,
            cands=[
                (S("≤", bound), M_SWAPPED_CONTEXT),
                (S("≥", Fr(m * (n_tests + 1)) / (n_tests + 1) - Fr(sum(scores), n_tests + 1)), M_ONE_TERM),
                (S("≥", m * (n_tests + 1) + sum(scores)), M_SIGN),
                (S(">", bound), M_STRICT),
            ],
            working=[f"({'+'.join(str(s) for s in scores)} + {v})/{n_tests + 1} ≥ {m}", f"{v} ≥ {need}."],
        )
    raise VerificationError("no average draw")


def _draw_word_between(rng):
    """``L op1 f + r k op2 U``: a fare or bill that must lie between two amounts."""
    for _ in range(_MAX_DRAWS):
        f, r = rng.randint(3, 12), rng.randint(2, 5)
        k_lo, k_hi = Fr(rng.randint(2, 12)) + Fr(rng.randint(0, 2), 2), 0
        lo_cost = int(f + r * k_lo) if (r * k_lo).denominator == 1 else None
        if lo_cost is None:
            continue
        L = lo_cost - rng.choice([0, 0, 1])
        U = L + rng.randint(8, 40)
        op1, op2 = rng.choice(("<", "≤")), rng.choice(("<", "≤"))
        lo, hi = Fr(L - f, r), Fr(U - f, r)
        if hi - lo < 2 or lo <= 0 or 100 % lo.denominator or 100 % hi.denominator:
            continue
        v, who = "k", rng.choice(_WHO)
        word1 = "more than" if op1 == "<" else "at least"
        word2 = "less than" if op2 == "<" else "no more than"
        sat = lambda x: _holds(L, op1, f + r * x) and _holds(f + r * x, op2, U)  # noqa: E731
        S = lambda l_, a_, h_, b_: _wrange(v, l_, a_, h_, b_)  # noqa: E731
        return dict(
            text=(
                f"A taxi charges a flag fall of ${f} plus ${r} per kilometre. {who}'s fare for a trip was {word1} ${L} but {word2} ${U}. "
                f"Let {v} be the distance travelled in kilometres. Write an inequality for {v} and solve it. {_ASKW.format(v=v)}"
            ),
            sat=sat,
            lo=lo,
            lo_op=op1,
            hi=hi,
            hi_op=op2,
            v=v,
            cands=[
                (S(Fr(L, r), op1, hi, op2), M_COMPOUND_PART),
                (S(lo, _STRICT_SWAP[op1], hi, _STRICT_SWAP[op2]), M_STRICT),
                (S(Fr(L + f, r), op1, Fr(U + f, r), op2), M_SIGN),
                (S(hi, op1, lo, op2), M_SWAPPED_CONTEXT),
            ],
            working=[f"{L} {op1} {f} + {r}{v} {op2} {U}", f"{_wrange(v, lo, op1, hi, op2)}."],
        )
    raise VerificationError("no between draw")


def _finish_word(d: dict, rng) -> dict | None:
    v = d["v"]
    if "lo" in d:
        _verify_range(d["sat"], d["lo"], d["lo_op"], d["hi"], d["hi_op"])
        ans = _wrange(v, d["lo"], d["lo_op"], d["hi"], d["hi_op"])
    else:
        _verify_single(d["sat"], d["op"], d["bound"])
        ans = _wsingle(v, d["op"], d["bound"])
    dis = _tagged(ans, d["cands"])
    if dis is None:
        return None
    return dict(
        text=d["text"],
        answer=ans,
        answer_text=ans,
        answer_latex=latex_answer(ans, plain=True),
        working=d["working"],
        distractors=dis,
    )


def _word_body(rng: random.Random, tier: str) -> dict:
    for _ in range(_MAX_DRAWS):
        if tier in ("easy", "medium"):
            d = _draw_word_basic(rng, tier)
        else:
            d = rng.choice(
                [lambda: _draw_word_plans(rng), lambda: _draw_word_average(rng), lambda: _draw_word_between(rng)]
            )()
        out = _finish_word(d, rng)
        if out is not None:
            return out
    raise VerificationError("no usable word problem")


@generator(
    "lin_ineq_word",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def lin_ineq_word(rng: random.Random, tier: str = "easy") -> dict:
    return _word_body(rng, tier)


# -- exam chains -----------------------------------------------------------------------------------------------------


def _int_extreme(op: str, bound: Fr) -> tuple[int, str]:
    """The greatest (for < and ≤) or least (for > and ≥) integer solution of ``x op bound``, and which it is."""
    if op == "≤":
        return math.floor(bound), "greatest"
    if op == "<":
        return math.ceil(bound) - 1, "greatest"
    if op == "≥":
        return math.ceil(bound), "least"
    return math.floor(bound) + 1, "least"


def _num_dist(answer: int, cands) -> list[Distractor] | None:
    out, seen = [], {str(answer)}
    for value, mid in cands:
        if str(value) in seen:
            continue
        seen.add(str(value))
        out.append(Distractor(str(value), mid, value=int(value)))
        if len(out) == 3:
            break
    return out if len(out) >= 2 else None


def _extreme_distractors(op: str, bound: Fr, n: int):
    """Wrong integers for 'the greatest/least integer solution': the rounding slip, the strict bound itself and
    the neighbours."""
    cands = []
    if op in ("<", "≤"):
        cands.append((math.ceil(bound) if op == "≤" else math.floor(bound), M_ROUNDS_WRONG))
        cands.append((math.ceil(bound), M_STRICT_VALUE if op == "<" else M_ROUNDS_WRONG))
    else:
        cands.append((math.floor(bound) if op == "≥" else math.ceil(bound), M_ROUNDS_WRONG))
        cands.append((math.floor(bound), M_STRICT_VALUE if op == ">" else M_ROUNDS_WRONG))
    cands += [(n - 1, M_OFF_BY_ONE), (n + 1, M_OFF_BY_ONE)]
    return cands


def _expr_choice(rng: random.Random, v: str, *, negative_slope: bool):
    k = rng.randint(2, 7) * (-1 if negative_slope else 1)
    c = _nz(rng, -9, 12)
    text = join_terms([(c, ""), (k, v)]) if k < 0 else join_terms([(k, v), (c, "")])
    return k, c, text


def _integer_chain(rng: random.Random, draw, *, negative_slope: bool) -> dict:
    for _ in range(_MAX_DRAWS):
        v = "x"
        d = draw(rng, v)
        _verify_single(d["sat"], d["op"], d["bound"])
        ans = _single_text(v, d["op"], d["bound"])
        dis_a = _tagged(ans, d["cands"])
        if dis_a is None:
            continue
        n, which = _int_extreme(d["op"], d["bound"])
        check(d["sat"](Fr(n)) and not d["sat"](Fr(n + 1 if which == "greatest" else n - 1)), "integer extreme is wrong")
        dis_b = _num_dist(n, _extreme_distractors(d["op"], d["bound"], n))
        k, c, etext = _expr_choice(rng, v, negative_slope=negative_slope)
        value = k * n + c
        if value == 0 or value == n:
            continue
        c_cands = [(k * w + c, mid) for w, mid in _extreme_distractors(d["op"], d["bound"], n)]
        c_cands.append((k * n - c, M_SUBSTITUTION))
        dis_c = _num_dist(value, c_cands)
        if dis_b is None or dis_c is None:
            continue
        return dict(
            text=f"Consider the inequality {d['expr']}.",
            parts=[
                dict(
                    prompt=f"Solve the inequality. {_ASK.format(v=v)}",
                    answer_kind="inequality",
                    answer=ans,
                    answer_text=ans,
                    answer_latex=latex_answer(ans, plain=True),
                    marks=2,
                    working=d["working"],
                    distractors=dis_a,
                ),
                dict(
                    prompt=f"Write down the {which} integer value of {v} that satisfies the inequality.",
                    answer_kind="number",
                    answer=n,
                    answer_text=str(n),
                    answer_latex=str(n),
                    marks=1,
                    working=[f"From {ans}, the {which} integer is {n}."],
                    distractors=dis_b,
                ),
                dict(
                    prompt=f"Hence find the value of {etext} when {v} is that integer.",
                    answer_kind="number",
                    answer=value,
                    answer_text=str(value),
                    answer_latex=str(value),
                    marks=1,
                    follow_through=f"{k}*part_b + ({c})",
                    working=[f"{etext} with {v} = {n} is {value}."],
                    distractors=dis_c,
                ),
            ],
        )
    raise VerificationError("no usable integer chain")


@generator("exd_x_ineq_one_step", level=10, presets={"exam": dict(tier="exam")})
def exd_x_ineq_one_step(rng: random.Random, tier: str = "exam") -> dict:
    return _integer_chain(
        rng,
        lambda r, v: r.choice([_draw_div, _draw_bracket_one_step, _draw_bracket_one_step])(r, v),
        negative_slope=False,
    )


@generator("exd_x_ineq_negative", level=10, presets={"exam": dict(tier="exam")})
def exd_x_ineq_negative(rng: random.Random, tier: str = "exam") -> dict:
    return _integer_chain(
        rng,
        lambda r, v: r.choice([_draw_b_minus_ax, _draw_both_negative, _draw_both_negative])(r, v),
        negative_slope=True,
    )


@generator("exd_x_ineq_two_step", level=11, presets={"exam": dict(tier="exam")})
def exd_x_ineq_two_step(rng: random.Random, tier: str = "exam") -> dict:
    """(a) solve a double inequality, (b) how many integers satisfy it, (c) their sum."""
    for _ in range(_MAX_DRAWS):
        v = "x"
        d = _draw_compound(rng, v)
        _verify_range(d["sat"], d["lo"], d["lo_op"], d["hi"], d["hi_op"])
        ans = _range_text(v, d["lo"], d["lo_op"], d["hi"], d["hi_op"])
        dis_a = _tagged(ans, d["cands"])
        if dis_a is None:
            continue
        ints = [i for i in range(math.floor(d["lo"]) - 1, math.ceil(d["hi"]) + 2) if d["sat"](Fr(i))]
        if not ints:
            continue
        check(ints == list(range(ints[0], ints[-1] + 1)), "integer solutions are not consecutive")
        count, total = len(ints), sum(ints)
        if not 3 <= count <= 12 or total == 0:
            continue
        flipped = [
            i
            for i in range(math.floor(d["lo"]) - 2, math.ceil(d["hi"]) + 3)
            if _holds(d["lo"], _STRICT_SWAP[d["lo_op"]], Fr(i)) and _holds(Fr(i), _STRICT_SWAP[d["hi_op"]], d["hi"])
        ]
        cnt_cands = [
            (len(flipped), M_OFF_BY_ONE),
            (math.floor(d["hi"]) - math.ceil(d["lo"]), M_OFF_BY_ONE),
            (count + 1, M_OFF_BY_ONE),
            (count - 1, M_OFF_BY_ONE),
        ]
        sum_cands = [
            (sum(flipped), M_OFF_BY_ONE),
            (total + ints[-1] + 1, M_OFF_BY_ONE),
            (total - ints[0], M_OFF_BY_ONE),
            (total + ints[0] - 1, M_OFF_BY_ONE),
        ]
        dis_b, dis_c = _num_dist(count, cnt_cands), _num_dist(total, sum_cands)
        if dis_b is None or dis_c is None:
            continue
        return dict(
            text=f"Consider the double inequality {d['expr']}.",
            parts=[
                dict(
                    prompt=f"Solve the double inequality. {_ASK.format(v=v)}",
                    answer_kind="inequality",
                    answer=ans,
                    answer_text=ans,
                    answer_latex=latex_answer(ans, plain=True),
                    marks=2,
                    working=d["working"],
                    distractors=dis_a,
                ),
                dict(
                    prompt=f"How many integer values of {v} satisfy the double inequality?",
                    answer_kind="number",
                    answer=count,
                    answer_text=str(count),
                    answer_latex=str(count),
                    marks=1,
                    working=[f"The integers are {', '.join(str(i) for i in ints)}: {count} of them."],
                    distractors=dis_b,
                ),
                dict(
                    prompt=f"Find the sum of all the integer values of {v} that satisfy the double inequality.",
                    answer_kind="number",
                    answer=total,
                    answer_text=str(total),
                    answer_latex=str(total),
                    marks=1,
                    working=[f"{' + '.join(paren_negative(i) for i in ints)} = {total}."],
                    distractors=dis_c,
                ),
            ],
        )
    raise VerificationError("no usable two-step exam draw")


@generator("exd_x_ineq_word", level=11, presets={"exam": dict(tier="exam")})
def exd_x_ineq_word(rng: random.Random, tier: str = "exam") -> dict:
    """(a) the inequality, (b) greatest/least whole number, (c) the cost or amount raised there (follow-through),
    (d) the range once a second constraint is added."""
    for _ in range(_MAX_DRAWS):
        who = rng.choice(_WHO)
        ctx = rng.choice([c for c in _MEDIUM if "amount" in c])
        op, r, f, c0, T, bound = _word_single(rng, ctx, tier="exam")
        if bound < 5:
            continue
        v = ctx["var"]
        n, which = _int_extreme(op, bound)
        sat = lambda x: _holds(r * x + c0, op, T)  # noqa: E731
        _verify_single(sat, op, bound)
        check(sat(Fr(n)) and not sat(Fr(n + 1 if which == "greatest" else n - 1)), "word extreme is wrong")
        ans_a = _wsingle(v, op, bound)
        dis_a = _tagged(ans_a, _word_cands(v, op, r, c0, T, bound))
        if dis_a is None:
            continue
        dis_b = _num_dist(n, _extreme_distractors(op, bound, n))
        value = r * n + c0
        c_cands = [(r * w + c0, mid) for w, mid in _extreme_distractors(op, bound, n)]
        c_cands += [(r * n, M_IGNORED_CONST), (T, M_SUBSTITUTION)]
        dis_c = _num_dist(value, c_cands)
        if dis_b is None or dis_c is None or value <= 0:
            continue
        # (d) a second constraint on the other side of the same variable.
        if which == "greatest":
            m = rng.randint(1, max(1, n - 2))
            lo, lo_op, hi, hi_op = Fr(m), "≤", bound, op
            clause = f"at least {m}"
            extra = lambda p: p >= m  # noqa: E731
        else:
            m = n + rng.randint(2, 12)
            lo, lo_op, hi, hi_op = bound, ("≤" if op == "≥" else "<"), Fr(m), "≤"
            clause = f"at most {m}"
            extra = lambda p: p <= m  # noqa: E731
        ans_d = _wrange(v, lo, lo_op, hi, hi_op)
        for p in _probes([lo, hi]):
            check((sat(p) and extra(p)) == (_holds(lo, lo_op, p) and _holds(p, hi_op, hi)), "range disagrees")
        S = lambda l_, a_, h_, b_: _wrange(v, l_, a_, h_, b_)  # noqa: E731
        if which == "greatest":
            d_cands = [
                (ans_a, M_ONE_CONSTRAINT),
                (S(lo, lo_op, hi, _STRICT_SWAP[hi_op]), M_STRICT),
                (S(hi, hi_op, lo, lo_op), M_SWAPPED_CONTEXT),
                (_wsingle(v, "≥", m), M_ONE_CONSTRAINT),
            ]
        else:
            d_cands = [
                (ans_a, M_ONE_CONSTRAINT),
                (S(lo, _STRICT_SWAP[lo_op], hi, hi_op), M_STRICT),
                (S(hi, hi_op, lo, lo_op), M_SWAPPED_CONTEXT),
                (_wsingle(v, "≤", m), M_ONE_CONSTRAINT),
            ]
        dis_d = _tagged(ans_d, d_cands)
        if dis_d is None:
            continue
        stem = (
            ctx["stem"].format(who=who, r=r, f=f, T=T, limit=rng.choice(_LIMIT[op])) + f" Let {v} be {ctx['meaning']}."
        )
        return dict(
            text=stem,
            parts=[
                dict(
                    prompt=f"Write an inequality for {v} and solve it. {_ASKW.format(v=v)}",
                    answer_kind="inequality",
                    answer=ans_a,
                    answer_text=ans_a,
                    answer_latex=latex_answer(ans_a, plain=True),
                    marks=2,
                    working=[f"{join_terms([(r, v), (c0, '')])} {op} {T}", f"{v} {op} {_rt(bound)}."],
                    distractors=dis_a,
                ),
                dict(
                    prompt=f"What is the {which} whole number value of {v} that satisfies your inequality?",
                    answer_kind="number",
                    answer=n,
                    answer_text=str(n),
                    answer_latex=str(n),
                    marks=1,
                    sensible_range=(0, None),
                    working=[f"{v} must be a whole number, so the {which} value is {n}."],
                    distractors=dis_b,
                ),
                dict(
                    prompt=f"Hence find {ctx['amount']} when {v} is that whole number.",
                    answer_kind="number",
                    answer=value,
                    answer_text=str(value),
                    answer_latex=str(value),
                    marks=1,
                    sensible_range=(0, None),
                    follow_through=f"{r}*part_b + ({c0})",
                    working=[f"{r} × {n} {'+' if c0 >= 0 else '-'} {abs(c0)} = {value}."],
                    distractors=dis_c,
                ),
                dict(
                    prompt=f"Suppose also that {v} must be {clause}. Write the possible values of {v} as a single inequality. {_ASKW.format(v=v)}",
                    answer_kind="inequality",
                    answer=ans_d,
                    answer_text=ans_d,
                    answer_latex=latex_answer(ans_d, plain=True),
                    marks=2,
                    working=[f"Combine {ans_a} with {v} {'≥' if which == 'greatest' else '≤'} {m}: {ans_d}."],
                    distractors=dis_d,
                ),
            ],
        )
    raise VerificationError("no usable word exam draw")
