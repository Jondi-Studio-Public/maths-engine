"""Generates `ucv1_proof_by_induction` questions and the exam chain `exd_x_uc_induction`.

``ucv1-proof-by-induction`` (``proof-and-foundations``): the working steps of a proof by induction, asked as typed
answers (the rest of the domain is multiple choice). A statement P(n) about a sum or about divisibility is shown
and the learner supplies one step of its proof: a base-case value, the (k+1)th term, the right-hand side P(k+1)
must reach, the factored result of S_k + term, or the rearrangement that makes a divisibility step go. No
question ever asks which line of a proof is wrong.

Tiers (topic ``ucv1_proof_by_induction``):

* easy   -- the common value of both sides at n = 1 (or f(1) for a divisibility statement), or the (k+1)th term
  of the sum as an expression in k;
* medium -- the right-hand side that the inductive step must reach, i.e. the closed form with n replaced by
  k + 1, typed as an expression (any equivalent form is accepted);
* hard   -- the sum of the first (k + 1) terms as a single fully factorised expression, from a hypothesis given in
  EXPANDED form (``answer_form`` "factored"); or, for divisibility, the number or expression m in
  f(k+1) = a·f(k) + m (the prompt fixes a, so m is unique), or f(k+1) - f(k) in factored form for n³ - n.

The exam tier (topic ``exd_x_uc_induction``) is a four-part chain for a sum: (a) the value at n = 1, (b) the
(k+1)th term, (c) the factored S_(k+1) (follow-through from (b)), (d) a use of the closed form: S_N for a stated N
(follow-through from (c)) or the least n with S_n > T. One exam variant in five is the divisibility chain:
f(1), m, f(k+1)/d in terms of the quotient q (follow-through from m), and a value.

Every closed form is checked by brute force for n = 1..8 and the inductive identity
closed(k) + term(k+1) = closed(k+1) is checked symbolically and at sample points, so the statement is true.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache

import sympy
from sympy import Integer, Rational, Symbol, expand, factor

from ..answers import Distractor
from ..core import (
    VerificationError,
    attempts,
    check,
    check_distinct,
    check_equivalent,
    generator,
    pick_distractors,
)

_factor = lru_cache(maxsize=None)(factor)
k = Symbol("k")
n = Symbol("n")
q = Symbol("q")


# ---------------------------------------------------------------------------
# Text of expressions


def _txt(expr) -> str:
    """Learner-facing text of a SymPy expression: ^ for powers, and a product written by juxtaposition ("2k",
    "(k + 1)(k + 2)", "k(k + 1)") wherever that cannot be misread; a factor that ends in an exponent keeps its
    explicit star ("2^k*k"), and so does a product of two numbers ("6*3^k")."""
    s = str(expr).replace("**", "^")
    out = []
    for pos, ch in enumerate(s):
        if ch != "*":
            out.append(ch)
            continue
        nxt = s[pos + 1]
        before = s[:pos]
        if before.endswith(")"):
            depth, idx = 0, len(before) - 1
            while idx >= 0:
                depth += before[idx] == ")"
                depth -= before[idx] == "("
                if depth == 0:
                    break
                idx -= 1
            ends_in_exponent = idx > 0 and before[idx - 1] == "^"
        else:
            m = re.search(r"([A-Za-z0-9]+)$", before)
            start = m.start() if m else len(before)
            ends_in_exponent = start > 0 and before[start - 1] == "^"
            if ends_in_exponent and m and m.group(1).isdigit() and nxt == "(":
                ends_in_exponent = False  # "(k + 1)^2(k + 2)^2" cannot be misread
            if m and m.group(1).isdigit() and nxt.isdigit():
                ends_in_exponent = True  # number times number: keep the star
        if ends_in_exponent or not (nxt.isalpha() or nxt == "("):
            out.append("*")
    return "".join(out)


def _num_text(value) -> str:
    return str(int(value)) if Rational(value).q == 1 else str(value)


def _lin(c: int, d: int, var: str = "n") -> str:
    """'3n + 1', '5n - 1', 'n' ..."""
    head = var if c == 1 else f"{c}{var}"
    if d == 0:
        return head
    return f"{head} {'+' if d > 0 else '-'} {abs(d)}"


# ---------------------------------------------------------------------------
# Sum families. Each: term(x), closed(x), a way to print the sum, and how to print the closed form.


def _superscripts(text: str) -> str:
    """'n^2(n + 1)^2/4' as 'n²(n + 1)²/4' (squares and cubes of a letter or bracket only)."""
    return re.sub(r"(?<=[A-Za-z)])\^([23])(?!\d)", lambda m: "²³"[int(m.group(1)) - 2], text)


class _Family:
    def __init__(
        self,
        key,
        term,
        closed,
        first_terms,
        general,
        rhs,
        partial,
        factorable,
        multiplier=1,
    ):
        self.key = key
        self.term = term  # x -> sympy expression
        self.closed = closed  # x -> sympy expression
        self.first_terms = first_terms  # [str, str, str]
        self.general = general  # str
        self.rhs = rhs if rhs is not None else _superscripts(_txt(_factor(closed(n))))  # str with n
        self.partial = partial  # k -> expression: P(k+1)'s right-hand side with one n not updated
        self.factorable = factorable

    def disp(self, expr):
        """`expr` as shown to the learner: factored for the polynomial families, as built otherwise."""
        return _factor(expr) if self.factorable else sympy.together(expr)

    def sum_text(self) -> str:
        return " + ".join(self.first_terms) + f" + … + {self.general}"

    def statement(self) -> str:
        return f"{self.sum_text()} = {self.rhs}"

    def brute(self, upto: int):
        return [sum((self.term(Integer(i)) for i in range(1, m + 1)), Integer(0)) for m in range(1, upto + 1)]


def _draw_family(rng: random.Random, *, factorable_only: bool = False) -> _Family:
    kinds = ["ipa", "ipa", "sq", "sq", "cube", "cube", "lin", "lin", "lin", "oddsq", "trip", "mult"]
    if not factorable_only:
        kinds += ["pow2", "frac", "frac2", "geo"]
    kind = rng.choice(kinds)
    if kind == "ipa":
        a = rng.randint(1, 5)
        return _Family(
            f"ipa{a}",
            lambda x, a=a: x * (x + a),
            lambda x, a=a: x * (x + 1) * (2 * x + 3 * a + 1) / 6,
            [f"{i}×{i + a}" for i in (1, 2, 3)],
            f"n(n + {a})",
            None,
            lambda kk, a=a: (kk + 1) * (kk + 2) * (2 * kk + 3 * a + 1) / 6,
            True,
        )
    if kind in ("sq", "cube", "oddsq", "trip"):
        c = rng.choice([1, 1, 2, 3, 4, 5])
        lead = "" if c == 1 else f"{c}×"
        glead = "" if c == 1 else str(c)
        if kind == "sq":
            return _Family(
                f"sq{c}",
                lambda x, c=c: c * x**2,
                lambda x, c=c: c * x * (x + 1) * (2 * x + 1) / 6,
                [f"{lead}{i}²" for i in (1, 2, 3)],
                f"{glead}n²",
                None,
                lambda kk, c=c: c * (kk + 1) * (kk + 2) * (2 * kk + 1) / 6,
                True,
            )
        if kind == "cube":
            return _Family(
                f"cube{c}",
                lambda x, c=c: c * x**3,
                lambda x, c=c: c * x**2 * (x + 1) ** 2 / 4,
                [f"{lead}{i}³" for i in (1, 2, 3)],
                f"{glead}n³",
                None,
                lambda kk, c=c: c * (kk + 1) ** 2 * (kk + 1) ** 2 / 4,
                True,
            )
        if kind == "oddsq":
            return _Family(
                f"oddsq{c}",
                lambda x, c=c: c * (2 * x - 1) ** 2,
                lambda x, c=c: c * x * (2 * x - 1) * (2 * x + 1) / 3,
                [f"{lead}{2 * i - 1}²" for i in (1, 2, 3)],
                f"{glead}(2n - 1)²",
                None,
                lambda kk, c=c: c * (kk + 1) * (2 * kk + 1) * (2 * kk + 1) / 3,
                True,
            )
        return _Family(
            f"trip{c}",
            lambda x, c=c: c * x * (x + 1) * (x + 2),
            lambda x, c=c: c * x * (x + 1) * (x + 2) * (x + 3) / 4,
            [f"{lead}{i}×{i + 1}×{i + 2}" for i in (1, 2, 3)],
            f"{glead}n(n + 1)(n + 2)",
            None,
            lambda kk, c=c: c * (kk + 1) * (kk + 2) * (kk + 3) * (kk + 3) / 4,
            True,
        )
    if kind == "lin":
        for _ in attempts(100, "a linear term"):
            p, c = rng.randint(2, 7), rng.randint(1, 6)
            if p - c >= 1 and (p - 2 * c) != 0 and not (p == 2 and c == 1):
                break
        return _Family(
            f"lin{p}_{c}",
            lambda x, p=p, c=c: p * x - c,
            lambda x, p=p, c=c: x * (p * (x + 1) - 2 * c) / 2,
            [str(p * i - c) for i in (1, 2, 3)],
            f"({p}n - {c})",
            None,
            lambda kk, p=p, c=c: (kk + 1) * (p * (kk) - 2 * c + p) / 2,
            True,
        )
    if kind == "mult":
        c = rng.randint(2, 6)
        return _Family(
            f"mult{c}",
            lambda x, c=c: c * x * (x + 1),
            lambda x, c=c: c * x * (x + 1) * (x + 2) / 3,
            [f"{c}×1×2", f"{c}×2×3", f"{c}×3×4"],
            f"{c}n(n + 1)",
            None,
            lambda kk, c=c: c * (kk + 1) * (kk + 2) * (kk + 2) / 3,
            True,
        )
    if kind == "pow2":
        r = rng.choice([2, 2, 3])
        closed = (lambda x: (x - 1) * 2 ** (x + 1) + 2) if r == 2 else (lambda x: ((2 * x - 1) * 3 ** (x + 1) + 3) / 4)
        return _Family(
            f"pow{r}",
            lambda x, r=r: x * Integer(r) ** x,
            closed,
            [f"1×{r}", f"2×{r}²", f"3×{r}³"],
            f"n×{r}^n",
            "(n - 1)2^(n + 1) + 2" if r == 2 else "((2n - 1)3^(n + 1) + 3)/4",
            (lambda kk: kk * 2 ** (kk + 1) + 2) if r == 2 else (lambda kk: ((2 * kk + 1) * 3 ** (kk + 1) + 3) / 4),
            False,
        )
    if kind == "frac":
        return _Family(
            "frac",
            lambda x: 1 / (x * (x + 1)),
            lambda x: x / (x + 1),
            ["1/(1×2)", "1/(2×3)", "1/(3×4)"],
            "1/(n(n + 1))",
            "n/(n + 1)",
            lambda kk: kk / (kk + 2),
            False,
        )
    if kind == "frac2":
        return _Family(
            "frac2",
            lambda x: 1 / ((2 * x - 1) * (2 * x + 1)),
            lambda x: x / (2 * x + 1),
            ["1/(1×3)", "1/(3×5)", "1/(5×7)"],
            "1/((2n - 1)(2n + 1))",
            "n/(2n + 1)",
            lambda kk: kk / (2 * kk + 3),
            False,
        )
    r = rng.choice([2, 3, 4, 5])
    return _Family(
        f"geo{r}",
        lambda x, r=r: Integer(r) ** (x - 1),
        lambda x, r=r: (Integer(r) ** x - 1) / (r - 1),
        ["1", str(r), str(r * r)],
        f"{r}^(n - 1)",
        f"{r}^n - 1" if r == 2 else f"({r}^n - 1)/{r - 1}",
        lambda kk, r=r: (Integer(r) ** kk - 1) / (r - 1),
        False,
    )


_VERIFIED: set[str] = set()


def _verified(fam: _Family) -> None:
    """The statement is true: brute-force n = 1..8, and the inductive identity symbolically and at points.
    Done once per distinct statement in a process (the family key names the statement)."""
    if fam.key in _VERIFIED:
        return
    sums = fam.brute(8)
    for m, value in enumerate(sums, start=1):
        check(sympy.nsimplify(fam.closed(Integer(m)) - value) == 0, f"{fam.key}: closed form wrong at n={m}")
    identity = fam.closed(k) + fam.term(k + 1) - fam.closed(k + 1)
    check(sympy.simplify(identity) == 0, f"{fam.key}: inductive identity fails")
    for point in (1, 2, 5, 9):
        check(identity.subs(k, point) == 0, f"{fam.key}: inductive identity fails at {point}")
    _VERIFIED.add(fam.key)


@lru_cache(maxsize=None)
def _expanded_over(expr) -> str:
    """A polynomial-over-integer expression as '(2k^3 + 9k^2 + 7k)/6' (or just the polynomial for a whole one)."""
    together = sympy.together(expand(expr))
    num, den = sympy.fraction(together)
    body = _txt(expand(num))
    return f"({body})/{den}" if den != 1 else body


def _factored_text(expr) -> str:
    return _txt(_factor(expr))


# ---------------------------------------------------------------------------
# Divisibility statements

_DIV_A_B = []  # (a, d, b) with d | a - 1 and b = -1 (mod d)
for _a in (3, 4, 5, 6, 7, 8, 9, 10, 11):
    for _d in range(2, _a):
        if (_a - 1) % _d == 0:
            for _b in (-1, _d - 1, 2 * _d - 1):
                if _b != 0 and (_a + _b) % _d == 0:
                    _DIV_A_B.append((_a, _d, _b))
_DIV_A_B = sorted(set(_DIV_A_B))


def _draw_div(rng: random.Random):
    a, d, b = rng.choice(_DIV_A_B)
    return a, d, b


def _f_text(a: int, b: int) -> str:
    return f"{a}^n {'+' if b > 0 else '-'} {abs(b)}"


# ---------------------------------------------------------------------------
# Distractor helpers


def _expr_dist(expr, tag: str) -> Distractor:
    text = _txt(expr)
    return Distractor(text, tag, text)


def _int_dist(value, tag: str) -> Distractor:
    return Distractor(str(int(value)), tag, int(value))


def _same_expr(a, b) -> bool:
    """Whether two expressions in k (or q) are identical: exact values at a few points first (the cheap way to
    say "different"), the full simplification only when every point agrees."""
    symbols = sorted((a - b).free_symbols, key=str)
    for point in (2, 3, 5, 7):
        if (a - b).subs({sym: point for sym in symbols}) != 0:
            return False
    return sympy.simplify(a - b) == 0


@lru_cache(maxsize=None)
def _distinct_exprs_cached(answer, cands):
    """Candidates that are genuinely different expressions from the answer and from each other."""
    out, seen = [], []
    for expr, tag in cands:
        try:
            check_distinct(answer, expr)
        except VerificationError:
            continue
        if any(_same_expr(expr, other) for other in seen):
            continue
        seen.append(expr)
        out.append(_expr_dist(expr, tag))
    return tuple(out)


def _distinct_exprs(answer, cands):
    return list(_distinct_exprs_cached(answer, tuple(cands)))


# ---------------------------------------------------------------------------
# Easy


def _easy_base_case(rng):
    for _ in attempts(30, "a base case that is not just 1"):
        fam = _draw_family(rng)
        if fam.term(Integer(1)) != 1 or rng.random() < 0.2:
            break
    _verified(fam)
    value = fam.term(Integer(1))
    check(sympy.nsimplify(fam.closed(Integer(1)) - value) == 0, "base case: the two sides differ at n = 1")
    answer = Rational(value)
    s2 = fam.brute(2)[1]
    num_at_1 = sympy.fraction(sympy.together(fam.closed(Integer(1))))[0]
    cands = [
        (Rational(s2), "base-case-evaluated-at-wrong-n"),
        (Rational(fam.closed(Integer(0))), "base-case-evaluated-at-wrong-n"),
        (Rational(fam.term(Integer(2))), "base-case-second-term-only"),
        (Rational(num_at_1), "base-case-drops-denominator"),
    ]
    out = []
    for val, tag in cands:
        if val == answer or any(val == v for v, _t in out):
            continue
        out.append((val, tag))
    if answer.q == 1:
        out = [(v, t) for v, t in out if v.q == 1]
    text_ans = _num_text(answer)
    distractors = [Distractor(_num_text(v), t, _value_for(v)) for v, t in out]
    check(len(distractors) >= 2, "too few base-case distractors")
    return dict(
        text=(
            f"A proof by induction of the statement P(n): {fam.statement()}, for all integers n ≥ 1, starts with the "
            "base case n = 1. What is the common value of the left-hand side and the right-hand side of P(1)?"
        ),
        answer=_value_for(answer),
        answer_text=text_ans,
        answer_kind="expression",
        working=[f"Left-hand side at n = 1 is {text_ans}; right-hand side at n = 1 is also {text_ans}."],
        distractors=distractors[:4],
    )


def _value_for(val):
    val = Rational(val)
    return int(val) if val.q == 1 else str(val)


def _easy_term(rng):
    fam = _draw_family(rng)
    _verified(fam)
    t_next = fam.term(k + 1)
    answer_text = _txt(t_next)
    cands = [
        (fam.term(k), "term-index-not-shifted"),
        (fam.term(k + 2), "term-index-over-shifted"),
        (fam.term(k - 1), "term-index-shifted-wrong-way"),
        (fam.term(k) + 1, "term-index-not-shifted"),
    ]
    distractors = _distinct_exprs(t_next, cands)
    check(len(distractors) >= 2, "too few term distractors")
    return dict(
        text=(
            f"A proof by induction of the statement P(n): {fam.statement()}, for all integers n ≥ 1, assumes P(k) is true "
            "for some integer k ≥ 1. In the inductive step the next term is added to both sides. "
            "Write the (k + 1)th term of the sum as an expression in k."
        ),
        answer=str(t_next),
        answer_text=answer_text,
        answer_kind="expression",
        working=[f"The nth term is {fam.general}; with n = k + 1 it is {answer_text}."],
        distractors=distractors[:4],
    )


def _easy_divisible_base(rng):
    a, d, b = _draw_div(rng)
    value = a + b
    check(value % d == 0, "f(1) is not divisible by d")
    cands = [
        (a**2 + b, "base-case-evaluated-at-wrong-n"),
        (1 + b, "base-case-evaluated-at-wrong-n"),
        (a * b, "base-case-evaluated-at-wrong-n"),
        (a - b, "base-case-evaluated-at-wrong-n"),
    ]
    distractors = []
    for val, tag in cands:
        if val != value and all(val != int(dd.text) for dd in distractors):
            distractors.append(_int_dist(val, tag))
    check(len(distractors) >= 2, "too few f(1) distractors")
    return dict(
        text=(
            f"A proof by induction of P(n): f(n) = {_f_text(a, b)} is divisible by {d}, for all integers n ≥ 1, starts "
            "with the base case n = 1. Find the value of f(1)."
        ),
        answer=value,
        answer_text=str(value),
        answer_kind="expression",
        working=[f"f(1) = {a} {'+' if b > 0 else '-'} {abs(b)} = {value}, which is {value // d} × {d}."],
        distractors=distractors[:4],
    )


# ---------------------------------------------------------------------------
# Medium


def _medium_target(rng):
    fam = _draw_family(rng)
    _verified(fam)
    target = fam.disp(fam.closed(k + 1))
    answer_text = _txt(target)
    cands = [
        (fam.disp(fam.closed(k)), "target-uses-k-not-k-plus-1"),
        (fam.disp(fam.partial(k)), "substitutes-k-plus-1-in-one-place-only"),
        (fam.disp(fam.closed(k) + fam.term(k)), "adds-kth-term-not-next"),
        (fam.disp(fam.closed(k) + 1), "adds-one-not-the-next-term"),
    ]
    distractors = _distinct_exprs(target, cands)
    check(len(distractors) >= 2, "too few target distractors")
    return dict(
        text=(
            f"The statement P(n): {fam.statement()} is to be proved by induction for all integers n ≥ 1. "
            "The inductive step assumes P(k) and must reach P(k + 1). "
            "Write the right-hand side of P(k + 1) as an expression in k."
        ),
        answer=str(target),
        answer_text=answer_text,
        answer_kind="expression",
        working=[f"Replace n by k + 1 in {fam.rhs}: {answer_text}."],
        distractors=distractors[:4],
    )


# ---------------------------------------------------------------------------
# Hard


def _hard_factored(rng):
    fam = _draw_family(rng, factorable_only=True)
    _verified(fam)
    s_next = fam.closed(k) + fam.term(k + 1)
    answer = _factor(s_next)
    check_equivalent(answer, fam.closed(k + 1), "S_k + term does not give P(k+1)'s right-hand side")
    answer_text = _txt(answer)
    cands = [
        (_factor(fam.closed(k)), "forgot-to-add-the-term"),
        (_factor(fam.closed(k) + fam.term(k)), "adds-kth-term-not-next"),
        (_factor(fam.partial(k)), "substitutes-k-plus-1-in-one-place-only"),
        (_factor(fam.closed(k) - fam.term(k + 1)), "subtracts-the-term-instead-of-adding"),
    ]
    distractors = _distinct_exprs(answer, cands)
    check(len(distractors) >= 2, "too few factored distractors")
    return dict(
        text=(
            f"The sum {fam.sum_text()} is being evaluated by induction. The inductive step assumes that the sum of "
            f"the first k terms is {_expanded_over(fam.closed(k))}. Add the (k + 1)th term and write the sum of the "
            "first (k + 1) terms as a single fully factorised expression in k."
        ),
        answer=str(answer),
        answer_text=answer_text,
        answer_kind="expression",
        answer_form="factored",
        working=[f"Sum of (k + 1) terms = {_expanded_over(fam.closed(k))} + {_txt(fam.term(k + 1))} = {answer_text}."],
        distractors=distractors[:4],
    )


def _hard_div_number(rng):
    a, d, b = _draw_div(rng)
    m = b * (1 - a)
    f = lambda e: Integer(a) ** e + b  # noqa: E731
    # second route: f(k+1) = a f(k) + m at several k
    for kk in (1, 2, 3, 4, 6):
        check(f(kk + 1) == a * f(kk) + m, "m does not make f(k+1) = a f(k) + m")
        check(m % d == 0, "m is not a multiple of d")
    cands = [
        (-m, "remainder-sign-reversed"),
        (-a * b, "remainder-drops-the-constant"),
        (b, "remainder-drops-the-constant"),
        (a * b, "remainder-drops-the-constant"),
    ]
    distractors = []
    for val, tag in cands:
        if val != m and all(val != int(x.text) for x in distractors):
            distractors.append(_int_dist(val, tag))
    check(len(distractors) >= 2, "too few m distractors")
    return dict(
        text=(
            f"P(n): f(n) = {_f_text(a, b)} is divisible by {d}, for all integers n ≥ 1. In the inductive step f(k) is "
            f"assumed divisible by {d}, and f(k + 1) is written in the form f(k + 1) = {a}f(k) + m, where m is a "
            "number. Find m."
        ),
        answer=m,
        answer_text=str(m),
        answer_kind="expression",
        working=[
            f"f(k + 1) = {a}·{a}^k {'+' if b > 0 else '-'} {abs(b)} = {a}({a}^k {'+' if b > 0 else '-'} {abs(b)}) "
            f"+ ({b} - {a * b}), so m = {m}, a multiple of {d}."
        ],
        distractors=distractors[:4],
    )


def _hard_div_power(rng):
    for _ in attempts(200, "a power difference"):
        a, b = rng.randint(4, 10), rng.randint(2, 8)
        if a - b >= 2 and a > b:
            break
    m = (a - b) * Integer(b) ** k
    f = lambda e: Integer(a) ** e - Integer(b) ** e  # noqa: E731
    check_equivalent(f(k + 1), a * f(k) + m, "m does not make f(k+1) = a f(k) + m")
    answer_text = _txt(m)
    cands = [
        ((a - b) * Integer(a) ** k, "rearrangement-uses-wrong-base"),
        (Integer(b) ** k, "rearrangement-drops-the-factor"),
        (-(a - b) * Integer(b) ** k, "remainder-sign-reversed"),
        ((a + b) * Integer(b) ** k, "uses-sum-not-difference-of-bases"),
    ]
    distractors = _distinct_exprs(m, cands)
    check(len(distractors) >= 2, "too few power-difference distractors")
    return dict(
        text=(
            f"P(n): f(n) = {a}^n - {b}^n is divisible by {a - b}, for all integers n ≥ 1. In the inductive step f(k) is "
            f"assumed divisible by {a - b}, and f(k + 1) is written in the form f(k + 1) = {a}f(k) + m. "
            "Find m as an expression in k."
        ),
        answer=str(m),
        answer_text=answer_text,
        answer_kind="expression",
        working=[f"{a}^(k + 1) - {b}^(k + 1) = {a}({a}^k - {b}^k) + ({a} - {b}){b}^k, so m = {answer_text}."],
        distractors=distractors[:4],
    )


_CUBIC = [
    ("n³ - n", 6, k**3 - k),
    ("n³ + 2n", 3, k**3 + 2 * k),
]


def _hard_div_cubic(rng):
    label, d, f_k = rng.choice(_CUBIC)
    diff = expand(f_k.subs(k, k + 1) - f_k)
    answer = _factor(diff)
    check(sympy.simplify(answer - diff) == 0, "factored difference disagrees")
    for kk in range(1, 8):
        check((f_k.subs(k, kk)) % d == 0, "statement not true")
    answer_text = _txt(answer)
    cands = [
        (diff + 1, "expansion-slip"),
        (f_k.subs(k, k + 1), "gives-f-of-k-plus-1-not-the-difference"),
        (_factor(f_k.subs(k, k + 1) + f_k), "adds-instead-of-subtracts"),
        (_factor(diff - 3 * k), "expansion-slip"),
    ]
    distractors = _distinct_exprs(answer, [(_factor(e) if i != 1 else e, t) for i, (e, t) in enumerate(cands)])
    check(len(distractors) >= 2, "too few cubic distractors")
    return dict(
        text=(
            f"P(n): f(n) = {label} is divisible by {d}, for all integers n ≥ 1. In the inductive step f(k + 1) is "
            "compared with f(k). Write f(k + 1) - f(k) as a fully factorised expression in k."
        ),
        answer=str(answer),
        answer_text=answer_text,
        answer_kind="expression",
        answer_form="factored",
        working=[f"f(k + 1) - f(k) = {_txt(expand(diff))} = {answer_text}."],
        distractors=distractors[:4],
    )


# ---------------------------------------------------------------------------
# Topic for the single-answer tiers


@generator(
    "ucv1_proof_by_induction",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_proof_by_induction(rng: random.Random, tier: str = "easy") -> dict:
    """A typed step of a proof by induction: the base-case value or the next term (easy), the target right-hand
    side (medium), the factored sum or the divisibility rearrangement (hard)."""
    pick = rng.random()
    if tier == "easy":
        return _easy_base_case(rng) if pick < 0.4 else _easy_term(rng) if pick < 0.85 else _easy_divisible_base(rng)
    if tier == "medium":
        return _medium_target(rng)
    if tier == "hard":
        if pick < 0.5:
            return _hard_factored(rng)
        if pick < 0.72:
            return _hard_div_number(rng)
        if pick < 0.88:
            return _hard_div_power(rng)
        return _hard_div_cubic(rng)
    raise VerificationError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain


def _exam_sum(rng: random.Random) -> dict:
    fam = _draw_family(rng, factorable_only=True)
    _verified(fam)
    base = fam.term(Integer(1))
    t_next = fam.term(k + 1)
    s_next = _factor(fam.closed(k) + t_next)
    check_equivalent(s_next, fam.closed(k + 1), "S_(k+1) disagrees with the closed form")
    hypothesis = _expanded_over(fam.closed(k))

    # (d): a use of the closed form
    if rng.random() < 0.6:
        use = "value"
        big = rng.randint(10, 25)
        d_answer = Integer(fam.closed(Integer(big)))
        check(d_answer == sum(fam.term(Integer(i)) for i in range(1, big + 1)), "S_N disagrees with brute force")
        d_prompt = f"Hence find the sum of the first {big} terms."
        d_ft = f"Subs(part_c, (k,), ({big - 1},))"
        d_cands = [
            (Integer(fam.closed(Integer(big - 1))), "off-by-one-term-count"),
            (Integer(fam.closed(Integer(big + 1))), "off-by-one-term-count"),
            (Integer(fam.closed(Integer(big)) - fam.term(Integer(big))), "off-by-one-term-count"),
            (
                Integer(Rational(big) * fam.term(Integer(big)) / 2),
                "treats-sum-as-arithmetic-series",
            ),
        ]
        marks_d = 2
    else:
        use = "least"
        for _ in attempts(200, "a threshold"):
            n0 = rng.randint(6, 14)
            lo, hi = fam.closed(Integer(n0 - 1)), fam.closed(Integer(n0))
            if hi - lo >= 4:
                break
        thr = int(lo) + rng.randint(1, int(hi - lo) - 2)
        brute = next(m for m in range(1, 60) if sum(fam.term(Integer(i)) for i in range(1, m + 1)) > thr)
        check(brute == n0, "least n disagrees with brute force")
        term_n = next((m for m in range(1, 2000) if fam.term(Integer(m)) > thr), None)
        d_answer = Integer(n0)
        d_prompt = f"Hence find the least number of terms for which the sum exceeds {thr}."
        d_ft = None
        d_cands = [
            (Integer(n0 - 1), "least-n-takes-last-failing-value"),
            (Integer(n0 + 1), "least-n-overshoots"),
        ] + ([(Integer(term_n), "term-compared-not-sum")] if term_n is not None else [])
        marks_d = 2
    d_text = str(int(d_answer))
    d_dist = []
    for val, tag in d_cands:
        if val != d_answer and all(val != Integer(x.text) for x in d_dist) and val >= 1:
            d_dist.append(_int_dist(val, tag))
    check(len(d_dist) >= 2, "too few use distractors")

    a_cands = [
        (fam.term(Integer(2)), "base-case-second-term-only"),
        (fam.brute(2)[1], "base-case-evaluated-at-wrong-n"),
        (fam.closed(Integer(0)), "base-case-evaluated-at-wrong-n"),
        (sympy.fraction(sympy.together(fam.closed(Integer(1))))[0], "base-case-drops-denominator"),
    ]
    a_dist = []
    for val, tag in a_cands:
        val = Rational(val)
        if val != base and all(_num_text(val) != x.text for x in a_dist) and (base.q != 1 or val.q == 1):
            a_dist.append(Distractor(_num_text(val), tag, _value_for(val)))
    b_cands = [
        (fam.term(k), "term-index-not-shifted"),
        (fam.term(k + 2), "term-index-over-shifted"),
        (fam.term(k - 1), "term-index-shifted-wrong-way"),
    ]
    b_dist = _distinct_exprs(t_next, b_cands)
    c_cands = [
        (_factor(fam.closed(k)), "forgot-to-add-the-term"),
        (_factor(fam.closed(k) + fam.term(k)), "adds-kth-term-not-next"),
        (_factor(fam.partial(k)), "substitutes-k-plus-1-in-one-place-only"),
        (_factor(fam.closed(k) - t_next), "subtracts-the-term-instead-of-adding"),
    ]
    c_dist = _distinct_exprs(s_next, c_cands)
    check(min(len(a_dist), len(b_dist), len(c_dist)) >= 2, "too few distractors in the induction chain")

    return dict(
        text=(f"Let P(n) be the statement {fam.statement()}, for integers n ≥ 1. P(n) is to be proved by induction."),
        parts=[
            dict(
                prompt="Verify P(1): write down the common value of the left-hand side and the right-hand side when n = 1.",
                answer_kind="expression",
                answer=_value_for(base),
                answer_text=_num_text(base),
                marks=1,
                working=[f"Both sides equal {_num_text(base)} when n = 1."],
                distractors=a_dist[:3],
            ),
            dict(
                prompt="In the inductive step P(k) is assumed true. Write the (k + 1)th term of the sum as an expression in k.",
                answer_kind="expression",
                answer=str(t_next),
                answer_text=_txt(t_next),
                marks=1,
                working=[f"The nth term is {fam.general}; with n = k + 1 it is {_txt(t_next)}."],
                distractors=b_dist[:3],
            ),
            dict(
                prompt=(
                    f"The assumption is that the sum of the first k terms is {hypothesis}. Add the (k + 1)th term "
                    "and write the sum of the first (k + 1) terms as a single fully factorised expression in k."
                ),
                answer_kind="expression",
                answer=str(s_next),
                answer_text=_txt(s_next),
                answer_form="factored",
                marks=3,
                follow_through=f"({fam.closed(k)}) + part_b",
                working=[f"{hypothesis} + {_txt(t_next)} = {_txt(s_next)}, which is the right-hand side of P(k + 1)."],
                distractors=c_dist[:3],
            ),
            dict(
                prompt=d_prompt,
                answer_kind="expression",
                answer=int(d_answer),
                answer_text=d_text,
                marks=marks_d,
                follow_through=d_ft,
                working=[f"Using the closed form, the answer is {d_text}."],
                distractors=d_dist[:3],
            ),
        ],
    )


def _exam_divisibility(rng: random.Random) -> dict:
    a, d, b = _draw_div(rng)
    f1 = a + b
    m = b * (1 - a)
    f = lambda e: Integer(a) ** e + b  # noqa: E731
    for kk in (1, 2, 3, 5):
        check(f(kk + 1) == a * f(kk) + m, "rearrangement wrong")
    # f(k) = d q  =>  f(k+1)/d = a q + m/d
    check(m % d == 0, "m not a multiple of d")
    quotient = a * q + Rational(m, d)
    big = rng.randint(3, 5)
    final = Integer(f(Integer(big)))
    check(final % d == 0, "f(N) not divisible by d")
    final_q = final // d
    a_cands = [
        (a**2 + b, "base-case-evaluated-at-wrong-n"),
        (1 + b, "base-case-evaluated-at-wrong-n"),
        (a * b, "base-case-evaluated-at-wrong-n"),
    ]
    b_cands = [
        (-m, "remainder-sign-reversed"),
        (-a * b, "remainder-drops-the-constant"),
        (b, "remainder-drops-the-constant"),
        (a * b, "remainder-drops-the-constant"),
    ]
    c_cands = [
        (a * q + m, "forgets-to-divide-the-remainder"),
        (a * q - Rational(m, d), "remainder-sign-reversed"),
        (q + Rational(m, d), "multiplier-dropped"),
        (a * q, "remainder-dropped"),
    ]
    d_cands = [
        (final, "quotient-not-taken"),
        (Integer(f(Integer(big + 1))) // d, "off-by-one-term-count"),
        (Integer(f(Integer(big - 1))) // d, "off-by-one-term-count"),
    ]

    def ints(cands, answer):
        out = []
        for val, tag in cands:
            if Rational(val) != Rational(answer) and Rational(val).q == 1 and all(str(int(val)) != x.text for x in out):
                out.append(_int_dist(val, tag))
        return out

    a_d, b_d, d_d = ints(a_cands, f1), ints(b_cands, m), ints(d_cands, final_q)
    c_d = _distinct_exprs(quotient, c_cands)
    check(min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2, "too few distractors in the divisibility chain")
    sign = "+" if b > 0 else "-"
    return dict(
        text=f"Let P(n) be the statement that f(n) = {_f_text(a, b)} is divisible by {d}, for all integers n ≥ 1. P(n) is to be proved by induction.",
        parts=[
            dict(
                prompt="Verify P(1): find the value of f(1).",
                answer_kind="expression",
                answer=f1,
                answer_text=str(f1),
                marks=1,
                working=[f"f(1) = {a} {sign} {abs(b)} = {f1} = {f1 // d} × {d}."],
                distractors=a_d[:3],
            ),
            dict(
                prompt=(
                    f"In the inductive step f(k) is assumed divisible by {d}. Write f(k + 1) in the form "
                    f"f(k + 1) = {a}f(k) + m, where m is a number, and state m."
                ),
                answer_kind="expression",
                answer=m,
                answer_text=str(m),
                marks=2,
                working=[f"f(k + 1) = {a}({a}^k {sign} {abs(b)}) + ({b} - {a * b}), so m = {m}."],
                distractors=b_d[:3],
            ),
            dict(
                prompt=f"Let f(k) = {d}q, where q is an integer. Write f(k + 1) ÷ {d} as an expression in q.",
                answer_kind="expression",
                answer=str(quotient),
                answer_text=_txt(quotient),
                marks=2,
                follow_through=f"{a}*q + part_b/{d}",
                working=[f"f(k + 1) = {a}·{d}q + ({m}), so f(k + 1) ÷ {d} = {_txt(quotient)}, an integer."],
                distractors=c_d[:3],
            ),
            dict(
                prompt=f"By the result, f({big}) is divisible by {d}. Find f({big}) ÷ {d}.",
                answer_kind="expression",
                answer=int(final_q),
                answer_text=str(int(final_q)),
                marks=1,
                working=[f"f({big}) = {int(final)}, and {int(final)} ÷ {d} = {int(final_q)}."],
                distractors=d_d[:3],
            ),
        ],
    )


@generator("exd_x_uc_induction", level=19, presets={"exam": dict(tier="exam")})
def exd_x_uc_induction(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a four-part induction chain, mostly for a sum, sometimes for divisibility."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _exam_divisibility(rng) if rng.random() < 0.2 else _exam_sum(rng)
