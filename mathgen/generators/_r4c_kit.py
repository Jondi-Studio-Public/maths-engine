"""Shared helpers for the `r4c_*` abstract-algebra generators: answer plumbing and arithmetic over Z_p[x].

This module is not a generator (its name starts with an underscore, so the generators package skips it).

Every wrong option is built from a named error and is run through the real grader for its part: an option the grader
would accept as correct is dropped, so each option a learner sees is marked wrong. A polynomial over Z_p is a list of
coefficients, lowest degree first, each in 0..p-1; the helpers keep that normal form.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd
from types import SimpleNamespace

from ..answers import Distractor, exact_value
from ..core import check
from ..grading import GRADERS
from ..grading.typed import set_answer_text

# -- answer plumbing --------------------------------------------------------------------------------------------------


def _accepted(kind, answer, answer_text, reply) -> bool:
    probe = SimpleNamespace(
        answer=answer, answer_text=answer_text, text="", answer_alt=None, answer_form=None, answer_kind=kind
    )
    return bool(GRADERS[kind](probe, reply).correct)


def int_options(value, slips, keep=3, *, lo=None, hi=None):
    """Distractors for a whole-number answer: `slips` is a list of (wrong value, misconception id) in order of
    likelihood; repeats, the answer itself, out-of-range values and anything the number grader accepts are dropped."""
    text = str(value)
    answer = exact_value(value)
    out, seen = [], {text}
    for item in slips:
        if item is None or item[0] is None:
            continue
        v, tag = item
        if int(v) != v:
            continue
        v = int(v)
        if str(v) in seen or (lo is not None and v < lo) or (hi is not None and v > hi):
            continue
        if _accepted("number", answer, text, str(v)):
            continue
        seen.add(str(v))
        out.append(Distractor(str(v), tag, v))
        if len(out) == keep:
            break
    return out


def struct_options(kind, answer, answer_text, candidates, text_of, keep=3):
    """Distractors for a structured answer: `candidates` is a list of (value, misconception id); `text_of` writes a
    value as answer text. Each kept option differs from the key and is rejected by the real grader."""
    check(_accepted(kind, answer, answer_text, answer_text), "the grader rejects its own canonical answer")
    out, seen = [], {answer_text}
    for value, tag in candidates:
        if value is None or (kind == "set" and len({repr(x) for x in value}) != len(value)):
            continue
        text = text_of(value)
        if text in seen or _accepted(kind, answer, answer_text, text):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, value))
        if len(out) == keep:
            break
    return out


def tuple_text(values) -> str:
    return "(" + ", ".join(str(v) for v in values) + ")"


def number_part(
    prompt, value, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3, lo=None, hi=None
):
    """A part with a whole-number answer."""
    options = int_options(value, slips, keep, lo=lo, hi=hi)
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=exact_value(value),
        answer_text=str(value),
        marks=marks,
        working=working,
        distractors=options,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def set_part(prompt, key, marks, working, candidates, keep=3):
    """A part whose answer is a set of whole numbers; `candidates` is a list of (set, misconception id)."""
    value = sorted(key)
    text = set_answer_text(value)
    cands = [(sorted(v), tag) for v, tag in candidates if v is not None]
    options = struct_options("set", value, text, cands, set_answer_text, keep)
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    return dict(
        prompt=prompt,
        answer_kind="set",
        answer=value,
        answer_text=text,
        marks=marks,
        working=working,
        distractors=options,
    )


def ordered_part(prompt, key, marks, working, candidates, keep=3):
    """A part whose answer is a short ordered list of whole numbers, typed "(a, b, c)"."""
    value = list(key)
    text = tuple_text(value)
    cands = [(list(v), tag) for v, tag in candidates if v is not None]
    options = struct_options("ordered_list", value, text, cands, tuple_text, keep)
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    return dict(
        prompt=prompt,
        answer_kind="ordered_list",
        answer=value,
        answer_text=text,
        marks=marks,
        working=working,
        distractors=options,
    )


def expr_part(prompt, key, p, marks, working, candidates, keep=3):
    """A part whose answer is a polynomial (coefficient lists, lowest degree first)."""
    text, sym = poly_text(key), poly_sympy(key)
    options = poly_options(key, candidates, keep)
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    return dict(
        prompt=prompt,
        answer_kind="expression",
        answer=sym,
        answer_text=text,
        answer_latex=poly_latex(key),
        marks=marks,
        working=working,
        distractors=options,
    )


def number_question(text, value, working, slips, *, keep=4, lo=None, hi=None):
    """A single-answer whole-number question dict."""
    options = int_options(value, slips, keep, lo=lo, hi=hi)
    check(len(options) >= 2, "too few distractors")
    return dict(
        text=text,
        answer_kind="number",
        answer=exact_value(value),
        answer_text=str(value),
        answer_latex=str(value),
        working=working,
        distractors=options,
    )


def set_question(text, key, working, candidates, keep=3):
    value = sorted(key)
    ans_text = set_answer_text(value)
    cands = [(sorted(v), tag) for v, tag in candidates if v is not None]
    options = struct_options("set", value, ans_text, cands, set_answer_text, keep)
    check(len(options) >= 2, "too few distractors")
    return dict(text=text, answer_kind="set", answer=value, answer_text=ans_text, working=working, distractors=options)


def ordered_question(text, key, working, candidates, keep=3):
    value = list(key)
    ans_text = tuple_text(value)
    cands = [(list(v), tag) for v, tag in candidates if v is not None]
    options = struct_options("ordered_list", value, ans_text, cands, tuple_text, keep)
    check(len(options) >= 2, "too few distractors")
    return dict(
        text=text, answer_kind="ordered_list", answer=value, answer_text=ans_text, working=working, distractors=options
    )


def expr_question(text, key, working, candidates, keep=3):
    options = poly_options(key, candidates, keep)
    check(len(options) >= 2, "too few distractors")
    return dict(
        text=text,
        answer_kind="expression",
        answer=poly_sympy(key),
        answer_text=poly_text(key),
        answer_latex=poly_latex(key),
        working=working,
        distractors=options,
    )


# -- number theory ----------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def factorise(n: int) -> tuple[tuple[int, int], ...]:
    """The prime factorisation of n as ((prime, exponent), ...)."""
    out, d = [], 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            out.append((d, e))
        d += 1
    if n > 1:
        out.append((n, 1))
    return tuple(out)


def divisors(n: int) -> list[int]:
    out = [1]
    for prime, e in factorise(n):
        out = [d * prime**k for d in out for k in range(e + 1)]
    return sorted(out)


def totient(n: int) -> int:
    out = n
    for prime, _ in factorise(n):
        out = out // prime * (prime - 1)
    return out


def radical(n: int) -> int:
    out = 1
    for prime, _ in factorise(n):
        out *= prime
    return out


PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)


# -- polynomials over Z_p ---------------------------------------------------------------------------------------------

Poly = tuple[int, ...]  # coefficients, lowest degree first, no trailing zeros; () is the zero polynomial


def trim(c) -> Poly:
    c = list(c)
    while c and c[-1] == 0:
        c.pop()
    return tuple(c)


def padd(a, b, p) -> Poly:
    n = max(len(a), len(b))
    return trim(((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n))


def pmul(a, b, p) -> Poly:
    if not a or not b:
        return ()
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return trim(out)


def pdivmod(a, b, p):
    """Quotient and remainder of a by b over Z_p (b non-zero)."""
    a, b = list(a), trim(b)
    check(bool(b), "division by the zero polynomial")
    inv = pow(b[-1], -1, p)
    q = [0] * max(len(a) - len(b) + 1, 0)
    while len(trim(a)) >= len(b):
        a = list(trim(a))
        shift = len(a) - len(b)
        f = a[-1] * inv % p
        q[shift] = f
        for i, y in enumerate(b):
            a[shift + i] = (a[shift + i] - f * y) % p
    return trim(q), trim(a)


def monic(a, p) -> Poly:
    a = trim(a)
    if not a:
        return a
    inv = pow(a[-1], -1, p)
    return trim(c * inv % p for c in a)


def pgcd(a, b, p) -> Poly:
    """The monic gcd over Z_p."""
    a, b = trim(a), trim(b)
    while b:
        a, b = b, pdivmod(a, b, p)[1]
    return monic(a, p)


def peval(a, x, p) -> int:
    out = 0
    for c in reversed(a):
        out = (out * x + c) % p
    return out


def pdeg(a) -> int:
    return len(trim(a)) - 1


@lru_cache(maxsize=None)
def all_monic(deg: int, p: int):
    """Every monic polynomial of the degree over Z_p, as tuples."""
    from itertools import product

    return tuple(tuple(low) + (1,) for low in product(range(p), repeat=deg))


@lru_cache(maxsize=None)
def irreducibles(deg: int, p: int):
    """Every monic irreducible polynomial of the degree over Z_p (deg <= 3 by roots; larger by trial division)."""
    if deg == 1:
        return all_monic(1, p)
    found = []
    for f in all_monic(deg, p):
        if all(pdivmod(f, g, p)[1] for d in range(1, deg // 2 + 1) for g in (irreducibles(d, p) if d < deg else ())):
            found.append(f)
    return tuple(found)


def factor_degrees(f, p) -> list[int]:
    """Degrees of the irreducible factors of f over Z_p with multiplicity, smallest first (trial division)."""
    f, out = trim(f), []
    d = 1
    while pdeg(f) >= 2 * d:
        for g in irreducibles(d, p):
            while pdeg(f) >= d and not pdivmod(f, g, p)[1]:
                f = pdivmod(f, g, p)[0]
                out.append(d)
        d += 1
    if pdeg(f) >= 1:
        out.append(pdeg(f))
    return sorted(out)


# polynomial writing: "x^3 + 2x + 4"; the same polynomial as a SymPy string and as LaTeX


def _terms(a):
    for k in range(len(a) - 1, -1, -1):
        if a[k]:
            yield k, a[k]


def _signed(a, term):
    """Join the terms of a polynomial (coefficients may be negative) with " + " and " - "; `term(k, c)` writes |c|x^k."""
    out = ""
    for k, c in _terms(trim(a)):
        body = term(k, abs(c))
        if not out:
            out = ("-" if c < 0 else "") + body
        else:
            out += (" - " if c < 0 else " + ") + body
    return out or "0"


def poly_text(a) -> str:
    def term(k, c):
        coef = "" if c == 1 and k > 0 else str(c)
        return coef + ("x" if k == 1 else f"x^{k}" if k > 1 else "")

    return _signed(a, term)


def poly_sympy(a) -> str:
    return _signed(a, lambda k, c: f"{c}" if k == 0 else f"{c}*x" if k == 1 else f"{c}*x**{k}")


def poly_latex(a) -> str:
    def term(k, c):
        coef = "" if c == 1 and k > 0 else str(c)
        return coef + ("x" if k == 1 else f"x^{{{k}}}" if k > 1 else "")

    return _signed(a, term)


def poly_options(key, candidates, keep=3):
    """Distractors for a polynomial answer; candidates are (coefficient list, misconception id).

    Two polynomials with different coefficient lists are different polynomials, and the expression grader compares
    polynomials, so a candidate that differs from the key is rejected by it. The tests grade every option of every part,
    so the generator does not parse them with SymPy a second time.
    """
    text, want = poly_text(key), trim(key)
    out, seen = [], {text}
    for value, tag in candidates:
        if value is None or not trim(value) or trim(value) == want:
            continue
        t = poly_text(value)
        if t in seen:
            continue
        seen.add(t)
        out.append(Distractor(t, tag, poly_sympy(value)))
        if len(out) == keep:
            break
    return out


def gcd_all(*values: int) -> int:
    out = 0
    for v in values:
        out = gcd(out, v)
    return out
