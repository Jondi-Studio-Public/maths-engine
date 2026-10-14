"""Shared helpers for the `r4a_*` abstract-algebra generators: permutations, small groups and answer plumbing.

This module is not a generator (its name starts with an underscore, so the generators package skips it).

Permutations of {1, ..., n} are tuples of images: `p[i - 1]` is the image of `i`. `compose(p, q)` is "p after q", so
a product of cycles is read right to left, the way the questions state it. A cycle is written starting at its smallest
entry, and the cycles of a permutation are ordered by their first entry; that is the one normal form the `set` grader
can compare, so every question that asks for cycle notation states it.

The answer helpers build wrong options whose structured value is graded by the real grader for the part: a wrong
option that the grader would accept is dropped, so every option a learner sees is marked wrong.
"""

from __future__ import annotations

from functools import wraps
from math import gcd, lcm
from types import SimpleNamespace

from ..answers import Distractor, exact_value
from ..core import check
from ..grading import GRADERS
from ..grading.typed import set_answer_text
from .exd_x_r2g_support import latex_with_sets

Perm = tuple[int, ...]


# -- permutations ---------------------------------------------------------------------------------------------------


def identity(n: int) -> Perm:
    return tuple(range(1, n + 1))


def compose(p: Perm, q: Perm) -> Perm:
    """p after q: i -> p(q(i))."""
    return tuple(p[q[i] - 1] for i in range(len(q)))


def inverse(p: Perm) -> Perm:
    out = [0] * len(p)
    for i, v in enumerate(p, start=1):
        out[v - 1] = i
    return tuple(out)


def power(p: Perm, k: int) -> Perm:
    """p^k by repeated composition (k >= 0)."""
    out = identity(len(p))
    for _ in range(k):
        out = compose(p, out)
    return out


def cycles(p: Perm, *, fixed: bool = False) -> list[tuple[int, ...]]:
    """Disjoint cycles of `p`, each starting at its smallest entry, ordered by first entry; fixed points only if asked."""
    seen: set[int] = set()
    out = []
    for start in range(1, len(p) + 1):
        if start in seen:
            continue
        cycle, point = [], start
        while point not in seen:
            seen.add(point)
            cycle.append(point)
            point = p[point - 1]
        if fixed or len(cycle) > 1:
            out.append(tuple(cycle))
    return out


def from_cycles(cycle_list, n: int) -> Perm:
    """The product of `cycle_list` (cycles as tuples, not nec. disjoint), the rightmost cycle acting first."""
    out = identity(n)
    for cycle in reversed(cycle_list):
        step = list(identity(n))
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            step[a - 1] = b
        out = compose(tuple(step), out)
    return out


def order(p: Perm) -> int:
    lengths = [len(c) for c in cycles(p, fixed=True)]
    return lcm(*lengths)


def order_by_powers(p: Perm) -> int:
    """The order by repeated composition (an independent route to `order`)."""
    ident, q = identity(len(p)), p
    for k in range(1, 5041):
        if q == ident:
            return k
        q = compose(p, q)
    raise AssertionError("order not found")  # pragma: no cover


def sign(p: Perm) -> int:
    """+1 or -1 by cycle count."""
    return -1 if (len(p) - len(cycles(p, fixed=True))) % 2 else 1


def sign_by_inversions(p: Perm) -> int:
    """The sign by counting inversions (an independent route to `sign`)."""
    inversions = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return -1 if inversions % 2 else 1


def cycle_text(cycle) -> str:
    return "(" + " ".join(str(x) for x in cycle) + ")"


def product_text(cycle_list) -> str:
    """Cycles written side by side, "(1 4 2)(2 5 3)"."""
    return "".join(cycle_text(c) for c in cycle_list)


def one_line_text(p: Perm) -> str:
    return ", ".join(str(v) for v in p)


def set_of_cycles(p: Perm):
    """(answer value, answer text) for the non-trivial cycles of `p` as a `set` answer."""
    value = [list(c) for c in cycles(p)]
    return value, set_answer_text(value)


def cycles_answer(cycle_list):
    """(value, text) for a list of cycles already in normal form."""
    value = [list(c) for c in cycle_list]
    return value, set_answer_text(value)


# -- wrong options ----------------------------------------------------------------------------------------------------


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
    """Distractors for a structured answer. `candidates` is a list of (value, misconception id); `text_of` writes a
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


def cycles_options(key_cycles, candidates, keep=3):
    """Wrong options for a cycle-notation (`set`) answer from candidate cycle lists, in likelihood order."""
    value, text = cycles_answer(key_cycles)
    cands = [([list(c) for c in cyc], tag) for cyc, tag in candidates if cyc is not None]
    return struct_options("set", value, text, cands, set_answer_text, keep)


def pair_options(key, candidates, keep=3):
    """Wrong options for a two-number `ordered_list` answer such as [i, j]."""
    text = "(" + ", ".join(map(str, key)) + ")"
    cands = [(list(v), tag) for v, tag in candidates if v is not None]
    return struct_options("ordered_list", list(key), text, cands, lambda v: "(" + ", ".join(map(str, v)) + ")", keep)


# -- parts and questions ----------------------------------------------------------------------------------------------


def number_part(
    prompt, value, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3, lo=None, hi=None
):
    """A part with a whole-number answer. Mirrors the other exam kits."""
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


def cycles_part(prompt, key_cycles, marks, working, candidates, keep=3):
    value, text = cycles_answer(key_cycles)
    options = cycles_options(key_cycles, candidates, keep)
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


def pair_part(prompt, key, marks, working, candidates, keep=3):
    options = pair_options(key, candidates, keep)
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    text = "(" + ", ".join(map(str, key)) + ")"
    return dict(
        prompt=prompt,
        answer_kind="ordered_list",
        answer=list(key),
        answer_text=text,
        marks=marks,
        working=working,
        distractors=options,
    )


def set_part(prompt, key, marks, working, candidates, keep=3):
    """A part whose answer is a set of numbers (or of tuples); `candidates` is a list of (value, misconception id)."""
    value = [list(x) if isinstance(x, (list, tuple)) else x for x in key]
    text = set_answer_text(value)
    cands = [
        ([list(x) if isinstance(x, (list, tuple)) else x for x in v], tag) for v, tag in candidates if v is not None
    ]
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


def gcd_all(*values: int) -> int:
    out = 0
    for v in values:
        out = gcd(out, v)
    return out


def with_latex(build):
    """Decorate a generator so every prompt carries real LaTeX: set braces (`{1, 2}`) become maths, prose stays text."""

    @wraps(build)
    def wrapped(*args, **kwargs):
        d = build(*args, **kwargs)
        if d.get("parts"):
            d["latex"] = latex_with_sets(d["text"])
            for part in d["parts"]:
                part["prompt_latex"] = latex_with_sets(part["prompt"])
        else:
            d["latex"] = latex_with_sets(d["text"])
        return d

    return wrapped


def plain_latex(build):
    """Like `with_latex`, but the prompts are shown as plain text (for words made of single-letter symbols, `r^5 s`)."""

    @wraps(build)
    def wrapped(*args, **kwargs):
        d = build(*args, **kwargs)
        d["latex"] = "plain:" + d["text"]
        for part in d.get("parts") or []:
            part["prompt_latex"] = "plain:" + part["prompt"]
        return d

    return wrapped
