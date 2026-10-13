"""Source of `generate-sequence-from-rule` for the easy, medium and hard stages: find a term of a sequence from
the rule that defines it. The tiers escalate in the reasoning the learner must chain, not in the size of the numbers.

  easy   -- explicit linear rule t(n) = a n + b, n = 1 the first term; one substitution.
  medium -- explicit rule with a square or a negative coefficient (a n^2 + b n + c, or c - a n); two or three
            operations that must be done in the right order, sometimes asked as "the kth term".
  hard   -- recursive rule t(1) = p, t(n + 1) = r t(n) + d, asked for t(4) or t(5): the learner iterates k - 1 times.

Notation follows `exd_x_f0_sequence_recurrence` (the exam source): t(n), t(n + 1), joined with `join_terms`.
Every answer is computed by one route (substitution or a forward loop) and checked by a second (an independent
list of terms, or a closed form in exact rational arithmetic).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor
from ..core import check, generator, join_terms, pick_distractors


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _signed(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([-1, 1]) * rng.randint(lo, hi)


def _pack(text: str, value: int, working: list[str], candidates: list[tuple[int, str | None]]) -> dict:
    """Build the question dict; `candidates` are (wrong value, misconception id) in priority order."""
    answer_text = str(value)
    cands = [Distractor(str(v), tag, v) for v, tag in candidates]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=value,
        answer_text=answer_text,
        answer_latex=answer_text,
        answer_kind="number",
        working=working,
        distractors=pick_distractors(answer_text, cands),
    )


def _fill(wrong: list[tuple[int, str | None]], value: int) -> list[tuple[int, str | None]]:
    """Append untagged near-misses so a degenerate draw still offers options."""
    return wrong + [(value + 1, None), (value - 1, None), (value + 2, None)]


def _easy(rng: random.Random) -> dict:
    for _ in range(200):
        a = rng.randint(2, 9)
        b = _signed(rng, 1, 9)
        k = rng.randint(3, 8)
        value = a * k + b
        before = a * (k - 1) + b
        after = a * (k + 1) + b
        on_previous = a * before + b
        unsigned = a * k + abs(b)
        candidates = {value, before, after, on_previous, a * k} | ({unsigned} if b < 0 else set())
        if len(candidates) == (6 if b < 0 else 5):
            break
    else:
        check(False, "no usable easy draw")
    terms = [a * i + b for i in range(1, k + 1)]
    check(terms[-1] == value, "listing the terms disagrees with direct substitution")
    rule = join_terms([(a, "n"), (b, "")])
    text = f"A sequence has the rule t(n) = {rule}, where n = 1 gives the first term. Find t({k})."
    working = [
        f"t({k}) = {a} × {k} {'+' if b > 0 else '-'} {abs(b)} = {a * k} {'+' if b > 0 else '-'} {abs(b)} = {value}"
    ]
    wrong = [
        (before, "term-position-off-by-one"),
        (after, "term-position-off-by-one"),
        (on_previous, "explicit-rule-applied-to-previous-term"),
        (a * k, "dropped-constant-term"),
    ]
    if b < 0:
        wrong.append((unsigned, "substitution-sign-error"))
    return _pack(text, value, working, _fill(wrong, value))


def _medium_quadratic(rng: random.Random) -> dict:
    for _ in range(300):
        a = rng.randint(1, 4)
        b = rng.choice([0, 0, -1, 1, 2, 3, -2, -3, 4, -4])
        c = _signed(rng, 0, 8)
        k = rng.randint(3, 7)
        value = a * k * k + b * k + c
        before = a * (k - 1) ** 2 + b * (k - 1) + c
        after = a * (k + 1) ** 2 + b * (k + 1) + c
        doubled = a * 2 * k + b * k + c
        whole_square = (a * k) ** 2 + b * k + c
        no_const = a * k * k + b * k
        flipped = a * k * k - b * k + c
        if len({value, before, after, doubled, no_const}) == 5:
            break
    else:
        check(False, "no usable medium quadratic draw")
    terms = [a * i**2 + b * i + c for i in range(1, k + 1)]
    check(terms[-1] == value, "listing the terms disagrees with direct substitution")
    rule = join_terms([(a, "n^2"), (b, "n"), (c, "")])
    ask, label = _ask(rng, k)
    text = f"A sequence has the rule t(n) = {rule}, where n = 1 gives the first term. {ask}"
    working = [_quad_working(label, a, b, c, k, value)]
    wrong = [
        (doubled, "squared-as-doubled"),
        (no_const, "dropped-constant-term"),
        (before, "term-position-off-by-one"),
        (after, "term-position-off-by-one"),
    ]
    if a > 1:
        wrong.insert(1, (whole_square, "square-applied-to-coefficient"))
    if b < 0:
        wrong.insert(2, (flipped, "sign-error-linear-term"))
    return _pack(text, value, working, _fill(wrong, value))


def _quad_working(label: str, a: int, b: int, c: int, k: int, value: int) -> str:
    parts = [f"{k}^2" if a == 1 else f"{a} × {k}^2"]
    if b:
        parts.append(f"{'+' if b > 0 else '-'} {k}" if abs(b) == 1 else f"{'+' if b > 0 else '-'} {abs(b)} × {k}")
    if c:
        parts.append(f"{'+' if c > 0 else '-'} {abs(c)}")
    steps = f"{a * k * k}"
    if b:
        steps += f" {'+' if b > 0 else '-'} {abs(b * k)}"
    if c:
        steps += f" {'+' if c > 0 else '-'} {abs(c)}"
    return f"{label} = {' '.join(parts)} = {steps} = {value}"


def _ask(rng: random.Random, k: int) -> tuple[str, str]:
    """The request, and the label the working uses for the term."""
    if rng.random() < 0.5:
        return f"Find the {_ordinal(k)} term.", f"t({k})"
    return f"Find t({k}).", f"t({k})"


def _medium_negative(rng: random.Random) -> dict:
    for _ in range(300):
        a = rng.randint(2, 8)
        c = rng.randint(20, 60)
        k = rng.randint(3, 8)
        value = c - a * k
        reordered = (c - a) * k
        flipped = c + a * k
        before = c - a * (k - 1)
        after = c - a * (k + 1)
        on_previous = c - a * before
        if len({value, reordered, flipped, before, after, on_previous}) == 6:
            break
    else:
        check(False, "no usable medium negative-coefficient draw")
    terms = [c - a * i for i in range(1, k + 1)]
    check(terms[-1] == value, "listing the terms disagrees with direct substitution")
    rule = join_terms([(c, ""), (-a, "n")])
    ask, label = _ask(rng, k)
    text = f"A sequence has the rule t(n) = {rule}, where n = 1 gives the first term. {ask}"
    working = [f"{label} = {c} - {a} × {k} = {c} - {a * k} = {value}"]
    wrong = [
        (reordered, "wrong-order-of-operations"),
        (flipped, "sign-error-linear-term"),
        (before, "term-position-off-by-one"),
        (after, "term-position-off-by-one"),
        (on_previous, "explicit-rule-applied-to-previous-term"),
    ]
    return _pack(text, value, working, _fill(wrong, value))


def _medium(rng: random.Random) -> dict:
    return rng.choice([_medium_quadratic, _medium_negative])(rng)


def _hard(rng: random.Random) -> dict:
    for _ in range(500):
        p = rng.randint(-4, 8)
        r = rng.choice([2, 2, 3, -1, -2, 2, 3, -3, 1])
        d = _signed(rng, 1, 6)
        k = rng.choice([4, 5])
        terms = [p]
        for _i in range(k - 1):
            terms.append(r * terms[-1] + d)
        value = terms[-1]
        too_early = terms[-2]
        too_late = r * value + d
        no_constant = p * r ** (k - 1)
        add_first = p
        for _i in range(k - 1):
            add_first = r * (add_first + d)
        if len({value, too_early, too_late, no_constant, add_first}) == 5 and abs(value) <= 150:
            break
    else:
        check(False, "no usable hard draw")
    # Second route, exact: t(k) = r^(k-1) (p + d/(r-1)) - d/(r-1) when r != 1, else p + (k-1) d.
    if r == 1:
        closed = Fraction(p + (k - 1) * d)
    else:
        fixed = Fraction(d, 1 - r)
        closed = (p - fixed) * r ** (k - 1) + fixed
    check(closed == value, "closed form disagrees with the iterated terms")
    rule = join_terms([(r, "t(n)"), (d, "")])
    text = f"A sequence is defined by t(1) = {p} and t(n + 1) = {rule} for every n ≥ 1. Find t({k})."
    working = [f"t(1) = {p}"]
    for i in range(1, k):
        prev, nxt = terms[i - 1], terms[i]
        working.append(f"t({i + 1}) = {_apply(r, prev, d)} = {nxt}")
    wrong = [
        (too_early, "iteration-stopped-early"),
        (too_late, "term-position-off-by-one"),
        (no_constant, "dropped-constant-term"),
        (add_first, "wrong-order-of-operations"),
    ]
    return _pack(text, value, working, _fill(wrong, value))


def _apply(r: int, prev: int, d: int) -> str:
    """'2 × 3 + 1' for one step of the recursion, with a negative previous term bracketed."""
    shown = f"({prev})" if prev < 0 else str(prev)
    times = {1: shown, -1: f"-{shown}"}.get(r, f"{r} × {shown}")
    return f"{times} {'+' if d > 0 else '-'} {abs(d)}"


@generator(
    "sequence_term_from_rule",
    level=10,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def sequence_term_from_rule(rng: random.Random, tier: str = "easy") -> dict:
    builders = {"easy": _easy, "medium": _medium, "hard": _hard}
    check(tier in builders, f"unknown tier {tier!r}")
    return builders[tier](rng)
