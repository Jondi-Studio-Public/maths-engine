"""Generates `transition-matrix-steady-state` questions (easy, medium, hard) and the exam chain `r3c_steady_state_chain`.

Over many periods the shares of a population among three states settle to a distribution ``s`` that one more period
does not change: ``T s = s`` with the shares adding to 1. The movements are given in words (the same contexts as
`transition-matrix-states`); the learner finds the long-run shares by solving for ``s``, not by multiplying forward.

* easy: the long-run share of one named state, as a fraction;
* medium: the long-run fraction in two named states together, where one state's movements are given with "the
  rest" so a share is found from a column total;
* hard: one movement share is unknown and the long-run share of one state is stated, so the unknown is found and then
  the long-run share of a second state;
* exam (``r3c_steady_state_chain``): the long-run shares of two states, the third carried forward from them, and the
  long-run number in the third state out of a total.

Every transition matrix is built first (columns in tenths); the long-run shares are solved exactly and the denominator
is kept small so the shares are nice fractions.
"""

from __future__ import annotations

from fractions import Fraction

from math import gcd

from ..core import attempts, check, generator
from ._r3c_kit import (
    col,
    fraction_part,
    fraction_single,
    matrix_single,
    names_text,
    number_part,
    number_single,
    percent,
    random_transition,
    steady_state,
    transpose,
)
from .r3c_transition_states_build import _describe, _setup

_LONG_RUN = "In the long run the shares of the {noun} in each group settle to a fixed distribution."


def _lcm(values):
    out = 1
    for v in values:
        out = out * v.denominator // gcd(out, v.denominator)
    return int(out)


def _good(T, limit):
    s = steady_state(T)
    if s is None or any(v <= 0 for v in s) or _lcm(s) > limit:
        return None
    return s


def _slips_share(T, s, target):
    """Wrong long-run shares for ``target`` from the usual errors."""
    n = len(s)
    stay = [T[i][i] for i in range(n)]
    out = [(stay[target] / sum(stay), "r3c-ss-stay-share"), (Fraction(1, n), "r3c-ss-equal-shares")]
    flipped = steady_state(transpose(T))
    if flipped is not None:
        out.append((flipped[target], "r3c-ss-rows-columns-swapped"))
    other = (target + 1) % n
    out.append((s[other], "r3c-ss-wrong-state"))
    if s[target] > Fraction(1, 3):
        out.append((s[target] / sum(s[:-1]) if sum(s[:-1]) else None, "r3c-ss-not-normalised"))
    out.append((s[target] / max(s), "r3c-ss-not-normalised"))
    out.append((1 - s[target], "r3c-ss-complement"))
    return [(v, tag) for v, tag in out if v is not None and 0 < v < 1]


def _easy(rng):
    for _ in attempts(800, "an easy steady-state question"):
        ctx, names = _setup(rng, 3)
        T = random_transition(rng, 3, stay=(4, 7))
        s = _good(T, 30)
        if s is None or len(set(s)) < 3:
            continue
        target = rng.randrange(3)
        text = (
            ctx["intro"].format(names=names_text(names))
            + " "
            + _describe(ctx, T, names)
            + " "
            + _LONG_RUN.format(noun=ctx["noun"])
            + f" What fraction of the {ctx['noun']} are {ctx['at'](names[target])} in the long run?"
        )
        working = [
            "In the long run the shares s satisfy T s = s and add to 1.",
            *[f"{nm}: {s[i]}" for i, nm in enumerate(names)],
        ]
        q = fraction_single(text, s[target], working, _slips_share(T, s, target))
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _medium(rng):
    for _ in attempts(800, "a medium steady-state question"):
        ctx, names = _setup(rng, 3)
        T = random_transition(rng, 3, stay=(4, 7))
        s = _good(T, 40)
        if s is None or len(set(s)) < 3:
            continue
        a, b = rng.sample(range(3), 2)
        c = 3 - a - b
        text = (
            ctx["intro"].format(names=names_text(names))
            + " "
            + _describe(ctx, T, names, rest_for=(rng.randrange(3),))
            + " "
            + _LONG_RUN.format(noun=ctx["noun"])
            + f" What fraction of the {ctx['noun']} are {ctx['at'](names[a])} or {ctx['at'](names[b])} in the long run?"
        )
        key = s[a] + s[b]
        wrong = [
            (s[a], "r3c-ss-one-group-only"),
            (s[b], "r3c-ss-one-group-only"),
            (s[c], "r3c-ss-wrong-state"),
            (Fraction(2, 3), "r3c-ss-equal-shares"),
            (_slips_share(T, s, a)[0][0] + _slips_share(T, s, b)[0][0], "r3c-ss-stay-share"),
            (s[a] * s[b], "r3c-ss-product"),
        ]
        working = [
            "Solve T s = s with the shares adding to 1.",
            *[f"{nm}: {s[i]}" for i, nm in enumerate(names)],
            f"{s[a]} + {s[b]} = {key}.",
        ]
        q = fraction_single(text, key, working, [(v, t) for v, t in wrong if 0 < v < 1])
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _steady_with_unknown(T, k_from, k_to, x):
    """The long-run shares when the move from state ``k_from`` to ``k_to`` has share ``x`` (and the stay share of
    ``k_from`` takes up the slack)."""
    n = len(T)
    M = [list(r) for r in T]
    M[k_to][k_from] = Fraction(x)
    M[k_from][k_from] = 1 - sum(M[i][k_from] for i in range(n) if i != k_from)
    return M


def _hard(rng):
    for _ in attempts(1500, "a hard steady-state question"):
        ctx, names = _setup(rng, 3)
        T = random_transition(rng, 3, stay=(4, 7))
        k_from = rng.randrange(3)
        k_to = rng.choice([i for i in range(3) if i != k_from])
        x = Fraction(rng.randint(1, 4), 10)
        M = _steady_with_unknown(T, k_from, k_to, x)
        if M[k_from][k_from] < Fraction(2, 10):
            continue
        s = _good(M, 40)
        if s is None or len(set(s)) < 3:
            continue
        target = rng.randrange(3)
        ask = rng.choice([i for i in range(3) if i != target])
        # The stated share must identify x. With the unknown in one column, every unnormalised long-run entry is
        # linear in x, so the share is a ratio of two linear functions of x: it is either constant or strictly
        # monotone on the allowed range. Two different shares at two allowed values of x rule out a constant.
        other_x = x + Fraction(1, 10) if M[k_from][k_from] >= Fraction(1, 10) else x - Fraction(1, 10)
        s_other = _good(_steady_with_unknown(T, k_from, k_to, other_x), 10**6)
        if s_other is None or s_other[target] == s[target]:
            continue
        lines = []
        for j in range(3):
            if j == k_from:
                others = [(i, M[i][j]) for i in range(3) if i != j and i != k_to and M[i][j] > 0]
                bits = [f"{percent(p)} {ctx['to'](names[i])}" for i, p in others]
                bits.append(f"x% {ctx['to'](names[k_to])}")
                lines.append(
                    f"Of the {ctx['noun']} {ctx['at'](names[j])}, " + names_text(bits) + "; everyone else stays."
                )
            else:
                moves = [(i, M[i][j]) for i in range(3) if i != j and M[i][j] > 0]
                bits = [f"{percent(M[j][j])} {ctx['stay']}"] + [f"{percent(p)} {ctx['to'](names[i])}" for i, p in moves]
                lines.append(f"Of the {ctx['noun']} {ctx['at'](names[j])}, " + names_text(bits) + ".")
        text = (
            ctx["intro"].format(names=names_text(names))
            + " "
            + " ".join(lines)
            + f" In the long run {s[target]} of the {ctx['noun']} are {ctx['at'](names[target])}. "
            + f"What fraction of the {ctx['noun']} are {ctx['at'](names[ask])} in the long run?"
        )
        key = s[ask]
        wrong = []
        for slip_x in (x + Fraction(1, 10), x - Fraction(1, 10)):
            if slip_x > 0 and M[k_from][k_from] + x - slip_x >= 0:
                s_slip = _good(_steady_with_unknown(T, k_from, k_to, slip_x), 10**6)
                if s_slip is not None:
                    wrong.append((s_slip[ask], "r3c-ss-share-slip"))
        third = 3 - target - ask
        wrong += [
            (s[third], "r3c-ss-wrong-state"),
            (1 - key, "r3c-ss-complement"),
            (1 - s[target], "r3c-ss-complement"),
            (s[target], "r3c-ss-wrong-state"),
        ]
        working = [
            "Write the stay share of that group in terms of x, so every column adds to 1.",
            "Solve T s = s with the shares adding to 1, keeping x, and set the share of the stated group equal to the given fraction to find x.",
            f"x = {int(x * 100)}, and the long-run shares are {', '.join(f'{nm} {s[i]}' for i, nm in enumerate(names))}.",
        ]
        q = fraction_single(text, key, working, [(v, t) for v, t in wrong if 0 < v < 1])
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "r3c_transition_steady_state",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3c_transition_steady_state(rng, tier="easy"):
    """The long-run behaviour of a transition model; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


@generator("r3c_steady_state_chain", level=12, presets={"exam": dict(tier="exam")})
def r3c_steady_state_chain(rng, tier="exam"):
    """Exam chain: two long-run shares, the third from them, and the long-run count in the third state."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(800, "a steady-state chain"):
        ctx, names = _setup(rng, 3)
        T = random_transition(rng, 3, stay=(4, 7))
        s = _good(T, 40)
        if s is None or len(set(s)) < 3:
            continue
        d = _lcm(s)
        total = d * rng.randint(max(1, 500 // d), max(2, 2500 // d))
        stem = (
            ctx["intro"].format(names=names_text(names))
            + " "
            + _describe(ctx, T, names)
            + f" There are {total} {ctx['noun']} in all. In the long run the shares of the {ctx['noun']} in each group settle to a fixed distribution."
        )
        a, b, c = 0, 1, 2
        part_a = fraction_part(
            f"What fraction of the {ctx['noun']} are {ctx['at'](names[a])} in the long run?",
            s[a],
            3,
            ["Solve T s = s with the three shares adding to 1."],
            _slips_share(T, s, a),
        )
        part_b = fraction_part(
            f"What fraction of the {ctx['noun']} are {ctx['at'](names[b])} in the long run?",
            s[b],
            1,
            ["Use the same solution of T s = s."],
            _slips_share(T, s, b),
        )
        part_c = fraction_part(
            f"What fraction of the {ctx['noun']} are {ctx['at'](names[c])} in the long run?",
            s[c],
            1,
            [f"The three shares add to 1, so 1 - {s[a]} - {s[b]} = {s[c]}."],
            [
                (1 - s[a], "r3c-ss-complement"),
                (1 - s[b], "r3c-ss-complement"),
                (s[a] + s[b], "r3c-ss-sum-not-remainder"),
                (s[a] * s[b], "r3c-ss-product"),
            ],
            follow="1 - part_a - part_b",
        )
        counts = s[c] * total
        part_d = number_part(
            f"In the long run, how many of the {total} {ctx['noun']} are {ctx['at'](names[c])}?",
            counts,
            1,
            [f"{s[c]} of {total} is {int(counts)}."],
            [
                (total / 3, "r3c-ss-equal-shares"),
                (s[a] * total, "r3c-ss-wrong-state"),
                (s[b] * total, "r3c-ss-wrong-state"),
                ((1 - s[c]) * total, "r3c-ss-complement"),
            ],
            follow=f"part_c*{total}",
            lo=1,
            sensible=(0, total),
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
