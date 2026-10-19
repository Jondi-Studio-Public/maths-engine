"""Generates `transition-matrix-states` questions (easy, medium, hard) and the exam chain `r3c_transition_states_chain`.

A population is shared among three or four states (phone plans, shops, ways to travel, clubs). Each period every
member either stays or moves, so column ``j`` of the transition matrix ``T`` says where state ``j`` goes next and the
columns add to 1. The state vector then moves on by ``S(n+1) = T S(n)``. The prompt gives the movements in words,
never as a matrix; the learner builds ``T`` (or its action) and applies it.

* easy: three states, one period: the state vector after one period, as a column;
* medium: three states, two periods; one state's movements are given with "the rest" so a share is found from the
  column total;
* hard: four states, the starting numbers given as shares of a total, and the numbers in every state after three
  periods;
* exam (``r3c_transition_states_chain``): a missing "stay" share found from a column total, then the numbers in one
  state after one and two periods (carrying the share forward) and the change over the two periods.

Matrices are built first and the wording is written from them; every count in a question is a whole number.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r3c_kit import (
    col,
    mat_mul,
    mat_pow,
    names_text,
    number_part,
    number_single,
    matrix_single,
    percent,
    random_transition,
    transpose,
    vec,
)

_CONTEXTS = (
    dict(
        noun="customers",
        unit="month",
        intro="A phone company has three plans, {names}. Each month some customers change plan.",
        intro4="A phone company has four plans, {names}. Each month some customers change plan.",
        names=("Basic", "Plus", "Max", "Family"),
        at=lambda n: f"on the {n} plan",
        to=lambda n: f"change to the {n} plan",
        stay="stay on the same plan",
    ),
    dict(
        noun="shoppers",
        unit="week",
        intro="Shoppers in a town choose between {names}. Each week some of them change where they do their main shop.",
        intro4="Shoppers in a town choose between {names}. Each week some of them change where they do their main shop.",
        names=("Fresh Mart", "Green Valley", "Coastal Market", "Corner Co-op"),
        at=lambda n: f"shopping at {n}",
        to=lambda n: f"move to {n}",
        stay="shop at the same place",
    ),
    dict(
        noun="commuters",
        unit="year",
        intro="Commuters in a regional city travel to work by {names}. Each year some of them change how they travel.",
        intro4="Commuters in a regional city travel to work by {names}. Each year some of them change how they travel.",
        names=("train", "bus", "car", "bicycle"),
        at=lambda n: f"travelling by {n}",
        to=lambda n: f"change to the {n}" if n in ("train", "bus", "bicycle") else f"change to the {n}",
        stay="keep travelling the same way",
    ),
    dict(
        noun="members",
        unit="season",
        intro="A community sports centre has members in {names}. Each season some members change activity.",
        intro4="A community sports centre has members in {names}. Each season some members change activity.",
        names=("swimming", "tennis", "netball", "squash"),
        at=lambda n: f"in {n}",
        to=lambda n: f"change to {n}",
        stay="stay in the same activity",
    ),
)


def _setup(rng, n):
    ctx = rng.choice(_CONTEXTS)
    names = rng.sample(ctx["names"], n)
    return ctx, names


def _describe(ctx, T, names, *, rest_for=()):
    """One sentence per state saying where its members go. A state in ``rest_for`` has its largest move left as
    "the rest"."""
    n = len(names)
    out = []
    for j in range(n):
        moves = [(i, T[i][j]) for i in range(n) if i != j and T[i][j] > 0]
        bits = [f"{percent(T[j][j])} {ctx['stay']}"]
        if j in rest_for:
            big = max(moves, key=lambda m: m[1])
            moves.remove(big)
        bits += [f"{percent(p)} {ctx['to'](names[i])}" for i, p in moves]
        line = (
            f"Of the {ctx['noun']} {ctx['at'](names[j])}, "
            + ", ".join(bits[:-1])
            + (" and " if len(bits) > 1 else "")
            + bits[-1]
        )
        if j in rest_for:
            line += f"; the rest {ctx['to'](names[big[0]])}"
        out.append(line + ".")
    return " ".join(out)


def _start_text(ctx, names, S0):
    return ", ".join(f"{int(s)} {nm}" for s, nm in zip(S0, names[:-1])) + f" and {int(S0[-1])} {names[-1]}"


def _apply(T, S, k):
    return vec(mat_mul(mat_pow(T, k), col(S)))


def _whole(values):
    return all(v.denominator == 1 for v in values)


def _entry_slip(rng, S):
    i = rng.randrange(len(S))
    out = list(S)
    out[i] = out[i] + rng.choice((-1, 1)) * Fraction(max(10, int(S[i]) // 10))
    return out


def _vector_slips(rng, T, S0, steps):
    """Wrong state vectors from the usual errors."""
    n = len(S0)
    right = _apply(T, S0, steps)
    Tt = transpose(T)
    out = [(_apply(Tt, S0, steps), "r3c-ts-rows-columns-swapped")]
    out.append(
        ([T[i][i] * S0[i] for i in range(n)], "r3c-ts-stay-only")
        if steps == 1
        else (_apply(T, S0, 1), "r3c-ts-one-step-only")
    )
    if steps > 1:
        out.append(([sum(T[i][j] ** steps * S0[j] for j in range(n)) for i in range(n)], "r3c-ts-square-entries"))
        out.append(([S0[i] + steps * (_apply(T, S0, 1)[i] - S0[i]) for i in range(n)], "r3c-ts-double-the-change"))
    else:
        out.append((list(S0), "r3c-ts-no-change"))
    out.append((_entry_slip(rng, right), "r3c-ts-entry-slip"))
    out.append((right[::-1], "r3c-ts-order-reversed"))
    return [(col(v), tag) for v, tag in out]


_EXAMPLE = "Write the numbers as a column in the order {names}, for example [[100], [200], [300]]."


def _easy(rng):
    for _ in attempts(500, "an easy transition question"):
        ctx, names = _setup(rng, 3)
        T = random_transition(rng, 3)
        S0 = [Fraction(100 * rng.randint(4, 30)) for _ in range(3)]
        S1 = _apply(T, S0, 1)
        if not _whole(S1) or len(set(S1)) < 3 or S1 == S0:
            continue
        text = (
            ctx["intro"].format(names=names_text(names))
            + " "
            + _describe(ctx, T, names)
            + f" At the start of the first {ctx['unit']} there are {_start_text(ctx, names, S0)} {ctx['noun']}. "
            + f"How many {ctx['noun']} are in each group at the end of the {ctx['unit']}? "
            + _EXAMPLE.format(names=", ".join(names))
        )
        working = [
            f"Each group after one {ctx['unit']} is the stayers plus everyone who moves in from the other groups.",
            *[f"{nm}: {int(S1[i])}" for i, nm in enumerate(names)],
        ]
        q = matrix_single(text, col(S1), working, _vector_slips(rng, T, S0, 1))
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _medium(rng):
    for _ in attempts(500, "a medium transition question"):
        ctx, names = _setup(rng, 3)
        T = random_transition(rng, 3, stay=(4, 7))
        rest_for = (rng.randrange(3),)
        S0 = [Fraction(100 * rng.randint(4, 30)) for _ in range(3)]
        S2 = _apply(T, S0, 2)
        S1 = _apply(T, S0, 1)
        if not (_whole(S2) and _whole(S1)) or len(set(S2)) < 3 or S2 == S0:
            continue
        text = (
            ctx["intro"].format(names=names_text(names))
            + " "
            + _describe(ctx, T, names, rest_for=rest_for)
            + f" At the start there are {_start_text(ctx, names, S0)} {ctx['noun']}. "
            + f"How many {ctx['noun']} are in each group after two {ctx['unit']}s? "
            + _EXAMPLE.format(names=", ".join(names))
        )
        j = rest_for[0]
        working = [
            f"The {ctx['noun']} {ctx['at'](names[j])} move or stay with shares adding to 100%, so the share left for 'the rest' is found from the others.",
            f"After one {ctx['unit']}: {[int(v) for v in S1]}.",
            f"After two {ctx['unit']}s: {[int(v) for v in S2]}.",
        ]
        q = matrix_single(text, col(S2), working, _vector_slips(rng, T, S0, 2))
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _hard(rng):
    for _ in attempts(800, "a hard transition question"):
        ctx, names = _setup(rng, 4)
        T = random_transition(rng, 4, stay=(4, 7))
        total = 20000
        shares = [10 * rng.randint(1, 4) for _ in range(4)]
        if sum(shares) != 100:
            continue
        S0 = [Fraction(total * s, 100) for s in shares]
        steps = [_apply(T, S0, k) for k in range(4)]
        if not all(_whole(s) for s in steps) or len(set(steps[3])) < 4:
            continue
        share_text = (
            ", ".join(f"{s}% {nm}" for s, nm in zip(shares[:-1], names[:-1])) + f" and {shares[-1]}% {names[-1]}"
        )
        text = (
            ctx["intro4"].format(names=names_text(names))
            + " "
            + _describe(ctx, T, names)
            + f" There are {total} {ctx['noun']} in all, and at the start they are shared {share_text}. "
            + f"How many {ctx['noun']} are in each group after three {ctx['unit']}s? "
            + _EXAMPLE.format(names=", ".join(names)).replace("[[100], [200], [300]]", "[[100], [200], [300], [400]]")
        )
        slips = _vector_slips(rng, T, S0, 3)
        slips.insert(1, (col(steps[2]), "r3c-ts-step-count"))
        working = [
            "Find the number in each group from the shares, then move them on one period at a time.",
            *[f"After {k} {ctx['unit']}{'' if k == 1 else 's'}: {[int(v) for v in steps[k]]}" for k in (1, 2, 3)],
        ]
        q = matrix_single(text, col(steps[3]), working, slips)
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "r3c_transition_states",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3c_transition_states(rng, tier="easy"):
    """The state of a population after some periods of a transition model; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3c_transition_states_chain", level=12, presets={"exam": dict(tier="exam")})
def r3c_transition_states_chain(rng, tier="exam"):
    """Exam chain: a missing stay share, one period, two periods, and the change over the two periods."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(800, "a transition chain"):
        ctx, names = _setup(rng, 3)
        T = random_transition(rng, 3, stay=(4, 7))
        k = rng.randrange(3)  # the state whose stay share is hidden
        S0 = [Fraction(100 * rng.randint(4, 30)) for _ in range(3)]
        S1, S2 = _apply(T, S0, 1), _apply(T, S0, 2)
        if not (_whole(S1) and _whole(S2)) or S2[k] == S0[k]:
            continue
        stay = T[k][k]
        leave = 1 - stay
        flows_out = [(i, T[i][k]) for i in range(3) if i != k and T[i][k] > 0]
        if len(flows_out) < 1:
            continue
        # Describe every state except k in full; state k gives only the moves.
        lines = []
        for j in range(3):
            moves = [(i, T[i][j]) for i in range(3) if i != j and T[i][j] > 0]
            if j == k:
                bits = [f"{percent(p)} {ctx['to'](names[i])}" for i, p in moves]
                lines.append(
                    f"Of the {ctx['noun']} {ctx['at'](names[j])}, " + names_text(bits) + "; everyone else stays."
                )
            else:
                bits = [f"{percent(T[j][j])} {ctx['stay']}"] + [f"{percent(p)} {ctx['to'](names[i])}" for i, p in moves]
                lines.append(f"Of the {ctx['noun']} {ctx['at'](names[j])}, " + names_text(bits) + ".")
        stem = (
            ctx["intro"].format(names=names_text(names))
            + " "
            + " ".join(lines)
            + f" At the start there are {_start_text(ctx, names, S0)} {ctx['noun']}."
        )
        name = names[k]
        inflow = sum(T[k][j] * S0[j] for j in range(3) if j != k)
        # (a) the stay share of state k
        pct_stay = int(stay * 100)
        pct_leave = int(leave * 100)
        part_a = number_part(
            f"What percentage of the {ctx['noun']} {ctx['at'](name)} stay in that group from one {ctx['unit']} to the next?",
            pct_stay,
            1,
            [
                f"The shares leaving add to {pct_leave}%, and the shares of a group add to 100%, so {100} - {pct_leave} = {pct_stay}."
            ],
            [
                (pct_leave, "r3c-ts-leaving-not-staying"),
                (100 - int(max(p for _, p in flows_out) * 100), "r3c-ts-one-flow-only"),
                (pct_stay - 10, "r3c-ts-column-total"),
                (pct_stay + 10, "r3c-ts-column-total"),
            ],
            lo=0,
            sensible=(0, 100),
        )
        # (b) the number in k after one period, from part (a)
        c1 = S1[k]
        part_b = number_part(
            f"How many {ctx['noun']} are {ctx['at'](name)} after one {ctx['unit']}?",
            c1,
            2,
            [f"Stayers {int(stay * S0[k])} plus arrivals {int(inflow)} gives {int(c1)}."],
            [
                (leave * S0[k] + inflow, "r3c-ts-leaving-not-staying"),
                (stay * S0[k], "r3c-ts-ignores-inflows"),
                (inflow, "r3c-ts-ignores-stayers"),
                (S0[k] + inflow, "r3c-ts-nobody-leaves"),
            ],
            follow=f"part_a/100*{int(S0[k])} + {int(inflow)}",
            lo=1,
            sensible=(0, None),
        )
        # (c) the number in k after two periods
        inflow1 = sum(T[k][j] * S1[j] for j in range(3) if j != k)
        c2 = S2[k]
        part_c = number_part(
            f"How many {ctx['noun']} are {ctx['at'](name)} after two {ctx['unit']}s?",
            c2,
            3,
            [f"Stayers {int(stay * c1)} plus arrivals {int(inflow1)} gives {int(c2)}."],
            [
                (c1, "r3c-ts-one-step-only"),
                (stay * c1 + inflow, "r3c-ts-uses-start-values"),
                (leave * c1 + inflow1, "r3c-ts-leaving-not-staying"),
                (inflow1, "r3c-ts-ignores-stayers"),
            ],
            follow=f"part_a/100*part_b + {int(inflow1)}",
            lo=1,
            sensible=(0, None),
        )
        change = c2 - S0[k]
        part_d = number_part(
            f"By how many has the number {ctx['at'](name)} {'risen' if change > 0 else 'fallen'} over the two {ctx['unit']}s?",
            abs(change),
            1,
            [f"{int(c2)} compared with {int(S0[k])} at the start."],
            [
                (c2, "r3c-ts-count-not-change"),
                (abs(c2 - c1), "r3c-ts-wrong-interval"),
                (abs(c1 - S0[k]), "r3c-ts-wrong-interval"),
                (abs(change) + 100, "r3c-ts-entry-slip"),
            ],
            follow=f"Abs(part_c - {int(S0[k])})",
            lo=1,
            sensible=(0, None),
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
