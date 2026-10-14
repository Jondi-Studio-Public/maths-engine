"""Generators for `f10-read-a-simulation`: run a simulation that is already described, then estimate from its results.

The simulation is given, never designed: a rule says which random digits stand for what, the digits are listed, and the
learner reads the results. Three tiers answered as fractions, plus a multi-part exam chain:

* easy: one digit is one trial; count the trials where a rule digit appeared and estimate the chance.
* medium: a pair of digits is one trial (two days, two shots); estimate the chance of both, at least one, or exactly one.
* hard: a two-draw simulation without replacing, each trial recorded as the colours drawn in order; estimate a chance
  given the first (or second) draw, which is P(B given A) from two-draw counts.
* exam (`fr2d_read_simulation_exam`): count the successful trials, estimate, find the theoretical chance from the
  rule, and compare the count with what the theory expects.

The digits are drawn with the seeded rng, so the results are real random output, not built to hit a target. Every key is
re-derived from the printed text (a second count over the rendered string) and every option is a proper fraction
different from the key.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import exact_value
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    rational_text,
)

from ._fr2d_kit import FRACTION_SUFFIX, frac_d, int_d, is_proper, need, redraw

_CONTEXTS = (
    dict(
        rule="Each digit stands for one day. The {digits} stand for a day with rain, and the other digits "
        "stand for a dry day.",
        one="it rains on a day",
        both="it rains on both days",
        atleast="it rains on at least one of the two days",
        exactly="it rains on exactly one of the two days",
        what="a weather simulation",
    ),
    dict(
        rule="Each digit stands for one free throw. The {digits} stand for a throw that scores, and the "
        "other digits stand for a miss.",
        one="a free throw scores",
        both="both free throws score",
        atleast="at least one of the two free throws scores",
        exactly="exactly one of the two free throws scores",
        what="a basketball simulation",
    ),
    dict(
        rule="Each digit stands for one morning. The {digits} stand for a morning when the bus is late, "
        "and the other digits stand for a morning when it is on time.",
        one="the bus is late on a morning",
        both="the bus is late on both mornings",
        atleast="the bus is late on at least one of the two mornings",
        exactly="the bus is late on exactly one of the two mornings",
        what="a bus simulation",
    ),
    dict(
        rule="Each digit stands for one seed. The {digits} stand for a seed that grows, and the other "
        "digits stand for a seed that does not.",
        one="a seed grows",
        both="both seeds grow",
        atleast="at least one of the two seeds grows",
        exactly="exactly one of the two seeds grows",
        what="a planting simulation",
    ),
)

_EVENT_KEYS = ("both", "atleast", "exactly")


def _digits_text(lo, hi):
    return f"digits {lo} and {hi}" if hi == lo + 1 else f"digits {lo} to {hi}"


def _rule_range(rng, sizes):
    k = rng.choice(sizes)
    lo = rng.randint(0, 10 - k)
    return lo, lo + k - 1, k


def _digits(rng, n):
    return [rng.randint(0, 9) for _ in range(n)]


def _wins(d, lo, hi):
    return lo <= d <= hi


def _ask_fraction(rng, what, event):
    return (
        phrase(
            rng,
            (
                "Use the results to estimate the probability that {event}.",
                "From these results, estimate the probability that {event}.",
            ),
            event=event,
        )
        + f" {FRACTION_SUFFIX}"
    )


def _fraction_dict(stem, p, working, distractors):
    return dict(
        text=stem,
        answer_kind="fraction",
        answer=exact_value(p),
        answer_text=rational_text(p),
        answer_latex=latex_answer(rational_text(p)),
        working=working,
        distractors=distractors,
    )


# -- easy: one digit, one trial -------------------------------------------------------------------------------


def _easy(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(2000, "a digit list with a clear estimate"):
        lo, hi, k = _rule_range(rng, [2, 3, 4, 5])
        n = rng.choice([20, 25, 30])
        digits = _digits(rng, n)
        wins = sum(_wins(d, lo, hi) for d in digits)
        p = Fraction(wins, n)
        theory = Fraction(k, 10)
        if not 0 < p < 1 or p == theory or p == Fraction(1, 2):
            continue
        cands = [
            frac_d(1 - p, "fr2d-gives-the-opposite-event"),
            frac_d(Fraction(wins, n - wins), "fr2d-odds-for-probability"),
            frac_d(theory, "fr2d-gives-theory-not-simulation"),
            frac_d(Fraction(wins, 10), "fr2d-divides-by-the-digits-not-trials"),
            frac_d(Fraction(wins, n + wins), "fr2d-wrong-trial-total"),
        ]
        opts = pick_distractors(rational_text(p), [c for c in cands if is_proper(c)], 3)
        if len(opts) == 3:
            break
    shown = " ".join(map(str, digits))
    # second route: read the count back from the printed digits
    printed = [int(t) for t in shown.split()]
    check(
        sum(1 for d in printed if lo <= d <= hi) == wins and len(printed) == n, "count differs from the printed digits"
    )
    stem = (
        f"Here is {ctx['what']}. {ctx['rule'].format(digits=_digits_text(lo, hi))} Each digit is one trial. The {n} random digits were: "
        f"{shown}. {_ask_fraction(rng, ctx['what'], ctx['one'])}"
    )
    working = [
        f"Digits from {lo} to {hi} count as a success: {wins} of the {n} digits.",
        f"{wins}/{n} = {rational_text(p)}.",
    ]
    return _fraction_dict(stem, p, working, opts)


# -- medium: a pair of digits is one trial ----------------------------------------------------------------------


def _event_holds(key, a, b):
    return {"both": a and b, "atleast": a or b, "exactly": a != b}[key]


def _theory(key, p):
    return {"both": p * p, "atleast": 1 - (1 - p) ** 2, "exactly": 2 * p * (1 - p)}[key]


def _pairs_text(pairs):
    return " ".join(f"{a}{b}" for a, b in pairs)


def _medium(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(3000, "a list of pairs with a clear estimate"):
        lo, hi, k = _rule_range(rng, [3, 4, 5, 6])
        n = rng.choice([20, 25])
        key = rng.choice(_EVENT_KEYS)
        pairs = [(rng.randint(0, 9), rng.randint(0, 9)) for _ in range(n)]
        wins = sum(_event_holds(key, _wins(a, lo, hi), _wins(b, lo, hi)) for a, b in pairs)
        p = Fraction(wins, n)
        theory = _theory(key, Fraction(k, 10))
        if not 0 < p < 1 or p == theory:
            continue
        digit_wins = sum(_wins(a, lo, hi) + _wins(b, lo, hi) for a, b in pairs)
        both = sum(_wins(a, lo, hi) and _wins(b, lo, hi) for a, b in pairs)
        atleast = sum(_wins(a, lo, hi) or _wins(b, lo, hi) for a, b in pairs)
        confusions = {
            "both": (Fraction(atleast, n), "fr2d-or-instead-of-and"),
            "atleast": (Fraction(both, n), "fr2d-and-instead-of-or"),
            "exactly": (Fraction(atleast, n), "fr2d-counts-both-as-exactly-one"),
        }[key]
        cands = [
            frac_d(Fraction(digit_wins, 2 * n), "fr2d-counts-digits-not-trials"),
            frac_d(confusions[0], confusions[1]),
            frac_d(1 - p, "fr2d-gives-the-opposite-event"),
            frac_d(theory, "fr2d-gives-theory-not-simulation"),
            frac_d(Fraction(wins, 2 * n), "fr2d-counts-digits-not-trials"),
        ]
        opts = pick_distractors(rational_text(p), [c for c in cands if is_proper(c)], 3)
        if len(opts) == 3:
            break
    shown = _pairs_text(pairs)
    tokens = shown.split()
    check(len(tokens) == n, "the printed pairs do not match the number of trials")
    check(
        sum(_event_holds(key, lo <= int(t[0]) <= hi, lo <= int(t[1]) <= hi) for t in tokens) == wins,
        "count differs from the printed pairs",
    )
    stem = (
        f"Here is {ctx['what']}. {ctx['rule'].format(digits=_digits_text(lo, hi))} Two digits make one trial. The {n} trials were: "
        f"{shown}. {_ask_fraction(rng, ctx['what'], ctx[key])}"
    )
    description = {
        "both": "both digits in the pair are from",
        "atleast": "at least one digit in the pair is from",
        "exactly": "exactly one digit in the pair is from",
    }[key]
    working = [
        f"A trial is a success when {description} {lo} to {hi}.",
        f"{wins} of the {n} trials are successes.",
        f"{wins}/{n} = {rational_text(p)}.",
    ]
    return _fraction_dict(stem, p, working, opts)


# -- hard: two draws without replacing, conditional estimate --------------------------------------------------


_BAGS = (
    dict(
        lead="A bag holds {r} red and {b} blue counters. Two counters are drawn one after the other without putting "
        "the first back. This was simulated {n} times and each result was recorded as the colour of the first "
        "counter then the colour of the second (RB means red then blue).",
        letters=("R", "B"),
        names=("red", "blue"),
        first="the first counter is {a}",
        second="the second counter is {a}",
    ),
    dict(
        lead="A group has {r} girls and {b} boys. A captain and then a vice-captain are chosen at random from the "
        "group, and nobody can hold both jobs. This was simulated {n} times and each result was recorded as the "
        "captain then the vice-captain (GB means a girl captain and a boy vice-captain).",
        letters=("G", "B"),
        names=("a girl", "a boy"),
        first="the captain is {a}",
        second="the vice-captain is {a}",
    ),
)


def _simulate_pairs(rng, r, b, n, letters):
    bag = [letters[0]] * r + [letters[1]] * b
    out = []
    for _ in range(n):
        i, j = rng.sample(range(len(bag)), 2)
        out.append(bag[i] + bag[j])
    return out


def _hard(rng):
    ctx = rng.choice(_BAGS)
    L0, L1 = ctx["letters"]
    for _ in attempts(3000, "a two-draw simulation with a clear conditional estimate"):
        r, b = rng.randint(3, 6), rng.randint(3, 6)
        n = 30
        results = _simulate_pairs(rng, r, b, n, ctx["letters"])
        given_first = rng.random() < 0.75
        which_given = rng.choice([L0, L1])
        which_target = rng.choice([L0, L1])
        if given_first:
            cond = [t for t in results if t[0] == which_given]
            hit = [t for t in cond if t[1] == which_target]
        else:
            cond = [t for t in results if t[1] == which_given]
            hit = [t for t in cond if t[0] == which_target]
        if len(cond) < 8 or not 0 < len(hit) < len(cond):
            continue
        p = Fraction(len(hit), len(cond))
        n_cond, n_hit = len(cond), len(hit)
        # theoretical conditional chance of the target given the condition
        counts = {L0: r, L1: b}
        remaining = dict(counts)
        remaining[which_given] -= 1
        theory = Fraction(remaining[which_target], r + b - 1)
        other_cond = [t for t in results if (t[1] if given_first else t[0]) == which_target]
        swapped = Fraction(n_hit, len(other_cond)) if other_cond else None
        cands = [
            frac_d(Fraction(n_hit, n), "fr2d-joint-for-conditional"),
            frac_d(swapped, "fr2d-swaps-condition-and-target") if swapped is not None else None,
            frac_d(Fraction(n_cond, n), "fr2d-ignores-the-target"),
            frac_d(1 - p, "fr2d-gives-the-opposite-event"),
            frac_d(theory, "fr2d-gives-theory-not-simulation"),
            frac_d(Fraction(len(other_cond), n), "fr2d-ignores-the-condition"),
        ]
        opts = pick_distractors(rational_text(p), [c for c in cands if c is not None and is_proper(c)], 3)
        if len(opts) == 3:
            break
    shown = " ".join(results)
    tokens = shown.split()
    idx_c, idx_t = (0, 1) if given_first else (1, 0)
    check(
        Fraction(
            sum(1 for t in tokens if t[idx_c] == which_given and t[idx_t] == which_target),
            sum(1 for t in tokens if t[idx_c] == which_given),
        )
        == p,
        "conditional count differs from the printed results",
    )
    pos, side = ("first", "second") if given_first else ("second", "first")

    def clause(position, letter):
        return ctx[position].format(a=ctx["names"][ctx["letters"].index(letter)])

    stem = (
        ctx["lead"].format(r=r, b=b, n=n)
        + f" The {n} results were: {shown}. "
        + phrase(
            rng,
            (
                "Use the results to estimate the probability that {x}, given that {y}.",
                "Estimate, from the simulation, the probability that {x} when {y}.",
            ),
            x=clause(side, which_target),
            y=clause(pos, which_given),
        )
        + f" {FRACTION_SUFFIX}"
    )
    working = [
        f"Keep only the results where {clause(pos, which_given)}: {n_cond} of the {n}.",
        f"Of those, {n_hit} have {clause(side, which_target)}.",
        f"{n_hit}/{n_cond} = {rational_text(p)}.",
    ]
    return _fraction_dict(stem, p, working, opts)


@generator(
    "fr2d_read_simulation",
    level=8,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
@redraw
def fr2d_read_simulation(rng, tier: str = "easy"):
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# -- exam chain ---------------------------------------------------------------------------------------------------


@generator("fr2d_read_simulation_exam", level=9, presets={"exam": {"tier": "exam"}})
@redraw
def fr2d_read_simulation_exam(rng, tier: str = "exam"):
    ctx = rng.choice(_CONTEXTS)
    n = 25
    for _ in attempts(4000, "a simulation with clear counts for the chain"):
        k = rng.choice([2, 4, 6, 8])
        lo = rng.randint(0, 10 - k)
        hi = lo + k - 1
        key = rng.choice(_EVENT_KEYS)
        pairs = [(rng.randint(0, 9), rng.randint(0, 9)) for _ in range(n)]
        wins = sum(_event_holds(key, _wins(a, lo, hi), _wins(b, lo, hi)) for a, b in pairs)
        p_theory = _theory(key, Fraction(k, 10))
        expected = n * p_theory
        if expected.denominator != 1 or not 1 < wins < n - 1 or wins == expected:
            continue
        p_obs = Fraction(wins, n)
        diff = wins - int(expected)
        digit_wins = sum(_wins(a, lo, hi) + _wins(b, lo, hi) for a, b in pairs)
        break
    p1 = Fraction(k, 10)
    shown = _pairs_text(pairs)
    tokens = shown.split()
    check(
        sum(_event_holds(key, lo <= int(t[0]) <= hi, lo <= int(t[1]) <= hi) for t in tokens) == wins,
        "count differs from the printed pairs",
    )
    check(
        p_theory == {"both": p1**2, "atleast": 1 - (1 - p1) ** 2, "exactly": p1 * (1 - p1) * 2}[key], "theory differs"
    )
    event = ctx[key]
    stem = (
        f"Here is {ctx['what']}. {ctx['rule'].format(digits=_digits_text(lo, hi))} Two digits make one trial. The {n} trials were: "
        f"{shown}. In each trial, the event of interest is that {event}."
    )
    both_c = sum(_wins(a, lo, hi) and _wins(b, lo, hi) for a, b in pairs)
    atleast_c = sum(_wins(a, lo, hi) or _wins(b, lo, hi) for a, b in pairs)
    alt_count = {"both": atleast_c, "atleast": both_c, "exactly": atleast_c}[key]
    a_opts = pick_distractors(
        str(wins),
        [
            int_d(digit_wins, "fr2d-counts-digits-not-trials"),
            int_d(alt_count, "fr2d-or-and-confused"),
            int_d(n - wins, "fr2d-counts-the-opposite-event"),
            int_d(wins + 1, "fr2d-count-off-by-one"),
            int_d(max(0, wins - 1), "fr2d-count-off-by-one"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_cands = [
        frac_d(Fraction(wins, 2 * n), "fr2d-counts-digits-not-trials"),
        frac_d(1 - p_obs, "fr2d-gives-the-opposite-event"),
        frac_d(Fraction(alt_count, n), "fr2d-or-and-confused"),
        frac_d(p_theory, "fr2d-gives-theory-not-simulation"),
    ]
    c_cands = {
        "both": [
            frac_d(p1, "fr2d-one-digit-chance-for-the-pair"),
            frac_d(2 * p1, "fr2d-adds-chances-for-the-pair"),
            frac_d(1 - (1 - p1) ** 2, "fr2d-or-instead-of-and"),
            frac_d(p_obs, "fr2d-gives-simulation-not-theory"),
        ],
        "atleast": [
            frac_d(p1, "fr2d-one-digit-chance-for-the-pair"),
            frac_d(2 * p1, "fr2d-adds-chances-for-the-pair"),
            frac_d(p1**2, "fr2d-and-instead-of-or"),
            frac_d(p_obs, "fr2d-gives-simulation-not-theory"),
        ],
        "exactly": [
            frac_d(p1, "fr2d-one-digit-chance-for-the-pair"),
            frac_d(p1**2, "fr2d-and-instead-of-or"),
            frac_d(1 - (1 - p1) ** 2, "fr2d-counts-both-as-exactly-one"),
            frac_d(p_obs, "fr2d-gives-simulation-not-theory"),
        ],
    }[key] + [frac_d(1 - p_theory, "fr2d-gives-the-opposite-event")]
    b_opts = pick_distractors(rational_text(p_obs), [c for c in b_cands if is_proper(c)], 3)
    c_opts = pick_distractors(rational_text(p_theory), [c for c in c_cands if is_proper(c)], 3)
    d_opts = pick_distractors(
        str(diff),
        [
            int_d(-diff, "fr2d-difference-the-wrong-way"),
            int_d(wins, "fr2d-count-for-difference-from-expected"),
            int_d(int(expected), "fr2d-expected-for-difference"),
            int_d(wins + int(expected), "fr2d-adds-instead-of-compares"),
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    need(a_opts, b_opts, c_opts, d_opts)
    pt, po = rational_text(p_theory), rational_text(p_obs)
    parts = [
        dict(
            prompt=f"How many of the {n} trials had the event of interest?",
            answer_kind="number",
            answer=wins,
            answer_text=str(wins),
            answer_latex=latex_answer(str(wins)),
            marks=2,
            sensible_range=(0, n),
            working=[f"Check each pair against the digits {lo} to {hi}: {wins} trials qualify."],
            distractors=a_opts,
        ),
        dict(
            prompt=f"Use the simulation to estimate the probability of the event. {FRACTION_SUFFIX}",
            answer_kind="fraction",
            answer=exact_value(p_obs),
            answer_text=po,
            answer_latex=latex_answer(po),
            marks=1,
            sensible_range=(0, 1),
            follow_through=f"part_a/{n}",
            working=[f"{wins}/{n} = {po}."],
            distractors=b_opts,
        ),
        dict(
            prompt=(
                "Digits are equally likely, so use the rule to work out the theoretical probability of the event. "
                f"{FRACTION_SUFFIX}"
            ),
            answer_kind="fraction",
            answer=exact_value(p_theory),
            answer_text=pt,
            answer_latex=latex_answer(pt),
            marks=2,
            sensible_range=(0, 1),
            working=[
                f"One digit succeeds with probability {k}/10 = {rational_text(p1)}.",
                f"For the pair, the event has probability {pt}.",
            ],
            distractors=c_opts,
        ),
        dict(
            prompt=(
                f"In {n} trials, how many more trials than the theory expects had the event? "
                "Give a negative number if there were fewer."
            ),
            answer_kind="number",
            answer=diff,
            answer_text=str(diff),
            answer_latex=latex_answer(str(diff)),
            marks=2,
            sensible_range=(-n, n),
            follow_through=f"part_a-{n}*part_c",
            working=[f"Expected: {n} x {pt} = {int(expected)}.", f"{wins} - {int(expected)} = {diff}."],
            distractors=d_opts,
        ),
    ]
    return dict(text=stem, parts=parts)
