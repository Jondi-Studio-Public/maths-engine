"""Exam chain for `f10-conditional-probability-language`: reading a two-way table as a conditional probability.

The practice tiers only read the wording ("given", "knowing that", "if ... then", "of") and pick which event is the
condition. The exam question uses that wording to calculate. A group is sorted two ways and the four cell counts are
stated in a sentence. One event is the condition and one is the target, and either may be the "not" form of its pair:

(a) how many are in the group that satisfies the condition        whole number,  1 mark
(b) the probability of the target given the condition            fraction,      2 marks, follow-through (a)
(c) the probability of the target with no condition              fraction,      1 mark
(d) how many percentage points the condition moves it            1 d.p.,        2 marks, follow-through (b), (c)

Part (d) is where independence is read: a change of 0.0 means the events are independent, and a positive or negative
change says which way knowing the condition pushes the chance.

## Construction

Both row totals are multiples of 20 and each row's share of the target is a whole number of twentieths, so every cell
is a whole number. About a third of the draws make the two rows share the same proportion (independent events); the
rest differ by at least two twentieths. A draw is kept only when the change in percentage points is exact to one
decimal place, so the key never needs rounding. Part (b) is also recomputed from the four cells directly.

## Distractors

(a) the cell where both events happen, the complement group, the target's total.
(b) the cell over the whole group, the cell over the target's total (condition and target swapped), the unconditional
probability.
(c) the joint probability, the conditional probability, the probability of the opposite target.
(d) the sign reversed, "no change", the change measured as a percentage of the original probability, the joint
probability in place of the conditional one.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_r2x_kit import frac_part, num_part

# Each scenario: the group, the one-line introduction, then each event as (plural predicate, singular predicate).
_SCENARIOS = (
    dict(
        noun="students",
        one="student",
        intro="{n} students were asked how they get to school and whether they have a part-time job.",
        rows=(("walk to school", "walks to school"), ("are driven to school", "is driven to school")),
        cols=(
            ("have a part-time job", "has a part-time job"),
            ("do not have a part-time job", "does not have a part-time job"),
        ),
    ),
    dict(
        noun="members",
        one="member",
        intro="A community pool has {n} members. Each was asked whether they swim laps and whether they use the sauna.",
        rows=(("swim laps", "swims laps"), ("do not swim laps", "does not swim laps")),
        cols=(("use the sauna", "uses the sauna"), ("do not use the sauna", "does not use the sauna")),
    ),
    dict(
        noun="households",
        one="household",
        intro="A survey of {n} households in a suburb asked about pets and about solar panels.",
        rows=(("own a dog", "owns a dog"), ("do not own a dog", "does not own a dog")),
        cols=(("have solar panels", "has solar panels"), ("do not have solar panels", "does not have solar panels")),
    ),
    dict(
        noun="commuters",
        one="commuter",
        intro="{n} commuters at a railway station were asked how they travel and whether they bought a coffee.",
        rows=(("travel by train", "travels by train"), ("travel by bus", "travels by bus")),
        cols=(("bought a coffee", "bought a coffee"), ("did not buy a coffee", "did not buy a coffee")),
    ),
    dict(
        noun="players",
        one="player",
        intro="{n} players in a netball association were asked whether they play in the juniors and whether they umpire.",
        rows=(
            ("play in the juniors", "plays in the juniors"),
            ("do not play in the juniors", "does not play in the juniors"),
        ),
        cols=(("umpire", "umpires"), ("do not umpire", "does not umpire")),
    ),
)

_ASK_B = (
    "Using your answer to (a), find the probability that a {one} chosen at random {t_sg}, given that the {one} {c_sg}. "
    "Give the answer as a fraction in simplest form.",
    "Using your answer to (a): knowing that a {one} chosen at random {c_sg}, what is the probability that the {one} "
    "{t_sg}? Give the answer as a fraction in simplest form.",
    "Using your answer to (a): if a {one} chosen at random {c_sg}, what is the probability that the {one} {t_sg}? "
    "Give the answer as a fraction in simplest form.",
    "Using your answer to (a), what fraction of the {noun} who {c_pl} also {t_pl}? Give the answer as a fraction in "
    "simplest form.",
)


def _draw(rng, independent):
    """Cell counts ``n[row][col]``: rows are the first event pair, columns the second."""
    for _ in attempts(2000, "a table of whole-number cells"):
        r = [rng.choice(range(40, 181, 20)), rng.choice(range(40, 181, 20))]
        i1 = rng.randint(2, 18)
        i2 = i1 if independent else rng.choice([v for v in range(2, 19) if abs(v - i1) >= 2])
        n = [[r[0] * i1 // 20, r[0] - r[0] * i1 // 20], [r[1] * i2 // 20, r[1] - r[1] * i2 // 20]]
        if (n[0][0] * 20 != r[0] * i1) or (n[1][0] * 20 != r[1] * i2):
            continue
        return n
    raise AssertionError("unreachable")


@generator("exd_x_r2x_conditional_probability_language", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_r2x_conditional_probability_language(rng, tier: str = "exam"):
    """Count the group, find a conditional probability, find the plain one, then read independence from the change."""
    check(tier == "exam", f"unknown tier {tier!r}")
    sc = rng.choice(_SCENARIOS)
    independent = rng.random() < 0.3
    for _ in attempts(3000, "a draw with an exact one-decimal change"):
        n = _draw(rng, independent)
        row = rng.randint(0, 1)  # the condition
        col = rng.randint(0, 1)  # the target
        total = sum(map(sum, n))
        group = sum(n[row])
        target_total = n[0][col] + n[1][col]
        cell = n[row][col]
        cond = Fraction(cell, group)
        plain = Fraction(target_total, total)
        change = 100 * (cond - plain)
        if (change * 10).denominator != 1:
            continue
        if independent and change != 0:
            continue
        break
    # A second route for each part, from the four stated cells alone.
    flat = {(i, j): n[i][j] for i in (0, 1) for j in (0, 1)}
    check(group == flat[(row, 0)] + flat[(row, 1)], "group total disagrees with its cells")
    check(cond == Fraction(flat[(row, col)], flat[(row, 0)] + flat[(row, 1)]), "conditional probability disagrees")
    check((change == 0) == (cond == plain), "no change must coincide with equal probabilities")

    c_pl, c_sg = sc["rows"][row]
    t_pl, t_sg = sc["cols"][col]
    cell_text = []
    for i in (0, 1):
        for j in (0, 1):
            cell_text.append(f"{n[i][j]} {sc['rows'][i][0]} and {sc['cols'][j][0]}")
    stem = (
        sc["intro"].format(n=total) + f" Of them, {cell_text[0]}; {cell_text[1]}; {cell_text[2]}; and {cell_text[3]}."
    )
    noun, one = sc["noun"], sc["one"]
    t_opp = Fraction(total - target_total, total)

    def points(value):
        return 100 * Fraction(value)

    parts = [
        num_part(
            f"How many of the {total} {noun} {c_pl}?",
            group,
            0,
            1,
            [f"The number of {noun} who {c_pl} is the sum of its two cells: {n[row][0]} + {n[row][1]} = {group}."],
            [
                (cell, "r2x-cond-joint-not-group-total"),
                (total - group, "r2x-cond-complement-group"),
                (target_total, "r2x-cond-other-event-total"),
                (total - cell, "r2x-cond-complement-group"),
            ],
        ),
        frac_part(
            phrase(rng, _ASK_B, one=one, noun=noun, c_sg=c_sg, t_sg=t_sg, c_pl=c_pl, t_pl=t_pl),
            cond,
            2,
            [
                f"Only the {group} {noun} who {c_pl} are possible, so {cell} of those {group} {t_pl}.",
                f"{cell}/{group} = {cond.numerator}/{cond.denominator}.",
            ],
            [
                (Fraction(cell, total), "r2x-cond-whole-group-denominator"),
                (Fraction(cell, target_total), "r2x-cond-swapped-condition-target"),
                (plain, "r2x-cond-ignores-condition"),
                (Fraction(group - cell, group), "r2x-cond-complement-target"),
            ],
            follow=f"{cell}/part_a",
        ),
        frac_part(
            f"What is the probability that a {one} chosen at random from all {total} {noun} {t_sg}? "
            f"Give the answer as a fraction in simplest form.",
            plain,
            1,
            [f"{target_total} of the {total} {noun} {t_pl}, so the probability is {target_total}/{total}."],
            [
                (Fraction(cell, total), "r2x-cond-joint-for-single"),
                (cond, "r2x-cond-condition-applied-with-none-given"),
                (t_opp, "r2x-cond-complement-target"),
                (Fraction(group, total), "r2x-cond-condition-for-target"),
            ],
        ),
        num_part(
            f"Use your answers to (b) and (c) to work out how knowing that a {one} {c_sg} affects the probability "
            f"that the {one} {t_sg}. Give the change in percentage points, correct to 1 decimal place: positive "
            f"for an increase, negative for a decrease, and 0.0 for no change.",
            change,
            1,
            2,
            [
                f"Conditional probability minus plain probability: {cond.numerator}/{cond.denominator} - "
                f"{plain.numerator}/{plain.denominator}.",
                "Multiply the difference by 100 to give percentage points. Zero means the events are independent.",
            ],
            [
                (-change, "r2x-cond-sign-reversed"),
                (0, "r2x-cond-assumes-independent"),
                (100 * (cond / plain - 1), "r2x-cond-relative-not-points"),
                (points(Fraction(cell, total)) - points(plain), "r2x-cond-joint-for-conditional"),
                (points(cond), "r2x-cond-probability-not-change"),
            ],
            follow="100*(part_b - part_c)",
            sensible=None,
        ),
    ]
    return dict(text=stem, parts=parts)
