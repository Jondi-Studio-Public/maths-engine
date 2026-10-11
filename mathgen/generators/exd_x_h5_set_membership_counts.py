"""Exam-tier source for `senior-determine-and-use-the-formulas-for-finding-the`.

Registers the topic `exd_x_h5_set_membership_counts`.

The easy/medium/hard tiers apply inclusion-exclusion to named overlaps. This exam tier works with
*membership counts* instead: the rosters of three clubs are given only as sizes, so their sum counts every
student once per club they belong to. Writing ``e1, e2, e3`` for the numbers of students in exactly one, exactly
two and all three clubs, and ``e0`` for those in none:

    e0 + e1 + e2 + e3 = N              (every student)
    e1 + 2*e2 + 3*e3  = a + b + c      (every membership)

Two of the four counts are stated, one more is asked for, and the learner has to set up and solve the two
linear relations themselves (the choice of method is theirs). The four numbers are drawn first, the prompt is
built from them, and the stated pair plus the two relations pin the asked-for count uniquely.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_distinct, generator, phrase, pick_distractors

_CLUBS = (
    ("chess", "debating", "robotics"),
    ("drama", "choir", "orchestra"),
    ("swimming", "athletics", "tennis"),
    ("coding", "art", "science"),
    ("film", "photography", "journalism"),
)

# (known group, asked group) -- groups are 1 = exactly one, 2 = exactly two, 3 = all three.
_COMBOS = ((3, 2), (3, 1), (2, 3), (2, 1))
_GROUP_PHRASE = {1: "exactly one club", 2: "exactly two of the clubs", 3: "all three clubs"}

_OPENERS = (
    "A school has {N} students and three clubs. The {c1} club has {a} members, the {c2} club has {b} and the {c3} club has {c}.",
    "Of {N} students at a school, the {c1} club has {a} members, the {c2} club has {b} and the {c3} club has {c}.",
    "Three clubs run at a school with {N} students: {c1} ({a} members), {c2} ({b} members) and {c3} ({c} members).",
)
_NONE = (
    "{e0} students belong to none of the clubs and {k} students belong to {known}.",
    "{e0} students are in no club at all, while {k} students belong to {known}.",
)
_ASKS = (
    "How many students belong to {asked}?",
    "Find the number of students who belong to {asked}.",
)


def _solve(weights, union, memberships, known_group, known_value):
    """Solve the two membership relations for the two unknown groups, using the given per-group weights.

    Returns ``{group: count}`` for the two groups other than ``known_group`` as exact Fractions.
    """
    ga, gb = [g for g in (1, 2, 3) if g != known_group]
    rhs1 = Fraction(union - known_value)
    rhs2 = Fraction(memberships - weights[known_group] * known_value)
    wa, wb = weights[ga], weights[gb]
    xa = (rhs2 - wb * rhs1) / (wa - wb)
    return {ga: xa, gb: rhs1 - xa}


@generator("exd_x_h5_set_membership_counts", level=13, presets={"exam": dict(tier="exam")})
def exd_x_h5_set_membership_counts(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _attempt in range(200):
        # Draw the seven Venn regions first, so the three rosters are consistent with a real arrangement.
        only = [rng.randint(2, 8) for _ in range(3)]
        pair = [rng.randint(2, 6) for _ in range(3)]  # AB, AC, BC (exactly those two)
        triple = rng.randint(2, 6)
        e = {1: sum(only), 2: sum(pair), 3: triple}
        e0 = rng.randint(2, 12)
        known_group, asked_group = rng.choice(_COMBOS)
        n_total = e0 + e[1] + e[2] + e[3]
        a = only[0] + pair[0] + pair[1] + triple
        b = only[1] + pair[0] + pair[2] + triple
        c = only[2] + pair[1] + pair[2] + triple
        memberships = a + b + c
        check(memberships == e[1] + 2 * e[2] + 3 * e[3], "rosters must add to the membership total")
        union = n_total - e0
        answer = e[asked_group]
        weights = {1: 1, 2: 2, 3: 3}
        sol = _solve(weights, union, memberships, known_group, e[known_group])
        check(sol[asked_group] == answer, "membership relations do not return the drawn count")
        check(
            e0 + e[known_group] + sum(sol.values()) == n_total and a + b + c == memberships,
            "student and membership totals must both reproduce the drawn counts",
        )

        wrong = []
        for w in ({1: 1, 2: 2, 3: 2}, {1: 1, 2: 3, 3: 3}, {1: 1, 2: 1, 3: 3}):
            if w[1] == w[2] and known_group == 3 or w[2] == w[3] and known_group == 1:
                continue
            try:
                s = _solve(w, union, memberships, known_group, e[known_group])
            except ZeroDivisionError:
                continue
            wrong.append(s[asked_group])
        # Forgetting the students who are in no club: the union is taken to be the whole school.
        wrong.append(_solve(weights, n_total, memberships, known_group, e[known_group])[asked_group])
        cands = [v for v in wrong if v.denominator == 1 and v >= 0 and v != answer]
        if len(set(cands)) < 3:
            continue
        break
    else:
        raise RuntimeError("could not draw a set-membership question with three clean distractors")

    club_names = rng.choice(_CLUBS)
    asked = _GROUP_PHRASE[asked_group]
    known = _GROUP_PHRASE[known_group]
    opener = phrase(rng, _OPENERS, N=n_total, c1=club_names[0], c2=club_names[1], c3=club_names[2], a=a, b=b, c=c)
    middle = phrase(rng, _NONE, e0=e0, k=e[known_group], known=known)
    ask = phrase(rng, _ASKS, asked=asked)
    text = f"{opener} {middle} {ask}"
    answer_text = str(answer)
    for cand in cands:
        check_distinct(answer, int(cand), msg="a distractor equals the answer")
    distractors = pick_distractors(answer_text, [str(int(v)) for v in cands], n=3)
    check(len(distractors) == 3, "expected three distractors")
    return {
        "text": text,
        "latex": text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
    }
