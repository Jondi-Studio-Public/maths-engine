"""Generates `f10-spatial-algorithm-choose-test` questions and its exam chain. `answer: mcq` (the exam chain also has
typed number and ordered-list parts).

Scope: the procedures behind Years 7 to 10 geometry. A construction is a short numbered procedure; a congruence test
is a procedure that asks which facts about two triangles are enough. The learner chooses the right order for a scrambled
construction, or says which extra fact completes a named congruence test. Writing a new construction, proving a test
correct, and similarity are out of scope. Every question has exactly one answer.

Ladder (reasoning escalation, not size):
  easy:   choose the correct order for the four steps of a compass-and-ruler construction. Each step uses what the one before
          drew, so the order is unique.
  medium: the same for five-step constructions where the steps carry measurements (a triangle from three sides,
          a triangle from two sides and the angle between them, a regular hexagon in a circle, a bisected angle).
  hard:   two triangles are known to match in two facts; which one extra fact completes the named test (SAS
          or RHS)? The tempting wrong facts give two sides and an angle that is not between them, or complete a
          different test. (ASA is left out: once two angles are known the third follows, so more than one extra side
          would complete it.)
  exam:   a four-part chain on a right-angled triangle: its third side by Pythagoras' theorem (2 marks), the extra
          fact that makes a second triangle match it by the RHS test (2), the perimeter of the second triangle
          (follows through), and the steps for constructing it in order (3).

Congruence tests are checked by running the test on the set of facts, so exactly one offered fact completes the
named test.
"""

from __future__ import annotations

import math

from ..core import check, generator
from ._fr2e_kit import Redraw, draw, exact_number_part, letter_order, mcq_part, single, words_part

E_LISTED = "fr2e-spatial-steps-in-listed-order"
E_REVERSED = "fr2e-spatial-steps-in-reverse-order"
E_SWAPPED = "fr2e-spatial-two-steps-swapped"
E_SSA = "fr2e-spatial-angle-not-between-the-sides"
E_OTHER_TEST = "fr2e-spatial-fact-completes-a-different-test"
E_ANGLE_FOR_SIDE = "fr2e-spatial-angle-where-side-needed"
E_SIDE_FOR_ANGLE = "fr2e-spatial-side-where-angle-needed"
E_LEG_FOR_HYP = "fr2e-spatial-leg-given-for-hypotenuse"
E_SIDE_NOT_NEXT = "fr2e-spatial-side-not-next-to-the-angle"

# -- constructions ---------------------------------------------------------------------------------------------------


_PAIRS = (("A", "B"), ("M", "N"), ("X", "Y"), ("S", "T"), ("G", "H"))
_TRIPLES = (("P", "Q", "R"), ("A", "B", "C"), ("K", "L", "M"), ("X", "Y", "Z"), ("D", "E", "F"))
_QUADS = (("P", "X", "Y", "Z"), ("M", "S", "T", "U"), ("H", "J", "K", "L"), ("Q", "C", "D", "E"), ("V", "F", "G", "W"))


def _perp_bisector(r):
    L = r.choice((6, 7, 8, 9, 10))
    A, B = r.choice(_PAIRS)
    return "construct the perpendicular bisector of a segment", [
        f"Draw the segment {A}{B}, {L} cm long.",
        f"Open the compass to more than half the length of {A}{B}.",
        f"Put the compass point on {A} and draw an arc above and below {A}{B}.",
        f"Keeping the same width, put the compass point on {B} and draw arcs that cross the first two arcs.",
        "Join the two crossing points with a ruler.",
    ]


def _bisect_angle(r):
    t = r.choice((40, 50, 60, 70, 80, 90, 100, 110, 120))
    A, B, C = r.choice(_TRIPLES)
    return "bisect an angle", [
        f"Draw rays {B}{A} and {B}{C} so that angle {A}{B}{C} is {t}°.",
        f"Put the compass point on {B} and draw an arc that crosses both rays.",
        "Keeping the same width, put the compass point on each crossing in turn and draw an arc inside the angle, so the two arcs cross.",
        f"Draw a ray from {B} through the point where the two arcs cross.",
        f"Check with a protractor that each half of the angle measures {t // 2}°.",
    ]


def _triangle_sss(r):
    a, b, c = sorted(r.sample(range(4, 11), 3))
    if a + b <= c:
        raise Redraw("not a triangle")
    P, Q, R = r.choice(_TRIPLES)
    return "construct a triangle from its three side lengths", [
        f"Draw the side {P}{Q}, {c} cm long.",
        f"Open the compass to {a} cm, put the point on {P} and draw an arc.",
        f"Open the compass to {b} cm, put the point on {Q} and draw an arc that crosses the first arc at {R}.",
        f"Join {P} to {R} and {Q} to {R}.",
    ]


def _perp_from_point(r):
    P, X, Y, Z = r.choice(_QUADS)
    return "construct the perpendicular from a point to a line", [
        f"Put the compass point on {P} and draw an arc that crosses the line at two points, {X} and {Y}.",
        f"Put the compass point on {X} and draw an arc on the other side of the line from {P}.",
        f"Keeping the same width, put the compass point on {Y} and draw an arc that crosses the previous arc at {Z}.",
        f"Join {P} to {Z} with a ruler.",
    ]


def _hexagon(r):
    R = r.choice((3, 4, 5, 6))
    O, A, B = r.choice((("O", "A", "B"), ("C", "P", "Q"), ("M", "X", "Y")))
    return "construct a regular hexagon", [
        f"Draw a circle of radius {R} cm with centre {O}.",
        f"Mark a point {A} on the circle.",
        f"Keep the compass at {R} cm, put the point on {A} and mark point {B} on the circle.",
        f"Put the compass point on {B} and mark the next point, repeating until six points are marked.",
        "Join the six points in order.",
    ]


def _sixty(r):
    P, Q, R = r.choice(_TRIPLES)
    return "construct an angle of 60°", [
        f"Draw a line and mark a point {P} on it.",
        f"Put the compass point on {P} and draw an arc that crosses the line at {Q}.",
        f"Keeping the same width, put the compass point on {Q} and draw an arc that crosses the first arc at {R}.",
        f"Draw the ray from {P} through {R}, so that angle {Q}{P}{R} is 60°.",
    ]


def _triangle_sas(r):
    a, b = r.sample(range(4, 11), 2)
    t = r.choice((35, 40, 50, 55, 65, 70, 100, 110))
    P, Q, R = r.choice(_TRIPLES)
    return "construct a triangle from two sides and the angle between them", [
        f"Draw the side {P}{Q}, {a} cm long.",
        f"Put a protractor at {P} and mark an angle of {t}° from {P}{Q}.",
        f"Draw a ray from {P} through the mark.",
        f"Measure {b} cm along the ray from {P} and mark the point {R}.",
        f"Join {Q} to {R}.",
    ]


_EASY_BUILDERS = (_triangle_sss, _perp_from_point, _sixty)
_MEDIUM_BUILDERS = (_perp_bisector, _hexagon, _triangle_sas, _bisect_angle)


def _order_question(rng, builders, marks=1):
    title, steps = rng.choice(builders)(rng)
    lines, key = letter_order(rng, steps)
    wrong = [(sorted(key), E_LISTED), (list(reversed(key)), E_REVERSED)]
    for i in range(len(key) - 1):
        k = list(key)
        k[i], k[i + 1] = k[i + 1], k[i]
        wrong.append((k, E_SWAPPED))
    wrong = [(w, t) for w, t in wrong if w != key]
    stem = f"These steps to {title} have been shuffled: {' '.join(lines)}"
    part = mcq_part(
        "Which order of the steps is correct?",
        ", ".join(key),
        [(", ".join(w), t) for w, t in wrong],
        marks,
        ["Each step uses what the step before it drew: " + " then ".join(f"({c})" for c in key) + "."],
    )
    return single(stem, part)


def _easy(rng):
    return _order_question(rng, _EASY_BUILDERS)


def _medium(rng):
    return _order_question(rng, _MEDIUM_BUILDERS)


# -- congruence tests ------------------------------------------------------------------------------------------------

_LABELS = (("ABC", "DEF"), ("PQR", "STU"), ("KLM", "XYZ"), ("ABC", "PQR"))


class Facts:
    """Facts that two triangles share, as the set of corresponding parts known to be equal.

    Sides are named by their end points by index (0-1, 1-2, 2-0), angles by the index of their vertex. 'R' marks
    that both triangles have a right angle at vertex 1."""

    def __init__(self, names: tuple[str, str]):
        self.n1, self.n2 = names

    def side(self, i: int) -> str:
        return f"{self.n1[i]}{self.n1[(i + 1) % 3]} = {self.n2[i]}{self.n2[(i + 1) % 3]}"

    def angle(self, i: int) -> str:
        return f"∠{self.n1[i]} = ∠{self.n2[i]}"


def _sides_adjacent(v: int) -> tuple[int, int]:
    """Side indices (side i joins vertex i to i + 1) that meet at vertex v."""
    return (v - 1) % 3, v


def applies(test: str, facts: set) -> bool:
    """Whether the facts, as a set of ("s", i) sides, ("a", i) angles and ("r",) right angle at vertex 1, are enough."""
    sides = {f[1] for f in facts if f[0] == "s"}
    angles = {f[1] for f in facts if f[0] == "a"}
    if test == "SSS":
        return sides == {0, 1, 2} or {0, 1, 2} <= sides
    if test == "SAS":
        return any(set(_sides_adjacent(v)) <= sides and v in angles for v in range(3))
    if test == "ASA":
        # two angles and the side between them: the side joining vertices u and w is side u when w = u + 1
        return any(u in angles and (u + 1) % 3 in angles and u in sides for u in range(3))
    if test == "RHS":
        # right angle at vertex 1, hypotenuse is the side opposite it: side 2 (joins vertex 2 to vertex 0)
        return ("r",) in facts and 2 in sides and bool(sides & {0, 1})
    raise ValueError(test)


_TEST_NAMES = {
    "SSS": "SSS (three pairs of equal sides)",
    "SAS": "SAS (two pairs of equal sides and the angle between them)",
    "ASA": "ASA (two pairs of equal angles and the side between them)",
    "RHS": "RHS (a right angle, the hypotenuse and one other side)",
}

_COMPLETE = {
    "SSS": [("s", 0), ("s", 1), ("s", 2)],
    "RHS": [("r",), ("s", 2), ("s", 0)],
}


def _complete_set(test: str, rng):
    if test == "SAS":
        v = rng.randrange(3)
        return [("a", v)] + [("s", i) for i in _sides_adjacent(v)]
    if test == "ASA":
        u = rng.randrange(3)
        return [("a", u), ("a", (u + 1) % 3), ("s", u)]
    if test == "RHS":
        leg = rng.choice((0, 1))
        return [("r",), ("s", 2), ("s", leg)]
    return list(_COMPLETE[test])


def _fact_text(f: Facts, fact) -> str:
    return f.side(fact[1]) if fact[0] == "s" else f.angle(fact[1])


def _all_facts():
    return [("s", 0), ("s", 1), ("s", 2), ("a", 0), ("a", 1), ("a", 2)]


def _tag(test, given, extra) -> str:
    """The error that choosing `extra` stands for."""
    trial = set(given) | {extra}
    if any(applies(t, trial) for t in ("SSS", "SAS", "ASA", "RHS") if t != test):
        return E_OTHER_TEST
    if test == "RHS":
        return E_LEG_FOR_HYP if extra[0] == "s" else E_ANGLE_FOR_SIDE
    if test == "SAS":
        return E_SSA if extra[0] == "a" else E_SIDE_NOT_NEXT
    return E_ANGLE_FOR_SIDE


def _hard(rng):
    test = rng.choice(("SAS", "RHS"))
    f = Facts(rng.choice(_LABELS))
    complete = _complete_set(test, rng)
    check(applies(test, set(complete)), "the complete set must satisfy the test")
    if test == "RHS":
        missing = ("s", 2)  # the hypotenuse is the missing fact, so no second leg can also complete the test
    else:
        missing = rng.choice([c for c in complete])
    given = [c for c in complete if c != missing]
    given_set = set(given)
    check(not applies(test, given_set), "the given facts must not already satisfy the test")
    already = (
        {("a", 1)} if ("r",) in given_set else set()
    )  # a stated right angle already gives the equal angles at vertex 1
    candidates = [c for c in _all_facts() if c not in given_set and c != missing and c not in already]
    right = [c for c in _all_facts() if c not in given_set and applies(test, given_set | {c})]
    check(right == [missing], "exactly one extra fact completes the test")
    wrong = rng.sample(candidates, 3)
    check(all(not applies(test, given_set | {w}) for w in wrong), "no wrong fact may complete the test")
    names = f.n1 + " and " + f.n2
    given_text = " and ".join(
        ("both have a right angle at " + f.n1[1] + " and " + f.n2[1]) if g == ("r",) else _fact_text(f, g)
        for g in given
    )
    if ("r",) in given:
        others = [g for g in given if g != ("r",)]
        given_text = f"each has a right angle (at {f.n1[1]} and {f.n2[1]})" + "".join(
            f" and {_fact_text(f, g)}" for g in others
        )
    stem = (
        f"Triangles {f.n1} and {f.n2} are being compared. It is known that {given_text}. "
        f"Which one extra fact would let you show the triangles are congruent by the {test} test?"
    )
    key = _fact_text(f, missing)
    mc = mcq_part(
        "Choose the extra fact.",
        key,
        [(_fact_text(f, w), _tag(test, given_set, w)) for w in wrong],
        1,
        [
            f"The {_TEST_NAMES[test]} test needs: "
            + ", ".join(_fact_text(f, c) if c != ("r",) else "a right angle in each" for c in complete)
            + "."
        ],
    )
    out = single(stem, mc)
    out["text"] = stem
    return out


# -- exam ------------------------------------------------------------------------------------------------------------


def _exam(rng):
    k = rng.choice((1, 2, 3, 4))
    leg1, leg2 = 3 * k, 4 * k
    hyp = 5 * k
    check(hyp**2 == leg1**2 + leg2**2, "the triangle must be right-angled")
    n1, n2 = rng.choice(_LABELS)
    f = Facts((n1, n2))
    flip = rng.random() < 0.5
    known_leg, other_leg = (0, 1) if not flip else (1, 0)
    known_len = (leg1, leg2)[known_leg]
    a_, b_, c_ = n1
    d_, e_, f_ = n2
    hyp_name, hyp_name2 = f"{c_}{a_}", f"{f_}{d_}"
    leg_names = (f"{a_}{b_}", f"{b_}{c_}")
    leg_names2 = (f"{d_}{e_}", f"{e_}{f_}")
    stem = (
        f"Triangle {n1} has a right angle at {b_}, {leg_names[0]} = {leg1} cm and {leg_names[1]} = {leg2} cm. "
        f"Triangle {n2} has a right angle at {e_} and {leg_names2[known_leg]} = {leg_names[known_leg]}. "
        f"A student wants to show that triangle {n2} is congruent to triangle {n1}."
    )
    part_a = exact_number_part(
        f"Find the length of {hyp_name}, in centimetres.",
        hyp,
        2,
        [f"{hyp_name}² = {leg1}² + {leg2}² = {leg1**2 + leg2**2}.", f"{hyp_name} = {hyp}."],
        [
            (leg1 + leg2, "fr2e-spatial-legs-added-for-hypotenuse"),
            (leg1**2 + leg2**2, "fr2e-spatial-square-not-rooted"),
            (leg2**2 - leg1**2, "fr2e-spatial-pythagoras-subtracted"),
            (max(leg1, leg2), "fr2e-spatial-longer-leg-for-hypotenuse"),
        ],
        suffix="",
        sensible_range=(0, None),
    )
    given = {("r",), ("s", known_leg)}
    check(not applies("RHS", given) and applies("RHS", given | {("s", 2)}), "only the hypotenuse completes RHS")
    wrong_facts = [("s", other_leg), ("a", 0), ("a", 2)]
    check(all(not applies("RHS", given | {w}) for w in wrong_facts), "no other offered fact completes RHS")
    part_b = mcq_part(
        "Which extra fact would let the student show the triangles are congruent by the RHS test?",
        f"{hyp_name} = {hyp_name2}",
        [(f.side(w[1]) if w[0] == "s" else f.angle(w[1]), _tag("RHS", given, w)) for w in wrong_facts],
        2,
        [
            "RHS needs a right angle, the hypotenuse and one other side. The right angle and "
            f"{leg_names[known_leg]} = {leg_names2[known_leg]} are known, so the hypotenuse is needed."
        ],
    )
    perimeter = leg1 + leg2 + hyp
    part_c = exact_number_part(
        f"The triangles are congruent. Find the perimeter of triangle {n2}, in centimetres.",
        perimeter,
        1,
        [f"{leg_names2[0]} = {leg1}, {leg_names2[1]} = {leg2}, {hyp_name2} = {hyp}, so the perimeter is {perimeter}."],
        [
            (leg1 + leg2, "fr2e-spatial-hypotenuse-left-out-of-perimeter"),
            (hyp, "fr2e-spatial-one-side-for-perimeter"),
            (leg1 * leg2 // 2, "fr2e-spatial-area-for-perimeter"),
            (perimeter + hyp, "fr2e-spatial-hypotenuse-counted-twice"),
        ],
        suffix="",
        follow_through=f"{leg1} + {leg2} + part_a",
        sensible_range=(0, None),
    )
    steps = [
        f"Draw the side {leg_names2[known_leg]}, {known_len} cm long.",
        f"At {e_}, construct a perpendicular to {leg_names2[known_leg]}.",
        f"Open the compass to the length found in part (a), put the point on {d_ if known_leg == 0 else f_} and draw an arc that crosses the perpendicular at {f_ if known_leg == 0 else d_}.",
        f"Join {d_ if known_leg == 0 else f_} to {f_ if known_leg == 0 else d_}.",
    ]
    lines, key = letter_order(rng, steps)
    wrong = [(sorted(key), E_LISTED), (list(reversed(key)), E_REVERSED)]
    for i in range(3):
        kk = list(key)
        kk[i], kk[i + 1] = kk[i + 1], kk[i]
        wrong.append((kk, E_SWAPPED))
    wrong = [(w, t) for w, t in wrong if w != key]
    part_d = words_part(
        f"Triangle {n2} is to be constructed. These steps are shuffled: {' '.join(lines)} Give the letters of the steps in the order they should be done, separated by commas.",
        key,
        3,
        ["Draw the known leg, make the right angle at its end, then use the hypotenuse to find the last vertex."],
        wrong,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator(
    "fr2e_spatial_algorithm",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2e_spatial_algorithm(rng, tier: str = "easy"):
    """Order the steps of a construction, or choose the extra fact that completes a congruence test."""
    build = {"easy": _easy, "medium": _medium, "hard": _hard}.get(tier)
    check(build is not None, f"unknown tier {tier!r}")
    return draw(build, rng)


@generator("exd_x_fr2e_spatial_algorithm", level=8, presets={"exam": dict(tier="exam")})
def exd_x_fr2e_spatial_algorithm(rng, tier: str = "exam"):
    """A four-part chain on a right-angled triangle: its hypotenuse, the RHS test, the perimeter, the construction."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return draw(_exam, rng)
