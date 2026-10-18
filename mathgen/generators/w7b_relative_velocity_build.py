"""Generates `w7b_relative_velocity` questions (easy, medium, hard) and the exam chain `exd_x_w7b_relative_velocity`.

``relative-velocity-vectors`` (``vectors``, answer ``number``): the velocity of one moving object relative to another,
from components or from speeds on perpendicular bearings, and the closest-approach and interception set-ups that rest on it.

* easy: one component of the velocity of A relative to B (or of B relative to A) from two velocities in i, j form; the
  closing speed or the catch-up time of two vehicles on a straight road; or the reverse, one component of an object's
  own velocity from its velocity relative to the other.
* medium: the speed of A relative to B for motion at right angles (ships on cardinal bearings, a swimmer in a river, an
  aircraft in a crosswind) or from components; the coordinate of A's position relative to B after some hours.
* hard: the time of closest approach of two ships, the least distance between them, the time a chaser of known speed
  needs to intercept a moving target, or the component of the chaser's velocity.
* exam (``exd_x_w7b_relative_velocity``): a four-part closest-approach chain: relative speed, time of closest approach,
  least distance, and the time the ships first come within a stated range.

Every figure is built backwards from a chosen relative velocity (a Pythagorean multiple, so every speed and distance is
exact), a chosen time and a chosen perpendicular offset, and re-derived a second way: the time of closest approach from
the vertex of the parabola through three samples of the squared distance, the interception time from the quadratic
formula with an exact integer square root.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from ..core import attempts, check, generator, join_terms, paren_negative, phrase
from ._w6a_kit import enough, num_dist, num_fields, part, retry_draw

TOPIC = "w7b_relative_velocity"
EXAM_TOPIC = "exd_x_w7b_relative_velocity"

AREA = "relative-velocity"
MISCONCEPTIONS = [
    (
        "relvel-wrong-order",
        "Subtracts the velocities in the wrong order, giving the velocity of B relative to A where A relative to B was asked, or the reverse.",
        AREA,
    ),
    ("relvel-adds-velocities", "Adds the two velocities instead of subtracting to find a relative velocity.", AREA),
    (
        "relvel-ignores-reference",
        "Gives the velocity of the moving object itself, forgetting to subtract the velocity of the object it is measured against.",
        AREA,
    ),
    (
        "relvel-swaps-components",
        "Gives the other component of the relative velocity: the j-component where the i-component was asked, or the reverse.",
        AREA,
    ),
    (
        "relvel-1d-direction",
        "Adds the speeds of two vehicles moving the same way, or subtracts them when they move in opposite directions.",
        AREA,
    ),
    (
        "relvel-rearranges-wrongly",
        "Rearranges the relative velocity equation with a sign slip, so the unknown velocity has the wrong sign or the relative velocity is added where it should be subtracted.",
        AREA,
    ),
    ("relvel-catch-up-gives-speed", "Gives the closing speed where the time to catch up was asked.", AREA),
    ("relvel-catch-up-gives-gap", "Gives the starting gap where the time to catch up was asked.", AREA),
    (
        "relvel-adds-components",
        "Adds the sizes of the two components instead of using Pythagoras to find a speed.",
        AREA,
    ),
    (
        "relvel-forgets-root",
        "Stops at the sum of the squares of the components and does not take the square root.",
        AREA,
    ),
    ("relvel-takes-larger-component", "Takes the larger component as the speed and ignores the other.", AREA),
    (
        "relvel-adds-perpendicular-speeds",
        "Adds the two speeds for motion at right angles instead of combining them with Pythagoras.",
        AREA,
    ),
    (
        "relvel-subtracts-perpendicular-speeds",
        "Subtracts the two speeds for motion at right angles instead of combining them with Pythagoras.",
        AREA,
    ),
    (
        "relvel-position-ignores-start",
        "Finds only how far the relative position has changed and forgets the starting relative position.",
        AREA,
    ),
    (
        "relvel-position-uses-one-velocity",
        "Moves the relative position using one object's velocity instead of the relative velocity.",
        AREA,
    ),
    ("relvel-time-added", "Adds the time to the coordinate instead of multiplying the velocity component by it.", AREA),
    ("approach-sign-error", "Drops the minus sign in the time of closest approach and gives a negative time.", AREA),
    (
        "approach-divides-by-speed",
        "Divides the dot product of the relative position and velocity by the relative speed instead of by its square.",
        AREA,
    ),
    (
        "approach-equal-coordinate",
        "Takes the time when one coordinate of the two positions agrees as the time of closest approach.",
        AREA,
    ),
    ("approach-gives-square", "Gives the square of the least distance between the objects.", AREA),
    (
        "approach-distance-travelled",
        "Gives how far the relative position moves before the moment of closest approach instead of the least distance.",
        AREA,
    ),
    (
        "approach-adds-components",
        "Adds the sizes of the components of the closest separation instead of using Pythagoras.",
        AREA,
    ),
    (
        "approach-first-within-uses-closest",
        "Gives the time of closest approach where the time they first come within the stated range was asked.",
        AREA,
    ),
    (
        "approach-last-within",
        "Adds the half-duration to the time of closest approach and gives the time they leave the range.",
        AREA,
    ),
    (
        "approach-half-width",
        "Gives the half-duration spent within range instead of the time they first come within it.",
        AREA,
    ),
    (
        "intercept-ignores-motion",
        "Divides the starting separation by the chaser's speed as if the target were standing still.",
        AREA,
    ),
    (
        "intercept-gives-distance",
        "Gives the starting separation of the two boats where the time to intercept was asked.",
        AREA,
    ),
    (
        "intercept-gives-chaser-distance",
        "Gives how far the chaser travels where the time to intercept was asked.",
        AREA,
    ),
    (
        "intercept-negative-root",
        "Reports the negative root of the interception quadratic, a time before the chase began.",
        AREA,
    ),
    (
        "intercept-copies-target-velocity",
        "Gives the target's velocity component as the chaser's.",
        AREA,
    ),
    ("intercept-sign-flipped", "Gives the chaser's velocity component with the wrong sign.", AREA),
    (
        "intercept-relative-component",
        "Gives the component of the chaser's velocity relative to the target instead of its own velocity.",
        AREA,
    ),
]

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29))
_VEHICLES = (("Ship", "km/h", "km"), ("Aircraft", "km/h", "km"), ("Boat", "km/h", "km"), ("Drone", "m/s", "m"))


# -- helpers ------------------------------------------------------------------------------------------------------------


def _hours(t) -> str:
    """``1 hour`` or ``<t> hours``."""
    return "1 hour" if t == 1 else f"{t} hours"


def _vec(x: int, y: int) -> str:
    return join_terms([(x, "i"), (y, "j")])


def _pt(x: int, y: int) -> str:
    return f"({x}, {y})"


def _dist(correct, candidates, n: int = 4):
    """`num_dist`, with candidates of the answer's own shape (whole number or fraction) tried first."""
    whole = Fraction(correct).denominator == 1
    ordered = sorted(
        (c for c in candidates if c[0] is not None),
        key=lambda c: Fraction(c[0]).denominator == 1 and not whole or Fraction(c[0]).denominator != 1 and whole,
    )
    return num_dist(correct, ordered, n)


def _signed_pair(rng, a: int, b: int):
    """(±a, ±b), the two in either order."""
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    return rng.choice((-1, 1)) * x, rng.choice((-1, 1)) * y


def _answer(value) -> dict:
    return num_fields(value)


def _craft(rng):
    noun, unit, length = rng.choice(_VEHICLES)
    return noun, unit, length


# -- easy ---------------------------------------------------------------------------------------------------------------


def _easy_component(rng):
    noun, unit, _ = _craft(rng)
    for _ in attempts(500, "two velocities with a non-zero relative component"):
        va = (rng.randint(-9, 9), rng.randint(-9, 9))
        vb = (rng.randint(-9, 9), rng.randint(-9, 9))
        rel = (va[0] - vb[0], va[1] - vb[1])
        if rel[0] and rel[1] and va != vb and va[0] != 0 and va[1] != 0 and vb[0] != 0 and vb[1] != 0:
            break
    a_rel_b = rng.random() < 0.6
    comp = rng.choice((0, 1))
    first, second = ("A", "B") if a_rel_b else ("B", "A")
    mover, base = (va, vb) if a_rel_b else (vb, va)
    value = mover[comp] - base[comp]
    # second route: positions after two hours, relative displacement over those two hours.
    pos_mover = [2 * mover[0], 2 * mover[1]]
    pos_base = [2 * base[0], 2 * base[1]]
    check(Fraction(pos_mover[comp] - pos_base[comp], 2) == value, "relative component differs")
    other = mover[1 - comp] - base[1 - comp]
    cands = [
        (-value, "relvel-wrong-order"),
        (va[comp] + vb[comp], "relvel-adds-velocities"),
        (mover[comp], "relvel-ignores-reference"),
        (other, "relvel-swaps-components"),
    ]
    dist = _dist(value, cands)
    enough(dist)
    axis = "i" if comp == 0 else "j"
    stem = phrase(
        rng,
        (
            "{n} A has velocity {va} {u} and {n2} B has velocity {vb} {u}, where i points east and j points north. "
            "What is the {ax}-component of the velocity of {f} relative to {s}, in {u}?",
            "Relative to the ground, {n2} A moves with velocity {va} {u} and {n2} B with velocity {vb} {u} (i east, "
            "j north). Find the {ax}-component of the velocity of {f} relative to {s}, in {u}.",
        ),
        n=noun,
        n2=noun.lower() if noun != "Aircraft" else "aircraft",
        va=_vec(*va),
        vb=_vec(*vb),
        u=unit,
        ax=axis,
        f=first,
        s=second,
    )
    stem = stem.replace(f"{noun.lower()} A", f"{noun} A").replace(f"{noun.lower()} B", f"{noun} B")
    working = [
        f"The velocity of {first} relative to {second} is v{first} - v{second}.",
        f"Its {axis}-component is {mover[comp]} - ({base[comp]}) = {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _easy_straight(rng):
    noun, unit, _ = ("Car", "km/h", "km")
    same = rng.random() < 0.5
    catch_up = same and rng.random() < 0.5
    for _ in attempts(500, "two road speeds"):
        va, vb = rng.randint(40, 120), rng.randint(40, 120)
        if abs(va - vb) >= 5 and (not catch_up or va > vb):
            break
    rel = abs(va - vb) if same else va + vb
    way = "in the same direction" if same else "in opposite directions"
    if catch_up:
        hours = rng.randint(2, 9)
        gap = hours * (va - vb)
        check(Fraction(gap, va - vb) == hours, "catch-up time differs")
        cands = [
            (Fraction(gap, va + vb), "relvel-1d-direction"),
            (Fraction(gap, va), "relvel-ignores-reference"),
            (Fraction(gap, vb), "relvel-ignores-reference"),
            (va - vb, "relvel-catch-up-gives-speed"),
            (gap, "relvel-catch-up-gives-gap"),
        ]
        dist = _dist(hours, cands)
        enough(dist)
        stem = phrase(
            rng,
            (
                "Car A is {g} km behind car B on a straight road. Car A travels at {va} km/h and car B at {vb} km/h, "
                "both in the same direction. After how many hours does car A catch car B?",
                "On a straight road car B is {g} km ahead of car A. Both head the same way, A at {va} km/h and B at "
                "{vb} km/h. How many hours does A take to draw level with B?",
            ),
            g=gap,
            va=va,
            vb=vb,
        )
        working = [f"A closes the gap at {va} - {vb} = {va - vb} km/h.", f"{gap} / {va - vb} = {hours} hours."]
        return dict(text=stem, working=working, distractors=dist, **_answer(hours), answer_kind="number")
    wrong = va + vb if same else abs(va - vb)
    cands = [
        (wrong, "relvel-1d-direction"),
        (va, "relvel-ignores-reference"),
        (vb, "relvel-ignores-reference"),
        (rel + 10, "relvel-adds-velocities"),
    ]
    dist = _dist(rel, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "Two cars travel along a straight road {way}. Car A travels at {va} km/h and car B at {vb} km/h. "
            "What is the speed of A relative to B, in km/h?",
            "On a straight road car A moves at {va} km/h and car B at {vb} km/h, {way}. How fast does A appear to be "
            "moving to a passenger in B, in km/h?",
        ),
        way=way,
        va=va,
        vb=vb,
    )
    working = [
        f"Moving {way}, the relative speed is {'|' + str(va) + ' - ' + str(vb) + '|' if same else str(va) + ' + ' + str(vb)}.",
        f"So the relative speed is {rel} km/h.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(rel), answer_kind="number")


def _easy_reverse(rng):
    noun, unit, _ = _craft(rng)
    for _ in attempts(500, "a reverse relative-velocity draw"):
        known = (rng.randint(-9, 9), rng.randint(-9, 9))
        rel = (rng.randint(-9, 9), rng.randint(-9, 9))
        if all(known) and all(rel):
            break
    comp = rng.choice((0, 1))
    find_a = rng.random() < 0.5  # find vA from vB and vAB, else vB from vA and vAB
    if find_a:
        value = rel[comp] + known[comp]  # vA = vAB + vB
        cands = [
            (known[comp] - rel[comp], "relvel-rearranges-wrongly"),
            (rel[comp] - known[comp], "relvel-rearranges-wrongly"),
            (rel[comp], "relvel-ignores-reference"),
            (rel[1 - comp] + known[1 - comp], "relvel-swaps-components"),
        ]
        given = f"{noun} B has velocity {_vec(*known)} {unit}, and {noun} A moves with velocity {_vec(*rel)} {unit} relative to it"
        ask = f"Find the {'i' if comp == 0 else 'j'}-component of the velocity of {noun} A relative to the ground"
        steps = f"vA = (vA - vB) + vB, so the component is {rel[comp]} + ({known[comp]}) = {value}."
        # second route: vA - vB must give back the stated relative velocity
        check(value - known[comp] == rel[comp], "reverse key differs")
    else:
        value = known[comp] - rel[comp]  # vB = vA - vAB
        cands = [
            (known[comp] + rel[comp], "relvel-rearranges-wrongly"),
            (rel[comp] - known[comp], "relvel-rearranges-wrongly"),
            (rel[comp], "relvel-ignores-reference"),
            (known[1 - comp] - rel[1 - comp], "relvel-swaps-components"),
        ]
        given = f"{noun} A has velocity {_vec(*known)} {unit}, and moves with velocity {_vec(*rel)} {unit} relative to {noun} B"
        ask = f"Find the {'i' if comp == 0 else 'j'}-component of the velocity of {noun} B relative to the ground"
        steps = f"vB = vA - (vA - vB), so the component is {known[comp]} - ({rel[comp]}) = {value}."
        check(known[comp] - value == rel[comp], "reverse key differs")
    dist = _dist(value, cands)
    enough(dist)
    stem = f"With i east and j north, {given}. {ask}, in {unit}."
    return dict(text=stem, working=[steps], distractors=dist, **_answer(value), answer_kind="number")


# -- medium -------------------------------------------------------------------------------------------------------------


def _pyth_pair(rng, limit: int = 60):
    for _ in attempts(200, "a Pythagorean pair"):
        a, b, c = rng.choice(_TRIPLES)
        k = rng.randint(1, max(1, limit // c))
        if k * c <= limit:
            u, v = k * a, k * b
            if rng.random() < 0.5:
                u, v = v, u
            return u, v, k * c
    raise AssertionError  # pragma: no cover


_BEARING = {(90, 0): "east", (270, 0): "west", (90, 180): "east", (270, 180): "west"}


def _perp_cands(u, v, w):
    return [
        (u + v, "relvel-adds-perpendicular-speeds"),
        (abs(u - v), "relvel-subtracts-perpendicular-speeds"),
        (u * u + v * v, "relvel-forgets-root"),
        (max(u, v), "relvel-takes-larger-component"),
    ]


def _medium_perpendicular(rng):
    u, v, w = _pyth_pair(rng, 60)
    check(u * u + v * v == w * w, "not a Pythagorean pair")
    dist = _dist(w, _perp_cands(u, v, w))
    enough(dist)
    kind = rng.choice(("ships", "river", "wind"))
    if kind == "ships":
        ba = rng.choice((90, 270))
        bb = rng.choice((0, 180))
        if rng.random() < 0.5:
            ba, bb = bb, ba
        stem = phrase(
            rng,
            (
                "Ship A sails on a bearing of {ba:03d} at {u} km/h and ship B sails on a bearing of {bb:03d} at {v} km/h. "
                "What is the speed of A relative to B, in km/h?",
                "Ship A steams at {u} km/h on a bearing of {ba:03d}, and ship B at {v} km/h on a bearing of {bb:03d}. "
                "Find how fast ship A is moving relative to ship B, in km/h.",
            ),
            ba=ba,
            bb=bb,
            u=u,
            v=v,
        )
        working = [
            f"Bearings {ba:03d} and {bb:03d} are at right angles, so the relative velocity has components {u} and {v}.",
            f"Speed = sqrt({paren_negative(u)}^2 + {paren_negative(v)}^2) = {w} km/h.",
        ]
    elif kind == "river":
        stem = phrase(
            rng,
            (
                "A swimmer heads straight across a river at {u} m/s relative to the water. The river flows at {v} m/s "
                "parallel to its banks. What is the swimmer's speed relative to the bank, in m/s?",
                "A boat points directly across a river and moves at {u} m/s through the water. The current runs at {v} m/s "
                "along the banks. Find the boat's speed relative to the bank, in m/s.",
            ),
            u=u,
            v=v,
        )
        working = [
            "Velocity relative to the bank = velocity relative to the water + velocity of the water.",
            f"The two are at right angles, so the speed is sqrt({paren_negative(u)}^2 + {paren_negative(v)}^2) = {w} m/s.",
        ]
    else:
        stem = phrase(
            rng,
            (
                "An aircraft heads due north with an airspeed of {u} km/h. A wind blows from the west at {v} km/h. "
                "What is the speed of the aircraft relative to the ground, in km/h?",
                "A pilot points the aircraft due north, flying at {u} km/h through the air, while the air moves due east at "
                "{v} km/h. Find the aircraft's ground speed, in km/h.",
            ),
            u=u,
            v=v,
        )
        working = [
            "Ground velocity = velocity through the air + velocity of the air.",
            f"These are at right angles, so the ground speed is sqrt({paren_negative(u)}^2 + {paren_negative(v)}^2) = {w} km/h.",
        ]
    return dict(text=stem, working=working, distractors=dist, **_answer(w), answer_kind="number")


def _medium_components(rng):
    noun, unit, _ = _craft(rng)
    a, b, c = rng.choice(_TRIPLES[:3])
    k = rng.randint(1, min(3, 15 // max(a, b)))
    rel = _signed_pair(rng, k * a, k * b)
    for _ in attempts(500, "a second velocity"):
        vb = (rng.randint(-12, 12), rng.randint(-12, 12))
        va = (rel[0] + vb[0], rel[1] + vb[1])
        if va != vb and vb != (0, 0) and va != (0, 0) and abs(va[0]) <= 20 and abs(va[1]) <= 20:
            break
    w = k * c
    check((va[0] - vb[0]) ** 2 + (va[1] - vb[1]) ** 2 == w * w, "speed differs")
    dx, dy = va[0] - vb[0], va[1] - vb[1]
    cands = [
        (abs(dx) + abs(dy), "relvel-adds-components"),
        (dx * dx + dy * dy, "relvel-forgets-root"),
        (max(abs(dx), abs(dy)), "relvel-takes-larger-component"),
        (abs(va[0] + vb[0]) + abs(va[1] + vb[1]), "relvel-adds-velocities"),
    ]
    dist = _dist(w, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "{n} A has velocity {va} {u} and {n} B has velocity {vb} {u}, with i east and j north. What is the speed of A "
            "relative to B, in {u}?",
            "Relative to the ground, {n} A moves with velocity {va} {u} and {n} B with {vb} {u} (i east, j north). Find the "
            "speed of A relative to B, in {u}.",
        ),
        n=noun,
        va=_vec(*va),
        vb=_vec(*vb),
        u=unit,
    )
    working = [
        f"The velocity of A relative to B is ({_vec(*va)}) - ({_vec(*vb)}) = {_vec(dx, dy)}.",
        f"Its magnitude is sqrt({paren_negative(dx)}^2 + {paren_negative(dy)}^2) = {w} {unit}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(w), answer_kind="number")


def _medium_position(rng):
    noun, unit, length = _craft(rng)
    t = rng.randint(2, 8)
    for _ in attempts(500, "a position draw"):
        ra = (rng.randint(-30, 30), rng.randint(-30, 30))
        rb = (rng.randint(-30, 30), rng.randint(-30, 30))
        va = (rng.randint(-8, 8), rng.randint(-8, 8))
        vb = (rng.randint(-8, 8), rng.randint(-8, 8))
        comp = rng.choice((0, 1))
        dv = va[comp] - vb[comp]
        if dv and ra[comp] != rb[comp] and all(va) and all(vb):
            break
    r0 = ra[comp] - rb[comp]
    value = r0 + dv * t
    # second route: positions of each object at time t, subtracted.
    check((ra[comp] + va[comp] * t) - (rb[comp] + vb[comp] * t) == value, "relative position differs")
    cands = [
        (dv * t, "relvel-position-ignores-start"),
        (r0 + va[comp] * t, "relvel-position-uses-one-velocity"),
        (r0 + dv + t, "relvel-time-added"),
        (r0 - dv * t, "relvel-wrong-order"),
    ]
    dist = _dist(value, cands)
    enough(dist)
    axis = "x" if comp == 0 else "y"
    stem = (
        f"At time 0 {noun} A is at {_pt(*ra)} {length} and {noun} B is at {_pt(*rb)} {length}, with x measured east and y north. "
        f"A moves with constant velocity {_vec(*va)} {unit} and B with constant velocity {_vec(*vb)} {unit} (i east, j north). "
        f"After {_hours(t) if unit == 'km/h' else str(t) + ' seconds'}, what is the {axis}-coordinate of the position of A relative to B, in {length}?"
    )
    working = [
        f"The position of A relative to B starts at {r0} in the {axis}-direction and changes at {va[comp]} - ({vb[comp]}) = {dv} per unit time.",
        f"After {t}: {r0} + {paren_negative(dv)} x {t} = {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


# -- hard: closest approach ---------------------------------------------------------------------------------------------


def _vertex_time(r0, v) -> Fraction:
    """Time of least distance, from the vertex of the parabola through d^2 at t = 0, 1, 2 (no dot-product formula)."""

    def f(t):
        return (r0[0] + v[0] * t) ** 2 + (r0[1] + v[1] * t) ** 2

    f0, f1, f2 = f(0), f(1), f(2)
    qa = Fraction(f2 - 2 * f1 + f0, 2)
    qb = f1 - f0 - qa
    return -qb / (2 * qa)


def _approach(rng, triples=_TRIPLES[:2], k_max=3, j_max=7, t_lo=1, t_hi=14, fixed=None):
    """A closest-approach set-up: relative velocity k*(a, b) on a Pythagorean triple, offset j*(-b, a) at closest approach."""
    a, b, c = rng.choice(triples)
    if fixed:
        a, b, c = fixed["triple"]
    k = fixed["k"] if fixed else rng.randint(1, k_max if c <= 5 else 1)
    j = fixed["j"] if fixed else rng.randint(1, j_max)
    tstar = fixed["t"] if fixed else rng.randint(t_lo, t_hi if c <= 5 else min(t_hi, 7))
    sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
    swap = rng.random() < 0.5
    ua, ub = (b, a) if swap else (a, b)
    v = (sx * k * ua, sy * k * ub)  # relative velocity of A relative to B
    perp = (-v[1] // k * (1 if True else 1), v[0] // k)  # a lattice vector of length c, perpendicular to v
    side = rng.choice((-1, 1))
    w = (side * j * perp[0], side * j * perp[1])
    r0 = (w[0] - v[0] * tstar, w[1] - v[1] * tstar)  # position of A relative to B at t = 0
    dmin = j * c
    # second route: the vertex of d^2(t), and the least distance read back from it.
    check(_vertex_time(r0, v) == tstar, "closest-approach time differs")
    tv = _vertex_time(r0, v)
    check((r0[0] + v[0] * tv) ** 2 + (r0[1] + v[1] * tv) ** 2 == dmin * dmin, "least distance differs")
    check(v[0] * w[0] + v[1] * w[1] == 0, "offset not perpendicular")
    for _ in attempts(200, "a non-zero velocity for B"):
        vb = (rng.randint(-12, 12), rng.randint(-12, 12))
        va = (v[0] + vb[0], v[1] + vb[1])
        if vb != (0, 0) and va != (0, 0):
            break
    rb = (rng.randint(-20, 20), rng.randint(-20, 20))
    ra = (rb[0] + r0[0], rb[1] + r0[1])
    return dict(k=k, c=c, j=j, t=tstar, v=v, w=w, r0=r0, dmin=dmin, va=va, vb=vb, ra=ra, rb=rb, speed=k * c)


def _approach_text(s, noun, unit, length):
    return (
        f"At noon {noun} A is at {_pt(*s['ra'])} {length} and {noun} B is at {_pt(*s['rb'])} {length}, with x measured east and "
        f"y north of a harbour. A moves with constant velocity {_vec(*s['va'])} {unit} and B with constant velocity "
        f"{_vec(*s['vb'])} {unit} (i east, j north)."
    )


def _hard_approach_time(rng):
    s = _approach(rng)
    r0, v, t = s["r0"], s["v"], s["t"]
    cands = [
        (-t, "approach-sign-error"),
        (t * s["speed"], "approach-divides-by-speed"),
        (Fraction(-r0[0], v[0]) if v[0] else None, "approach-equal-coordinate"),
        (Fraction(-r0[1], v[1]) if v[1] else None, "approach-equal-coordinate"),
    ]
    dist = _dist(t, cands)
    enough(dist)
    noun, unit, length = ("Ship", "km/h", "km")
    stem = f"{_approach_text(s, noun, unit, length)} After how many hours from noon are the two ships closest together?"
    working = [
        f"The position of A relative to B is {_pt(*r0)} at noon and changes at {_vec(*v)} km/h.",
        "The distance is least when the relative position is perpendicular to the relative velocity.",
        f"That happens at t = {_hours(t)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(t), answer_kind="number")


def _hard_approach_distance(rng):
    s = _approach(rng)
    r0, v, t, w = s["r0"], s["v"], s["t"], s["w"]
    d = s["dmin"]
    cands = [
        (d * d, "approach-gives-square"),
        (t * s["speed"], "approach-distance-travelled"),
        (abs(w[0]) + abs(w[1]), "approach-adds-components"),
        (d + s["speed"], "approach-distance-travelled"),
    ]
    dist = _dist(d, cands)
    enough(dist)
    stem = f"{_approach_text(s, 'Ship', 'km/h', 'km')} What is the least distance between the two ships, in km?"
    working = [
        f"The relative position is {_pt(*r0)} + t({_vec(*v)}) and the distance is least at t = {t}.",
        f"There the relative position is {_pt(*w)}, so the least distance is {d} km.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(d), answer_kind="number")


# -- hard: interception -------------------------------------------------------------------------------------------------


def _quad_root(d0, vb, speed) -> list[Fraction]:
    """Roots t of |d0 + vb t| = speed * t, from the quadratic formula with an exact integer square root."""
    qa = vb[0] ** 2 + vb[1] ** 2 - speed**2
    qb = 2 * (d0[0] * vb[0] + d0[1] * vb[1])
    qc = d0[0] ** 2 + d0[1] ** 2
    disc = qb * qb - 4 * qa * qc
    root = isqrt(disc)
    check(root * root == disc, "interception discriminant not a square")
    return sorted([Fraction(-qb - root, 2 * qa), Fraction(-qb + root, 2 * qa)])


def _intercept(rng):
    for _ in attempts(2000, "an interception set-up"):
        a, b, c = rng.choice(_TRIPLES[:3])
        m = rng.randint(1, 3) if c <= 13 else 1
        speed = m * c
        va = _signed_pair(rng, m * a, m * b)
        a2, b2, c2 = rng.choice(_TRIPLES[:2])
        k = rng.randint(1, 2)
        rel = _signed_pair(rng, k * a2, k * b2)
        vb = (va[0] - rel[0], va[1] - rel[1])
        if vb[0] ** 2 + vb[1] ** 2 >= speed * speed or vb == (0, 0) or va == (0, 0):
            continue
        if abs(vb[0]) > 20 or abs(vb[1]) > 20:
            continue
        t = rng.randint(1, max(2, min(10, 80 // (k * c2))))
        d0 = (rel[0] * t, rel[1] * t)
        ra = (rng.randint(-15, 15), rng.randint(-15, 15))
        rb = (ra[0] + d0[0], ra[1] + d0[1])
        roots = _quad_root(d0, vb, speed)
        check(roots[1] == t and roots[0] < 0, "interception time differs")
        return dict(speed=speed, va=va, vb=vb, rel=rel, t=t, d0=d0, ra=ra, rb=rb, neg=roots[0], dist0=k * c2 * t)
    raise AssertionError  # pragma: no cover


def _intercept_text(s):
    return (
        f"A coastguard boat at {_pt(*s['ra'])} km can travel at a constant speed of {s['speed']} km/h in any direction. "
        f"A smuggler's boat at {_pt(*s['rb'])} km moves with constant velocity {_vec(*s['vb'])} km/h (x east, y north; i east, j north). "
        "The coastguard sets off now on a straight course to intercept the smuggler."
    )


def _hard_intercept_time(rng):
    s = _intercept(rng)
    t = s["t"]
    cands = [
        (Fraction(s["dist0"], s["speed"]), "intercept-ignores-motion"),
        (s["dist0"], "intercept-gives-distance"),
        (s["neg"], "intercept-negative-root"),
        (s["speed"] * t, "intercept-gives-chaser-distance"),
    ]
    dist = _dist(t, cands)
    enough(dist)
    stem = f"{_intercept_text(s)} After how many hours does the coastguard boat reach the smuggler?"
    working = [
        "Let the interception happen after t hours: the separation vector plus the target's displacement has length speed x t.",
        f"Solving the resulting quadratic gives one positive root, t = {_hours(t)}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(t), answer_kind="number")


def _hard_intercept_component(rng):
    s = _intercept(rng)
    comp = rng.choice((0, 1))
    value = s["va"][comp]
    # second route: the chaser flies from ra to rb + vb t in t hours.
    t = s["t"]
    check(Fraction(s["rb"][comp] + s["vb"][comp] * t - s["ra"][comp], t) == value, "velocity component differs")
    cands = [
        (s["vb"][comp], "intercept-copies-target-velocity"),
        (-value, "intercept-sign-flipped"),
        (s["rel"][comp], "intercept-relative-component"),
        (s["va"][1 - comp], "relvel-swaps-components"),
    ]
    dist = _dist(value, cands)
    enough(dist)
    axis = "i" if comp == 0 else "j"
    stem = f"{_intercept_text(s)} What is the {axis}-component of the coastguard boat's velocity, in km/h?"
    working = [
        f"The boat meets the smuggler after {_hours(t)}, at {_pt(*(s['rb'][0] + s['vb'][0] * t, s['rb'][1] + s['vb'][1] * t))}.",
        f"Its {axis}-component is the displacement in that direction divided by {t}, which is {value} km/h.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


_EASY = (_easy_component, _easy_straight, _easy_reverse)
_MEDIUM = (_medium_perpendicular, _medium_components, _medium_position)
_HARD = (_hard_approach_time, _hard_approach_distance, _hard_intercept_time, _hard_intercept_component)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w7b_relative_velocity(rng, tier="easy"):
    """Relative velocity, closest approach and interception; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if not (pool is not None):
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam ---------------------------------------------------------------------------------------------------------------

# (j, k*h, m): the least distance is j*c; staying within m*c needs half-duration h = (k*h)/k hours
_EXAM_RANGES = ((3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10), (5, 12, 13), (12, 5, 13), (9, 12, 15), (8, 15, 17))


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w7b_relative_velocity(rng, tier="exam"):
    """Exam chain: relative speed, time of closest approach, least distance, and when the ships first come within range."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a closest-approach chain"):
        j, kh, m = rng.choice(_EXAM_RANGES)
        k = rng.choice([x for x in (1, 2, 3) if kh % x == 0])
        half = kh // k
        t = half + rng.randint(1, 8)
        s = _approach(rng, triples=((3, 4, 5),), fixed=dict(triple=(3, 4, 5), k=k, j=j, t=t))
        c = 5
        rng_dist = m * c
        # second route for the range: |r0 + v t|^2 = D^2 has roots t* -/+ h with h from the quadratic formula
        qa = s["v"][0] ** 2 + s["v"][1] ** 2
        qb = 2 * (s["r0"][0] * s["v"][0] + s["r0"][1] * s["v"][1])
        qc = s["r0"][0] ** 2 + s["r0"][1] ** 2 - rng_dist**2
        disc = qb * qb - 4 * qa * qc
        root = isqrt(disc)
        if root * root != disc:
            continue
        first = Fraction(-qb - root, 2 * qa)
        if first == t - half and first > 0:
            break
    else:
        raise AssertionError  # pragma: no cover
    D = rng_dist
    v, r0 = s["v"], s["r0"]
    speed, dmin = s["speed"], s["dmin"]
    ra, rb, va, vb = s["ra"], s["rb"], s["va"], s["vb"]

    a_dist = _dist(
        speed,
        [
            (abs(v[0]) + abs(v[1]), "relvel-adds-components"),
            (v[0] ** 2 + v[1] ** 2, "relvel-forgets-root"),
            (max(abs(v[0]), abs(v[1])), "relvel-takes-larger-component"),
            (abs(va[0] + vb[0]) + abs(va[1] + vb[1]), "relvel-adds-velocities"),
        ],
    )
    b_dist = _dist(
        t,
        [
            (-t, "approach-sign-error"),
            (t * speed, "approach-divides-by-speed"),
            (Fraction(-r0[0], v[0]) if v[0] else None, "approach-equal-coordinate"),
            (Fraction(-r0[1], v[1]) if v[1] else None, "approach-equal-coordinate"),
        ],
    )
    c_dist = _dist(
        dmin,
        [
            (dmin * dmin, "approach-gives-square"),
            (t * speed, "approach-distance-travelled"),
            (abs(s["w"][0]) + abs(s["w"][1]), "approach-adds-components"),
            (dmin + speed, "approach-distance-travelled"),
        ],
    )
    first_time = t - half
    d_dist = _dist(
        first_time,
        [
            (t, "approach-first-within-uses-closest"),
            (t + half, "approach-last-within"),
            (half, "approach-half-width"),
            (t - 2 * half if t != 2 * half else None, "approach-half-width"),
        ],
    )
    for d in (a_dist, b_dist, c_dist, d_dist):
        enough(d)
    return dict(
        text=f"{_approach_text(s, 'Ship', 'km/h', 'km')}",
        parts=[
            part(
                "Find the speed of A relative to B, in km/h.",
                "number",
                num_fields(speed),
                1,
                [
                    f"The relative velocity is {_vec(*v)} km/h, so the speed is sqrt({paren_negative(v[0])}^2 + {paren_negative(v[1])}^2) = {speed} km/h."
                ],
                a_dist,
                sensible_range=(0, None),
            ),
            part(
                "How many hours after noon are the ships closest together?",
                "number",
                num_fields(t),
                2,
                [
                    f"The relative position {_pt(*r0)} + t({_vec(*v)}) is perpendicular to the relative velocity at t = {t}."
                ],
                b_dist,
                sensible_range=(0, None),
            ),
            part(
                "What is the least distance between the ships, in km?",
                "number",
                num_fields(dmin),
                2,
                [f"At t = {t} the relative position has length {dmin} km."],
                c_dist,
                follow_through=f"sqrt(({r0[0]} + ({v[0]})*part_b)**2 + ({r0[1]} + ({v[1]})*part_b)**2)",
                sensible_range=(0, None),
            ),
            part(
                f"The ships can see each other when they are within {D} km. At what time, in hours after noon, do they first come within {D} km?",
                "number",
                num_fields(first_time),
                2,
                [
                    f"The half-duration within range is sqrt({D}^2 - {dmin}^2) / {speed} = {_hours(half)}, so the time is {t} - {half} = {first_time}."
                ],
                d_dist,
                follow_through=f"part_b - sqrt({D}**2 - part_c**2)/part_a",
                sensible_range=(0, None),
            ),
        ],
    )
