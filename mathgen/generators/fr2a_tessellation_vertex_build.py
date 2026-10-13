"""``fr2a_tessellation_vertex``: angles of regular polygons that meet at a point in a tiling.

Skill ``f10-tessellation-vertex-angles`` (AC9M6SP03). Regular polygons tile a surface when the angles of the shapes
that meet at a point add to 360 degrees. Every answer is a whole number.

Tiers:

- easy:   one interior angle of a named regular polygon, or how many of one polygon fit round a point.
- medium: two regular polygons are named and meet a third at a point with nothing left over: how many sides does the
          third one have? (The learner needs both the interior angle of a regular polygon and the sum at a point.)
- hard:   a vertex with several polygons where one group of identical polygons is unknown (how many sides has each?),
          or identical polygons are placed round a point so they just touch and the size of the gap that is left is
          asked.

The vertex configurations for the medium and hard tiers are found by search (every multiset of regular polygons whose
interior angles add to exactly 360 degrees), not typed in, and each answer is found a second way from the exterior angle.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, phrase, pick_distractors

from .exd_x_sc3_support import redraw

_SIDES = (3, 4, 5, 6, 8, 9, 10, 12)
_WIDE = (3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24)
_NAMES = {
    3: "equilateral triangle",
    4: "square",
    5: "regular pentagon",
    6: "regular hexagon",
    7: "regular heptagon",
    8: "regular octagon",
    9: "regular nonagon",
    10: "regular decagon",
    12: "regular dodecagon",
}


def _name(n: int) -> str:
    return _NAMES.get(n, f"regular polygon with {n} sides")


def _a(n: int) -> str:
    """The shape with its article: 'an equilateral triangle', 'a square', 'a regular polygon with 15 sides'."""
    name = _name(n)
    return ("an " if name[0] in "aeiou" else "a ") + name


_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _plural(n: int) -> str:
    return {3: "equilateral triangles", 4: "squares"}.get(n, _name(n) + "s")


def interior(n: int) -> Fraction:
    """Interior angle of a regular n-gon from the angle sum of an n-gon."""
    return Fraction(180 * (n - 2), n)


def interior_by_exterior(n: int) -> Fraction:
    """The same angle from the exterior angle (360 / n), the second route."""
    return 180 - Fraction(360, n)


@lru_cache(maxsize=None)
def _vertex_configs(parts: int) -> tuple[tuple[int, ...], ...]:
    """Every multiset of ``parts`` regular polygons (sides from _WIDE) whose interior angles add to 360 degrees."""
    return tuple(c for c in combinations_with_replacement(_WIDE, parts) if sum(interior(n) for n in c) == 360)


def sides_for_angle(theta: Fraction) -> Fraction:
    """Sides of the regular polygon whose interior angle is ``theta`` degrees: n = 360 / (180 - theta)."""
    if not 0 < theta < 180:
        return Fraction(0)  # no regular polygon has this interior angle
    return Fraction(360) / (180 - theta)


def _ints(values):
    return [Distractor(str(int(v)), tag, int(v)) for v, tag in values if Fraction(v).denominator == 1 and v > 0]


def _finish(text, key, cands, working):
    key_text = str(int(key))
    dis = pick_distractors(key_text, [c for c in cands if c.text != key_text], 3)
    check(len(dis) >= 3, f"only {len(dis)} distractors for {key_text}")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=int(key),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        answer_kind="number",
        working=working,
        distractors=dis,
    )


# ------------------------------------------------------------------------------------------ easy


def _easy_missing_angle(rng):
    """Three tiles meet at a point; two angles are given: the third is what is left of 360 degrees."""
    angles = sorted({int(interior(n)) for n in _SIDES if n != 3} | {60})
    for _ in attempts(200, "two angles that leave a positive third angle"):
        a, b = rng.choice(angles), rng.choice(angles)
        third = 360 - a - b
        if 60 <= third < 180 and third not in (a, b):
            break
    text = phrase(
        rng,
        (
            "Three tiles meet at one point of a tiling with no gaps. Two of the tile angles at the point are "
            f"{a}° and {b}°. What is the size of the third angle, in degrees?",
            f"Round a point in a tiling there are three tile corners with no gaps. Two of them are {a}° and {b}°. "
            "How big is the third corner, in degrees?",
        ),
    )
    cands = _ints(
        [
            (180 - a - b, "fr2a-straight-line-for-point"),
            (a + b, "fr2a-angles-total-for-gap"),
            (360 - a, "fr2a-one-angle-taken-for-all"),
            (360 - b, "fr2a-one-angle-taken-for-all"),
        ]
    )
    working = [f"The angles round a point add to 360°.", f"360° − {a}° − {b}° = {third}°."]
    return _finish(text, third, cands, working)


def _easy(rng):
    flavour = rng.randrange(3)
    if flavour == 2:
        return _easy_missing_angle(rng)
    if flavour == 0:
        n = rng.choice((5, 8, 9, 10, 12, 6, 3))
        theta = interior(n)
        check(theta == interior_by_exterior(n) and theta.denominator == 1, "interior angle routes disagree")
        text = phrase(
            rng,
            (
                "A tiler is choosing between shapes. What is the size of one interior angle of {shape}, in degrees?",
                "One corner of {shape} is cut from a tile. What is the size of the interior angle at that corner, in degrees?",
                "What is the size of each interior angle of {shape}, in degrees?",
            ),
            shape=_a(n),
        )
        cands = _ints(
            [
                (Fraction(360, n), "fr2a-exterior-for-interior"),
                (Fraction(180 * (n - 2)), "fr2a-angle-sum-for-one-angle"),
                (Fraction(180, n), "fr2a-straight-angle-over-sides"),
                (360 - theta, "fr2a-reflex-for-interior"),
                (Fraction(180 * (n - 1), n), "fr2a-sum-of-n-minus-one-triangles"),
            ]
        )
        working = [
            f"The exterior angles of a regular polygon add to 360°, so each exterior angle is 360° ÷ {n} = {int(Fraction(360, n))}°.",
            f"An interior angle and its exterior angle make 180°, so the interior angle is 180° − {int(Fraction(360, n))}° = {int(theta)}°.",
        ]
        return _finish(text, theta, cands, working)
    # how many of one polygon fit round a point with nothing left over: only the triangle, square and hexagon
    n = rng.choice((3, 4, 6))
    theta = interior(n)
    k = Fraction(360) / theta
    check(k.denominator == 1, "the shapes do not fit round a point exactly")
    text = phrase(
        rng,
        (
            "{shape_plural} are fitted together round a point with no gaps and no overlaps. How many fit round the point?",
            "A floor is tiled with {shape_plural_lower}. How many tiles meet at each corner point?",
        ),
        shape_plural=_plural(n).capitalize(),
        shape_plural_lower=_plural(n),
    )
    cands = _ints(
        [
            (Fraction(n), "fr2a-sides-for-count"),
            (Fraction(180) / theta, "fr2a-straight-line-for-point"),
            (theta, "fr2a-angle-for-count"),
            (k + 1, "fr2a-count-off-by-one"),
            (k - 1, "fr2a-count-off-by-one"),
            (Fraction(360, n), "fr2a-exterior-for-interior"),
        ]
    )
    working = [
        f"One interior angle of a {_name(n)} is {int(theta)}°.",
        f"The angles round a point add to 360°, so 360° ÷ {int(theta)}° = {int(k)} tiles.",
    ]
    return _finish(text, k, cands, working)


# ------------------------------------------------------------------------------------------ medium


def _medium(rng):
    configs = [c for c in _vertex_configs(3)]
    cfg = list(rng.choice(configs))
    rng.shuffle(cfg)
    unknown = cfg[2]
    a, b = cfg[0], cfg[1]
    theta_a, theta_b = interior(a), interior(b)
    missing = 360 - theta_a - theta_b
    n = sides_for_angle(missing)
    check(
        n == unknown and interior_by_exterior(int(n)) == missing, "third polygon's sides differ between the two routes"
    )
    if a == b:
        known = f"two {_plural(a)}"
    else:
        known = f"{_a(a)} and {_a(b)}"
    text = phrase(
        rng,
        (
            "At one corner point of a tiling, {known} meet a third regular polygon with no gaps and no overlaps. "
            "How many sides does the third polygon have?",
            "A tiling pattern has {known} and one more regular polygon fitting exactly round a point. "
            "How many sides has the other polygon?",
        ),
        known=known,
    )
    cands = _ints(
        [
            (missing, "fr2a-angle-for-sides"),
            (Fraction(360) / missing, "fr2a-360-over-interior"),
            (n + 1, "fr2a-count-off-by-one"),
            (n - 1, "fr2a-count-off-by-one"),
            (Fraction(a + b), "fr2a-sides-added"),
            (180 - missing, "fr2a-exterior-for-interior"),
        ]
    )
    working = [
        f"Interior angles: {_name(a)} {int(theta_a)}°, {_name(b)} {int(theta_b)}°.",
        f"The third angle is 360° − {int(theta_a)}° − {int(theta_b)}° = {int(missing)}°.",
        f"Its exterior angle is 180° − {int(missing)}° = {int(180 - missing)}°, so it has 360° ÷ {int(180 - missing)}° = {int(n)} sides.",
    ]
    return _finish(text, n, cands, working)


# ------------------------------------------------------------------------------------------ hard


def _hard_identical(rng):
    """A vertex of 4 to 6 polygons where m >= 2 identical ones are unknown."""
    options = []
    for parts in (4, 5, 6):
        for cfg in _vertex_configs(parts):
            for n in set(cfg):
                m = cfg.count(n)
                rest = [x for x in cfg if x != n]
                # the other polygons are known, so they must not include more of the unknown kind
                if m >= 2 and rest and len(set(rest)) <= 2:
                    options.append((cfg, n, m, tuple(rest)))
    cfg, n, m, rest = rng.choice(options)
    rest_angle = sum(interior(x) for x in rest)
    theta = (360 - rest_angle) / m
    check(sides_for_angle(theta) == n, "identical polygons' sides differ between the two routes")
    counts: dict[int, int] = {}
    for x in rest:
        counts[x] = counts.get(x, 0) + 1
    words = []
    for x, c in sorted(counts.items()):
        words.append(_a(x) if c == 1 else f"{_WORDS[c]} {_plural(x)}")
    known = " and ".join(words)
    text = phrase(
        rng,
        (
            "{known} and {m} identical regular polygons meet at a point with no gaps and no overlaps. "
            "How many sides does each of the identical polygons have?",
            "At one point of a tiling there are {known}, and {m} more regular polygons that are all the same. "
            "How many sides does each of those {m} polygons have?",
        ),
        known=known,
        m=_WORDS[m],
    )
    text = text[0].upper() + text[1:]
    cands = _ints(
        [
            (theta, "fr2a-angle-for-sides"),
            (Fraction(360) - rest_angle, "fr2a-shared-angle-not-split"),
            (sides_for_angle(360 - rest_angle), "fr2a-shared-angle-not-split"),
            (Fraction(360) / theta, "fr2a-360-over-interior"),
            (n + 1, "fr2a-count-off-by-one"),
            (n - 1, "fr2a-count-off-by-one"),
            (180 - theta, "fr2a-exterior-for-interior"),
        ]
    )
    working = [
        "The known polygons use "
        + " + ".join(f"{int(interior(x))}°" for x in rest)
        + f" = {int(rest_angle)}° of the 360°.",
        f"The other {_WORDS[m]} share the remaining {int(360 - rest_angle)}°, so each interior angle is {int(theta)}°.",
        f"Its exterior angle is {int(180 - theta)}°, so it has 360° ÷ {int(180 - theta)}° = {int(n)} sides.",
    ]
    return _finish(text, n, cands, working)


def _hard_gap(rng):
    n = rng.choice((5, 8, 9, 10, 12))
    theta = interior(n)
    k = int(Fraction(360) // theta)
    gap = 360 - k * theta
    check(gap > 0 and gap == 360 - sum(interior_by_exterior(n) for _ in range(k)), "gap differs between the two routes")
    text = phrase(
        rng,
        (
            f"{_WORDS[k].capitalize()} {_plural(n)} are placed round a point, each sharing a side with the next. "
            "What is the size of the gap that is left, in degrees?",
            f"{_WORDS[k].capitalize()} {_plural(n)} are fitted round a point, each touching the next along a side. "
            "How many degrees of the 360° round the point are left uncovered?",
        ),
    )
    text = text[0].upper() + text[1:]
    cands = _ints(
        [
            (180 - k * theta if 180 - k * theta > 0 else theta - gap, "fr2a-straight-line-for-point"),
            (k * theta, "fr2a-angles-total-for-gap"),
            (360 - theta, "fr2a-one-angle-taken-for-all"),
            (gap + theta, "fr2a-gap-with-one-tile-too-few"),
            (theta - gap if theta - gap > 0 else gap * 2, "fr2a-angle-minus-gap"),
            (Fraction(360, n), "fr2a-exterior-for-interior"),
        ]
    )
    working = [
        f"One interior angle of a {_name(n)} is {int(theta)}°.",
        f"{_WORDS[k].capitalize()} of them use {k} × {int(theta)}° = {int(k * theta)}°.",
        f"The gap is 360° − {int(k * theta)}° = {int(gap)}°.",
    ]
    return _finish(text, gap, cands, working)


def _hard(rng):
    return _hard_identical(rng) if rng.random() < 0.6 else _hard_gap(rng)


@generator(
    "fr2a_tessellation_vertex",
    level=6,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2a_tessellation_vertex(rng, tier="easy"):
    """Interior angles of regular polygons and the angles that meet at a point of a tiling."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    return redraw(build, rng, "a question with enough distinct wrong answers")
