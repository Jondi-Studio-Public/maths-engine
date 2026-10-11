"""Generates `f10c_congruence_criteria` questions for `f10-congruence-similarity-criteria`: two triangles are
described in words and numbers (the sides, angles and right angle that are known about each, with the vertex
correspondence stated), and the learner picks the one conclusion those facts justify.

Answer space (a closed list of labels, offered four at a time, exactly one correct):

- ``Congruent by SSS / SAS / ASA / AAS / RHS`` (a real congruence test is met);
- ``Similar by AA / SSS (proportional sides) / SAS (proportional sides)`` (a real similarity test is met and the
  triangles are not the same size);
- ``Neither can be concluded from this information`` (no test is met);
- the two classic false tests ``Congruent by SSA`` and ``Congruent by AAA``, which are only ever distractors.

Tiers escalate by what has to be decided, not by bigger numbers:

- **easy**: three matched parts are listed in the natural order for the test (side, angle, side for SAS, and so
  on) and the triangles are the same size. The decision is naming the congruence test from the *kinds* of parts.
- **medium**: adds the same-size / different-size decision (sides all in proportion, so a similarity test rather
  than a congruence one), the angles-only test AA, and the "not enough" answer (a side ratio that breaks on one
  side, or too few matched parts).
- **hard**: the parts are listed out of order and named by vertex, so the learner must decide from the letters
  whether a stated angle sits *between* two stated sides (SAS) or not (SSA, which proves nothing); a right
  angle between two legs is SAS but a right angle with the hypotenuse is RHS; and a third angle has to be found
  from the angle sum before two triangles can be matched by AA, or shown not to match.

Verification: every question is built from a chosen answer and then re-derived by `_valid_labels`, which applies
each test's definition to the stated facts (not to the case the facts were built from). The answer must be the
only congruence or similarity test the facts meet, no offered distractor may be met, and "neither" requires that
no test is met at all.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations

from ..core import check, generator, phrase

# -- Labels -----------------------------------------------------------------

C_SSS = "Congruent by SSS"
C_SAS = "Congruent by SAS"
C_ASA = "Congruent by ASA"
C_AAS = "Congruent by AAS"
C_RHS = "Congruent by RHS"
C_SSA = "Congruent by SSA"
C_AAA = "Congruent by AAA"
S_AA = "Similar by AA"
S_SSS = "Similar by SSS (proportional sides)"
S_SAS = "Similar by SAS (proportional sides)"
NEITHER = "Neither can be concluded from this information"

CONGRUENT_VALID = [C_SSS, C_SAS, C_ASA, C_AAS, C_RHS]
FALSE_TESTS = [C_SSA, C_AAA]
SIMILAR = [S_AA, S_SSS, S_SAS]
ALL_LABELS = CONGRUENT_VALID + FALSE_TESTS + SIMILAR + [NEITHER]

# Note ids (they must match the `misconceptions` of the skill file).
M_SSA = "ssa-accepted"
M_AAA = "angles-only-congruent"
M_REJECT = "rejects-sufficient-conditions"
M_MISLABEL = "mislabels-the-criterion"
M_SIM_AS_CONG = "similar-treated-as-congruent"
M_SOME_RATIOS = "checks-only-some-ratios"
M_ANY_ANGLE = "treats-any-angle-as-included"
M_RHS_SAS = "confuses-rhs-with-sas"
M_SKIP_THIRD = "skips-the-third-angle"
M_WRONG_VERTEX = "matches-angles-at-wrong-vertices"

TIERS = ("easy", "medium", "hard")

_NAME_SETS = [
    ("ABC", "DEF"),
    ("PQR", "XYZ"),
    ("KLM", "UVW"),
    ("GHJ", "STU"),
    ("LMN", "RST"),
    ("ABC", "PQR"),
]

_EDGES = [(0, 1), (1, 2), (0, 2)]


# -- Facts about a triangle --------------------------------------------------


class Tri:
    """What is stated about one triangle: side lengths by vertex pair and angles by vertex."""

    def __init__(self, sides=None, angles=None):
        self.sides = dict(sides or {})  # (i, j) with i < j -> int cm
        self.angles = dict(angles or {})  # vertex -> int degrees


def _third(v1: int, v2: int) -> int:
    return ({0, 1, 2} - {v1, v2}).pop()


def _triangle_ok(a: int, b: int, c: int) -> bool:
    return len({a, b, c}) == 3 and a + b > c + 1 and a + c > b + 1 and b + c > a + 1


def _draw_sides(rng, lo: int, hi: int):
    for _ in range(400):
        s = [rng.randint(lo, hi) for _ in range(3)]
        if _triangle_ok(*s):
            return s
    raise RuntimeError("f10c_congruence_criteria: could not draw a triangle")


def _scaled(t: Tri, k: int) -> Tri:
    return Tri({e: v * k for e, v in t.sides.items()}, dict(t.angles))


def _angle_value(rng, avoid=()) -> int:
    while True:
        a = rng.randint(30, 85)
        if a not in avoid:
            return a


# -- The independent definition of each test ---------------------------------


def _known_angles(t: Tri) -> dict:
    """Stated angles plus a third one found from the angle sum when exactly two are stated."""
    known = dict(t.angles)
    if len(known) == 2:
        v = _third(*known)
        known[v] = 180 - sum(known.values())
    return known


def _valid_labels(t1: Tri, t2: Tri) -> set:
    """Every valid test (named exactly as an option) that the stated facts meet, from the definitions."""
    k1, k2 = _known_angles(t1), _known_angles(t2)
    # A corresponding pair of angles that differ rules out similarity, so no test applies.
    if any(k1[v] != k2[v] for v in k1 if v in k2):
        return set()
    common = [e for e in t1.sides if e in t2.sides]
    ratios = {Fraction(t2.sides[e], t1.sides[e]) for e in common}
    if len(ratios) > 1:
        return set()
    k = next(iter(ratios)) if ratios else None
    stated_match = [v for v in t1.angles if v in t2.angles and t1.angles[v] == t2.angles[v]]
    known_match = [v for v in k1 if v in k2 and k1[v] == k2[v]]
    out = set()

    def sas_vertices():
        for e1, e2 in combinations(common, 2):
            shared = set(e1) & set(e2)
            if len(shared) == 1 and next(iter(shared)) in stated_match:
                yield next(iter(shared))

    if len(common) == 3:
        out.add(S_SSS)
    if any(True for _ in sas_vertices()):
        out.add(S_SAS)
    if len(known_match) >= 2:
        out.add(S_AA)
    if k == 1:
        if len(common) == 3:
            out.add(C_SSS)
        if any(True for _ in sas_vertices()):
            out.add(C_SAS)
        for u, w in combinations(stated_match, 2):
            if (min(u, w), max(u, w)) in common:
                out.add(C_ASA)
            if any(e != (min(u, w), max(u, w)) for e in common):
                out.add(C_AAS)
        for v in stated_match:
            if t1.angles[v] == 90:
                hyp = tuple(sorted({0, 1, 2} - {v}))
                legs = [e for e in common if v in e]
                if hyp in common and legs:
                    out.add(C_RHS)
    return out


# -- Case builders ------------------------------------------------------------
# Each returns (t1, t2, order, label, case) where `order` lists the part keys in the natural reading order
# (("s", edge) or ("a", vertex)).


def _pair(base: Tri, k: int, swap: bool):
    big = _scaled(base, k)
    return (big, base) if swap else (base, big)


def _case_sss(rng, k, swap):
    s = _draw_sides(rng, 4, 14 if k == 1 else 8)
    base = Tri(dict(zip(_EDGES, s)))
    t1, t2 = _pair(base, k, swap)
    return t1, t2, [("s", e) for e in _EDGES]


def _case_sas(rng, k, swap, angle=None):
    v = rng.randrange(3)
    e1, e2 = sorted(e for e in _EDGES if v in e)
    lo, hi = (4, 14) if k == 1 else (3, 8)
    a, b = rng.sample(range(lo, hi + 1), 2)
    ang = angle if angle is not None else rng.choice([x for x in range(30, 131) if x != 90])
    base = Tri({e1: a, e2: b}, {v: ang})
    t1, t2 = _pair(base, k, swap)
    return t1, t2, [("s", e1), ("a", v), ("s", e2)]


def _case_asa(rng, swap):
    u, w = sorted(rng.sample(range(3), 2))
    a1 = _angle_value(rng)
    a2 = _angle_value(rng, avoid=(a1,))
    base = Tri({(u, w): rng.randint(4, 14)}, {u: a1, w: a2})
    t1, t2 = _pair(base, 1, swap)
    return t1, t2, [("a", u), ("s", (u, w)), ("a", w)]


def _case_aas(rng, swap):
    u, w = sorted(rng.sample(range(3), 2))
    x = _third(u, w)
    a1 = _angle_value(rng)
    a2 = _angle_value(rng, avoid=(a1,))
    side = tuple(sorted((rng.choice([u, w]), x)))
    base = Tri({side: rng.randint(4, 14)}, {u: a1, w: a2})
    t1, t2 = _pair(base, 1, swap)
    return t1, t2, [("a", u), ("a", w), ("s", side)]


def _case_rhs(rng, swap):
    v = rng.randrange(3)
    hyp = tuple(sorted({0, 1, 2} - {v}))
    leg = tuple(sorted((v, rng.choice(hyp))))
    leg_len = rng.randint(5, 12)
    hyp_len = leg_len + rng.randint(2, 7)
    base = Tri({hyp: hyp_len, leg: leg_len}, {v: 90})
    t1, t2 = _pair(base, 1, swap)
    return t1, t2, [("a", v), ("s", hyp), ("s", leg)]


def _case_sas_right(rng, swap):
    v = rng.randrange(3)
    e1, e2 = sorted(e for e in _EDGES if v in e)
    a, b = rng.sample(range(5, 13), 2)
    base = Tri({e1: a, e2: b}, {v: 90})
    t1, t2 = _pair(base, 1, swap)
    return t1, t2, [("s", e1), ("a", v), ("s", e2)]


def _case_aa(rng, k, swap):
    """Two stated angles, no more; with a size cue (one side pair in ratio k) when k > 1."""
    u, w = sorted(rng.sample(range(3), 2))
    a1 = _angle_value(rng)
    a2 = _angle_value(rng, avoid=(a1,))
    sides = {}
    if k > 1:
        sides[rng.choice(_EDGES)] = rng.randint(3, 8)
    base = Tri(sides, {u: a1, w: a2})
    t1, t2 = _pair(base, k, swap)
    order = [("a", u), ("a", w)] + [("s", e) for e in sides]
    return t1, t2, order


def _case_no_ratio(rng, k, swap):
    """All three sides stated, but one pair breaks the ratio: not similar, not congruent."""
    s = _draw_sides(rng, 4 if k == 1 else 3, 12 if k == 1 else 8)
    base = Tri(dict(zip(_EDGES, s)))
    for _ in range(100):
        bad = rng.randrange(3)
        delta = rng.choice([-2, -1, 1, 2, 3])
        big_sides = [v * k + (delta if i == bad else 0) for i, v in enumerate(s)]
        if big_sides[bad] > 0 and _triangle_ok(*big_sides) and Fraction(big_sides[bad], s[bad]) != k:
            big = Tri(dict(zip(_EDGES, big_sides)))
            t1, t2 = (big, base) if swap else (base, big)
            return t1, t2, [("s", e) for e in _EDGES]
    raise RuntimeError("f10c_congruence_criteria: could not break a ratio")


def _case_too_few(rng, k, swap):
    """Only two matched parts stated: two sides, or a side and an angle."""
    lo, hi = (4, 14) if k == 1 else (3, 8)
    if rng.random() < 0.5:
        e1, e2 = rng.sample(_EDGES, 2)
        a, b = rng.sample(range(lo, hi + 1), 2)
        base = Tri({e1: a, e2: b})
        order = [("s", e1), ("s", e2)]
    else:
        e = rng.choice(_EDGES)
        v = rng.randrange(3)
        base = Tri({e: rng.randint(lo, hi)}, {v: _angle_value(rng)})
        order = [("s", e), ("a", v)]
    t1, t2 = _pair(base, k, swap)
    return t1, t2, order


def _case_ssa(rng, k, swap):
    """Two sides and an angle that is NOT between them (the side opposite the angle is the shorter one, so two
    different triangles fit and nothing is proved)."""
    for _ in range(400):
        v = rng.randrange(3)
        x = rng.choice([w for w in range(3) if w != v])
        y = _third(v, x)
        ang = rng.randint(30, 65)
        b = rng.randint(8, 16) if k == 1 else rng.randint(5, 9)
        lo = math.ceil(b * math.sin(math.radians(ang))) + 1
        if lo > b - 1:
            continue
        a = rng.randint(lo, b - 1)
        base = Tri({tuple(sorted((v, x))): b, tuple(sorted((x, y))): a}, {v: ang})
        t1, t2 = _pair(base, k, swap)
        return t1, t2, [("s", tuple(sorted((v, x)))), ("s", tuple(sorted((x, y)))), ("a", v)]
    raise RuntimeError("f10c_congruence_criteria: could not draw an SSA configuration")


def _case_aa_derived(rng, match: bool):
    """Two angles stated in each triangle, at partly different vertices. The third angle of each is found by
    the angle sum; the triangles match (AA) only if every corresponding angle agrees."""
    while True:
        p = rng.randint(35, 80)
        q = rng.randint(35, 80)
        r = 180 - p - q
        if r >= 30 and len({p, q, r}) == 3:
            break
    angs = [p, q, r]
    rng.shuffle(angs)
    v1 = sorted(rng.sample(range(3), 2))
    t1 = Tri(angles={v: angs[v] for v in v1})
    shared = rng.choice(v1)
    other = rng.choice([v for v in range(3) if v not in v1])
    v2 = sorted([shared, other])
    t2_angles = {v: angs[v] for v in v2}
    if not match:
        v = other
        for _ in range(100):
            wrong = rng.randint(35, 100)
            if wrong != angs[v] and 180 - angs[shared] - wrong > 20:
                t2_angles[v] = wrong
                break
    t2 = Tri(angles=t2_angles)
    return t1, t2, [("a", v) for v in v1]


# -- Rendering ---------------------------------------------------------------


def _fmt_part(kind, key, tri: Tri, names: str, reverse: bool) -> str:
    if kind == "s":
        i, j = key
        pair = names[j] + names[i] if reverse else names[i] + names[j]
        return f"{pair} = {tri.sides[key]} cm"
    return f"angle {names[key]} = {tri.angles[key]} degrees"


def _join(parts) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _part_keys(tri: Tri):
    return [("s", e) for e in tri.sides] + [("a", v) for v in tri.angles]


def _render(rng, tier, t1, t2, order, names):
    n1, n2 = names
    parts = []
    for tri, nm in ((t1, n1), (t2, n2)):
        keys = [key for key in order if key[1] in (tri.sides if key[0] == "s" else tri.angles)]
        for key in _part_keys(tri):
            if key not in keys:
                keys.append(key)
        if tier == "hard":
            rng.shuffle(keys)
        parts.append(
            _join([_fmt_part(kd, ky, tri, nm, tier == "hard" and kd == "s" and rng.random() < 0.4) for kd, ky in keys])
        )
    corr = f"({n1[0]} with {n2[0]}, {n1[1]} with {n2[1]} and {n1[2]} with {n2[2]})"
    templates = [
        "Triangle {n1} has {p1}. Triangle {n2} has {p2}. The vertices correspond in the order {corr}. "
        "Using only this information, which one of the following conclusions is justified?",
        "In triangle {n1}, {p1}. In triangle {n2}, {p2}. Corresponding vertices are {corr}. "
        "Which one of the following conclusions can be justified directly from this information alone?",
        "Two triangles are described. Triangle {n1}: {p1}. Triangle {n2}: {p2}. Vertices correspond in the "
        "order {corr}. Which conclusion is justified by exactly this information?",
    ]
    return phrase(rng, templates, n1=n1, n2=n2, p1=parts[0], p2=parts[1], corr=corr)


# -- Explanations --------------------------------------------------------------

_WHY = {
    C_SSS: "All three pairs of sides are equal, so the triangles are congruent (SSS).",
    C_SAS: "Two pairs of sides are equal and the equal angles lie between those sides, so the triangles are congruent (SAS).",
    C_ASA: "Two pairs of angles are equal and the equal sides lie between those angles, so the triangles are congruent (ASA).",
    C_AAS: "Two pairs of angles and a side not between them are equal, so the triangles are congruent (AAS).",
    C_RHS: "Each triangle has a right angle, and the hypotenuse and one other side are equal, so the triangles are congruent (RHS).",
    S_AA: "Two pairs of corresponding angles are equal, so the triangles are similar (AA).",
    S_SSS: "All three pairs of sides are in the same ratio, so the triangles are similar (SSS ratio).",
    S_SAS: "Two pairs of sides are in the same ratio and the angles between them are equal, so the triangles are similar (SAS ratio).",
    NEITHER: "The stated facts do not meet the conditions of any congruence or similarity test.",
}


# -- Distractor tags -----------------------------------------------------------


def _tag(answer: str, label: str, case: str) -> str:
    if label == NEITHER:
        return M_SKIP_THIRD if case == "aa_derived" else M_REJECT
    if label == C_SSA:
        return M_SSA
    if label == C_AAA:
        return M_AAA
    if case in ("ssa", "ssa_ratio") and label in (C_SAS, S_SAS):
        return M_ANY_ANGLE
    if case == "sas_right" and label == C_RHS:
        return M_RHS_SAS
    if case == "rhs" and label == C_SAS:
        return M_RHS_SAS
    if case in ("no_ratio", "no_ratio_same") and label in (C_SSS, S_SSS):
        return M_SOME_RATIOS
    if case == "aa_unmatched" and label == S_AA:
        return M_WRONG_VERTEX
    if label in CONGRUENT_VALID and answer in SIMILAR:
        return M_SIM_AS_CONG
    return M_MISLABEL


_MUST = {
    "ssa": [C_SSA, C_SAS],
    "ssa_ratio": [C_SSA, S_SAS],
    "sas_right": [C_RHS],
    "rhs": [C_SAS],
    "no_ratio": [S_SSS],
    "no_ratio_same": [C_SSS],
    "aa_derived": [NEITHER],
    "aa_unmatched": [S_AA],
    "sss_ratio": [C_SSS],
    "sas_ratio": [C_SAS],
    "aa": [C_AAA],
    "aa_sized": [C_AAA],
}


def _eligible(answer: str, valid: set) -> list:
    out = []
    for label in ALL_LABELS:
        if label == answer or label in valid:
            continue  # a met test is never offered as a wrong answer
        if answer in CONGRUENT_VALID and label in SIMILAR:
            continue  # same-size triangles are also similar, so a similarity label would not be wrong
        if answer == C_RHS and label == C_SSA:
            continue  # RHS is the one valid special case of SSA
        if {answer, label} == {C_ASA, C_AAS}:
            continue  # two angles and a side meet both, whichever way the side sits, once the third angle is found
        out.append(label)
    return out


def _distractors(rng, answer, valid, case):
    pool = _eligible(answer, valid)
    chosen = [m for m in _MUST.get(case, []) if m in pool]
    rest = [p for p in pool if p not in chosen]
    rng.shuffle(rest)
    chosen = (chosen + rest)[:3]
    return chosen


# -- The tier builders -----------------------------------------------------------


def _pick(rng, table):
    r = rng.random() * sum(w for _, w in table)
    for item, w in table:
        r -= w
        if r <= 0:
            return item
    return table[-1][0]


def _build_case(rng, tier):
    """Returns (t1, t2, order, label, case)."""
    swap = rng.random() < 0.5
    if tier == "easy":
        label = rng.choice(CONGRUENT_VALID)
        case = {C_SSS: "sss", C_SAS: "sas", C_ASA: "asa", C_AAS: "aas", C_RHS: "rhs"}[label]
        if case == "sss":
            t1, t2, order = _case_sss(rng, 1, swap)
        elif case == "sas":
            t1, t2, order = _case_sas(rng, 1, swap)
        elif case == "asa":
            t1, t2, order = _case_asa(rng, swap)
        elif case == "aas":
            t1, t2, order = _case_aas(rng, swap)
        else:
            t1, t2, order = _case_rhs(rng, swap)
        return t1, t2, order, label, case
    if tier == "medium":
        case = _pick(
            rng,
            [
                ("sss", 9), ("sas", 9), ("asa", 9), ("aas", 9), ("rhs", 9),
                ("sss_ratio", 12), ("sas_ratio", 12), ("aa", 6), ("aa_sized", 6),
                ("no_ratio", 6), ("no_ratio_same", 4), ("too_few", 9),
            ],
        )  # fmt: skip
    else:
        case = _pick(
            rng,
            [
                ("ssa", 8), ("ssa_ratio", 6), ("sas_vertex", 6), ("sas_ratio_vertex", 6), ("rhs", 14),
                ("sas_right", 14), ("aa_derived", 14), ("aa_unmatched", 10), ("sss_ratio", 6),
                ("aas", 8), ("no_ratio", 8),
            ],
        )  # fmt: skip
    k = rng.choice([2, 3, 4])
    if case == "sss":
        t1, t2, order = _case_sss(rng, 1, swap)
        label = C_SSS
    elif case == "sas" or case == "sas_vertex":
        t1, t2, order = _case_sas(rng, 1, swap)
        label = C_SAS
    elif case == "asa":
        t1, t2, order = _case_asa(rng, swap)
        label = C_ASA
    elif case == "aas":
        t1, t2, order = _case_aas(rng, swap)
        label = C_AAS
    elif case == "rhs":
        t1, t2, order = _case_rhs(rng, swap)
        label = C_RHS
    elif case == "sas_right":
        t1, t2, order = _case_sas_right(rng, swap)
        label = C_SAS
    elif case == "sss_ratio":
        t1, t2, order = _case_sss(rng, k, swap)
        label = S_SSS
    elif case in ("sas_ratio", "sas_ratio_vertex"):
        t1, t2, order = _case_sas(rng, k, swap)
        label = S_SAS
        case = "sas_ratio"
    elif case == "aa":
        t1, t2, order = _case_aa(rng, 1, swap)
        label = S_AA
    elif case == "aa_sized":
        t1, t2, order = _case_aa(rng, k, swap)
        label = S_AA
    elif case == "no_ratio":
        t1, t2, order = _case_no_ratio(rng, k, swap)
        label = NEITHER
    elif case == "no_ratio_same":
        t1, t2, order = _case_no_ratio(rng, 1, swap)
        label = NEITHER
    elif case == "too_few":
        t1, t2, order = _case_too_few(rng, rng.choice([1, 2, 3]), swap)
        label = NEITHER
    elif case == "ssa":
        t1, t2, order = _case_ssa(rng, 1, swap)
        label = NEITHER
    elif case == "ssa_ratio":
        t1, t2, order = _case_ssa(rng, 2, swap)
        label = NEITHER
    elif case == "aa_derived":
        t1, t2, order = _case_aa_derived(rng, True)
        label = S_AA
    else:  # aa_unmatched
        t1, t2, order = _case_aa_derived(rng, False)
        label = NEITHER
    return t1, t2, order, label, case


@generator(
    "f10c_congruence_criteria",
    level=8,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def f10c_congruence_criteria(rng, tier="easy"):
    """Name the congruence or similarity test that two described triangles satisfy, or say that none applies
    (see the module docstring for the answer space and the tiers)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {list(TIERS)}")
    names = rng.choice(_NAME_SETS)
    t1, t2, order, answer, case = _build_case(rng, tier)

    valid = _valid_labels(t1, t2)
    if answer == NEITHER:
        check(not valid, f"f10c_congruence_criteria: 'neither' keyed but the facts meet {sorted(valid)}")
    else:
        check(answer in valid, f"f10c_congruence_criteria: keyed {answer!r} but the facts meet {sorted(valid)}")
        others = {v for v in valid if v != answer}
        if answer in CONGRUENT_VALID:
            check(
                all(v in SIMILAR for v in others),
                f"f10c_congruence_criteria: two congruence tests met ({sorted(valid)})",
            )
        else:
            check(not others, f"f10c_congruence_criteria: two tests met ({sorted(valid)})")

    distractors = _distractors(rng, answer, valid, case)
    check(len(distractors) == 3, "f10c_congruence_criteria: fewer than three distractors available")
    check(
        not (set(distractors) & valid) and answer not in distractors and len(set(distractors)) == 3,
        "f10c_congruence_criteria: a distractor is met by the facts or repeats",
    )

    text = _render(rng, tier, t1, t2, order, names)
    notes = {d: _tag(answer, d, case) for d in distractors}
    working = [
        _part_summary(t1, t2),
        _WHY[answer],
        "Only the answer's own test is fully met by the stated facts.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        working=working,
        distractors=distractors,
        option_notes=notes,
    )


def _part_summary(t1: Tri, t2: Tri) -> str:
    k1, k2 = _known_angles(t1), _known_angles(t2)
    sides = [e for e in t1.sides if e in t2.sides]
    angs = [v for v in k1 if v in k2 and k1[v] == k2[v]]
    ratios = sorted({str(Fraction(t2.sides[e], t1.sides[e])) for e in sides})
    return (
        f"Sides stated in both triangles: {len(sides)} (side ratios {', '.join(ratios) or 'none'}); "
        f"corresponding angles known to be equal: {len(angs)}."
    )
