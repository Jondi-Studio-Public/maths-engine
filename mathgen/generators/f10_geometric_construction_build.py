"""Following a stated geometric construction algorithm after one step is refined.

- ``f10_geometric_construction_algorithm`` (``f10-geometric-construction-algorithm``,
  AC9M9SP03 and AC9M10SP03, domain ``geometry``, answer: ``number``) -- a
  construction is given as numbered steps, one step is then REFINED (its
  setting changed, or the step replaced), and the learner finds the length or
  angle the refined construction produces. The learner follows and reasons
  about the algorithm; they do not author one, and nothing is asked to be
  explained or proved.

Three tiers, each a different reasoning shape:

- easy: one construction, one theorem. Either the arc-and-compass perpendicular
  bisector (the compass setting is refined; find the distance from the arcs'
  meeting point to the midpoint) or a point marked on a semicircle (a chord
  length is refined; find the other chord, by the angle-in-a-semicircle
  theorem). One Pythagorean step on integer sides.
- medium: an angle algorithm. Rays are built from OA by named angle steps
  (equilateral-triangle arcs turn 60 degrees, a perpendicular turns 90 degrees,
  bisecting halves the angle so far); one step is replaced; find the final angle.
  No Pythagoras: the reasoning is tracing how a replaced step changes what
  every later step starts from.
- hard: a five-step construction (perpendicular bisector, a point marked on it,
  a join) containing a compass setting that does NOT affect the result. The
  refinement changes the marked distance, the segment length, or only the
  decoy compass setting; the learner must work out which steps feed the length
  asked for. Integer sides by construction (Pythagorean triples).

Every answer is an integer, every distractor is the result of one named error,
and the answer is re-derived from the RENDERED text (a regex parse, then an
integer square root), independently of how construction chose the numbers.
"""

from __future__ import annotations

import math
import re

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    pick_distractors,
)

# -- misconception ids (must match the skill YAML) ---------------------------------
M_IGNORES = "ignores-the-refinement"
M_REPORTS_SETTING = "reports-the-compass-setting"
M_REPORTS_OLD_SETTING = "reports-the-original-compass-setting"
M_SUBTRACTS = "subtracts-instead-of-pythagoras"
M_INSERTS = "adds-the-refinement-as-an-extra-step"
M_STOPS = "stops-at-the-refined-step"
M_REVERSED = "applies-the-steps-out-of-order"
M_ADDS_LEGS = "adds-the-two-legs"
M_REPORTS_LEG = "reports-one-leg"
M_REPORTS_CHORD = "reports-the-given-chord"

# -- integer right triangles -------------------------------------------------------


def _build_triples(limit: int):
    out = []
    for a in range(3, limit):
        for b in range(a + 1, limit):
            c2 = a * a + b * b
            c = math.isqrt(c2)
            if c <= limit and c * c == c2:
                out.append((a, b, c))
    return tuple(out)


_TRIPLES = _build_triples(100)
# For each leg value x: the (other leg, hypotenuse) pairs.
_BY_LEG: dict[int, list[tuple[int, int]]] = {}
for _a, _b, _c in _TRIPLES:
    _BY_LEG.setdefault(_a, []).append((_b, _c))
    _BY_LEG.setdefault(_b, []).append((_a, _c))
# For each hypotenuse c: the (leg, other leg) pairs, each leg listed once per ordering.
_BY_HYP: dict[int, list[tuple[int, int]]] = {}
for _a, _b, _c in _TRIPLES:
    _BY_HYP.setdefault(_c, []).append((_a, _b))
    _BY_HYP.setdefault(_c, []).append((_b, _a))

# Half-lengths with at least two partners, so a refinement changes the answer.
_HALF_LENGTHS = tuple(sorted(x for x, v in _BY_LEG.items() if len(v) >= 2 and 5 <= x <= 40))
_HYPOTENUSES = tuple(sorted(c for c, v in _BY_HYP.items() if len(v) >= 4 and c <= 100))

_LETTERS = (
    {"a": "A", "b": "B", "p": "P", "m": "M", "c": "C"},
    {"a": "X", "b": "Y", "p": "Q", "m": "N", "c": "R"},
    {"a": "D", "b": "E", "p": "S", "m": "T", "c": "U"},
    {"a": "K", "b": "L", "p": "F", "m": "G", "c": "H"},
)


def _pick_distractors(rng, answer: int, pairs: list[tuple[str, int]]):
    """Dedupe to distinct values that differ from the answer; returns (strings, notes), or None if fewer than 3."""
    seen, keep = {answer}, []
    for note, value in pairs:
        if value is None or value <= 0 or value in seen:
            continue
        seen.add(value)
        keep.append((note, value))
    if len(keep) < 3:
        return None
    if len(keep) > 3:  # show three of the four, so every misconception is exercised across seeds
        keep = [keep[i] for i in sorted(rng.sample(range(len(keep)), 3))]
    check_pairwise_distinct(answer, *[v for _, v in keep], msg="f10_geometric_construction: distractor collision")
    strings = pick_distractors(str(answer), [str(v) for _, v in keep])
    notes = {str(v): n for n, v in keep if str(v) in strings}
    return strings, notes


# -- easy ----------------------------------------------------------------------------

_EASY_Q_PERP = (
    "After the refinement, how far is {p} from {m}, in cm?",
    "How far is {p} from {m} once the refinement is made, in cm?",
    "Find the distance from {p} to {m} after the refinement, in cm.",
)
_EASY_Q_THALES = (
    "After the refinement, how long is {b}{c}, in cm?",
    "How long is {b}{c} once the refinement is made, in cm?",
    "Find the length of {b}{c} after the refinement, in cm.",
)


def _easy_perp(rng):
    letters = rng.choice(_LETTERS)
    for _ in range(100):
        h = rng.choice(_HALF_LENGTHS)
        (m1, r1), (m2, r2) = rng.sample(_BY_LEG[h], 2)
        # compasses: r1 first, refined to r2
        cands = [
            (M_IGNORES, m1),
            (M_REPORTS_SETTING, r2),
            (M_SUBTRACTS, r2 - h),
            (M_REPORTS_LEG, h),
        ]
        picked = _pick_distractors(rng, m2, cands)
        if picked and m2 != h:
            break
    else:  # pragma: no cover
        raise RuntimeError("f10_geometric_construction easy perpendicular: no draw")
    A, B, P, M = letters["a"], letters["b"], letters["p"], letters["m"]
    text = (
        f"A construction algorithm finds the perpendicular bisector of a segment. "
        f"Step 1: Draw segment {A}{B}, {2 * h} cm long. "
        f"Step 2: Set the compasses to {r1} cm. "
        f"Step 3: Draw an arc centred on {A} and an arc centred on {B}, meeting at point {P} above {A}{B}. "
        f"Step 4: Mark the midpoint {M} of {A}{B} and join {P}{M}. "
        f"A refinement changes Step 2 so that the compasses are set to {r2} cm instead. "
        + rng.choice(_EASY_Q_PERP).format(p=P, m=M)
    )
    working = [
        f"{P} is {r2} cm from both {A} and {B}, and {M} is the midpoint of {A}{B}, so {A}{M} = {h} cm "
        f"and angle {P}{M}{A} is a right angle.",
        f"{P}{M}^2 = {r2}^2 - {h}^2 = {r2 * r2} - {h * h} = {m2 * m2}.",
        f"{P}{M} = {m2} cm.",
    ]
    return text, m2, picked, working


def _easy_thales(rng):
    letters = rng.choice(_LETTERS)
    for _ in range(100):
        d = rng.choice(_HYPOTENUSES)
        legs = sorted({x for pair in _BY_HYP[d] for x in pair})
        a1, a2 = rng.sample(legs, 2)
        partner = {x: y for x, y in _BY_HYP[d]}
        answer = partner[a2]
        cands = [
            (M_IGNORES, partner[a1]),
            (M_REPORTS_SETTING, a2),
            (M_SUBTRACTS, d - a2),
            (M_REPORTS_CHORD, a1),
        ]
        picked = _pick_distractors(rng, answer, cands)
        if picked:
            break
    else:  # pragma: no cover
        raise RuntimeError("f10_geometric_construction easy semicircle: no draw")
    A, B, C = letters["a"], letters["b"], letters["c"]
    text = (
        f"A construction algorithm marks a point on a semicircle. "
        f"Step 1: Draw segment {A}{B}, {d} cm long, and draw a semicircle with {A}{B} as its diameter. "
        f"Step 2: Set the compasses to {a1} cm. "
        f"Step 3: Draw an arc centred on {A} that cuts the semicircle at point {C}. "
        f"Step 4: Join {B}{C}. "
        f"A refinement changes Step 2 so that the compasses are set to {a2} cm instead. "
        + rng.choice(_EASY_Q_THALES).format(b=B, c=C)
    )
    working = [
        f"{C} is on the semicircle with diameter {A}{B}, so angle {A}{C}{B} is a right angle (angle in a semicircle).",
        f"{A}{C} = {a2} cm after the refinement, so {B}{C}^2 = {d}^2 - {a2}^2 = {d * d} - {a2 * a2} = {answer * answer}.",
        f"{B}{C} = {answer} cm.",
    ]
    return text, answer, picked, working


_EASY_PERP_RE = re.compile(
    r"Draw segment (\w)(\w), (\d+) cm long\. Step 2: Set the compasses to (\d+) cm\. .*?"
    r"set to (\d+) cm instead\."
)
_EASY_THALES_RE = re.compile(
    r"Draw segment (\w)(\w), (\d+) cm long, and draw a semicircle.*?Set the compasses to (\d+) cm\. "
    r".*?set to (\d+) cm instead\."
)


def _check_easy(text: str, answer: int) -> None:
    if "perpendicular bisector" in text:
        m = _EASY_PERP_RE.search(text)
        check(m is not None, "f10_geometric_construction: could not parse the easy perpendicular text")
        length, new_r = int(m.group(3)), int(m.group(5))
        half = length // 2
        check(length % 2 == 0 and new_r > half, "f10_geometric_construction: arcs would not meet")
        got = math.isqrt(new_r * new_r - half * half)
        check(got * got == new_r * new_r - half * half, "f10_geometric_construction: P to M is not whole")
    else:
        m = _EASY_THALES_RE.search(text)
        check(m is not None, "f10_geometric_construction: could not parse the easy semicircle text")
        diameter, new_leg = int(m.group(3)), int(m.group(5))
        check(new_leg < diameter, "f10_geometric_construction: arc cannot reach the semicircle")
        rest = diameter * diameter - new_leg * new_leg
        got = math.isqrt(rest)
        check(got * got == rest, "f10_geometric_construction: chord is not whole")
    check(got == answer, "f10_geometric_construction: easy re-derivation disagrees with answer_text")


# -- medium: angle algorithm ---------------------------------------------------------

_START = {
    30: "Construct ray OB so that angle AOB is 30 degrees (an equilateral-triangle angle, bisected).",
    45: "Construct ray OB so that angle AOB is 45 degrees (a right angle, bisected).",
    60: "Construct ray OB with equilateral-triangle arcs so that angle AOB is 60 degrees.",
    90: "Construct ray OB perpendicular to OA, so that angle AOB is 90 degrees.",
    120: "Construct ray OB with two equilateral-triangle steps, so that angle AOB is 120 degrees.",
    150: "Construct ray OB so that angle AOB is 150 degrees (a right angle plus an equilateral-triangle angle).",
}
_OPS = {
    "bisect": "Bisect the angle between OA and the latest ray, and draw the new ray along the bisector.",
    "double": "Copy the angle between OA and the latest ray on to the far side of the latest ray, and draw the new ray.",
    "add30": "From the latest ray, turn a further 30 degrees (an equilateral-triangle angle, bisected), and draw the new ray.",
    "add60": "From the latest ray, use equilateral-triangle arcs to turn a further 60 degrees, and draw the new ray.",
    "add90": "From the latest ray, construct a perpendicular to turn a further 90 degrees, and draw the new ray.",
    "back60": "From the latest ray, use equilateral-triangle arcs to turn 60 degrees back towards OA, and draw the new ray.",
}
_OP_BY_TEXT = {v: k for k, v in _OPS.items()}
_START_BY_TEXT = {v: k for k, v in _START.items()}
_OP_FN = {
    "bisect": lambda a: a / 2,
    "double": lambda a: a * 2,
    "add30": lambda a: a + 30,
    "add60": lambda a: a + 60,
    "add90": lambda a: a + 90,
    "back60": lambda a: a - 60,
}


def _apply(angle: int, op: str):
    """The angle from OA to the latest ray after one op, or None unless a whole number of degrees in (0, 180]."""
    new = _OP_FN[op](angle)
    if new != int(new) or not 0 < new <= 180:
        return None
    return int(new)


def _run(start: int, ops):
    angle, trail = start, [start]
    for op in ops:
        angle = _apply(angle, op)
        if angle is None:
            return None, None
        trail.append(angle)
    return angle, trail


def _medium(rng):
    for _ in range(500):
        start = rng.choice(list(_START))
        ops = [rng.choice(list(_OPS)) for _ in range(3)]
        k = rng.randrange(3)  # which op (0-based among the three) the refinement replaces
        new_op = rng.choice([o for o in _OPS if o != ops[k]])
        refined = list(ops)
        refined[k] = new_op
        orig, _ = _run(start, ops)
        answer, trail = _run(start, refined)
        if orig is None or answer is None:
            continue
        inserted_ops = ops[:k] + [new_op] + ops[k:]
        inserted, _ = _run(start, inserted_ops)
        stopped = trail[k + 1]  # after the refined step
        reordered, _ = _run(start, list(reversed(refined)))
        picked = _pick_distractors(
            rng,
            answer,
            [
                (M_IGNORES, orig),
                (M_INSERTS, inserted),
                (M_STOPS, stopped),
                (M_REVERSED, reordered),
            ],
        )
        if picked:
            break
    else:  # pragma: no cover
        raise RuntimeError("f10_geometric_construction medium: no draw")
    steps = [_START[start]] + [_OPS[o] for o in ops]
    body = " ".join(f"Step {i}: {s}" for i, s in enumerate(steps, start=1))
    text = (
        "A construction algorithm builds angles at O by turning anticlockwise from ray OA. "
        f"{body} "
        f"A refinement replaces Step {k + 2} with: {_OPS[new_op]} "
        "After the refinement, what is the size of the angle between OA and the final ray, in degrees?"
    )
    names = {
        "bisect": "halve it",
        "double": "double it",
        "add30": "add 30",
        "add60": "add 60",
        "add90": "add 90",
        "back60": "subtract 60",
    }
    working = [
        f"Step 1 gives {trail[0]} degrees.",
        *[
            f"Step {i + 2} ({names[o]}): {trail[i]} -> {trail[i + 1]} degrees."
            + (" (the refined step)" if i == k else "")
            for i, o in enumerate(refined)
        ],
        f"The final angle is {answer} degrees.",
    ]
    return text, answer, picked, working


_MED_RE = re.compile(
    r"Step 1: (.*?) Step 2: (.*?) Step 3: (.*?) Step 4: (.*?) A refinement replaces Step (\d) with: (.*?) After"
)


def _check_medium(text: str, answer: int) -> None:
    m = _MED_RE.search(text)
    check(m is not None, "f10_geometric_construction: could not parse the medium text")
    start = _START_BY_TEXT[m.group(1)]
    ops = [_OP_BY_TEXT[m.group(i)] for i in (2, 3, 4)]
    ops[int(m.group(5)) - 2] = _OP_BY_TEXT[m.group(6)]
    angle = start
    for op in ops:
        angle = _OP_FN[op](angle)
    check(angle == answer, "f10_geometric_construction: medium re-derivation disagrees with answer_text")


# -- hard: propagation with a decoy ---------------------------------------------------

_HARD_Q = (
    "After the refinement, how long is {a}{c}, in cm?",
    "How long is {a}{c} once the refinement is made, in cm?",
    "Find the length of {a}{c} after the refinement, in cm.",
)


def _hard(rng):
    letters = rng.choice(_LETTERS)
    for _ in range(300):
        target = rng.choice(("mark", "segment", "compass"))
        if target == "mark":
            h = rng.choice(_HALF_LENGTHS)
            (d1, c1), (d2, c2) = rng.sample(_BY_LEG[h], 2)
            seg_half, mark, answer = h, d2, c2
            r = rng.randint(max(h, d1) + 1, max(h, d1) + 15)
            r2 = None
            cands = [(M_IGNORES, c1), (M_ADDS_LEGS, h + d2), (M_REPORTS_SETTING, r), (M_REPORTS_LEG, d2)]
            L1 = L2 = 2 * h
            d_first = d1
        elif target == "segment":
            d = rng.choice(_HALF_LENGTHS)
            (h1, c1), (h2, c2) = rng.sample(_BY_LEG[d], 2)
            seg_half, mark, answer = h2, d, c2
            r = rng.randint(max(h1, h2) + 1, max(h1, h2) + 15)
            r2 = None
            cands = [(M_IGNORES, c1), (M_ADDS_LEGS, h2 + d), (M_REPORTS_SETTING, r), (M_REPORTS_LEG, d)]
            L1, L2 = 2 * h1, 2 * h2
            d_first = d
        else:
            h = rng.choice(_HALF_LENGTHS)
            d, c = rng.choice(_BY_LEG[h])
            seg_half, mark, answer = h, d, c
            r = rng.randint(h + 1, h + 15)
            r2 = rng.randint(h + 1, h + 25)
            if r2 == r:
                continue
            cands = [
                (M_REPORTS_SETTING, r2),
                (M_REPORTS_OLD_SETTING, r),
                (M_ADDS_LEGS, h + d),
                (M_REPORTS_LEG, d),
            ]
            L1 = L2 = 2 * h
            d_first = d
        picked = _pick_distractors(rng, answer, cands)
        if picked:
            break
    else:  # pragma: no cover
        raise RuntimeError("f10_geometric_construction hard: no draw")
    A, B, P, M, C = letters["a"], letters["b"], letters["p"], letters["m"], letters["c"]
    if target == "mark":
        refinement = f"Step 4 so that {M}{C} = {mark} cm"
    elif target == "segment":
        refinement = f"Step 1 so that {A}{B} is {L2} cm long"
    else:
        refinement = f"Step 2 so that the compasses are set to {r2} cm"
    text = (
        f"A construction algorithm has five steps. "
        f"Step 1: Draw segment {A}{B}, {L1} cm long. "
        f"Step 2: Set the compasses to {r} cm and draw an arc centred on {A} and an arc centred on {B}, "
        f"meeting at point {P} above {A}{B}. "
        f"Step 3: Mark the midpoint {M} of {A}{B} and draw line {P}{M}. "
        f"Step 4: On line {P}{M}, mark point {C} above {A}{B} with {M}{C} = {d_first} cm. "
        f"Step 5: Join {A} to {C}. "
        f"A refinement changes {refinement}. " + rng.choice(_HARD_Q).format(a=A, c=C)
    )
    if target == "compass":
        working = [
            f"Only Steps 1, 3, 4 and 5 decide the length of {A}{C}: the arcs just locate {P} and the line {P}{M}.",
            f"{A}{M} = {seg_half} cm and {M}{C} = {mark} cm, with a right angle at {M}.",
            f"{A}{C}^2 = {seg_half}^2 + {mark}^2 = {answer * answer}, so {A}{C} = {answer} cm (unchanged by the refinement).",
        ]
    else:
        working = [
            f"After the refinement {A}{M} = {seg_half} cm and {M}{C} = {mark} cm, and {P}{M} is perpendicular to {A}{B}.",
            f"{A}{C}^2 = {seg_half}^2 + {mark}^2 = {seg_half * seg_half} + {mark * mark} = {answer * answer}.",
            f"{A}{C} = {answer} cm.",
        ]
    return text, answer, picked, working


_HARD_RE = re.compile(
    r"Step 1: Draw segment (\w)(\w), (\d+) cm long\. Step 2: Set the compasses to (\d+) cm .*?"
    r"Step 4: On line \w\w, mark point (\w) above \w\w with \w\w = (\d+) cm\. "
    r"Step 5: Join \w to \w\. A refinement changes (Step 4 so that \w\w = (\d+) cm|"
    r"Step 1 so that \w\w is (\d+) cm long|Step 2 so that the compasses are set to (\d+) cm)\."
)


def _check_hard(text: str, answer: int) -> None:
    m = _HARD_RE.search(text)
    check(m is not None, "f10_geometric_construction: could not parse the hard text")
    length, compass, mark = int(m.group(3)), int(m.group(4)), int(m.group(6))
    if m.group(8) is not None:  # Step 4 refined
        mark = int(m.group(8))
    elif m.group(9) is not None:  # Step 1 refined
        length = int(m.group(9))
    else:  # Step 2 refined -- the decoy
        compass = int(m.group(10))
    check(length % 2 == 0 and compass * 2 > length, "f10_geometric_construction: arcs would not meet")
    sq = (length // 2) ** 2 + mark * mark
    got = math.isqrt(sq)
    check(got * got == sq, "f10_geometric_construction: AC is not whole")
    check(got == answer, "f10_geometric_construction: hard re-derivation disagrees with answer_text")


# -- the generator ---------------------------------------------------------------------


@generator(
    "f10_geometric_construction_algorithm",
    level=9,
    presets={
        "easy": dict(variant="single_theorem"),
        "medium": dict(variant="angle_steps"),
        "hard": dict(variant="propagation"),
    },
)
def f10_geometric_construction_algorithm(rng, variant="single_theorem"):
    if variant == "single_theorem":
        text, answer, picked, working = (_easy_perp if rng.random() < 0.5 else _easy_thales)(rng)
        _check_easy(text, answer)
    elif variant == "angle_steps":
        text, answer, picked, working = _medium(rng)
        _check_medium(text, answer)
    elif variant == "propagation":
        text, answer, picked, working = _hard(rng)
        _check_hard(text, answer)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    distractors, notes = picked
    answer_text = str(answer)
    return {
        "text": text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "option_notes": notes,
        "working": working,
    }
