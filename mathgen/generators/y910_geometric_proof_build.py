"""Give the reason for each step of a short geometric proof (Year 10: formulate proofs of geometric facts).

A proof is a numbered list of statements and the learner supplies the REASON for each, which is the part of a proof
a computer can mark. The reasons are a numbered bank of eight (the ones the proof needs plus believable decoys such
as "corresponding angles" for "alternate angles"), shuffled for every question, so the numbers carry no meaning
from one question to the next. The learner types the number of the reason for step 1, step 2, ... in order. Guessing
a three-step proof from eight reasons is a 1 in 512 chance, and a five-step one is worse.

Ladder (reasoning escalation, not length alone):

* easy: three steps of angle chasing with parallel lines or crossing lines (co-interior angles from alternate
  angles; alternate angles from vertically opposite and corresponding angles; vertically opposite angles equal).
* medium: three or four steps that combine angle facts (the angle sum of a triangle from a parallel line, the
  exterior angle theorem, the apex angle of an isosceles triangle).
* hard: five steps that go through congruent triangles (base angles of an isosceles triangle, the diagonals of a
  parallelogram bisect each other, a point on a perpendicular bisector is equidistant from the ends), each needing
  the correct congruence test and the step that turns congruence into equal parts.
* exam: the same kind of proof, but each reason is a separate multiple-choice part (six or more reasons offered),
  followed by a number that uses the proved result.

Every proof is also checked numerically: the proof is built on random coordinates and every statement is evaluated.
"""

from __future__ import annotations

import math
import random

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer

TOPIC = "y910_geometric_proof_reasons"
TIERS = ("easy", "medium", "hard")

# reason key -> (text, class). Reasons of one class are the believable decoys for each other.
REASONS: dict[str, tuple[str, str]] = {
    "alt": ("Alternate angles on parallel lines are equal", "parallel"),
    "corr": ("Corresponding angles on parallel lines are equal", "parallel"),
    "coint": ("Co-interior angles on parallel lines add to 180°", "parallel"),
    "vert": ("Vertically opposite angles are equal", "lines"),
    "line": ("Angles on a straight line add to 180°", "lines"),
    "point": ("Angles at a point add to 360°", "lines"),
    "tri": ("The angles in a triangle add to 180°", "triangle"),
    "quad": ("The angles in a quadrilateral add to 360°", "triangle"),
    "ext": ("An exterior angle of a triangle equals the sum of the two opposite interior angles", "triangle"),
    "iso": ("Angles opposite equal sides of a triangle are equal", "triangle"),
    "sub": ("Combining the earlier steps: equal quantities can be substituted or subtracted", "combine"),
    "adj": ("A whole angle is the sum of the two parts it is made of", "combine"),
    "given": ("This equal length or angle is stated in the question", "fact"),
    "mid": ("Definition of a midpoint: it divides a line into two equal parts", "fact"),
    "common": ("A side shared by both triangles", "fact"),
    "ppar": ("Opposite sides of a parallelogram are equal", "fact"),
    "perp": ("Definition of perpendicular: the lines meet at right angles", "fact"),
    "sss": ("The triangles are congruent: three pairs of equal sides (SSS)", "test"),
    "sas": ("The triangles are congruent: two sides and the angle between them (SAS)", "test"),
    "asa": ("The triangles are congruent: two angles and the side between them (ASA)", "test"),
    "rhs": ("The triangles are congruent: right angle, hypotenuse and a side (RHS)", "test"),
    "cpct": ("Matching parts of congruent triangles are equal", "match"),
}
_NOTES = {
    ("parallel", "parallel"): "parallel-line-angle-reasons-confused",
    ("lines", "lines"): "straight-line-and-crossing-line-reasons-confused",
    ("parallel", "lines"): "parallel-line-reason-confused-with-a-line-fact",
    ("lines", "parallel"): "parallel-line-reason-confused-with-a-line-fact",
    ("triangle", "triangle"): "triangle-angle-reasons-confused",
    ("test", "test"): "congruence-tests-confused",
    ("test", "match"): "congruence-test-confused-with-matching-parts",
    ("match", "test"): "congruence-test-confused-with-matching-parts",
    ("fact", "fact"): "stated-fact-reasons-confused",
}
# decoys offered beyond the reasons a proof needs, per reason class
_DECOY_POOL = {
    "parallel": ["alt", "corr", "coint", "vert", "line"],
    "lines": ["vert", "line", "point", "alt", "coint"],
    "triangle": ["tri", "quad", "ext", "iso", "line"],
    "combine": ["sub", "adj"],
    "fact": ["given", "mid", "common", "ppar", "perp"],
    "test": ["sss", "sas", "asa", "rhs"],
    "match": ["cpct"],
}


def _note(true_key: str, wrong_key: str) -> str:
    a, b = REASONS[true_key][1], REASONS[wrong_key][1]
    return _NOTES.get((a, b)) or _NOTES.get((b, a)) or "reason-given-for-a-different-step"


# ---------------------------------------------------------------------------------------------------- geometry
def _ang(p, v, q) -> float:
    """The angle at v between vp and vq, in degrees."""
    a1 = math.atan2(p[1] - v[1], p[0] - v[0])
    a2 = math.atan2(q[1] - v[1], q[0] - v[0])
    d = abs(math.degrees(a1 - a2)) % 360
    return 360 - d if d > 180 else d


def _d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _eq(a, b) -> bool:
    return abs(a - b) < 1e-7


def _letters(rng, n: int) -> list[str]:
    pool = [c for c in "ABCDEFGHJKLMNPQRSTUVWXYZ"]
    return rng.sample(pool, n)


class Proof:
    def __init__(self, given: str, prove: str):
        self.given, self.prove = given, prove
        self.steps: list[tuple[str, str]] = []  # (statement, reason key)
        self.application: tuple[str, int, tuple[int, int], list[tuple[int, str]]] | None = (
            None  # prompt, value, range, wrong
        )

    def step(self, statement: str, reason: str, holds: bool) -> None:
        check(holds, f"proof step is false on the test coordinates: {statement}")
        check(reason in REASONS, f"unknown reason {reason!r}")
        self.steps.append((statement, reason))


# each template returns a Proof; coordinates are drawn at random and every statement is evaluated on them


def _t_coint(rng) -> Proof:
    P, Q, R, S, A, B = _letters(rng, 6)
    t = rng.uniform(0.3, 1.4) * rng.choice([1, -1])
    a, b = (0.0, 0.0), (t, -1.0)
    p, q, r, s = (-1.0, 0.0), (1.0, 0.0), (t - 1.0, -1.0), (t + 1.0, -1.0)
    pr = Proof(
        f"{P}{Q} is parallel to {R}{S}. The line {A}{B} crosses {P}{Q} at {A} and {R}{S} at {B}, with {R} and {P} on the same side of {A}{B}.",
        f"∠{P}{A}{B} + ∠{A}{B}{R} = 180°",
    )
    pr.step(f"∠{P}{A}{B} = ∠{A}{B}{S}", "alt", _eq(_ang(p, a, b), _ang(a, b, s)))
    pr.step(f"∠{A}{B}{S} + ∠{A}{B}{R} = 180°", "line", _eq(_ang(a, b, s) + _ang(a, b, r), 180))
    pr.step(f"∠{P}{A}{B} + ∠{A}{B}{R} = 180°", "sub", _eq(_ang(p, a, b) + _ang(a, b, r), 180))
    theta = rng.randrange(52, 128, 2)
    pr.application = (
        f"Given that ∠{P}{A}{B} = {theta}°, find ∠{A}{B}{R}.",
        180 - theta,
        (0, 180),
        [
            (theta, "co-interior-angles-taken-as-equal"),
            (abs(90 - theta), "co-interior-angles-taken-to-add-to-90"),
            (360 - theta - 100, "arithmetic-slip-in-the-final-step"),
        ],
    )
    return pr


def _t_vert(rng) -> Proof:
    A, B, C, D, E = _letters(rng, 5)
    t = rng.uniform(25, 75)
    e = (0.0, 0.0)
    a = (math.cos(math.radians(180 + t)), math.sin(math.radians(180 + t)))
    b = (-a[0], -a[1])
    c = (math.cos(math.radians(t + 70)), math.sin(math.radians(t + 70)))
    d = (-c[0], -c[1])
    pr = Proof(
        f"The straight lines {A}{B} and {C}{D} cross at {E}.",
        f"∠{A}{E}{C} = ∠{B}{E}{D}",
    )
    pr.step(f"∠{A}{E}{C} + ∠{C}{E}{B} = 180°", "line", _eq(_ang(a, e, c) + _ang(c, e, b), 180))
    pr.step(f"∠{C}{E}{B} + ∠{B}{E}{D} = 180°", "line", _eq(_ang(c, e, b) + _ang(b, e, d), 180))
    pr.step(f"∠{A}{E}{C} = ∠{B}{E}{D}", "sub", _eq(_ang(a, e, c), _ang(b, e, d)))
    theta = rng.randrange(35, 145, 5)
    pr.application = (
        f"Given that ∠{A}{E}{C} = {theta}°, find ∠{C}{E}{B}.",
        180 - theta,
        (0, 180),
        [
            (theta, "adjacent-angles-on-a-line-taken-as-equal"),
            (abs(90 - theta), "angles-on-a-line-taken-to-add-to-90"),
            (360 - theta, "angles-on-a-line-taken-to-add-to-360"),
        ],
    )
    return pr


def _t_alt(rng) -> Proof:
    P, Q, R, S, A, B, T = _letters(rng, 7)
    t = rng.uniform(0.3, 1.4) * rng.choice([1, -1])
    a, b = (0.0, 0.0), (t, -1.0)
    tt = (-t * 0.7, 0.7)  # T on the transversal above A
    p, q, s = (-1.0, 0.0), (1.0, 0.0), (t + 1.0, -1.0)
    pr = Proof(
        f"{P}{Q} is parallel to {R}{S}. The line {T}{B} crosses {P}{Q} at {A} and {R}{S} at {B}, with {T} above {P}{Q}.",
        f"∠{P}{A}{B} = ∠{A}{B}{S}",
    )
    pr.step(f"∠{P}{A}{B} = ∠{Q}{A}{T}", "vert", _eq(_ang(p, a, b), _ang(q, a, tt)))
    pr.step(f"∠{Q}{A}{T} = ∠{A}{B}{S}", "corr", _eq(_ang(q, a, tt), _ang(a, b, s)))
    pr.step(f"∠{P}{A}{B} = ∠{A}{B}{S}", "sub", _eq(_ang(p, a, b), _ang(a, b, s)))
    theta = rng.randrange(48, 132, 3)
    pr.application = (
        f"Given that ∠{Q}{A}{T} = {theta}°, find ∠{A}{B}{S}.",
        theta,
        (0, 180),
        [
            (180 - theta, "alternate-angles-taken-to-add-to-180"),
            (abs(90 - theta), "angles-taken-to-add-to-90"),
            (theta + 10, "arithmetic-slip-in-the-final-step"),
        ],
    )
    return pr


def _t_anglesum(rng) -> Proof:
    A, B, C, D, E = _letters(rng, 5)
    b, c = (0.0, 0.0), (rng.uniform(3, 6), 0.0)
    a = (rng.uniform(0.5, c[0] - 0.5), rng.uniform(1.5, 3.5))
    d, e = (a[0] - 1.5, a[1]), (a[0] + 1.5, a[1])
    pr = Proof(
        f"{A}{B}{C} is a triangle. The line {D}{E} passes through {A} and is parallel to {B}{C}, with {D} on the same side as {B}.",
        f"∠{A}{B}{C} + ∠{B}{A}{C} + ∠{A}{C}{B} = 180°",
    )
    pr.step(f"∠{D}{A}{B} = ∠{A}{B}{C}", "alt", _eq(_ang(d, a, b), _ang(a, b, c)))
    pr.step(f"∠{E}{A}{C} = ∠{A}{C}{B}", "alt", _eq(_ang(e, a, c), _ang(a, c, b)))
    pr.step(
        f"∠{D}{A}{B} + ∠{B}{A}{C} + ∠{E}{A}{C} = 180°", "line", _eq(_ang(d, a, b) + _ang(b, a, c) + _ang(e, a, c), 180)
    )
    pr.step(
        f"∠{A}{B}{C} + ∠{B}{A}{C} + ∠{A}{C}{B} = 180°", "sub", _eq(_ang(a, b, c) + _ang(b, a, c) + _ang(a, c, b), 180)
    )
    x, y = rng.randrange(35, 80, 1), rng.randrange(40, 85, 1)
    pr.application = (
        f"Given that ∠{A}{B}{C} = {x}° and ∠{A}{C}{B} = {y}°, find ∠{B}{A}{C}.",
        180 - x - y,
        (0, 180),
        [
            (x + y, "forgot-to-subtract-from-the-total"),
            (180 - x, "subtracted-only-one-angle"),
            (360 - x - y, "angle-sum-taken-as-360"),
        ],
    )
    return pr


def _t_exterior(rng) -> Proof:
    A, B, C, D = _letters(rng, 4)
    b, c = (0.0, 0.0), (rng.uniform(3, 5), 0.0)
    a = (rng.uniform(0.5, 4.5), rng.uniform(1.5, 3.5))
    d = (c[0] + 2.0, 0.0)
    pr = Proof(
        f"{A}{B}{C} is a triangle, and the side {B}{C} is extended to {D}.",
        f"∠{A}{C}{D} = ∠{B}{A}{C} + ∠{A}{B}{C}",
    )
    pr.step(
        f"∠{B}{A}{C} + ∠{A}{B}{C} + ∠{A}{C}{B} = 180°", "tri", _eq(_ang(b, a, c) + _ang(a, b, c) + _ang(a, c, b), 180)
    )
    pr.step(f"∠{A}{C}{B} + ∠{A}{C}{D} = 180°", "line", _eq(_ang(a, c, b) + _ang(a, c, d), 180))
    pr.step(f"∠{A}{C}{D} = ∠{B}{A}{C} + ∠{A}{B}{C}", "sub", _eq(_ang(a, c, d), _ang(b, a, c) + _ang(a, b, c)))
    x, y = rng.randrange(30, 75, 1), rng.randrange(35, 80, 1)
    pr.application = (
        f"Given that ∠{B}{A}{C} = {x}° and ∠{A}{B}{C} = {y}°, find ∠{A}{C}{D}.",
        x + y,
        (0, 180),
        [
            (180 - x - y, "interior-angle-found-instead-of-the-exterior-angle"),
            (180 - x, "subtracted-only-one-angle"),
            (abs(x - y), "subtracted-the-two-angles"),
        ],
    )
    return pr


def _t_apex(rng) -> Proof:
    A, B, C = _letters(rng, 3)
    half = rng.uniform(1.5, 3.0)
    b, c, a = (-half, 0.0), (half, 0.0), (0.0, rng.uniform(2, 5))
    pr = Proof(f"In triangle {A}{B}{C}, {A}{B} = {A}{C}.", f"∠{B}{A}{C} = 180° − 2 × ∠{A}{B}{C}")
    pr.step(f"∠{A}{B}{C} = ∠{A}{C}{B}", "iso", _eq(_ang(a, b, c), _ang(a, c, b)))
    pr.step(
        f"∠{B}{A}{C} + ∠{A}{B}{C} + ∠{A}{C}{B} = 180°", "tri", _eq(_ang(b, a, c) + _ang(a, b, c) + _ang(a, c, b), 180)
    )
    pr.step(f"∠{B}{A}{C} = 180° − 2 × ∠{A}{B}{C}", "sub", _eq(_ang(b, a, c), 180 - 2 * _ang(a, b, c)))
    x = rng.randrange(42, 78, 2)
    pr.application = (
        f"Given that ∠{A}{B}{C} = {x}°, find ∠{B}{A}{C}.",
        180 - 2 * x,
        (0, 180),
        [
            (180 - x, "subtracted-only-one-angle"),
            (2 * x, "doubled-the-base-angle-only"),
            (90 - x, "angle-sum-taken-as-90"),
        ],
    )
    return pr


def _t_iso(rng) -> Proof:
    A, B, C, M = _letters(rng, 4)
    half = rng.uniform(1.5, 3.0)
    b, c, a = (-half, 0.0), (half, 0.0), (0.0, rng.uniform(2, 5))
    m = (0.0, 0.0)
    pr = Proof(
        f"In triangle {A}{B}{C}, {A}{B} = {A}{C}, and {M} is the midpoint of {B}{C}.",
        f"∠{A}{B}{C} = ∠{A}{C}{B}",
    )
    pr.step(f"{A}{B} = {A}{C}", "given", _eq(_d(a, b), _d(a, c)))
    pr.step(f"{B}{M} = {M}{C}", "mid", _eq(_d(b, m), _d(m, c)))
    pr.step(f"{A}{M} = {A}{M}", "common", True)
    pr.step(f"△{A}{B}{M} ≅ △{A}{C}{M}", "sss", _eq(_d(a, b), _d(a, c)) and _eq(_d(b, m), _d(c, m)))
    pr.step(f"∠{A}{B}{C} = ∠{A}{C}{B}", "cpct", _eq(_ang(a, b, c), _ang(a, c, b)))
    k = rng.choice([(5, 12, 13), (3, 4, 5), (8, 15, 17), (6, 8, 10)])
    f = rng.choice([1, 2])
    leg, alt, hyp = k[0] * f, k[1] * f, k[2] * f
    pr.application = (
        f"Given that {A}{B} = {hyp} cm and {B}{C} = {2 * leg} cm, find {A}{M}, in centimetres.",
        alt,
        (0, 60),
        [
            (hyp - leg, "subtracted-instead-of-using-pythagoras"),
            (leg, "used-half-the-base-as-the-height"),
            (hyp - 2 * leg if hyp > 2 * leg else hyp + leg, "subtracted-instead-of-using-pythagoras"),
            (2 * leg, "used-the-whole-base-as-the-height"),
        ],
    )
    return pr


def _t_para(rng) -> Proof:
    A, B, C, D, E = _letters(rng, 5)
    a, b = (0.0, 0.0), (rng.uniform(3, 5), 0.0)
    shear = (rng.uniform(0.5, 1.5), rng.uniform(1.5, 3))
    d = shear
    c = (b[0] + shear[0], shear[1])
    e = ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2)
    pr = Proof(
        f"{A}{B}{C}{D} is a parallelogram. Its diagonals {A}{C} and {B}{D} cross at {E}.",
        f"{A}{E} = {E}{C}",
    )
    pr.step(f"∠{E}{A}{B} = ∠{E}{C}{D}", "alt", _eq(_ang(e, a, b), _ang(e, c, d)))
    pr.step(f"∠{E}{B}{A} = ∠{E}{D}{C}", "alt", _eq(_ang(e, b, a), _ang(e, d, c)))
    pr.step(f"{A}{B} = {C}{D}", "ppar", _eq(_d(a, b), _d(c, d)))
    pr.step(f"△{A}{B}{E} ≅ △{C}{D}{E}", "asa", _eq(_d(a, e), _d(c, e)))
    pr.step(f"{A}{E} = {E}{C}", "cpct", _eq(_d(a, e), _d(e, c)))
    ac = rng.randrange(10, 40, 2)
    pr.application = (
        f"Given that {A}{C} = {ac} cm, find {A}{E}, in centimetres.",
        ac // 2,
        (0, 40),
        [
            (ac, "took-the-whole-diagonal"),
            (ac // 4 if ac % 4 == 0 else ac // 2 + 2, "halved-twice"),
            (ac // 2 + 2, "arithmetic-slip-in-the-final-step"),
        ],
    )
    return pr


def _t_bisector(rng) -> Proof:
    A, B, M, P = _letters(rng, 4)
    half = rng.uniform(1.5, 3.0)
    a, b, m = (-half, 0.0), (half, 0.0), (0.0, 0.0)
    p = (0.0, rng.uniform(2, 5))
    pr = Proof(
        f"{M} is the midpoint of {A}{B}, and {P}{M} is perpendicular to {A}{B}.",
        f"{P}{A} = {P}{B}",
    )
    pr.step(f"{A}{M} = {M}{B}", "mid", _eq(_d(a, m), _d(m, b)))
    pr.step(f"∠{P}{M}{A} = ∠{P}{M}{B}", "perp", _eq(_ang(p, m, a), _ang(p, m, b)))
    pr.step(f"{P}{M} = {P}{M}", "common", True)
    pr.step(f"△{P}{M}{A} ≅ △{P}{M}{B}", "sas", _eq(_d(a, m), _d(b, m)) and _eq(_ang(p, m, a), _ang(p, m, b)))
    pr.step(f"{P}{A} = {P}{B}", "cpct", _eq(_d(p, a), _d(p, b)))
    k = rng.choice([(5, 12, 13), (3, 4, 5), (8, 15, 17), (6, 8, 10)])
    f = rng.choice([1, 2])
    leg, alt, hyp = k[0] * f, k[1] * f, k[2] * f
    pr.application = (
        f"Given that {A}{B} = {2 * leg} cm and {P}{M} = {alt} cm, find {P}{A}, in centimetres.",
        hyp,
        (0, 80),
        [
            (leg + alt, "added-instead-of-using-pythagoras"),
            (alt, "gave-a-leg-for-the-hypotenuse"),
            (2 * leg, "gave-a-leg-for-the-hypotenuse"),
        ],
    )
    return pr


_TEMPLATES = {
    "easy": [_t_coint, _t_vert, _t_alt],
    "medium": [_t_anglesum, _t_exterior, _t_apex],
    "hard": [_t_iso, _t_para, _t_bisector],
}


# ---------------------------------------------------------------------------------------------------- the bank
def _bank(rng, proof: Proof, size: int = 8) -> list[str]:
    """Reason keys offered: every one the proof uses, then decoys of the same classes until `size`."""
    used: list[str] = []
    for _s, k in proof.steps:
        if k not in used:
            used.append(k)
    pool: list[str] = []
    for k in used:
        for d in _DECOY_POOL[REASONS[k][1]]:
            if d not in used and d not in pool:
                pool.append(d)
    # a proof that uses a congruence test must not offer a second test that also fits the facts
    if "asa" in used:
        pool = [d for d in pool if d not in ("aas",)]
    if "sss" in used:
        pool = [d for d in pool if d not in ("aas",)]
    rng.shuffle(pool)
    # decoys from other classes if the pool is short
    rest = [k for k in REASONS if k not in used and k not in pool]
    rng.shuffle(rest)
    bank = used + pool + rest
    chosen = bank[:size]
    check(all(k in chosen for k in used), "bank is missing a needed reason")
    rng.shuffle(chosen)
    return chosen


def _wrong_sequences(rng, proof: Proof, bank: list[str], numbers: dict[str, int], n: int = 3):
    """Wrong answers, each changing one or two steps to a believable decoy. Returns Distractors."""
    key = [numbers[k] for _s, k in proof.steps]
    candidates: list[tuple[list[int], str]] = []
    for i, (_s, true) in enumerate(proof.steps):
        # a decoy of the same class first; the proof must still be wrong at step i
        decoys = [k for k in bank if k != true]
        decoys.sort(key=lambda k: (REASONS[k][1] != REASONS[true][1], rng.random()))
        for wrong in decoys[:2]:
            seq = list(key)
            seq[i] = numbers[wrong]
            candidates.append((seq, _note(true, wrong)))
    rng.shuffle(candidates)
    out: list[Distractor] = []
    seen = {tuple(key)}
    used_steps: set[int] = set()
    for seq, note in candidates:
        if tuple(seq) in seen:
            continue
        step = next(i for i, (a, b) in enumerate(zip(seq, key)) if a != b)
        if step in used_steps and len(out) < n - 1:
            continue
        seen.add(tuple(seq))
        used_steps.add(step)
        out.append(Distractor(", ".join(map(str, seq)), note, list(seq)))
        if len(out) == n:
            break
    return out


def _statement_lines(proof: Proof) -> str:
    return " ".join(f"Step {i}: {s}." for i, (s, _k) in enumerate(proof.steps, 1))


@generator(TOPIC, level=10, presets={t: {"tier": t} for t in TIERS})
def y910_geometric_proof_reasons(rng, tier="easy"):
    """Match the reasons to the steps of a short geometric proof; see the module docstring."""
    check(tier in _TEMPLATES, f"unknown tier {tier!r}")
    proof = rng.choice(_TEMPLATES[tier])(rng)
    bank = _bank(rng, proof)
    numbers = {k: i + 1 for i, k in enumerate(bank)}
    key = [numbers[k] for _s, k in proof.steps]
    distractors = _wrong_sequences(rng, proof, bank, numbers)
    check(len(distractors) >= 3, "too few wrong reason sequences")
    reasons = " ".join(f"{numbers[k]}. {REASONS[k][0]}." for k in sorted(bank, key=lambda k: numbers[k]))
    n = len(proof.steps)
    text = (
        f"A proof is being written. {proof.given} To prove: {proof.prove}. "
        f"{_statement_lines(proof)} "
        f"Reasons: {reasons} "
        f"Write the number of the reason for each of the {n} steps, in order, separated by commas."
    )
    answer_text = ", ".join(map(str, key))
    working = [f"Step {i}: {s}. Reason {numbers[k]}: {REASONS[k][0]}." for i, (s, k) in enumerate(proof.steps, 1)]
    return dict(
        text=text,
        answer=key,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
        answer_kind="ordered_list",
    )


# ---------------------------------------------------------------------------------------------------- exam
_EXAM_TEMPLATES = [_t_anglesum, _t_exterior, _t_iso, _t_para, _t_bisector]


@generator("exd_x_y910_geometric_proof", level=10, presets={"exam": dict(tier="exam")})
def exd_x_y910_geometric_proof(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the reason for three steps of a proof (one multiple-choice part each), then a number from it."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(100, "an exam proof"):
        proof = rng.choice(_EXAM_TEMPLATES)(rng)
        # steps whose reasons are all different, skipping 'given' and 'common' which prove nothing about the method
        steps = [(i, s, k) for i, (s, k) in enumerate(proof.steps, 1) if k not in ("given", "common")]
        seen: set[str] = set()
        picks = []
        for i, s, k in steps:
            if k not in seen:
                seen.add(k)
                picks.append((i, s, k))
        if len(picks) >= 3:
            break
    picks = picks[:3] if len(picks) == 3 else [picks[0], picks[len(picks) // 2], picks[-1]]
    n = len(proof.steps)
    text = (
        f"A proof is being written. {proof.given} To prove: {proof.prove}. {_statement_lines(proof)} "
        f"Each part asks for the reason for one step."
    )
    parts = []
    marks = [1, 1, 1]
    for (i, s, k), mk in zip(picks, marks):
        options = [x for x in REASONS if x != k]
        same = [x for x in options if REASONS[x][1] == REASONS[k][1]]
        other = [x for x in options if REASONS[x][1] != REASONS[k][1]]
        rng.shuffle(same)
        rng.shuffle(other)
        wrong = (same + other)[:5]
        if "asa" in (k, *[kk for _s, kk in proof.steps]):
            wrong = [x for x in wrong if x != "aas"]
        ds = [Distractor(REASONS[w][0], _note(k, w), REASONS[w][0]) for w in wrong]
        parts.append(
            dict(
                prompt=f"Which reason justifies step {i}: {s}?",
                answer_kind="mcq",
                answer=REASONS[k][0],
                answer_text=REASONS[k][0],
                answer_latex="plain:" + REASONS[k][0],
                marks=mk,
                working=[f"Step {i}: {s}. The reason is: {REASONS[k][0]}."],
                distractors=ds,
            )
        )
    prompt, value, (low, high), wrong_values = proof.application
    value_text = str(int(value))
    parts.append(
        dict(
            prompt=f"Use the result of the proof. {prompt}",
            answer_kind="number",
            answer=int(value),
            answer_text=value_text,
            answer_latex=latex_answer(value_text),
            marks=2,
            sensible_range=(low, high),
            working=[f"The proved result gives the answer {value_text}."],
            distractors=_number_distractors(int(value), wrong_values),
        )
    )
    return dict(text=text, parts=parts)


def _number_distractors(value: int, wrong_values):
    """Tagged wrong whole numbers from the errors the proved result guards against."""
    out = []
    seen = {str(value)}
    for v, note in wrong_values:
        if v <= 0 or str(v) in seen:
            continue
        seen.add(str(v))
        out.append(Distractor(str(v), note, v))
    for v in (value + 10, value - 10, value + 5, value - 5, value + 20):
        if len(out) >= 3:
            break
        if v > 0 and str(v) not in seen:
            seen.add(str(v))
            out.append(Distractor(str(v), "arithmetic-slip-in-the-final-step", v))
    return out[:3]
