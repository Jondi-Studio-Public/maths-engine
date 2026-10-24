"""Practical right- and non-right-angled triangle trigonometry
(``senior-solve-practical-problems-involving-the-trigonometry-of-right``, ACMGM037, domain trigonometry,
answer: number).

Skill ``name``: "Solve practical problems involving the
trigonometry of right-angled and non-right-angled triangles, including
problems involving angles of elevation and depression and the use of
bearings in navigation".

Five shipping skills already cover every SINGLE-TRIANGLE case this name could mean: ``right-triangle-angle``
(one right triangle, inverse trig), ``sine-rule``/``cosine-rule`` (one non-right triangle, one rule), and the
elevation/bearings skills' own hard/exam tiers, which chain two triangles but always reduce to tan/Pythagoras
alone (their two triangles are always collinear or joined by an exact 90-degree turn). The kernel this skill
alone owns is a GENUINELY two-step chain: a ground triangle PQT that is NOT reducible to a right triangle
(solved by the sine or cosine rule) feeding its derived side into a SECOND, right-angled elevation
triangle -- a chain none of those neighbours contains.

Geometry (every tier): P and Q are two points on level ground; T is the
base of a tower/flagpole/building of unknown height. The ground triangle
PQT has PQ = d, angle QPT = phi (at P), angle PQT = psi (at Q); these three
values (ASA) minimally and fully determine the triangle -- the third angle
(at T) and the two remaining sides are all derived, never independently
stated (an overdetermined figure is a defect).

- easy (``direct_angles``): d, phi, psi and the elevation angle theta from
  P are all given directly. PT is found via the sine rule, then H = PT
  tan(theta).
- medium (``swap_orientation``): the SAME chain, but the elevation is taken
  from Q instead of P, so the learner must derive QT (paired with phi, the
  angle at P) rather than PT (paired with psi, the angle at Q) -- a
  genuinely different sine-rule pairing, not a relabelling.
- hard (``cosine_rule_sas``): a different ground-triangle configuration --
  two sides PQ = d, QT = e and the INCLUDED angle at Q (psi) are given
  (SAS), so PT is found via the COSINE rule instead, then the same
  elevation chain from P.
- exam (``bearings_ground_triangle``, stages: [exam]): the interior angle
  at P (phi) is not given directly -- instead two bearings (of Q, and of
  the tower's base T) are given as seen from P, and phi must first be
  derived as the bearing difference (handling the 0/360 wraparound
  explicitly), before the same sine-rule-then-elevation chain as easy.

Every tier's in-generator check is a genuinely different equation route
from its own construction (see the `_verify_*` docstrings below), and the
companion oracle parses
only the rendered `question.text`/`answer_text` and never imports this
module.
"""

from __future__ import annotations

import math

from ..core import (
    check,
    check_distinct,
    generator,
    phrase,
    pick_distractors,
    rounds_unambiguously,
    stated_precision_tolerance,
)

_RESAMPLE_LIMIT = 400

# Unrounded-value safety margin for a misconception candidate against the
# true (unrounded) answer -- comfortably larger than the real grader's own
# 1-d.p. tolerance (0.05) so double-rounding (unrounded draw -> 1 d.p.
# display -> grader comparison) can never let a candidate collide.
_MIN_GAP = 0.06

# Ground-triangle angle pool for phi (at P) and psi (at Q). Bounded well inside the required
# [15, 165] figure-sensibility band: every one of phi, psi and the derived angle at T is kept in
# [20, 140] by the rejection loop below.
PHI_POOL = list(range(25, 111))
PSI_POOL = list(range(25, 111))

# Elevation/depression angle pool, matching the existing elevation skill's
# own convention: excludes 45 (tan(45) = 1 makes height == leg exactly, a
# guaranteed trailing-zero answer at 1 d.p.).
THETA_POOL = [t for t in range(15, 76) if t != 45]

D_LO, D_HI = 20, 150  # ground baseline PQ, metres
E_LO, E_HI = 20, 150  # second ground side QT (hard tier only), metres

_H_LO, _H_HI = 5.0, 300.0  # plausible tower/flagpole/building height, metres


def _round1(v: float) -> str:
    return f"{v:.1f}"


def _valid_ground_triangle(phi: int, psi: int) -> bool:
    angle_t = 180 - phi - psi
    return 20 <= phi <= 140 and 20 <= psi <= 140 and 20 <= angle_t <= 140


def _far_enough(candidate: float, answer: float, min_gap: float = _MIN_GAP) -> bool:
    return abs(candidate - answer) > min_gap


def _interior_from_bearings(bearing_q: int, bearing_t: int) -> int:
    """The interior angle QPT at P from the two bearings taken at P, via
    modular difference then supplement -- the CONSTRUCTION route. Bearings
    wrap at 0/360; the interior angle between two rays is never more than
    180, so a raw difference over 180 must be replaced by its supplement."""
    diff = (bearing_t - bearing_q) % 360
    return diff if diff <= 180 else 360 - diff


# ---------------------------------------------------------------------------
# Independent checks -- each a genuinely different equation route from the
# construction it verifies (see each docstring).
# ---------------------------------------------------------------------------


def _verify_pt_via_qt_cosine(d: float, phi: float, psi: float, angle_t: float, pt: float) -> float:
    """Independent check for PT (easy and exam tiers): derive QT via the
    sine rule using phi -- the OTHER angle from psi, which the direct
    construction used -- then verify PT via the cosine rule at vertex Q
    using PQ, QT and psi. A transcription slip in the direct sine-rule
    division (construction) would not also appear in this cosine-rule
    route, which never divides by sin(psi) at all."""
    qt = d * math.sin(math.radians(phi)) / math.sin(math.radians(angle_t))
    pt_check = math.sqrt(d * d + qt * qt - 2 * d * qt * math.cos(math.radians(psi)))
    check(
        abs(pt_check - pt) < 1e-6,
        f"practical_triangle: cosine-rule cross-check for PT disagrees "
        f"({pt_check} vs {pt}; d={d}, phi={phi}, psi={psi})",
    )
    return pt_check


def _verify_qt_via_pt_cosine(d: float, phi: float, psi: float, angle_t: float, qt: float) -> float:
    """Symmetric independent check for QT (medium tier): derive PT via the
    sine rule using psi, then verify QT via the cosine rule at vertex P
    using PQ, PT and phi."""
    pt = d * math.sin(math.radians(psi)) / math.sin(math.radians(angle_t))
    qt_check = math.sqrt(d * d + pt * pt - 2 * d * pt * math.cos(math.radians(phi)))
    check(
        abs(qt_check - qt) < 1e-6,
        f"practical_triangle: cosine-rule cross-check for QT disagrees "
        f"({qt_check} vs {qt}; d={d}, phi={phi}, psi={psi})",
    )
    return qt_check


def _verify_cosine_sas(d: float, e: float, psi: float, pt: float) -> None:
    """Independent check for the hard tier's cosine-rule (SAS) construction:
    a right-triangle decomposition (drop a perpendicular from T onto line
    PQ; the foot is e*cos(psi) from Q, at height e*sin(psi)) instead of the
    law-of-cosines formula used at construction. This never calls cos(psi)
    inside a subtraction-from-(d^2+e^2) expression, so a sign or factor
    slip in the construction formula would not survive here."""
    foot = e * math.cos(math.radians(psi))
    height = e * math.sin(math.radians(psi))
    pt_check = math.hypot(d - foot, height)
    check(
        abs(pt_check - pt) < 1e-6,
        f"practical_triangle: right-triangle decomposition disagrees with "
        f"cosine rule ({pt_check} vs {pt}; d={d}, e={e}, psi={psi})",
    )


def _verify_bearing_interior(bearing_q: int, bearing_t: int, phi: float) -> None:
    """Independent check for the exam tier's bearing-derived interior angle:
    unit compass vectors' dot product + acos, instead of the modular
    difference-then-supplement used at construction. This route never
    computes a modular difference at all, so it independently confirms the
    wraparound case (bearing_t - bearing_q crossing 0/360) was handled
    correctly."""
    vq = (math.sin(math.radians(bearing_q)), math.cos(math.radians(bearing_q)))
    vt = (math.sin(math.radians(bearing_t)), math.cos(math.radians(bearing_t)))
    dot = max(-1.0, min(1.0, vq[0] * vt[0] + vq[1] * vt[1]))
    phi_check = math.degrees(math.acos(dot))
    check(
        abs(phi_check - phi) < 1e-6,
        f"practical_triangle: dot-product bearing check disagrees with "
        f"modular difference ({phi_check} vs {phi}; bq={bearing_q}, bt={bearing_t})",
    )


def _verify_elevation(leg: float, theta_deg: float, height: float) -> None:
    """Independent elevation check: recompute via the hypotenuse (leg /
    cos(theta), then hyp * sin(theta)) instead of the direct
    leg * tan(theta) multiplication used at construction."""
    hyp = leg / math.cos(math.radians(theta_deg))
    h_check = hyp * math.sin(math.radians(theta_deg))
    check(
        abs(h_check - height) < 1e-6,
        f"practical_triangle: sin/cos elevation cross-check disagrees "
        f"({h_check} vs {height}; leg={leg}, theta={theta_deg})",
    )


# ---------------------------------------------------------------------------
# Draw + reject loops, one per tier. Each returns a plain dict once every
# plausibility guard (no sliver triangle, no implausible distance, no
# implausible height) and every distractor-gap guard passes; each raises via
# `check(False, ...)` if `_RESAMPLE_LIMIT` draws never satisfy them.
# ---------------------------------------------------------------------------


def _draw_easy(rng) -> dict:
    for _ in range(_RESAMPLE_LIMIT):
        phi = rng.choice(PHI_POOL)
        psi = rng.choice(PSI_POOL)
        if not _valid_ground_triangle(phi, psi):
            continue
        d = rng.randint(D_LO, D_HI)
        theta = rng.choice(THETA_POOL)
        angle_t = 180 - phi - psi

        pt = d * math.sin(math.radians(psi)) / math.sin(math.radians(angle_t))
        height = pt * math.tan(math.radians(theta))
        if not (_H_LO <= height <= _H_HI):
            continue
        if not rounds_unambiguously(height, 1):
            continue
        answer_text = _round1(height)
        if answer_text.endswith(".0"):
            continue

        # Misconceptions (unrounded, as final heights):
        wrong_leg = d * math.tan(math.radians(theta))  # tan(d) not tan(PT)
        pt_trig_swap = d * math.cos(math.radians(psi)) / math.sin(math.radians(angle_t))
        h_trig_swap = pt_trig_swap * math.tan(math.radians(theta))  # sin<->cos in sine rule
        pt_wrong_vertex = d * math.sin(math.radians(phi)) / math.sin(math.radians(angle_t))
        h_wrong_vertex = pt_wrong_vertex * math.tan(math.radians(theta))  # used QT (phi) not PT (psi)
        cands = [wrong_leg, h_trig_swap, h_wrong_vertex]
        if not all(c > 1.0 for c in cands):
            continue  # a non-positive or near-zero candidate height is trivially eliminable, not a real distractor
        if not all(_far_enough(c, height) for c in cands):
            continue

        return dict(
            d=d,
            phi=phi,
            psi=psi,
            theta=theta,
            angle_t=angle_t,
            pt=pt,
            height=height,
            answer_text=answer_text,
            cands=cands,
        )
    check(False, "practical_triangle easy: no well-conditioned draw found")


def _draw_medium(rng) -> dict:
    for _ in range(_RESAMPLE_LIMIT):
        phi = rng.choice(PHI_POOL)
        psi = rng.choice(PSI_POOL)
        if not _valid_ground_triangle(phi, psi):
            continue
        d = rng.randint(D_LO, D_HI)
        theta = rng.choice(THETA_POOL)
        angle_t = 180 - phi - psi

        qt = d * math.sin(math.radians(phi)) / math.sin(math.radians(angle_t))
        height = qt * math.tan(math.radians(theta))
        if not (_H_LO <= height <= _H_HI):
            continue
        if not rounds_unambiguously(height, 1):
            continue
        answer_text = _round1(height)
        if answer_text.endswith(".0"):
            continue

        wrong_leg = d * math.tan(math.radians(theta))
        qt_trig_swap = d * math.cos(math.radians(phi)) / math.sin(math.radians(angle_t))
        h_trig_swap = qt_trig_swap * math.tan(math.radians(theta))
        qt_wrong_vertex = d * math.sin(math.radians(psi)) / math.sin(math.radians(angle_t))
        h_wrong_vertex = qt_wrong_vertex * math.tan(math.radians(theta))  # used PT (psi) not QT (phi)
        cands = [wrong_leg, h_trig_swap, h_wrong_vertex]
        if not all(c > 1.0 for c in cands):
            continue  # a non-positive or near-zero candidate height is trivially eliminable, not a real distractor
        if not all(_far_enough(c, height) for c in cands):
            continue

        return dict(
            d=d,
            phi=phi,
            psi=psi,
            theta=theta,
            angle_t=angle_t,
            qt=qt,
            height=height,
            answer_text=answer_text,
            cands=cands,
        )
    check(False, "practical_triangle medium: no well-conditioned draw found")


def _draw_hard(rng) -> dict:
    for _ in range(_RESAMPLE_LIMIT):
        d = rng.randint(D_LO, D_HI)
        e = rng.randint(E_LO, E_HI)
        psi = rng.choice(PSI_POOL)
        theta = rng.choice(THETA_POOL)

        pt_sq = d * d + e * e - 2 * d * e * math.cos(math.radians(psi))
        if pt_sq <= 0:
            continue
        pt = math.sqrt(pt_sq)
        if not (5.0 <= pt <= 400.0):
            continue

        # Sanity: the fully-determined triangle's OTHER two angles must also
        # be non-sliver, even though only d, e, psi were stated.
        angle_t_check = math.degrees(math.acos((pt * pt + e * e - d * d) / (2 * pt * e)))
        angle_p_check = 180 - psi - angle_t_check
        if not (20 <= angle_t_check <= 140 and 20 <= angle_p_check <= 140):
            continue

        height = pt * math.tan(math.radians(theta))
        if not (_H_LO <= height <= _H_HI):
            continue
        if not rounds_unambiguously(height, 1):
            continue
        answer_text = _round1(height)
        if answer_text.endswith(".0"):
            continue

        pt_sign_error = math.sqrt(d * d + e * e + 2 * d * e * math.cos(math.radians(psi)))
        h_sign_error = pt_sign_error * math.tan(math.radians(theta))  # + instead of -
        pt_missing2 = math.sqrt(d * d + e * e - d * e * math.cos(math.radians(psi)))
        h_missing2 = pt_missing2 * math.tan(math.radians(theta))  # dropped the factor of 2
        h_wrong_leg = d * math.tan(math.radians(theta))  # tan(d) not tan(PT)
        cands = [h_sign_error, h_missing2, h_wrong_leg]
        if not all(_far_enough(c, height) for c in cands):
            continue

        return dict(d=d, e=e, psi=psi, theta=theta, pt=pt, height=height, answer_text=answer_text, cands=cands)
    check(False, "practical_triangle hard: no well-conditioned draw found")


def _draw_exam(rng) -> dict:
    for _ in range(_RESAMPLE_LIMIT):
        phi = rng.choice(PHI_POOL)
        psi = rng.choice(PSI_POOL)
        if not _valid_ground_triangle(phi, psi):
            continue
        bearing_q = rng.randint(0, 359)
        direction = rng.choice((1, -1))
        bearing_t = (bearing_q + direction * phi) % 360
        phi_from_bearings = _interior_from_bearings(bearing_q, bearing_t)
        check(
            phi_from_bearings == phi,
            f"practical_triangle exam: bearing construction disagrees with drawn phi ({phi_from_bearings} vs {phi})",
        )

        d = rng.randint(D_LO, D_HI)
        theta = rng.choice(THETA_POOL)
        angle_t = 180 - phi - psi

        pt = d * math.sin(math.radians(psi)) / math.sin(math.radians(angle_t))
        height = pt * math.tan(math.radians(theta))
        if not (_H_LO <= height <= _H_HI):
            continue
        if not rounds_unambiguously(height, 1):
            continue
        answer_text = _round1(height)
        if answer_text.endswith(".0"):
            continue

        wrong_leg = d * math.tan(math.radians(theta))
        pt_trig_swap = d * math.cos(math.radians(psi)) / math.sin(math.radians(angle_t))
        h_trig_swap = pt_trig_swap * math.tan(math.radians(theta))
        pt_wrong_vertex = d * math.sin(math.radians(phi)) / math.sin(math.radians(angle_t))
        h_wrong_vertex = pt_wrong_vertex * math.tan(math.radians(theta))
        cands = [wrong_leg, h_trig_swap, h_wrong_vertex]
        if not all(c > 1.0 for c in cands):
            continue  # a non-positive or near-zero candidate height is trivially eliminable, not a real distractor
        if not all(_far_enough(c, height) for c in cands):
            continue

        return dict(
            d=d,
            phi=phi,
            psi=psi,
            theta=theta,
            angle_t=angle_t,
            pt=pt,
            height=height,
            bearing_q=bearing_q,
            bearing_t=bearing_t,
            direction=direction,
            answer_text=answer_text,
            cands=cands,
        )
    check(False, "practical_triangle exam: no well-conditioned draw found")


# Wording -- 3 structurally distinct phrasings per tier, sharing the SAME literal key phrases ("PQ =
# {d} m", "angle QPT = {phi} degrees"...) regardless of sentence order, so the independent oracle
# can extract every value with one anchored regex per fact rather than one regex per template.

_EASY_TEMPLATES = (
    "P and Q are two points on level ground, with PQ = {d} m. T is the base of a tower. "
    "Angle QPT = {phi} degrees and angle PQT = {psi} degrees. The angle of elevation from P "
    "to the top of the tower is {theta} degrees. Find the height of the tower, to 1 decimal place.",
    "A tower stands at point T. Two ground points P and Q satisfy PQ = {d} m, angle QPT = "
    "{phi} degrees and angle PQT = {psi} degrees. Standing at P, an observer finds that the "
    "angle of elevation from P to the top of the tower is {theta} degrees. Find the height of "
    "the tower, to 1 decimal place.",
    "T is the base of a tower on level ground. The angle of elevation from P to the top of the "
    "tower is {theta} degrees. P and Q are ground points with PQ = {d} m; angle QPT = {phi} "
    "degrees and angle PQT = {psi} degrees. Find the height of the tower, to 1 decimal place.",
)

_MEDIUM_TEMPLATES = (
    "P and Q are two points on level ground, with PQ = {d} m. T is the base of a tower. "
    "Angle QPT = {phi} degrees and angle PQT = {psi} degrees. The angle of elevation from Q "
    "to the top of the tower is {theta} degrees. Find the height of the tower, to 1 decimal place.",
    "A tower stands at point T. Two ground points P and Q satisfy PQ = {d} m, angle QPT = "
    "{phi} degrees and angle PQT = {psi} degrees. Standing at Q, an observer finds that the "
    "angle of elevation from Q to the top of the tower is {theta} degrees. Find the height of "
    "the tower, to 1 decimal place.",
    "T is the base of a tower on level ground. The angle of elevation from Q to the top of the "
    "tower is {theta} degrees. P and Q are ground points with PQ = {d} m; angle QPT = {phi} "
    "degrees and angle PQT = {psi} degrees. Find the height of the tower, to 1 decimal place.",
)

_HARD_TEMPLATES = (
    "P, Q and T are three points, with T the base of a tower. PQ = {d} m and QT = {e} m, with "
    "angle PQT = {psi} degrees between them. The angle of elevation from P to the top of the "
    "tower is {theta} degrees. Find the height of the tower, to 1 decimal place.",
    "A site plan shows PQ = {d} m and QT = {e} m, with the included angle PQT = {psi} degrees. "
    "T is the base of a tower. The angle of elevation from P to the top of the tower is {theta} "
    "degrees. Find the height of the tower, to 1 decimal place.",
    "T is the base of a tower. Given PQ = {d} m, QT = {e} m and angle PQT = {psi} degrees, and "
    "that the angle of elevation from P to the top of the tower is {theta} degrees, find the "
    "height of the tower, to 1 decimal place.",
)

_EXAM_TEMPLATES = (
    "P and Q are ground points with PQ = {d} m and angle PQT = {psi} degrees, where T is the "
    "base of a tower. From P, the bearing of Q from P is {bq} degrees and the bearing of T "
    "from P is {bt} degrees. The angle of elevation from P to the top of the tower is {theta} "
    "degrees. Find the height of the tower, to 1 decimal place.",
    "A tower's base is at T. From P, the bearing of Q from P is {bq} degrees and the bearing "
    "of T from P is {bt} degrees. PQ = {d} m and angle PQT = {psi} degrees. The angle of "
    "elevation from P to the top of the tower is {theta} degrees. Find the height of the "
    "tower, to 1 decimal place.",
    "T is the base of a tower. The angle of elevation from P to the top of the tower is {theta} "
    "degrees. From P, the bearing of Q from P is {bq} degrees and the bearing of T from P is "
    "{bt} degrees, with PQ = {d} m and angle PQT = {psi} degrees. Find the height of the tower, "
    "to 1 decimal place.",
)


def _build_result(question_text: str, working: list[str], draw: dict) -> dict:
    answer_text = draw["answer_text"]
    height = draw["height"]
    cands = draw["cands"]
    check_distinct(height, *cands, msg="practical_triangle: a distractor equals the answer")
    candidates = [_round1(c) for c in cands]
    distractors = pick_distractors(answer_text, candidates, tolerance=stated_precision_tolerance(1))
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=float(answer_text),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _easy_tier(rng) -> dict:
    draw = _draw_easy(rng)
    d, phi, psi, theta, angle_t, pt, height = (draw[k] for k in ("d", "phi", "psi", "theta", "angle_t", "pt", "height"))
    pt_check = _verify_pt_via_qt_cosine(d, phi, psi, angle_t, pt)
    _verify_elevation(pt_check, theta, height)

    question_text = phrase(rng, _EASY_TEMPLATES, d=d, phi=phi, psi=psi, theta=theta)
    working = [
        f"Ground triangle PQT: angle T = 180 - {phi} - {psi} = {angle_t} degrees.",
        f"Sine rule: PT = PQ × sin(angle PQT) / sin(angle T) = {d} × sin({psi}°) / sin({angle_t}°) = {round(pt, 4)} m.",
        f"Elevation from P: height = PT × tan({theta}°) = {draw['answer_text']} m (to 1 decimal place).",
    ]
    return _build_result(question_text, working, draw)


def _medium_tier(rng) -> dict:
    draw = _draw_medium(rng)
    d, phi, psi, theta, angle_t, qt, height = (draw[k] for k in ("d", "phi", "psi", "theta", "angle_t", "qt", "height"))
    qt_check = _verify_qt_via_pt_cosine(d, phi, psi, angle_t, qt)
    _verify_elevation(qt_check, theta, height)

    question_text = phrase(rng, _MEDIUM_TEMPLATES, d=d, phi=phi, psi=psi, theta=theta)
    working = [
        f"Ground triangle PQT: angle T = 180 - {phi} - {psi} = {angle_t} degrees.",
        f"Sine rule: QT = PQ × sin(angle QPT) / sin(angle T) = {d} × sin({phi}°) / sin({angle_t}°) = {round(qt, 4)} m.",
        f"Elevation from Q: height = QT × tan({theta}°) = {draw['answer_text']} m (to 1 decimal place).",
    ]
    return _build_result(question_text, working, draw)


def _hard_tier(rng) -> dict:
    draw = _draw_hard(rng)
    d, e, psi, theta, pt, height = (draw[k] for k in ("d", "e", "psi", "theta", "pt", "height"))
    _verify_cosine_sas(d, e, psi, pt)
    _verify_elevation(pt, theta, height)

    question_text = phrase(rng, _HARD_TEMPLATES, d=d, e=e, psi=psi, theta=theta)
    working = [
        f"Cosine rule (SAS): PT^2 = PQ^2 + QT^2 - 2 × PQ × QT × cos(angle PQT) "
        f"= {d}^2 + {e}^2 - 2 × {d} × {e} × cos({psi}°).",
        f"PT = {round(pt, 4)} m.",
        f"Elevation from P: height = PT × tan({theta}°) = {draw['answer_text']} m (to 1 decimal place).",
    ]
    return _build_result(question_text, working, draw)


def _exam_tier(rng) -> dict:
    draw = _draw_exam(rng)
    d, phi, psi, theta, angle_t, pt, height, bearing_q, bearing_t = (
        draw[k] for k in ("d", "phi", "psi", "theta", "angle_t", "pt", "height", "bearing_q", "bearing_t")
    )
    _verify_bearing_interior(bearing_q, bearing_t, phi)
    pt_check = _verify_pt_via_qt_cosine(d, phi, psi, angle_t, pt)
    _verify_elevation(pt_check, theta, height)

    question_text = phrase(
        rng,
        _EXAM_TEMPLATES,
        d=d,
        psi=psi,
        theta=theta,
        bq=f"{bearing_q:03d}",
        bt=f"{bearing_t:03d}",
    )
    working = [
        f"Interior angle at P: the bearing difference between T and Q from P, adjusted for "
        f"wraparound, gives angle QPT = {phi} degrees.",
        f"Ground triangle PQT: angle T = 180 - {phi} - {psi} = {angle_t} degrees.",
        f"Sine rule: PT = PQ × sin(angle PQT) / sin(angle T) = {d} × sin({psi}°) / sin({angle_t}°) = {round(pt, 4)} m.",
        f"Elevation from P: height = PT × tan({theta}°) = {draw['answer_text']} m (to 1 decimal place).",
    ]
    return _build_result(question_text, working, draw)


@generator(
    "p14a_practical_triangle",
    level=13,
    presets={
        "easy": dict(variant="direct_angles"),
        "medium": dict(variant="swap_orientation"),
        "hard": dict(variant="cosine_rule_sas"),
        "exam": dict(variant="bearings_ground_triangle"),
    },
)
def p14a_practical_triangle(rng, variant="direct_angles"):
    """Solve practical problems involving the trigonometry of right-angled
    AND non-right-angled triangles, chaining a non-right ground triangle
    (solved by the sine or cosine rule) into a right-angled elevation leg --
    the genuinely two-step kernel none of the five single-triangle shipping
    neighbours (right-triangle-angle, sine-rule, cosine-rule, the elevation
    skill's collinear two-observation tiers, the bearings skill's
    always-90-degree-turn tiers) ever requires (ACMGM037)."""
    if variant == "direct_angles":
        return _easy_tier(rng)
    if variant == "swap_orientation":
        return _medium_tier(rng)
    if variant == "cosine_rule_sas":
        return _hard_tier(rng)
    if variant == "bearings_ground_triangle":
        return _exam_tier(rng)
    raise ValueError(f"unknown variant: {variant}")
