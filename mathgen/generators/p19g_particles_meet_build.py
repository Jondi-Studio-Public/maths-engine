"""Generates `p19g_particles_meet_mcq` questions: decide whether two particles with stated position-vector
functions ever meet.

Do two particles meet?

senior-examine-the-position-of-two-particles-each-described (ACMSM106, domain
vectors). Scope: given two
particles whose position vectors are stated as vector functions of time,
determine whether the particles meet -- occupy the same position at the same
time -- choosing yes or no from the offered options.

``p19g_particles_meet_mcq`` is an ``answer: mcq`` classification. The natural
underlying question is binary (do they meet, yes/no), which the answer-space requirements
forbid shipping as a two-option mcq, so this ships FOUR options with exactly
one true, verified by construction on every generated instance (never
assumed to be mutually exclusive by design).

Mathematics. Particle A: r_A(t) = pA + t*vA. Particle B: r_B(t) = pB + t*vB,
both in R^2 (R^1 time domain, t >= 0). Solving r_A(t) = r_B(t) component-wise
is a genuinely different operation from ``vector-addition`` (the active
prerequisite), which only ever adds/subtracts two GIVEN vectors and never
involves time or a system to solve.

Given the final (already-constructed) pA, vA, pB, vB -- never the
construction's own bookkeeping of "which case I meant to build" -- exactly
one of three real outcomes holds, independently re-derived every time by
``_base_options``/``_window_options`` from ``_solve2``:

  - vA, vB not parallel (``cross(vA, vB) != 0``): the two INFINITE lines
    through pA/vA and pB/vB cross at exactly one point, reached by particle A
    at a unique t = sA and particle B at a unique t = sB (Cramer's rule).
      - sA == sB (and >= 0): the particles MEET, at that shared time.
      - sA != sB: the paths CROSS (same point in space) but each particle is
        there at a DIFFERENT time -- they never actually meet.
  - vA, vB parallel (``cross(vA, vB) == 0``) and the two lines are distinct
    (not coincident, verified by ``cross(vA, pB - pA) != 0``): the paths
    NEVER cross at all, for any pair of times.

The exam tier adds a genuine kind escalation: a stated time WINDOW
(0 <= t <= T). This turns "meet" into two mutually exclusive REAL
sub-cases -- a meeting inside the window (a genuine "yes") and a meeting
that exists mathematically but occurs at t > T (a genuine, specific "no,
outside the window") -- so all four exam-tier options are real, mutually
exclusive verdicts, no fixed filler decoy needed.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, rational_text

# ============================================================================
# Exact vector arithmetic (Fraction throughout -- never float).
# ============================================================================


def _pos(p, v, t):
    return (p[0] + v[0] * t, p[1] + v[1] * t)


def _cross2(vA, vB):
    return vA[0] * vB[1] - vA[1] * vB[0]


def _solve2(pA, vA, pB, vB):
    """(sA, sB) with pA + sA*vA == pB + sB*vB, or None if vA, vB are
    parallel (``_cross2(vA, vB) == 0``). Cramer's rule, exact Fraction
    arithmetic -- this is the ONE independent-of-construction recompute
    every option-builder below calls fresh, on the FINAL vectors, never on
    whatever the construction step believed it was building (re-derive,
    don't trust)."""
    cross = _cross2(vA, vB)
    if cross == 0:
        return None
    dx, dy = pB[0] - pA[0], pB[1] - pA[1]
    sA = (dx * vB[1] - vB[0] * dy) / cross
    sB = (vA[1] * dx - vA[0] * dy) / cross
    return sA, sB


def _nz(rng, lo, hi):
    return Fraction(rng.choice([v for v in range(lo, hi + 1) if v != 0]))


def _coef_hard(rng, lo, hi):
    """A hard-tier coefficient: an integer, or (half the time) a half-integer
    -- the "coefficients include fractions" escalation."""
    if rng.random() < 0.5:
        return Fraction(rng.randint(lo, hi))
    odd = rng.choice([v for v in range(2 * lo + 1, 2 * hi, 2) if v != 0])
    return Fraction(odd, 2)


def _nz_hard(rng, lo, hi):
    while True:
        v = _coef_hard(rng, lo, hi)
        if v != 0:
            return v


# ============================================================================
# Formatting
# ============================================================================


def _vt_str(mag: Fraction) -> str:
    if mag == 1:
        return "t"
    mag_str = rational_text(mag)
    return f"({mag_str})t" if mag.denominator != 1 else f"{mag_str}t"


def _comp(p: Fraction, v: Fraction) -> str:
    """One coordinate function ``p + v*t`` the way a student would write it:
    drops a zero position ("-2t" rather than "0 - 2t") or a zero velocity
    ("5" rather than "5 + 0t"), and parenthesises a non-integer coefficient
    ("(1/2)t") so it is never misread as "1/2 * t" glued onto the next
    character."""
    if v == 0:
        return rational_text(p)
    sign = "+" if v > 0 else "-"
    vt = _vt_str(abs(v))
    if p == 0:
        return vt if v > 0 else f"-{vt}"
    return f"{rational_text(p)} {sign} {vt}"


def _fmt_point(pt) -> str:
    return f"({rational_text(pt[0])}, {rational_text(pt[1])})"


def _fmt_t(t) -> str:
    return rational_text(t)


def _opt_meet_text(point, t) -> str:
    """A "Yes" option verbose enough to match the "No" templates' own
    length/specificity (defect avoided: the un-padded "Yes" template was
    ~3x shorter than every "No" template, so "always pick the longest option"
    scored 71% on easy). Restating BOTH particles' own position at the claimed
    time is genuine, checkable content -- the option's claim IS exactly that
    r_A(t) and r_B(t) both equal ``point`` -- never filler prose."""
    pt = _fmt_point(point)
    t_str = _fmt_t(t)
    return (
        f"Yes -- the two particles meet at {pt} when t = {t_str}: particle A's "
        f"own position at that time is {pt}, and so is particle B's -- the same "
        f"point at the same time."
    )


def _opt_meet_in_window_text(point, t, T) -> str:
    pt = _fmt_point(point)
    t_str = _fmt_t(t)
    return (
        f"Yes -- the two particles meet at {pt} when t = {t_str}: particle A's "
        f"own position at that time is {pt}, and so is particle B's, within the "
        f"stated window 0 <= t <= {_fmt_t(T)}."
    )


def _opt_meet_outside_window_text(point, t, T) -> str:
    return (
        f"No -- a genuine meeting point exists at t = {_fmt_t(t)}, giving position "
        f"{_fmt_point(point)}, but this is outside the stated window 0 <= t <= {_fmt_t(T)}, "
        f"so no meeting occurs within it."
    )


def _opt_cross_text(point, ta, tb) -> str:
    return (
        f"No -- the two paths cross at {_fmt_point(point)}, but particle A is there "
        f"when t = {_fmt_t(ta)} while particle B is there when t = {_fmt_t(tb)} -- "
        f"different times, so they never actually meet."
    )


def _opt_never_text() -> str:
    return (
        "No -- the two paths never cross at all: no pair of times, equal or "
        "different, ever places the two particles at the same point; no real "
        "solution exists."
    )


def _coin(pA, vA, pB, vB) -> int:
    """A deterministic 0/1 drawn from the final vectors (so option builders
    keep their four-argument signature): which of two equally coherent decoy
    constructions to use, so the key's point is repeated in about half the
    instances and unique in the rest -- neither is a tell."""
    total = sum((abs(x.numerator) * 7 + x.denominator * 3) for x in (*pA, *vA, *pB, *vB))
    return (total // 3) % 2


def _wrong_time_yes(pA, vA, pB, vB, t):
    """A "Yes" claim at time ``t`` at particle A's position there: false
    whenever B is not at that point at ``t`` (checked), and every quoted
    number is checkable by plugging ``t`` into the two stated paths."""
    point = _pos(pA, vA, t)
    check(_pos(pB, vB, t) != point, "a wrong-time yes decoy is accidentally true")
    return _opt_meet_text(point, t)


def _wrong_time_yes_b(pA, vA, pB, vB, t):
    """As ``_wrong_time_yes`` but quoting particle B's position at ``t``."""
    point = _pos(pB, vB, t)
    check(_pos(pA, vA, t) != point, "a wrong-time yes decoy is accidentally true")
    return _opt_meet_text(point, t)


# ============================================================================
# The independent option-builder (no window): exactly one of MEET / CROSS /
# NEVER is real. Every distractor is a claim about the SAME two paths: a wrong-time "Yes" at a point one of the paths really passes through,
# and a "cross at different times" claim built from a real point of the paths.
# No distractor uses an invented point, and there is no fixed t = 0 decoy, so
# neither "the odd-one-out Yes" nor "the repeated point" identifies the key.
# ============================================================================


def _base_options(pA, vA, pB, vB):
    check(pA != pB, "the two particles must not start at the same point")
    solved = _solve2(pA, vA, pB, vB)
    one = Fraction(1)

    if solved is not None:
        sA, sB = solved
        if sA == sB:
            check(sA >= 0, "a constructed meeting time must not be negative")
            t0 = sA
            point = _pos(pA, vA, t0)
            check(_pos(pB, vB, t0) == point, "meet point disagrees between the two particles")
            answer = _opt_meet_text(point, t0)
            # Cross claim at the TRUE point with invented different times:
            # false because both particles reach it at t0.
            check(_pos(pB, vB, t0 + one) != point, "the fake-cross decoy is accidentally true")
            if _coin(pA, vA, pB, vB):
                opt_cross = _opt_cross_text(point, t0, t0 + one)
            else:
                q = _pos(pA, vA, t0 + one)
                check(_pos(pB, vB, t0) != q, "the fake-cross decoy is accidentally true")
                opt_cross = _opt_cross_text(q, t0 + one, t0)
            opt_never = _opt_never_text()
            opt_wrong = _wrong_time_yes(pA, vA, pB, vB, t0 + one)
            verdict, flavour = "MEET", "meet"
            distractors = [opt_cross, opt_never, opt_wrong]
        else:
            ta, tb = sA, sB
            point = _pos(pA, vA, ta)
            check(_pos(pB, vB, tb) == point, "cross point disagrees between the two particles")
            answer = _opt_cross_text(point, ta, tb)
            # "Yes" at the true crossing point at A's time: false, B is not
            # there at ta (ta != tb and B passes through the point once).
            check(_pos(pB, vB, ta) != point, "the fake-meet decoy is accidentally true")
            if _coin(pA, vA, pB, vB):
                opt_meet = _opt_meet_text(point, ta)
            else:
                opt_meet = _wrong_time_yes(pA, vA, pB, vB, tb + one)
            opt_never = _opt_never_text()
            opt_wrong = _wrong_time_yes_b(pA, vA, pB, vB, ta)
            verdict, flavour = "NO", "cross_different_times"
            distractors = [opt_meet, opt_never, opt_wrong]
    else:
        crossline = _cross2(vA, (pB[0] - pA[0], pB[1] - pA[1]))
        check(crossline != 0, "parallel velocity vectors must trace distinct, non-coincident lines")
        answer = _opt_never_text()
        opt_meet = _wrong_time_yes(pA, vA, pB, vB, one)
        opt_meet_b = _wrong_time_yes_b(pA, vA, pB, vB, one)
        fake_ta, fake_tb = one, Fraction(2)
        fake_cross_point = _pos(pA, vA, fake_ta)
        check(_pos(pB, vB, fake_tb) != fake_cross_point, "the fake-cross decoy is accidentally true")
        opt_cross = _opt_cross_text(fake_cross_point, fake_ta, fake_tb)
        if _coin(pA, vA, pB, vB):
            fake_cross_point = _pos(pB, vB, fake_tb)
            check(_pos(pA, vA, fake_ta) != fake_cross_point, "the fake-cross decoy is accidentally true")
            opt_cross = _opt_cross_text(fake_cross_point, fake_ta, fake_tb)
            opt_meet_b = _wrong_time_yes_b(pA, vA, pB, vB, fake_tb)
        verdict, flavour = "NO", "never_intersect"
        distractors = [opt_meet, opt_cross, opt_meet_b]

    check(len({answer, *distractors}) == 4, "the four mcq options are not all distinct")
    return answer, distractors, verdict, flavour


# ============================================================================
# The independent option-builder WITH a stated time window (exam tier): four
# REAL, mutually exclusive verdicts -- no fixed filler decoy needed. Decoys
# quote real points of the two paths (never an invented point).
# ============================================================================


def _window_options(pA, vA, pB, vB, T, claim_pick=0):
    check(pA != pB, "the two particles must not start at the same point")
    solved = _solve2(pA, vA, pB, vB)
    one = Fraction(1)

    def fake_in():
        # "Yes within the window" at A's position at a claimed time in
        # (0, T]: false, B is not there (checked). ``claim_pick`` chooses
        # among the claimable times so the claim is not always the window end.
        n = int(T)
        cands = [Fraction(k) for k in range(1, n + 1) if _pos(pB, vB, Fraction(k)) != _pos(pA, vA, Fraction(k))]
        check(bool(cands), "no false in-window claim is available")
        tc = cands[claim_pick % len(cands)]
        pt = _pos(pA, vA, tc)
        check(_pos(pB, vB, tc) != pt and 0 < tc <= T, "the fake in-window decoy is accidentally true")
        return _opt_meet_in_window_text(pt, tc, T)

    def fake_out():
        pt = _pos(pA, vA, T + one)
        check(_pos(pB, vB, T + one) != pt, "the fake outside-window decoy is accidentally true")
        return _opt_meet_outside_window_text(pt, T + one, T)

    if solved is not None:
        sA, sB = solved
        if sA == sB:
            t0 = sA
            check(t0 >= 0, "a constructed meeting time must not be negative")
            point = _pos(pA, vA, t0)
            check(_pos(pB, vB, t0) == point, "meet point disagrees between the two particles")
            in_window = t0 <= T
            check(_pos(pB, vB, t0 + one) != point, "the fake-cross decoy is accidentally true")
            if _coin(pA, vA, pB, vB):
                opt_cross = _opt_cross_text(point, t0, t0 + one)
            else:
                q = _pos(pA, vA, t0 + one)
                check(_pos(pB, vB, t0) != q, "the fake-cross decoy is accidentally true")
                opt_cross = _opt_cross_text(q, t0 + one, t0)
            opt_never = _opt_never_text()
            if in_window:
                answer = _opt_meet_in_window_text(point, t0, T)
                distractors = [fake_out(), opt_cross, opt_never]
                verdict, flavour = "MEET_IN_WINDOW", "meet_in_window"
            else:
                answer = _opt_meet_outside_window_text(point, t0, T)
                distractors = [fake_in(), opt_cross, opt_never]
                verdict, flavour = "MEET_OUTSIDE_WINDOW", "meet_outside_window"
        else:
            ta, tb = sA, sB
            point = _pos(pA, vA, ta)
            check(_pos(pB, vB, tb) == point, "cross point disagrees between the two particles")
            answer = _opt_cross_text(point, ta, tb)
            distractors = [fake_in(), fake_out(), _opt_never_text()]
            verdict, flavour = "NO", "cross_different_times"
    else:
        crossline = _cross2(vA, (pB[0] - pA[0], pB[1] - pA[1]))
        check(crossline != 0, "parallel velocity vectors must trace distinct, non-coincident lines")
        answer = _opt_never_text()
        fake_ta, fake_tb = one, Fraction(2)
        fake_cross_point = _pos(pA, vA, fake_ta)
        check(_pos(pB, vB, fake_tb) != fake_cross_point, "the fake-cross decoy is accidentally true")
        opt_cross = _opt_cross_text(fake_cross_point, fake_ta, fake_tb)
        verdict, flavour = "NO", "never_intersect"
        distractors = [fake_in(), fake_out(), opt_cross]

    check(len({answer, *distractors}) == 4, "the four mcq options are not all distinct")
    return answer, distractors, verdict, flavour


# ============================================================================
# Construction (draw the verdict FIRST, then build vectors to realize it).
# ============================================================================

_INTROS = (
    "Two particles move in a plane so that their position vectors, in metres, are given below.",
    "The position vectors of two particles, in metres, are functions of time t (in seconds), given below.",
    "Below are the position vectors of two particles, as functions of time t.",
)


def _render(pA, vA, pB, vB, *, window=None):
    intro = _INTROS[0]
    ax, ay = _comp(pA[0], vA[0]), _comp(pA[1], vA[1])
    bx, by = _comp(pB[0], vB[0]), _comp(pB[1], vB[1])
    if window is None:
        domain = "for t >= 0"
        question = "Do the two particles meet?"
    else:
        domain = f"for 0 <= t <= {_fmt_t(window)}"
        question = "Do the two particles meet within this time window?"
    text = f"{intro} Particle A: r_A(t) = ({ax}, {ay}). Particle B: r_B(t) = ({bx}, {by}), {domain}. {question}"
    return text


def _manhattan(pA, pB) -> Fraction:
    """|pA - pB| in the taxicab metric -- the concrete "how far apart do the
    two particles start" measure the fixed "Yes at t=0" decoy's own
    plausibility is bounded by (see ``max_gap`` below)."""
    return abs(pA[0] - pB[0]) + abs(pA[1] - pB[1])


def _draw_p0(rng, coord_lo, coord_hi, p0_half):
    """The shared meeting/crossing point. With ``p0_half`` (hard tier) about
    half the draws put at least one coordinate on a half-integer, so the key's
    quoted point is fractional as often as it is whole (a whole-number point
    would otherwise single out the key when the velocities are half-integers)."""
    x, y = rng.randint(coord_lo, coord_hi), rng.randint(coord_lo, coord_hi)
    P0 = [Fraction(x), Fraction(y)]
    if p0_half and rng.random() < 0.5:
        i = rng.choice((0, 1)) if rng.random() < 0.6 else None
        for j in (i,) if i is not None else (0, 1):
            P0[j] += Fraction(1, 2)
    return (P0[0], P0[1])


def _draw_meet(
    rng, *, coord_lo, coord_hi, vel_lo, vel_hi, t_lo, t_hi, shared_coord, coef_fn, max_gap=None, p0_half=False
):
    for _ in range(500):
        t0 = Fraction(rng.randint(t_lo, t_hi))
        if t0 == 0:
            continue
        P0 = _draw_p0(rng, coord_lo, coord_hi, p0_half)
        vA = (coef_fn(rng, vel_lo, vel_hi), coef_fn(rng, vel_lo, vel_hi))
        if shared_coord and rng.random() < 0.5:
            vB = (vA[0], coef_fn(rng, vel_lo, vel_hi))
        else:
            vB = (coef_fn(rng, vel_lo, vel_hi), coef_fn(rng, vel_lo, vel_hi))
        if _cross2(vA, vB) == 0:
            continue
        pA = (P0[0] - t0 * vA[0], P0[1] - t0 * vA[1])
        pB = (P0[0] - t0 * vB[0], P0[1] - t0 * vB[1])
        if pA == pB:
            continue
        if max_gap is not None and _manhattan(pA, pB) > max_gap:
            continue
        return pA, vA, pB, vB, t0
    raise RuntimeError("could not construct a MEET instance")


def _draw_cross(
    rng, *, coord_lo, coord_hi, vel_lo, vel_hi, t_lo, t_hi, shared_coord, coef_fn, max_gap=None, p0_half=False
):
    for _ in range(500):
        ta = Fraction(rng.randint(t_lo, t_hi))
        tb = Fraction(rng.randint(t_lo, t_hi))
        if ta == tb or ta == 0 or tb == 0:
            continue
        P0 = _draw_p0(rng, coord_lo, coord_hi, p0_half)
        vA = (coef_fn(rng, vel_lo, vel_hi), coef_fn(rng, vel_lo, vel_hi))
        if shared_coord and rng.random() < 0.5:
            vB = (vA[0], coef_fn(rng, vel_lo, vel_hi))
        else:
            vB = (coef_fn(rng, vel_lo, vel_hi), coef_fn(rng, vel_lo, vel_hi))
        if _cross2(vA, vB) == 0:
            continue
        pA = (P0[0] - ta * vA[0], P0[1] - ta * vA[1])
        pB = (P0[0] - tb * vB[0], P0[1] - tb * vB[1])
        if pA == pB:
            continue
        if max_gap is not None and _manhattan(pA, pB) > max_gap:
            continue
        return pA, vA, pB, vB
    raise RuntimeError("could not construct a CROSS instance")


# Default k pool for NEVER's parallel scale factor (integers AND
# half-integers -- the "coefficients include fractions" hard-tier
# escalation). Easy/medium pass ``_K_INTEGER_ONLY`` instead: hard's own
# coef_fn (``_nz_hard``) already renders half-integer velocities on meet/
# cross draws too, so a fractional k there is not a tell; easy/medium's
# meet/cross draws use integer-only ``_nz``, so a fractional k on a NEVER
# draw would be a "see a '/' -> pick never" giveaway.
_K_MIXED = tuple(Fraction(n, d) for n in (-3, -2, -1, 1, 2, 3) for d in (1, 2) if Fraction(n, d) != 1)
_K_INTEGER_ONLY = tuple(Fraction(n) for n in (-3, -2, -1, 2, 3))


def _draw_never(rng, *, coord_lo, coord_hi, vel_lo, vel_hi, coef_fn, max_gap=None, k_choices=_K_MIXED):
    for _ in range(500):
        vA = (coef_fn(rng, vel_lo, vel_hi), coef_fn(rng, vel_lo, vel_hi))
        pA = (Fraction(rng.randint(coord_lo, coord_hi)), Fraction(rng.randint(coord_lo, coord_hi)))
        perp = (-vA[1], vA[0])
        m = Fraction(rng.randint(-3, 3))
        k = rng.choice(k_choices)
        vB = (k * vA[0], k * vA[1])
        pB = (pA[0] + perp[0] + m * vA[0], pA[1] + perp[1] + m * vA[1])
        if _cross2(vA, vB) != 0:
            continue
        if pA == pB:
            continue
        if max_gap is not None and _manhattan(pA, pB) > max_gap:
            continue
        return pA, vA, pB, vB
    raise RuntimeError("could not construct a NEVER instance")


# ============================================================================
# Tiers
# ============================================================================


def _build(
    rng,
    *,
    coord_lo,
    coord_hi,
    vel_lo,
    vel_hi,
    t_lo,
    t_hi,
    shared_coord,
    coef_fn,
    never_weight,
    max_gap=None,
    never_k_choices=_K_MIXED,
    p0_half=False,
):
    r = rng.random()
    if never_weight > 0 and r >= 1 - never_weight:
        pA, vA, pB, vB = _draw_never(
            rng,
            coord_lo=coord_lo,
            coord_hi=coord_hi,
            vel_lo=vel_lo,
            vel_hi=vel_hi,
            coef_fn=coef_fn,
            max_gap=max_gap,
            k_choices=never_k_choices,
        )
    elif r < 0.30:
        pA, vA, pB, vB, _ = _draw_meet(
            rng,
            coord_lo=coord_lo,
            coord_hi=coord_hi,
            vel_lo=vel_lo,
            vel_hi=vel_hi,
            t_lo=t_lo,
            t_hi=t_hi,
            shared_coord=shared_coord,
            coef_fn=coef_fn,
            max_gap=max_gap,
            p0_half=p0_half,
        )
    else:
        pA, vA, pB, vB = _draw_cross(
            rng,
            coord_lo=coord_lo,
            coord_hi=coord_hi,
            vel_lo=vel_lo,
            vel_hi=vel_hi,
            t_lo=t_lo,
            t_hi=t_hi,
            shared_coord=shared_coord,
            coef_fn=coef_fn,
            max_gap=max_gap,
            p0_half=p0_half,
        )
    answer, distractors, verdict, flavour = _base_options(pA, vA, pB, vB)
    text = _render(pA, vA, pB, vB)
    return dict(
        text=text,
        answer=answer,
        answer_text=answer,
        distractors=distractors,
        working=[_WHY[flavour]],
    )


_WHY = {
    "meet": "Solve r_A(t) = r_B(t) for each component: the same value of t works in both, so the particles meet.",
    "cross_different_times": "The paths cross, but the particles reach the crossing point at different times, so they never meet.",
    "never_intersect": "The velocity vectors are parallel and the two paths are different lines, so the paths never meet.",
    "meet_in_window": "The same value of t satisfies r_A(t) = r_B(t) in both components, and it lies inside the time window, so the particles meet during it.",
    "meet_outside_window": "The same value of t satisfies r_A(t) = r_B(t) in both components, but it lies outside the time window, so the particles do not meet during it.",
}


def _build_easy(rng):
    # never_weight=1/3 makes NEVER-cross a real, live draw at this tier (it was previously never drawn at all here); max_gap=6 keeps the
    # fixed "Yes at t=0" decoy's own claim (particle A's t=0 position equals
    # pB) close enough to the true pA that it is never trivially rejectable
    # on sight -- half the tier's own coordinate span (-6..6, span 12).
    # never_k_choices=_K_INTEGER_ONLY: easy's meet/cross draws use integer-
    # only _nz coefficients, so a fractional NEVER-draw scale factor would be
    # a "see a '/' -> pick never" tell.
    return _build(
        rng,
        coord_lo=-6,
        coord_hi=6,
        vel_lo=-4,
        vel_hi=4,
        t_lo=1,
        t_hi=3,
        shared_coord=False,
        coef_fn=_nz,
        never_weight=1 / 3,
        max_gap=6,
        never_k_choices=_K_INTEGER_ONLY,
    )


def _build_medium(rng):
    # Same rebalance as easy: never_weight raised from 0.25 to
    # 1/3 so no single flavour (meet 30%, cross ~36.7%, never ~33.3%) exceeds
    # the 40% key-share ceiling; max_gap=8 is half this tier's own coordinate
    # span (-8..8, span 16), for the same decoy-coherence reason as easy.
    # never_k_choices=_K_INTEGER_ONLY: same "no fraction tell" reason as easy
    # -- medium's meet/cross draws also use integer-only _nz.
    return _build(
        rng,
        coord_lo=-8,
        coord_hi=8,
        vel_lo=-5,
        vel_hi=5,
        t_lo=1,
        t_hi=5,
        shared_coord=True,
        coef_fn=_nz,
        never_weight=1 / 3,
        max_gap=8,
        never_k_choices=_K_INTEGER_ONLY,
    )


def _build_hard(rng):
    # max_gap=9 (half the tier's own coordinate span, -9..9): the fixed
    # decoy is offered at hard too, and the pre-fix gap there (median 17,
    # max 105.5 over a span of only 18) was just as "off by an absurd
    # multiple of the scene's own scale" as easy/medium -- bounded here for
    # the same coherence reason. never_k_choices is
    # left at the default _K_MIXED: hard's own coef_fn (_nz_hard) already
    # renders half-integer velocities on meet/cross draws too, so a
    # fractional k here is not a tell.
    return _build(
        rng,
        coord_lo=-9,
        coord_hi=9,
        vel_lo=-6,
        vel_hi=6,
        t_lo=1,
        t_hi=6,
        shared_coord=True,
        coef_fn=_nz_hard,
        never_weight=0.35,
        max_gap=9,
        p0_half=True,
    )


def _build_exam(rng):
    T = Fraction(rng.randint(2, 5))
    r = rng.random()
    if r < 0.30:
        pA, vA, pB, vB, _ = _draw_meet(
            rng,
            coord_lo=-8,
            coord_hi=8,
            vel_lo=-5,
            vel_hi=5,
            t_lo=1,
            t_hi=int(T),
            shared_coord=False,
            coef_fn=_nz,
        )
    elif r < 0.55:
        pA, vA, pB, vB, _ = _draw_meet(
            rng,
            coord_lo=-8,
            coord_hi=8,
            vel_lo=-5,
            vel_hi=5,
            t_lo=int(T) + 1,
            t_hi=int(T) + 4,
            shared_coord=False,
            coef_fn=_nz,
        )
    elif r < 0.80:
        pA, vA, pB, vB = _draw_cross(
            rng,
            coord_lo=-8,
            coord_hi=8,
            vel_lo=-5,
            vel_hi=5,
            t_lo=1,
            t_hi=6,
            shared_coord=False,
            coef_fn=_nz,
        )
    else:
        pA, vA, pB, vB = _draw_never(rng, coord_lo=-8, coord_hi=8, vel_lo=-5, vel_hi=5, coef_fn=_nz)

    answer, distractors, verdict, flavour = _window_options(pA, vA, pB, vB, T, rng.randrange(1000))
    text = _render(pA, vA, pB, vB, window=T)
    return dict(
        text=text,
        answer=answer,
        answer_text=answer,
        distractors=distractors,
        working=[_WHY[flavour]],
    )


@generator(
    "p19g_particles_meet_mcq",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19g_particles_meet_mcq(rng, tier="easy"):
    """Given two particles' position vectors as functions of time, determine
    whether the particles meet -- occupy the same position at the same time.
    Four options, exactly one true: a genuine meeting (with point/time), a
    same-SPACE-different-time crossing, a never-intersect (parallel, distinct
    paths), and (exam tier only) a genuine meeting that falls outside a
    stated time window. The verdict is drawn FIRST (construct-to-target),
    then the vectors are built to realize it; the option-builder never trusts
    that intent, re-deriving the true verdict from the final vectors."""
    if tier not in ("easy", "medium", "hard", "exam"):
        raise ValueError(f"tier must be one of easy/medium/hard/exam; got {tier!r}")
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    return _build_exam(rng)
