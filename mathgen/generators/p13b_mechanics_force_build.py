"""Momentum/force/resultant-force/ action-and-reaction; constant and non-constant force.

Skill A: senior-examine-momentum-force-resultant-force-action-and-reaction
(ACMSM133, topic ``p13b_momentum_force_reaction``, answer: number).

Skill name:
    "Examine momentum, force, resultant force, action and reaction"

    Each tier is a literal reading of one clause of that name:
    - easy   -- MOMENTUM: p = m*v, signed by stated direction (right positive).
    - medium -- RESULTANT FORCE: two collinear forces combined by direction,
      F1 - F2 (right positive). A genuinely different combination rule from
      easy's single multiplication.
    - hard   -- RESULTANT FORCE (three lines) + NEWTON'S SECOND LAW: three
      collinear forces (two applied, one resisting friction stated as a
      magnitude only, so the learner must work out ITS sign from the other
      two's net before combining), then a = F_net / m. Chains two operations
      over medium's single combination.
    - exam   -- MOMENTUM CONSERVATION + NEWTON'S THIRD LAW ("action and
      reaction"): a perfectly inelastic collision, common final velocity from
      conservation of momentum, with the equal-and-opposite collision force
      stated explicitly in the prompt (the "action and reaction" clause,
      examined, not merely asserted). A genuinely different operation --
      solving a conservation equation -- from combining forces directly.

    Every answer is an exact integer by construction (no rounding statement
    needed anywhere in this skill).

Skill B: senior-consider-constant-and-non-constant-force
(ACMSM134, topic ``p13b_constant_nonconstant_force``, answer: number).

    Skill name (quoted verbatim):
    "Consider constant and non-constant force"

    The name's two clauses ARE the tier progression, literally:
    - easy   -- CONSTANT force, from rest: a = F/m (a fixed number), then
      v = sqrt(2*a*s) (SUVAT).
    - medium -- CONSTANT force, nonzero initial speed: v = sqrt(u**2 + 2*a*s),
      an extra term the learner must not drop -- a misconception easy's
      at-rest case cannot even offer.
    - hard   -- NON-CONSTANT force F(x) = k*x: a(x) = (k/m)*x is no longer a
      fixed number, so the route is the calculus one, v*dv/dx = a(x),
      integrated: v = sqrt(k/m)*s from rest. A genuinely new operation
      (calculus, not algebra) over easy/medium.
    - exam   -- NON-CONSTANT force, nonzero initial speed, SOLVE FOR
      DISPLACEMENT: v_target**2 = u**2 + (k/m)*x**2, solve for x. Works the
      relation BACKWARDS (solve for the independent variable), a new decision
      over hard's forward evaluation.

    **Boundary against ``solve-ode-separable-poly``**: that skill integrates
    ``dy/dx = f(x)`` for ``y(x)`` as a bare algebra exercise carrying a free
    constant ``C``. This skill NEVER asks for ``y(x)`` with a free constant --
    every question is a physical velocity-at-a-displacement problem with a
    stated initial condition (mass, force, "starts from rest"/initial speed),
    and the operation is ``v dv/dx = a(x)`` (which needs a square root to
    isolate ``v``), never ``dy/dx = f(x)``.

    Every answer is an exact integer by construction (perfect-square /
    Pythagorean-triple construction -- see each tier's comment).
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from sympy import Derivative, Function, Rational, solve, symbols

from ..core import check, check_equivalent, generator, latex_answer, paren_negative, phrase, pick_distractors

x_sym, v_sym, k_sym, m_sym, u_sym = symbols("x v k m u")
mA_sym, mB_sym, vA_sym, vB_sym, vf_sym = symbols("mA mB vA vB vf")

# Solved ONCE at import time, symbolically, so each question only substitutes
# numbers into an already-solved expression (`scripts/check_generator_speed.py`
# flags a per-question `sympy.solve` call over 15ms median -- both exam tiers
# hit that with a fresh solve() per draw; solving once here keeps the
# independent-check route identical while making it cheap per question).
_A_EXAM_SOLUTION = solve(mA_sym * vA_sym + mB_sym * vB_sym - (mA_sym + mB_sym) * vf_sym, vf_sym)[0]
_B_EXAM_SOLUTIONS = solve(v_sym**2 - u_sym**2 - (k_sym / m_sym) * x_sym**2, x_sym)


# =====================================================================
# Local helper (one-off to this module -- see docstring note below).
# =====================================================================


def _keep_distinct(answer: int, *candidates: int) -> list[int]:
    """Drop any candidate that is numerically equal to ``answer``.

    Some misconception candidates collide with the true answer only for specific small-integer draws (an
    expected structural coincidence, not a construction bug), so this is a soft filter, not a hard-failing
    check. This mirrors the locally-scoped ``_add_distinct`` helper already shipped in
    ``mathgen/generators/p12c_trig_apply_implicit_diff_build.py`` -- kept local here too.
    """
    out = []
    for c in candidates:
        if c != answer and c not in out:
            out.append(c)
    return out


def _fmt_int(value) -> str:
    iv = int(value)
    check(iv == value, f"expected an exact integer, got {value!r}")
    return str(iv)


# =====================================================================
# Skill A: momentum, resultant force, Newton's second and third laws
# =====================================================================


@generator(
    "p13b_momentum_force_reaction",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13b_momentum_force_reaction(rng, tier="easy"):
    """See module docstring for the tier ladder. Every answer is an exact
    integer."""
    if tier == "easy":
        m = rng.randint(2, 12)
        v = rng.randint(1, 15)
        direction = rng.choice(("right", "left"))
        sign = 1 if direction == "right" else -1
        answer = sign * m * v
        # Independent check: Fraction arithmetic in a different order
        # (multiply v by m instead of m by v, then apply the sign last).
        check_equivalent(
            Fraction(v) * Fraction(m) * sign, answer, f"easy momentum mismatch: m={m}, v={v}, dir={direction}"
        )

        working = [
            f"Taking right as positive, v = {sign * v} m/s, so p = mv = {m} × {paren_negative(sign * v)} = {answer} kg·m/s."
        ]
        unsigned = m * v  # forgot the direction sign entirely
        added = m + v  # adds instead of multiplies
        cands = _keep_distinct(answer, unsigned, -unsigned, added)

        text = phrase(
            rng,
            (
                "An object of mass {m} kg moves at {v} m/s to the {dir}. Find its "
                "momentum in kg·m/s (take right as positive).",
                "A {m} kg object travels at {v} m/s, moving to the {dir}. Calculate "
                "its momentum in kg·m/s (right is the positive direction).",
                "An object with mass {m} kg has a velocity of {v} m/s directed to "
                "the {dir}. Determine its momentum in kg·m/s (positive direction "
                "is to the right).",
            ),
            m=m,
            v=v,
            dir=direction,
        )

    elif tier == "medium":
        F2 = rng.randint(1, 15)
        F1 = rng.randint(1, 15)
        while F1 == F2:
            F1 = rng.randint(1, 15)
        answer = F1 - F2
        # Independent check: Fraction arithmetic in a different order (negate
        # F2 first, then add to F1).
        check_equivalent(Fraction(F1) + (-Fraction(F2)), answer, f"medium resultant mismatch: F1={F1}, F2={F2}")

        added = F1 + F2  # forgets a force's sign, adds instead
        working = [f"Resultant = {F1} - {F2} = {answer} N (right is positive, left is negative)."]
        only_F1 = F1  # ignores F2 entirely
        only_F2 = F2  # ignores F1 entirely (uses F2's magnitude)
        cands = _keep_distinct(answer, added, only_F1, only_F2)

        text = phrase(
            rng,
            (
                "Two horizontal forces act on an object: {F1} N to the right and "
                "{F2} N to the left. Find the resultant force in N (take right as "
                "positive).",
                "An object experiences two horizontal forces, {F1} N acting right "
                "and {F2} N acting left. Calculate the resultant force in N (right "
                "is positive).",
                "A body has two horizontal forces applied to it: {F1} N pushing "
                "right and {F2} N pushing left. Determine the resultant force in N "
                "(positive means to the right).",
            ),
            F1=F1,
            F2=F2,
        )

    elif tier == "hard":
        m = rng.randint(2, 6)
        # Draw the ANSWER (acceleration/m) directly and uniformly first, then
        # derive k1 (net applied/m) from it by adding the friction/m ratio in
        # the same direction. Deriving k1 first and computing the answer from
        # it (the original construction) concentrated |answer| = 1 at ~46% of
        # draws, because a small |k1| forces a small friction range and
        # k1 - sign(k1)*k2 collapses towards +-1 whenever |k1| is small --
        # measured over 200 seeds (`answer_not_concentrated_at_200_seeds`).
        # Drawing the answer first and adding friction on top keeps it
        # uniform over N equally-likely nonzero values by construction.
        # Range widened from (-8, 9) (16 values, measured 15 distinct / 40
        # seeds with '-5' at 18%) to (-14, 15) (28 values, measured 23
        # distinct / 40 seeds, top share 10%) -- pool-variety check.
        answer = rng.choice([i for i in range(-14, 15) if i != 0])  # = a, in m/s^2
        k2 = rng.randint(1, 5)  # friction / m, > 0
        sign_a = 1 if answer > 0 else -1
        k1 = answer + sign_a * k2  # net applied / m; |k1| > k2 by construction
        net_applied = m * k1
        friction = m * k2
        # Construct F1 (right), F2 (left) so F1 - F2 = net_applied, both > 0.
        # `lower` guarantees F1 = F2 + net_applied >= 1 even when
        # net_applied is strongly negative (a fixed range like [1, 20] can
        # under-shoot and never produce a positive F1 there).
        lower = max(1, 1 - net_applied)
        F2 = rng.randint(lower, lower + 19)
        F1 = F2 + net_applied
        # Independent check: Fraction arithmetic re-derived in a different
        # order -- combine the two applied forces first (F1 minus F2), then
        # subtract the friction's OWN derived sign (opposite to the net of
        # the two applied forces), rather than the m*k1/m*k2 route above.
        net_applied_fr = Fraction(F1) - Fraction(F2)
        friction_sign = 1 if net_applied_fr > 0 else -1
        net_force_fr = net_applied_fr - friction_sign * Fraction(friction)
        check_equivalent(
            net_force_fr / Fraction(m), answer, f"hard acceleration mismatch: F1={F1}, F2={F2}, f={friction}, m={m}"
        )

        fric_dir = "left" if net_applied > 0 else "right"
        working = [
            f"Applied force = {F1} - {F2} = {net_applied} N.",
            f"Friction opposes the motion, so it acts {fric_dir}: net force = {net_applied} {'-' if net_applied > 0 else '+'} {friction} = {net_applied - (1 if net_applied > 0 else -1) * friction} N.",
            f"a = {net_applied - (1 if net_applied > 0 else -1) * friction} / {m} = {answer} m/s².",
        ]
        forgot_friction = k1  # ignores friction entirely
        sign_error_friction = k1 + (1 if k1 > 0 else -1) * k2  # adds friction the wrong way
        forgot_divide_by_mass = net_applied - (1 if k1 > 0 else -1) * friction  # net FORCE, not a
        cands = _keep_distinct(answer, forgot_friction, sign_error_friction, forgot_divide_by_mass)

        text = phrase(
            rng,
            (
                "Three collinear horizontal forces act on an object of mass {m} kg: "
                "{F1} N to the right, {F2} N to the left, and a resisting friction "
                "force of magnitude {f} N that opposes the object's resulting "
                "direction of motion. Find the resulting acceleration of the object "
                "in m/s² (take right as positive).",
                "An object of mass {m} kg has three collinear horizontal forces "
                "acting on it: {F1} N pushing right, {F2} N pushing left, and "
                "friction of magnitude {f} N opposing whichever direction the "
                "object ends up accelerating in. Find the resulting acceleration in "
                "m/s² (right positive).",
                "A {m} kg object experiences {F1} N to the right, {F2} N to the "
                "left, and a friction force of magnitude {f} N that resists the "
                "direction the object ends up moving in. Find the resulting "
                "acceleration in m/s² (positive is to the right).",
            ),
            m=m,
            F1=F1,
            F2=F2,
            f=friction,
        )

    else:  # exam
        # Draw the ANSWER (v_final, the common post-collision velocity)
        # directly and uniformly first from a wide integer range, then derive
        # vA and vB algebraically so that momentum conservation holds EXACTLY
        # by construction -- no reject-and-redraw needed at all.
        #
        # For any mA, mB with d = gcd(mA, mB), mA/d and mB/d are integers, and
        #     vA = v_final + (mB/d)*n,  vB = v_final - (mA/d)*n
        # satisfy mA*vA + mB*vB = (mA+mB)*v_final for EVERY integer n (expand
        # and the n terms cancel). `n` is set to the smallest positive integer
        # that forces vA > 0 (A moving right) and vB < 0 (B moving toward A,
        # per the stated convention) simultaneously.
        #
        # The old construction rejected-and-redrew mA, mB, vA, vB until
        # (mA+mB) happened to divide mA*vA+mB*vB -- small integer quotients
        # pass that divisibility test far more often than large ones, which
        # concentrated v_final at '1' 25% of the time over 40 seeds
        # (measured via the pool-variety check). Drawing v_final first and
        # deriving vA/vB removes the rejection (and its concentration)
        # entirely: every (v_final, mA, mB, n) draw is accepted. Range widened
        # to (-15, 16) and mA/mB to [2, 6] (was [1, 6]; mA=mB=1 forced very
        # large n for some v_final, pushing vA/vB implausibly high) --
        # measured 24 distinct / 40 seeds, top share 12.5%.
        v_final = rng.choice([i for i in range(-15, 16) if i != 0])
        mA = rng.randint(2, 6)
        mB = rng.randint(2, 6)
        d = gcd(mA, mB)
        mA_d, mB_d = mA // d, mB // d
        n = max(1, v_final // mA_d + 1, (-v_final) // mB_d + 1)
        vA = v_final + mB_d * n
        vB = v_final - mA_d * n
        check(
            vA > 0 and vB < 0,
            f"exam: expected vA>0, vB<0, got vA={vA}, vB={vB} (v_final={v_final}, mA={mA}, mB={mB}, n={n})",
        )

        # Independent check: sympy.solve of mA*vA + mB*vB - (mA+mB)*v == 0,
        # independent of the arithmetic route above.
        solved_val = _A_EXAM_SOLUTION.subs({mA_sym: mA, mB_sym: mB, vA_sym: vA, vB_sym: vB})
        check_equivalent(solved_val, v_final, f"exam conservation mismatch: mA={mA}, mB={mB}, vA={vA}, vB={vB}")

        answer = v_final
        working = [
            f"Momentum before = {mA} × {vA} + {mB} × ({vB}) = {mA * vA + mB * vB} kg·m/s.",
            f"Momentum after = ({mA} + {mB}) v = {mA + mB}v, so v = {mA * vA + mB * vB} / {mA + mB} = {v_final} m/s.",
        ]
        total_p = mA * vA + mB * vB
        wrong_denominator = None
        if mA != mB and total_p % (mA - mB) == 0:
            wrong_denominator = total_p // (mA - mB)  # mA - mB instead of mA + mB
        forgot_B_term = None
        if (mA * vA) % (mA + mB) == 0:
            forgot_B_term = (mA * vA) // (mA + mB)  # forgets B's momentum term
        sign_flip = -v_final
        ignores_conservation = vA  # assumes A's own speed persists
        cands_all = [c for c in (wrong_denominator, forgot_B_term, sign_flip, ignores_conservation) if c is not None]
        cands = _keep_distinct(answer, *cands_all)

        text = phrase(
            rng,
            (
                "Two objects moving towards each other collide and stick together "
                "(a perfectly inelastic collision). Object A has mass {mA} kg "
                "moving at {vA} m/s in the positive direction; object B has mass "
                "{mB} kg moving at {absVB} m/s in the negative direction. By "
                "Newton's third law, the force object A exerts on object B during "
                "the collision is equal in magnitude and opposite in direction to "
                "the force object B exerts on object A. Using conservation of "
                "momentum, find the common velocity of the combined objects after "
                "the collision, in m/s (positive direction as stated).",
                "Object A (mass {mA} kg, velocity {vA} m/s in the positive "
                "direction) collides with object B (mass {mB} kg, velocity {absVB} "
                "m/s in the negative direction) and the two stick together. By "
                "Newton's third law, the collision force each object exerts on the "
                "other is equal and opposite. Use conservation of momentum to find "
                "the common velocity after the collision, in m/s (positive as "
                "stated).",
                "Two objects collide and stick together (perfectly inelastic): "
                "object A, {mA} kg at {vA} m/s in the positive direction, and "
                "object B, {mB} kg at {absVB} m/s in the negative direction. "
                "Newton's third law means the force A exerts on B during the "
                "collision equals in magnitude, and opposes in direction, the force "
                "B exerts on A. Find the common velocity of the combined objects "
                "after the collision, in m/s (positive direction as stated), using "
                "conservation of momentum.",
            ),
            mA=mA,
            mB=mB,
            vA=vA,
            absVB=-vB,
        )

    answer_text = _fmt_int(answer)
    cands_text = [_fmt_int(c) for c in cands]
    distractors = pick_distractors(answer_text, cands_text)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


# =====================================================================
# Skill B: constant vs non-constant force
# =====================================================================

# Primitive Pythagorean triples (odd_leg, even_leg, hyp). even_leg is always
# divisible by 4, which several tiers below rely on to keep a misconception
# candidate an exact integer.
_TRIPLES = (
    (3, 4, 5),
    (5, 12, 13),
    (15, 8, 17),
    (7, 24, 25),
    (21, 20, 29),
    (9, 40, 41),
)


@generator(
    "p13b_constant_nonconstant_force",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13b_constant_nonconstant_force(rng, tier="easy"):
    """See module docstring for the tier ladder. Every answer is an exact
    integer, constructed via a perfect square (easy/medium) or a Pythagorean
    triple (medium/exam -- see ``_TRIPLES``)."""
    if tier == "easy":
        # a = 2*t, s = t*n**2  =>  2*a*s = (2*t*n)**2, so v = 2*t*n exactly.
        # t widened from [1, 5] to [1, 14] (n unchanged at [1, 4] to keep
        # s = t*n**2 from blowing up quadratically) -- measured 11 distinct
        # answers over 40 seeds at [1,5]/[1,4] ('6' at 18%), 23 distinct at
        # [1,14]/[1,4], top share 12.5%.
        t = rng.randint(1, 14)
        n = rng.randint(1, 4)
        m = rng.randint(1, 8)
        a = 2 * t
        s = t * n * n
        v = 2 * t * n
        F = m * a
        # Independent check: Fraction re-derivation (a = F/m, computed from
        # the stated F and m, not from the construction's own `a`).
        a_check = Fraction(F, m)
        check_equivalent(a_check, a, f"easy a mismatch: F={F}, m={m}")
        check(v * v == 2 * a * s, f"easy v**2 mismatch: v={v}, a={a}, s={s}")

        working = [
            f"a = F/m = {F}/{m} = {a} m/s².",
            f"v² = u² + 2as = 0 + 2 × {a} × {s} = {2 * a * s}, so v = {v} m/s.",
        ]
        forgot_sqrt = 2 * a * s  # confuses v with v**2 = 2*a*s
        sign_flip = -v
        cands = _keep_distinct(v, forgot_sqrt, sign_flip)

        text = phrase(
            rng,
            (
                "An object of mass {m} kg experiences a constant net force of "
                "{F} N in its direction of motion. Starting from rest, find its "
                "speed in m/s after it has travelled {s} m.",
                "A {m} kg object starts from rest and experiences a constant net "
                "force of {F} N in its direction of motion. Find its speed in m/s "
                "once it has travelled {s} m.",
                "Starting from rest, an object of mass {m} kg is acted on by a "
                "constant net force of {F} N in its direction of motion. Find its "
                "speed in m/s after travelling {s} m.",
            ),
            m=m,
            F=F,
            s=s,
        )
        answer = v

    elif tier == "medium":
        odd_leg, even_leg, hyp = rng.choice(_TRIPLES)
        # k widened from [1, 3] to [1, 4] -- measured 18 distinct / 40 seeds
        # at [1,3] (under the 20 bar), 20 distinct, top share 7.5% at [1,4].
        k = rng.randint(1, 4)
        u = odd_leg * k
        w = even_leg * k  # w is even (even_leg divisible by 4)
        v = hyp * k
        a = 2
        s = (w // 2) ** 2
        m = rng.randint(1, 8)
        F = m * a
        check(u * u + w * w == v * v, f"medium triple mismatch: u={u}, w={w}, v={v}")
        # Independent check: Fraction re-derivation of a from F, m.
        a_check = Fraction(F, m)
        check_equivalent(a_check, a, f"medium a mismatch: F={F}, m={m}")
        check(u * u + 2 * a * s == v * v, f"medium v**2 mismatch: u={u}, a={a}, s={s}, v={v}")

        working = [
            f"a = F/m = {F}/{m} = {a} m/s².",
            f"v² = u² + 2as = {u}² + 2 × {a} × {s} = {u * u} + {2 * a * s} = {v * v}, so v = {v} m/s.",
        ]
        forgot_u = w  # sqrt(2*a*s) alone, drops the u**2 term
        forgot_sqrt = v * v  # confuses v with v**2
        sign_flip = -v
        cands = _keep_distinct(v, forgot_u, forgot_sqrt, sign_flip)

        text = phrase(
            rng,
            (
                "An object of mass {m} kg experiences a constant net force of "
                "{F} N in its direction of motion. It already has a speed of "
                "{u} m/s in that direction. Find its speed in m/s after it has "
                "travelled a further {s} m.",
                "A {m} kg object, already moving at {u} m/s, experiences a "
                "constant net force of {F} N in its direction of motion. Find its "
                "speed in m/s once it has travelled a further {s} m.",
                "An object of mass {m} kg moving at {u} m/s experiences a "
                "constant net force of {F} N in its direction of motion. Find its "
                "speed in m/s after travelling {s} more metres.",
            ),
            m=m,
            F=F,
            u=u,
            s=s,
        )
        answer = v

    elif tier == "hard":
        c = rng.randint(2, 5)
        m = rng.randint(1, 6)
        s = rng.randint(1, 10)
        k = m * c * c
        v = c * s
        # Independent check: sympy, solving v*dv/dx = (k/m)*x with v(0)=0 as
        # a SEPARATE symbolic derivation from the closed form above.
        v_fn = Function("v")
        ode_lhs = v_fn(x_sym) * Derivative(v_fn(x_sym), x_sym) - Rational(k, m) * x_sym
        # v(x)**2 = (k/m)*x**2 + C, C = 0 from v(0) = 0 (rest); take the
        # positive root, which is v = sqrt(k/m) * x = c * x since k = m*c**2.
        v_expr_check = (Rational(k, m) * x_sym**2) ** Rational(1, 2)
        check_equivalent(v_expr_check.subs(x_sym, s), v, f"hard v mismatch: k={k}, m={m}, s={s}")
        check(
            ode_lhs.subs(v_fn(x_sym), c * x_sym).doit() == 0,
            f"hard: v=c*x does not satisfy v*dv/dx=(k/m)*x for k={k}, m={m}",
        )

        working = [
            f"Work done = integral of {k}x from 0 to {s} = {k}/2 × {s}² = {k * s * s / 2:g} J.",
            f"This equals the gain in kinetic energy: (1/2) × {m} × v² = {k * s * s / 2:g}, so v² = {c * c * s * s} and v = {v} m/s.",
        ]
        forgot_sqrt = v * v  # v**2 instead of v
        sign_flip = -v
        force_at_rest = 0  # treats F(0)=0 as the constant force throughout
        cands = _keep_distinct(v, forgot_sqrt, sign_flip, force_at_rest)

        text = phrase(
            rng,
            (
                "An object of mass {m} kg, initially at rest at x = 0, "
                "experiences a force F(x) = {k}·x N (in the direction of "
                "increasing x) while it is at position x m. Find its speed in "
                "m/s when it reaches x = {s} m.",
                "Starting from rest at x = 0, an object of mass {m} kg "
                "experiences a force F(x) = {k}·x N (in the direction of "
                "increasing x) at position x m. Find its speed in m/s when it "
                "reaches x = {s} m.",
                "An object of mass {m} kg starts from rest at x = 0 and "
                "experiences a force F(x) = {k}·x N (direction of increasing "
                "x) at position x m. Find its speed, in m/s, when x = {s} m.",
            ),
            m=m,
            k=k,
            s=s,
        )
        answer = v

    else:  # exam
        odd_leg, even_leg, hyp = rng.choice(_TRIPLES)
        # `c` used to multiply BOTH the answer's scale and v_target
        # (`u = odd_leg*c*j2`, `v_target = hyp*c*j2`, `answer = even_leg*j2`)
        # -- v_target scaled with c for zero benefit to answer variety
        # (worst case was hyp=41, c=5, j2=3 => v_target=615 m/s, measured
        # before the fix). Decoupled: c now scales `k = m*c*c` only (still varies
        # the F(x) = k*x coefficient the learner reads) and divides back OUT
        # of the answer (`answer = even_leg*j2 // c`), while u and v_target
        # no longer carry a `c` factor at all -- v_target = hyp*j2 exactly,
        # independent of c, so widening j2 for variety no longer inflates
        # the ceiling. `even_leg*j2` is always divisible by c in {1, 2}
        # because every `even_leg` in `_TRIPLES` is divisible by 4 (see that
        # comment). Verified: u**2 + (c*answer)**2 == (odd_leg*j2)**2 +
        # (even_leg*j2)**2 == j2**2*hyp**2 == v_target**2 for either c,
        # by the check() below (not just asserted here).
        # Measured 12 distinct / 40 seeds ('40' at 25%) at the old c in
        # [2,5]/j2 in [1,3]; 25 distinct, top share 7.5%, worst-case
        # v_target = 41*10 = 410 (vs. 615 before) at c in {1,2}/j2 in [1,10].
        c = rng.choice((1, 2))
        j2 = rng.randint(1, 10)
        m = rng.randint(1, 6)
        k = m * c * c
        u = odd_leg * j2
        v_target = hyp * j2
        answer = even_leg * j2 // c  # x
        check(
            u * u + (c * answer) ** 2 == v_target**2,
            f"exam triple mismatch: u={u}, c={c}, x={answer}, v_target={v_target}",
        )
        # Independent check: sympy.solve on
        # v_target**2 - u**2 - (k/m)*x**2 == 0, a separate symbolic route.
        subs = {u_sym: u, k_sym: k, m_sym: m, v_sym: v_target}
        candidates_solved = [s.subs(subs) for s in _B_EXAM_SOLUTIONS]
        positive_solutions = [s for s in candidates_solved if s.is_positive]
        check(len(positive_solutions) == 1, f"exam: expected exactly one positive root, got {positive_solutions}")
        check_equivalent(positive_solutions[0], answer, f"exam x mismatch: u={u}, k={k}, m={m}, v_target={v_target}")

        # Treats a as constant (a = c**2), the WRONG SUVAT form:
        # x = (v_target**2 - u**2) / (2*c**2). Integer since even_leg is
        # divisible by 4.
        working = [
            f"Work done = integral of {k}x from 0 to x = {k}x²/2 and equals the change in kinetic energy: {k}x²/2 = (1/2) × {m} × ({v_target}² - {u}²).",
            f"{k}x² = {m} × {v_target**2 - u**2}, so x² = {m * (v_target**2 - u**2) // k}.",
            f"x = {answer} m.",
        ]
        wrong_suvat = (v_target**2 - u**2) // (2 * c * c)
        forgot_u = v_target // c  # ignores the initial-speed term entirely
        sign_flip = -answer
        cands = _keep_distinct(answer, wrong_suvat, forgot_u, sign_flip)

        text = phrase(
            rng,
            (
                "An object of mass {m} kg, at x = 0, has an initial speed of "
                "{u} m/s in the direction of increasing x. It then experiences a "
                "force F(x) = {k}·x N while at position x m. Find the "
                "displacement x (in m) at which its speed reaches {v_target} m/s.",
                "At x = 0, an object of mass {m} kg has an initial speed of "
                "{u} m/s in the direction of increasing x, and experiences a "
                "force F(x) = {k}·x N at position x m. Find the displacement "
                "x (in m) at which the speed reaches {v_target} m/s.",
                "An object of mass {m} kg starts at x = 0 with an initial speed "
                "of {u} m/s in the direction of increasing x and experiences a "
                "force F(x) = {k}·x N at position x m. Find the displacement "
                "x, in m, at which its speed reaches {v_target} m/s.",
            ),
            m=m,
            u=u,
            k=k,
            v_target=v_target,
        )

    answer_text = _fmt_int(answer)
    cands_text = [_fmt_int(c) for c in cands]
    distractors = pick_distractors(answer_text, cands_text)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )
