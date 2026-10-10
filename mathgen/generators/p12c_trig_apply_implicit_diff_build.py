"""Trigonometric derivative applications; implicit differentiation.

Skill 1: senior-use-trigonometric-functions-and-their-derivatives-to-solve
(ACMMM103, topic ``p12c_trig_deriv_apply``, answer: number).
Skill 2: senior-use-implicit-differentiation-to-determine-the-gradient-of
(ACMSM128, topic ``p12c_implicit_diff_gradient``, answer: expression).

**The ``differentiate-trig`` boundary (skill 1).**
``differentiate-trig`` (``skills/differentiation/differentiate-trig.yaml``,
``mathgen.generators.calculus.differentiation(kinds=("trig",...))``) asks the
learner to differentiate a bare ``a*sin(bx) + c*cos(dx))`` expression -- no
scenario, no stated practical quantity, answer is always the symbolic
derivative itself. Skill 1 below never asks "differentiate" in the abstract:
every question states a practical oscillating scenario with a model already
given as ``H(t) = a*sin(b*t) + c``, and always asks for a NUMBER (a rate at a
stated time, a time at which the rate hits a target, or the larger of two
stated rates) -- never the bare symbolic derivative.

**The ``senior-find-the-slope-of-a-tangent-and-the`` boundary (skill 1).**
That skill (``skills/differentiation/senior-find-the-slope-of-a-tangent-and-the.yaml``,
topic ``find_the_slope_of_a_tangent_and_the``) is polynomial-only -- its
generator never touches ``sin``/``cos`` -- and always asks for the tangent
LINE (a slope-intercept expression), never a rate. Skill 1 below never asks
for a tangent line or a line equation; its answer is always a number.

**Design choice: sin-only, never cos-based.** The specification allowed
``H(t) = a*sin(bt) + c`` "(or with cos)". A cos-based model's derivative is
``-a*b*sin(bt)``, which is IDENTICALLY ZERO at ``t = 0`` for every ``a``,
``b`` -- the exact instant the easy tier asks about -- so a cos-based easy
question would always have the answer 0. Sin-only avoids that degenerate
collision entirely (``a*b*cos(0) = a*b``, never 0 since ``a``, ``b`` are
drawn positive).

**Design choice: medium's "nice" angle is restricted to a RATIONAL-cosine
set.** The specification says "a multiple of pi/6 or pi/4" as an example of a
nice angle; pi/4 and pi/6 themselves give an irrational cos (surd), which
would force a genuinely non-exact answer that must be reported to a stated
decimal precision. Restricting medium's target angle to
``{0, pi/3, pi/2, 2*pi/3, pi, 4*pi/3, 3*pi/2, 5*pi/3}`` keeps every medium
answer an EXACT multiple of one-half (cos in ``{1, 1/2, 0, -1/2, -1}``), so
medium never needs a rounding statement at all -- the "round to a stated
number of decimal places" policy is reserved for hard/exam, where the
answer is a TIME (an inverse-cosine value), genuinely irrational in general.

**The implicit-differentiation ``solve-ode-exponential`` boundary (skill 2).**
``solve-ode-exponential`` (``skills/differential-equations/solve-ode-exponential.yaml``)
solves ``dy/dx = ky`` for ``y`` as an EXPLICIT function of ``x``. Skill 2
below never solves a differential equation and never produces ``y`` as a
function of ``x`` -- the curve is given as an implicit relation
``F(x, y) = constant``, and the question is to differentiate both sides with
respect to ``x`` (treating ``y`` as ``y(x)`` via the chain rule) and solve
the resulting linear-in-``y'`` equation for ``dy/dx``.

As a fallback, this generator instead moves the PRODUCT-RULE cross term to medium
and builds two further, genuinely different escalations for hard and exam:

- **easy**: ``A*x**2 + B*y**2 = C`` -- power rule (x-term) + basic chain rule
  (y-term, coefficient ``u' = 1`` never appears since there is no product).
- **medium**: ``A*x**2 + B*x*y + C*y**2 = D`` -- adds an ``xy`` cross term,
  the product rule's SIMPLEST case (``u = x``, ``u' = 1`` trivially).
- **hard**: ``A*x**2*y + B*y**3 = C`` -- the product rule's GENERAL case
  (``u = x**2``, ``u' = 2x`` -- the learner must differentiate BOTH factors of
  the product, not carry a trivial ``u' = 1``), combined with the chain+power
  rule on a CUBED ``y`` term. Genuinely different decision from medium: at
  medium, "differentiate the x-factor" is a no-op; at hard, it is not.
- **exam**: ``A*x**3 + B*x*y**2 + C*y**3 = D`` -- a product rule term where
  the SECOND factor itself needs the chain rule (``d/dx[y**2] = 2*y*y'``,
  nested inside the product rule), a further genuinely different combined
  manoeuvre from hard's (where the product rule's second factor was the bare
  ``y``, needing no further rule).
"""

from __future__ import annotations

from sympy import Rational, acos, cos, diff, idiff, pi, sin, sympify, symbols

from ..core import (
    check,
    check_equivalent,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    symbolically_equal,
)

t, x, y = symbols("t x y")


# =====================================================================
# Skill 1: trigonometric-derivative practical applications
# =====================================================================

# Scenarios: (noun phrase, function symbol, value unit, time unit).
_SCENARIOS = (
    ("the height above the ground of a point on a Ferris wheel", "h", "metres", "seconds"),
    ("the water level at a tidal gauge", "h", "metres", "hours"),
    ("the temperature inside a greenhouse", "T", "degrees Celsius", "hours"),
    ("the horizontal displacement of a swinging pendulum bob", "x", "centimetres", "seconds"),
    ("the height of a piston in an engine cylinder", "h", "millimetres", "seconds"),
)

# Singular form of each scenario unit, so a target of magnitude 1 never
# reads "1 centimetres" (the "singular_count_plural" text defect).
_UNIT_SINGULAR = {
    "metres": "metre",
    "degrees Celsius": "degree Celsius",
    "centimetres": "centimetre",
    "millimetres": "millimetre",
}


def _rate_unit_for(count, unit: str, time_unit: str) -> str:
    """ "<unit>/<time_unit>", with the value unit singularised when the
    stated count's magnitude is exactly 1."""
    word = _UNIT_SINGULAR[unit] if abs(count) == 1 else unit
    return f"{word}/{time_unit}"


# Nice angles (in units of pi) with an EXACT rational cosine, excluding 0
# (that is the easy tier's own t = 0 case). Each maps to a distinct cos
# value, so two DIFFERENT angles from this set can never tie on cos.
_NICE_ANGLES = (
    (Rational(1, 3), Rational(1, 2)),  # pi/3 -> cos = 1/2
    (Rational(1, 2), Rational(0)),  # pi/2 -> cos = 0
    (Rational(2, 3), Rational(-1, 2)),  # 2pi/3 -> cos = -1/2
    (Rational(1, 1), Rational(-1)),  # pi -> cos = -1
)
# Exam draws its PAIR of times from a pool excluding cos = 0: with all four `_NICE_ANGLES`, the pi/2
# entry forces the LARGER rate to be exactly 0 whenever it is paired with either negative-cos entry
# (2 of 6 possible pairs), concentrating the exam answer on 0 at ~33% over 200 seeds -- measured,
# over the 25% answer-concentration bar. Dropping pi/2 from the exam pool removes that structural
# collapse; medium still draws pi/2.
_EXAM_NICE_ANGLES = tuple(pair for pair in _NICE_ANGLES if pair[1] != 0)
# Medium draws ONE angle, so repeated cos values are harmless there; the two extra angles below
# dilute pi/2's cos = 0 from 1 in 4 to 1 in 6.
_MEDIUM_NICE_ANGLES = _NICE_ANGLES + (
    (Rational(4, 3), Rational(-1, 2)),  # 4pi/3 -> cos = -1/2
    (Rational(5, 3), Rational(1, 2)),  # 5pi/3 -> cos = 1/2
)
# Targets (k) for the hard-tier inverse lookup, excluding k=1 (angle 0,
# duplicates the easy tier's own answer shape).
_HARD_TARGETS = (Rational(-1), Rational(-1, 2), Rational(0), Rational(1, 2))


def _fmt_option(value) -> str:
    """Any numeric sympy value (rational or irrational) as a plain decimal: an exact whole number renders with no
    decimal point at all; an exact half-integer (every medium/easy rate and evaluation-tier value here IS one,
    since cos of every nice angle in ``_NICE_ANGLES`` is a multiple of 1/2) renders to exactly 1 decimal
    place; anything else (an irrational "evaluates H instead of H'" distractor) rounds to 1 decimal place too,
    so the answer is never the only decimal-shaped option on offer --. Never a needless trailing ``.0`` on a
    whole number."""
    value = sympify(value)
    if value.is_Rational and value.q == 1:
        return str(value.p)
    return f"{float(value):.1f}"


def _add_distinct(pool: list, answer, *candidates) -> None:
    """Append each candidate to ``pool`` only if it is genuinely distinct
    from ``answer`` -- some structural collisions here are EXPECTED (a
    dropped-b-factor misconception equals the true rate whenever b = 1; a
    sign-flip misconception equals the true rate whenever the rate is 0), so
    silently dropping the colliding candidate (rather than hard-failing via
    `check_distinct`, which is for catching a construction BUG, not an
    expected structural coincidence) is the correct response here."""
    for candidate in candidates:
        if not symbolically_equal(answer, candidate):
            pool.append(candidate)


def _pi_text(coeff: Rational) -> str:
    """A rational multiple of pi as learner-facing text: 0 -> "0",
    1 -> "π", 1/3 -> "π/3", 2/3 -> "2π/3"."""
    coeff = Rational(coeff)
    if coeff == 0:
        return "0"
    num, den = coeff.p, coeff.q
    num_part = "π" if num == 1 else f"{num}π"
    return num_part if den == 1 else f"{num_part}/{den}"


@generator(
    "p12c_trig_deriv_apply",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12c_trig_deriv_apply(rng, tier="easy"):
    """H(t) = a*sin(b*t) + c models a stated practical oscillating quantity
    (t >= 0, t in the scenario's own time unit). dH/dt = a*b*cos(b*t).

    easy:   rate of change AT t = 0 (dH/dt|_0 = a*b, no trig evaluation).
    medium: rate of change at a stated t where b*t is an exact-cosine angle.
    hard:   find the time t in [0, pi/b] (cos monotonic there, so the target
            has exactly one solution) at which the rate equals a stated
            exact target; reported to 2 decimal places.
    exam:   compare the rate at two stated (distinct, exact-angle) times and
            report the value of the LARGER rate -- a forward
            evaluate-and-compare, a genuinely different decision from hard's
    inverse lookup.
    """
    noun, sym, unit, time_unit = rng.choice(_SCENARIOS)
    a = rng.randint(1, 9)
    b = rng.randint(1, 4)
    c = rng.randint(-9, 9)

    a_text = "" if a == 1 else f"{a}·"
    b_text = "t" if b == 1 else f"{b}·t"
    sin_term = f"{a_text}sin({b_text})"
    c_text = "" if c == 0 else f" {'+' if c >= 0 else '-'} {abs(c)}"
    model_text = f"{sym}(t) = {sin_term}{c_text}"
    rate_unit = f"{unit}/{time_unit}"

    H = a * sin(b * t) + c
    dH = diff(H, t)  # a*b*cos(b*t) -- differentiated symbolically, the
    # independent route against which every evaluated tier is checked.

    if tier == "easy":
        rate = Rational(a * b)
        check_equivalent(dH.subs(t, 0), rate, f"easy dH/dt(0) mismatch: a={a}, b={b}")
        answer_text = _fmt_option(rate)

        value_instead = H.subs(t, 0)  # evaluates H, not H'
        sign_flip = -rate  # collides with `rate` when rate = 0
        no_chain_factor = Rational(a)  # dropped the b factor; collides when b = 1
        cands_val = []
        _add_distinct(cands_val, rate, value_instead, sign_flip, no_chain_factor)
        cands = [_fmt_option(v) for v in cands_val]
        text = (
            f"{noun[0].upper()}{noun[1:]} is modelled by {model_text}, where t is the time "
            f"in {time_unit} since t = 0 ({sym} in {unit}). Find the instantaneous rate of "
            f"change of {sym} with respect to t at t = 0, in {rate_unit}."
        )

    elif tier == "medium":
        angle_coeff, cos_val = rng.choice(_MEDIUM_NICE_ANGLES)
        t1 = angle_coeff / b
        rate = Rational(a * b) * cos_val
        check_equivalent(
            dH.subs(t, angle_coeff * pi / b), rate, f"medium dH/dt mismatch: a={a}, b={b}, angle={angle_coeff}"
        )
        answer_text = _fmt_option(rate)

        value_instead = H.subs(t, angle_coeff * pi / b)  # evaluates H, not H'
        sign_flip = -rate  # collides with `rate` when rate = 0 (cos_val = 0)
        no_chain_factor = Rational(a) * cos_val  # dropped the b factor; collides when b = 1
        ignores_trig = Rational(a * b)  # ignores cos entirely, as if cos(bt) = 1 always;
        # distinct from `rate` on every draw since `_NICE_ANGLES` excludes
        # cos = 1 (angle 0, the easy tier's own case) at medium.
        cands_val = []
        _add_distinct(cands_val, rate, value_instead, sign_flip, no_chain_factor, ignores_trig)
        cands = [_fmt_option(v) for v in cands_val]
        text = (
            f"{noun[0].upper()}{noun[1:]} is modelled by {model_text}, where t is the time "
            f"in {time_unit} since t = 0 ({sym} in {unit}). Find the instantaneous rate of "
            f"change of {sym} with respect to t at t = {_pi_text(t1)} {time_unit}, in {rate_unit}."
        )

    elif tier == "hard":
        k = rng.choice(_HARD_TARGETS)
        target = Rational(a * b) * k
        # angle = arccos(k) in [0, pi], t = angle / b, the UNIQUE solution in
        # [0, pi/b] since cos(b*t) is strictly monotonic there.
        angle = acos(k)
        t_true = float(angle) / b
        t_rounded = round(t_true, 2)
        # Not sitting on the rounding boundary: the nearest exact multiple of
        # 0.005 (the midpoint between two 2-d.p. roundings) must be
        # comfortably far from `t_true`, not merely that `round()` picked a
        # side (which is true by definition and proves nothing).
        nearest_boundary = round(t_true * 200) / 200
        check(abs(t_true - nearest_boundary) > 0.0002, f"hard t too close to a rounding boundary: {t_true}")
        # Independent check: cos(b * (angle/b)) recovers k exactly.
        check_equivalent(cos(b * (angle / b)), k, f"hard target/angle mismatch: b={b}, k={k}")
        bound = float(pi) / b
        answer_text = f"{t_rounded:.2f}"

        target_text = _fmt_option(target)
        # Misconception: solves H(t) = target (a VALUE equation) instead of
        # dH/dt = target (the RATE equation) -- forgets to divide the
        # inverse-cosine angle by b when isolating t.
        forgot_divide_by_b = round(float(angle), 2)
        # Misconception: sign error on the target -- looks up arccos(-k)
        # instead of arccos(k).
        wrong_angle = acos(-k) if k != 0 else (pi - angle)
        sign_error_t = round(float(wrong_angle) / b, 2)
        cands = []
        if b != 1 and abs(forgot_divide_by_b - t_rounded) > 0.02:
            cands.append(f"{forgot_divide_by_b:.2f}")
        if abs(sign_error_t - t_rounded) > 0.02:
            cands.append(f"{sign_error_t:.2f}")
        cands.append(f"{-t_rounded:.2f}")
        text = (
            f"{noun[0].upper()}{noun[1:]} is modelled by {model_text}, where t is the time "
            f"in {time_unit} since t = 0 ({sym} in {unit}). For 0 ≤ t ≤ {bound:.2f}, find "
            f"the time t at which the rate of change of {sym} equals {target_text} "
            f"{_rate_unit_for(target, unit, time_unit)}. Give your answer correct to 2 decimal places."
        )

    else:  # exam
        angle1, cos1 = rng.choice(_EXAM_NICE_ANGLES)
        remaining = [pair for pair in _EXAM_NICE_ANGLES if pair[0] != angle1]
        angle2, cos2 = rng.choice(remaining)
        t1, t2 = angle1 / b, angle2 / b
        rate1 = Rational(a * b) * cos1
        rate2 = Rational(a * b) * cos2
        check_equivalent(dH.subs(t, angle1 * pi / b), rate1, f"exam dH/dt(t1) mismatch: angle1={angle1}")
        check_equivalent(dH.subs(t, angle2 * pi / b), rate2, f"exam dH/dt(t2) mismatch: angle2={angle2}")
        check(rate1 != rate2, f"exam rates collided: a={a}, b={b}, angle1={angle1}, angle2={angle2}")
        larger = max(rate1, rate2)
        smaller = min(rate1, rate2)
        answer_text = _fmt_option(larger)

        value1 = H.subs(t, angle1 * pi / b)
        value2 = H.subs(t, angle2 * pi / b)
        value_instead = max(value1, value2)  # evaluates H, not H', at both times
        sign_flip = -larger  # collides with `smaller` when the two rates are symmetric about 0
        cands_val = []
        _add_distinct(cands_val, larger, smaller, value_instead, sign_flip)
        cands = [_fmt_option(v) for v in cands_val]
        text = (
            f"{noun[0].upper()}{noun[1:]} is modelled by {model_text}, where t is the time "
            f"in {time_unit} since t = 0 ({sym} in {unit}). Compare the instantaneous rate of "
            f"change of {sym} at t = {_pi_text(t1)} {time_unit} with the rate at t = {_pi_text(t2)} "
            f"{time_unit}. State the value of the LARGER of the two rates, in {rate_unit}."
        )

    if tier != "hard":
        # Shape every option like the key: a whole-number key keeps whole
        # numbers (an irrational "evaluates H, not H'" value is rounded to
        # one), a half-integer key shows every option to one decimal place
        # ("-18.0" beside "4.5"), so the decimal point never gives it away.
        if "." in answer_text:
            cands = [f"{float(c):.1f}".replace("-0.0", "0.0") for c in cands]
        else:
            cands = [str(round(float(c))) for c in cands]
    distractors = pick_distractors(answer_text, cands)

    return dict(
        text=text,
        latex=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )


# =====================================================================
# Skill 2: implicit differentiation for the gradient of implicit curves
# =====================================================================


def _typed(expr) -> str:
    return str(expr).replace("**", "^")


def _term(coef: int, var_text: str) -> str:
    """A coefficient-times-variable term for the question prose: omits a
    coefficient of 1 (never "1*x^2" -- a student would just write "x^2")."""
    return var_text if coef == 1 else f"{coef}*{var_text}"


@generator(
    "p12c_implicit_diff_gradient",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p12c_implicit_diff_gradient(rng, tier="easy"):
    """An implicit curve F(x, y) = constant. Find dy/dx via implicit
    differentiation. See module docstring for the tier ladder and the
    medium-redesign decision.
    """
    if tier == "easy":
        A = rng.randint(1, 6)
        B = rng.randint(1, 6)
        C = rng.randint(10, 60)
        lhs = A * x**2 + B * y**2
        rhs = C
        N, D = A * x, B * y
        true_yprime = -N / D
        eq_text = f"{_term(A, 'x^2')} + {_term(B, 'y^2')} = {C}"
        sign_flip = N / D
        forgot_yprime_factor = Rational(-A, B) * x  # missing y in the denominator
        dropped_B = -A * x / y
        cands_expr = [sign_flip, forgot_yprime_factor, dropped_B]

    elif tier == "medium":
        A = rng.randint(1, 6)
        B = rng.randint(1, 6)
        C = rng.randint(1, 6)
        D = rng.randint(10, 60)
        lhs = A * x**2 + B * x * y + C * y**2
        rhs = D
        N = 2 * A * x + B * y
        Den = B * x + 2 * C * y
        true_yprime = -N / Den
        eq_text = f"{_term(A, 'x^2')} + {_term(B, 'x*y')} + {_term(C, 'y^2')} = {D}"
        sign_flip = N / Den
        forgot_product_rule = -(2 * A * x) / Den  # dropped the "+By" piece of N
        forgot_chain_on_y2 = -(2 * A * x + B * y + 2 * C * y) / (B * x)  # y^2 term shifted to N
        cands_expr = [sign_flip, forgot_product_rule, forgot_chain_on_y2]

    elif tier == "hard":
        A = rng.randint(1, 5)
        B = rng.randint(1, 5)
        C = rng.randint(10, 60)
        lhs = A * x**2 * y + B * y**3
        rhs = C
        N = 2 * A * x * y
        Den = A * x**2 + 3 * B * y**2
        true_yprime = -N / Den
        eq_text = f"{_term(A, 'x^2*y')} + {_term(B, 'y^3')} = {C}"
        sign_flip = N / Den
        # treats x^2 as a constant multiplier when differentiating the
        # x^2*y product -- misses the "2xy" term that x^2's own power rule
        # contributes, so the x^2*y term contributes only x^2*y' to the
        # y'-bearing side and the numerator (which came entirely from that
        # missed term) is identically zero.
        forgot_u_prime = Rational(0)
        # forgets the chain rule on y^3 -- treats d/dx[B*y**3] as 3*B*y**2
        # (no y' factor), so that term is added as a bare term instead of
        # joining the y'-bearing side: 2*A*x*y + A*x**2*y' + 3*B*y**2 = 0.
        forgot_chain_on_y3 = -(2 * A * x * y + 3 * B * y**2) / (A * x**2)
        cands_expr = [sign_flip, forgot_u_prime, forgot_chain_on_y3]

    else:  # exam
        A = rng.randint(1, 4)
        B = rng.randint(1, 4)
        C = rng.randint(1, 4)
        D = rng.randint(10, 60)
        lhs = A * x**3 + B * x * y**2 + C * y**3
        rhs = D
        N = 3 * A * x**2 + B * y**2
        Den = 2 * B * x * y + 3 * C * y**2
        true_yprime = -N / Den
        eq_text = f"{_term(A, 'x^3')} + {_term(B, 'x*y^2')} + {_term(C, 'y^3')} = {D}"
        sign_flip = N / Den
        # forgets the chain rule on y^2 when differentiating the x*y^2
        # product's second factor -- treats it as a constant, dropping the
        # "2*B*x*y*y'" term from the denominator entirely.
        forgot_product_chain = -(3 * A * x**2 + B * y**2) / (3 * C * y**2)
        # forgets the chain rule on y^3 -- shifts that term's contribution
        # to the numerator instead of the denominator.
        forgot_chain_on_y3 = -(3 * A * x**2 + B * y**2 + 3 * C * y**2) / (2 * B * x * y)
        cands_expr = [sign_flip, forgot_product_chain, forgot_chain_on_y3]

    # Independent check: sympy.idiff, a route independent of however the
    # term-by-term construction above assembled `true_yprime`.
    oracle = idiff(lhs - rhs, y, x)
    check_equivalent(oracle, true_yprime, f"idiff disagrees: tier={tier}, eq={eq_text}")

    # Some candidates collide with the true answer only for specific SMALL
    # coefficient draws (e.g. easy's `dropped_B` collides exactly when
    # B = 1) -- an expected structural coincidence, not a construction bug,
    # so a colliding candidate is dropped rather than hard-failing the build
    # (see `_add_distinct`'s docstring, skill 1, above).
    distinct_cands = []
    _add_distinct(distinct_cands, true_yprime, *cands_expr)
    check(len(distinct_cands) >= 1, f"implicit-diff: every candidate collided: tier={tier}, eq={eq_text}")

    answer_text = _typed(true_yprime)
    cands_text = [_typed(c) for c in distinct_cands]
    distractors = pick_distractors(answer_text, cands_text)

    eq_prose = eq_text.replace("*", "·")  # display multiplication, never a bare "*"
    text = phrase(
        rng,
        (
            "The curve is defined implicitly by {eq}. Use implicit differentiation to find dy/dx in terms of x and y.",
            "A curve satisfies the implicit equation {eq}. Find dy/dx in terms of x "
            "and y using implicit differentiation.",
            "Consider the curve given implicitly by {eq}. Use implicit differentiation "
            "to express dy/dx in terms of x and y.",
        ),
        eq=eq_prose,
    )

    return dict(
        text=text,
        latex=text,
        answer=str(true_yprime),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )
