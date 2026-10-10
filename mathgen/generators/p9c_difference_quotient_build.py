"""The difference-quotient co-located pair.

Two skills, one file, one author -- a similarity check flagged them as 0.50-similarity duplicate candidates of EACH
OTHER, and the split is a NOTATION-vs-BEHAVIOUR distinction that has to hold from the
rendered question text, not the title.

**The one-sentence test for each row:**

- ``senior_examine_the_behaviour_of_the_difference_quotient_frac``
  (``senior-examine-the-behaviour-of-the-difference-quotient-frac``,
  ACMMM081, topic ``p9c_difference_quotient_behaviour``) asks about a
  SEQUENCE of shrinking h-values and what number they approach -- the
  learner reads a numeric TREND, never picks a Delta x, never computes a
  single ratio from scratch.
- ``senior_use_the_notation_frac_delta_y_delta_x``
  (``senior-use-the-notation-frac-delta-y-delta-x``, ACMMM079, topic
  ``p9c_delta_y_delta_x_notation``) applies the delta-notation to compute a
  SINGLE quotient at ONE stated delta-x -- never a sequence, never a limit,
  never "as h->0".

**The ACMMM082 boundary (`senior-define-the-derivative-f-left-x-right-as`,
`mathgen/generators/p5e_first_principles_build.py`), already shipped.** That
skill is ALGEBRAIC/SYMBOLIC: it forms and simplifies the difference quotient
as an EXPRESSION IN x AND h (``answer: expression``, no specific point), and
its hard tier takes the algebraic limit to produce f'(x) as an expression in
x alone. Neither skill below EVER asks for an expression in x/h or in x
alone; both evaluate at a stated NUMERIC point and both answer with a plain
number (row 1: an integer; row 2: an exact fraction).
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Rational, diff, expand, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    pick_distractors,
    rational_text,
)

# Module-scope symbols. `x`/`h` for row 1's difference quotient in the
# defining variable and its increment; `dx` for row 2's Leibniz increment
# (kept distinct from `h` even though both play the same algebraic role, so
# the two skills' own working never shares a symbol name by accident).
_x, _h = symbols("x h")
_dx = symbols("dx")


def _prose(expr) -> str:
    """Learner-facing rendering: ``**`` -> ``^``, ``*`` -> the middle dot
    (never a bare asterisk -- `mathgen/text_quality.py`'s `star_operator`
    check would flag a literal ``*``). Mirrors `p5e_first_principles_build.py`'s
    own `_prose` helper; not shared centrally because it is two lines and
    used only by these two topics."""
    return str(expr).replace("**", "^").replace("*", "·")


def _dp3(value: Fraction) -> str:
    """Exact 3-decimal-place text for a `Fraction` known to have a
    denominator dividing 1000 -- every displayed table value in row 1, by
    construction (`A` an integer, `h` at most 3 decimal places). Deliberately
    NOT a float format: `f"{float(v):.3f}"` risks binary round-off exactly at
    the precision boundary this skill is teaching, so the exact digits are
    read off the `Fraction` directly instead."""
    scaled = value * 1000
    check(scaled.denominator == 1, f"table value {value} is not exact to 3 dp")
    n = int(scaled)
    sign = "-" if n < 0 else ""
    n = abs(n)
    whole, frac = divmod(n, 1000)
    return f"{sign}{whole}.{frac:03d}"


# --- Row 1: examine the behaviour of the difference quotient as h -> 0 -----

_POS_H = (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000))
_NEG_H = tuple(-v for v in _POS_H)
# Fixed display labels for the h-values themselves (never reformatted to 3
# dp -- only the computed Q(h) column needs the fixed precision; the h
# column is the natural "0.1, 0.01, 0.001" a learner is shown elsewhere).
_POS_H_LABELS = ("0.1", "0.01", "0.001")
_NEG_H_LABELS = ("-0.1", "-0.01", "-0.001")

_P_POOL = tuple(v for v in range(-6, 7) if v != 0)


@generator(
    "p9c_difference_quotient_behaviour",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9c_difference_quotient_behaviour(rng, tier="easy"):
    """Read a numeric TREND of the difference quotient Q(h) = (f(p+h)-f(p))/h
    at several shrinking h and state the value it approaches as h -> 0.

    f(x) = A*x^2 + B*x + C, evaluated at one stated nonzero point x = p.

    easy:   B = 0. One-sided h = 0.1, 0.01, 0.001. Answer = 2*A*p.
    medium: B nonzero. Same one-sided sequence. Answer = 2*A*p + B.
    hard:   B nonzero (as medium). BOTH a positive and a negative one-sided
            sequence are shown; the learner must recognise the two trends
            converge to the SAME value (two-sided convergence), a
            qualitatively different check from reading a longer table.
            Answer is still 2*A*p + B.

    Never asks for an expression in x/h and never picks a Delta x -- see the
    module docstring's ACMMM082 boundary.
    """
    a = rng.randrange(1, 6)  # [1, 5]
    c = rng.randrange(-9, 10)
    p = rng.choice(_P_POOL)

    if tier == "easy":
        b = 0
    else:
        for _attempt in range(50):
            b_pool = [v for v in range(-9, 10) if v != 0]
            b = rng.choice(b_pool)
            if 2 * a * p + b != 0:  # guards the sign-error distractor collision
                break

    # Construction: derive the closed form by actual algebra, not a hard-coded formula -- expand
    # cancels the common factor of h exactly.
    f = a * _x**2 + b * _x + c
    dq = expand((f.subs(_x, _x + _h) - f) / _h)
    dq_at_p = dq.subs(_x, p)
    answer_expr = dq_at_p.subs(_h, 0)
    check(answer_expr.is_Integer, f"limit is not an integer: a={a}, b={b}, c={c}, p={p}")
    answer = int(answer_expr)

    # Independent check: the general derivative operator, a route that never
    # touches the specific quotient's algebra.
    f_general = a * _x**2 + b * _x + c
    check_equivalent(
        diff(f_general, _x).subs(_x, p), answer, f"independent derivative check failed: a={a}, b={b}, c={c}, p={p}"
    )

    def q_fraction(h_value: Fraction) -> Fraction:
        """Q(h) at the drawn p, as an exact `Fraction` -- `dq_at_p` is linear
        in h so this substitution never leaves the rationals."""
        rational = dq_at_p.subs(_h, Rational(h_value.numerator, h_value.denominator))
        return Fraction(int(rational.p), int(rational.q))

    pos_fractions = [q_fraction(hv) for hv in _POS_H]
    pos_lines = ", ".join(f"Q({label}) = {_dp3(qv)}" for label, qv in zip(_POS_H_LABELS, pos_fractions))

    f_prose = _prose(expand(f))
    if tier != "hard":
        question_text = (
            f"f(x) = {f_prose}. At x = {p}, the difference quotient "
            f"Q(h) = (f(x+h) - f(x)) / h is computed for several small "
            f"positive values of h: {pos_lines}. As h approaches 0, what "
            f"value does the difference quotient approach?"
        )
    else:
        neg_fractions = [q_fraction(hv) for hv in _NEG_H]
        neg_lines = ", ".join(f"Q({label}) = {_dp3(qv)}" for label, qv in zip(_NEG_H_LABELS, neg_fractions))
        question_text = (
            f"f(x) = {f_prose}. At x = {p}, the difference quotient "
            f"Q(h) = (f(x+h) - f(x)) / h is computed for several small "
            f"values of h approaching 0 from both directions: positive "
            f"h gives {pos_lines}; negative h gives {neg_lines}. What "
            f"single value do both sequences approach as h approaches 0?"
        )

    answer_text = str(answer)

    # Misconceptions. The key is a whole number, so every option is too: the
    # old "reads off the least-converged table value" distractor was a
    # 3-decimal-place number and singled out the key by its shape.
    sign_error = -answer
    function_value = a * p**2 + b * p + c  # reads f(p) instead of the slope
    forgot_p = 2 * a + b  # differentiates but drops the factor of p
    check_distinct(answer, sign_error, msg=f"sign_error equals answer: a={a}, b={b}, p={p}")
    candidates_text = [str(sign_error)]
    if tier == "easy":
        forgot_power_multiply = a * p
        check_distinct(answer, forgot_power_multiply, msg=f"forgot_power_multiply equals answer: a={a}, p={p}")
        candidates_text.append(str(forgot_power_multiply))
    else:
        dropped_linear_term = 2 * a * p
        check_distinct(answer, dropped_linear_term, msg=f"dropped_linear_term equals answer: a={a}, b={b}, p={p}")
        candidates_text.append(str(dropped_linear_term))
        candidates_text.append(str(2 * a * p - b))  # sign slip on the linear term
    candidates_text.extend([str(function_value), str(forgot_p), str(answer + 1), str(answer - 1)])

    distractors = pick_distractors(answer_text, candidates_text)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer),
        "distractors": distractors,
    }


# --- Row 2: use the notation delta-y/delta-x for one stated quotient -------

_DX_POSITIVE = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 4))
_DX_NEGATIVE = tuple(-v for v in _DX_POSITIVE)


def _linear_f(m, k):
    return lambda v: m * v + k


def _quadratic_f(a, b, c):
    return lambda v: a * v**2 + b * v + c


@generator(
    "p9c_delta_y_delta_x_notation",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9c_delta_y_delta_x_notation(rng, tier="easy"):
    """Use delta-y/delta-x for the SINGLE, fully stated quotient
    (f(p+dx)-f(p))/dx at one stated point p and one stated dx -- never a
    sequence, never a limit, never "as h -> 0".

    easy:   f(x) = M*x + K, M nonzero. dx from a POSITIVE pool excluding 1.
            delta_y/delta_x = M exactly, independent of p and dx.
    medium: f(x) = A*x^2 + B*x + C, B nonzero. Same positive dx pool.
            delta_y/delta_x = A*(2*p + dx) + B -- now genuinely depends on
            p and dx.
    hard:   same quadratic formula, dx drawn from the NEGATIVE pool -- the
            escalation is sign management, not magnitude.

    Never shows a sequence of h and never asks "as h -> 0" -- see the module
    docstring's ACMMM082 boundary.
    """
    p = rng.choice(_P_POOL)

    if tier == "easy":
        dx_pool = _DX_POSITIVE
        for _attempt in range(50):
            m_pool = [v for v in range(-9, 10) if v != 0]
            m = rng.choice(m_pool)
            if m not in (1, -1):  # guards the reciprocal-error collision
                break
        k = rng.randrange(-9, 10)
        dx = rng.choice(dx_pool)
        f = _linear_f(m, k)
        f_sym = m * _x + k
    else:
        dx_pool = _DX_POSITIVE if tier == "medium" else _DX_NEGATIVE
        dx = rng.choice(dx_pool)
        a = rng.randrange(1, 6)
        c = rng.randrange(-9, 10)
        for _attempt in range(50):
            b_pool = [v for v in range(-9, 10) if v != 0]
            b = rng.choice(b_pool)
            answer_probe = a * (2 * p + dx) + b
            if answer_probe not in (0, 1, -1):  # guards sign-error/reciprocal-error
                break
        f = _quadratic_f(a, b, c)
        f_sym = a * _x**2 + b * _x + c

    # Construction: plain Fraction arithmetic on the specific numbers, no
    # symbols at all.
    p_val = Fraction(p)
    f_p = f(p_val)
    f_p_dx = f(p_val + dx)
    delta_y = f_p_dx - f_p
    answer = delta_y / dx
    check(isinstance(answer, Fraction) or isinstance(answer, int), f"answer is not an exact fraction: tier={tier}")
    answer = Fraction(answer)

    # Independent check: general symbolic formula in x and dx, cancelled by
    # `expand` exactly as row 1's closed form is, THEN substituted with the
    # specific p/dx and compared -- a genuinely different route from the raw
    # arithmetic above.
    general = expand((f_sym.subs(_x, _x + _dx) - f_sym) / _dx)
    general_at_point = general.subs({_x: p, _dx: Rational(dx.numerator, dx.denominator)})
    check_equivalent(general_at_point, answer, f"independent symbolic check failed: tier={tier}, p={p}, dx={dx}")

    answer_payload = {"p": answer.numerator, "q": answer.denominator}
    answer_text = rational_text(answer)

    f_prose = _prose(expand(f_sym))
    dx_text = rational_text(dx)
    question_text = f"y = f(x) = {f_prose}. At x = {p}, x changes by δx = {dx_text}. Find δy/δx for this change."

    # Misconceptions.
    forgot_to_divide = delta_y
    check_distinct(answer, forgot_to_divide, msg=f"forgot_to_divide equals answer: tier={tier}, p={p}, dx={dx}")
    reciprocal_error = Fraction(1, 1) / answer
    check_distinct(answer, reciprocal_error, msg=f"reciprocal_error equals answer: tier={tier}, p={p}, dx={dx}")
    sign_error = -answer
    check_distinct(answer, sign_error, msg=f"sign_error equals answer: tier={tier}, p={p}, dx={dx}")

    candidates_text = [
        rational_text(Fraction(forgot_to_divide)),
        rational_text(reciprocal_error),
        rational_text(sign_error),
    ]
    if tier == "hard":
        ignored_delta_x_sign = a * (2 * p + abs(dx)) + b
        check_distinct(answer, ignored_delta_x_sign, msg=f"ignored_delta_x_sign equals answer: p={p}, dx={dx}")
        candidates_text.append(rational_text(Fraction(ignored_delta_x_sign)))

    distractors = pick_distractors(answer_text, candidates_text)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer_payload,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer),
        "distractors": distractors,
    }
