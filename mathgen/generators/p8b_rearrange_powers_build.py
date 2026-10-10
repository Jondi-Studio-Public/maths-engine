"""Deliberately its own module: it never imports from, and is never imported by, that module, so nothing pinned
there can move.

Do NOT set `answer_form` on the Question this generator returns -- a
rearranged formula has no single required shape.

**Grading-gap history:** three cases were built and then removed -- `E = mc²` for `c`, `P = I²R` for `I`, and
`y = a(x-h)²+k` for `x` -- because their canonical answer was a quotient of two square roots with a *free
symbol* in the denominator (`sqrt(P)/sqrt(R)`, `sqrt(E)/sqrt(m)`, `h + sqrt(y-k)/sqrt(a)`).
`mathgen/grading/typed.py`'s `symbolically_equal` has no positivity assumptions when it sympifies a learner's
typed text, so it cannot prove `sqrt(P/R)` (the form a learner naturally writes) equal to `sqrt(P)/sqrt(R)` --
they genuinely differ once the free symbol is allowed to be negative. Confirmed empirically
(`grade_expression` rejects the natural combined form for all three; `regrade_sweep.py` independently caught
it). This does NOT affect a case whose only sqrt-denominator symbol is `pi` (a provably-positive literal, not
a free symbol) -- `circle_r`'s `sqrt(A)/sqrt(pi)` and `sphere_r`'s cube-root-over-`pi` form both correctly
accept their combined alternative (`sqrt(A/pi)`, `(6*V/pi)**(1/3)/2`), confirmed the same way. See
`test_alternative_radical_form_is_accepted_for_every_case` in this module's test file, which pins the property
generically (not the three known-bad formulas by name) so a future case with this shape fails automatically.
"""

import re

from sympy import Rational, symbols, solve, Eq, pi, sqrt, latex

from ..core import generator, check, check_equivalent, check_distinct, pick_distractors


def _display(expr) -> str:
    """Learner-facing rendering of a sympy expression: ``T^2*g/(4*pi^2)``
    becomes ``T^2g/(4π^2)``.

    ``Question.answer`` keeps ``str(expr)`` (Python/sympy notation) because
    the expression grader sympifies it; ``answer_text``, ``working`` and
    distractors use this instead (same convention as
    ``p3c_indices_build.py``'s own ``_display``), because build topics are
    held to the learner-facing text bar in ``mathgen/text_quality.py`` that
    flags raw ``**`` and a bare ``*`` straight after a digit
    -- the legacy ``rearrange`` topic in
    ``foundations_algebra.py`` predates that bar and is exempt (its module
    name doesn't end in ``_build``), but this module is a `_build` module and
    is not. ``**`` becomes ``^`` (the grader already accepts ``^`` from a
    learner); a ``*`` immediately after a digit (``4*pi``, coefficient times
    a following symbol/paren) is dropped for implicit multiplication, which
    the grader's ``implicit_multiplication_application`` transform also
    accepts on the way back in. A ``*`` between two letters (``T*g``) is left
    alone -- collapsing it too would merge adjacent one-letter symbol names
    into a single unparseable token (``pi`` immediately followed by ``l``
    becomes the identifier ``pil``, not ``pi`` times ``l``) -- and it is not
    what the bad-text patterns flag: ``star_operator``'s regex only matches a
    bare ``*`` or a digit-then-``*`` run, never letter-then-``*``.

    Reverted: `answer_text` is exactly what `scripts/regrade_sweep.py` and the real grading pipeline grade
    when a learner's answer matches the canonical text verbatim, and
    `mathgen/grading/typed.py:_parse_expression` cannot round-trip unicode `√` (U+221A is not a Python
    identifier character, so it silently mis-parses rather than raising) or unicode `π` (U+03C0 IS a valid
    identifier, so it parses as a free `Symbol('π')`, not `sympy.pi` -- not algebraically equal to the
    constant).
    """
    s = str(expr).replace("**", "^")
    s = re.sub(r"(?<=\d)\*", "", s)
    return s


# Symbols distinct from foundations_algebra's own `_l`/`_A`/... (this is a
# new module; it defines its own), one small symbol table per (formula)
# group below. Assumptions are chosen per quantity's actual physical range,
# not uniformly `positive=True` any more -- each declaration below says why.
_A, _r = symbols("A r", positive=True)  # circle area, radius: both positive
_T, _l, _g = symbols("T l g", positive=True)  # pendulum period/length/g: all positive

# The pendulum-period formula's un-squared T/(2*pi) ratio -- shared by the
# correct l/g derivations below and by the "squared only one side"
# misconception, which is literally this ratio left unsquared.
_RATIO = _T / (2 * pi)

_V = symbols("V", positive=True)

# Even root to undo, explicit root (Pythagoras c=sqrt(a**2+b**2) for a): all
# three are lengths, so `positive=True`. Distinct python names (`_py*`) so a
# future addition reusing plain `a`/`c` for an unrelated quantity can't be
# misread as this module's symbol table.
_pyA, _pyB, _pyC = symbols("a b c", positive=True)

# Genuine multi-step transposition, no power/root sign issue (SUVAT
# s=ut+(1/2)*a*t**2 for a, and for u): displacement/velocity/acceleration
# are all signed in general (a SUVAT problem regularly has a negative
# acceleration, a reversing velocity, or a negative displacement from the
# start point) so `real=True`; time is the one quantity here that is always
# positive.
_s, _u, _a3 = symbols("s u a", real=True)
_t = symbols("t", positive=True)


# (formula, target) cases, chosen uniformly per call. All are hard-tier by
# construction: isolating the target needs a square root or a squaring/cubing
# step, never a single division/multiplication (mirrors the existing
# `_REARRANGE_HARD` single-formula hard tier, just with many formulas
# instead of one). `note`, when non-empty, states in the question text
# itself the physical reason only one root is wanted -- never left implicit.
# `key` disambiguates cases that share a `target_name` (several formulas
# below reuse the letters `r` and `a` for physically different quantities,
# same as the pre-existing `l`/`g` reuse of one formula for two targets) --
# misconception construction below switches on `key`, never on
# `target_name` alone.
_CASES = [
    dict(
        key="circle_r",
        target_name="r",
        target=_r,
        text="A = πr²",
        tex=r"A = \pi r^2",
        lhs=_A,
        rhs=pi * _r**2,
        note="r is the radius of a circle, so r > 0",
    ),
    dict(
        key="pendulum_l",
        target_name="l",
        target=_l,
        text="T = 2π√(l/g)",
        tex=r"T = 2\pi\sqrt{l/g}",
        lhs=_T,
        rhs=2 * pi * sqrt(_l / _g),
        note="",
    ),
    dict(
        key="pendulum_g",
        target_name="g",
        target=_g,
        text="T = 2π√(l/g)",
        tex=r"T = 2\pi\sqrt{l/g}",
        lhs=_T,
        rhs=2 * pi * sqrt(_l / _g),
        note="",
    ),
    dict(
        key="sphere_r",
        target_name="r",
        target=_r,
        text="V = (4/3)πr³",
        tex=r"V = \frac{4}{3}\pi r^3",
        lhs=_V,
        rhs=Rational(4, 3) * pi * _r**3,
        # Odd power: a real cube root is unique for any sign of its
        # argument, so taking it never introduces a second candidate root
        # the way an even power does -- no positive-root note is needed
        # (confirmed empirically: solve() below returns exactly one root
        # under `r > 0` with no extra filtering required).
        note="",
    ),
    dict(
        key="pythagoras_a",
        target_name="a",
        target=_pyA,
        text="c = √(a² + b²)",
        tex=r"c = \sqrt{a^2 + b^2}",
        lhs=_pyC,
        rhs=sqrt(_pyA**2 + _pyB**2),
        note="a is a length, so a > 0",
    ),
    dict(
        key="suvat_a",
        target_name="a",
        target=_a3,
        text="s = ut + ½at²",
        tex=r"s = ut + \frac{1}{2}at^2",
        lhs=_s,
        rhs=_u * _t + Rational(1, 2) * _a3 * _t**2,
        note="",
    ),
    dict(
        key="suvat_u",
        target_name="u",
        target=_u,
        text="s = ut + ½at²",
        tex=r"s = ut + \frac{1}{2}at^2",
        lhs=_s,
        rhs=_u * _t + Rational(1, 2) * _a3 * _t**2,
        note="",
    ),
]


def _misconceptions(key, ans_expr):
    """Two symbolic misconception candidates per case, checked
    `check_distinct` against the abstract symbols (not a per-seed numeric
    substitution -- there is nothing to draw here, the formula itself is the
    fixed content and rng only picks among the cases), so `check_distinct`
    proves each candidate differs from the answer for EVERY value the
    symbols could take, not just the values a seed happens to sample."""
    if key == "circle_r":
        return [
            -ans_expr,  # kept both signs implicitly: offered the negative root
            _A / pi,  # isolated r**2 and never took the root at all
        ]
    if key == "pendulum_l":
        return [
            _g * _RATIO,  # squared only one side (ratio left unsquared)
            _RATIO**2 / _g,  # divided by g instead of multiplying by it
        ]
    if key == "pendulum_g":
        return [
            _l / _RATIO,  # squared only one side (ratio left unsquared)
            _l * _RATIO**2,  # multiplied by l instead of dividing by it
        ]
    if key == "sphere_r":
        return [
            Rational(3) * _V / (4 * pi),  # isolated r**3 and never took the root at all
            _V ** Rational(1, 3),  # forgot the 4/(3*pi) coefficient entirely
        ]
    if key == "pythagoras_a":
        return [
            sqrt(_pyC**2 + _pyB**2),  # kept the addition -- never moved b**2 across
            _pyC - _pyB,  # thought the root distributes over the sum
        ]
    if key == "suvat_a":
        return [
            2 * (_s - _u) / _t**2,  # dropped the t multiplying u
            (_s - _u * _t) / _t**2,  # forgot the factor of 2
        ]
    if key == "suvat_u":
        return [
            _s / _t - _a3 * _t,  # forgot the factor of 1/2
            _s - _a3 * _t / 2,  # forgot to divide s by t
        ]
    raise AssertionError(f"no misconceptions defined for case key {key!r}")


@generator("rearrange_powers_roots", level=7)
def rearrange_powers_roots(rng):
    case = rng.choice(_CASES)
    target_name, target = case["target_name"], case["target"]
    lhs, rhs = case["lhs"], case["rhs"]

    solutions = solve(Eq(lhs, rhs), target)
    # No remaining case needs branch selection (the one that did, `vertex_x`,
    # was removed -- see module docstring's "grading-gap" section), so every
    # case is held to exactly one root.
    check(len(solutions) == 1, f"expected exactly one root for {case['key']}, got {solutions}")
    ans_expr = solutions[0]
    # Independent check: substitute the derived expression back into the
    # ORIGINAL unsolved equation -- a different computational route from
    # solve()'s own algorithm, same style as the existing `rearrange` topic.
    check_equivalent(lhs.subs(target, ans_expr), rhs.subs(target, ans_expr))
    ans_str = str(ans_expr)

    cand_exprs = _misconceptions(case["key"], ans_expr)
    check_distinct(ans_expr, *cand_exprs)
    ans_display = _display(ans_expr)
    cands = [_display(c) for c in cand_exprs]

    note_suffix = f" ({case['note']})" if case["note"] else ""
    text = f"Make {target_name} the subject of the formula {case['text']}{note_suffix}"
    return dict(
        text=text,
        latex=case["tex"],
        answer=ans_str,
        answer_text=ans_display,
        answer_latex=latex(ans_expr),
        working=[f"Start from {case['text']}", f"{target_name} = {ans_display}"],
        distractors=pick_distractors(ans_display, cands),
    )
