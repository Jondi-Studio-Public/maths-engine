"""Generates `p13c_antideriv_notation` questions (ACMMM115): use antiderivative notation and give the result as
an expression.

Kernel: the notation ``\\int f(x) dx`` read BACKWARDS. Every premise states a
DIFFERENTIATION fact -- ``d/dx(F(x)) = f(x)`` for a concrete ``F(x)`` the
learner is TOLD -- and the question asks the learner to translate that fact
into integral notation: write down what ``\\int f(x) dx`` equals, i.e. state
``F(x) + C``. Nothing is computed from scratch; ``F`` is always already in
the premise.

**The high-risk boundary this design exists to avoid:** two shipping
skills already own "compute an antiderivative from a bare integrand" --
``skills/integration/integrate-polynomial.yaml`` ("Find ∫(-6x^2 - 14x) dx" -> derive the antiderivative by
applying the reverse power rule) and ``skills/integration/integrate-trig.yaml`` (same shape, trig integrands).
This generator NEVER hands the learner a bare ``f(x)`` and asks for ``∫f(x)dx`` computed from nothing --
``F(x)`` is always given, so the operation being assessed is reading the notation, not deriving an
antiderivative. Verified by direct render comparison against both sibling skills: every question here states a
stated F and its stated derivative fact BEFORE asking for the integral notation, which neither sibling skill's
prompt does.

**The ``solve-ode-separable-poly`` boundary**
(``skills/differential-equations/solve-ode-separable-poly.yaml``): that skill
states ``dy/dx = <polynomial>`` and asks the learner to solve the ODE (an
integration-from-a-bare-derivative problem, matching the same
integrate-from-nothing shape as the two integration skills above, just
phrased as an ODE). It never states a paired ``F(x)`` fact in its premise
either. This generator's premise always states BOTH members of a
differentiation pair (``F`` and its derivative), which `solve-ode-*` never
does.

**The ``+C`` grading finding, proven not assumed** (``mathgen/grading/typed.py::
_expected_expression``, confirmed by a direct ``grade_expression`` call in
this module's own test file): ``question.answer`` holds the antiderivative
WITHOUT ``+ C`` (e.g. ``"x**4"``); ``question.answer_text`` holds the same
expression WITH ``" + C"`` appended (e.g. ``"x**4 + C"``) -- the exact
convention ``integrate-polynomial``/``integrate-trig`` already use.
``_expected_expression`` treats the DISPLAYED ``answer_text`` as authoritative
whenever it differs from the internal ``answer`` by exactly one free symbol
(the constant of integration), so a learner who drops ``+ C`` is compared
against the full ``F(x) + C`` and correctly REJECTED, while a learner who
types the full ``F(x) + C`` is ACCEPTED. Every constant-of-integration
distractor in this module is spelled with the same letter ``C`` the
canonical uses -- a different letter (``D``) would be a different free
symbol to the grader and would not reliably fire the "dropped +C"
misconception it is meant to test.

**Difficulty ladder** (every tier ``answer: expression``, one grader kind):
- easy: ``F(x) = a*x^n`` (single power term). The differentiation fact is
  stated directly; write ``∫f(x) dx``.
- medium: ``F(x)`` a 2-3 term polynomial sum. Same shape, more terms to
  carry into the ``+ C`` answer -- a bigger computation, not a new decision.
- hard: ``F(x)`` is exponential (``a*e^(kx)``), sine (``a*sin(kx)``) or
  cosine (``a*cos(kx)``) -- a function the learner CANNOT reach via the
  polynomial power rule at all. The differentiation fact is stated in full,
  so the ONLY route to the answer is reading what the notation asserts, not
  "applying the power rule harder."

  **History: why the hard tier's third family is ``cos``, not
  ``ln``.** The hard tier originally drew ``ln`` (``F = a*ln(x)``,
  ``f = a/x``). ACARA senior students are taught the domain-safe answer form
  ``a*ln|x| + C`` for this family, but ``mathgen/grading/typed.py::_parse_expression``
  cannot parse the ``|x|`` bar-absolute-value notation at all: ``|`` is not
  valid Python syntax, so the primary sympy parse throws and the code falls
  through to ``spoken_to_expr`` (a voice-transcript parser), which silently
  discards the ``ln`` and returns the bare symbol ``x`` -- confirmed
  directly, ``grade_expression`` on a genuinely correct ``-2*ln|x| + C``
  answer returned ``correct=False``. This is a real parser gap in
  ``mathgen/grading/typed.py``, not a narrow symbolic-equivalence miss, and there is
  no existing bar-notation handling anywhere in that module to lean on.
  cross-cutting); the fix here is by construction instead: the ``ln`` family
  (and its ``_draw_ln``/``_LN_FACT_TEMPLATES`` machinery) has been removed
  from the hard tier and replaced with ``cos`` (``_draw_cos``), a third
  family that is equally unreachable via the polynomial power rule but
  needs no absolute-value qualifier and no domain caveat. Do not
  reintroduce ``ln`` here without first fixing the ``|...|`` parse gap in
  ``mathgen/grading/typed.py``.
- exam: linearity from TWO stated facts -- one polynomial ``F``, one
  exponential-or-sine ``G`` -- combined into a single stated derivative
  expression ``p*f(x) + q*g(x)``; the learner must recognise each additive
  piece traces back to one of the two given facts and recover
  ``p*F(x) + q*G(x) + C``. A genuinely new decision (combining two notation
  facts linearly), not present at any lower tier and not merely a bigger
number range.
  confirms this as a genuine escalation.

**Misconceptions** (checked distinct via a real sympy ``Symbol('C')``, every
seed, unconditionally):
- drops ``+ C`` entirely (grader-enforced, see the finding above);
- answers ``f(x)`` itself instead of ``F(x) + C`` (confuses which side of
  the notation is being asked for -- the single most on-contract
  misconception this skill exists to catch);
- (hard/exam only) answers ``F'(x) + C`` where ``F'(x) != f(x)`` as stated --
  a differentiation slip carried into the notation, built per-family so it
  is genuinely a DIFFERENT value from the stated ``f(x)``;
- (exam only) drops one of the two linear terms, or drops both coefficients.
"""

from __future__ import annotations

import re

from sympy import cos, diff, exp, sin, symbols

from ..core import (
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
    symbolically_equal,
)

x, C = symbols("x C")


def _add_distinct(pool: list, answer_symbolic, *candidates) -> None:
    """Append each ``(expr, has_c)`` candidate only if its symbolic form
    (``expr + C`` when ``has_c``, else ``expr``) is genuinely a different
    VALUE from `answer_symbolic` and from every candidate already kept --
    some collisions are expected structural coincidences (e.g. a family
    drawn with a coefficient of 1 could make a "dropped coefficient"
    candidate equal a lower-priority one), not construction bugs, matching
    ``p13c_instantaneous_rate_context_build.py``'s own ``_add_distinct``.
    """
    kept_symbolic = [answer_symbolic]
    for expr, has_c in candidates:
        if expr is None:
            continue
        symbolic = expr + C if has_c else expr
        if any(symbolically_equal(symbolic, k) for k in kept_symbolic):
            continue
        kept_symbolic.append(symbolic)
        pool.append((expr, has_c))


def _log_to_ln(text: str) -> str:
    """Sympy's ``ln`` is a pure display alias of ``log`` -- there is no separate ``ln`` AST node, so
    ``str(sympy.ln(x))`` is literally ``"log(x)"``. A plain string replace is grader-safe EVERYWHERE this
    module builds a string (prompt, ``answer_text``, every distractor):
    `mathgen/grading/typed.py:grade_expression` already accepts `ln(x)`, `log(x)` and bare-juxtaposition
    spellings as equivalent, so swapping the DISPLAYED spelling from `log(` to `ln(` changes nothing about
    what the grader compares. Senior (ACARA) audiences read `log(x)` as base-10, so every learner-facing
    string in this module should read `ln(x)`, never `log(x)`.
    """
    return text.replace("log(", "ln(")


def _exp_to_e_caret(text: str) -> str:
    """PROMPT-ONLY display transform: ``exp(<inner>)`` -> ``e^(<inner>)``.

    NEVER apply this to a string the grader parses (`answer_text`, a
    distractor). Confirmed directly:
    ``_parse_expression('e^(3*x)')`` returns ``Symbol('e') ** (3*x)``, NOT
    ``sympy.exp(3*x)`` -- there is no special-casing of the literal
    character ``e`` as Euler's number anywhere in that parser. Introducing
    the string `"e^"` into a graded field adds a spurious free symbol `e`
    in addition to `C`, which breaks `_expected_expression`'s "displayed
    answer_text is authoritative when it differs from the internal answer
    by exactly one free symbol" branch and would turn the canonical answer
    itself into a false reject.
    """
    return re.sub(r"exp\(([^()]*(?:\([^()]*\)[^()]*)*)\)", r"e^(\1)", text)


def _disp(expr) -> str:
    """Human-readable display for the PROMPT text only: Python's "**" power
    operator becomes "^", and "*" becomes "·" (a real multiplication dot,
    never a literal asterisk) -- `mathgen.text_quality`'s `star_operator`/
    `double_star` checks forbid both anywhere in `Question.text`, matching
    the convention `p12c_trig_apply_implicit_diff_build.py`'s own hand-built
    prompt strings already use. Also applies the `log(` -> `ln(` senior
    notation fix and, PROMPT-ONLY, `exp(...)` -> `e^(...)` (see
    `_log_to_ln`/`_exp_to_e_caret` -- neither is safe on a graded string).
    """
    text = str(expr).replace("**", "^").replace("*", "·")
    text = _log_to_ln(text)
    text = _exp_to_e_caret(text)
    return text


def _value(expr) -> str:
    """Canonical VALUE string for `answer_text`/a distractor: only "**" is
    converted to "^" here -- `double_star` is forbidden in every
    learner-facing field, but a bare "*" is fine outside the prompt itself
    (matches `differential_equation`'s own `_ode_exponential` convention,
    `str(ans).replace('**', '^')`). Also applies the `log(` -> `ln(` senior
    notation fix (grader-safe, see `_log_to_ln`) -- but NEVER the
    `exp(...)` -> `e^(...)` prompt-only transform, which would introduce a
    spurious free symbol into a graded field (see `_exp_to_e_caret`)."""
    return _log_to_ln(str(expr).replace("**", "^"))


def _display(expr, has_c: bool) -> str:
    return f"{_value(expr)} + C" if has_c else _value(expr)


def _rand_nonzero(rng, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


# --- Polynomial family: F = sum of power terms, f built independently via
# the explicit power-rule formula (never via `diff`), so `check_equivalent`
# below is a genuinely independent route. ---------------------------------


def _rand_poly_terms(rng, count, power_lo, power_hi, coef_lo, coef_hi):
    powers = rng.sample(range(power_lo, power_hi + 1), k=count)
    return [(_rand_nonzero(rng, coef_lo, coef_hi), p) for p in sorted(powers, reverse=True)]


def _poly_from_terms(terms):
    return sum(c * x**p for c, p in terms)


def _poly_derivative_from_terms(terms):
    return sum(c * p * x ** (p - 1) for c, p in terms)


_FACT_TEMPLATES = (
    "Given that d/dx({Fstr}) = {fstr}, write down ∫({fstr}) dx.",
    "The derivative of F(x) = {Fstr} is {fstr}. Find ∫({fstr}) dx.",
    "It is known that d/dx[{Fstr}] = {fstr}. Write ∫ {fstr} dx.",
)

_EXAM_FACT_TEMPLATES = (
    "Given that d/dx({F1str}) = {f1str} and d/dx({Gstr}) = {gstr}, find ∫({combined}) dx.",
    "It is known that the derivative of F(x) = {F1str} is {f1str}, and the "
    "derivative of G(x) = {Gstr} is {gstr}. Find ∫({combined}) dx.",
)


def _easy(rng):
    a = _rand_nonzero(rng, -9, 9)
    n = rng.randint(2, 5)
    terms = [(a, n)]
    F = _poly_from_terms(terms)
    f = _poly_derivative_from_terms(terms)
    check_equivalent(diff(F, x), f, f"easy: derivative mismatch a={a} n={n}")

    answer_symbolic = F + C
    cands: list = []
    _add_distinct(
        cands,
        answer_symbolic,
        (F, False),  # dropped +C
        (f, True),  # answered f(x) itself instead of F(x)+C
    )
    check_distinct(answer_symbolic, *(e + C if hc else e for e, hc in cands), msg="easy: distractor equals answer")

    Fstr, fstr = _disp(F), _disp(f)
    text = phrase(rng, _FACT_TEMPLATES, Fstr=Fstr, fstr=fstr)
    working = [
        f"The fact d/dx({Fstr}) = {fstr} says {Fstr} is an antiderivative of {fstr}.",
        f"So ∫({fstr}) dx = {Fstr} + C.",
    ]
    return _finish(F, cands, text, working)


def _medium(rng):
    count = rng.choice([2, 3])
    terms = _rand_poly_terms(rng, count, power_lo=2, power_hi=6, coef_lo=-9, coef_hi=9)
    F = _poly_from_terms(terms)
    f = _poly_derivative_from_terms(terms)
    check_equivalent(diff(F, x), f, f"medium: derivative mismatch terms={terms}")

    answer_symbolic = F + C
    cands: list = []
    _add_distinct(
        cands,
        answer_symbolic,
        (F, False),
        (f, True),
    )
    check_distinct(answer_symbolic, *(e + C if hc else e for e, hc in cands), msg="medium: distractor equals answer")

    Fstr, fstr = _disp(F), _disp(f)
    text = phrase(rng, _FACT_TEMPLATES, Fstr=Fstr, fstr=fstr)
    working = [
        f"The fact d/dx({Fstr}) = {fstr} says {Fstr} is an antiderivative of {fstr}.",
        f"So ∫({fstr}) dx = {Fstr} + C.",
    ]
    return _finish(F, cands, text, working)


def _draw_exp(rng):
    a = _rand_nonzero(rng, -5, 5)
    k = rng.randint(2, 5)
    F = a * exp(k * x)
    f = a * k * exp(k * x)
    # Sign-error differentiation slip. NOTE: "forgot the chain-rule factor k"
    # (a*exp(kx), i.e. differentiating e^(kx) as if it reproduced itself with
    # no inner-derivative multiplier) was tried first and rejected -- for
    # this family that value is ALGEBRAICALLY IDENTICAL to F itself (an
    # exponential's structural self-reproduction under differentiation), so
    # it collapses to the correct answer rather than a genuinely wrong one.
    # A sign error is always distinct from both f (k != -1, never true here)
    # and F (k != 1, guaranteed by the k in [2, 5] range).
    wrong_deriv = -f
    return F, f, wrong_deriv


def _draw_sin(rng):
    a = _rand_nonzero(rng, -5, 5)
    k = rng.randint(2, 5)
    F = a * sin(k * x)
    f = a * k * cos(k * x)
    wrong_deriv = -f  # sign-error differentiation slip; see _draw_exp
    return F, f, wrong_deriv


def _draw_cos(rng):
    a = _rand_nonzero(rng, -5, 5)
    k = rng.randint(2, 5)
    F = a * cos(k * x)
    f = -a * k * sin(k * x)
    wrong_deriv = -f  # sign-error differentiation slip; see _draw_exp
    return F, f, wrong_deriv


def _hard(rng):
    family = rng.choice(["exp", "sin", "cos"])
    if family == "exp":
        F, f, wrong_deriv = _draw_exp(rng)
    elif family == "sin":
        F, f, wrong_deriv = _draw_sin(rng)
    else:
        F, f, wrong_deriv = _draw_cos(rng)
    templates = _FACT_TEMPLATES
    check_equivalent(diff(F, x), f, f"hard[{family}]: derivative mismatch F={F}")

    answer_symbolic = F + C
    cands: list = []
    _add_distinct(
        cands,
        answer_symbolic,
        (F, False),  # dropped +C
        (f, True),  # answered f(x) itself instead of F(x)+C
        (wrong_deriv, True),  # F'(x)+C where F'(x) != stated f(x)
    )
    check_distinct(
        answer_symbolic, *(e + C if hc else e for e, hc in cands), msg=f"hard[{family}]: distractor equals answer"
    )

    Fstr, fstr = _disp(F), _disp(f)
    text = phrase(rng, templates, Fstr=Fstr, fstr=fstr)
    working = [
        f"The fact d/dx({Fstr}) = {fstr} says {Fstr} is an antiderivative of {fstr}.",
        f"So ∫({fstr}) dx = {Fstr} + C.",
    ]
    return _finish(F, cands, text, working)


def _exam(rng):
    a1 = _rand_nonzero(rng, -9, 9)
    n1 = rng.randint(2, 4)
    F1 = a1 * x**n1
    f1 = a1 * n1 * x ** (n1 - 1)
    check_equivalent(diff(F1, x), f1, f"exam: poly derivative mismatch a1={a1} n1={n1}")

    g_family = rng.choice(["exp", "sin"])
    if g_family == "exp":
        G, g, _ = _draw_exp(rng)
    else:
        G, g, _ = _draw_sin(rng)
    check_equivalent(diff(G, x), g, f"exam[{g_family}]: derivative mismatch G={G}")

    p = rng.randint(2, 5)
    q = rng.randint(2, 5)

    combined_integrand = p * f1 + q * g
    full_F = p * F1 + q * G
    check_equivalent(diff(full_F, x), combined_integrand, f"exam: linearity mismatch p={p} q={q}")

    # The two "drops one linear term or their coefficient" candidates below
    # are offered in a RANDOMISED order (never a fixed priority) so both are
    # genuinely exercisable across seeds -- `pick_distractors`'s default
    # n=3 keeps only the first 3 candidates it sees, so a fixed order would
    # let one permanently dominate the other and never reach a learner.
    drop_term_or_coeff = [(p * F1, True), (F1 + G, True)]
    rng.shuffle(drop_term_or_coeff)

    answer_symbolic = full_F + C
    cands: list = []
    _add_distinct(
        cands,
        answer_symbolic,
        (full_F, False),  # dropped +C
        (combined_integrand, True),  # answered the combined f(x) itself
        *drop_term_or_coeff,  # dropped the second term entirely, or both coefficients
    )
    check_distinct(answer_symbolic, *(e + C if hc else e for e, hc in cands), msg="exam: distractor equals answer")

    F1str, f1str, Gstr, gstr = _disp(F1), _disp(f1), _disp(G), _disp(g)
    combined_str = _disp(combined_integrand)
    text = phrase(
        rng,
        _EXAM_FACT_TEMPLATES,
        F1str=F1str,
        f1str=f1str,
        Gstr=Gstr,
        gstr=gstr,
        combined=combined_str,
    )
    working = [
        f"Split the integrand: {combined_str} = {p}({f1str}) + {q}({gstr}).",
        f"By linearity, ∫({combined_str}) dx = {p}∫({f1str}) dx + {q}∫({gstr}) dx = {p}({F1str}) + {q}({Gstr}) + C.",
        f"So ∫({combined_str}) dx = {_disp(full_F)} + C.",
    ]
    return _finish(full_F, cands, text, working)


def _finish(F, cands, text, working=None):
    answer_text = f"{_value(F)} + C"
    distractor_strs = pick_distractors(answer_text, [_display(e, hc) for e, hc in cands])
    # `sympy.latex(exp(2*x))` already renders "e^{2 x}" (no fix needed), but
    # `sympy.latex(log(x))`/`sympy.latex(ln(x))` both render
    # "\log{\left(x \right)}" -- senior (ACARA) audiences read \log as
    # base-10, so fix the LaTeX key the same way as every plain-text field.
    answer_latex = latex_answer(str(F)).replace("\\log{", "\\ln{") + " + C"
    return dict(
        text=text,
        latex=text,
        answer=str(F),
        answer_text=answer_text,
        answer_latex=answer_latex,
        distractors=distractor_strs,
        working=list(working or []),
    )


@generator(
    "p13c_antideriv_notation",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13c_antideriv_notation(rng, tier="easy"):
    """See module docstring for the full ladder, the direction-flip design
    and the boundaries against `integrate-polynomial`/`integrate-trig`/
    `solve-ode-separable-poly`.

    easy:   F(x) = a*x^n, a single power term; write ∫f(x) dx from the
            stated differentiation fact.
    medium: F(x) a 2-3 term polynomial sum; same shape, more terms.
    hard:   F(x) exponential/sine/cosine -- unreachable via the polynomial
            power rule; the ONLY route is reading the stated notation fact.
            (Not ln -- see the module docstring's history note.)
    exam:   linearity from two stated facts (one polynomial, one
            exponential-or-sine); recover p*F(x) + q*G(x) + C.
    """
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    return _exam(rng)
