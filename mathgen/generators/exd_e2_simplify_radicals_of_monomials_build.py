"""Generates `exd_e2_simplify_radicals_of_monomials` questions: simplify the radicals of monomials into
coefficient and variable-power form.

Built by FORWARD CONSTRUCTION only (a mandatory discipline here): the
answer's coefficient ``K`` and each variable's own net exponent ``e_v`` are
drawn FIRST, then the two radicands are raised FROM them -- never the other
way around, so nothing here is a perfect-power coincidence that has to be
retried for. The identities this relies on (never a per-seed coincidence,
see the module-level constants below):

    k_S = s*s                       (a perfect square, always)
    k_C = (K*s)**3                  (a perfect cube, always)
    cbrt(k_C) / sqrt(k_S) == K      exactly, for every (K, s)

    s_v = 2*d_v                     (even, always)
    c_v = 3*(e_v + d_v)             (a nonnegative multiple of 3, always)
    c_v/3 - s_v/2 == e_v            identically, for every (e_v, d_v)

``d_v`` is drawn from {1, 2} for EVERY variable (never 0) -- not just the
"small nonnegative integer" range, but a subset of it,
deliberately chosen so every question's square-root radicand carries an
actual variable factor for every variable in play. That is what guarantees
the "own wrong form is rejected" proof (see ``TestOwnFormRejectedProof`` in
this module's test file) holds on EVERY question, not just most of them: the
per-root-extracted-but-uncancelled fraction always has a compound
denominator (a numeric factor times at least one variable power) and so
always needs a bracket, which is what ``answer_form="expanded"`` rejects on.
Numerator-variable exponents ``e_v`` are drawn from {1, 2, 3}; the (at most
one) denominator variable's ``e_v`` is drawn from ``range(-d_v, 0)``, which
is always non-empty because ``d_v >= 1``.

The canonical answer is always a bare monomial fraction: a coefficient,
zero or more numerator variable powers, and AT MOST ONE denominator
variable's Pow (never a bracket -- "/" and "**"/"^" precedence make a single
denominator Pow unambiguous). That is what lets this skill reuse the
EXISTING ``answer_form="expanded"`` (mathgen/grading/typed.py) rather than needing a
new form: the canonical never needs a bracket, so grading it against itself
always passes the "no leftover non-enclosing (" check, while both
rejection-target forms below always carry one (see "The two rejection
targets").

Root-index pairing (cube root over square root) is fixed across every tier.
Four tiers, genuine reasoning escalation:

  easy   -- ONE variable, e_v always >= 1 (no denominator variable at all --
            the answer is a bare K*a**e monomial, no fraction).
  medium -- TWO variables, still every e_v >= 1 (more bookkeeping across two
            variables, still no denominator step).
  hard   -- TWO or THREE variables (chosen per seed), exactly ONE drawn with
            e_v < 0 -- the new reasoning step over medium is placing a
            variable under the fraction bar, not just computing another
            numerator exponent.
  exam   -- THREE variables, one denominator variable,
            PLUS a genuinely new element over hard: the cube
            root's own coefficient may be a NEGATIVE perfect cube (a real,
            well-defined cube root unlike a square root of a negative
            number), exercised via a sign draw hard never makes. The
            generator's own sqrt radicand is unaffected (always positive --
            k_S = s*s is a square, never negative).

The two rejection targets, and why only one is load-bearing
-------------------------------------------------------------
1. Roots-extracted-but-uncancelled (PRIMARY, required): e.g. for the module
   docstring's worked K=3, s=2, single-variable-a, d_a=1, e_a=2 example
   (k_C=216, k_S=4, c_a=9, s_a=2) -- ``6*a**3/(2*a)``. This is PROVABLY
   value-equal to the canonical ``3*a**2`` by ordinary polynomial
   arithmetic (integer exponents throughout, no root-branch ambiguity), so
   its rejection is a clean demonstration that the FORM check, not a lucky
   value-inequality, is what catches it. Measured directly (see this
   is True, and the real grader still rejects the former against the
   latter's ``answer_form="expanded"``, because the raw text carries a
   non-fully-enclosing "(" around the compound denominator.
2. The fully unextracted ``cbrt(...)/sqrt(...)`` text -- included as an
   ADDITIONAL, honestly-caveated check only. Measured directly (same
   example): ``symbolically_equal("cbrt(216*a**9)/sqrt(4*a**2)", "3*a**2")``
   is FALSE for generic (non-positive-assumption) symbols -- sympy cannot
   prove ``(a**9)**(1/3) == a**3`` without knowing ``a`` is positive, and a
   learner's typed text always parses to plain, assumption-free symbols.
   So this form's rejection is NOT reliable evidence that the FORM check
   (rather than plain value-inequality) is what caught it; only proof 1 is
   load-bearing.

Independent check, second route (every tier, unconditionally)
------------------------------------------------------------- Every question rebuilds ``real_root(k_C *
prod(a**c_v), 3) / sqrt(k_S * prod(a**s_v))`` from k_C/k_S/c_v/s_v directly -- never from this module's own
K/e_v values -- with sympy symbols declared ``positive=True`` for this internal check ONLY -- and confirms it
agrees with the monomial this module constructed via ``check_equivalent``.

**Deviation from the obvious construction, justified and documented
here**: the obvious choice would be ``Pow(..., Rational(1,3))`` for this route.
Measured directly: plain ``Pow(negative, Rational(1, 3))`` in sympy returns
the PRINCIPAL COMPLEX cube root, not the real negative one (e.g.
``(-91125)**Rational(1,3)`` numerically evaluates to a complex number, not
``-45``) -- so a literal ``Pow`` call would make this independent check
fail (or assert equality against a complex number) on every exam-tier
question with a negative-signed coefficient, which is exactly the tier this
route most needs to prove. ``sympy.real_root`` is used for the cube root
instead (confirmed directly: ``real_root(-91125, 3) == -45``), and plain
``Pow``/``sqrt`` is kept for the square root (its own radicand is always
non-negative by construction, so no branch issue there).

Misconceptions (every one, ``check_distinct`` unconditionally, every tier)
----------------------------------------------------------------------------
- **wrong-root-index**: applies the WRONG divisor to each variable's own
  cube/square exponent (halves the cube exponent instead of thirding it,
  thirds the square exponent instead of halving it -- ``c_v // 2 - s_v //
  3`` in place of the correct ``c_v // 3 - s_v // 2``), coefficient left
  correct (the description is specifically an exponent-handling error, not
  a coefficient one). Measured over the module's own (d_v, e_v) ranges
  ({1,2} x {1,2,3} for numerator variables): this always yields a strictly
  POSITIVE wrong exponent for every numerator-type variable, which alone
  guarantees the whole monomial differs from the answer regardless of what
  happens to the (at most one) denominator variable's own wrong exponent --
  proved by ``check_distinct`` on every question, not merely argued here.
- **forgot-the-divide**: reports the cube-root part's own simplification
  alone (``e_v + d_v`` for every variable, i.e. ``c_v // 3`` without ever
  subtracting the square root's own ``d_v``), coefficient ``K_signed * s``
  (the un-divided cube coefficient) -- always a genuinely different integer
  from ``K_signed`` because ``s >= 2`` unconditionally.
- **sign-or-placement-error** (hard/exam only -- no denominator variable
  exists at easy/medium to misplace): moves the one denominator variable
  into the numerator instead of leaving it under the fraction bar --
  differs from the answer in that variable's own exponent sign, which is
  always nonzero by construction.

Every distractor is built with the SAME ``_answer_text`` formatter as the
canonical, so no option is identifiable by shape alone (bare monomial
fraction, at most one denominator Pow, never a bracket).
"""

from __future__ import annotations

import random

from sympy import Symbol, real_root, sqrt, symbols

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, phrase, pick_distractors

VAR_POOL = ["a", "b", "c"]
SYM = dict(zip(VAR_POOL, symbols(" ".join(VAR_POOL))))
POS = {v: Symbol(v, positive=True) for v in VAR_POOL}

_SENTENCE_PATTERNS = (
    "Simplify {expr}.",
    "Simplify {expr}, giving your answer in simplest form.",
    "Simplify {expr} as a single term.",
)


def _fmt_pow(var: str, exp: int) -> str:
    """One variable factor, student-style: bare for exponent 1, '{var}^{exp}'
    otherwise -- never Python's '**' (that is reserved for this module's own
    internal sympy/test-only text, never learner-facing text)."""
    return var if exp == 1 else f"{var}^{exp}"


def _monomial_text(coef: int, exps: dict[str, int]) -> str:
    """'{coef}a^2b' style (implicit multiplication, matching the exam
    exam paper convention of perfect-power coefficients and mathgen.text_quality's
    "star_operator" prompt check -- a bare "*" between factors is a flagged
    defect in `.text`), variables in a fixed alphabetical order so the same
    drawn values always render identically regardless of dict order.
    `sympy.parsing.sympy_parser.implicit_multiplication_application` (used
    throughout `mathgen/grading/typed.py`'s `_parse_expression`) parses this back
    unambiguously -- confirmed directly: `'3a**2c' -> 3*a**2*c`."""
    parts = [str(coef)] + [_fmt_pow(v, exps[v]) for v in sorted(exps)]
    return "".join(parts)


def _answer_text(coef: int, numer_exps: dict[str, int], denom_var: str | None, denom_exp: int | None) -> str:
    """The canonical's own raw text: a bare monomial numerator over AT MOST
    ONE denominator variable Pow, never a bracket."""
    numerator = _monomial_text(coef, numer_exps)
    if denom_var is None:
        return numerator
    return f"{numerator}/{_fmt_pow(denom_var, denom_exp)}"


def _monomial_expr(sym_table: dict, coef, exps: dict[str, int]):
    total = coef
    for v, e in exps.items():
        total *= sym_table[v] ** e
    return total


def _draw_variable(rng: random.Random, negative: bool) -> tuple[int, int]:
    """(d_v, e_v) for one variable. `d_v` is drawn from {1, 2} -- never 0,
    see module docstring for why this is what keeps the square-root
    radicand's own extracted form always compound (own-wrong-form-rejected
    proof). `e_v` is then drawn from a RANGE that keeps `e_v + d_v >= 0` by
    construction (never a redraw): {1, 2, 3} for a numerator variable,
    `range(-d_v, 0)` for the (at most one) denominator variable -- always
    non-empty because `d_v >= 1`."""
    d_v = rng.choice([1, 2])
    if negative:
        e_v = rng.randint(-d_v, -1)
    else:
        e_v = rng.randint(1, 3)
    return d_v, e_v


def _build(rng: random.Random, num_vars: int, allow_denom: bool, allow_negative_coef: bool) -> dict:
    vars_list = sorted(rng.sample(VAR_POOL, num_vars))
    denom_var = rng.choice(vars_list) if allow_denom else None

    d: dict[str, int] = {}
    e: dict[str, int] = {}
    for v in vars_list:
        d[v], e[v] = _draw_variable(rng, negative=(v == denom_var))

    sign = -1 if (allow_negative_coef and rng.choice([True, False])) else 1
    K = rng.randint(2, 9)
    s = rng.randint(2, 5)
    K_signed = sign * K
    k_S = s * s
    k_C = (K_signed * s) ** 3

    c = {v: 3 * (e[v] + d[v]) for v in vars_list}
    s_exp = {v: 2 * d[v] for v in vars_list}

    numer_exps = {v: e[v] for v in vars_list if e[v] > 0}
    denom_exp = -e[denom_var] if denom_var else None
    check(bool(numer_exps), "at least one numerator variable must survive")

    answer_expr = _monomial_expr(SYM, K_signed, numer_exps)
    if denom_var:
        answer_expr /= SYM[denom_var] ** denom_exp
    answer_text = _answer_text(K_signed, numer_exps, denom_var, denom_exp)

    # Independent check, second route -- see module docstring for why
    # `real_root` (not plain `Pow`) is required for the cube root, and why
    # this whole route uses `positive=True` symbols, never `SYM`.
    cube_radicand_pos = _monomial_expr(POS, k_C, c)
    square_radicand_pos = _monomial_expr(POS, k_S, s_exp)
    independent_answer = real_root(cube_radicand_pos, 3) / sqrt(square_radicand_pos)
    answer_expr_pos = _monomial_expr(POS, K_signed, numer_exps)
    if denom_var:
        answer_expr_pos /= POS[denom_var] ** denom_exp
    check_equivalent(
        answer_expr_pos,
        independent_answer,
        "independent check: cbrt(k_C*prod)/sqrt(k_S*prod), recomputed straight from k_C/k_S/c_v/s_v, "
        "does not match the constructed monomial answer",
    )

    # A denominator variable's own e_v + d_v may land on exactly 0 (e.g.
    # e_v=-1, d_v=1), which makes c_v=0 -- that variable genuinely has no
    # cube-root factor at all (a**0 == 1), so it is dropped from the
    # RADICAND TEXT entirely rather than displayed as a confusing "a^0".
    # s_exp is never 0 (d_v >= 1 unconditionally), so no filtering is needed
    # there, but the same filter is applied for symmetry/defensiveness.
    cube_text = _monomial_text(k_C, {v: x for v, x in c.items() if x > 0})
    square_text = _monomial_text(k_S, {v: x for v, x in s_exp.items() if x > 0})
    expr_text = f"∛({cube_text}) ÷ √({square_text})"
    text = phrase(rng, _SENTENCE_PATTERNS, expr=expr_text)

    # -- wrong-root-index: wrong divisor on every variable's own exponent --
    wrong_e = {v: (c[v] // 2) - (s_exp[v] // 3) for v in vars_list}
    wri_numer = {v: x for v, x in wrong_e.items() if x > 0}
    wri_denom_candidates = [v for v, x in wrong_e.items() if x < 0]
    check(len(wri_denom_candidates) <= 1, "wrong-root-index: more than one negative exponent would need a bracket")
    check(bool(wri_numer), "wrong-root-index: at least one numerator variable must survive")
    wri_denom_var = wri_denom_candidates[0] if wri_denom_candidates else None
    wri_denom_exp = -wrong_e[wri_denom_var] if wri_denom_var else None
    wri_text = _answer_text(K_signed, wri_numer, wri_denom_var, wri_denom_exp)
    wri_expr = _monomial_expr(SYM, K_signed, wri_numer)
    if wri_denom_var:
        wri_expr /= SYM[wri_denom_var] ** wri_denom_exp
    check_distinct(answer_expr, wri_expr, msg="wrong-root-index distractor must be a genuinely different value")

    # -- forgot-the-divide: cube-root part alone, square-root divisor never applied --
    ftd_exps = {v: e[v] + d[v] for v in vars_list}
    check(all(x >= 0 for x in ftd_exps.values()), "forgot-the-divide: an exponent went negative")
    ftd_numer = {v: x for v, x in ftd_exps.items() if x > 0}
    ftd_coef = K_signed * s
    ftd_text = _answer_text(ftd_coef, ftd_numer, None, None)
    ftd_expr = _monomial_expr(SYM, ftd_coef, ftd_numer)
    check_distinct(answer_expr, ftd_expr, msg="forgot-the-divide distractor must be a genuinely different value")

    cand_texts = [wri_text, ftd_text]

    # -- sign-or-placement-error: hard/exam only, when a denominator variable exists --
    if denom_var:
        spe_numer = dict(numer_exps)
        spe_numer[denom_var] = -e[denom_var]
        spe_text = _answer_text(K_signed, spe_numer, None, None)
        spe_expr = _monomial_expr(SYM, K_signed, spe_numer)
        check_distinct(
            answer_expr, spe_expr, msg="sign-or-placement-error distractor must be a genuinely different value"
        )
        cand_texts.append(spe_text)

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_expr),
        distractors=pick_distractors(answer_text, cand_texts),
        answer_form="expanded",
    )


def _easy(rng: random.Random) -> dict:
    return _build(rng, num_vars=1, allow_denom=False, allow_negative_coef=False)


def _medium(rng: random.Random) -> dict:
    return _build(rng, num_vars=2, allow_denom=False, allow_negative_coef=False)


def _hard(rng: random.Random) -> dict:
    num_vars = rng.choice([2, 3])
    return _build(rng, num_vars=num_vars, allow_denom=True, allow_negative_coef=False)


def _exam(rng: random.Random) -> dict:
    return _build(rng, num_vars=3, allow_denom=True, allow_negative_coef=True)


@generator(
    "exd_e2_simplify_radicals_of_monomials",
    level=10,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e2_simplify_radicals_of_monomials(rng: random.Random, tier: str = "easy") -> dict:
    """Simplify cbrt(radicand) / sqrt(radicand) to a bare monomial fraction.
    See module docstring for the four tiers' escalation."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    if tier == "exam":
        return _exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
