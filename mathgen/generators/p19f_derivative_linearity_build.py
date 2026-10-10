"""Generates `p19f_derivative_linearity` questions (ACMMM090): use the linearity properties of the derivative,
answered as an expression.

Scope: given two differentiable functions f and g whose derivatives
f'(x) and g'(x) are stated as expressions, and constants a and b, compute the
derivative of a*f(x) + b*g(x) as a*f'(x) + b*g'(x).

**Sharp neighbours** (never rendered by this generator):
``differentiate-polynomial`` differentiates a bare polynomial FROM SCRATCH --
this skill is NEVER handed a polynomial to differentiate, only
already-differentiated expressions f'(x)/g'(x) to scale and combine.
``senior-define-the-derivative-f-left-x-right-as`` is the first-principles
limit definition -- never used here. ``senior-apply-the-product-quotient-
and-chain-rule-to`` differentiates a COMPOSITE expression (a product,
quotient or composition) -- this skill never multiplies or composes
functions, only scales and adds two already-given derivatives.

**Difficulty ladder** (every tier ``answer: expression``, one grader kind):
- easy: a, b positive small integers; f'(x)/g'(x) each a single term or
  binomial, already simplified.
- medium: a or b negative (sign management enters).
- hard: f'(x)/g'(x) are handed in an UNSIMPLIFIED form (e.g. "3x + 2x"
  instead of "5x") -- the learner must simplify each derivative before
  applying linearity, a genuine extra reasoning step, not a bigger range.
- exam: the combination itself is stated in WORDS instead of bare a/b
  ("h(x) is twice f(x) minus three times g(x); find h'(x)") -- the learner
  must translate English into the coefficients before applying linearity, a
  decision absent from every lower tier.

**Independent check.** The canonical text is built via `join_terms`/
`fmt_signed` from a raw coefficient dict (never `str(sympy_expr)`); the
independent route re-parses that same coefficient dict into a fresh sympy
expression and `check_equivalent`s it against `a*fp_sym + b*gp_sym` built
directly from f'/g''s own raw term lists in sympy -- two separately-typed
routes to the same value, never a hand-rolled `simplify(...) == 0`.

**Misconceptions**: dropped one coefficient (reports `f' + b*g'`, silently
treating a as 1); a sign error on one term; swapped which constant
multiplies which derivative (`b*f' + a*g'` when a != b); at the exam tier,
an English-translation slip that drops the minus sign the words state.
Every candidate checked UNEQUAL to the answer via `check_distinct`/
`check_pairwise_distinct`, unconditionally.
"""

from __future__ import annotations

from sympy import Symbol
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_PARSE_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    fmt_signed,
    generator,
    join_terms,
    latex_answer,
    pairwise_distinct,
    pick_distractors,
)

_x = Symbol("x")


def _rand_nonzero(rng, lo, hi):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _rand_not_one(rng, lo, hi):
    """Nonzero, excluding +1 -- `a` must never be 1, or the "dropped a"
    misconception (`f' + b*g'`, i.e. treating `a` as 1) collapses onto the
    real answer."""
    return rng.choice([v for v in range(lo, hi + 1) if v not in (0, 1)])


def _term_list(rng, *, unsimplified: bool, deg_lo=0, deg_hi=2, coef_lo=-9, coef_hi=9, n_terms=None):
    """A small polynomial as [(coef, power), ...], sorted by descending
    power, with distinct powers -- the canonical, SIMPLIFIED representation
    of f'(x)/g'(x). ``unsimplified`` is applied by the caller (splitting one
    term's coefficient into two addends for display only); this helper only
    ever returns a simplified term list."""
    if n_terms is None:
        n_terms = rng.choice([1, 2])
    powers = rng.sample(range(deg_lo, deg_hi + 1), k=n_terms)
    terms = [(_rand_nonzero(rng, coef_lo, coef_hi), p) for p in sorted(powers, reverse=True)]
    return terms


def _term_text(terms) -> str:
    """Render a simplified [(coef, power), ...] term list the way a student
    would write it, using `join_terms` (never `str(sympy_expr)`, which would
    also auto-simplify a deliberately-unsimplified hard-tier display)."""
    parts = [(coef, "" if power == 0 else ("x" if power == 1 else f"x^{power}")) for coef, power in terms]
    return join_terms(parts)


def _split_unsimplified(rng, terms):
    """For the hard tier: split ONE term's coefficient c into c1 + c2 (both
    nonzero, c1 + c2 == c) so the displayed derivative reads e.g. "3x + 2x"
    instead of the already-combined "5x". Returns the display term list
    (with the split applied) alongside the still-simplified `terms` used for
    every symbolic computation."""
    idx = rng.randrange(len(terms))
    coef, power = terms[idx]
    # split into two nonzero, non-equal-to-coef parts summing to coef
    for _attempt in range(50):
        c1 = rng.randint(1, max(2, abs(coef) + 3)) * (1 if coef >= 0 else -1)
        c2 = coef - c1
        if c1 != 0 and c2 != 0:
            break
    else:
        c1, c2 = coef, 0  # pathological fallback, never hit in practice
    display_terms = terms[:idx] + [(c1, power), (c2, power)] + terms[idx + 1 :]
    return display_terms


def _sym_from_terms(terms):
    expr = 0
    for coef, power in terms:
        expr += coef * _x**power
    return expr


def _fmt_expr_terms(terms) -> str:
    """Build the display string for a full symbolic sum of (coef, power)
    terms in x, collapsing equal powers -- used for the canonical answer
    a*f' + b*g', where powers from f' and g' may collide."""
    by_power: dict[int, int] = {}
    for coef, power in terms:
        by_power[power] = by_power.get(power, 0) + coef
    ordered = sorted(((c, p) for p, c in by_power.items() if c != 0), key=lambda t: -t[1])
    if not ordered:
        return "0"
    return _term_text(ordered)


def _scale(terms, k):
    return [(k * coef, power) for coef, power in terms]


def _combine(*term_lists):
    combined: list[tuple[int, int]] = []
    for tl in term_lists:
        combined.extend(tl)
    return combined


def _draw_fg(rng, *, a, b, coef_lo=-9, coef_hi=9):
    """Draw f'(x), g'(x) as two SEPARATE simplified term lists, redrawing
    (up to 50 attempts) until the resulting answer and every misconception
    candidate this skill offers (dropped-a, sign-error, and swapped when
    a != b) are pairwise distinct -- an accidental proportionality between
    two independently-drawn polynomials is rare but real, and a collision
    there must resample rather than raise. The final `check_pairwise_distinct`
    in `_build_question` is the hard backstop if 50 attempts are exhausted."""
    f_terms = g_terms = None
    for _attempt in range(50):
        f_terms = _term_list(rng, unsimplified=False, coef_lo=coef_lo, coef_hi=coef_hi)
        g_terms = _term_list(rng, unsimplified=False, coef_lo=coef_lo, coef_hi=coef_hi)
        fp_sym = _sym_from_terms(f_terms)
        gp_sym = _sym_from_terms(g_terms)
        answer_sym = a * fp_sym + b * gp_sym
        if answer_sym == 0:
            continue
        candidates = [fp_sym + b * gp_sym, a * fp_sym - b * gp_sym]
        if a != b:
            candidates.append(b * fp_sym + a * gp_sym)
        if pairwise_distinct(answer_sym, *candidates):
            return f_terms, g_terms
    return f_terms, g_terms  # pragma: no cover -- exhausted only pathologically


def _build_question(rng, *, a, b, f_terms, g_terms, f_display_terms, g_display_terms, prompt_text_fn):
    """Shared construction for every tier: given the drawn a, b and f'/g'
    term lists (plus their possibly-unsimplified DISPLAY term lists), build
    the canonical answer, its independent check, the misconception
    candidates and the final Question dict."""
    fp_sym = _sym_from_terms(f_terms)
    gp_sym = _sym_from_terms(g_terms)
    answer_sym = a * fp_sym + b * gp_sym
    check(answer_sym != 0, f"degenerate combination: a={a} b={b} f'={fp_sym} g'={gp_sym}")

    answer_terms = _combine(_scale(f_terms, a), _scale(g_terms, b))
    answer_text = _fmt_expr_terms(answer_terms)

    # Independent check: re-parse the DISPLAYED answer_text with sympy (a
    # different route from the coefficient-dict construction above) and
    # compare against a*fp_sym + b*gp_sym built straight from f'/g's own raw
    # sympy expressions.
    reparsed = parse_expr(answer_text.replace("^", "**"), transformations=_PARSE_TRANSFORMATIONS)
    check_equivalent(
        reparsed, answer_sym, f"independent re-parse check failed: a={a} b={b} answer_text={answer_text!r}"
    )

    # Misconceptions.
    dropped_a = _fmt_expr_terms(_combine(f_terms, _scale(g_terms, b)))
    dropped_a_sym = fp_sym + b * gp_sym
    sign_error_terms = _combine(_scale(f_terms, a), _scale(g_terms, -b))
    sign_error_sym = a * fp_sym - b * gp_sym
    swapped_terms = _combine(_scale(f_terms, b), _scale(g_terms, a))
    swapped_sym = b * fp_sym + a * gp_sym

    candidates_sym = [dropped_a_sym, sign_error_sym]
    candidates_text = [dropped_a, _fmt_expr_terms(sign_error_terms)]
    if a != b:
        candidates_sym.append(swapped_sym)
        candidates_text.append(_fmt_expr_terms(swapped_terms))

    check(
        pairwise_distinct(answer_sym, *candidates_sym),
        f"misconception collision before check: a={a} b={b} f'={fp_sym} g'={gp_sym}",
    )
    check_distinct(answer_sym, *candidates_sym, msg=f"distractor equals answer: a={a} b={b} f'={fp_sym} g'={gp_sym}")
    check_pairwise_distinct(
        answer_sym, *candidates_sym, msg=f"distractor collision: a={a} b={b} f'={fp_sym} g'={gp_sym}"
    )

    text = prompt_text_fn(f_display_terms, g_display_terms, a, b)
    distractors = pick_distractors(answer_text, candidates_text)

    return dict(
        text=text,
        latex=text,
        answer=str(answer_sym),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )


def _easy(rng):
    a = rng.randint(2, 6)
    b = rng.randint(1, 6)
    f_terms, g_terms = _draw_fg(rng, a=a, b=b, coef_lo=-9, coef_hi=9)
    f_disp, g_disp = f_terms, g_terms

    def prompt(f_disp, g_disp, a, b):
        return (
            f"f'(x) = {_term_text(f_disp)} and g'(x) = {_term_text(g_disp)}. "
            f"h(x) = {fmt_signed(a, 'f(x)')} + {b}g(x). Find h'(x)."
        )

    return _build_question(
        rng,
        a=a,
        b=b,
        f_terms=f_terms,
        g_terms=g_terms,
        f_display_terms=f_disp,
        g_display_terms=g_disp,
        prompt_text_fn=prompt,
    )


def _medium(rng):
    a = _rand_not_one(rng, -6, 6)
    b = _rand_nonzero(rng, -6, 6)
    if a > 0 and b > 0:  # medium's own escalation: at least one of a, b negative
        b = -b
    f_terms, g_terms = _draw_fg(rng, a=a, b=b, coef_lo=-9, coef_hi=9)
    f_disp, g_disp = f_terms, g_terms

    def prompt(f_disp, g_disp, a, b):
        b_term = fmt_signed(b, "g(x)")
        sign = "+" if b >= 0 else "-"
        b_mag_term = fmt_signed(abs(b), "g(x)")
        return (
            f"f'(x) = {_term_text(f_disp)} and g'(x) = {_term_text(g_disp)}. "
            f"h(x) = {fmt_signed(a, 'f(x)')} {sign} {b_mag_term}. Find h'(x)."
        )

    return _build_question(
        rng,
        a=a,
        b=b,
        f_terms=f_terms,
        g_terms=g_terms,
        f_display_terms=f_disp,
        g_display_terms=g_disp,
        prompt_text_fn=prompt,
    )


def _hard(rng):
    a = _rand_not_one(rng, -6, 6)
    b = _rand_nonzero(rng, -6, 6)
    f_terms, g_terms = _draw_fg(rng, a=a, b=b, coef_lo=-9, coef_hi=9)
    f_disp = _split_unsimplified(rng, f_terms)
    g_disp = _split_unsimplified(rng, g_terms)

    def prompt(f_disp, g_disp, a, b):
        sign = "+" if b >= 0 else "-"
        b_mag_term = fmt_signed(abs(b), "g(x)")
        return (
            f"f'(x) = {_term_text(f_disp)} and g'(x) = {_term_text(g_disp)}. "
            f"h(x) = {fmt_signed(a, 'f(x)')} {sign} {b_mag_term}. Find h'(x), "
            f"simplifying f'(x) and g'(x) first."
        )

    return _build_question(
        rng,
        a=a,
        b=b,
        f_terms=f_terms,
        g_terms=g_terms,
        f_display_terms=f_disp,
        g_display_terms=g_disp,
        prompt_text_fn=prompt,
    )


_NUMBER_WORDS = {1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times"}


def _exam(rng):
    a = rng.randint(2, 6)  # exam tier: keep a positive, spelled with a word
    b = _rand_nonzero(rng, -6, 6)
    while abs(b) == a:
        b = _rand_nonzero(rng, -6, 6)  # avoid a == |b| collapsing the swap misconception
    f_terms, g_terms = _draw_fg(rng, a=a, b=b, coef_lo=-9, coef_hi=9)
    f_disp, g_disp = f_terms, g_terms

    def prompt(f_disp, g_disp, a, b):
        # h(x) = a*f(x) + b*g(x), spelled in words: b >= 0 reads as "plus
        # <word> g(x)"; b < 0 reads as "minus <word> g(x)" -- consistent
        # with the actual signed combination `_build_question` computes.
        # The English-translation-slip misconception (reading the stated
        # sign backwards) is exactly `_build_question`'s own general
        # sign-error candidate (a*f' - b*g'), so no extra distractor logic
        # is needed here.
        a_word = _NUMBER_WORDS[a]
        b_word = _NUMBER_WORDS[abs(b)]
        connective = "plus" if b >= 0 else "minus"
        return (
            f"f'(x) = {_term_text(f_disp)} and g'(x) = {_term_text(g_disp)}. "
            f"h(x) is {a_word} f(x) {connective} {b_word} g(x). Find h'(x)."
        )

    return _build_question(
        rng,
        a=a,
        b=b,
        f_terms=f_terms,
        g_terms=g_terms,
        f_display_terms=f_disp,
        g_display_terms=g_disp,
        prompt_text_fn=prompt,
    )


@generator(
    "p19f_derivative_linearity",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19f_derivative_linearity(rng, tier="easy"):
    """See module docstring for the full difficulty ladder and boundaries.

    easy:   a, b positive small integers; f'/g' each a single term or
            binomial, already simplified.
    medium: a or b negative.
    hard:   f'/g' handed unsimplified (e.g. "3x + 2x"); the learner must
            simplify before combining.
    exam:   the combination stated in WORDS ("h(x) is twice f(x) minus
            three times g(x)") instead of bare a/b.
    """
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    return _exam(rng)
