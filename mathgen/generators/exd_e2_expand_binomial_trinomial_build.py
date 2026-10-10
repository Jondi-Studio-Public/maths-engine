"""Generates `expand_binomial_trinomial` questions (`skills/expressions-equivalence/expand-binomial-
trinomial.yaml`): expand a binomial multiplied by a trinomial.

The hazard: ``grade_expression`` compares by VALUE, so a learner who types the
question's own unexpanded product back -- ``(x + a)(2x + 3y - 5)`` -- is by value correct and the skill
teaches nothing. ``answer_form="expanded"`` is set on every question here. Unlike the algebraic-fraction generators of the same family, no special
construction effort is needed to make the canonical bracket-free: every ``answer_text`` below is built
ENTIRELY from ``mathgen.core.join_terms``/``fmt_signed`` -- those two helpers never emit a "(" character -- so
a fully expanded, fully-collected polynomial answer is bracket-free by construction, and the unexpanded
question text (always literally ``(binomial)(second factor)``, two bracketed groups multiplied together)
always carries one. Confirmed directly against the real grader in this module's own test file (the "own wrong
form is rejected" proof), not merely asserted.

Four sources, genuine reasoning escalation (not range):

- **easy** -- binomial x binomial, ONE variable, numeric coefficients:
  ``(x + p)(qx + r)``, with ``q`` drawn to exclude +-1 so this never
  coincides with the existing monic ``expand_binomial_product_monic``
  (``p7b_quadratics_build.py``) or ``difference-of-squares`` draw shapes --
  both of those are always monic in the second factor.
- **medium** -- binomial x TRINOMIAL, TWO variables, numeric coefficients:
  ``(x + p)(qx + ry + s)``. New reasoning step: six partial products instead
  of four, and a cross term ``xy`` that never combines with anything else.
- **hard** -- binomial x trinomial, two variables, ONE coefficient replaced
  by a symbolic parameter ``a`` (always the binomial's own second term):
  ``(x + a)(qx + ry + s)``. The learner must carry ``a`` through the
  expansion algebraically (it survives into the ``x``, ``y`` and constant
  coefficients of the answer) rather than collapsing it to a number.
- **exam** -- an exam-style shape, TWO symbolic parameters:
  ``(x + a)(2x + by - 5)`` style -- ``a`` as at hard, plus a SECOND parameter
  ``b`` that replaces exactly one of the trinomial's three numeric
  coefficients, chosen at random per seed. Varying WHICH coefficient ``b``
  replaces (rather than always the ``y`` coefficient, as a single worked
  example might show) is what gives this tier real prompt-shape variety
  bare symbol name or its textual position, and a fixed slot would only ever

Construction, not rejection sampling: every coefficient (``p``/``q``/``r``/``s``) is drawn nonzero by
construction (``_signed_nonzero``), and the symbolic parameters replace an ENTIRE coefficient slot outright
(coefficient magnitude 1) rather than being drawn and retried -- so there is no rejection loop anywhere in
this module, and every misconception distractor is guaranteed distinct from the answer by the SAME
nonzero-by-construction argument (see ``_make``'s three ``check_distinct`` calls), not by chance.

Independent check, second route (every tier):
each question hand-builds its answer via ``join_terms``/``fmt_signed`` from
the same six (four at easy) raw partial-product monomials it also displays
as ``working`` -- never via ``sympy.expand`` -- and then separately builds
real ``sympy`` expressions for the binomial and second factor from the SAME
drawn values and confirms ``sympy.expand(binomial_expr * factor_expr)``
agrees via ``check_equivalent``. The hand-built string construction and
sympy's own ``expand()`` share no code, so this is a genuine second route,
not a re-run of the same arithmetic (``t2_polynomial_combine_build.py`` is
the precedent for this exact discipline; ``p7b_quadratics_build.py``'s own
independent check is numeric substitution instead, because ITS construction
already calls ``sympy.expand()`` as its primary route and so cannot reuse it
as a second one).

Misconceptions (every tier unless noted, ``check_distinct`` unconditionally):

- **only-multiplied-first-terms** -- distributes the binomial's leading
  ``x`` across the second factor but drops the second binomial term's own
  distribution entirely (a genuine partial FOIL/expansion).
- **sign-error-on-one-term** -- flips the sign of exactly one of the second
  factor's original terms (chosen at random per seed: ``x``/``y``/const, or
  just ``x``/const at easy, which has no ``y`` term to flip) before
  distributing.
- **dropped-cross-term** (medium tier and up only) -- omits the ``xy``
  monomial entirely from the final answer. NOT exercisable at easy: a
  binomial x binomial product has no cross term to drop in the first place
  (its two variables never differ), so this candidate is only built when
  ``has_y`` is true.

Every distractor is formatted exactly like the answer -- built from the same
raw-term list via the same ``join_terms`` call -- so no option is
identifiable by shape alone.
"""

from __future__ import annotations

import random

from sympy import Symbol, expand, symbols

from ..core import check_distinct, check_equivalent, generator, join_terms, latex_answer, pick_distractors

x, y = symbols("x y")
a_sym = Symbol("a")
b_sym = Symbol("b")

# Every reachable ``var`` token this module ever hand-builds (base shape
# 'x^2'/'xy'/'x'/'y'/'' optionally prefixed by the symbolic content
# multiplying it: '', 'a', 'b' or 'ab') -- a fixed display order, x^2 down to
# the constant family, plain-numeric before a-prefixed before b-prefixed
# before both. Not every entry is reachable by every tier (only exam ever
# produces a 'b'/'ab' prefix), which is harmless: `_merge` only ever looks up
# a var it actually built.
_VAR_ORDER = {
    "x^2": 0,
    "ax^2": 1,
    "bx^2": 2,
    "abx^2": 3,
    "xy": 10,
    "axy": 11,
    "bxy": 12,
    "abxy": 13,
    "x": 20,
    "ax": 21,
    "bx": 22,
    "abx": 23,
    "y": 30,
    "ay": 31,
    "by": 32,
    "aby": 33,
    "": 40,
    "a": 41,
    "b": 42,
    "ab": 43,
}


def _signed_nonzero(rng: random.Random, lo: int, hi: int) -> int:
    """A nonzero integer of magnitude in [lo, hi], random sign. lo must be
    >= 1 -- this is not a general nonzero-in-range draw, it is specifically
    "pick a magnitude, then a sign", so every coefficient in this module is
    guaranteed nonzero without ever retrying a draw."""
    magnitude = rng.randint(lo, hi)
    return magnitude if rng.choice([True, False]) else -magnitude


def _combine(m: tuple, slot: tuple, base_var: str) -> tuple[int, str]:
    """The single monomial `m * slot * base_var`, as a (coef, var) pair.

    `m` and `slot` are each ("num", value) or ("sym", letter) -- `m` is
    always ("num", p) or ("sym", "a") (the binomial's own second term);
    `slot` is one of the second factor's three coefficients, always
    ("num", value) except at exam tier where exactly one slot is
    ("sym", "b"). `base_var` already encodes the x/y content contributed
    independently of m and slot: 'x^2', 'xy', 'x', 'y' or '' (bare
    constant)."""
    m_kind, m_val = m
    s_kind, s_val = slot
    if m_kind == "num" and s_kind == "num":
        return m_val * s_val, base_var
    if m_kind == "num" and s_kind == "sym":
        return m_val, f"{s_val}{base_var}"
    if m_kind == "sym" and s_kind == "num":
        return s_val, f"{m_val}{base_var}"
    letters = "".join(sorted({m_val, s_val}))
    return 1, f"{letters}{base_var}"


def _scale(term: tuple[int, str], sign: int) -> tuple[int, str]:
    coef, var = term
    return coef * sign, var


def _foil_halves(m, cx, cy, cc, has_y: bool) -> tuple[list, list]:
    """The raw partial products, split into the "x times the second factor"
    half and the "m times the second factor" half -- three entries each
    (two, when `has_y` is False and there is no xy/y contribution)."""
    first = [_combine(("num", 1), cx, "x^2")]
    if has_y:
        first.append(_combine(("num", 1), cy, "xy"))
    first.append(_combine(("num", 1), cc, "x"))
    second = [_combine(m, cx, "x")]
    if has_y:
        second.append(_combine(m, cy, "y"))
    second.append(_combine(m, cc, ""))
    return first, second


def _merge(terms: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Collect true like terms (identical `var`) by summing their integer
    coefficients, in `_VAR_ORDER`'s fixed display order. Two raw terms with
    DIFFERENT var strings are never merged even if a human might call them
    both "the x term" -- e.g. a numeric "x" term and a symbolic "ax" term
    are different monomials once `a` is tracked, exactly the discipline that
    keeps the result bracket-free (no "(coef1 + coef2*a)x" grouping is ever
    attempted; sympy's own default expansion makes the identical choice)."""
    totals: dict[str, int] = {}
    for coef, var in terms:
        totals[var] = totals.get(var, 0) + coef
    return sorted(((c, v) for v, c in totals.items()), key=lambda cv: _VAR_ORDER[cv[1]])


def _var_to_expr(var: str):
    """The sympy multiplier for one `var` token -- the independent-check
    bridge from this module's own hand-built (coef, var) pairs into a real
    sympy expression, so `check_equivalent` can compare against a
    genuinely-independent `sympy.expand()` call (see module docstring)."""
    s = var
    prefix = 1
    if s.startswith("ab"):
        prefix, s = a_sym * b_sym, s[2:]
    elif s.startswith("a"):
        prefix, s = a_sym, s[1:]
    elif s.startswith("b"):
        prefix, s = b_sym, s[1:]
    suffix = {"x^2": x**2, "xy": x * y, "x": x, "y": y, "": 1}
    if s not in suffix:
        raise ValueError(f"unrecognised var token {var!r}")
    return prefix * suffix[s]


def _terms_to_expr(terms: list[tuple[int, str]]):
    total = 0
    for coef, var in terms:
        total += coef * _var_to_expr(var)
    return total


def _slot_entry(slot: tuple, base_var: str) -> tuple[int, str]:
    """One second-factor coefficient as a `join_terms`-ready (coef, var)
    pair -- a numeric slot keeps its own value; a symbolic ('b') slot is
    coefficient 1 on a 'b'-prefixed var, i.e. always shown as a bare
    unit-coefficient symbol ('by', 'bx', 'b'), matching the
    "2x + by - 5" style -- this module never draws a symbolic slot with any
    other magnitude."""
    kind, val = slot
    return (val, base_var) if kind == "num" else (1, f"b{base_var}")


def _binomial_text(m: tuple) -> str:
    kind, val = m
    if kind == "sym":
        return "x + a"
    return join_terms([(1, "x"), (val, "")])


def _second_factor_text(cx, cy, cc, has_y: bool) -> str:
    entries = [_slot_entry(cx, "x")]
    if has_y:
        entries.append(_slot_entry(cy, "y"))
    entries.append(_slot_entry(cc, ""))
    return join_terms(entries)


def _slot_value_expr(slot: tuple):
    kind, val = slot
    return b_sym if kind == "sym" else val


def _binomial_expr(m: tuple):
    kind, val = m
    return x + (a_sym if kind == "sym" else val)


def _second_factor_expr(cx, cy, cc, has_y: bool):
    expr = _slot_value_expr(cx) * x + _slot_value_expr(cc)
    if has_y:
        expr += _slot_value_expr(cy) * y
    return expr


def _sign_error_variant(m, cx, cy, cc, has_y: bool, flip: str) -> list[tuple[int, str]]:
    """The raw terms with exactly one of the second factor's ORIGINAL three
    terms sign-flipped before distributing -- both the "x times that slot"
    and the "m times that slot" partial products flip together, which is
    what "flip the term before distributing" actually means."""
    first, second = _foil_halves(m, cx, cy, cc, has_y)
    first, second = list(first), list(second)
    first_index = {"x": 0, "y": 1 if has_y else None, "const": len(first) - 1}
    second_index = {"x": 0, "y": 1 if has_y else None, "const": len(second) - 1}
    fi, si = first_index[flip], second_index[flip]
    if fi is not None:
        first[fi] = _scale(first[fi], -1)
    if si is not None:
        second[si] = _scale(second[si], -1)
    return first + second


def _make(rng: random.Random, m: tuple, cx: tuple, cy: tuple | None, cc: tuple, has_y: bool) -> dict:
    first, second = _foil_halves(m, cx, cy, cc, has_y)
    raw_terms = first + second
    merged = _merge(raw_terms)
    answer_text = join_terms(merged)
    answer_expr = _terms_to_expr(merged)

    # Independent check, second route: real sympy expressions built from the
    # SAME m/cx/cy/cc, multiplied and expanded by sympy itself -- see module
    # docstring for why this is genuinely independent of the join_terms
    # construction above.
    check_equivalent(
        answer_expr,
        expand(_binomial_expr(m) * _second_factor_expr(cx, cy, cc, has_y)),
        "expand_binomial_trinomial: hand-built expansion must agree with sympy's independent expand()",
    )

    binomial_text = _binomial_text(m)
    factor_text = _second_factor_text(cx, cy, cc, has_y)
    product_text = f"({binomial_text})({factor_text})"
    text = f"Expand and simplify {product_text}."

    # only-multiplied-first-terms: drops the "m times the second factor"
    # distribution entirely -- guaranteed nonzero because m is always
    # nonzero (drawn nonzero, or the symbol 'a') and the second factor is
    # never the zero polynomial (every slot is drawn nonzero too).
    only_first = _merge(first)
    only_first_text = join_terms(only_first)
    check_distinct(
        answer_expr,
        _terms_to_expr(only_first),
        msg="expand_binomial_trinomial: only-multiplied-first-terms distractor must be a genuinely different value",
    )
    cand_texts = [only_first_text]

    # sign-error-on-one-term: guaranteed nonzero for the same reason -- the
    # flipped slot's own two partial products are never both zero.
    flip_choices = ["x", "y", "const"] if has_y else ["x", "const"]
    flip = rng.choice(flip_choices)
    sign_error_terms = _merge(_sign_error_variant(m, cx, cy, cc, has_y, flip))
    sign_error_text = join_terms(sign_error_terms)
    check_distinct(
        answer_expr,
        _terms_to_expr(sign_error_terms),
        msg="expand_binomial_trinomial: sign-error-on-one-term distractor must be a genuinely different value",
    )
    cand_texts.append(sign_error_text)

    # dropped-cross-term: medium tier and up only -- see module docstring
    # for why this is not exercisable at easy (no xy term exists to drop).
    if has_y:
        dropped = [first[0], first[2], *second]  # first[1] is the xy raw term
        dropped_terms = _merge(dropped)
        dropped_text = join_terms(dropped_terms)
        check_distinct(
            answer_expr,
            _terms_to_expr(dropped_terms),
            msg="expand_binomial_trinomial: dropped-cross-term distractor must be a genuinely different value",
        )
        cand_texts.append(dropped_text)

    raw_line = join_terms(raw_terms)
    working = [
        f"{product_text} = {raw_line}",
        f"= {answer_text}",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_expr),
        working=working,
        distractors=pick_distractors(answer_text, cand_texts),
        answer_form="expanded",
    )


def _easy(rng: random.Random) -> dict:
    p = _signed_nonzero(rng, 1, 9)
    q = rng.choice([v for v in range(-6, 7) if v not in (0, 1, -1)])
    r = _signed_nonzero(rng, 1, 9)
    return _make(rng, ("num", p), ("num", q), None, ("num", r), has_y=False)


def _medium(rng: random.Random) -> dict:
    p = _signed_nonzero(rng, 1, 8)
    q = _signed_nonzero(rng, 1, 8)
    r = _signed_nonzero(rng, 1, 8)
    s = _signed_nonzero(rng, 1, 8)
    return _make(rng, ("num", p), ("num", q), ("num", r), ("num", s), has_y=True)


def _hard(rng: random.Random) -> dict:
    q = _signed_nonzero(rng, 1, 9)
    r = _signed_nonzero(rng, 1, 9)
    s = _signed_nonzero(rng, 1, 9)
    return _make(rng, ("sym", "a"), ("num", q), ("num", r), ("num", s), has_y=True)


def _exam(rng: random.Random) -> dict:
    slot = rng.choice(["x", "y", "const"])
    q = _signed_nonzero(rng, 1, 9)
    r = _signed_nonzero(rng, 1, 9)
    s = _signed_nonzero(rng, 1, 9)
    cx = ("sym", "b") if slot == "x" else ("num", q)
    cy = ("sym", "b") if slot == "y" else ("num", r)
    cc = ("sym", "b") if slot == "const" else ("num", s)
    return _make(rng, ("sym", "a"), cx, cy, cc, has_y=True)


@generator(
    "exd_e2_expand_binomial_trinomial",
    level=10,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e2_expand_binomial_trinomial(rng: random.Random, tier: str = "easy") -> dict:
    """Expand a binomial times a second factor (a binomial at easy, a
    trinomial at every tier above it) into a fully expanded, fully-collected
    polynomial. See module docstring for the four tiers' escalation."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    if tier == "exam":
        return _exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
