"""Polynomial definition and classification.

Two genuinely different graded operations, drawn roughly evenly per tier
(``rng.choice`` between the two modes, never a params-driven switch -- no
extra ``params`` beyond the difficulty preset's own ``tier`` were needed):

- Mode A ("Classify this polynomial..."): given a genuine polynomial, state
  its degree and whether it is a monomial/binomial/trinomial. ``hard``
  escalates to two-variable terms, where reading only the ``x`` exponent of
  the highest-degree term (ignoring its ``y`` contribution) is a real,
  exercisable misconception, not a magnitude bump.
- Mode B ("Is this a polynomial? If not, why not..."): given an expression
  that may or may not be a polynomial, name which of four fixed categories
  applies -- a genuine polynomial, or one of three violations (negative
  exponent, fractional exponent, variable in the denominator). ``hard``
  pairs a constant-denominator polynomial trap (``x/2``, still a polynomial)
  against the variable-denominator violation (``2/x``, not one).

No sympy: every polynomial is built directly as a list of
``(coefficient, variable-text)`` terms with an explicit, tracked exponent
pair, fed straight into ``core.join_terms`` -- there is no expression to
parse or expand, so there is no ``_poly_display``-style helper to hand-roll.
Variable-text shape is pinned exactly: ``""`` for a constant,
``"x"``/``"x^{p}"`` for a single-variable term, and the x-part then y-part
concatenated with no separator for a two-variable term (``"x^2y"``, not
``"x^2 y"``).
"""

from __future__ import annotations

from ..core import check, check_distinct, generator, join_terms, latex_answer, pick_distractors

_TYPE_NAMES = {1: "monomial", 2: "binomial", 3: "trinomial"}
# Adjacent term-count label: monomial/trinomial each have exactly one
# neighbour; binomial has two, so a fixed choice would offer the same
# neighbour every time -- drawn with rng.choice at the call site instead.
_ADJACENT_TYPES = {1: (2,), 2: (1, 3), 3: (2,)}

_CATEGORY_POLYNOMIAL = "Yes, it is a polynomial"
_CATEGORY_NEGATIVE_EXPONENT = "No - it has a negative exponent"
_CATEGORY_FRACTIONAL_EXPONENT = "No - it has a fractional exponent"
_CATEGORY_VARIABLE_DENOMINATOR = "No - the variable is in the denominator"
_VIOLATION_CATEGORIES = (
    _CATEGORY_NEGATIVE_EXPONENT,
    _CATEGORY_FRACTIONAL_EXPONENT,
    _CATEGORY_VARIABLE_DENOMINATOR,
)
_ALL_CATEGORIES = (_CATEGORY_POLYNOMIAL,) + _VIOLATION_CATEGORIES

# Clean padding terms for the hard-tier "not a polynomial" case (never a
# constant -- a plain digit term would never be confused with the violation
# term anyway, but keeping every padding term genuinely a variable term
# keeps the "one clear-cut reason" question unambiguous).
_PADDING_POOL = ((1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (1, 1))


def _var_text(ex: int, ey: int = 0) -> str:
    """One term's variable-text, pinned exactly: "" for a
    constant, "x"/"x^p" for a single-variable term (never "x^1"), x-part
    then y-part concatenated with no separator for a two-variable term."""
    if ex == 0 and ey == 0:
        return ""
    x_part = "" if ex == 0 else ("x" if ex == 1 else f"x^{ex}")
    y_part = "" if ey == 0 else ("y" if ey == 1 else f"y^{ey}")
    return x_part + y_part


def _nonzero_coef(rng, lo: int = -9, hi: int = 9) -> int:
    coef = 0
    while coef == 0:
        coef = rng.randint(lo, hi)
    return coef


def _check_terms_distinct(terms) -> None:
    """Defensive, cheap: no two terms may share variable-text, or join_terms
    would silently combine them as like terms and the declared term count
    would be wrong."""
    var_texts = [var for _coef, var, _ex, _ey in terms]
    check(len(var_texts) == len(set(var_texts)), "duplicate variable-text terms would combine as like terms")


def _draw_single_var_terms(rng, degree_lo: int, degree_hi: int):
    """A genuine single-variable polynomial: (coef, var, ex, ey) terms whose
    exponents are distinct, the true degree always included, 1-3 terms."""
    degree = rng.randint(degree_lo, degree_hi)
    max_terms = min(3, degree + 1)  # exponents 0..degree are the only pool
    term_count = rng.randint(1, max_terms)
    exponents = {degree}
    while len(exponents) < term_count:
        exponents.add(rng.randint(0, degree))
    terms = [(_nonzero_coef(rng), _var_text(e, 0), e, 0) for e in sorted(exponents, reverse=True)]
    _check_terms_distinct(terms)
    return terms, degree


def _draw_two_var_terms(rng, degree_lo: int, degree_hi: int, max_attempts: int = 300):
    """A genuine two-variable polynomial for the hard tier. Every term but
    one has total degree strictly below the drawn degree, so the
    maximum-degree term is always unique -- no tie for "the" highest-degree
    term, which the x-exponent-only distractor depends on."""
    for _ in range(max_attempts):
        degree = rng.randint(degree_lo, degree_hi)
        term_count = rng.randint(1, 3)
        ex_top = rng.randint(0, degree)
        pairs = {(ex_top, degree - ex_top)}
        inner_attempts = 0
        while len(pairs) < term_count and inner_attempts < 100:
            inner_attempts += 1
            d2 = rng.randint(0, degree - 1)
            ex2 = rng.randint(0, d2)
            pairs.add((ex2, d2 - ex2))
        if len(pairs) == term_count:
            # Total order, not hash order: highest total degree first, and a
            # deterministic tie-break on ex so the term ORDER (hence the
            # rendered text, hence determinism) never depends on set/hash
            # iteration order.
            ordered = sorted(pairs, key=lambda p: (-(p[0] + p[1]), -p[0]))
            terms = [(_nonzero_coef(rng), _var_text(ex, ey), ex, ey) for ex, ey in ordered]
            _check_terms_distinct(terms)
            return terms, degree
    raise RuntimeError(f"could not draw {term_count} distinct two-variable terms at degree {degree}")


def _mode_a_question(rng, tier: str):
    if tier == "hard":
        terms, degree = _draw_two_var_terms(rng, 1, 5)
    elif tier == "medium":
        terms, degree = _draw_single_var_terms(rng, 1, 5)
    else:
        terms, degree = _draw_single_var_terms(rng, 1, 3)

    term_count = len(terms)
    type_label = _TYPE_NAMES[term_count]
    expr = join_terms([(coef, var) for coef, var, _ex, _ey in terms])
    answer_text = f"Degree {degree}, {type_label}"

    top_ex, top_ey = terms[0][2], terms[0][3]
    x_only_offered = tier == "hard" and top_ey > 0

    # +/-1, whichever stays >=0 (degree is always >=1 so both do) -- drawn so
    # the correct option isn't predictably always the smaller of the two
    # same-type degree values on offer. When the x-exponent-only distractor
    # is offered it is always < degree, so the wrong-degree distractor is
    # pinned to +1 to keep the two from ever colliding.
    delta = 1 if x_only_offered else rng.choice((-1, 1))
    if degree + delta < 0:
        delta = 1
    wrong_degree_val = degree + delta

    adjacent_type = rng.choice(_ADJACENT_TYPES[term_count])
    wrong_degree_right_type = f"Degree {wrong_degree_val}, {type_label}"
    right_degree_wrong_type = f"Degree {degree}, {_TYPE_NAMES[adjacent_type]}"
    wrong_degree_wrong_type = f"Degree {wrong_degree_val}, {_TYPE_NAMES[adjacent_type]}"

    candidates = [wrong_degree_right_type, right_degree_wrong_type]
    if x_only_offered:
        x_only_degree = top_ex
        check_distinct(degree, x_only_degree, msg="x-exponent-only distractor must differ from the true degree")
        candidates.append(f"Degree {x_only_degree}, {type_label}")
    else:
        candidates.append(wrong_degree_wrong_type)

    text = f"Classify this polynomial by degree and number of terms: {expr}"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


def _polynomial_terms_for_tier(rng, tier: str):
    """A genuine polynomial's (coef, var) term list for Mode B's "yes"
    branch, reusing Mode A's own per-tier term machinery. Hard tier also
    exercises the constant-denominator trap (``x/2``, still a polynomial,
    the deliberate pair for the variable-denominator violation below) on a
    measured, non-trivial fraction of seeds."""
    if tier == "hard":
        terms, _degree = _draw_two_var_terms(rng, 1, 5)
        pairs = [(coef, var) for coef, var, _ex, _ey in terms]
        if rng.random() < 0.7:
            denom = rng.randint(2, 5)
            trap = (rng.choice((1, -1)), f"x/{denom}")
            if len(pairs) < 3:
                pairs = pairs + [trap]
            else:
                pairs = list(pairs)
                pairs[rng.randrange(len(pairs))] = trap
        return pairs
    if tier == "medium":
        terms, _degree = _draw_single_var_terms(rng, 1, 5)
    else:
        terms, _degree = _draw_single_var_terms(rng, 1, 3)
    return [(coef, var) for coef, var, _ex, _ey in terms]


def _violation_term(rng, category: str, tier: str) -> tuple[int, str]:
    """One term embodying exactly one of the three fixed violation reasons,
    displayed exactly as follows: a negative exponent with the caret
    directly against the minus (no space, so it never trips the
    ``signed_after_operator`` text-quality pattern, which requires a space
    between an operator and a following minus-digit); a parenthesised
    fraction after a caret; a variable in the denominator."""
    if category == _CATEGORY_NEGATIVE_EXPONENT:
        k = rng.randint(1, 2) if tier == "easy" else rng.randint(1, 4)
        return _nonzero_coef(rng), f"x^-{k}"
    if category == _CATEGORY_FRACTIONAL_EXPONENT:
        num, den = rng.choice(((1, 2), (1, 3), (2, 3), (3, 4), (1, 4)))
        return _nonzero_coef(rng), f"x^({num}/{den})"
    # variable denominator: a plain numerator/x[^p] string, not a
    # coefficient*variable product -- fed into join_terms as (coef, var)
    # like every other term so no join is hand-formatted, but the "variable
    # text" here is the whole fraction.
    num = rng.randint(2, 9)
    if tier != "easy" and rng.random() < 0.5:
        var = f"{num}/x^{rng.randint(2, 3)}"
    else:
        var = f"{num}/x"
    return rng.choice((1, -1)), var


def _mode_b_question(rng, tier: str):
    category = rng.choice(_ALL_CATEGORIES)
    if category == _CATEGORY_POLYNOMIAL:
        pairs = _polynomial_terms_for_tier(rng, tier)
    else:
        coef, var = _violation_term(rng, category, tier)
        pairs = [(coef, var)]
        if tier == "hard":
            padding_count = rng.randint(0, 2)
            padding = rng.sample(_PADDING_POOL, k=padding_count)
            pairs = pairs + [(_nonzero_coef(rng), _var_text(ex, ey)) for ex, ey in padding]
        rng.shuffle(pairs)

    check(len(pairs) == len({var for _coef, var in pairs}), "duplicate variable-text terms would combine as like terms")
    expr = join_terms(pairs)
    text = f"Is this a polynomial? If not, why not: {expr}"

    # Drop the mathematically-equivalent label from the candidate pool rather than relying on
    # `pick_distractors` (which only drops an identical VALUE, not an equivalent CONCEPT under a
    # different display form). "fractional exponent" names a genuinely different reason (a radical,
    # not an inverse power) and is never dropped.
    excluded = {category}
    if category == _CATEGORY_NEGATIVE_EXPONENT:
        excluded.add(_CATEGORY_VARIABLE_DENOMINATOR)
    elif category == _CATEGORY_VARIABLE_DENOMINATOR:
        excluded.add(_CATEGORY_NEGATIVE_EXPONENT)
    candidates = [c for c in _ALL_CATEGORIES if c not in excluded]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=category,
        answer_text=category,
        answer_latex=latex_answer(category, plain=True),
        distractors=pick_distractors(category, candidates, n=3),
    )


@generator(
    "polynomial_classify",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def polynomial_classify(rng, tier: str = "easy"):
    mode = rng.choice(("A", "B"))
    if mode == "A":
        return _mode_a_question(rng, tier)
    return _mode_b_question(rng, tier)
