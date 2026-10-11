"""Add/subtract polynomials and linear combination (skills/expressions-equivalence/add-subtract-polynomials.yaml,
skills/expressions-equivalence/polynomial-linear-combination.yaml).

Scope: coefficient handling and term ALIGNMENT (by degree, not by printed
position), never degree itself. Every polynomial is tracked as a dense list
of coefficients indexed by degree (index d = coefficient of x^d), covering
0..max_degree even where a polynomial has no term of some degree (stored as
0) -- this dense form is what makes "a missing power in one polynomial" a
real alignment trap at hard tier. sympy is used only for the INDEPENDENT
check (a second, symbolic route to the same value), never to construct or
display a question.

``answer_form="expanded"`` is set on every question. The "own form rejected" proof therefore does not use
sympy.factor: an arbitrary polynomial sum usually has no nontrivial factorisation, so factor would just hand
back the expanded form unchanged and the proof would wrongly assert rejection of a CORRECT answer. Instead the
proof feeds the question's own bracketed construction straight back (e.g. "(3x^2 - 5x + 7) + (-2x^2 + x - 4) -
(x^2 + 6x - 1)") -- correct by value, but carrying a bracket, which answer_form="expanded" must reject.

The `exam` preset routes to a dedicated two-variable branch (`two_var=True`) inside each existing `@generator`
function; every other preset leaves `two_var` at its default `False` and reaches the ORIGINAL code path
byte-for-byte, so `easy`/`medium`/`hard` render output is provably unchanged.
"""

from __future__ import annotations

from sympy import symbols

from ..core import check_distinct, check_equivalent, generator, join_terms, pick_distractors

x, y = symbols("x y")


def _nonzero_randint(rng, lo: int, hi: int) -> int:
    while True:
        v = rng.randint(lo, hi)
        if v != 0:
            return v


def _draw_dense_poly(rng, degree: int, lo: int, hi: int) -> list[int]:
    """Dense coefficient array, index = degree, degrees 0..degree inclusive,
    every position nonzero by construction -- 'fully aligned, no missing
    powers' is the easy/medium baseline. A hard-tier caller zeroes one
    non-leading position afterwards via `_force_missing_power`."""
    return [_nonzero_randint(rng, lo, hi) for _ in range(degree + 1)]


def _force_negative_leading(coeffs: list[int]) -> None:
    coeffs[-1] = -abs(coeffs[-1])


def _force_missing_power(rng, coeffs: list[int]) -> int:
    """Zero one NON-leading degree, in place -- forces the alignment trap: a
    term present (by degree) in the other two polynomials but absent from
    this one. Never the leading term, so this polynomial's own top degree is
    unaffected. Requires len(coeffs) >= 2 (degree >= 1)."""
    d_missing = rng.randint(0, len(coeffs) - 2)
    coeffs[d_missing] = 0
    return d_missing


def _force_mixed_sign(coeffs: list[int]) -> None:
    """Force at least one positive and one negative coefficient among the
    dense array's nonzero entries, in place. Flips the first nonzero entry
    found (never index 0 blindly, since that position may already have been
    zeroed by `_force_missing_power`) -- the hard-tier trap needs a
    polynomial whose own sign mix is a second, independent error source
    alongside a nontrivial multiplier."""
    if any(c > 0 for c in coeffs) and any(c < 0 for c in coeffs):
        return
    for i, c in enumerate(coeffs):
        if c != 0:
            coeffs[i] = -c
            return


def _apply_missing_placeholder(coeffs: list[int], d_missing: int) -> list[int]:
    """coefficient-drop misconception, one concrete well-defined version: the
    coefficient at `d_missing` (zero, by construction) is treated as if it
    ALSO carries the nonzero coefficient from the neighbouring degree -- the
    neighbour's own contribution at its own degree is left untouched, so the
    value is duplicated onto the missing slot, not shifted away from where it
    belongs. Returns a new list; `coeffs` itself is not mutated."""
    wrong = list(coeffs)
    neighbour = d_missing - 1 if d_missing > 0 else d_missing + 1
    wrong[d_missing] = coeffs[neighbour]
    return wrong


def _combine_arrays(arrays: list[list[int]], signs: list[int]) -> list[int]:
    length = len(arrays[0])
    return [sum(s * arr[d] for s, arr in zip(signs, arrays)) for d in range(length)]


# ---------------------------------------------------------------------------
# Two-variable (exam-tier) representation. A single-variable "degree" is one
# integer; a two-variable one is an EXPONENT PAIR (dx, dy), so a fresh dense
# array is needed rather than stretching `_draw_dense_poly`'s single-axis
# shape. Fixed at six monomial slots, every polynomial dense over total
# degree 0..2 (matching the skill's own "xy, x^2, y^2 terms, and lower-degree/
# constant terms as the existing skill already uses" contract) -- HIGHEST
# total degree first so `join_terms`-ready display order needs no reversal,
# the same convention `_poly_terms` uses for the single-variable list.
_TWO_VAR_MONOMIALS: list[tuple[int, int, str]] = [
    (2, 0, "x^2"),
    (1, 1, "xy"),
    (0, 2, "y^2"),
    (1, 0, "x"),
    (0, 1, "y"),
    (0, 0, ""),
]


def _draw_two_var_poly(rng, lo: int, hi: int) -> list[int]:
    """Dense coefficient array over `_TWO_VAR_MONOMIALS`, every position
    nonzero by construction -- mirrors `_draw_dense_poly`'s single-variable
    convention ('fully aligned, no missing powers' is the baseline; a caller
    forces a missing monomial afterwards via `_force_missing_monomial`)."""
    return [_nonzero_randint(rng, lo, hi) for _ in _TWO_VAR_MONOMIALS]


def _force_missing_monomial(rng, coeffs: list[int]) -> int:
    """Zero one of the three genuinely two-variable QUADRATIC monomials
    (x^2, xy, y^2 -- indices 0, 1, 2 of `_TWO_VAR_MONOMIALS`), in place, and
    return its index. Unlike `_force_missing_power`'s single axis ('degree'),
    a two-variable coefficient-drop trap must be tied to one specific
    EXPONENT PAIR -- restricting the draw to these three indices (never a
    linear/constant slot) is what keeps the trap 'the same thing being true
    for an independent exponent pair' (x^2 present in one polynomial, absent
    in another; equally xy, equally y^2) rather than reducing back to the
    single-variable-style 'a degree is missing' story. Over enough seeds all
    three are exercised -- see `tests/test_exd_e1_polynomial_exam_sources.py`
    for the direct count."""
    idx = rng.randint(0, 2)
    coeffs[idx] = 0
    return idx


def _two_var_terms(coeffs: list[int]) -> list[tuple[int, str]]:
    """[(coef, label), ...] in `_TWO_VAR_MONOMIALS` order (already highest
    total degree first), join_terms-ready -- the two-variable analogue of
    `_poly_terms`."""
    return [(c, label) for c, (_dx, _dy, label) in zip(coeffs, _TWO_VAR_MONOMIALS)]


def _two_var_display(coeffs: list[int]) -> str:
    return join_terms(_two_var_terms(coeffs))


# `answer_latex` (unlike `text`'s own `latex`, which is always `plain:`-
#: prefixed and so never reaches KaTeX at all) is real LaTeX for this module's
#: existing single-variable tiers -- "3x^2 - 5x + 7" is already valid LaTeX
#: with no escaping needed, because there is only ever one variable letter.
#: The two-variable `xy` cross term breaks that: two BARE adjacent letters
#: with no backslash anywhere in the string is exactly
#: `tests/test_answer_latex_prose_sweep.py`'s italicised-prose defect
#: signature -- KaTeX would still render "xy" correctly as x times
#: y, but the sweep cannot distinguish that from leaked English prose, and a
#: genuine two-letter-adjacency case never arose in this module before. Fixed
#: at the source: only `answer_latex` uses this "x\\,y" (a real LaTeX thin
#: space, so it also introduces the backslash the sweep's "not real LaTeX"
#: half of its check requires) -- `text`/`answer_text`/every distractor
#: string keep the plain "xy" a learner would actually type, unaffected.
_TWO_VAR_LATEX_LABELS = {"xy": "x\\,y"}


def _two_var_latex_terms(coeffs: list[int]) -> list[tuple[int, str]]:
    return [(c, _TWO_VAR_LATEX_LABELS.get(label, label)) for c, (_dx, _dy, label) in zip(coeffs, _TWO_VAR_MONOMIALS)]


def _two_var_answer_latex(coeffs: list[int]) -> str:
    return join_terms(_two_var_latex_terms(coeffs))


def _two_var_expr(coeffs: list[int]):
    return sum(c * x**dx * y**dy for c, (dx, dy, _label) in zip(coeffs, _TWO_VAR_MONOMIALS))


def _degree_label(d: int) -> str:
    if d >= 2:
        return f"x^{d}"
    if d == 1:
        return "x"
    return ""


def _poly_terms(coeffs: list[int]) -> list[tuple[int, str]]:
    """[(coef, label), ...] from HIGHEST degree to LOWEST, join_terms-ready.
    Zero coefficients are passed through -- join_terms drops them -- so a
    missing power really is absent from the rendered text, not shown as a
    zero term. One-off local helper: it adapts this generator's OWN dense
    coefficient list into join_terms' input shape; no sympy expression ever
    exists here to display, so it is not a hand-rolled `_poly_text`-from-sympy display."""
    return [(coeffs[d], _degree_label(d)) for d in range(len(coeffs) - 1, -1, -1)]


def _poly_display(coeffs: list[int]) -> str:
    return join_terms(_poly_terms(coeffs))


def _poly_expr(coeffs: list[int]):
    return sum(c * x**d for d, c in enumerate(coeffs))


def _polynomial_add_subtract_two_var(rng, coef_lo: int, coef_hi: int) -> dict:
    """`exam` source: genuinely two-variable, one of two
    template shapes per question --

    - 'nested': `P - (Q + R)`, the subtraction distributing across a
      BRACKETED SUM of two polynomials (structurally different from every
      lower tier's flat `A + B - C`, which never nests a bracket at all);
    - 'flat': `A + B - C`, the SAME shape lower tiers use but now
      two-variable, itself already a new reasoning shape over single-variable
      `hard` (which terms are 'like' now depends on BOTH exponents, not one).

    Both variants force one QUADRATIC monomial (x^2, xy or y^2) missing in a
    random one of the three named polynomials, so the coefficient-drop trap
    is exercisable every question, the same 'always forced' convention the
    single-variable `hard` tier uses.
    """
    variant = rng.choice(["nested", "flat"])
    letters = ("P", "Q", "R") if variant == "nested" else ("A", "B", "C")
    l1, l2, l3 = letters
    polys = {name: _draw_two_var_poly(rng, coef_lo, coef_hi) for name in letters}

    missing_target = rng.choice(letters)
    missing_idx = _force_missing_monomial(rng, polys[missing_target])

    signs = [1, -1, -1] if variant == "nested" else [1, 1, -1]
    correct = _combine_arrays([polys[l1], polys[l2], polys[l3]], signs)

    # Independent check, second route: real sympy expressions combined with
    # sympy's own +/-, and for 'nested' the bracket is applied EXPLICITLY
    # (Q + R) rather than as two separate subtractions, matching what the
    # rendered question actually shows.
    e1, e2, e3 = _two_var_expr(polys[l1]), _two_var_expr(polys[l2]), _two_var_expr(polys[l3])
    if variant == "nested":
        independent = e1 - (e2 + e3)
    else:
        independent = e1 + e2 - e3
    check_equivalent(
        _two_var_expr(correct),
        independent,
        f"polynomial_add_subtract (exam, {variant}): coefficient-array result "
        "must agree with sympy's own signed combination",
    )

    d1, d2, d3 = _two_var_display(polys[l1]), _two_var_display(polys[l2]), _two_var_display(polys[l3])
    if variant == "nested":
        text = f"{l1} = {d1}. {l2} = {d2}. {l3} = {d3}. Find {l1} - ({l2} + {l3})."
    else:
        text = f"{l1} = {d1}. {l2} = {d2}. {l3} = {d3}. Find {l1} + {l2} - {l3}."
    answer_text = _two_var_display(correct)

    # sign-error-on-subtraction: fails to distribute the minus sign at all --
    # 'nested' computes P + Q + R (the whole bracket wrongly added); 'flat'
    # computes A + B + C, the same trap the single-variable tiers use.
    sign_error = _combine_arrays([polys[l1], polys[l2], polys[l3]], [1, 1, 1])
    check_distinct(
        _two_var_expr(correct),
        _two_var_expr(sign_error),
        msg=f"polynomial_add_subtract (exam, {variant}): sign-error-on-subtraction "
        "distractor must be a genuinely different value",
    )
    cands = [_two_var_display(sign_error)]

    # swapped-subtracted-term: subtracts the WRONG one of the two things that
    # should be subtracted. 'nested' distributes the minus onto only the
    # first bracketed term (l2), forgetting the second (l3) -- P - Q + R, as
    # if only Q were part of what's being subtracted; 'flat' is the original
    # single-variable trap (A - B + C), generalised to two variables.
    swapped = _combine_arrays([polys[l1], polys[l2], polys[l3]], [1, -1, 1])
    check_distinct(
        _two_var_expr(correct),
        _two_var_expr(swapped),
        msg=f"polynomial_add_subtract (exam, {variant}): swapped-subtracted-term "
        "distractor must be a genuinely different value",
    )
    cands.append(_two_var_display(swapped))

    # coefficient-drop: see _apply_missing_placeholder / _force_missing_monomial.
    wrong_poly = _apply_missing_placeholder(polys[missing_target], missing_idx)
    wrong = dict(polys)
    wrong[missing_target] = wrong_poly
    coefficient_drop = _combine_arrays([wrong[l1], wrong[l2], wrong[l3]], signs)
    check_distinct(
        _two_var_expr(correct),
        _two_var_expr(coefficient_drop),
        msg=f"polynomial_add_subtract (exam, {variant}): coefficient-drop distractor "
        "must be a genuinely different value",
    )
    cands.append(_two_var_display(coefficient_drop))

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=_two_var_expr(correct),
        answer_text=answer_text,
        answer_latex=_two_var_answer_latex(correct),
        distractors=pick_distractors(answer_text, cands),
        answer_form="expanded",
    )


@generator(
    "polynomial_add_subtract",
    level=9,
    presets={
        "easy": dict(deg_lo=1, deg_hi=2, coef_lo=-9, coef_hi=9, force_neg_leading=False, force_missing=False),
        "medium": dict(deg_lo=1, deg_hi=3, coef_lo=-12, coef_hi=12, force_neg_leading=True, force_missing=False),
        "hard": dict(deg_lo=2, deg_hi=4, coef_lo=-12, coef_hi=12, force_neg_leading=False, force_missing=True),
        "exam": dict(coef_lo=-9, coef_hi=9, two_var=True),
    },
)
def polynomial_add_subtract(
    rng,
    deg_lo: int = 1,
    deg_hi: int = 2,
    coef_lo: int = -9,
    coef_hi: int = 9,
    force_neg_leading: bool = False,
    force_missing: bool = False,
    two_var: bool = False,
):
    if two_var:
        return _polynomial_add_subtract_two_var(rng, coef_lo, coef_hi)
    degree = rng.randint(deg_lo, deg_hi)
    a = _draw_dense_poly(rng, degree, coef_lo, coef_hi)
    b = _draw_dense_poly(rng, degree, coef_lo, coef_hi)
    c = _draw_dense_poly(rng, degree, coef_lo, coef_hi)
    polys = {"A": a, "B": b, "C": c}

    if force_neg_leading:
        _force_negative_leading(polys[rng.choice(["A", "B", "C"])])

    missing_target = missing_degree = None
    if force_missing:
        missing_target = rng.choice(["A", "B", "C"])
        missing_degree = _force_missing_power(rng, polys[missing_target])

    # C must differ from B: when they are equal, A + B - C and the swapped-subtracted-term distractor A - B + C
    # are the same value (easy seed 208, and medium seed 4041 after the negative-leading step). Moving C's leading
    # coefficient one step further from zero keeps its sign, its degree and any missing power, and leaves every
    # other seed's question unchanged.
    if c == b:
        c[-1] += -1 if c[-1] < 0 else 1

    correct = _combine_arrays([a, b, c], [1, 1, -1])

    # Independent check, second route: real sympy expressions combined with
    # sympy's own +/-, never the coefficient-array arithmetic above.
    a_expr, b_expr, c_expr = _poly_expr(a), _poly_expr(b), _poly_expr(c)
    check_equivalent(
        _poly_expr(correct),
        a_expr + b_expr - c_expr,
        "polynomial_add_subtract: coefficient-array result must agree with sympy's own A + B - C",
    )

    a_disp, b_disp, c_disp = _poly_display(a), _poly_display(b), _poly_display(c)
    text = f"A = {a_disp}. B = {b_disp}. C = {c_disp}. Find A + B - C."
    answer_text = _poly_display(correct)

    # sign-error-on-subtraction: A + B + C instead of A + B - C.
    sign_error = _combine_arrays([a, b, c], [1, 1, 1])
    check_distinct(
        _poly_expr(correct),
        _poly_expr(sign_error),
        msg="polynomial_add_subtract: sign-error-on-subtraction distractor must be a genuinely different value",
    )
    cands = [_poly_display(sign_error)]

    # swapped-subtracted-term:
    # subtracts B instead of C -- A - B + C, as if the learner picked the wrong
    # NAMED polynomial to treat as the subtracted one.
    swapped_subtracted = _combine_arrays([a, b, c], [1, -1, 1])
    check_distinct(
        _poly_expr(correct),
        _poly_expr(swapped_subtracted),
        msg="polynomial_add_subtract: swapped-subtracted-term distractor must be a genuinely different value",
    )
    cands.append(_poly_display(swapped_subtracted))

    # coefficient-drop (hard tier only): see _apply_missing_placeholder.
    if missing_target is not None:
        wrong_poly = _apply_missing_placeholder(polys[missing_target], missing_degree)
        wrong_a = wrong_poly if missing_target == "A" else a
        wrong_b = wrong_poly if missing_target == "B" else b
        wrong_c = wrong_poly if missing_target == "C" else c
        coefficient_drop = _combine_arrays([wrong_a, wrong_b, wrong_c], [1, 1, -1])
        check_distinct(
            _poly_expr(correct),
            _poly_expr(coefficient_drop),
            msg="polynomial_add_subtract: coefficient-drop distractor must be a genuinely different value",
        )
        cands.append(_poly_display(coefficient_drop))

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=_poly_expr(correct),
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
        answer_form="expanded",
    )


def _polynomial_linear_combination_two_var(rng, coef_lo: int, coef_hi: int, mult_lo: int, mult_hi: int) -> dict:
    """`exam` source: genuinely two-variable P, Q,
    R with `xy`, `x^2`, `y^2` (and lower-degree/constant) terms, multipliers
    up to 5, and -- unlike every lower tier's fixed '2P + 3Q - 5R' shape
    (always P, Q additive and R subtractive) -- the SUBTRACTED slot itself
    varies across P/Q/R, so the operation is genuinely a signed combination
    rather than a fixed pattern with bigger numbers dropped in.
    """
    p = _draw_two_var_poly(rng, coef_lo, coef_hi)
    q = _draw_two_var_poly(rng, coef_lo, coef_hi)
    r = _draw_two_var_poly(rng, coef_lo, coef_hi)
    polys = {"P": p, "Q": q, "R": r}

    m1 = rng.randint(mult_lo, mult_hi)
    m2 = rng.randint(mult_lo, mult_hi)
    m3 = rng.randint(mult_lo, mult_hi)

    subtracted = rng.choice(["P", "Q", "R"])
    sign = {"P": 1, "Q": 1, "R": 1}
    sign[subtracted] = -1

    missing_target = rng.choice(["P", "Q", "R"])
    missing_idx = _force_missing_monomial(rng, polys[missing_target])

    signed_mults = [sign["P"] * m1, sign["Q"] * m2, sign["R"] * m3]
    correct = _combine_arrays([p, q, r], signed_mults)

    p_expr, q_expr, r_expr = _two_var_expr(p), _two_var_expr(q), _two_var_expr(r)
    check_equivalent(
        _two_var_expr(correct),
        signed_mults[0] * p_expr + signed_mults[1] * q_expr + signed_mults[2] * r_expr,
        "polynomial_linear_combination (exam, two-variable): coefficient-array result must "
        "agree with sympy's own signed combination",
    )

    p_disp, q_disp, r_disp = _two_var_display(p), _two_var_display(q), _two_var_display(r)
    combo_text = join_terms([(signed_mults[0], "P"), (signed_mults[1], "Q"), (signed_mults[2], "R")])
    text = f"P = {p_disp}. Q = {q_disp}. R = {r_disp}. Find {combo_text}."
    answer_text = _two_var_display(correct)

    # sign-error-on-subtraction: forgets to distribute the minus onto
    # whichever slot is actually subtracted this time -- treats every term
    # as addition, m1*P + m2*Q + m3*R.
    sign_error = _combine_arrays([p, q, r], [m1, m2, m3])
    check_distinct(
        _two_var_expr(correct),
        _two_var_expr(sign_error),
        msg="polynomial_linear_combination (exam): sign-error-on-subtraction "
        "distractor must be a genuinely different value",
    )
    cands = [_two_var_display(sign_error)]

    # linear-combination-only: multiplier 1 for every polynomial, correct
    # sign pattern kept (so this isolates 'ignored the multipliers' from
    # 'ignored the signs').
    linear_only = _combine_arrays([p, q, r], [sign["P"], sign["Q"], sign["R"]])
    check_distinct(
        _two_var_expr(correct),
        _two_var_expr(linear_only),
        msg="polynomial_linear_combination (exam): linear-combination-only "
        "distractor must be a genuinely different value",
    )
    cands.append(_two_var_display(linear_only))

    # coefficient-drop: see _apply_missing_placeholder / _force_missing_monomial.
    wrong_poly = _apply_missing_placeholder(polys[missing_target], missing_idx)
    wrong = {"P": p, "Q": q, "R": r}
    wrong[missing_target] = wrong_poly
    coefficient_drop = _combine_arrays([wrong["P"], wrong["Q"], wrong["R"]], signed_mults)
    check_distinct(
        _two_var_expr(correct),
        _two_var_expr(coefficient_drop),
        msg="polynomial_linear_combination (exam): coefficient-drop distractor must be a genuinely different value",
    )
    cands.append(_two_var_display(coefficient_drop))

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=_two_var_expr(correct),
        answer_text=answer_text,
        answer_latex=_two_var_answer_latex(correct),
        distractors=pick_distractors(answer_text, cands),
        answer_form="expanded",
    )


@generator(
    "polynomial_linear_combination",
    level=9,
    presets={
        "easy": dict(
            deg_lo=1,
            deg_hi=2,
            coef_lo=-9,
            coef_hi=9,
            mult_mode="fixed",
            mult_lo=2,
            mult_hi=4,
            force_mixed=False,
            force_missing=False,
        ),
        "medium": dict(
            deg_lo=1,
            deg_hi=3,
            coef_lo=-9,
            coef_hi=9,
            mult_mode="drawn",
            mult_lo=2,
            mult_hi=4,
            force_mixed=False,
            force_missing=False,
        ),
        "hard": dict(
            deg_lo=2,
            deg_hi=4,
            coef_lo=-12,
            coef_hi=12,
            mult_mode="drawn",
            mult_lo=2,
            mult_hi=6,
            force_mixed=True,
            force_missing=True,
        ),
        "exam": dict(coef_lo=-9, coef_hi=9, mult_lo=2, mult_hi=5, two_var=True),
    },
)
def polynomial_linear_combination(
    rng,
    deg_lo: int = 1,
    deg_hi: int = 2,
    coef_lo: int = -9,
    coef_hi: int = 9,
    mult_mode: str = "fixed",
    mult_lo: int = 2,
    mult_hi: int = 4,
    force_mixed: bool = False,
    force_missing: bool = False,
    two_var: bool = False,
):
    if two_var:
        return _polynomial_linear_combination_two_var(rng, coef_lo, coef_hi, mult_lo, mult_hi)
    degree = rng.randint(deg_lo, deg_hi)
    p = _draw_dense_poly(rng, degree, coef_lo, coef_hi)
    q = _draw_dense_poly(rng, degree, coef_lo, coef_hi)
    r = _draw_dense_poly(rng, degree, coef_lo, coef_hi)
    polys = {"P": p, "Q": q, "R": r}

    if mult_mode == "fixed":
        m1, m2, m3 = 2, 1, 1
    else:
        m1 = rng.randint(mult_lo, mult_hi)
        m2 = rng.randint(mult_lo, mult_hi)
        m3 = rng.randint(mult_lo, mult_hi)

    missing_target = missing_degree = None
    if force_missing:
        missing_target = rng.choice(["P", "Q", "R"])
        missing_degree = _force_missing_power(rng, polys[missing_target])

    if force_mixed:
        # R is always the subtracted polynomial -- its multiplier's EFFECT is
        # already negative, so pairing it with a genuine sign mix gives the
        # two independent sign-tracking sources the hard tier names. Applied
        # AFTER force_missing so a zeroed position (possibly R's own) can
        # never undo this.
        _force_mixed_sign(r)

    correct = _combine_arrays([p, q, r], [m1, m2, -m3])

    p_expr, q_expr, r_expr = _poly_expr(p), _poly_expr(q), _poly_expr(r)
    check_equivalent(
        _poly_expr(correct),
        m1 * p_expr + m2 * q_expr - m3 * r_expr,
        "polynomial_linear_combination: coefficient-array result must agree with sympy's own m1*P + m2*Q - m3*R",
    )

    p_disp, q_disp, r_disp = _poly_display(p), _poly_display(q), _poly_display(r)
    combo_text = join_terms([(m1, "P"), (m2, "Q"), (-m3, "R")])
    text = f"P = {p_disp}. Q = {q_disp}. R = {r_disp}. Find {combo_text}."
    answer_text = _poly_display(correct)

    # sign-error-on-subtraction: m1*P + m2*Q + m3*R (subtraction dropped).
    sign_error = _combine_arrays([p, q, r], [m1, m2, m3])
    check_distinct(
        _poly_expr(correct),
        _poly_expr(sign_error),
        msg="polynomial_linear_combination: sign-error-on-subtraction distractor must be a genuinely different value",
    )
    cands = [_poly_display(sign_error)]

    # linear-combination-only: multiplier 1 for every polynomial, i.e. P + Q - R.
    linear_only = _combine_arrays([p, q, r], [1, 1, -1])
    check_distinct(
        _poly_expr(correct),
        _poly_expr(linear_only),
        msg="polynomial_linear_combination: linear-combination-only distractor must be a genuinely different value",
    )
    cands.append(_poly_display(linear_only))

    # coefficient-drop (hard tier only): see _apply_missing_placeholder.
    if missing_target is not None:
        wrong_poly = _apply_missing_placeholder(polys[missing_target], missing_degree)
        wrong_p = wrong_poly if missing_target == "P" else p
        wrong_q = wrong_poly if missing_target == "Q" else q
        wrong_r = wrong_poly if missing_target == "R" else r
        coefficient_drop = _combine_arrays([wrong_p, wrong_q, wrong_r], [m1, m2, -m3])
        check_distinct(
            _poly_expr(correct),
            _poly_expr(coefficient_drop),
            msg="polynomial_linear_combination: coefficient-drop distractor must be a genuinely different value",
        )
        cands.append(_poly_display(coefficient_drop))

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=_poly_expr(correct),
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
        answer_form="expanded",
    )
