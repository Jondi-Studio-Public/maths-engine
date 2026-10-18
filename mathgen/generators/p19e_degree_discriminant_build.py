"""Generates `p19e_poly_degree_leading` and `p19e_discriminant_zero_count` questions: a polynomial's degree and
leading coefficient, and the number of zeros from the discriminant.

Two generators, both senior ``acara-senior-v8-4``, ``response_family:
ordered_structures``, ``answer: pair``:

- ``p19e_poly_degree_leading``
  (``senior-identify-the-coefficients-and-the-degree-of-a``, ACMMM015) --
  given a stated polynomial in one variable, state its degree and the
  coefficient of its highest-degree (leading) term. The skill narrows
  "the coefficients and the degree" to the single leading coefficient paired
  with the degree (the pair that together describes end behaviour); the full
  coefficient list is dropped and never built.

- ``p19e_discriminant_zero_count``
  (``senior-find-turning-points-and-zeros-of-quadratics-and``, ACMMM010) --
  given a quadratic ``ax^2 + bx + c`` with stated integer a, b, c, compute the
  discriminant ``b^2 - 4ac`` and state the number of real zeros (0, 1 or 2)
  it implies. Never asks for the turning point or the zeros themselves --
  that is `quadratic-vertex` / `solve-quadratics-formula` /
  `factorise-quadratic`'s own content.

Both answers are LABELLED pairs (`mathgen/grading/typed.py::grade_pair`,
`_labelled_pair_verdict`), graded order-independently by label name; every
prompt states the label shape explicitly ("Give your answer as ...").
"""

from __future__ import annotations

from sympy import Integer, Poly, Symbol, discriminant as sympy_discriminant, expand

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    fmt_signed,
    generator,
    join_terms,
    paren_negative,
    phrase,
    pick_distractors,
)

X = Symbol("x")


# ============================================================================
# Shared: labelled-pair finalisation
# ============================================================================


def _latex_label(label: str) -> str:
    """A single-letter label ("D") is already valid maths; a multi-letter
    word ("degree", "coefficient", "zeros") must be wrapped in `\\text{}` or
    KaTeX renders it as italicised one-letter-variables-multiplied-together
    prose."""
    return label if len(label) == 1 else rf"\text{{{label}}}"


def _finalise_pair(text, first, second, extra_candidates, working, label_a, label_b, n_distractors=3):
    """Build the dict for a labelled `answer: pair` question: `answer_text`
    is "`{label_a} = {first}, {label_b} = {second}`", graded order-
    independently by label (`mathgen/grading/typed.py::_labelled_pair_verdict`).
    `answer_latex` states the same pair with each word label wrapped in
    `\\text{}` (prose in maths mode needs `\\text{}`), never plain text.

    `extra_candidates` is a list of `(value_a, value_b)` tuples, each a
    genuinely different misconception pair. Duplicates of the answer (or of
    each other) are dropped before formatting; `check_pairwise_distinct` is
    then run over the answer and every KEPT distractor, unconditionally.
    """
    answer_tuple = (first, second)
    answer_text = f"{label_a} = {first}, {label_b} = {second}"
    answer_latex = rf"{_latex_label(label_a)} = {first},\; {_latex_label(label_b)} = {second}"
    seen = {answer_tuple}
    candidate_tuples = []
    for cand in extra_candidates:
        if cand in seen:
            continue
        seen.add(cand)
        candidate_tuples.append(cand)
    formatted = [f"{label_a} = {a}, {label_b} = {b}" for a, b in candidate_tuples]
    distractors = pick_distractors(answer_text, formatted, n=n_distractors)
    kept_tuples = [candidate_tuples[formatted.index(d)] for d in distractors]
    check_pairwise_distinct(
        answer_tuple,
        *kept_tuples,
        msg="p19e: a distractor pair collides with the answer or another distractor",
    )
    check(len(distractors) >= 1, "p19e: no distractor survived construction")
    return dict(
        text=text,
        latex=text,
        answer={label_a: first, label_b: second},
        answer_text=answer_text,
        answer_latex=answer_latex,
        working=working,
        distractors=distractors,
    )


# ============================================================================
# p19e_poly_degree_leading (ACMMM015)
# ============================================================================


def _term_var(k: int) -> str:
    if k == 0:
        return ""
    if k == 1:
        return "x"
    return f"x^{k}"


def _draw_nonzero(rng, lo=-9, hi=9):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _verify_additive_poly(coeffs: dict[int, int], degree: int, lc: int) -> None:
    """Independent route: build the sympy expression from the SAME coeff
    dict and read degree/leading coefficient off sympy's own `Poly` -- a
    genuinely different piece of code from this module's `max()`-based
    read, proven to agree by `Poly.expand`'s own term-collection logic."""
    expr = sum(Integer(c) * X**k for k, c in coeffs.items())
    p = Poly(expand(expr), X)
    check_equivalent(p.degree(), degree, "p19e_poly_degree_leading: degree disagrees with sympy Poly")
    check_equivalent(p.LC(), lc, "p19e_poly_degree_leading: leading coefficient disagrees with sympy Poly")


def _generic_distractors(degree, lc, first_deg, first_coef, count_as_degree):
    return [
        (lc, degree),  # swapped (LC, degree)
        (first_deg, first_coef),  # the first-WRITTEN term's own power/coefficient
        (count_as_degree, lc),  # mistakes a count of terms/factors for the degree
        (degree, -lc),  # sign error on the leading coefficient only
    ]


_ANSWER_FORMAT_D = " Give your answer as degree = ..., coefficient = ..."

_EASY_TEMPLATES = (
    "Consider the polynomial P(x) = {poly}. State its degree and the coefficient of its leading (highest-degree) term.",
    "For the polynomial {poly}, give the degree and the leading coefficient.",
    "The polynomial {poly} is written in one variable. What is its degree, and what is the "
    "coefficient of its highest-degree term?",
)
_MEDIUM_TEMPLATES = (
    "The terms of a polynomial are given, not necessarily in order of degree: {poly}. What "
    "is its degree, and what is the coefficient of its highest-degree term?",
    "Consider P(x) = {poly}, with its terms listed in no particular order. State the degree "
    "and the leading coefficient.",
    "For the polynomial {poly} (terms not written in order of degree), give the degree and the leading coefficient.",
)
_HARD_TEMPLATES = (
    "Without expanding, state the degree and the leading coefficient of the polynomial {poly}.",
    "The polynomial {poly} is given as a product of factors. Without multiplying it out, "
    "find its degree and its leading coefficient.",
    "Consider P(x) = {poly}, written as a product of factors. State the degree and the "
    "leading coefficient of P(x) without expanding it.",
)
_EXAM_TEMPLATES = (
    "Consider the polynomial P(x) = {poly}. Some like terms have not yet been combined. "
    "State the degree and the leading coefficient of P(x) once like terms are combined.",
    "The terms of a polynomial are given, with some like terms still to be combined: {poly}. "
    "Combine like terms first, then state the polynomial's degree and its leading coefficient.",
    "For P(x) = {poly}, first combine any like terms. What is the resulting degree, and what "
    "is the coefficient of the resulting leading term?",
)


def _make_easy(rng):
    d = rng.randint(3, 5)
    lc = rng.choice([c for c in range(-6, 7) if abs(c) >= 2])
    coeffs = {d: lc}
    for k in range(d - 1, -1, -1):
        if rng.random() < 0.75:
            coeffs[k] = _draw_nonzero(rng)
    if len(coeffs) < 2:
        k = rng.randrange(0, d)
        coeffs[k] = _draw_nonzero(rng)
    degree = max(coeffs)
    lc_final = coeffs[degree]
    _verify_additive_poly(coeffs, degree, lc_final)

    order = sorted(coeffs, reverse=True)
    terms = [(coeffs[k], _term_var(k)) for k in order]
    poly_text = join_terms(terms)
    text = phrase(rng, _EASY_TEMPLATES, poly=poly_text) + _ANSWER_FORMAT_D

    first_deg, first_coef = order[0], coeffs[order[0]]
    cands = _generic_distractors(degree, lc_final, first_deg, first_coef, len(coeffs))
    working = [
        f"Highest power present is x^{degree}, with coefficient {lc_final}.",
        f"degree = {degree}, coefficient = {lc_final}.",
    ]
    return _finalise_pair(text, degree, lc_final, cands, working, "degree", "coefficient")


def _make_medium(rng):
    d = rng.randint(3, 5)
    lc = rng.choice([-1, 1]) if rng.random() < 0.5 else rng.choice([c for c in range(-6, 7) if abs(c) >= 2])
    coeffs = {d: lc}
    for k in range(d - 1, -1, -1):
        if rng.random() < 0.75:
            coeffs[k] = _draw_nonzero(rng)
    if len(coeffs) < 2:
        k = rng.randrange(0, d)
        coeffs[k] = _draw_nonzero(rng)
    degree = max(coeffs)
    lc_final = coeffs[degree]
    _verify_additive_poly(coeffs, degree, lc_final)

    order = sorted(coeffs, reverse=True)
    shuffled = list(order)
    tries = 0
    while True:
        rng.shuffle(shuffled)
        tries += 1
        if shuffled[0] != degree or tries >= 20:
            break
    if shuffled[0] == degree and len(shuffled) > 1:
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]

    terms = [(coeffs[k], _term_var(k)) for k in shuffled]
    poly_text = join_terms(terms)
    text = phrase(rng, _MEDIUM_TEMPLATES, poly=poly_text) + _ANSWER_FORMAT_D

    first_deg, first_coef = shuffled[0], coeffs[shuffled[0]]
    cands = _generic_distractors(degree, lc_final, first_deg, first_coef, len(coeffs))
    working = [
        "Terms are not written in order of degree.",
        f"Highest power present is x^{degree}, with coefficient {lc_final}.",
        f"degree = {degree}, coefficient = {lc_final}.",
    ]
    return _finalise_pair(text, degree, lc_final, cands, working, "degree", "coefficient")


def _linear_factor(rng, *, allow_zero_const=False):
    c = rng.choice([-3, -2, -1, 1, 2, 3])
    if allow_zero_const and rng.random() < 0.4:
        d = 0
    else:
        d = _draw_nonzero(rng, -8, 8)
    if d == 0:
        display = fmt_signed(c, "x")
    else:
        display = f"({fmt_signed(c, 'x')} {'+' if d > 0 else '-'} {abs(d)})"
    return dict(degree=1, lc=c, display=display, expr=Integer(c) * X + Integer(d))


def _quad_factor(rng):
    c = rng.choice([-3, -2, -1, 1, 2, 3])
    d = _draw_nonzero(rng, -8, 8)
    display = f"({fmt_signed(c, 'x^2')} {'+' if d > 0 else '-'} {abs(d)})"
    return dict(degree=2, lc=c, display=display, expr=Integer(c) * X**2 + Integer(d))


def _squared_factor(rng):
    c = rng.choice([-3, -2, -1, 1, 2, 3])
    d = _draw_nonzero(rng, -8, 8)
    inner_display = f"({fmt_signed(c, 'x')} {'+' if d > 0 else '-'} {abs(d)})"
    return dict(degree=2, lc=c * c, display=inner_display + "^2", expr=(Integer(c) * X + Integer(d)) ** 2)


def _build_hard_poly(rng):
    variant = rng.choice(["linear_quad", "linear_squared", "three_linear"])
    if variant == "linear_quad":
        factors = [_linear_factor(rng), _quad_factor(rng)]
    elif variant == "linear_squared":
        factors = [_linear_factor(rng, allow_zero_const=True), _squared_factor(rng)]
    else:
        factors = [_linear_factor(rng) for _ in range(3)]

    degree = sum(f["degree"] for f in factors)
    lc = 1
    for f in factors:
        lc *= f["lc"]
    poly_text = "".join(f["display"] for f in factors)

    expr = Integer(1)
    for f in factors:
        expr = expr * f["expr"]
    p = Poly(expand(expr), X)
    check_equivalent(p.degree(), degree, "p19e_poly_degree_leading (hard): degree disagrees with sympy expansion")
    check_equivalent(p.LC(), lc, "p19e_poly_degree_leading (hard): leading coefficient disagrees with sympy expansion")

    first = factors[0]
    max_factor_degree = max(f["degree"] for f in factors)
    sum_lcs = sum(f["lc"] for f in factors)
    return poly_text, degree, lc, first["degree"], first["lc"], len(factors), max_factor_degree, sum_lcs


def _make_hard(rng):
    (poly_text, degree, lc, first_deg, first_coef, num_factors, max_factor_degree, sum_lcs) = _build_hard_poly(rng)
    text = phrase(rng, _HARD_TEMPLATES, poly=poly_text) + _ANSWER_FORMAT_D
    cands = _generic_distractors(degree, lc, first_deg, first_coef, num_factors)
    cands.append((max_factor_degree, sum_lcs))
    working = [
        f"degree = sum of factor degrees = {degree}.",
        f"leading coefficient = product of factor leading coefficients = {lc}.",
    ]
    return _finalise_pair(text, degree, lc, cands, working, "degree", "coefficient", n_distractors=4)


def _build_exam_poly(rng):
    apparent_deg = rng.randint(3, 5)
    apparent_coef = _draw_nonzero(rng)
    true_deg = rng.randint(1, apparent_deg - 1)
    true_coef = _draw_nonzero(rng)

    extra_terms = [(apparent_deg, apparent_coef), (true_deg, true_coef), (apparent_deg, -apparent_coef)]
    lower_degs = list(range(0, true_deg))
    rng.shuffle(lower_degs)
    n_extra = rng.randint(0, min(2, len(lower_degs))) if lower_degs else 0
    for k in lower_degs[:n_extra]:
        extra_terms.append((k, _draw_nonzero(rng)))
    rng.shuffle(extra_terms)

    # Keep the two cancelling apparent-degree terms apart so the question
    # reads as a genuine "combine like terms" task, not a visible adjacent pair.
    tries = 0
    while True:
        idxs = [i for i, (k, _) in enumerate(extra_terms) if k == apparent_deg]
        if len(idxs) < 2 or abs(idxs[0] - idxs[1]) > 1 or tries >= 15:
            break
        rng.shuffle(extra_terms)
        tries += 1

    raw: dict[int, int] = {}
    for k, c in extra_terms:
        raw[k] = raw.get(k, 0) + c
    coeffs = {k: v for k, v in raw.items() if v != 0}
    check(bool(coeffs), "p19e_poly_degree_leading (exam): cancellation left no terms")
    degree = max(coeffs)
    lc = coeffs[degree]
    check(degree == true_deg, "p19e_poly_degree_leading (exam): apparent-degree term did not fully cancel")
    check(lc == true_coef, "p19e_poly_degree_leading (exam): true leading coefficient mismatch")

    expr = sum(Integer(c) * X**k for k, c in extra_terms)
    p = Poly(expand(expr), X)
    check_equivalent(p.degree(), degree, "p19e_poly_degree_leading (exam): degree disagrees with sympy expansion")
    check_equivalent(p.LC(), lc, "p19e_poly_degree_leading (exam): leading coefficient disagrees with sympy expansion")

    poly_text = join_terms([(c, _term_var(k)) for k, c in extra_terms])
    first_deg, first_coef = extra_terms[0]
    return poly_text, degree, lc, apparent_deg, apparent_coef, first_deg, first_coef, len(extra_terms)


def _make_exam(rng):
    (poly_text, degree, lc, apparent_deg, apparent_coef, first_deg, first_coef, num_terms) = _build_exam_poly(rng)
    text = phrase(rng, _EXAM_TEMPLATES, poly=poly_text) + _ANSWER_FORMAT_D
    cands = _generic_distractors(degree, lc, first_deg, first_coef, num_terms)
    cands.append((apparent_deg, apparent_coef))
    working = [
        f"The x^{apparent_deg} terms cancel: they sum to 0.",
        f"Highest surviving power is x^{degree}, with coefficient {lc}.",
        f"degree = {degree}, coefficient = {lc}.",
    ]
    return _finalise_pair(text, degree, lc, cands, working, "degree", "coefficient", n_distractors=4)


@generator(
    "p19e_poly_degree_leading",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19e_poly_degree_leading(rng, tier="easy"):
    """Given a stated polynomial in one variable, state its degree and the
    coefficient of its highest-degree (leading) term. See module docstring
    for the four-tier reasoning escalation."""
    if tier == "easy":
        return _make_easy(rng)
    if tier == "medium":
        return _make_medium(rng)
    if tier == "hard":
        return _make_hard(rng)
    if tier == "exam":
        return _make_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ============================================================================
# p19e_discriminant_zero_count (ACMMM010)
# ============================================================================


def _classify(disc: int) -> int:
    if disc > 0:
        return 2
    if disc == 0:
        return 1
    return 0


def _verify_discriminant(a: int, b: int, c: int, disc: int) -> None:
    """Independent route: sympy's own `discriminant()` on the quadratic
    expression, a different piece of code from this module's `b*b - 4*a*c`."""
    expr = Integer(a) * X**2 + Integer(b) * X + Integer(c)
    check_equivalent(
        sympy_discriminant(expr, X),
        disc,
        "p19e_discriminant_zero_count: discriminant disagrees with sympy.discriminant",
    )


def _quad_text(a: int, b: int, c: int) -> str:
    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def _row2_distractors(a, b, c, disc):
    """Every candidate's zero-count value stays in {0, 1, 2} -- `_classify` always returns one of those, and the
    two `*_count_map`s below are hand-built to only ever land in that set."""
    alt1 = b * b + 4 * a * c  # sign error: forgot the minus on 4ac
    alt2 = b - 4 * a * c  # forgot to square b
    alt3 = -(b * b) - 4 * a * c  # sign error on b^2 itself
    count = _classify(disc)
    wrong_count_map = {0: 1, 1: 2, 2: 1}  # boundary near-miss for the true D
    inverted_count_map = {0: 2, 1: 0, 2: 0}  # inverts the sign rule (D>0 <-> D<0)
    return [
        (alt1, _classify(alt1)),
        (alt2, _classify(alt2)),
        (alt3, _classify(alt3)),
        (disc, wrong_count_map[count]),
        (disc, inverted_count_map[count]),
    ]


_ANSWER_FORMAT_DISC = " Give your answer as D = ..., zeros = ..."

_EASY_TEMPLATES2 = (
    "Consider the quadratic {poly}. Compute its discriminant and state how many real zeros it has.",
    "For the quadratic {poly}, find the discriminant and the number of real zeros (0, 1 or 2).",
    "The quadratic {poly} has some number of real solutions. Find its discriminant, and "
    "state how many real zeros it has.",
)
_MEDIUM_TEMPLATES2 = (
    "Consider the quadratic {poly}. Compute its discriminant and state how many real zeros it has.",
    "For the quadratic {poly}, find the discriminant and the number of real zeros (0, 1 or 2).",
    "The quadratic {poly} has some number of real solutions. Find its discriminant, and "
    "state how many real zeros it has.",
)
_HARD_TEMPLATES2 = (
    "Consider the quadratic {poly}. Identify a, b and c, then compute the discriminant and "
    "state how many real zeros the equation has.",
    "For the quadratic {poly}, find the discriminant and the number of real zeros (0, 1 or 2).",
    "The quadratic {poly} has some number of real solutions. Find its discriminant, and "
    "state how many real zeros it has.",
)
_EXAM_TEMPLATES2 = (
    "Consider the quadratic {poly_eq}. Compute its discriminant and state how many real zeros it has.",
    "Let f(x) = {poly_expr}. Find the discriminant of f, and state how many real zeros f has.",
    "For the curve y = {poly_expr}, find the discriminant of the quadratic and state how "
    "many x-intercepts it has (0, 1 or 2).",
)


def _draw_easy_quadratic(rng, target):
    for _ in range(2000):
        a = rng.choice([1, 2, 3, 4, 5])
        b = _draw_nonzero(rng, -9, 9)
        c = _draw_nonzero(rng, -9, 9)
        disc = b * b - 4 * a * c
        if _classify(disc) == target:
            return a, b, c, disc
    raise RuntimeError(f"p19e_discriminant_zero_count (easy): failed to draw for target {target}")


def _draw_medium_quadratic(rng, target):
    if target == 1:
        for _ in range(2000):
            a = rng.choice([-3, -2, -1, 1, 2, 3])
            b = _draw_nonzero(rng, -10, 10)
            if (b * b) % (4 * a) != 0:
                continue
            c = (b * b) // (4 * a)
            disc = b * b - 4 * a * c
            if _classify(disc) == 1:
                return a, b, c, disc
        raise RuntimeError("p19e_discriminant_zero_count (medium): failed to draw count-1")
    # Broad signed sampling (a itself may be negative, unlike the easy
    # tier's a > 0 only): whenever a and c land on opposite signs, -4ac is
    # POSITIVE -- the double-negative sign trap the tier is built to teach --
    # and this happens on a solid fraction of draws without being forced,
    # so both count=0 and count=2 stay reachable (a forced opposite-sign
    # constraint would make count=0 essentially unreachable, since opposite
    # signs push the discriminant positive).
    for _ in range(3000):
        a = _draw_nonzero(rng, -6, 6)
        b = _draw_nonzero(rng, -10, 10)
        c = _draw_nonzero(rng, -6, 6)
        disc = b * b - 4 * a * c
        if _classify(disc) == target:
            return a, b, c, disc
    raise RuntimeError(f"p19e_discriminant_zero_count (medium): failed to draw for target {target}")


def _draw_hard_quadratic(rng, target):
    if target == 1:
        # A missing-term draw (b=0 AND c=0, i.e.
        # a*x^2=0) was previously the ONLY way this tier reached D=0 --
        # 58/200 draws were that one degenerate shape. Build a genuine full
        # trinomial perfect square instead, with an implicit +/-1 on `a`
        # (printed as "x^2" / "-x^2", never "1x^2"): (x + r)^2 or
        # -(x + r)^2 for a nonzero root r, so a = +/-1, b = 2*a*r,
        # c = a*r*r -- both b and c are always present and nonzero, and
        # `ax^2 = 0` never appears again. (e.g. r=-5 -> x^2 - 10x + 25;
        # a=-1, r=3 -> -x^2 + 6x - 9.)
        for _ in range(2000):
            sign = rng.choice([1, -1])
            root = _draw_nonzero(rng, -9, 9)
            a = sign
            b = 2 * sign * root
            c = sign * root * root
            disc = b * b - 4 * a * c
            if b != 0 and c != 0 and _classify(disc) == 1:
                return a, b, c, disc, "none"
        raise RuntimeError("p19e_discriminant_zero_count (hard): failed to draw count-1 perfect square")
    for _ in range(3000):
        # `missing` is redrawn every attempt (not fixed before the loop):
        # missing == "c" gives disc = b*b, always > 0, so it can NEVER reach
        # target == 0 -- fixing it up front made every target-0 draw with
        # that unlucky first choice retry forever on an impossible branch.
        missing = rng.choice(["b", "c"])
        a = rng.choice([-3, -2, -1, 1, 2, 3])
        if missing == "b":
            b = 0
            c = _draw_nonzero(rng, -12, 12)
        else:
            c = 0
            b = _draw_nonzero(rng, -12, 12)
        disc = b * b - 4 * a * c
        if _classify(disc) == target:
            return a, b, c, disc, missing
    raise RuntimeError(f"p19e_discriminant_zero_count (hard): failed to draw for target {target}")


def _draw_exam_quadratic(rng, target):
    # A quadratic's discriminant is always == b*b mod 4 (4ac contributes 0
    # mod 4), and squares mod 4 are only 0 or 1 -- so Delta = -1 or -2 is
    # mathematically impossible for ANY integer a, b, c. -3 (b odd) and -4
    # (b even) are the two smallest-magnitude achievable negative values.
    pool = {1: [0], 2: [1, 4], 0: [-3, -4]}[target]
    target_delta = rng.choice(pool)
    for _ in range(4000):
        a = rng.choice([-2, -1, 1, 2])
        b = _draw_nonzero(rng, -10, 10)
        num = b * b - target_delta
        if num % (4 * a) != 0:
            continue
        c = num // (4 * a)
        if c == 0:
            continue  # keep all three terms visible at the exam tier
        disc = b * b - 4 * a * c
        if disc == target_delta and _classify(disc) == target:
            return a, b, c, disc
    raise RuntimeError(f"p19e_discriminant_zero_count (exam): failed to draw for target {target}")


def _make_disc_easy(rng):
    target = rng.choice([0, 1, 2])
    a, b, c, disc = _draw_easy_quadratic(rng, target)
    _verify_discriminant(a, b, c, disc)
    count = _classify(disc)
    text = phrase(rng, _EASY_TEMPLATES2, poly=_quad_text(a, b, c) + " = 0") + _ANSWER_FORMAT_DISC
    cands = _row2_distractors(a, b, c, disc)
    working = [
        f"D = {paren_negative(b)}^2 - 4({a})({c}) = {disc}.",
        f"{count} real zero(s).",
    ]
    return _finalise_pair(text, disc, count, cands, working, "D", "zeros", n_distractors=4)


def _make_disc_medium(rng):
    target = rng.choice([0, 1, 2])
    a, b, c, disc = _draw_medium_quadratic(rng, target)
    _verify_discriminant(a, b, c, disc)
    count = _classify(disc)
    text = phrase(rng, _MEDIUM_TEMPLATES2, poly=_quad_text(a, b, c) + " = 0") + _ANSWER_FORMAT_DISC
    cands = _row2_distractors(a, b, c, disc)
    working = [
        f"D = {paren_negative(b)}^2 - 4({a})({c}) = {disc}.",
        f"{count} real zero(s).",
    ]
    return _finalise_pair(text, disc, count, cands, working, "D", "zeros", n_distractors=4)


def _make_disc_hard(rng):
    target = rng.choice([0, 1, 2])
    a, b, c, disc, missing = _draw_hard_quadratic(rng, target)
    _verify_discriminant(a, b, c, disc)
    count = _classify(disc)
    text = phrase(rng, _HARD_TEMPLATES2, poly=_quad_text(a, b, c) + " = 0") + _ANSWER_FORMAT_DISC
    cands = _row2_distractors(a, b, c, disc)
    working = [
        f"a = {a}, b = {b}, c = {c}.",
        f"D = {paren_negative(b)}^2 - 4({a})({c}) = {disc}.",
        f"{count} real zero(s).",
    ]
    return _finalise_pair(text, disc, count, cands, working, "D", "zeros", n_distractors=4)


def _make_disc_exam(rng):
    target = rng.choice([0, 1, 2])
    a, b, c, disc = _draw_exam_quadratic(rng, target)
    _verify_discriminant(a, b, c, disc)
    count = _classify(disc)
    poly_expr = _quad_text(a, b, c)
    text = phrase(rng, _EXAM_TEMPLATES2, poly_eq=poly_expr + " = 0", poly_expr=poly_expr) + _ANSWER_FORMAT_DISC
    cands = _row2_distractors(a, b, c, disc)
    working = [
        f"D = {paren_negative(b)}^2 - 4({a})({c}) = {disc}.",
        f"{count} real zero(s).",
    ]
    return _finalise_pair(text, disc, count, cands, working, "D", "zeros", n_distractors=4)


@generator(
    "p19e_discriminant_zero_count",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19e_discriminant_zero_count(rng, tier="easy"):
    """Given a quadratic ax^2+bx+c with stated integer a, b, c, compute its
    discriminant b^2-4ac and state the number of real zeros (0, 1 or 2) it
    implies. See module docstring for the four-tier reasoning escalation."""
    if tier == "easy":
        return _make_disc_easy(rng)
    if tier == "medium":
        return _make_disc_medium(rng)
    if tier == "hard":
        return _make_disc_hard(rng)
    if tier == "exam":
        return _make_disc_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
