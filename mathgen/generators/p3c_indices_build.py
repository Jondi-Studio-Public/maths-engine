"""Index laws review and radicals/fractional indices, both senior ``acara-senior-v8-4``, domain
``functions-sequences``, ``answer: expression``/``answer: number``.

Two generators:

- ``index_laws_review`` (``senior-review-indices-including-fractional-indices-and-the-index``,
  ACMMM061) -- apply the product, quotient and power-of-a-power index laws to combine two or
  three like-base index terms with a symbolic base ``x``, including simple fractional exponents
  (1/2, 1/3, 2/3). Every canonical answer is a simplified power of ``x``, graded by sympy
expression equivalence.
  ``exam_algebraic_fraction``: an algebraic fraction (both numerator and denominator a numeric
  perfect-power coefficient times a power of ``x``) raised to a fractional index, composing the
  fraction-simplification step with the same perfect-power/fractional-power idea the ``hard``
  ``power_fraction`` variant already proves for a single term.

- ``radicals_fractional_indices`` (``senior-use-radicals-and-convert-to-and-from-fractional``,
  ACMMM062) -- evaluate a numeral presented in fractional-index or worded-radical notation to
  its exact integer value. Includes both notations the descriptor names ("convert to and from
  fractional indices"), displayed as either fractional-index numeral form or a worded
  root-then-power description, always evaluated to the same exact integer.

Both generators use sympy for expression construction and simplify-based verification;
radicals_fractional_indices uses only integer arithmetic for answers (no decimals).
"""

from __future__ import annotations

from math import gcd

from sympy import symbols, Rational, latex, sympify

from ..core import check, check_distinct, check_equivalent, generator, phrase, pick_distractors


def _xp(e) -> str:
    """x raised to e, writing plain x for exponent 1."""
    return "x" if e == 1 else f"x^{e}"


def _display(expr) -> str:
    """Learner-facing rendering of a sympy power: ``x^5``, ``x^(3/2)``.

    ``Question.answer`` keeps ``str(expr)`` because the expression grader
    sympifies it; only ``answer_text``, ``text`` and distractors use ``^``
    (the grader already accepts ``^`` from the learner).
    """
    return str(expr).replace("**", "^")


x = symbols("x")

# A POSITIVE symbol, used only by the exam variant below. `(x**-4)**Rational(3,4)` is not equal to
# `x**-3` for a general real x (an even power inside a fractional outer power hits a genuine
# branch-cut ambiguity for negative x -- checked directly: `sympy.simplify` on the assumption-free
# `x` returns `.is_zero is None`, not `True`), so the nested algebraic-fraction exam item is only
# exact once x is known positive, the standard convention for this class of "simplify to a single
# power" item. The module-level plain `x` above is untouched and still used by every existing
# variant -- this is an additive, second symbol, never a reassignment.
x_pos = symbols("x", positive=True)

# =============================================================================
# index_laws_review (Skill A)
# =============================================================================


def _verify_index_law_combination(original_expr, answer_expr) -> None:
    """Independent check: use sympy's own simplification engine on the
    unsimplified original expression, comparing against the hand-derived answer.
    A genuinely different route from construction."""
    check_equivalent(
        original_expr,
        answer_expr,
        "hand-computed exponent arithmetic disagrees with sympy's own "
        "simplification of the original unsimplified expression",
    )


def _build_easy_product(rng) -> dict:
    """Easy: single application of the product law. x**a * x**b = x**(a+b)."""
    a = rng.randint(2, 9)
    b = rng.randint(1, 9)

    # Build the unsimplified original expression (sympy auto-combines, but
    # construction vs simplification are still different routes).
    original_expr = x**a * x**b

    # Hand-computed exponent (not via sympy simplification).
    ans_exp = a + b
    answer_expr = x**ans_exp

    _verify_index_law_combination(original_expr, answer_expr)

    answer_text = _display(answer_expr)

    text = f"Simplify {_xp(a)} × {_xp(b)}, giving your answer as a single power of x."
    latex_text = rf"{latex(x**a)} \cdot {latex(x**b)}"

    # Misconceptions:
    #  - multiplies-instead-of-adding: x**(a*b)
    #  - subtracts-instead-of-adding: x**(a-b)
    wrong1 = x ** (a * b)
    wrong2 = x ** (a - b)

    candidates = [_display(wrong1), _display(wrong2)]

    return dict(
        text=text,
        latex=latex_text,
        answer=str(answer_expr),
        answer_text=answer_text,
        answer_latex=latex(answer_expr),
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


def _build_medium_product_quotient(rng) -> dict:
    """Medium: product AND quotient combined. (x**a * x**b) / x**c = x**(a+b-c)."""
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    # Guarantee a+b-c >= 1, i.e. final exponent is positive.
    c = rng.randint(1, a + b - 1)

    original_expr = (x**a * x**b) / x**c

    # Hand-computed exponent.
    ans_exp = a + b - c
    answer_expr = x**ans_exp

    _verify_index_law_combination(original_expr, answer_expr)

    answer_text = _display(answer_expr)

    text = f"Simplify ({_xp(a)} × {_xp(b)}) ÷ {_xp(c)}, giving your answer as a single power of x."

    # Misconceptions:
    #  - ignores-the-division: x**(a+b)
    #  - treats-division-as-multiplication: x**(a+b+c)
    wrong3 = x ** (a + b)
    wrong4 = x ** (a + b + c)

    candidates = [_display(wrong3), _display(wrong4)]

    return dict(
        text=text,
        latex=rf"(x^{{{a}}} \cdot x^{{{b}}}) / x^{{{c}}}",
        answer=str(answer_expr),
        answer_text=answer_text,
        answer_latex=latex(answer_expr),
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


def _build_hard_power_fraction(rng) -> dict:
    """Hard: power-of-a-power law combined with product/quotient, and a fractional
    base exponent. (x**p)**q * x**r / x**s = x**(p*q + r - s)."""
    FRACS = [Rational(1, 2), Rational(1, 3), Rational(2, 3)]
    p = rng.choice(FRACS)
    q = rng.randint(2, 4)
    r = rng.randint(1, 9)
    s = rng.randint(1, 9)

    original_expr = (x**p) ** q * x**r / x**s

    # Hand-computed exponent using sympy.Rational arithmetic.
    ans_exp = p * q + r - s
    answer_expr = x**ans_exp

    _verify_index_law_combination(original_expr, answer_expr)

    answer_text = _display(answer_expr)

    # Render p as a string (e.g., "1/2").
    p_str = str(p)

    text = f"Simplify (x^({p_str}))^{q} × {_xp(r)} ÷ {_xp(s)}, giving your answer as a single power of x."

    # For the latex, hand-write to avoid confusion with precedence.
    # (x**(1/2))**3 to show the fractional exponent clearly.
    latex_text = rf"(x^{{{p}}})^{{{q}}} \cdot x^{{{r}}} / x^{{{s}}}"

    # Misconceptions:
    #  - power-of-power-adds-instead-of-multiplies: (p+q) + r - s instead of (p*q) + r - s
    #  - quotient-sign-flip: (p*q) + r + s instead of (p*q) + r - s
    #  - reads-fractional-exponent-as-coefficient: p * x**(q + r - s)

    wrong5_exp = p + q + r - s
    wrong5 = x**wrong5_exp

    wrong6_exp = p * q + r + s
    wrong6 = x**wrong6_exp

    wrong7 = p * x ** (q + r - s)

    candidates = [_display(wrong5), _display(wrong6), _display(wrong7)]

    return dict(
        text=text,
        latex=latex_text,
        answer=str(answer_expr),
        answer_text=answer_text,
        answer_latex=latex(answer_expr),
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


def _build_exam_algebraic_fraction(rng) -> dict:
    """Exam: an algebraic FRACTION -- numerator and
    denominator both powers of x with numeric coefficients that are perfect
    powers -- raised to a fractional index. Composes two steps the lower
    tiers already exercise separately: simplify the fraction inside (subtract
    exponents; reduce the numeric coefficient to a ratio of two coprime prime
    powers) THEN apply the fractional power `hard`'s ``power_fraction``
    variant already proves for a single perfect-power coefficient/base -- to
    both the numeric coefficient (comes out exact by construction, reused
    from that same idea) and the symbolic exponent (multiply).

    Backwards construction for exactness: pick
    the fractional index n/d, two distinct small primes p != q, and the
    coefficient exponent multiplier equal to d itself -- so the fraction's
    numerator coefficient is p**d and its denominator coefficient is q**d.
    Dividing gives (p/q)**d; raising to n/d gives (p/q)**n exactly (no
    residual surd, matching the exam tier's own 256**(3/4) = 4**3 = 64
    exactness), a genuine integer power of a genuinely reduced (p and q are
    distinct primes, so p**n/q**n is already in lowest terms) rational. The x
    exponents are built the same way: their difference is forced to a
    multiple of d, so dividing by the fraction bar and then applying n/d
    lands on an exact integer exponent too.
    """
    d = rng.choice([2, 3, 4])
    n = rng.choice([i for i in range(1, d) if gcd(i, d) == 1])
    p, q = rng.sample([2, 3, 5], 2)

    k = d  # coefficient exponent multiplier: t=1, kept minimal for size.
    num_coef = p**k
    den_coef = q**k

    m = rng.choice([1, 2])
    # base_exp >= 2 so neither x-exponent ever renders as the bare "x^1" a
    # learner never sees on paper.
    base_exp = rng.randint(2, 3)
    num_smaller = rng.choice([True, False])
    if num_smaller:
        a, b = base_exp, base_exp + m * d
    else:
        a, b = base_exp + m * d, base_exp

    diff = a - b  # an exact multiple of d, signed either way.
    final_x_exp = (diff // d) * n  # exact: diff is a multiple of d by construction.
    coef_num_final = p**n
    coef_den_final = q**n

    # Independent route: build the ORIGINAL unsimplified nested
    # expression from num_coef/den_coef/a/b/n/d directly -- none of which
    # were derived from final_x_exp/coef_num_final/coef_den_final above --
    # and let sympy's own simplification confirm it equals the
    # hand-constructed answer.
    original_expr = ((num_coef * x_pos**a) / (den_coef * x_pos**b)) ** Rational(n, d)
    answer_expr = Rational(coef_num_final, coef_den_final) * x_pos**final_x_exp
    check_equivalent(
        original_expr,
        answer_expr,
        "exam nested-fraction simplification disagrees with sympy's own "
        "simplification of the original unsimplified expression",
    )

    answer_text = _display(answer_expr)

    text = phrase(
        rng,
        [
            "Simplify (({num_coef}x^{a})/({den_coef}x^{b}))^({n}/{d}), giving your answer as a single power of x.",
            "Write (({num_coef}x^{a})/({den_coef}x^{b}))^({n}/{d}) as a single power of x.",
            "Express (({num_coef}x^{a})/({den_coef}x^{b}))^({n}/{d}) in simplified form as a single power of x.",
        ],
        num_coef=num_coef,
        a=a,
        den_coef=den_coef,
        b=b,
        n=n,
        d=d,
    )

    latex_text = rf"\left(\frac{{{num_coef}x^{{{a}}}}}{{{den_coef}x^{{{b}}}}}\right)^{{{n}/{d}}}"

    # Misconceptions:
    #  - exam-exponent-subtraction-direction: subtracted the exponents the
    #    wrong way round (x**8/x**4 instead of x**4/x**8) -- the FINAL x
    #    exponent comes out sign-flipped, coefficient correct.
    #  - exam-power-applied-to-coefficient-only: applied the fractional power
    #    to the numeric coefficient but left the x exponent as the raw,
    #    unreduced difference (never multiplied by n/d).
    #  - exam-power-applied-to-exponent-only: applied the fractional power to
    #    the x exponent but left the numeric coefficient as the raw,
    #    unreduced ratio p**d/q**d (never raised to n/d).
    #  - exam-wrong-root-index (only when d == 4, the exam tier's own case --
    #    see below for why this is the only value of d that stays exact):
    #    took the coefficient's SQUARE root (index 2) regardless of the
    #    actual index denominator d, e.g. a fourth root mistaken for a
    #    square root. The correct step raises (p/q)**d to the power n/d; this
    #    misconception raises it to n/2 instead: (p/q)**(d*n/2). That is only
    #    guaranteed to stay a clean rational (no surd, so no shape/format
    #    tell separate from the misconception itself) for every valid n when
    #    d == 4 (d*n/2 = 2n, always an integer); d == 3 would need n even,
    #    which is only half of its valid n choices, and d == 2 collapses to
    #    the correct exponent itself (n/2 == n/d), not a genuinely different
    #    value at all -- so this candidate is honestly not exercisable at
    #    every d, only ever offered when it is exact by construction.
    wrong_sign = Rational(coef_num_final, coef_den_final) * x_pos ** (-final_x_exp)
    wrong_exp_unreduced = Rational(coef_num_final, coef_den_final) * x_pos**diff
    wrong_coef_unreduced = Rational(num_coef, den_coef) * x_pos**final_x_exp
    candidate_exprs = [wrong_sign, wrong_exp_unreduced, wrong_coef_unreduced]
    if d == 4:
        wrong_root = Rational(num_coef, den_coef) ** Rational(n, 2) * x_pos**final_x_exp
        candidate_exprs.append(wrong_root)

    check_distinct(answer_expr, *candidate_exprs, msg="an exam-source distractor equals the correct answer")

    # `pick_distractors(n=3)` below only ever shows 3 of these -- when a
    # fourth (`wrong_root`, d == 4 only) exists, rotate which three are
    # offered by seed rather than always dropping the same one, so
    # `exam-wrong-root-index` is genuinely exercised across seeds and not
    # dead weight in the skill's misconceptions list.
    if len(candidate_exprs) > 3:
        candidate_exprs = rng.sample(candidate_exprs, 3)

    candidates = [_display(w) for w in candidate_exprs]

    return dict(
        text=text,
        latex=latex_text,
        answer=str(answer_expr),
        answer_text=answer_text,
        answer_latex=latex(answer_expr),
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


@generator(
    "index_laws_review",
    level=11,
    presets={
        "easy": dict(variant="product"),
        "medium": dict(variant="product_quotient"),
        "hard": dict(variant="power_fraction"),
        "exam": dict(variant="exam_algebraic_fraction"),
    },
)
def index_laws_review(rng, variant="product"):
    """Apply the product, quotient and power-of-a-power index laws to combine
    like-base index terms with a symbolic base, including simple fractional
    exponents. Return the simplified single power of x."""
    if variant == "product":
        return _build_easy_product(rng)
    if variant == "product_quotient":
        return _build_medium_product_quotient(rng)
    if variant == "power_fraction":
        return _build_hard_power_fraction(rng)
    if variant == "exam_algebraic_fraction":
        return _build_exam_algebraic_fraction(rng)
    raise ValueError(f"unknown variant: {variant!r}")


# =============================================================================
# radicals_fractional_indices (Skill B)
# =============================================================================


def _find_integer_root(n: int, r: int) -> int | None:
    """Brute-force search for the exact integer r-th root of n. Small values
    by construction (n <= 243), so this is cheap. Deliberately does not use
    round(n ** (1/r)) or any floating-point path."""
    for candidate in range(1, n + 1):
        if candidate**r == n:
            return candidate
    return None


def _verify_radical_evaluation(n: int, r: int, m: int, answer: int) -> None:
    """Independent check: brute-force root search, genuinely different route
    from construction."""
    found = _find_integer_root(n, r)
    check(found is not None, f"{n} is not an exact integer {r}th power")
    check(found**m == answer, "brute-force root search disagrees with the constructed answer")


def _build_easy_root_only(rng) -> dict:
    """Easy: single-step root extraction. Evaluate n**(1/r)."""
    r = rng.choice([2, 3])
    b = rng.randint(2, 6)
    n = b**r

    answer = b
    _verify_radical_evaluation(n, r, 1, answer)

    answer_text = str(answer)

    text = f"Evaluate {n}^(1/{r})."

    # Misconceptions:
    #  - forgets-the-root: n**1 = n
    #  - off-by-one-power: b**(1+1) = b**2 (for m=1 this means b**2)
    #  - uses-the-root-index: r
    wrong1 = n**1
    wrong3 = b**2  # off by one from b**1
    wrong4 = r

    candidates = [str(wrong1), str(wrong3), str(wrong4)]

    return dict(
        text=text,
        latex=rf"{{{{{{n}}}}}}^{{{{1/{r}}}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


def _build_medium_root_then_power(rng) -> dict:
    """Medium: worded radical form -- root extraction then raising to a power.
    Example: "Evaluate the square root of 16, raised to the power of 3." = 2^3 = 8"""
    r = rng.choice([2, 3])
    # Exclude trivial m == r case.
    m = rng.choice([v for v in range(2, r + 3) if v != r])
    b = rng.randint(2, 4)
    n = b**r

    answer = b**m
    _verify_radical_evaluation(n, r, m, answer)

    answer_text = str(answer)

    ordinal = {2: "square", 3: "cube"}[r]
    text = f"Evaluate the {ordinal} root of {n}, raised to the power of {m}."

    # Misconceptions:
    #  - forgets-the-root: n**m
    #  - stops-after-the-root: b (only applies when m > 1, which is always true here)
    #  - off-by-one-power: b**(m+1)
    #  - uses-the-root-index: r
    wrong1 = n**m
    wrong2 = b
    wrong3 = b ** (m + 1)
    wrong4 = r

    candidates = [str(wrong1), str(wrong2), str(wrong3), str(wrong4)]

    return dict(
        text=text,
        latex=rf"\sqrt[{r}]{{{n}}}^{{{m}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


def _build_hard_unfamiliar_root(rng) -> dict:
    """Hard: direct fractional-index notation with a less-familiar root index.
    Evaluate n**(m/r) where r is 4 or 5, and m < r."""
    r = rng.choice([4, 5])
    # Ensure 2 <= m < r, so m/r is a genuine partial exponent.
    m = rng.choice([v for v in range(2, r) if v != r])
    b = rng.randint(2, 3)
    n = b**r

    answer = b**m
    _verify_radical_evaluation(n, r, m, answer)

    answer_text = str(answer)

    text = f"Evaluate {n}^({m}/{r})."

    # Misconceptions:
    #  - forgets-the-root: n**m
    #  - stops-after-the-root: b
    #  - off-by-one-power: b**(m+1)
    #  - uses-the-root-index: r
    wrong1 = n**m
    wrong2 = b
    wrong3 = b ** (m + 1)
    wrong4 = r

    candidates = [str(wrong1), str(wrong2), str(wrong3), str(wrong4)]

    return dict(
        text=text,
        latex=rf"{{{{{{n}}}}}}^{{{{{m}/{r}}}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, candidates, n=3),
    )


@generator(
    "radicals_fractional_indices",
    level=11,
    presets={
        "easy": dict(variant="root_only"),
        "medium": dict(variant="root_then_power"),
        "hard": dict(variant="unfamiliar_root"),
    },
)
def radicals_fractional_indices(rng, variant="root_only"):
    """Evaluate a numeral presented in fractional-index or worded-radical
    notation to its exact integer value. Includes both notations (fractional-index
    numeral form or worded root-then-power description)."""
    if variant == "root_only":
        return _build_easy_root_only(rng)
    if variant == "root_then_power":
        return _build_medium_root_then_power(rng)
    if variant == "unfamiliar_root":
        return _build_hard_unfamiliar_root(rng)
    raise ValueError(f"unknown variant: {variant!r}")
