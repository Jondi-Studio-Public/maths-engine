"""Design constraint: every question is
constructed so the fully-cancelled answer always lands as a single monomial
numerator (a coefficient and, at most, ONE surviving variable power -- never
an addition or subtraction) over a bare positive integer denominator. That
guarantees the canonical's raw text never contains "(" at all, so
`answer_form="expanded"` (mathgen/grading/typed.py) accepts it, while every question's
own uncancelled quotient/product text -- built from two (or three) fractions
each individually bracketed -- always carries brackets and is rejected by the
same mechanism.

Every tier is forward-constructed: the reduced coefficient is always derived via `fractions.Fraction`, which
reduces exactly and can never fail or need a retry, and every variable exponent is a small integer
addition/subtraction computed directly -- there is no path in this module that draws numbers and then
discovers, after the fact, whether they "worked". The only `while` loops here are bounded wording guards
(never redraw two source numbers so they render as a visually-confusing coincidence, e.g. "6x/6") or a
degenerate-answer guard (denominator reducing to 1, which is mathematically fine but a duller question) --
never a correctness retry., even though the answer is not always chosen strictly before every source number is
drawn.

Four tiers, genuine reasoning escalation:
  easy   -- multiply two single-variable (x) fractions; the numeric
            coefficients cross-cancel and x cancels completely (the
            question's own worked shape, "(2x/3) x (5/(4x))"). The rendered
            operator is the multiplication sign U+00D7, never a bare "x" --
            the variable and the operator would otherwise collide.
  medium -- DIVIDE two single-variable (x) fractions -- invert-and-multiply
            is a step easy never needs; x still cancels completely.
  hard   -- two variables, divide, matching an exam-style item's own shape
            ("(5xy/3) / (10x/(7y))"): a coefficient cancellation AND a
            per-variable step -- x cancels fully, y's powers combine to y^2.
  exam   -- a genuine escalation over hard, not a bigger number. Two
            clears from shape variety alone:
              * "chain3": hard's own two fractions, divided, THEN a third,
                purely numeric fraction multiplied or divided in -- a chained
                three-fraction operation hard never has.
              * "power": one single-variable (y) fraction SQUARED, then
                divided by a second -- an exponent-of-a-fraction step no
                lower tier has. Always a division (never a multiply) so
                forgot-to-invert is exercisable on every exam-power seed.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from sympy import Rational, symbols

from ..core import check_distinct, check_equivalent, fmt_signed, generator, phrase, pick_distractors

x, y = symbols("x y")


# --- shared text/value helpers ----------------------------------------------


def _term(coef: int, var: str = "") -> str:
    """One coefficient*variable term, student-style ('27y^2', '4x', '9')."""
    return fmt_signed(coef, var)


def _frac_text(num_coef: int, num_var: str, den_coef: int, den_var: str) -> str:
    """'num/den', bracketing the denominator only when it is itself compound
    (coefficient*variable) -- a bare integer or a bare variable denominator
    never needs one. Never used for the canonical answer (whose denominator
    is always a bare positive integer by construction, see module docstring)."""
    num = _term(num_coef, num_var)
    den = _term(den_coef, den_var)
    if den_var:
        den = f"({den})"
    return f"{num}/{den}"


def _wrap(text: str) -> str:
    return f"({text})"


def _reduced(num: int, den: int) -> tuple[int, int]:
    """A distractor's coefficient is always DISPLAYED in lowest terms, even
    when the misconception's own story is "left an extra factor
    un-cancelled" -- what changed is the VALUE (num/den, deliberately not
    reduced to num0/den0), never how that value is then written down."""
    frac = Fraction(num, den)
    return frac.numerator, frac.denominator


def _answer_text(coef_num: int, var: str, coef_den: int) -> str:
    """The canonical's own raw text: a bare monomial numerator over a bare
    positive integer denominator, NEVER a bracket -- the `expanded` contract
    this whole module exists to satisfy."""
    numerator = _term(coef_num, var)
    return numerator if coef_den == 1 else f"{numerator}/{coef_den}"


#: The prompt-parsing regexes in this module's own test file key off the
#: literal word "Simplify" at the start of every question, so every template
#: here keeps that word -- only the FRAME around the expression varies.
_SENTENCE_PATTERNS = (
    "Simplify {expr}.",
    "Simplify {expr}, giving your answer in simplest form.",
    "Simplify {expr} to a single term.",
)


def _sentence(rng, expr_text: str) -> str:
    return phrase(rng, _SENTENCE_PATTERNS, expr=expr_text)


def _coprime_answer(rng, lo: int = 2, hi: int = 9) -> tuple[int, int]:
    """A coprime (P, Q) pair with Q != 1 -- P/Q always reduces to a genuine
    fraction (Q == 1 is mathematically fine but a duller question, so this
    is a wording guard, not a correctness retry: math.gcd never fails, and
    the loop only ever redraws because Q happened to divide P away)."""
    while True:
        p0 = rng.randint(lo, hi)
        q0 = rng.randint(lo, hi)
        g0 = gcd(p0, q0)
        p, q = p0 // g0, q0 // g0
        if q != 1:
            return p, q


def _gh(rng, lo: int = 2, hi: int = 4) -> tuple[int, int]:
    """Two distinct small forced-common-factors, g and h -- g cross-cancels
    between the two given fractions during combination, h is left internal
    to one fraction so it must ALSO be cancelled on its own. Distinct only
    as a wording guard (so a render never shows the same factor twice and
    reads as a typo); both g and h are >= 2 unconditionally, which is what
    keeps 'coefficient-not-reduced' genuinely exercisable every seed."""
    g = rng.randint(lo, hi)
    h = rng.randint(lo, hi)
    while h == g:
        h = rng.randint(lo, hi)
    return g, h


# --- easy: multiply, one variable, fully cancels ----------------------------


def _easy(rng):
    p, q = _coprime_answer(rng)
    g = rng.randint(2, 4)
    k = rng.randint(2, 9)
    while k == p * g:  # wording guard: never render "(kx/k) ..." (a == b)
        k = rng.randint(2, 9)
    a, b = p * g, k  # F1 = a*x/b
    c, d = k, q * g  # F2 = c/(d*x)

    f1_text = _frac_text(a, "x", b, "")
    f2_text = _frac_text(c, "", d, "x")
    text = _sentence(rng, f"{_wrap(f1_text)} × {_wrap(f2_text)}")

    literal_expr = (a * x / b) * (Rational(c) / (d * x))
    check_equivalent(literal_expr, Rational(p, q), "easy: literal product does not reduce to the constructed answer")

    answer_expr = Rational(p, q)
    answer_text = _answer_text(p, "", q)

    # dropped-a-variable-power: fails to cancel x, leaves it in the answer.
    dropped_var_expr = Rational(p, q) * x
    dropped_var_text = _answer_text(p, "x", q)
    # coefficient-not-reduced: cancels g from the denominator side (d) but
    # leaves the numerator's own copy of g in place -- g*P/Q, a genuinely
    # different value since g >= 2 unconditionally.
    cnr_num, cnr_den = _reduced(p * g, q)
    coeff_not_reduced_expr = Rational(cnr_num, cnr_den)
    coeff_not_reduced_text = _answer_text(cnr_num, "", cnr_den)

    check_distinct(
        answer_expr,
        dropped_var_expr,
        coeff_not_reduced_expr,
        msg="easy: distractor must be a genuinely different value",
    )

    return text, answer_expr, answer_text, [dropped_var_text, coeff_not_reduced_text]


# --- medium: divide, one variable, fully cancels ----------------------------


def _medium(rng):
    p, q = _coprime_answer(rng)
    g, h = _gh(rng)
    a, b = p * h, g * h  # F1 = a*x/b
    c, d = q, g  # F2 = c*x/d

    f1_text = _frac_text(a, "x", b, "")
    f2_text = _frac_text(c, "x", d, "")
    text = _sentence(rng, f"{_wrap(f1_text)} / {_wrap(f2_text)}")

    literal_expr = (a * x / b) / (Rational(c) * x / d)
    check_equivalent(literal_expr, Rational(p, q), "medium: literal quotient does not reduce to the constructed answer")

    answer_expr = Rational(p, q)
    answer_text = _answer_text(p, "", q)

    # forgot-to-invert: multiplies the second fraction straight through
    # instead of inverting it -- a*c*x^2/(b*d): x SURVIVES at power 2
    # instead of cancelling, a genuinely different value.
    fi = Fraction(a * c, b * d)
    forgot_invert_expr = Rational(fi.numerator, fi.denominator) * x**2
    forgot_invert_text = _answer_text(fi.numerator, "x^2", fi.denominator)
    # dropped-a-variable-power: fails to cancel x, leaves it in the answer.
    dropped_var_expr = Rational(p, q) * x
    dropped_var_text = _answer_text(p, "x", q)
    # coefficient-not-reduced: F1's own internal factor h left un-cancelled.
    cnr_num, cnr_den = _reduced(p * h, q)
    coeff_not_reduced_expr = Rational(cnr_num, cnr_den)
    coeff_not_reduced_text = _answer_text(cnr_num, "", cnr_den)

    check_distinct(
        answer_expr,
        forgot_invert_expr,
        dropped_var_expr,
        coeff_not_reduced_expr,
        msg="medium: distractor must be a genuinely different value",
    )

    return text, answer_expr, answer_text, [forgot_invert_text, dropped_var_text, coeff_not_reduced_text]


# --- hard: two variables, divide, x cancels, y combines to y^2 -------------


def _hard(rng):
    p, q = _coprime_answer(rng)
    g, h = _gh(rng)
    m1, n1 = p * h, g * h  # F1 = m1*x*y/n1
    m2, n2 = q, g  # F2 = m2*x/(n2*y)

    f1_text = _frac_text(m1, "xy", n1, "")
    f2_text = _frac_text(m2, "x", n2, "y")
    text = _sentence(rng, f"{_wrap(f1_text)} / {_wrap(f2_text)}")

    literal_expr = (m1 * x * y / n1) / (Rational(m2) * x / (n2 * y))
    check_equivalent(
        literal_expr, Rational(p, q) * y**2, "hard: literal quotient does not reduce to the constructed answer"
    )

    answer_expr = Rational(p, q) * y**2
    answer_text = _answer_text(p, "y^2", q)

    # forgot-to-invert: multiplies straight through -- (m1*m2*x^2)/(n1*n2):
    # x SURVIVES at power 2 (instead of cancelling) and y cancels (instead
    # of combining) -- a different surviving variable entirely.
    fi = Fraction(m1 * m2, n1 * n2)
    forgot_invert_expr = Rational(fi.numerator, fi.denominator) * x**2
    forgot_invert_text = _answer_text(fi.numerator, "x^2", fi.denominator)
    # dropped-a-variable-power: treats y as cancelling fully, like x, instead
    # of combining -- states the coefficient alone, no surviving variable.
    dropped_var_expr = Rational(p, q)
    dropped_var_text = _answer_text(p, "", q)
    # coefficient-not-reduced: F1's own internal factor h left un-cancelled.
    cnr_num, cnr_den = _reduced(p * h, q)
    coeff_not_reduced_expr = Rational(cnr_num, cnr_den) * y**2
    coeff_not_reduced_text = _answer_text(cnr_num, "y^2", cnr_den)

    check_distinct(
        answer_expr,
        forgot_invert_expr,
        dropped_var_expr,
        coeff_not_reduced_expr,
        msg="hard: distractor must be a genuinely different value",
    )

    return text, answer_expr, answer_text, [forgot_invert_text, dropped_var_text, coeff_not_reduced_text]


# --- exam variant 1: hard's own quotient, chained with a third fraction ----


def _exam_chain3(rng):
    p, q = _coprime_answer(rng)
    g, h = _gh(rng)
    m1, n1 = p * h, g * h  # F1 = m1*x*y/n1
    m2, n2 = q, g  # F2 = m2*x/(n2*y)
    m3 = rng.randint(2, 9)
    n3 = rng.randint(2, 9)
    while n3 == m3:  # wording guard, not a correctness retry
        n3 = rng.randint(2, 9)
    op2 = rng.choice(["*", "/"])
    op2_symbol = "×" if op2 == "*" else "/"

    f1_text = _frac_text(m1, "xy", n1, "")
    f2_text = _frac_text(m2, "x", n2, "y")
    f3_text = _frac_text(m3, "", n3, "")
    text = _sentence(rng, f"{_wrap(f1_text)} / {_wrap(f2_text)} {op2_symbol} {_wrap(f3_text)}")

    intermediate = (m1 * x * y / n1) / (Rational(m2) * x / (n2 * y))
    third = Rational(m3, n3)
    literal_expr = intermediate * third if op2 == "*" else intermediate / third

    base = Fraction(p, q)
    op2_factor = Fraction(m3, n3) if op2 == "*" else Fraction(n3, m3)
    coef = base * op2_factor
    check_equivalent(
        literal_expr,
        Rational(coef.numerator, coef.denominator) * y**2,
        "exam-chain3: literal chain does not reduce to the constructed answer",
    )

    answer_expr = Rational(coef.numerator, coef.denominator) * y**2
    answer_text = _answer_text(coef.numerator, "y^2", coef.denominator)

    # forgot-to-invert (the F1/F2 step -- always present, op1 is fixed "/"):
    # multiplies F1*F2 straight, then still applies op2 with F3 correctly.
    fi_base = Fraction(m1 * m2, n1 * n2) * op2_factor
    forgot_invert_expr = Rational(fi_base.numerator, fi_base.denominator) * x**2
    forgot_invert_text = _answer_text(fi_base.numerator, "x^2", fi_base.denominator)
    # dropped-a-variable-power: y cancels fully instead of combining to y^2
    # -- same coefficient handling as the answer, no surviving variable.
    dropped_var_expr = Rational(coef.numerator, coef.denominator)
    dropped_var_text = _answer_text(coef.numerator, "", coef.denominator)
    # coefficient-not-reduced: F1's own internal factor h left un-cancelled.
    cnr = (base * h) * op2_factor
    coeff_not_reduced_expr = Rational(cnr.numerator, cnr.denominator) * y**2
    coeff_not_reduced_text = _answer_text(cnr.numerator, "y^2", cnr.denominator)

    check_distinct(
        answer_expr,
        forgot_invert_expr,
        dropped_var_expr,
        coeff_not_reduced_expr,
        msg="exam-chain3: distractor must be a genuinely different value",
    )

    return text, answer_expr, answer_text, [forgot_invert_text, dropped_var_text, coeff_not_reduced_text]


# --- exam variant 2: one fraction squared, then divided --------------------


def _exam_power(rng):
    while True:
        j = rng.randint(2, 4)
        base1 = rng.randint(2, 4)
        base2 = rng.randint(2, 4)
        if base1 == base2:  # wording guard, not a correctness retry
            continue
        m1, n1 = base1 * j, base2 * j  # F1 = m1*y/n1 (internal factor j)
        m2 = rng.randint(2, 9)
        n2 = rng.randint(2, 9)
        if m2 == n2:  # wording guard, not a correctness retry
            continue
        coef = Fraction(m1 * m1 * n2, n1 * n1 * m2)
        if coef.denominator != 1:  # degenerate-answer guard, not correctness
            break
    p, q = coef.numerator, coef.denominator

    f1_text = _frac_text(m1, "y", n1, "")
    f2_text = _frac_text(m2, "y", n2, "")
    text = _sentence(rng, f"{_wrap(f1_text)}^2 / {_wrap(f2_text)}")

    literal_expr = (Rational(m1) * y / n1) ** 2 / (Rational(m2) * y / n2)
    check_equivalent(
        literal_expr,
        Rational(p, q) * y,
        "exam-power: literal power expression does not reduce to the constructed answer",
    )

    answer_expr = Rational(p, q) * y
    answer_text = _answer_text(p, "y", q)

    # forgot-to-invert: multiplies by F2 straight instead of inverting it --
    # (m1^2*m2)*y^3/(n1^2*n2): y ends at power 3, not 1.
    fi = Fraction(m1 * m1 * m2, n1 * n1 * n2)
    forgot_invert_expr = Rational(fi.numerator, fi.denominator) * y**3
    forgot_invert_text = _answer_text(fi.numerator, "y^3", fi.denominator)
    # dropped-a-variable-power: forgets the square (uses F1 unsquared), so y
    # cancels fully against F2's own y instead of leaving y^1 behind.
    dv = Fraction(m1 * n2, n1 * m2)
    dropped_var_expr = Rational(dv.numerator, dv.denominator)
    dropped_var_text = _answer_text(dv.numerator, "", dv.denominator)
    # coefficient-not-reduced: F1's own internal factor j left un-cancelled
    # (one factor of it, not the j^2 the square actually contributes).
    cnr_num, cnr_den = _reduced(p * j, q)
    coeff_not_reduced_expr = Rational(cnr_num, cnr_den) * y
    coeff_not_reduced_text = _answer_text(cnr_num, "y", cnr_den)

    check_distinct(
        answer_expr,
        forgot_invert_expr,
        dropped_var_expr,
        coeff_not_reduced_expr,
        msg="exam-power: distractor must be a genuinely different value",
    )

    return text, answer_expr, answer_text, [forgot_invert_text, dropped_var_text, coeff_not_reduced_text]


@generator(
    "exd_e2_algebraic_fraction_mul_div",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e2_algebraic_fraction_mul_div(rng, tier: str = "medium"):
    if tier == "easy":
        text, answer_expr, answer_text, cands = _easy(rng)
    elif tier == "medium":
        text, answer_expr, answer_text, cands = _medium(rng)
    elif tier == "hard":
        text, answer_expr, answer_text, cands = _hard(rng)
    elif tier == "exam":
        variant = rng.choice(["chain3", "power"])
        if variant == "chain3":
            text, answer_expr, answer_text, cands = _exam_chain3(rng)
        else:
            text, answer_expr, answer_text, cands = _exam_power(rng)
    else:
        raise ValueError(f"unknown tier {tier!r}")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
        answer_form="expanded",
    )
