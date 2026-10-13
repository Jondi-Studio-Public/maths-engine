"""Foundational algebra: like terms, expanding/simplifying expressions,
reading expression structure, translating words, linear relations from
points/tables, sequences, ratio word problems, coordinates, inequalities.

These sit "below" or alongside levels 7-10 in difficulty terms but reuse
nearby level numbers (no unique level is required -- see README's levels
table)."""

import math
import re
from fractions import Fraction

from sympy import Symbol, symbols, expand, latex, Rational, solve, Eq, Poly, pi, simplify, sympify, together

from ..core import (
    draw_until,
    generator,
    check,
    pick_distractors,
    rational_text,
    fmt_signed,
    join_terms,
    check_equivalent,
    check_distinct,
    pairwise_distinct,
    symbolically_equal,
    phrase,
)

x = symbols("x")
y = symbols("y")


@generator("arithmetic_properties", level=3)
def arithmetic_properties(rng):
    a, b, c = (rng.randint(2, 9) for _ in range(3))
    kind = rng.choice(("commutative", "associative", "identity", "distributive"))
    if kind == "commutative":
        text, latex_text = f"Which property is shown by {a} + {b} = {b} + {a}?", rf"{a}+{b}={b}+{a}"
        working = ["The order of the addends changes, but the sum does not."]
    elif kind == "associative":
        text, latex_text = (
            f"Which property is shown by ({a} + {b}) + {c} = {a} + ({b} + {c})?",
            rf"({a}+{b})+{c}={a}+({b}+{c})",
        )
        working = ["Only the grouping changes; the numbers stay in the same order."]
    elif kind == "identity":
        text, latex_text = f"Which property is shown by {a} + 0 = {a}?", rf"{a}+0={a}"
        working = ["Adding zero leaves a number unchanged."]
    else:
        text, latex_text = (
            f"Which property is shown by {a}({b} + {c}) = {a * b} + {a * c}?",
            rf"{a}({b}+{c})={a * b}+{a * c}",
        )
        working = [f"The factor {a} is multiplied by both terms inside the brackets."]
    names = ["commutative", "associative", "identity", "distributive"]
    return dict(
        text=text,
        latex=latex_text,
        answer=kind,
        answer_text=kind.title(),
        working=working,
        distractors=[name.title() for name in names if name != kind],
    )


@generator("variable_concept", level=4)
def variable_concept(rng):
    contexts = [
        (
            "In p + 7 = 19, what does p represent?",
            "An unknown number",
            ["The plus sign", "Seven objects", "A unit label"],
        ),
        (
            "The cost of n identical tickets is 12n dollars. What does n represent?",
            "The number of tickets",
            ["The cost of one ticket", "The total cost", "The currency"],
        ),
        (
            "In A = s², the side length s can change. What is s?",
            "A variable quantity",
            ["Always the number 2", "The area", "A multiplication sign"],
        ),
    ]
    text, answer, distractors = rng.choice(contexts)
    return dict(
        text=text,
        latex=r"\text{Choose the meaning of the variable.}",
        answer=answer,
        answer_text=answer,
        working=["A variable is a symbol that can represent an unknown or changing quantity."],
        distractors=distractors,
    )


@generator("read_expression", level=5)
def read_expression(rng):
    n = rng.randint(2, 9)
    choices = [
        (
            f"3x + {n}",
            f"three times x, then add {n}",
            [f"three times the sum of x and {n}", f"x plus three times {n}", f"divide x by 3, then add {n}"],
        ),
        (
            f"3(x + {n})",
            f"three times the sum of x and {n}",
            [f"three times x, then add {n}", f"x plus three times {n}", f"the sum of 3 and x, then add {n}"],
        ),
        (
            f"(x + {n})/2",
            f"the sum of x and {n}, divided by two",
            [f"x plus {n} divided by two", f"twice the sum of x and {n}", f"x divided by the sum of {n} and two"],
        ),
    ]
    expression, answer, distractors = rng.choice(choices)
    return dict(
        text=f"Which phrase correctly describes {expression}?",
        latex=expression,
        answer=answer,
        answer_text=answer,
        working=["Use the brackets to identify which operation happens as a group."],
        distractors=distractors,
    )


@generator("define_variable", level=5)
def define_variable(rng):
    contexts = [
        (
            "A cinema ticket costs $14. You want to represent the number of tickets bought.",
            "Let t be the number of tickets bought.",
            ["Let t be $14.", "Let tickets be the total cost.", "Let t be the cinema."],
        ),
        (
            "A tank contains some litres of water and loses 5 litres each hour.",
            "Let w be the starting litres of water.",
            ["Let w be 5 hours.", "Let w be the tank.", "Let 5 be the unknown water."],
        ),
        (
            "A rectangle has an unknown width and a known length of 8 cm.",
            "Let w be the width in centimetres.",
            ["Let w be 8 cm.", "Let w be the rectangle.", "Let length be the unknown width."],
        ),
    ]
    situation, answer, distractors = rng.choice(contexts)
    return dict(
        text=f"{situation} Which is the clearest variable definition?",
        latex=r"\text{Choose the clearest variable definition.}",
        answer=answer,
        answer_text=answer,
        working=["A useful definition names the symbol, the quantity it represents, and its unit when relevant."],
        distractors=distractors,
    )


@generator(
    "like_terms",
    level=7,
    presets={
        "easy": dict(kind="positive", n_terms=3, coef_hi=9),
        "medium": dict(kind="signed", n_terms=3, coef_hi=9),
        "hard": dict(kind="signed", n_terms=4, coef_hi=15),
    },
)
def like_terms(rng, kind="positive", n_terms=3, coef_hi=9):
    if kind == "positive":
        coefs = [rng.randint(1, coef_hi) for _ in range(n_terms)]
    else:
        coefs = [rng.randint(1, coef_hi) * rng.choice([1, -1]) for _ in range(n_terms)]
    total = sum(coefs)
    check(sum(c * x for c in coefs).coeff(x, 1) == total)

    def term_str(c, first):
        if first:
            return "x" if c == 1 else ("-x" if c == -1 else f"{c}x")
        sign = "+" if c >= 0 else "-"
        mag = abs(c)
        return f"{sign} x" if mag == 1 else f"{sign} {mag}x"

    text = " ".join(term_str(c, i == 0) for i, c in enumerate(coefs))
    ans_str = "0" if total == 0 else ("x" if total == 1 else ("-x" if total == -1 else f"{total}x"))
    # Misconceptions: summed magnitudes and ignored sign; multiplied the
    # coefficients instead of summing; dropped the last term.
    ignore_sign = sum(abs(c) for c in coefs)
    mult_coefs = 1
    for c in coefs:
        mult_coefs *= c
    drop_last = sum(coefs[:-1])

    def _cx(c):
        return "0" if c == 0 else fmt_signed(c, "x")

    cands = [_cx(ignore_sign), _cx(mult_coefs), _cx(drop_last)]
    return dict(
        text=f"Simplify {text}",
        latex=text,
        answer=ans_str,
        answer_text=ans_str,
        answer_latex=ans_str,
        distractors=pick_distractors(ans_str, cands),
    )


@generator(
    "expand_bracket",
    level=7,
    presets={
        "easy": dict(kind="single", hi=9),
        "medium": dict(kind="single", hi=15),
        "hard": dict(kind="distributive", hi=9),
    },
)
def expand_bracket(rng, kind="single", hi=9):
    if kind == "distributive":
        a = rng.randint(1, hi) * rng.choice([1, -1])
        b = rng.randint(-hi, hi) or 1
        c = rng.randint(1, hi) * rng.choice([1, -1])
        d = rng.randint(-hi, hi) or 1
        ans = expand((a * x + b) * (c * x + d))
        check(ans.coeff(x, 2) == a * c and ans.coeff(x, 1) == a * d + b * c and ans.coeff(x, 0) == b * d)
        # Misconceptions: FOIL'd only the "first" and "last" terms (dropped
        # the cross terms); sign error on the middle (cross-term) coefficient.
        firsts_lasts = a * c * x**2 + b * d
        sign_err = a * c * x**2 - (a * d + b * c) * x + b * d
        cands = [str(firsts_lasts), str(sign_err)]
        bs = "+" if b >= 0 else "-"
        ds = "+" if d >= 0 else "-"
        ax, cx = fmt_signed(a, "x"), fmt_signed(c, "x")
        text = f"Expand ({ax} {bs} {abs(b)})({cx} {ds} {abs(d)})"
    elif kind == "single_trinomial":
        # A genuine reasoning escalation over kind == "single", never a bigger `hi` -- THREE terms
        # inside the bracket across TWO variables, with `a` drawn negative-or-fractional. A negative
        # multiplier alone would NOT measure as an escalation: `mathgen.text_quality._mask_digits`
        # masks a leading sign together with its digit run, so "Expand -3(x + 4)" and "Expand 3(x +
        # 5)" mask to the identical shape. The structural changes that actually move the token
        # sequence are the extra term/variable (always present) and, when `a` is fractional, the
        # literal "/" in the prompt -- a "/" is never swallowed by that mask, unlike a bare sign.
        a = rng.choice([-3, -2, 2, 3, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3)])
        b = rng.randint(-hi, hi) or 1
        c = rng.randint(-hi, hi) or 1
        d = rng.randint(-hi, hi) or 1
        ans = expand(a * (b * x + c * y + d))
        # Independent check, from scratch rather than re-reading `ans`: pull
        # the coefficient on x, the coefficient on y, and what is left once
        # both variables are zeroed out, each via a different sympy route
        # (coeff extraction / substitution) than the generator's own expand()
        # call -- proving the answer distributes across all THREE terms, no
        # more and no fewer.
        check_equivalent(ans.coeff(x, 1), a * b, "x-term coefficient wrong")
        check_equivalent(ans.coeff(y, 1), a * c, "y-term coefficient wrong")
        check_equivalent(ans.subs({x: 0, y: 0}), a * d, "constant term wrong")
        # Misconceptions: partial distribution onto the x-term only (left the
        # y-term and the constant un-multiplied); partial distribution onto
        # the x- and y-terms but forgot the constant (this doubles as the
        # fractional-multiplier "forgot one term" trap whenever `a` is a
        # fraction); lost/flipped the multiplier's sign entirely.
        only_x = expand(a * b * x + c * y + d)
        forgot_const = expand(a * b * x + a * c * y + d)
        sign_flip = expand(-a * (b * x + c * y + d))
        check_distinct(ans, only_x, forgot_const, sign_flip)
        cands = [str(only_x), str(forgot_const), str(sign_flip)]
        a_text = rational_text(a)
        inner = join_terms([(b, "x"), (c, "y"), (d, "")])
        text = f"Expand {a_text}({inner})"
    else:
        a = rng.randint(2, hi) * rng.choice([1, -1])
        b = rng.randint(-hi, hi) or 1
        c = rng.randint(-hi, hi) or 1
        ans = expand(a * (b * x + c))
        check(ans.coeff(x, 1) == a * b and ans.coeff(x, 0) == a * c)
        # Misconceptions: distributed onto the x-term only (left the constant
        # un-multiplied); sign error on the constant term.
        only_x = a * b * x + c
        sign_err = a * b * x - a * c
        cands = [str(only_x), str(sign_err)]
        cs = "+" if c >= 0 else "-"
        bx = fmt_signed(b, "x")
        text = f"Expand {a}({bx} {cs} {abs(c)})"
    ans_str = str(ans)
    return dict(
        text=text,
        latex=text,
        answer=ans_str,
        answer_text=ans_str,
        answer_latex=latex(ans),
        distractors=pick_distractors(ans_str, cands),
        answer_form="expanded",
    )


@generator(
    "simplify_expression",
    level=7,
    presets={
        "easy": dict(kind="brackets", hi=9),
        "medium": dict(kind="multi", hi=9),
        "hard": dict(kind="multi", hi=15),
    },
)
def simplify_expression(rng, kind="brackets", hi=9):
    if kind == "brackets":
        a = rng.randint(2, hi) * rng.choice([1, -1])
        b = rng.randint(-hi, hi) or 1
        c = rng.randint(-hi, hi) or 1
        d = rng.randint(-hi, hi)
        ans = expand(a * (b * x + c) + d * x)
        check(ans.coeff(x, 1) == a * b + d and ans.coeff(x, 0) == a * c)
        # Misconceptions: forgot to distribute onto the constant; forgot the
        # extra d*x term outside the bracket entirely.
        forgot_const = expand(a * b * x + c + d * x)
        forgot_dx = a * b * x + a * c
        cands = [str(forgot_const), str(forgot_dx)]
        cs = "+" if c >= 0 else "-"
        ds = "+" if d >= 0 else "-"
        bx = fmt_signed(b, "x")
        dx_mag = "x" if abs(d) == 1 else f"{abs(d)}x"
        text = f"Simplify {a}({bx} {cs} {abs(c)}) {ds} {dx_mag}"
    else:
        a = rng.randint(1, hi)
        b = rng.randint(1, hi)
        c = rng.randint(2, hi) * rng.choice([1, -1])
        d = rng.randint(-hi, hi) or 1
        e = rng.randint(-hi, hi)
        ans = expand(a * x + c * (b * x + d) - e)
        check(ans.coeff(x, 1) == a + c * b and ans.coeff(x, 0) == c * d - e)
        # Misconceptions: forgot to distribute c onto the constant d; added e
        # instead of subtracting it.
        forgot_const = expand(a * x + c * b * x + d - e)
        wrong_e_sign = expand(a * x + c * (b * x + d) + e)
        cands = [str(forgot_const), str(wrong_e_sign)]
        ds = "+" if d >= 0 else "-"
        bx = fmt_signed(b, "x")
        mid = f"({bx} {ds} {abs(d)})"
        text = join_terms([(a, "x"), (c, mid), (-e, "")])
    ans_str = str(ans)
    return dict(
        text=text,
        latex=text,
        answer=ans_str,
        answer_text=ans_str,
        answer_latex=latex(ans),
        distractors=pick_distractors(ans_str, cands),
        answer_form="expanded",
    )


def _rand_poly_terms(rng, n, hi, max_pow):
    powers = rng.sample(range(1, max_pow + 1), k=min(n, max_pow))
    coefs = [rng.randint(1, hi) * rng.choice([1, -1]) for _ in powers]
    return sorted(zip(coefs, powers), key=lambda t: -t[1])


@generator(
    "expression_parts",
    level=7,
    presets={
        "easy": dict(kind="term_count", hi=9),
        "medium": dict(kind="coefficient", hi=9),
        "hard": dict(kind="constant", hi=15),
    },
)
def expression_parts(rng, kind="coefficient", hi=9, n_terms=2, max_pow=3):
    terms = _rand_poly_terms(rng, n_terms, hi, max_pow)
    const = rng.randint(-hi, hi)
    expr = sum(c * x**p for c, p in terms) + const
    text_expr = str(expr)
    if kind == "term_count":
        ans = len(terms) + (1 if const != 0 else 0)
        cands = [ans - 1, ans + 1, len(terms)]
        prompt = f"How many terms does {text_expr} have?"
    elif kind == "coefficient":
        c, p = rng.choice(terms)
        ans = c
        # Misconceptions: flipped the sign; off-by-one; quoted another
        # term's coefficient (always available since n_terms >= 2).
        other = next((tc for tc, tp in terms if tp != p), c * p)
        cands = [-c, c + 1, other]
        power_str = "x" if p == 1 else f"x^{p}"
        prompt = f"What is the coefficient of {power_str} in {text_expr}?"
    else:  # constant
        ans = const
        other = terms[-1][0] if terms else 0
        cands = [other, -const, const + 1 if const != -1 else const - 1]
        prompt = f"What is the constant term in {text_expr}?"
    return dict(
        text=prompt,
        latex=prompt,
        answer=int(ans),
        answer_text=str(ans),
        answer_latex=str(ans),
        distractors=pick_distractors(str(ans), [str(c_) for c_ in cands]),
    )


def _expression_from_words_structure(rng, coef_hi, const_hi):
    """Hard tier: the wording forces a structure decision (reversed
    subtraction, or a bracketed sum/difference), not just a coefficient and
    constant. Draws are resampled until every distractor differs in value
    from the answer."""
    while True:
        a = rng.randint(2, coef_hi)
        b = rng.randint(1, const_hi)
        tmpl = rng.randint(0, 2)
        if tmpl == 0:
            words = f"subtract {a} times a number from {b}"
            ans_expr = b - a * x
            # order not reversed; wrong operation; roles of a and b swapped
            cands = [a * x - b, b + a * x, a - b * x]
        elif tmpl == 1:
            words = f"{a} times the sum of a number and {b}"
            ans_expr = a * (x + b)
            # bracket dropped; multiplier on the constant only; roles swapped
            cands = [a * x + b, x + a * b, a + b * x]
        else:
            words = f"{a} times the difference between a number and {b}"
            ans_expr = a * (x - b)
            cands = [a * x - b, x - a * b, a - b * x]
        # Resample a draw where a distractor equals the answer or another
        # distractor (e.g. a == b collapses two of them).
        if not pairwise_distinct(ans_expr, *cands):
            continue
        check_distinct(ans_expr, *cands)
        ans_str = str(ans_expr)
        return dict(
            text=f"Write an expression for: {words}",
            latex=rf"\text{{{words}}}",
            answer=ans_str,
            answer_text=ans_str,
            answer_latex=latex(ans_expr),
            distractors=pick_distractors(ans_str, [str(c) for c in cands]),
        )


@generator(
    "expression_from_words",
    level=7,
    presets={
        "easy": dict(coef_hi=5, const_hi=9),
        "medium": dict(coef_hi=9, const_hi=15),
        "hard": dict(coef_hi=12, const_hi=20),
    },
)
def expression_from_words(rng, coef_hi=9, const_hi=15, kind="linear"):
    if kind == "structure":
        # Dispatched BEFORE the shared draws below so the default path's RNG
        # order is untouched (EXD-001).
        return _expression_from_words_structure(rng, coef_hi, const_hi)
    a = rng.randint(2, coef_hi)
    b = rng.randint(-const_hi, const_hi) or 1
    tmpl = rng.randint(0, 2)
    if b >= 0:
        words = [
            f"{a} times a number, plus {b}",
            f"{b} more than {a} times a number",
            f"a number multiplied by {a}, then increased by {b}",
        ][tmpl]
    else:
        words = [
            f"{a} times a number, minus {abs(b)}",
            f"{abs(b)} less than {a} times a number",
            f"a number multiplied by {a}, then decreased by {abs(b)}",
        ][tmpl]
    ans_expr = a * x + b
    ans_str = str(ans_expr)
    # Misconceptions: dropped the coefficient (just "x + b"); reversed the
    # subtraction order (the classic "N less than ..." error); flipped the
    # sign of the constant.
    cands = [str(x + b), str(b - a * x), str(a * x - b)]
    return dict(
        text=f"Write an expression for: {words}",
        latex=rf"\text{{{words}}}",
        answer=ans_str,
        answer_text=ans_str,
        answer_latex=str(ans_expr),
        distractors=pick_distractors(ans_str, cands),
    )


@generator(
    "gradient",
    level=7,
    presets={
        "easy": dict(kind="points", hi=9),
        "medium": dict(kind="table", hi=9),
        "hard": dict(kind="context", hi=15),
    },
)
def gradient(rng, kind="points", hi=9):
    if kind == "context":
        # Dispatched BEFORE the shared `-hi..hi` draw below: a tank/ time-elapsed scenario needs its
        # own independent, non-negative time construction (see `_gradient_context`'s docstring), not
        # the shared signed-coordinate draw that `points`/`table` legitimately use. Fix.
        return _gradient_context(rng, hi)
    # Backwards: choose a clean gradient (as a fraction), then two points
    # that realise it exactly.
    denom = rng.randint(1, 4)
    num = rng.randint(-hi, hi) or 1
    m = Rational(num, denom)
    x1 = rng.randint(-hi, hi)
    y1 = rng.randint(-hi, hi)
    k = rng.randint(1, 3) * rng.choice([1, -1])
    dx = denom * k
    x2, y2 = x1 + dx, y1 + num * k
    check(Rational(y2 - y1, x2 - x1) == m)
    ans_text = rational_text(m)
    # Misconceptions: inverted the ratio (run over rise); added instead of
    # subtracting; sign error from listing points in the "wrong" order.
    cand_vals = []
    if y2 != y1:
        cand_vals.append(Rational(x2 - x1, y2 - y1))
    if x2 + x1 != 0:
        cand_vals.append(Rational(y2 + y1, x2 + x1))
    if x1 != x2:
        cand_vals.append(Rational(y2 - y1, x1 - x2))
    cands = [rational_text(c) for c in cand_vals]
    if kind == "table":
        text = f"x: {x1}, {x2}   y: {y1}, {y2}. Find the gradient."
    else:
        text = f"Find the gradient of the line through ({x1}, {y1}) and ({x2}, {y2})"
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer={"p": int(m.p), "q": int(m.q)},
        answer_text=ans_text,
        answer_latex=latex(m),
        working=[
            f"m = (y2 - y1)/(x2 - x1) = ({join_terms([(y2, ''), (-y1, '')])})"
            f"/({join_terms([(x2, ''), (-x1, '')])}) = {ans_text}"
        ],
        distractors=pick_distractors(ans_text, cands),
    )


def _gradient_context(rng, hi):
    """`kind="context"`: a tank's water
    level read at two points in elapsed time. Unlike `points`/`table`,
    where both coordinates are legitimately signed, this scenario needs a
    non-negative starting time and a STRICTLY LATER second time -- elapsed
    time can never be zero or negative -- so it draws `x1`/`x2`
    independently rather than reusing `gradient`'s shared `-hi..hi` draw
    (that shared draw is exactly what produced
    "1 minutes" and "after -9 minutes"). Water level (`y1`/`y2`) stays
    signed: a negative reading relative to a reference mark is physically
    sensible for a tank scenario, so only the elapsed-time draw needs its
    own construction.
    """
    denom = rng.randint(1, 4)
    num = rng.randint(-hi, hi) or 1
    m = Rational(num, denom)
    x1 = rng.randint(0, hi)
    y1 = rng.randint(-hi, hi)
    k = rng.randint(1, 3)  # strictly positive: elapsed time always moves forward
    dx = denom * k
    x2, y2 = x1 + dx, y1 + num * k
    check(Rational(y2 - y1, x2 - x1) == m)
    check(x2 > x1, "context scenario must have strictly positive elapsed time")
    ans_text = rational_text(m)
    # Same misconception set as the shared branch above: inverted ratio,
    # added instead of subtracted, sign error from the "wrong" point order.
    cand_vals = []
    if y2 != y1:
        cand_vals.append(Rational(x2 - x1, y2 - y1))
    if x2 + x1 != 0:
        cand_vals.append(Rational(y2 + y1, x2 + x1))
    if x1 != x2:
        cand_vals.append(Rational(y2 - y1, x1 - x2))
    cands = [rational_text(c) for c in cand_vals]
    x1_word = "minute" if x1 == 1 else "minutes"
    x2_word = "minute" if x2 == 1 else "minutes"
    text = (
        f"A tank's water level was {y1} cm after {x1} {x1_word} and {y2} cm after {x2} "
        f"{x2_word}. Find the rate of change of water level (cm per minute)."
    )
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer={"p": int(m.p), "q": int(m.q)},
        answer_text=ans_text,
        answer_latex=latex(m),
        working=[
            f"m = (y2 - y1)/(x2 - x1) = ({join_terms([(y2, ''), (-y1, '')])})"
            f"/({join_terms([(x2, ''), (-x1, '')])}) = {ans_text}"
        ],
        distractors=pick_distractors(ans_text, cands),
    )


@generator(
    "linear_from_table",
    level=7,
    presets={
        "easy": dict(kind="intercept", hi=9),
        "medium": dict(kind="equation", hi=9),
        "hard": dict(kind="equation", hi=15),
    },
)
def linear_from_table(rng, kind="equation", hi=9):
    if kind == "equation_context":
        return _linear_from_table_equation_context(rng, hi)
    if kind == "intercept_context":
        return _linear_from_table_intercept_context(rng, hi)
    m = rng.randint(-hi, hi) or 1
    b = rng.randint(-hi, hi)
    xs = sorted(rng.sample(range(-hi, hi + 1), 3))
    ys = [m * xv + b for xv in xs]
    check(all(m * xv + b == yv for xv, yv in zip(xs, ys)))
    table_text = "x: " + ", ".join(str(v) for v in xs) + "   y: " + ", ".join(str(v) for v in ys)
    if kind == "intercept":
        ans = b
        # Misconceptions: quoted a y-value directly; quoted the gradient
        # instead; flipped the sign of the intercept.
        cands = [ys[0], m, -b]
        return dict(
            text=f"{table_text}. Find the y-intercept.",
            latex=table_text,
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            working=[f"b = y when x = 0; using the pattern, b = {b}"],
            distractors=pick_distractors(str(ans), [str(c) for c in cands]),
        )
    m_str = "" if m == 1 else ("-" if m == -1 else str(m))
    ans_text = f"y = {m_str}x" if b == 0 else f"y = {m_str}x {'+' if b >= 0 else '-'} {abs(b)}"
    # Misconceptions: swapped m and b; sign error on the intercept; sign
    # error on the gradient.
    cands = [
        "y = " + join_terms([(b, "x"), (m, "")]),
        "y = " + join_terms([(m, "x"), (-b, "")]),
        "y = " + join_terms([(-m, "x"), (b, "")]),
    ]
    return dict(
        text=f"{table_text}. Find the equation of the line (y = mx + b).",
        latex=table_text,
        answer={"m": m, "b": b},
        answer_text=ans_text,
        answer_latex=ans_text,
        working=[f"m = (y2-y1)/(x2-x1) = {m}", f"b = {b}"],
        distractors=pick_distractors(ans_text, cands),
    )


# --------------------------------------------------------------------------- linear_from_table's
# "_context" kinds: a short real-world scenario ("a hire company charges a flat fee plus a daily
# rate") stands in for the table, translated into the SAME m/b construction and answer shape the
# table kinds above use. `kind="equation_context"` (below) asks for the full y = mx + b;
# `kind="intercept_context"` (dispatched right after it in the same file) asks only for the
# y-intercept from an equivalent scenario and reuses `_linear_from_table_scenario` below rather than
# re-deriving its own m/b/wording.
#
# m and b are drawn STRICTLY POSITIVE (`1..hi`, not `0..hi`) for these two kinds only -- a
# construction constraint local to this branch, not a change to the existing `equation`/`intercept`
# kinds' `-hi..hi` draw above. A hire-company flat fee and daily rate can't sensibly be negative,
# and a "$0 flat fee" scenario reads oddly, so both are drawn from `1..hi` rather than merely
# `0..hi`.
_LINEAR_TABLE_CONTEXT_TEMPLATES = [
    "A hire company charges a flat fee of ${b} plus ${m} per day.",
    "A car hire company charges a flat fee of ${b}, then ${m} for each day of hire.",
    "A tool hire company charges ${b} up front and ${m} for every extra day the tool is kept.",
]


def _linear_from_table_scenario(rng, hi):
    """Shared helper for `linear_from_table`'s `kind="equation_context"`
    and `kind="intercept_context"`.

       Signature: ``_linear_from_table_scenario(rng: random.Random, hi: int) ->
       tuple[int, int, str]``, returning ``(m, b, scenario)`` where ``m`` is the
       per-day rate, ``b`` is the flat fee (both drawn `1..hi`, see the module
       comment above), and ``scenario`` is one wording from
       `_LINEAR_TABLE_CONTEXT_TEMPLATES`, filled and ready for the caller to
       append its own kind-specific question sentence after it (all three
       templates share the "flat fee ... per day ..." framing, so any of them
       reads correctly regardless of which specific question follows). Callers
       build their own answer/distractors/working from the returned `m`/`b`;
       this function does no rounding or coercion -- they are plain ints exactly
       as drawn.
    """
    m = rng.randint(1, hi)
    b = rng.randint(1, hi)
    scenario = phrase(rng, _LINEAR_TABLE_CONTEXT_TEMPLATES, m=m, b=b)
    # Independent check: the two stated facts ("$b flat fee" / "$m per day")
    # really do appear in the rendered scenario text -- this is the
    # direct-by-construction check used in place of a
    # from-scratch re-derivation, since m/b are not solved for here, only
    # substituted into a template; it still catches a template that drops
    # one of its own slots.
    check(f"${b}" in scenario and f"${m}" in scenario, "context scenario is missing one of its own drawn m/b values")
    return m, b, scenario


def _linear_from_table_equation_context(rng, hi):
    """`kind="equation_context"`: ask for the full y = mx + b from
    the hire-company scenario built by `_linear_from_table_scenario`."""
    m, b, scenario = _linear_from_table_scenario(rng, hi)
    m_str = "" if m == 1 else str(m)
    ans_text = f"y = {m_str}x + {b}"
    text = f"{scenario} Write the equation for the total cost y after x days (y = mx + b)."
    # Misconceptions -- on which NUMBER in the prose is the rate vs the fixed
    # fee, distinct from the table kinds' column-reading misconceptions above:
    #  - swapped m and b (used the flat fee as the rate and vice versa);
    #  - used only the per-day rate, dropped the flat fee (y = mx);
    #  - used only the flat fee, dropped the rate (y = b, no x term).
    swapped = "y = " + join_terms([(b, "x"), (m, "")])
    dropped_fee = "y = " + join_terms([(m, "x")])
    dropped_rate = "y = " + join_terms([(b, "")])
    cands = [swapped, dropped_fee, dropped_rate]
    return dict(
        text=text,
        latex=text,
        answer={"m": m, "b": b},
        answer_text=ans_text,
        answer_latex=ans_text,
        working=[f"Rate: ${m} per day, so m = {m}", f"Flat fee: ${b}, so b = {b}", f"y = mx + b = {ans_text}"],
        distractors=pick_distractors(ans_text, cands),
    )


def _linear_from_table_intercept_context(rng, hi):
    """`kind="intercept_context"`: ask ONLY for the y-intercept
    (the flat fee) from the hire-company scenario built by
    `_linear_from_table_scenario` -- the direct sibling of
    `_linear_from_table_equation_context` above, but the question and answer
    shape match this skill's own `kind="intercept"` branch (a bare number),
    not the `equation` shape.

    No extra independent check is added here: `_linear_from_table_scenario`
    already asserts (via its own `check()`) that both `$b` and `$m` appear
    verbatim in the rendered `scenario` text, which is exactly the "the
    rendered scenario actually contains the stated b value" check --
    duplicating it here would just re-run the same regex
    against the same string.
    """
    m, b, scenario = _linear_from_table_scenario(rng, hi)
    days = rng.randint(2, 6)
    total_after_days = m * days + b
    ans = b
    ans_text = str(ans)
    text = f"{scenario} What is the flat fee, in dollars (the y-intercept)?"
    # Misconceptions -- on which quantity answers "find the intercept",
    # distinct from the table kind's column-reading misconceptions and from
    # the equation-context kind's misconceptions (that kind swaps m/b when writing the
    # WHOLE equation; here the two wrong readings are specific to answering
    # for b alone):
    #  - reported the per-day RATE (m) instead of the flat fee (b) -- the
    #    direct sibling of the equation-context "swapped m and b";
    #  - reported the TOTAL cost after `days` days instead of the intercept
    #    itself -- misread "find the starting value" as "find the cost".
    cands = [m, total_after_days]
    return dict(
        text=text,
        latex=text,
        answer=ans,
        answer_text=ans_text,
        answer_latex=ans_text,
        working=[f"The flat fee is charged before any days are added, so b = {b} (the y-intercept)"],
        distractors=pick_distractors(ans_text, [str(c) for c in cands]),
    )


@generator(
    "sequence",
    level=10,
    presets={
        "easy": dict(kind="common_diff", hi=9),
        "medium": dict(kind="use_rule", hi=9),
        "hard": dict(kind="nth_term_rule", hi=9),
    },
)
def sequence(rng, kind="common_diff", hi=9, n_terms=4):
    a1 = rng.randint(-hi, hi)
    d = rng.randint(-hi, hi) or 1
    if kind == "common_diff_nonconsecutive":
        for _ in range(50):
            p1 = rng.randint(2, 4)
            gap = rng.randint(2, 5)
            p2 = p1 + gap
            t1 = a1 + d * (p1 - 1)
            t2 = a1 + d * (p2 - 1)
            check((t2 - t1) == d * gap)
            cands = [t2 - t1, -d, d + 1, d - 1]
            if len({c for c in cands if c != d}) >= 3:
                break
            a1 = rng.randint(-hi, hi)
            d = rng.randint(-hi, hi) or 1
        else:
            check(False, "could not find distinct distractors for common_diff_nonconsecutive")
        return dict(
            text=f"In a sequence, term {p1} is {t1} and term {p2} is {t2}. Find the common difference.",
            latex=f"a_{{{p1}}}={t1},\\ a_{{{p2}}}={t2}",
            answer=d,
            answer_text=str(d),
            answer_latex=str(d),
            working=[f"Term {p1} and term {p2} span {gap} steps: ({t2} - {t1}) / {gap} = {d}."],
            distractors=pick_distractors(str(d), [str(c) for c in cands]),
        )
    if kind == "nth_term_rule_nonconsecutive":
        for _ in range(50):
            p1 = rng.randint(2, 4)
            gap = rng.randint(2, 5)
            p2 = p1 + gap
            t1 = a1 + d * (p1 - 1)
            t2 = a1 + d * (p2 - 1)
            check((t2 - t1) == d * gap)
            rule = expand(a1 + d * (x - 1))
            ans_str = str(rule)
            cands = [
                str(expand(a1 + d * x)),
                str(expand(d + d * (x - 1))),
                str(expand(rule + 1)),
                str(expand(rule - 1)),
            ]
            if len({c for c in cands if c != ans_str}) >= 3:
                break
            a1 = rng.randint(-hi, hi)
            d = rng.randint(-hi, hi) or 1
        else:
            check(False, "could not find distinct distractors for nth_term_rule_nonconsecutive")
        return dict(
            text=f"In a sequence, term {p1} is {t1} and term {p2} is {t2}. Find the nth-term rule (use x for the term number).",
            latex=f"a_{{{p1}}}={t1},\\ a_{{{p2}}}={t2}",
            answer=ans_str,
            answer_text=ans_str,
            answer_latex=latex(rule),
            working=[
                f"Difference: ({t2} - {t1}) / {gap} = {d}. First term: {t1} - {d}({p1} - 1) = {a1}. a_x = {a1} + ({d})(x - 1) = {rule}."
            ],
            distractors=pick_distractors(ans_str, cands),
        )
    if kind == "use_rule_inverse":
        for _ in range(50):
            rule = expand(a1 + d * (x - 1))
            n_target = rng.randint(n_terms + 1, n_terms + 8)
            value = int(rule.subs(x, n_target))
            cands = [n_target + 1, n_target - 1, value]
            if len({c for c in cands if c != n_target}) >= 3:
                break
            a1 = rng.randint(-hi, hi)
            d = rng.randint(-hi, hi) or 1
        else:
            check(False, "could not find distinct distractors for use_rule_inverse")
        check(a1 + d * (n_target - 1) == value)
        return dict(
            text=f"A sequence follows a_x = {rule}. Which value of x gives a_x = {value}?",
            latex=latex(rule),
            answer=n_target,
            answer_text=str(n_target),
            answer_latex=str(n_target),
            working=[f"Solve {rule} = {value} for x: x = {n_target}."],
            distractors=pick_distractors(str(n_target), [str(c) for c in cands]),
        )
    terms = [a1 + d * i for i in range(n_terms)]
    check(all(terms[i + 1] - terms[i] == d for i in range(n_terms - 1)))
    terms_text = ", ".join(str(t) for t in terms) + ", ..."
    if kind == "common_diff":
        ans = d
        # Misconceptions: quoted the second term itself; flipped the sign;
        # quoted the first term.
        cands = [terms[1], -d, terms[0]]
        return dict(
            text=f"Find the common difference of the sequence: {terms_text}",
            latex=terms_text,
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            distractors=pick_distractors(str(ans), [str(c) for c in cands]),
        )
    rule = expand(a1 + d * (x - 1))
    if kind == "nth_term_rule":
        ans_str = str(rule)
        # Misconceptions: used d*x + a1 without the "-1" offset; used d
        # itself as the constant term instead of a1; off-by-one constant.
        cands = [str(expand(a1 + d * x)), str(expand(d + d * (x - 1))), str(expand(rule + 1))]
        return dict(
            text=f"Find the nth-term rule for the sequence: {terms_text} (use x for the term number)",
            latex=terms_text,
            answer=ans_str,
            answer_text=ans_str,
            answer_latex=latex(rule),
            working=[f"a_x = {a1} + ({d})(x - 1) = {rule}"],
            distractors=pick_distractors(ans_str, cands),
        )
    # use_rule
    n = rng.randint(n_terms + 1, n_terms + 8)
    ans = rule.subs(x, n)
    check(ans == a1 + d * (n - 1))
    # Misconceptions: used n instead of (n-1) (off-by-one term index);
    # used (n-2); quoted d*n alone.
    cands = [a1 + d * n, a1 + d * (n - 2), d * n]
    return dict(
        text=f"A sequence follows a_x = {rule}. Find the {n}th term.",
        latex=latex(rule),
        answer=int(ans),
        answer_text=str(ans),
        answer_latex=str(ans),
        distractors=pick_distractors(str(ans), [str(c) for c in cands]),
    )


@generator(
    "word_problem",
    level=7,
    presets={
        "easy": dict(coef_hi=5, sol_hi=12),
        "medium": dict(coef_hi=9, sol_hi=20),
        "hard": dict(coef_hi=12, sol_hi=30),
    },
)
def word_problem(rng, kind="linear_eq", coef_hi=9, sol_hi=15):
    if kind == "linear_eq_reverse":
        return _word_problem_linear_eq_reverse(rng, coef_hi, sol_hi)
    x0 = rng.randint(1, sol_hi)
    a = rng.randint(2, coef_hi)
    b = rng.randint(-20, 20)
    c = a * x0 + b
    check(solve(Eq(a * x + b, c), x) == [x0])
    b_phrase = f"add {b}" if b >= 0 else f"subtract {abs(b)}"
    text = f"I think of a number, multiply it by {a}, then {b_phrase}. The result is {c}. What was the number?"
    # Misconceptions: forgot to undo the add/subtract step; divided the
    # result by a without removing b first; undid the steps in reverse (added
    # b back instead of subtracting it).
    cands = [Rational(c, a), c - b if b >= 0 else c + b, c + b]
    return dict(
        text=text,
        latex=text,
        answer=x0,
        answer_text=str(x0),
        answer_latex=str(x0),
        working=[f"{a}x {'+' if b >= 0 else '-'} {abs(b)} = {c}", f"x = {x0}"],
        distractors=pick_distractors(str(x0), [rational_text(c_) for c_ in cands]),
    )


def _word_problem_linear_eq_reverse(rng, coef_hi, sol_hi):
    # A genuine order-of-operations trap, not a range change. The narrative is read left-to-right in
    # construction order ("increased... then multiplied"), but undoing it correctly means divide by
    # `a` FIRST, then subtract `b` -- the OPPOSITE order from the `kind="linear_eq"` shape above,
    # whose narrative order ("multiply, then add/subtract") happens to already match its own undo
    # order (subtract-then-divide). `b` is drawn NON-negative only here (a single "increased by"
    # phrasing, not `linear_eq`'s bidirectional add/subtract) so this hard tier isolates the
    # reverse-order trap on its own instead of compounding it with a sign trap too (a judgement call, documented here).
    x0 = rng.randint(1, sol_hi)
    a = rng.randint(2, coef_hi)
    b = rng.randint(1, 20)
    c = a * (x0 + b)
    # Construct backwards from x0, exactly like the kind="linear_eq" shape
    # above: solve() from scratch is the primary check.
    check(solve(Eq(a * (x + b), c), x) == [x0])
    # Independent check, a different computational route (substitution
    # rather than solving): x0 plugged back into the unsolved bracket form
    # must reproduce c.
    check_equivalent(a * (x0 + b), c, "reverse word-problem answer does not satisfy a(x+b)=c")
    # The first is the canonical phrasing.
    templates = [
        f"A number is increased by {b}, then the result is multiplied by {a}, giving {c}. What was the number?",
        f"Take a number, increase it by {b}, then multiply the result by {a}. The answer is {c}. What was the number?",
        f"First, a number is increased by {b}. Multiplying that result by {a} gives {c}. What was the number?",
    ]
    text = rng.choice(templates)
    # Misconceptions:
    #  - divided by a correctly (reaching x0 + b) but then undid the steps in
    #    the SAME order as construction, adding b back instead of subtracting;
    #  - subtracted b before dividing, applying last-operation-first as if
    #    this were the standard ax+b=c shape, ignoring that the bracket
    #    changes the required order;
    #  - forgot the bracket/multiplier entirely and computed c - b as the
    #    whole answer.
    added_back = Rational(x0 + 2 * b)
    subtract_first = Rational(c - b, a)
    forgot_bracket = Rational(c - b)
    check_distinct(x0, added_back, subtract_first, forgot_bracket)
    cands = [added_back, subtract_first, forgot_bracket]
    return dict(
        text=text,
        latex=text,
        answer=x0,
        answer_text=str(x0),
        answer_latex=str(x0),
        working=[f"{a}(x + {b}) = {c}", f"x + {b} = {c // a if c % a == 0 else Rational(c, a)}", f"x = {x0}"],
        distractors=pick_distractors(str(x0), [rational_text(c_) for c_ in cands]),
    )


def _coordinates_fourth_vertex(rng, hi):
    """Three corners of an axis-parallel rectangle are given; write the fourth.
    The answer's x (y) is the x (y) value that appears once among the three."""
    while True:
        x1, x2 = rng.randint(-hi, hi), rng.randint(-hi, hi)
        y1, y2 = rng.randint(-hi, hi), rng.randint(-hi, hi)
        if x1 == x2 or y1 == y2:
            continue
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        missing = rng.randrange(4)
        ax, ay = corners[missing]
        given = [c for i, c in enumerate(corners) if i != missing]
        rng.shuffle(given)
        diagonal = corners[(missing + 2) % 4]
        ans_text = f"({ax}, {ay})"
        cands = [
            f"({diagonal[0]}, {diagonal[1]})",  # copied the diagonal vertex
            f"({ay}, {ax})",  # swapped x and y
            f"({-ax}, {-ay})",  # reversed both signs
        ]
        distractors = pick_distractors(ans_text, cands)
        if len(distractors) == 3:
            break
    listed = [f"{label}({px}, {py})" for label, (px, py) in zip("ABC", given)]
    text = (
        f"Three vertices of a rectangle with sides parallel to the axes are "
        f"{listed[0]}, {listed[1]} and {listed[2]}. "
        f"Write the coordinates (x, y) of the fourth vertex D."
    )
    return dict(
        text=text,
        latex=text,
        answer={"x": ax, "y": ay},
        answer_text=ans_text,
        answer_latex=ans_text,
        distractors=distractors,
    )


@generator(
    "coordinates",
    level=7,
    presets={
        "easy": dict(hi=5),
        "medium": dict(hi=9),
        "hard": dict(hi=15),
    },
)
def coordinates(rng, kind="read", hi=9):
    if kind == "fourth_vertex":
        return _coordinates_fourth_vertex(rng, hi)
    if hi < 1:
        raise ValueError("coordinates needs hi >= 1: the point must not be the origin, so some coordinate is nonzero")
    # Redraw until the point is not the origin (a point "0 units from the origin" has no direction words).
    px, py = draw_until(
        rng,
        lambda r: (r.randint(-hi, hi), r.randint(-hi, hi)),
        lambda point: point != (0, 0),
        what="a point other than the origin",
    )
    h_word = f"{abs(px)} {'unit' if abs(px) == 1 else 'units'} {'right' if px >= 0 else 'left'}"
    v_word = f"{abs(py)} {'unit' if abs(py) == 1 else 'units'} {'up' if py >= 0 else 'down'}"
    text = f"A point is {h_word} and {v_word} from the origin. Write its coordinates (x, y)."
    ans_text = f"({px}, {py})"
    # Misconceptions: swapped x and y; flipped the x-sign; flipped the y-sign.
    cands = [f"({py}, {px})", f"({-px}, {py})", f"({px}, {-py})"]
    return dict(
        text=text,
        latex=text,
        answer={"x": px, "y": py},
        answer_text=ans_text,
        answer_latex=ans_text,
        distractors=pick_distractors(ans_text, cands),
    )


_INEQ_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}


def _inequality_holds(a, b, c, op, v):
    lhs = a * v + b
    return {"<": lhs < c, ">": lhs > c, "<=": lhs <= c, ">=": lhs >= c}[op]


@generator(
    "inequality",
    level=7,
    presets={
        "easy": dict(kind="one_step", hi=9),
        "medium": dict(kind="two_step", hi=9),
        "hard": dict(kind="neg_coef", hi=9),
    },
)
def inequality(rng, kind="one_step", hi=9):
    op = rng.choice(["<", ">", "<=", ">="])
    if kind == "one_step":
        a, b = 1, rng.randint(-hi, hi)
    elif kind == "neg_coef":
        a, b = rng.randint(1, hi) * -1, rng.randint(-hi, hi)
    else:  # two_step
        a, b = rng.randint(2, hi), rng.randint(-hi, hi)
    c = rng.randint(-hi, hi)
    bound = Rational(c - b, a)
    ans_op = op if a > 0 else _INEQ_FLIP[op]
    # Independent check: nudge x from the bound on each side and confirm the
    # *original* (unsolved) inequality holds only on the claimed side.
    eps = Rational(1, 100)
    inside = bound + eps if ans_op in (">", ">=") else bound - eps
    outside = bound - eps if ans_op in (">", ">=") else bound + eps
    check(bool(_inequality_holds(a, b, c, op, inside)) and not bool(_inequality_holds(a, b, c, op, outside)))
    bound_text = rational_text(bound)
    ans_text = f"x {ans_op} {bound_text}"
    sign = "+" if b >= 0 else "-"
    lhs_text = fmt_signed(a, "x")
    text = f"Solve {lhs_text} {sign} {abs(b)} {op} {c}" if b != 0 else f"Solve {lhs_text} {op} {c}"
    # Misconceptions: got the direction of the inequality backwards (a mix-up
    # independent of any sign-of-a reasoning); forgot to isolate x at all
    # (quoted the unsolved bound c); sign error moving b to the other side;
    # off-by-one on the bound.
    flipped_dir_text = f"x {_INEQ_FLIP[ans_op]} {bound_text}"
    unsolved_text = f"x {op} {c}"
    wrong_bound = rational_text(Rational(c + b, a))
    sign_err_text = f"x {ans_op} {wrong_bound}"
    off_by_one_text = f"x {ans_op} {rational_text(bound + 1)}"
    cands = [flipped_dir_text, unsolved_text, sign_err_text, off_by_one_text]
    return dict(
        text=text,
        latex=text,
        answer=ans_text,
        answer_text=ans_text,
        answer_latex=ans_text,
        distractors=pick_distractors(ans_text, cands),
    )


# ---------------------------------------------------------------------------
# rearrange: make a named letter the subject of a small, fixed bank of real
# formulas. Symbols are single letters distinct from the module's `x`/`y` so
# an "expression" answer like "(v - u)/t" exercises the grader on letters
# other than x/y.
# Each formula: display text, TeX, lhs and rhs as expression strings, and the number of algebraic "steps" needed to
# isolate each rhs-variable as subject (1 = a single division/multiplication or one addition; 2 = an additive move
# then a division, or two multiplicative moves). Difficulty presets select by step count; `hard` draws from the
# squared formulas below, each solved for a letter that appears to the first power. Every target appears in the
# right-hand side as a polynomial of degree 1, which the distractor construction relies on.
_REARRANGE_BANK = (
    # (text, tex, lhs, rhs, {target: steps})
    ("A = lw", "A = lw", "A", "l*w", {"l": 1, "w": 1}),
    ("P = 2l + 2w", "P = 2l + 2w", "P", "2*l + 2*w", {"l": 2, "w": 2}),
    ("C = 2πr", r"C = 2\pi r", "C", "2*pi*r", {"r": 1}),
    ("F = ma", "F = ma", "F", "m*a", {"m": 1, "a": 1}),
    ("v = u + at", "v = u + at", "v", "u + a*t", {"u": 1, "a": 2, "t": 2}),
    ("s = vt", "s = vt", "s", "v*t", {"v": 1, "t": 1}),
    ("V = lwh", "V = lwh", "V", "l*w*h", {"l": 1, "w": 1, "h": 1}),
    ("P = IV", "P = IV", "P", "I*V", {"I": 1, "V": 1}),
    ("V = IR", "V = IR", "V", "I*R", {"I": 1, "R": 1}),
    ("Q = It", "Q = It", "Q", "I*t", {"I": 1, "t": 1}),
    ("W = Fd", "W = Fd", "W", "F*d", {"F": 1, "d": 1}),
    ("p = mv", "p = mv", "p", "m*v", {"m": 1, "v": 1}),
    ("E = Pt", "E = Pt", "E", "P*t", {"P": 1, "t": 1}),
    ("m = DV", "m = DV", "m", "D*V", {"D": 1, "V": 1}),
    ("C = πd", r"C = \pi d", "C", "pi*d", {"d": 1}),
    ("D = 2r", "D = 2r", "D", "2*r", {"r": 1}),
    ("A = bh", "A = bh", "A", "b*h", {"b": 1, "h": 1}),
    ("P = R - C", "P = R - C", "P", "R - C", {"R": 1, "C": 1}),
    ("T = a + b + c", "T = a + b + c", "T", "a + b + c", {"a": 1, "b": 1, "c": 1}),
    ("d = st", "d = st", "d", "s*t", {"s": 1, "t": 1}),
    ("C = F + rt", "C = F + rt", "C", "F + r*t", {"F": 1, "r": 2, "t": 2}),
    ("T = nP", "T = nP", "T", "n*P", {"n": 1, "P": 1}),
    ("A = ½bh", "A = \\frac{1}{2}bh", "A", "b*h/2", {"b": 2, "h": 2}),
    ("y = mx + c", "y = mx + c", "y", "m*x + c", {"m": 2, "x": 2, "c": 1}),
    ("F = 9C/5 + 32", "F = \\frac{9C}{5} + 32", "F", "9*C/5 + 32", {"C": 2}),
    ("C = 5(F - 32)/9", "C = \\frac{5(F - 32)}{9}", "C", "5*(F - 32)/9", {"F": 2}),
    ("M = (a + b)/2", "M = \\frac{a + b}{2}", "M", "(a + b)/2", {"a": 2, "b": 2}),
    ("I = Prt/100", "I = \\frac{Prt}{100}", "I", "P*r*t/100", {"P": 2, "r": 2, "t": 2}),
    ("P = 2a + b", "P = 2a + b", "P", "2*a + b", {"a": 2, "b": 1}),
    ("C = 40 + 15h", "C = 40 + 15h", "C", "40 + 15*h", {"h": 2}),
    ("P = nR - C", "P = nR - C", "P", "n*R - C", {"n": 2, "R": 2, "C": 1}),
    ("A = 2πrh", "A = 2\\pi rh", "A", "2*pi*r*h", {"r": 2, "h": 2}),
    ("V = ⅓Ah", "V = \\frac{1}{3}Ah", "V", "A*h/3", {"A": 2, "h": 2}),
    ("s = ½(a + b + c)", "s = \\frac{a + b + c}{2}", "s", "(a + b + c)/2", {"a": 2, "b": 2, "c": 2}),
    ("d = s + vt", "d = s + vt", "d", "s + v*t", {"s": 1, "v": 2, "t": 2}),
    ("P = A + Dgh", "P = A + Dgh", "P", "A + D*g*h", {"A": 1, "D": 2, "g": 2, "h": 2}),
    ("y = ax + b", "y = ax + b", "y", "a*x + b", {"a": 2, "x": 2, "b": 1}),
    ("T = pn + f", "T = pn + f", "T", "p*n + f", {"p": 2, "n": 2, "f": 1}),
)
_REARRANGE_HARD_BANK = (
    ("v² = u² + 2as", "v^2 = u^2 + 2as", "v**2", "u**2 + 2*a*s", {"a": 2, "s": 2}),
    ("E = ½mv²", "E = \\frac{1}{2}mv^2", "E", "m*v**2/2", {"m": 2}),
    ("K = ½mv²", "K = \\frac{1}{2}mv^2", "K", "m*v**2/2", {"m": 2}),
    ("U = ½kx²", "U = \\frac{1}{2}kx^2", "U", "k*x**2/2", {"k": 2}),
    ("s = ut + ½at²", "s = ut + \\frac{1}{2}at^2", "s", "u*t + a*t**2/2", {"u": 2, "a": 2}),
    ("h = ut - ½gt²", "h = ut - \\frac{1}{2}gt^2", "h", "u*t - g*t**2/2", {"u": 2, "g": 2}),
    ("P = I²R", "P = I^2R", "P", "I**2*R", {"R": 2}),
    ("F = mv²/r", "F = \\frac{mv^2}{r}", "F", "m*v**2/r", {"m": 2}),
    ("V = πr²h", "V = \\pi r^2h", "V", "pi*r**2*h", {"h": 2}),
    ("V = ⅓πr²h", "V = \\frac{1}{3}\\pi r^2h", "V", "pi*r**2*h/3", {"h": 2}),
    ("A = 2πr² + 2πrh", "A = 2\\pi r^2 + 2\\pi rh", "A", "2*pi*r**2 + 2*pi*r*h", {"h": 2}),
    ("A = πr² + πrl", "A = \\pi r^2 + \\pi rl", "A", "pi*r**2 + pi*r*l", {"l": 2}),
    ("A = πr² + πrs", "A = \\pi r^2 + \\pi rs", "A", "pi*r**2 + pi*r*s", {"s": 2}),
    ("E = ½mv² + mgh", "E = \\frac{1}{2}mv^2 + mgh", "E", "m*v**2/2 + m*g*h", {"m": 2, "g": 2, "h": 2}),
    ("F = Gmn/r²", "F = \\frac{Gmn}{r^2}", "F", "G*m*n/r**2", {"G": 2, "m": 2, "n": 2}),
    ("d = ½gt²", "d = \\frac{1}{2}gt^2", "d", "g*t**2/2", {"g": 2}),
    ("E = mc²", "E = mc^2", "E", "m*c**2", {"m": 2}),
    ("F = mw²r", "F = mw^2r", "F", "m*w**2*r", {"m": 2, "r": 2}),
    ("P = p + ½Dv²", "P = p + \\frac{1}{2}Dv^2", "P", "p + D*v**2/2", {"p": 2, "D": 2}),
    ("y = ax² + c", "y = ax^2 + c", "y", "a*x**2 + c", {"a": 2, "c": 2}),
    ("A = P(1 + r)²", "A = P(1 + r)^2", "A", "P*(1 + r)**2", {"P": 2}),
    ("W = ½mv² - ½mu²", "W = \\frac{1}{2}mv^2 - \\frac{1}{2}mu^2", "W", "m*v**2/2 - m*u**2/2", {"m": 2}),
    ("s = vt - ½at²", "s = vt - \\frac{1}{2}at^2", "s", "v*t - a*t**2/2", {"v": 2, "a": 2}),
    ("E = ½CV²", "E = \\frac{1}{2}CV^2", "E", "C*V**2/2", {"C": 2}),
    ("I = ½mr²", "I = \\frac{1}{2}mr^2", "I", "m*r**2/2", {"m": 2}),
)
_BARE_LETTER = re.compile(r"\b[A-Za-z]\b")
_LETTER_RUN = re.compile(r"(\\[A-Za-z]+)|([A-Za-z]{3,})")


def _rearrange_formula(entry):
    text, tex, lhs_src, rhs_src, steps = entry
    letters = {ch: Symbol(ch) for ch in _BARE_LETTER.findall(lhs_src + " " + rhs_src)}
    # A run of three variable letters ("mgh") would read as a word to the prompt checks; thin spaces keep them apart.
    tex = _LETTER_RUN.sub(lambda m: m.group(1) or r"\,".join(m.group(2)), tex)
    return dict(
        text=text,
        tex=tex,
        lhs=sympify(lhs_src, locals=letters),
        rhs=sympify(rhs_src, locals=letters),
        steps=steps,
        symbols=letters,
    )


_REARRANGE_FORMULAS = [_rearrange_formula(e) for e in _REARRANGE_BANK]
_REARRANGE_HARD = [_rearrange_formula(e) for e in _REARRANGE_HARD_BANK]
# One (formula, target) pair per row, so a seed draws uniformly over every question the tier can ask.
_REARRANGE_EASY = [(f, t) for f in _REARRANGE_FORMULAS for t, n in f["steps"].items() if n == 1]
_REARRANGE_MEDIUM = [(f, t) for f in _REARRANGE_FORMULAS for t, n in f["steps"].items() if n == 2]
_REARRANGE_HARD_POOL = [(f, t) for f in _REARRANGE_HARD for t in f["steps"]]


_REARRANGE_SOLVED: dict = {}


def _rearrange_solved(text, target_name, lhs, rhs, target):
    """Solved form, answer text, LaTeX and misconception candidates for one (formula, target), built by sympy once
    and reused: solving and simplifying are the slow part of a draw."""
    key = (text, target_name)
    if key not in _REARRANGE_SOLVED:
        solutions = solve(Eq(lhs, rhs), target)
        check(len(solutions) == 1)
        ans_expr = solutions[0]
        # Misconceptions, from the coefficient/remainder decomposition of rhs as a polynomial in the target: forgot
        # to divide by the coefficient at the end; sign error moving the additive remainder across the equals sign;
        # divided the wrong way round (inverted numerator/denominator).
        poly = Poly(rhs, target)
        coeff, remainder = poly.nth(1), poly.nth(0)
        cands = []
        if coeff != 1:
            cands.append(str(together(lhs - remainder)))
        if remainder != 0:
            cands.append(str(together((lhs + remainder) / coeff)))
        if not symbolically_equal(lhs, remainder):
            cands.append(str(together(coeff / (lhs - remainder))))
        _REARRANGE_SOLVED[key] = (ans_expr, str(ans_expr), latex(ans_expr), tuple(cands))
    return _REARRANGE_SOLVED[key]


@generator(
    "rearrange",
    level=7,
    presets={
        "easy": dict(n_steps=1),
        "medium": dict(n_steps=2),
        "hard": dict(n_steps=2, hard=True),
    },
)
def rearrange(rng, n_steps=1, hard=False):
    if hard:
        formula, target_name = rng.choice(_REARRANGE_HARD_POOL)
    else:
        formula, target_name = rng.choice(_REARRANGE_EASY if n_steps == 1 else _REARRANGE_MEDIUM)
    target = formula["symbols"][target_name]
    lhs, rhs = formula["lhs"], formula["rhs"]
    ans_expr, ans_str, ans_latex, cands = _rearrange_solved(formula["text"], target_name, lhs, rhs, target)
    # Independent check: substitute the derived expression back into the
    # ORIGINAL (unsolved) equation. Algebraic substitution + simplification
    # is a different computational route from solve()'s equation-solving.
    check_equivalent(lhs.subs(target, ans_expr), rhs.subs(target, ans_expr))

    text = f"Make {target_name} the subject of the formula {formula['text']}"
    return dict(
        text=text,
        latex=formula["tex"],
        answer=ans_str,
        answer_text=ans_str,
        answer_latex=ans_latex,
        working=[f"Start from {formula['text']}", f"{target_name} = {ans_str}"],
        distractors=pick_distractors(ans_str, list(cands)),
    )


# ---------------------------------------------------------------------------
# factor_gcf: pick coprime inner coefficients and a numeric (and, on hard, a
# variable) common factor, then EXPAND to build the question -- the answer is
# the factored form we started from. The `expression` grader accepts any
# equivalent expression by default, which would wrongly accept the unfactored
# question itself as "correct"; grading.grade_expression therefore requires a
# "(" in the learner's input whenever the canonical answer_text has one (see
# mathgen/grading/typed.py), so an unfactored answer is rejected here specifically
# because factor_gcf's answer_text always contains a bracket.
def _coprime_coeffs(rng, n, hi):
    while True:
        coeffs = [rng.randint(1, hi)] + [rng.randint(1, hi) * rng.choice([1, -1]) for _ in range(n - 1)]
        g = 0
        for c in coeffs:
            g = math.gcd(g, abs(c))
        if g == 1:
            return coeffs


@generator(
    "factor_gcf",
    level=7,
    presets={
        "easy": dict(hi=9, n_terms=2, var_factor=False),
        "medium": dict(hi=15, n_terms=3, var_factor=False),
        "hard": dict(hi=9, n_terms=2, var_factor=True),
    },
)
def factor_gcf(rng, hi=9, n_terms=2, var_factor=False):
    g = rng.randint(2, hi)
    inner_coeffs = _coprime_coeffs(rng, n_terms, hi)
    if var_factor:
        # Factor a common g*x out of a 2-term expression:
        # g*c1*x**2 + g*c2*x = g*x*(c1*x + c2)
        powers = [1, 0]
        prefix = f"{g}*x"
        outer = g * x
    else:
        powers = list(range(n_terms - 1, -1, -1))
        prefix = str(g)
        outer = g
    inner_expr = sum(c * x**p for c, p in zip(inner_coeffs, powers))
    expr = expand(outer * inner_expr)
    # Independent check: recompute the numeric GCF from the EXPANDED
    # expression's own coefficients (not from inner_coeffs) via a plain
    # integer-gcd sweep, and (for var_factor) confirm every term really is
    # divisible by x (no leftover constant term).
    ig = 0
    for c in Poly(expr, x).all_coeffs():
        ig = math.gcd(ig, abs(int(c)))
    check(ig == g)
    if var_factor:
        check(expr.coeff(x, 0) == 0)

    ans_str = f"{prefix}*({inner_expr})"
    expr_text = str(expr)
    # Misconceptions: divided only the first term by the common factor (left
    # the rest un-divided); sign error on the last bracket term; confused
    # factorising with simply restating the (unfactored) expression.
    undivided_rest = [c * g if i > 0 else c for i, c in enumerate(inner_coeffs)]
    divide_first_only_expr = sum(c * x**p for c, p in zip(undivided_rest, powers))
    divide_first_only = f"{prefix}*({divide_first_only_expr})"
    sign_flip_coeffs = list(inner_coeffs)
    sign_flip_coeffs[-1] = -sign_flip_coeffs[-1]
    sign_flip_expr = sum(c * x**p for c, p in zip(sign_flip_coeffs, powers))
    sign_flip = f"{prefix}*({sign_flip_expr})"
    cands = [divide_first_only, sign_flip, expr_text]

    return dict(
        text=f"Factor out the common factor: {expr_text}",
        latex=latex(expr),
        answer=ans_str,
        answer_text=ans_str,
        answer_latex=rf"{prefix}\left({latex(inner_expr)}\right)",
        working=[f"Common factor: {prefix}", f"{expr_text} = {ans_str}"],
        distractors=pick_distractors(ans_str, cands),
        answer_form="factored",
    )


# ---------------------------------------------------------------------------
# interpret_notation: substitute given integer values into a fixed piece of
# conventional notation (implied multiplication, powers, brackets, a
# fraction, and the classic "(2x)^2 vs 2x^2" contrast) and evaluate.
_NOTATION_KINDS = ("mult", "square", "distribute", "divide", "product", "bracket_sq", "coef_sq")


@generator(
    "interpret_notation",
    level=7,
    presets={
        "easy": dict(kinds=("mult", "square", "divide", "product"), hi=9),
        "medium": dict(kinds=("mult", "square", "distribute", "divide", "product"), hi=9),
        "hard": dict(kinds=("distribute", "bracket_sq", "coef_sq", "product"), hi=12),
    },
)
def interpret_notation(rng, kinds=_NOTATION_KINDS, hi=9):
    kind = rng.choice(list(kinds))
    yv = None
    if kind == "product":
        xv = rng.randint(-hi, hi) or 1
        yv = rng.randint(-hi, hi) or 1
        expr = x * y
        ans = xv * yv
        text = f"If x = {xv} and y = {yv}, what is xy?"
        latex_q = rf"x = {xv},\ y = {yv}:\quad xy"
        # Misconception: added instead of multiplied.
        cands = [str(xv + yv), str(xv - yv), str(-ans)]
    elif kind == "divide":
        xv = (rng.randint(1, max(1, hi // 4)) * 4) * rng.choice([1, -1])
        expr = x / 4
        ans = xv // 4  # xv is always an exact multiple of 4 by construction
        text = f"If x = {xv}, what is x/4?"
        latex_q = rf"x = {xv}:\quad \dfrac{{x}}{{4}}"
        # Misconceptions: inverted the division (computed 4/x); subtracted 4
        # instead of dividing by it.
        cands = [rational_text(Rational(4, xv)), str(xv - 4), str(-ans)]
    elif kind == "mult":
        xv = rng.randint(-hi, hi) or 1
        expr = 3 * x
        ans = 3 * xv
        text = f"If x = {xv}, what is 3x?"
        latex_q = rf"x = {xv}:\quad 3x"
        # Misconceptions: read "3x" as "3 + x"; confused with "x cubed".
        cands = [str(3 + xv), str(xv**3), str(-ans)]
    elif kind == "square":
        xv = rng.randint(-hi, hi) or 1
        expr = x**2
        ans = xv**2
        text = f"If x = {xv}, what is x²?"
        latex_q = rf"x = {xv}:\quad x^2"
        # Misconceptions: read "squared" as "times 2"; added 2 instead.
        cands = [str(2 * xv), str(xv + 2), str(-ans)]
    elif kind == "distribute":
        xv = rng.randint(-hi, hi) or 1
        expr = 2 * (x + 1)
        ans = 2 * (xv + 1)
        text = f"If x = {xv}, what is 2(x + 1)?"
        latex_q = rf"x = {xv}:\quad 2(x + 1)"
        # Misconceptions: distributed onto the x-term only, not the "+1";
        # dropped the multiplier entirely.
        cands = [str(2 * xv + 1), str(xv + 1), str(2 * (xv - 1))]
    elif kind == "bracket_sq":
        xv = rng.randint(-hi, hi) or 1
        expr = (2 * x) ** 2
        ans = (2 * xv) ** 2
        text = f"If x = {xv}, what is (2x)²?"
        latex_q = rf"x = {xv}:\quad (2x)^2"
        # Misconception: squared only x, applying the factor of 2 afterwards
        # (computed 2x^2 instead of (2x)^2).
        cands = [str(2 * xv**2), str(4 * xv), str(-ans)]
    else:  # coef_sq
        xv = rng.randint(-hi, hi) or 1
        expr = 2 * x**2
        ans = 2 * xv**2
        text = f"If x = {xv}, what is 2x²?"
        latex_q = rf"x = {xv}:\quad 2x^2"
        # Misconception: squared the coefficient along with x (computed
        # (2x)^2 instead of 2x^2).
        cands = [str((2 * xv) ** 2), str(2 * xv), str(-ans)]
    # Independent check: substitute into the sympy expression directly
    # (rather than the hand-computed Python arithmetic above) and confirm it
    # matches.
    subs_map = {x: xv, y: yv} if yv is not None else {x: xv}
    check(int(expr.subs(subs_map)) == ans)
    ans_str = str(int(ans))
    return dict(
        text=text,
        latex=latex_q,
        answer=int(ans),
        answer_text=ans_str,
        answer_latex=ans_str,
        distractors=pick_distractors(ans_str, cands),
    )


# ---------------------------------------------------------------------------
# like_terms_select (mcq): "which of these is a like term to <query>?" The
# correct option shares the query's exact variable/power structure with a
# different coefficient; the three distractors change one variable's
# identity, change one variable's power, or drop variables entirely (a plain
# constant) -- the three classic "not actually a like term" mistakes.
_LIKE_TERM_LETTERS = ["x", "y", "z", "a", "b", "n", "m", "p", "q", "r"]


def _monomial(coef, powers):
    expr = coef
    for v, p in powers.items():
        expr = expr * symbols(v) ** p
    return expr


@generator(
    "like_terms_select",
    level=7,
    presets={
        "easy": dict(n_vars=1, pow_hi=2, coef_hi=9),
        "medium": dict(n_vars=1, pow_hi=3, coef_hi=12),
        "hard": dict(n_vars=2, pow_hi=3, coef_hi=15),
    },
)
def like_terms_select(rng, n_vars=1, pow_hi=2, coef_hi=9):
    letters = rng.sample(_LIKE_TERM_LETTERS, k=n_vars)
    powers = {v: rng.randint(1, pow_hi) for v in letters}
    q_coef = rng.randint(2, coef_hi)
    query = _monomial(q_coef, powers)

    # Correct option: same variables/powers, a different coefficient.
    like_coef = rng.randint(1, coef_hi)
    while like_coef == q_coef:
        like_coef = rng.randint(1, coef_hi)
    like_coef *= rng.choice([1, -1])
    like_term = _monomial(like_coef, powers)
    check(simplify(like_term / query).is_number)

    # Distractor: different variable (swap one letter for an unused one).
    swap_var, swap_power = rng.choice(list(powers.items()))
    unused = [l for l in _LIKE_TERM_LETTERS if l not in letters]
    new_letter = rng.choice(unused)
    diff_var_powers = {v: p for v, p in powers.items() if v != swap_var}
    diff_var_powers[new_letter] = swap_power
    diff_var_term = _monomial(rng.randint(1, coef_hi) * rng.choice([1, -1]), diff_var_powers)
    check(not simplify(diff_var_term / query).is_number)

    # Distractor: different power on one of the existing variables.
    change_var = rng.choice(letters)
    old_power = powers[change_var]
    new_power = old_power + rng.choice([1, -1]) if old_power > 1 else old_power + 1
    diff_pow_powers = {**powers, change_var: new_power}
    diff_pow_term = _monomial(rng.randint(1, coef_hi) * rng.choice([1, -1]), diff_pow_powers)
    check(not simplify(diff_pow_term / query).is_number)

    # Distractor: a plain constant, no variable part at all.
    const_term = rng.randint(1, coef_hi) * rng.choice([1, -1])

    query_str = str(query)
    like_str = str(like_term)
    cands = [str(diff_var_term), str(diff_pow_term), str(const_term)]
    return dict(
        text=f"Which of these is a like term to {query_str}?",
        latex=latex(query),
        answer=like_str,
        answer_text=like_str,
        answer_latex=latex(like_term),
        working=["A like term has exactly the same variables raised to exactly the same powers."],
        distractors=pick_distractors(like_str, cands),
    )
