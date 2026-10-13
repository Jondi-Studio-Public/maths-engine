"""Levels 1-6: addition, subtraction, multiplication, division, fractions, powers."""

from fractions import Fraction

from sympy import Rational, latex

from ..core import draw_until, generator, check, pick_distractors, rational_text, paren_negative


@generator(
    "addition",
    level=1,
    presets={
        "easy": dict(lo=1, hi=20),
        "medium": dict(lo=1, hi=99),
        "hard": dict(lo=100, hi=999),
    },
)
def addition(rng, lo=1, hi=99):
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    ans = a + b
    return dict(
        text=f"{a} + {paren_negative(b)}",
        latex=f"{a} + {paren_negative(b)}",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        # Misconceptions: carry error (+/-10), off-by-one slip.
        distractors=pick_distractors(str(ans), [ans + 10, ans - 10, ans + 1, ans - 1]),
    )


@generator(
    "subtraction",
    level=2,
    presets={
        "easy": dict(lo=1, hi=20),
        "medium": dict(lo=1, hi=99),
        "hard": dict(lo=100, hi=999, allow_negative=True),
    },
)
def subtraction(rng, lo=1, hi=99, allow_negative=False):
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    if not allow_negative and b > a:
        a, b = b, a
    ans = a - b
    # Misconceptions: borrowing error (+/-10), off-by-one; drop negatives
    # when the level doesn't allow them (they'd be a giveaway).
    cands = [ans + 10, ans - 10, ans + 1, ans - 1]
    if not allow_negative:
        cands = [c for c in cands if c >= 0]
    return dict(
        text=f"{a} - {paren_negative(b)}",
        latex=f"{a} - {paren_negative(b)}",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        distractors=pick_distractors(str(ans), cands),
    )


@generator(
    "multiplication",
    level=3,
    presets={
        "easy": dict(lo=2, hi=9, lo2=2, hi2=9),
        "medium": dict(lo=2, hi=12, lo2=2, hi2=19),
        "hard": dict(lo=11, hi=25, lo2=11, hi2=99),
    },
)
def multiplication(rng, lo=2, hi=12, lo2=2, hi2=19):
    a, b = rng.randint(lo, hi), rng.randint(lo2, hi2)
    ans = a * b
    return dict(
        text=f"{a} \u00d7 {b}",
        latex=rf"{a} \times {b}",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        # Misconceptions: one row of the times table off.
        distractors=pick_distractors(str(ans), [a * (b + 1), a * (b - 1), (a + 1) * b, (a - 1) * b]),
    )


@generator(
    "division",
    level=4,
    presets={
        "easy": dict(q_lo=2, q_hi=9, d_lo=2, d_hi=9),
        "medium": dict(q_lo=2, q_hi=12, d_lo=2, d_hi=12),
        "hard": dict(q_lo=11, q_hi=25, d_lo=11, d_hi=19),
    },
)
def division(rng, q_lo=2, q_hi=12, d_lo=2, d_hi=12):
    # Backwards: pick quotient and divisor, build the dividend.
    q, d = rng.randint(q_lo, q_hi), rng.randint(d_lo, d_hi)
    n = q * d
    return dict(
        text=f"{n} \u00f7 {d}",
        latex=rf"{n} \div {d}",
        answer=q,
        answer_text=str(q),
        answer_latex=str(q),
        # Misconceptions: off-by-one quotient; answered with the divisor.
        distractors=pick_distractors(str(q), [q + 1, q - 1, d]),
    )


@generator(
    "fractions",
    level=5,
    presets={
        "easy": dict(max_den=6),
        "medium": dict(max_den=12),
        "hard": dict(max_den=20),
    },
)
def fractions(rng, max_den=12, ops=("+", "-", "*"), three_term=False):
    # Results must stay positive. The smallest proper fractions are 1/2 (max_den 2) and 1/3 (max_den 3), so:
    # a two-term subtraction needs two different proper fractions (max_den >= 3), and a three-term
    # expression that only subtracts needs a first fraction larger than the other two added (max_den >= 4,
    # e.g. 3/4 - 1/4 - 1/3). Below those no draw can pass, and the redraw loops would never finish.
    if three_term:
        term_ops = [o for o in ("+", "-") if o in ops] or ["+", "-"]
        if term_ops == ["-"] and max_den < 4:
            raise ValueError(
                "fractions needs max_den >= 4 when three_term=True and ops only subtracts: "
                "the result must stay positive"
            )

        def draw_expression(r):
            d1, d2, d3 = (r.randint(2, max_den) for _ in range(3))
            n1, n2, n3 = r.randint(1, d1 - 1), r.randint(1, d2 - 1), r.randint(1, d3 - 1)
            op1, op2 = r.choice(term_ops), r.choice(term_ops)
            f1, f2, f3 = Rational(n1, d1), Rational(n2, d2), Rational(n3, d3)
            partial = f1 + f2 if op1 == "+" else f1 - f2
            ans = partial + f3 if op2 == "+" else partial - f3
            return d1, d2, d3, n1, n2, n3, op1, op2, f1, f2, f3, partial, ans

        # keep results positive at this level
        d1, d2, d3, n1, n2, n3, op1, op2, f1, f2, f3, partial, ans = draw_until(
            rng, draw_expression, lambda expr: expr[-1] > 0, what="a three-term sum with a positive result"
        )
        sym = {"+": "+", "-": "-"}
        text = f"{n1}/{d1} {sym[op1]} {n2}/{d2} {sym[op2]} {n3}/{d3}"
        ltx = rf"\frac{{{n1}}}{{{d1}}} {sym[op1]} \frac{{{n2}}}{{{d2}}} {sym[op2]} \frac{{{n3}}}{{{d3}}}"
        # Independent check: recompute with stdlib Fractions, not SymPy Rationals.
        g1, g2, g3 = Fraction(n1, d1), Fraction(n2, d2), Fraction(n3, d3)
        gpartial = g1 + g2 if op1 == "+" else g1 - g2
        gans = gpartial + g3 if op2 == "+" else gpartial - g3
        check(gans == Fraction(int(ans.p), int(ans.q)))
        # Misconceptions, formatted identically to the answer.
        cands = [
            f1 + f2 + f3,  # ignored the operators, added everything
            partial,  # stopped after the first two terms
            f3,  # answered with only the last term
        ]
        cands = [c for c in cands if c != ans]
        return dict(
            text=text,
            latex=ltx,
            answer={"p": int(ans.p), "q": int(ans.q)},
            answer_text=rational_text(ans),
            answer_latex=latex(ans),
            distractors=pick_distractors(rational_text(ans), [rational_text(c) for c in cands]),
        )
    if "-" in ops and max_den < 3:
        raise ValueError(
            "fractions needs max_den >= 3 when ops includes '-': a subtraction must stay positive, "
            "which needs two different proper fractions"
        )
    op = rng.choice(list(ops))

    def draw_pair(r):
        d1, d2 = r.randint(2, max_den), r.randint(2, max_den)
        n1, n2 = r.randint(1, d1 - 1), r.randint(1, d2 - 1)
        f1, f2 = Rational(n1, d1), Rational(n2, d2)
        ans = {"+": f1 + f2, "-": f1 - f2, "*": f1 * f2, "/": f1 / f2}[op]
        return d1, d2, n1, n2, f1, f2, ans

    # keep results positive at this level
    d1, d2, n1, n2, f1, f2, ans = draw_until(
        rng, draw_pair, lambda pair: op != "-" or pair[-1] > 0, what="a pair of fractions with a positive result"
    )
    sym = {"+": "+", "-": "-", "*": "\u00d7", "/": "\u00f7"}[op]
    lsym = {"+": "+", "-": "-", "*": r"\times", "/": r"\div"}[op]
    # Independent check: recompute with stdlib Fractions, not SymPy Rationals.
    g1, g2 = Fraction(n1, d1), Fraction(n2, d2)
    check({"+": g1 + g2, "-": g1 - g2, "*": g1 * g2, "/": g1 / g2}[op] == Fraction(int(ans.p), int(ans.q)))
    # Misconceptions, formatted identically to the answer.
    cands = []
    if op == "+":
        if d1 + d2 != 0:
            cands.append(Rational(n1 + n2, d1 + d2))  # added tops and bottoms
        cands += [
            Rational(n1 + n2, max(d1, d2)),  # added tops, kept a bottom
            f1 * f2,
        ]  # multiplied instead
    elif op == "-":
        if d1 != d2:
            cands.append(abs(Rational(n1 - n2, d1 - d2)))  # subtracted tops and bottoms
        cands += [abs(Rational(n1 - n2, max(d1, d2))), f1 + f2]
    elif op == "*":
        cands += [
            Rational(n1 * n2, d1 + d2),  # multiplied tops, added bottoms
            f1 + f2,  # added instead
            Rational(n1 * d2, d1 * n2) if n2 else None,
        ]  # divided instead
    else:  # "/": multiplied instead of dividing; inverted the wrong fraction; added instead
        cands += [
            f1 * f2,  # multiplied instead of dividing
            f2 / f1 if f1 else None,  # inverted the wrong fraction
            f1 + f2,
        ]  # added instead
    cands = [c for c in cands if c is not None]
    return dict(
        text=f"{n1}/{d1} {sym} {n2}/{d2}",
        latex=rf"\frac{{{n1}}}{{{d1}}} {lsym} \frac{{{n2}}}{{{d2}}}",
        answer={"p": int(ans.p), "q": int(ans.q)},
        answer_text=rational_text(ans),
        answer_latex=latex(ans),
        distractors=pick_distractors(rational_text(ans), [rational_text(c) for c in cands]),
    )


@generator(
    "powers",
    level=6,
    presets={
        "easy": dict(base_hi=9, exp_hi=2),
        "medium": dict(base_hi=12, exp_hi=4),
        "hard": dict(base_hi=15, exp_hi=5),
    },
)
def powers(rng, base_hi=12, exp_hi=4):
    b = rng.randint(2, base_hi)
    e = rng.randint(2, exp_hi if b <= 5 else min(exp_hi, 3))
    ans = b**e
    sup = str(e).translate(str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"))
    return dict(
        text=f"{b}{sup}",
        latex=f"{b}^{{{e}}}",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        # Misconceptions: multiplied base by exponent; one power short/long.
        distractors=pick_distractors(str(ans), [b * e, b ** (e - 1), b ** (e + 1)]),
    )
