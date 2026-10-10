"""Foundational arithmetic: order of operations, decimals, signed integer
arithmetic, ratio, and plug-into-a-formula questions. These sit "below"
levels 1-10 in difficulty terms but reuse nearby level numbers (no unique
level is required -- see README's levels table)."""

import math
from fractions import Fraction

from sympy import Rational, sympify

from ..core import generator, check, check_pairwise_distinct, pick_distractors, rational_text, join_terms
from .primary_math import _aligned


def _left_to_right(vals, ops):
    """Evaluate strictly left-to-right, ignoring precedence -- the classic
    BODMAS misconception, used as a distractor."""
    acc = vals[0]
    for v, op in zip(vals[1:], ops):
        if op == "+":
            acc += v
        elif op == "-":
            acc -= v
        else:
            acc *= v
    return acc


def _apply(a, b, op):
    return a * b if op == "*" else (a + b if op == "+" else a - b)


def _reverse_precedence_eval(vals, ops):
    """Apply +/- before * -- the opposite-of-BODMAS misconception."""
    vals, ops = list(vals), list(ops)
    i = 0
    while i < len(ops):
        if ops[i] != "*":
            v = _apply(vals[i], vals[i + 1], ops[i])
            vals[i : i + 2] = [v]
            del ops[i]
        else:
            i += 1
    acc = vals[0]
    for v in vals[1:]:
        acc *= v
    return acc


def _flip_last_additive(vals, ops, bracket_at):
    """Flip the sign of the last +/- operator -- always changes the result
    (since every value is >= 1), so it never runs dry as a distractor even
    when left-to-right/reverse-precedence happen to collide with the answer."""
    ops2 = list(ops)
    for i in range(len(ops2) - 1, -1, -1):
        if ops2[i] in ("+", "-"):
            ops2[i] = "-" if ops2[i] == "+" else "+"
            return _precedence_eval(vals, ops2, bracket_at)
    return None


def _precedence_eval(vals, ops, bracket_at):
    """Manual BODMAS evaluator (brackets, then *, then +/- left to right) --
    a route independent of sympify()'s parser, used to verify the display
    string was built correctly."""
    vals, ops = list(vals), list(ops)
    if bracket_at is not None:
        v = _apply(vals[bracket_at], vals[bracket_at + 1], ops[bracket_at])
        vals[bracket_at : bracket_at + 2] = [v]
        del ops[bracket_at]
    i = 0
    while i < len(ops):
        if ops[i] == "*":
            vals[i : i + 2] = [vals[i] * vals[i + 1]]
            del ops[i]
        else:
            i += 1
    acc = vals[0]
    for v, op in zip(vals[1:], ops):
        acc = acc + v if op == "+" else acc - v
    return acc


@generator(
    "order_of_operations",
    level=4,
    presets={
        "easy": dict(terms=3, hi=9, use_brackets=False),
        "medium": dict(terms=4, hi=12, use_brackets=False),
        "hard": dict(terms=4, hi=12, use_brackets=True),
    },
)
def order_of_operations(rng, terms=3, hi=9, use_brackets=False):
    vals = [rng.randint(1, hi) for _ in range(terms)]
    ops = [rng.choice(["+", "-", "*"]) for _ in range(terms - 1)]
    bracket_at = rng.randint(0, terms - 2) if (use_brackets and terms >= 3) else None
    tokens = [str(v) for v in vals]
    if bracket_at is not None:
        tokens[bracket_at] = f"({tokens[bracket_at]}"
        tokens[bracket_at + 1] = f"{tokens[bracket_at + 1]})"
    parts = [tokens[0]]
    for i, op in enumerate(ops):
        parts.append(op)
        parts.append(tokens[i + 1])
    expr_str = " ".join(parts)
    ans = int(sympify(expr_str))
    # Independent check: a hand-rolled precedence evaluator, not sympify's parser.
    check(_precedence_eval(vals, ops, bracket_at) == ans)
    disp = expr_str.replace("*", "×")
    # Misconceptions: evaluated strictly left-to-right (ignoring precedence);
    # applied +/- before * (opposite-of-BODMAS); summed everything regardless
    # of operator; flipped the sign of the last term; dropped the last term
    # (or, if bracketed, respected precedence but ignored the brackets).
    cands = [str(_left_to_right(vals, ops)), str(_reverse_precedence_eval(vals, ops)), str(sum(vals))]
    flip = _flip_last_additive(vals, ops, bracket_at)
    if flip is not None:
        cands.append(str(flip))
    # Arithmetic slip on the first operand -- always differs (all values >=
    # 1), so it backstops the associative all-same-operator case where every
    # other misconception above happens to collide with the real answer.
    bumped = [vals[0] + 1] + vals[1:]
    cands.append(str(_precedence_eval(bumped, ops, bracket_at)))
    if bracket_at is not None:
        cands.append(str(_precedence_eval(vals, ops, None)))
    else:
        cands.append(str(_precedence_eval(vals[:-1], ops[:-1], None)))
    return dict(
        text=f"Evaluate {disp}",
        latex=disp,
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        distractors=pick_distractors(str(ans), cands),
    )


@generator(
    "decimals",
    level=5,
    presets={
        "easy": dict(dp=1, hi=9),
        "medium": dict(dp=1, hi=20),
        "hard": dict(dp=2, hi=20),
    },
)
def decimals(rng, dp=1, hi=12, ops=("+", "-", "*")):
    op = rng.choice(list(ops))
    scale = 10**dp
    while True:
        n1 = rng.randint(1, hi * scale)
        n2 = rng.randint(1, hi * scale)
        d1, d2 = Rational(n1, scale), Rational(n2, scale)
        ans = {"+": d1 + d2, "-": d1 - d2, "*": d1 * d2, "/": d1 / d2}[op]
        if op != "-" or ans > 0:
            break
    # Independent check: recompute with stdlib Fraction, not SymPy Rational.
    g1, g2 = Fraction(n1, scale), Fraction(n2, scale)
    check({"+": g1 + g2, "-": g1 - g2, "*": g1 * g2, "/": g1 / g2}[op] == Fraction(int(ans.p), int(ans.q)))
    sym = {"+": "+", "-": "-", "*": "×", "/": "÷"}[op]
    a_disp, b_disp = f"{float(d1):g}", f"{float(d2):g}"
    ans_val = float(ans)
    # Misconceptions: misplaced the decimal point one place either way;
    # performed the "opposite" operation.
    alt_op = {"+": "-", "-": "+", "*": "/", "/": "*"}[op]
    alt_val = float({"+": d1 + d2, "-": d1 - d2, "*": d1 * d2, "/": d1 / d2}[alt_op])
    cands = [f"{ans_val * 10:g}", f"{ans_val / 10:g}", f"{alt_val:g}"]
    answer_text = f"{ans_val:g}"
    exact = Fraction(int(ans.p), int(ans.q))
    places = next((p for p in range(4) if (exact * 10**p).denominator == 1), None)
    # Show every option at the answer's own number of decimal places so the key
    # cannot be spotted by its shape. A quotient that does not terminate is
    # already shown to six significant figures; its options match that.
    shown_places = len(answer_text.partition(".")[2])
    slips = (
        exact * 10,
        exact / 10,
        Fraction(alt_val).limit_denominator(10**6),
        exact * 100,
        exact + Fraction(1, 10 ** (places if places is not None else shown_places)),
    )
    if places is not None:
        answer_text, cands = _aligned(exact, slips, exact_only=True)
    else:
        _, cands = _aligned(Fraction(answer_text), slips, shown_places)
    return dict(
        text=f"{a_disp} {sym} {b_disp}",
        latex=f"{a_disp} {sym} {b_disp}",
        answer=ans_val,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
    )


@generator(
    "integer_arithmetic",
    level=2,
    presets={
        "easy": dict(hi=9),
        "medium": dict(hi=20),
        "hard": dict(hi=50),
    },
)
def integer_arithmetic(rng, hi=20, ops=("+", "-", "*", "/")):
    op = rng.choice(list(ops))
    sym = {"+": "+", "-": "-", "*": "×", "/": "÷"}[op]
    if op == "/":
        # Backwards: pick quotient and divisor first for a clean division.
        d = rng.randint(1, hi) * rng.choice([1, -1]) or 1
        q = rng.randint(1, hi) * rng.choice([1, -1]) or 1
        n = q * d
        ans = q
        check(Fraction(n, d) == Fraction(q, 1))
        # Misconceptions: dropped a negative sign; off-by-one on the quotient.
        cands = [-ans, abs(ans), ans + 1, ans - 1]
        text = f"{n} {sym} {d}"
    else:
        a = rng.randint(-hi, hi) or 1
        b = rng.randint(-hi, hi) or 1
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        # Misconceptions: dropped the negative sign(s), working with |a|, |b|;
        # flipped the sign of the final result; off-by-one.
        pos_version = {"+": abs(a) + abs(b), "-": abs(a) - abs(b), "*": abs(a) * abs(b)}[op]
        cands = [pos_version, -ans, ans + 1, ans - 1]
        if op == "*":
            text = f"{a} {sym} {b}"
        else:
            # "+"/"-" are shown as a signed sum so a negative b never
            # collides with the operator symbol ("17 + -14" / "17 - -14").
            b_eff = b if op == "+" else -b
            text = join_terms([(a, ""), (b_eff, "")])
    return dict(
        text=text,
        latex=text,
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        distractors=pick_distractors(str(ans), [str(c) for c in cands]),
    )


_RATIO_UNIT_PAIRS = (("m", "cm", 100), ("kg", "g", 1000), ("km", "m", 1000), ("hours", "minutes", 60))


@generator(
    "ratio",
    level=5,
    presets={
        "easy": dict(kind="simplify", hi=20),
        "medium": dict(kind="share", hi=60),
        "hard": dict(kind="unit_rate", hi=100),
    },
)
def ratio(rng, kind="simplify", hi=20):
    if kind == "mixed_units":
        # Branch before any draw the other kinds make (EXD-001).
        big, small, f = rng.choice(_RATIO_UNIT_PAIRS)
        p, q = rng.randint(1, 9), rng.randint(1, 9)
        while p == q or math.gcd(p, q) != 1:
            p, q = rng.randint(1, 9), rng.randint(1, 9)
        k = rng.randint(1, 3)
        a, b = p * k, q * k * f
        check(Fraction(a * f, b) == Fraction(p, q))
        ans_text = f"{p}:{q}"

        def _reduced(x, y):
            g = math.gcd(x, y)
            return f"{x // g}:{y // g}"

        # Misconceptions: ignored the units and reduced as written; reversed
        # the ratio; scaled the big amount by the factor a second time.
        cands = [_reduced(a, b), f"{q}:{p}", _reduced(a * f * f, b)]
        vals = [Fraction(int(c.split(":")[0]), int(c.split(":")[1])) for c in cands]
        check_pairwise_distinct(Fraction(p, q), *vals)
        return dict(
            text=f"Write the ratio {a} {big} to {b} {small} in simplest form.",
            latex=f"{a}\\text{{ {big} }}:{b}\\text{{ {small} }}",
            answer={"p": p, "q": q},
            answer_text=ans_text,
            answer_latex=ans_text,
            working=[
                f"{a} {big} = {a * f} {small}",
                f"{a * f}:{b} = {p}:{q} (divide both by {k * f})",
            ],
            distractors=pick_distractors(ans_text, cands),
        )
    if kind == "simplify":
        p, q = rng.randint(1, hi), rng.randint(1, hi)
        while math.gcd(p, q) != 1:
            p, q = rng.randint(1, hi), rng.randint(1, hi)
        g = rng.randint(2, 9)
        a, b = p * g, q * g
        check(Fraction(a, b) == Fraction(p, q))
        ans_text = f"{p}:{q}"
        cands = [f"{q}:{p}", f"{a}:{b}", f"{p + 1}:{q}"]
        return dict(
            text=f"Simplify the ratio {a}:{b}",
            latex=f"{a}:{b}",
            answer={"p": p, "q": q},
            answer_text=ans_text,
            answer_latex=ans_text,
            distractors=pick_distractors(ans_text, cands),
        )
    if kind == "share":
        p, q = rng.randint(1, 9), rng.randint(1, 9)
        unit = rng.randint(2, hi)
        total = (p + q) * unit
        share_a = p * unit
        check(share_a + q * unit == total)
        # Misconceptions: gave the other party's share; gave the whole total;
        # split it evenly instead of by ratio.
        cands = [q * unit, total, total // 2]
        return dict(
            text=f"Share {total} in the ratio {p}:{q}. Find the first share.",
            latex=rf"{total} \text{{ in ratio }} {p}:{q}",
            answer=share_a,
            answer_text=str(share_a),
            answer_latex=str(share_a),
            working=[f"1 part = {total}/({p}+{q}) = {unit}", f"First share = {p} × {unit} = {share_a}"],
            distractors=pick_distractors(str(share_a), [str(c) for c in cands]),
        )
    # unit_rate
    rate = rng.randint(2, hi)
    units = rng.randint(2, 12)
    cost = rate * units
    check(Fraction(cost, units) == rate)
    # Misconceptions: gave the number of units instead of the rate; gave the
    # total cost; off-by-one on the rate.
    cands = [units, cost, rate + 1]
    return dict(
        text=f"{units} items cost {cost} in total. Find the cost per item (unit rate).",
        latex=f"{cost} / {units}",
        answer=rate,
        answer_text=str(rate),
        answer_latex=str(rate),
        working=[f"rate = {cost} / {units} = {rate}"],
        distractors=pick_distractors(str(rate), [str(c) for c in cands]),
    )


_FORMULAS = ("rectangle_area", "triangle_area", "speed", "simple_interest", "perimeter")


def _formula_rearrange(rng, hi, kind):
    """Give the formula's result and all but one input; the learner rearranges
    to find the missing input. Built forward from integers so the answer is whole."""
    if kind == "rectangle_area":
        l, w = rng.randint(2, hi), rng.randint(2, hi)
        A = l * w
        text = f"A rectangle has area {A} and length {l}. Use A = l × w to find its width."
        formula, ans = "A = l × w", w
        # multiplied instead of dividing; added; subtracted
        cands = [A * l, A + l, A - l]
    elif kind == "perimeter":
        l, w = rng.randint(2, hi), rng.randint(2, hi)
        P = 2 * (l + w)
        text = f"A rectangle has perimeter {P} and length {l}. Use P = 2(l + w) to find its width."
        formula, ans = "P = 2(l + w)", w
        # subtracted the length once without halving; halved but forgot to subtract; forgot to halve
        cands = [P - l, Rational(P, 2), P - 2 * l]
    elif kind == "triangle_area":
        b = rng.randrange(2, hi, 2) or 2
        h = rng.randint(2, hi)
        A = b * h // 2
        text = f"A triangle has area {A} and base {b}. Use A = ½bh to find its height."
        formula, ans = "A = ½bh", h
        # multiplied instead of dividing; divided without doubling; added
        cands = [A * b, Rational(A, b), A + b]
    elif kind == "speed":
        t = rng.randint(2, 10)
        s = rng.randint(2, max(2, hi // t))
        text = f"A car travels at {s} km/h for {t} hours. Use speed = distance / time to find the distance it travels (km)."
        formula, ans = "s = d / t", s * t
        # divided as if finding speed; added the given values
        cands = [Rational(s, t), s + t, s * t * t]
    else:  # simple_interest
        P = rng.choice([100, 200, 500, 1000])
        R = rng.choice([2, 3, 4, 5, 10])
        T = rng.randint(1, 5)
        I = P * R * T // 100
        check(Rational(P * R * T, 100) == I)
        text = (
            f"Simple interest of £{I} is earned on £{P} over {T} years. Use I = PRT/100 to find the annual rate R (%)."
        )
        formula, ans = "I = PRT/100", R
        # substituted into the formula as if the unknown were the result;
        # forgot the 100; forgot to divide by the time
        cands = [Rational(P * T * I, 100), Rational(I, P * T), Rational(100 * I, P)]
    # generic off-by-one fillers guarantee three distractors after collisions
    cands = cands + [ans + 1, ans - 1, ans + 2]
    # whole-number answer, so keep whole-number distractors only (the skill
    # contract rejects a decimal distractor beside an integer key)
    texts = [str(int(c)) for c in cands if c == int(c)]
    distractors = pick_distractors(str(ans), texts)
    check(len(distractors) == 3 and str(ans) not in distractors)
    return dict(
        text=text,
        latex=formula,
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        distractors=distractors,
    )


@generator(
    "formula_plugin",
    level=10,
    presets={
        "easy": dict(hi=12),
        "medium": dict(hi=20),
        "hard": dict(hi=30),
    },
)
def formula_plugin(rng, hi=20, formulas=_FORMULAS, unknown="result"):
    kind = rng.choice(list(formulas))
    if unknown == "input":
        return _formula_rearrange(rng, hi, kind)
    if kind == "rectangle_area":
        l, w = rng.randint(2, hi), rng.randint(2, hi)
        ans = l * w
        text = f"A rectangle has length {l} and width {w}. Use A = l × w to find its area."
        formula, cands = "A = l × w", [2 * (l + w), l + w]
    elif kind == "triangle_area":
        b = rng.randrange(2, hi, 2) or 2
        h = rng.randint(2, hi)
        ans = Rational(b * h, 2)
        text = f"A triangle has base {b} and height {h}. Use A = ½bh to find its area."
        formula, cands = "A = ½bh", [b * h, b + h]
    elif kind == "speed":
        t = rng.randint(2, 10)
        d = t * rng.randint(2, max(2, hi // t))
        ans = d // t
        text = f"A car travels {d} km in {t} hours. Use speed = distance / time to find its speed (km/h)."
        formula, cands = "s = d / t", [d * t, t]
    elif kind == "simple_interest":
        P = rng.choice([100, 200, 500, 1000])
        R = rng.choice([2, 3, 4, 5, 10])
        T = rng.randint(1, 5)
        ans = Rational(P * R * T, 100)
        text = f"Find the simple interest on £{P} at {R}% per year for {T} years. Use I = PRT/100."
        formula, cands = "I = PRT/100", [P * R * T, P + R + T]
    else:  # perimeter
        l, w = rng.randint(2, hi), rng.randint(2, hi)
        ans = 2 * (l + w)
        text = f"A rectangle has length {l} and width {w}. Use P = 2(l + w) to find its perimeter."
        formula, cands = "P = 2(l + w)", [l * w, l + w]
    ans_val = ans if isinstance(ans, int) else float(ans)
    ans_text = str(ans_val) if isinstance(ans_val, int) else f"{ans_val:g}"
    # A generic off-by-a-bit filler keeps distractor coverage up even when a
    # formula-specific misconception happens to collide with the real answer.
    cands = cands + [ans_val + 1, ans_val - 1] if isinstance(ans_val, int) else cands + [ans_val * 2]
    return dict(
        text=text,
        latex=formula,
        answer=ans_val,
        answer_text=ans_text,
        answer_latex=ans_text,
        distractors=pick_distractors(ans_text, [f"{c:g}" if not isinstance(c, int) else str(c) for c in cands]),
    )
