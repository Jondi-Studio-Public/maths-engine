"""Generates `r4e_hybrid_functions` questions (easy, medium, hard) and the exam chain `r4e_hybrid_functions_chain`.

Skill `hybrid-functions` (answer `fraction`): functions defined by different rules on different intervals.

* easy: evaluate f at a point, choosing the rule from the inequality; the point is often the join itself, where the
  inequality (< or >=) decides which rule applies;
* medium: find the constant k that makes f continuous at the join, by equating the two rules at the join;
* hard: either k appears in both rules (so the equation has k on both sides), or there are two joins and two unknown
  constants and the question asks for their sum;
* exam: f(x) = x^2 + k for x < a and px + q for a <= x <= b: k for continuity, f at a point following through from k,
  the domain, and the range (which depends on whether the vertex of x^2 + k lies inside the first interval).

Every key is checked by evaluating both sides of each join with SymPy and by SymPy's limit from the left and right.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, paren_negative, phrase
from .r4e_common import Redraw, draw, fraction_part, inequality_part, num, single

x, k, m = sympy.symbols("x k m")


def fr(v) -> Fraction:
    v = sympy.Rational(v)
    return Fraction(int(v.p), int(v.q))


def rule_text(expr) -> str:
    poly = sympy.Poly(sympy.expand(expr), x)
    terms = []
    for (power,), coef in poly.terms():
        terms.append((int(coef), "" if power == 0 else ("x" if power == 1 else f"x^{power}")))
    return join_terms(terms)


def rule_text_k(expr) -> str:
    """A rule with a parameter k written as a learner would (k x, x^2 + k)."""
    poly = sympy.Poly(sympy.expand(expr), x, k)
    terms = []
    for (px, pk), coef in sorted(poly.terms(), key=lambda mono: (-mono[0][0], -mono[0][1])):
        name = ("" if px == 0 else ("x" if px == 1 else f"x^{px}")) + ("" if pk == 0 else "k")
        if px and pk:
            name = ("x" if px == 1 else f"x^{px}") + "k" if False else ("kx" if px == 1 else f"kx^{px}")
        terms.append((int(coef), name))
    return join_terms(terms)


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(500, "an evaluation whose three rule values differ"):
        a = rng.choice((-2, -1, 0, 1, 2, 3))
        b = a + rng.choice((2, 3))
        three = rng.random() < 0.5
        r1 = rng.choice((1, 2, 3, -1, -2)) * x + rng.choice((-3, -1, 2, 4, 5))
        r2 = rng.choice((1, 2, -1)) * x**2 + rng.choice((-4, -2, 1, 3))
        r3 = rng.choice((1, 2, 3, -1)) * x + rng.choice((-5, -2, 0, 3, 6))
        c = rng.choice((a, a, b, a - 1, a + 1, b + 1, b - 1))
        # which rule applies: x < a -> r1; a <= x < b -> r2 (or x >= a -> r2 when two rules); x >= b -> r3
        if three:
            rules = [r1, r2, r3]
            applies = 0 if c < a else (1 if c < b else 2)
        else:
            rules = [r1, r2]
            applies = 0 if c < a else 1
        values = [rule.subs(x, c) for rule in rules]
        if len(set(values)) == len(values) and (c in (a, b) or rng.random() < 0.5):
            break
    key = values[applies]
    # independent check: SymPy Piecewise
    pw = sympy.Piecewise(*([(r1, x < a), (r2, x < b), (r3, True)] if three else [(r1, x < a), (r2, True)]))
    check(pw.subs(x, c) == key, "the piecewise value disagrees")
    wrong = [(fr(v), "hybrid-wrong-rule") for i, v in enumerate(values) if i != applies]
    if c == a:
        wrong.insert(0, (fr(values[0]), "hybrid-boundary-point-in-wrong-piece"))
    if three and c == b:
        wrong.insert(0, (fr(values[1]), "hybrid-boundary-point-in-wrong-piece"))
    if three:
        stem = (
            f"A function is defined by f(x) = {rule_text(r1)} when x < {a}, f(x) = {rule_text(r2)} when {a} ≤ x < {b}, "
            f"and f(x) = {rule_text(r3)} when x ≥ {b}."
        )
    else:
        stem = phrase(
            rng,
            (
                "A function is defined by f(x) = {r1} when x < {a}, and f(x) = {r2} when x ≥ {a}.",
                "The function f is given by f(x) = {r1} for x < {a} and f(x) = {r2} for x ≥ {a}.",
            ),
            r1=rule_text(r1),
            r2=rule_text(r2),
            a=a,
        )
    which = [f"x < {a}", f"{a} ≤ x < {b}" if three else f"x ≥ {a}", f"x ≥ {b}"][applies]
    working = [
        f"At x = {c} the condition {which} holds, so f({c}) uses the rule {rule_text(rules[applies])}.",
        f"f({c}) = {key}.",
    ]
    return single(stem, fraction_part(f"Find f({c}).", fr(key), 1, working, wrong))


# -- medium -------------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(500, "a continuity constant distinct from its distractors"):
        a = rng.choice((-3, -2, -1, 1, 2, 3, 4))
        n_, p_ = rng.choice((-4, -2, -1, 1, 2, 3, 5)), rng.choice((-3, -1, 2, 4))
        form = rng.choice(("slope", "constant"))
        if form == "slope":
            # k x + n for x < a, x^2 + p for x >= a: k a + n = a^2 + p
            left, right = k * x + n_, x**2 + p_
            wrong_eq = [
                (sympy.Rational(a * a + p_ + n_, a), "continuity-sign-error"),
                (sympy.Rational(a * a + p_, a), "continuity-forgets-constant"),
                (sympy.Rational(2 * a, 1), "continuity-matches-derivatives"),
            ]
        else:
            # x^2 + k for x < a, n x + p for x >= a
            left, right = x**2 + k, n_ * x + p_
            wrong_eq = [
                (n_ * a + p_ + a * a, "continuity-sign-error"),
                (n_ * a + p_, "continuity-forgets-square"),
                (n_, "continuity-matches-derivatives"),
            ]
        sol = sympy.solve(sympy.Eq(left.subs(x, a), right.subs(x, a)), k)
        key = sol[0]
        wrong = [(fr(v) if False else v, t) for v, t in wrong_eq]
        wrong = [(sympy.Rational(v), t) for v, t in wrong]
        if all(w != key for w, _ in wrong) and abs(key) <= 40:
            break
    # the left and right limits at the join agree for this k, and only for this k
    lim_l = sympy.limit(left.subs(k, key), x, a, "-")
    lim_r = sympy.limit(right.subs(k, key), x, a, "+")
    check(lim_l == lim_r == right.subs(x, a), "f is not continuous at the join")
    stem = (
        f"The function f is defined by f(x) = {rule_text_k(left)} for x < {a} and f(x) = {rule_text_k(right)} for x ≥ {a}, "
        "where k is a constant."
    )
    working = [
        f"For continuity at x = {a} the two rules must give the same value there.",
        f"{sympy.sstr(left.subs(x, a)).replace('*', '')} = {sympy.sstr(right.subs(x, a)).replace('*', '')}, so k = {num(key)}.",
    ]
    return single(
        stem,
        fraction_part(
            "Find the value of k for which f is continuous at x = " + str(a) + ".",
            fr(key),
            2,
            working,
            [(fr(v), t) for v, t in wrong],
        ),
    )


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_both(rng):
    """k appears in both rules: x^2 + k x for x < a, c x - k for x >= a."""
    for _ in attempts(500, "a fractional or integer k distinct from its distractors"):
        a = rng.choice((1, 2, 3, -2, -3, 4))
        c = rng.choice((2, 3, 4, 5, -1, -2))
        left, right = x**2 + k * x, c * x - k
        sol = sympy.solve(sympy.Eq(left.subs(x, a), right.subs(x, a)), k)
        if len(sol) != 1:
            continue
        key = sol[0]
        slip = sympy.solve(sympy.Eq(left.subs(x, a), (c * x + k).subs(x, a)), k)  # sign of -k dropped
        one_sided = sympy.Rational(c * a - a * a, a) if a else None  # k from the left rule's k x term only
        wrong = [
            (slip[0] if slip else None, "continuity-parameter-sign-slip"),
            (one_sided, "continuity-collects-k-one-side-only"),
            (
                sympy.Rational(c * a, a + 1) * -1
                if False
                else (c * a + a * a) / sympy.Integer(a + 1)
                if a != -1
                else None,
                "continuity-sign-error",
            ),
            (sympy.Rational(c * a + a * a, a - 1) if a != 1 else None, "continuity-sign-error"),
            (sympy.Rational(c * a - a * a, 2 * a + 1), "continuity-collects-k-one-side-only"),
        ]
        wrong = [(sympy.Rational(v), t) for v, t in wrong if v is not None and v != key]
        if len(wrong) >= 2 and abs(key.p) <= 60 and key.q <= 6 and any((v.q == 1) == (key.q == 1) for v, _ in wrong):
            break
    check(sympy.limit(left.subs(k, key), x, a, "-") == sympy.limit(right.subs(k, key), x, a, "+"), "not continuous")
    stem = f"The function f is defined by f(x) = {rule_text_k(left)} for x < {a} and f(x) = {rule_text_k(right)} for x ≥ {a}, where k is a constant."
    working = [
        f"At x = {a} the rules give {sympy.sstr(left.subs(x, a)).replace('*', ' ')} and {sympy.sstr(right.subs(x, a)).replace('*', ' ')}.",
        f"Setting them equal gives k = {num(key)}.",
    ]
    return single(
        stem,
        fraction_part(
            f"Find the value of k for which f is continuous at x = {a}.",
            fr(key),
            3,
            working,
            [(fr(v), t) for v, t in wrong],
        ),
    )


def _hard_two_joins(rng):
    """Two joins, two constants: 2x + k for x < a, m for a <= x < b, x^2 + q for x >= b; ask for k + m."""
    for _ in attempts(500, "integer k and m with a sum distinct from its distractors"):
        a = rng.choice((-2, -1, 0, 1))
        b = a + rng.choice((2, 3))
        s = rng.choice((1, 2, 3))
        q0 = rng.choice((-4, -3, -1, 2))
        # rules: s*x + k (x < a), m (a <= x < b), x^2 + q0 (x >= b)
        m_val = b * b + q0
        k_val = m_val - s * a
        key = k_val + m_val
        wrong = [
            (k_val, "continuity-reports-one-constant"),
            (m_val, "continuity-reports-one-constant"),
            (k_val - m_val, "continuity-combines-constants-wrongly"),
            (m_val - s * b, "continuity-joins-at-wrong-point"),
        ]
        if key not in {w for w, _ in wrong} and abs(key) < 40 and m_val != 0 and k_val != 0:
            break
    check(sympy.limit(s * x + k_val, x, a, "-") == m_val, "left join disagrees")
    check(sympy.limit(x**2 + q0, x, b, "+") == m_val, "right join disagrees")
    stem = (
        f"The function f is defined by f(x) = {rule_text_k(s * x + k)} for x < {a}, f(x) = m for {a} ≤ x < {b}, "
        f"and f(x) = {rule_text(x**2 + q0)} for x ≥ {b}, where k and m are constants. f is continuous for all real x."
    )
    working = [
        f"At x = {b}: m = {paren_negative(b)}^2 + {paren_negative(q0)} = {m_val}.",
        f"At x = {a}: {s * a} + k = m, so k = {k_val}. Hence k + m = {key}.",
    ]
    return single(stem, fraction_part("Find the value of k + m.", key, 3, working, wrong))


@generator(
    "r4e_hybrid_functions",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4e_hybrid_functions(rng, tier: str = "easy"):
    """Hybrid (piecewise) functions: evaluation, continuity at a join, two joins; see the module docstring."""
    if tier == "easy":
        return draw(_easy, rng)
    if tier == "medium":
        return draw(_medium, rng)
    if tier == "hard":
        return draw(_hard_both if rng.random() < 0.55 else _hard_two_joins, rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ---------------------------------------------------------------------------------------------------


def _chain(rng):
    for _ in attempts(500, "a continuity constant, a first-piece point and a range that differ from their distractors"):
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        p = rng.choice((1, 2, 3))
        q = rng.choice((-4, -2, -1, 1, 2, 3))
        b = a + rng.choice((2, 3, 4))
        k_val = p * a + q - a * a
        c = a - rng.choice((1, 2, 3))
        if k_val != 0 and c != 0 and k_val != q:
            break
    k_val = sympy.Integer(k_val)
    # continuity, limit from each side
    check(sympy.limit(x**2 + k_val, x, a, "-") == sympy.limit(p * x + q, x, a, "+"), "not continuous")
    f_c = c * c + k_val
    # range: first piece x < a gives (a^2 + k, oo) with the vertex inside when a > 0, else [k, oo)
    if a > 0:
        range_ivs, floor, why = [(k_val, None, True, False)], k_val, f"the vertex x = 0 lies in x < {a}"
    else:
        range_ivs, floor, why = [(a * a + k_val, None, True, False)], a * a + k_val, f"x < {a} is left of the vertex"
    top_piece2 = p * b + q
    check(top_piece2 >= floor, "the second piece falls below the floor of the range")
    # independent check of the range by sampling both pieces
    samples = [sympy.Rational(i, 4) for i in range(-80, 80)]
    values = [(s * s + k_val) if s < a else (p * s + q) for s in samples if s <= b]
    check(min(values) >= floor, "a sampled value falls below the stated floor of the range")
    stem = (
        f"A function is defined by f(x) = x^2 + k for x < {a} and f(x) = {rule_text(p * x + q)} for {a} ≤ x ≤ {b}, "
        "where k is a constant. The function is continuous at the join."
    )
    part_a = fraction_part(
        "Find the value of k.",
        fr(k_val),
        2,
        [f"At x = {a}: {paren_negative(a)}^2 + k = {p * a + q}, so k = {k_val}."],
        [
            (fr(p * a + q + a * a), "continuity-sign-error"),
            (fr(p * a + q), "continuity-forgets-square"),
            (fr(a * a - (p * a + q)), "continuity-subtraction-reversed"),
        ],
    )
    part_b = fraction_part(
        f"Find f({c}).",
        fr(f_c),
        1,
        [f"{c} < {a}, so f({c}) = {paren_negative(c)}^2 + k = {f_c}."],
        [
            (fr(p * c + q), "hybrid-wrong-rule"),
            (fr(c * c), "hybrid-forgets-parameter"),
            (fr(c * c - k_val), "continuity-sign-error"),
        ],
        follow_through=f"{c * c} + part_a",
    )
    part_c = inequality_part(
        "State the domain of f.",
        [(None, b, False, True)],
        1,
        [f"The rules cover x < {a} and {a} ≤ x ≤ {b}, so the domain is x ≤ {b}."],
        [
            ([(None, b)], "hybrid-domain-endpoint-open-closed"),
            ([(None, a)], "hybrid-domain-first-rule-only"),
            ([(a, b, True, True)], "hybrid-domain-last-rule-only"),
        ],
    )
    part_d = inequality_part(
        "State the range of f.",
        range_ivs,
        3,
        [
            f"The first rule gives y ≥ {floor} ({why}); the second gives y from {p * a + q} to {top_piece2}, which lies inside it."
        ],
        [
            (
                [(k_val, None, True, False)] if a < 0 else [(a * a + k_val, None, True, False)],
                "hybrid-range-vertex-misplaced",
            ),
            ([(floor, None, False, False)], "hybrid-range-endpoint-open-closed"),
            ([(p * a + q, top_piece2, True, True)], "hybrid-range-last-rule-only"),
        ],
        var="y",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("r4e_hybrid_functions_chain", level=17, presets={"exam": dict(tier="exam")})
def r4e_hybrid_functions_chain(rng, tier: str = "exam"):
    """x^2 + k then a line: k for continuity, f at a point, the domain and the range."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
