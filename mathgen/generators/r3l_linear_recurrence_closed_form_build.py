"""Generates `uc2-linear-recurrence-closed-form` questions (easy, medium, hard) and the exam chain
`exd_x_r3l_linear_recurrence_closed_form`.

``uc2-linear-recurrence-closed-form`` (``discrete-mathematics``, answer ``expression``): the closed form (a formula in
n) of a sequence given by a second-order linear recurrence a(n) = p a(n-1) + q a(n-2) + k and two initial terms. No
characteristic equation is supplied and no root is named.

* easy: homogeneous (k = 0) with two distinct non-zero integer roots; the closed form is c1 r1^n + c2 r2^n.
* medium: homogeneous with a repeated root r; the closed form is (A + B n) r^n.
* hard: a constant term k. Either the characteristic roots are distinct and neither is 1 (a constant particular
  solution C = k / (1 - p - q), so the closed form is c1 r1^n + c2 r2^n + C), or one root is 1 and the particular
  solution is C n (the closed form is c1 r^n + c2 + C n).
* exam (``exd_x_r3l_linear_recurrence_closed_form``): a recurrence with a constant term. (a) the two roots of the
  characteristic equation, (b) the constant particular solution C, (c) the closed form, (d) the eighth term (follows
  through from (c)).

Every closed form is checked against terms 0 to 9 generated directly from the recurrence (the tests also compare it with
SymPy's `rsolve`); every distractor is checked to differ from the true sequence at some term in 0 to 9.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, paper_notation, paren_negative, phrase, pick_distractors

_n = sympy.symbols("n")
_ROOTS = (-3, -2, -1, 2, 3, 4)


def _terms(p, q, k, a0, a1, count=10):
    out = [a0, a1]
    while len(out) < count:
        out.append(p * out[-1] + q * out[-2] + k)
    return out


def _seq_of(expr, count=10):
    return [sympy.nsimplify(expr.subs(_n, i)) for i in range(count)]


def _matches(expr, p, q, k, a0, a1) -> bool:
    return _seq_of(expr) == [sympy.Integer(v) for v in _terms(p, q, k, a0, a1)]


def _form(expr) -> str:
    """The expression as typed: 3*2^n - 2*3^n, with no SymPy-isms such as `2**n`."""
    return str(expr).replace("**", "^")


def _rec_text(p, q, k=0) -> str:
    def term(c, sub):
        if c == 0:
            return ""
        mag = abs(c)
        body = f"a{sub}" if mag == 1 else f"{mag}a{sub}"
        return body

    parts = []
    for c, sub in ((p, "ₙ₋₁"), (q, "ₙ₋₂")):
        if c == 0:
            continue
        body = term(c, sub)
        if not parts:
            parts.append(("−" if c < 0 else "") + body)
        else:
            parts.append(("− " if c < 0 else "+ ") + body)
    if k:
        parts.append(("− " if k < 0 else "+ ") + str(abs(k)))
    return "aₙ = " + " ".join(parts)


def _char_text(p, q) -> str:
    """The characteristic equation r² − pr − q = 0 with tidy signs: r² − 2r − 3 = 0, r² + 8r + 16 = 0."""
    out = "r²"
    if p:
        out += (" − " if p > 0 else " + ") + (f"{abs(p)}r" if abs(p) != 1 else "r")
    if q:
        out += (" − " if q > 0 else " + ") + str(abs(q))
    return out + " = 0"


def _factor_text(r) -> str:
    """(r − 4) or (r + 4)."""
    return f"(r − {r})" if r > 0 else f"(r + {-r})"


def _plus(c) -> str:
    """` + 5` or ` − 5` for a signed constant."""
    return f" + {c}" if c > 0 else f" − {-c}"


def _fixed_text(p, q, k) -> str:
    """`2C + 8C − 18` for C = pC + qC + k, skipping a zero coefficient."""
    out = ""
    for c in (p, q):
        if c == 0:
            continue
        body = "C" if abs(c) == 1 else f"{abs(c)}C"
        out += (("−" if c < 0 else "") + body) if not out else ((" − " if c < 0 else " + ") + body)
    return out + (_plus(k) if out else str(k))


def _solve_check(p, q, k, a0, a1, expr):
    """The closed form must reproduce terms 0 to 9 generated directly from the recurrence."""
    check(_matches(expr, p, q, k, a0, a1), "closed form does not reproduce the recurrence")


def _expr_distractors(candidates, p, q, k, a0, a1, true_expr, text_true):
    """Distractors from `(expr, misconception)` pairs, dropping any that is the true sequence (or repeats)."""
    out, seen = [], {text_true}
    true_seq = _seq_of(true_expr)
    for expr, tag in candidates:
        if expr is None:
            continue
        txt = _form(expr)
        if txt in seen:
            continue
        if _seq_of(expr) == true_seq:
            continue
        seen.add(txt)
        out.append(Distractor(txt, tag, txt))
    return out


def _fit(r1, r2, a0, a1):
    """c1, c2 with c1 + c2 = a0 and c1 r1 + c2 r2 = a1 (exact)."""
    c1 = sympy.Rational(a1 - r2 * a0, r1 - r2)
    return c1, a0 - c1


def _easy_instance(rng):
    for _ in attempts(500, "distinct integer roots"):
        r1, r2 = rng.sample(_ROOTS, 2)
        c1, c2 = rng.choice([-3, -2, -1, 1, 2, 3, 4]), rng.choice([-3, -2, -1, 1, 2, 3])
        p, q = r1 + r2, -r1 * r2
        a0, a1 = c1 + c2, c1 * r1 + c2 * r2
        if a0 == 0 or a1 == 0 or abs(p) > 6 or abs(q) > 12:
            continue
        return p, q, r1, r2, c1, c2, a0, a1
    raise AssertionError  # pragma: no cover


def _easy(rng):
    p, q, r1, r2, c1, c2, a0, a1 = _easy_instance(rng)
    expr = c1 * sympy.Integer(r1) ** _n + c2 * sympy.Integer(r2) ** _n
    _solve_check(p, q, 0, a0, a1, expr)
    text = _form(expr)
    swapped = c2 * sympy.Integer(r1) ** _n + c1 * sympy.Integer(r2) ** _n
    flipped = c1 * sympy.Integer(-r1) ** _n + c2 * sympy.Integer(-r2) ** _n
    ones = sympy.Integer(r1) ** _n + sympy.Integer(r2) ** _n
    raw = a0 * sympy.Integer(r1) ** _n + a1 * sympy.Integer(r2) ** _n
    cands = [
        (swapped, "r3l-recurrence-coefficients-swapped"),
        (flipped, "r3l-recurrence-root-sign"),
        (raw, "r3l-recurrence-initial-conditions-misapplied"),
        (ones, "r3l-recurrence-initial-conditions-ignored"),
    ]
    distractors = _expr_distractors(cands, p, q, 0, a0, a1, expr, text)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=(
            f"A sequence has a₀ = {a0}, a₁ = {a1} and satisfies {_rec_text(p, q)} for n ≥ 2. "
            "Find a closed form for aₙ in terms of n."
        ),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        working=[
            f"Try aₙ = rⁿ: this gives {_char_text(p, q)}, with roots {r1} and {r2}.",
            f"aₙ = A·({r1})ⁿ + B·({r2})ⁿ. From a₀ = {a0} and a₁ = {a1}: A = {c1}, B = {c2}.",
            f"So aₙ = {paper_notation(text)}.",
        ],
        distractors=distractors,
    )


def _medium_instance(rng):
    for _ in attempts(500, "a repeated root"):
        r = rng.choice([-3, -2, 2, 3, 4, -4])
        A, B = rng.choice([-3, -2, -1, 1, 2, 3, 4]), rng.choice([-3, -2, -1, 1, 2, 3])
        p, q = 2 * r, -r * r
        a0, a1 = A, (A + B) * r
        if a1 == 0 or abs(a1) > 40:
            continue
        return p, q, r, A, B, a0, a1
    raise AssertionError  # pragma: no cover


def _medium(rng):
    p, q, r, A, B, a0, a1 = _medium_instance(rng)
    expr = (A + B * _n) * sympy.Integer(r) ** _n
    _solve_check(p, q, 0, a0, a1, expr)
    text = _form(expr)
    cands = [
        (A * sympy.Integer(r) ** _n, "r3l-recurrence-repeated-root-single-term"),
        ((A - B * _n) * sympy.Integer(r) ** _n, "r3l-recurrence-repeated-root-coefficient"),
        ((A + (a1 - A) * _n) * sympy.Integer(r) ** _n, "r3l-recurrence-repeated-root-coefficient"),
        ((A + B) * sympy.Integer(r) ** _n, "r3l-recurrence-repeated-root-no-n"),
        ((A + B * _n) * sympy.Integer(-r) ** _n, "r3l-recurrence-root-sign"),
    ]
    distractors = _expr_distractors(cands, p, q, 0, a0, a1, expr, text)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(
            rng,
            [
                "A sequence has a₀ = {a0}, a₁ = {a1} and satisfies {rec} for n ≥ 2. Find a closed form for aₙ in "
                "terms of n.",
                "A sequence has a₀ = {a0}, a₁ = {a1} and satisfies {rec} for n ≥ 2. Write aₙ in closed form.",
                "A sequence has a₀ = {a0}, a₁ = {a1} and satisfies {rec} for n ≥ 2. Find an expression for aₙ in "
                "terms of n.",
            ],
            a0=a0,
            a1=a1,
            rec=_rec_text(p, q),
        ),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        working=[
            f"{_char_text(p, q)} factorises as {_factor_text(r)}² = 0, so the root {r} is repeated and aₙ = (A + Bn)·({r})ⁿ.",
            f"a₀ = {a0} gives A = {A}; a₁ = (A + B)·({r}) = {a1} gives B = {B}.",
            f"So aₙ = {paper_notation(text)}.",
        ],
        distractors=distractors,
    )


def _hard_instance(rng, resonant):
    for _ in attempts(1000, "a constant-term recurrence"):
        if resonant:
            r1 = rng.choice([-2, 2, 3, -3])
            C = rng.choice([-3, -2, -1, 1, 2, 3])
            c1, c2 = rng.choice([-2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2, 4])
            p, q = 1 + r1, -r1
            k = C * (1 - r1)
            a0, a1 = c1 + c2, c1 * r1 + c2 + C
            expr = c1 * sympy.Integer(r1) ** _n + c2 + C * _n
            roots = (1, r1)
        else:
            r1, r2 = rng.sample([-3, -2, 2, 3, 4], 2)
            p, q = r1 + r2, -r1 * r2
            if p + q == 1:
                continue
            C = rng.choice([-4, -3, -2, -1, 1, 2, 3, 5])
            k = C * (1 - p - q)
            c1, c2 = rng.choice([-2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2, 3])
            a0, a1 = c1 + c2 + C, c1 * r1 + c2 * r2 + C
            expr = c1 * sympy.Integer(r1) ** _n + c2 * sympy.Integer(r2) ** _n + C
            roots = (r1, r2)
        if k == 0 or abs(k) > 40 or abs(p) > 7 or abs(q) > 12 or a0 == 0 or a1 == 0:
            continue
        return p, q, k, roots, C, (c1, c2), a0, a1, expr
    raise AssertionError  # pragma: no cover


def _hard_distractors(p, q, k, roots, C, a0, a1, expr, text, resonant):
    r1, r2 = roots
    cands = []
    # the constant term ignored: fit the homogeneous solution to the initial terms
    c1, c2 = _fit(r1, r2, a0, a1) if r1 != r2 else (None, None)
    if c1 is not None:
        hom = c1 * sympy.Integer(r1) ** _n + c2 * sympy.Integer(r2) ** _n
        cands.append((hom, "r3l-recurrence-constant-ignored"))
    if not resonant:
        # particular solution found with the wrong sign, coefficients refitted
        for tag, Cw in (("r3l-recurrence-particular-sign", -C), ("r3l-recurrence-particular-is-k", k)):
            d1, d2 = _fit(r1, r2, a0 - Cw, a1 - Cw)
            cands.append((d1 * sympy.Integer(r1) ** _n + d2 * sympy.Integer(r2) ** _n + Cw, tag))
        # particular solution added but the coefficients not refitted
        h1, h2 = _fit(r1, r2, a0, a1)
        cands.append(
            (h1 * sympy.Integer(r1) ** _n + h2 * sympy.Integer(r2) ** _n + C, "r3l-recurrence-particular-not-fitted")
        )
    else:
        other = r2 if r1 == 1 else r1
        cands.append((sympy.Integer(other) ** _n * (a0 - 0) + C * _n, "r3l-recurrence-resonance-missed"))
        d1, d2 = _fit(1, other, a0, a1)
        cands.append((d1 + d2 * sympy.Integer(other) ** _n + k * _n, "r3l-recurrence-particular-is-k"))
        cands.append((d1 + d2 * sympy.Integer(other) ** _n - C * _n, "r3l-recurrence-particular-sign"))
    return _expr_distractors(cands, p, q, k, a0, a1, expr, text)


def _hard(rng):
    resonant = rng.random() < 0.3
    p, q, k, roots, C, (c1, c2), a0, a1, expr = _hard_instance(rng, resonant)
    _solve_check(p, q, k, a0, a1, expr)
    text = _form(expr)
    distractors = _hard_distractors(p, q, k, roots, C, a0, a1, expr, text, resonant)
    check(len(distractors) >= 2, "too few distractors")
    if resonant:
        working = [
            f"The characteristic equation {_char_text(p, q)} has roots {roots[0]} and {roots[1]}; one of them is 1, so a constant cannot be a particular solution.",
            f"Try aₙ = Cn: Cn = {p}C(n − 1) + ({q})C(n − 2){_plus(k)} gives C = {C}.",
            f"Then aₙ = A·({roots[1]})ⁿ + B{_plus(C)}n, and a₀, a₁ give A = {c1}, B = {c2}: aₙ = {paper_notation(text)}.",
        ]
    else:
        working = [
            f"The characteristic equation {_char_text(p, q)} has roots {roots[0]} and {roots[1]}.",
            f"A constant particular solution C satisfies C = {_fixed_text(p, q, k)}, so C = {C}.",
            f"aₙ = A({roots[0]})ⁿ + B({roots[1]})ⁿ{_plus(C)}; a₀ = {a0} and a₁ = {a1} give A = {c1}, B = {c2}: aₙ = {paper_notation(text)}.",
        ]
    return dict(
        text=(
            f"A sequence has a₀ = {a0}, a₁ = {a1} and satisfies {_rec_text(p, q, k)} for n ≥ 2. "
            "Find a closed form for aₙ in terms of n."
        ),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        working=working,
        distractors=distractors,
    )


@generator(
    "r3l_linear_recurrence_closed_form",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3l_linear_recurrence_closed_form(rng, tier="easy"):
    """The closed form of a second-order linear recurrence; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------------


@generator("exd_x_r3l_linear_recurrence_closed_form", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r3l_linear_recurrence_closed_form(rng, tier="exam"):
    """Exam chain: characteristic roots, the constant particular solution, the closed form, then a term."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p, q, k, roots, C, (c1, c2), a0, a1, expr = _hard_instance(rng, False)
    r1, r2 = roots
    _solve_check(p, q, k, a0, a1, expr)
    text = _form(expr)
    index = rng.choice([6, 7, 8])
    value = int(expr.subs(_n, index))
    check(value == _terms(p, q, k, a0, a1, index + 1)[index], "term differs from the recurrence")
    roots_text = f"{min(roots)}, {max(roots)}"
    a_dist = pick_distractors(
        roots_text,
        [
            Distractor(f"{min(-r1, -r2)}, {max(-r1, -r2)}", "r3l-recurrence-root-sign", sorted([-r1, -r2])),
            Distractor(f"{min(p, q)}, {max(p, q)}", "r3l-recurrence-coefficients-as-roots", sorted([p, q])),
            Distractor(
                f"{min(r1 * r2, p)}, {max(r1 * r2, p)}", "r3l-recurrence-coefficients-as-roots", sorted([r1 * r2, p])
            ),
        ],
        3,
        unordered=True,
    )
    b_cands = [
        Distractor(str(-C), "r3l-recurrence-particular-sign", -C),
        Distractor(str(k), "r3l-recurrence-particular-is-k", k),
        Distractor(str(C + 1), "r3l-recurrence-particular-is-k", C + 1),
    ]
    if p + q != 0:
        b_cands.insert(
            1,
            Distractor(
                str(sympy.Rational(k, p + q)) if (p + q) != 0 and k % (p + q) == 0 else str(k + 1),
                "r3l-recurrence-particular-sign",
                int(k // (p + q)) if k % (p + q) == 0 else k + 1,
            ),
        )
    b_dist = pick_distractors(str(C), b_cands, 3)
    c_dist = _hard_distractors(p, q, k, roots, C, a0, a1, expr, text, False)[:3]
    seq = _terms(p, q, k, a0, a1, index + 1)
    d_cands = [
        Distractor(str(seq[index - 1]), "r3l-recurrence-index-off-by-one", seq[index - 1]),
        Distractor(str(seq[index + 0] - C), "r3l-recurrence-particular-not-fitted", seq[index] - C),
        Distractor(
            str(int(expr.subs(_n, index + 1))), "r3l-recurrence-index-off-by-one", int(expr.subs(_n, index + 1))
        ),
        Distractor(
            str(int(sum((c * sympy.Integer(r) ** index for c, r in ((c1, r1), (c2, r2)))))),
            "r3l-recurrence-constant-ignored",
            int(sum((c * sympy.Integer(r) ** index for c, r in ((c1, r1), (c2, r2))))),
        ),
    ]
    d_dist = pick_distractors(str(value), d_cands, 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    return dict(
        text=(f"A sequence has a₀ = {a0}, a₁ = {a1} and satisfies {_rec_text(p, q, k)} for n ≥ 2."),
        parts=[
            dict(
                prompt="Write down the two roots of the characteristic equation of the homogeneous recurrence aₙ = "
                + _rec_text(p, q).split("= ", 1)[1]
                + ".",
                answer_kind="list",
                answer=sorted(roots),
                answer_text=roots_text,
                marks=2,
                working=[f"{_char_text(p, q)} factorises as {_factor_text(r1)}{_factor_text(r2)} = 0."],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the constant C for which aₙ = C satisfies the full recurrence for all n.",
                answer_kind="fraction",
                answer=exact_value(Fraction(C)),
                answer_text=str(C),
                marks=2,
                working=[f"C = {_fixed_text(p, q, k)} gives C(1 − {paren_negative(p)} − ({q})) = {k}, so C = {C}."],
                distractors=b_dist,
            ),
            dict(
                prompt="Find a closed form for aₙ in terms of n.",
                answer_kind="expression",
                answer=text,
                answer_text=text,
                marks=4,
                working=[f"aₙ = A({r1})ⁿ + B({r2})ⁿ{_plus(C)}; the initial terms give A = {c1}, B = {c2}."],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Hence find a{''.join('₀₁₂₃₄₅₆₇₈₉'[int(ch)] for ch in str(index))}.",
                answer_kind="number",
                answer=value,
                answer_text=str(value),
                marks=1,
                follow_through=f"Subs(part_c, (n,), ({index},))",
                working=[f"Substitute n = {index} into the closed form: {value}."],
                distractors=d_dist,
            ),
        ],
    )
