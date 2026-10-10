"""Generates `p19f_second_derivative_test` questions: find a function's single critical point and classify it
with the second derivative test.

The second derivative test.

Skill: senior-understand-and-use-the-second-derivative-test-for
(ACMMM111, topic ``p19f_second_derivative_test``, answer: ordered_list).

Given a function with EXACTLY ONE critical point (by construction, on every
draw), find that critical point's x-value and use the second derivative
test to classify it as a local maximum or a local minimum. The skill
never answers "inconclusive" -- a function where the test is genuinely
inconclusive (f''(x0) == 0) is excluded by construction, not handled as a
third label.

**Why ``ordered_list``, not ``pair``.** The contract answer is (x-value, classification word) -- a mixed
numeric/string pair. `grade_pair`'s component extraction (`extract_values`) parses NUMBERS out of the
learner's text; a text label like "local maximum" yields zero extracted values, so a `pair` answer can never
grade that component correctly. `grade_ordered_list` compares item-by-item POSITIONALLY instead, and its own
per-item rule (numeric item -> parsed value; string item -> case-insensitive text compare) is exactly what a
(number, word) pair needs.

**Construction, by tier (guarantees the invariant exactly, never measured
after the fact):**

- easy/medium: f(x) = A*x^2 + B*x + C, A != 0. The critical x-value x0 is
  drawn FIRST, then B = -2*A*x0, so f'(x) = 2*A*x + B has its one root
  exactly at x0 and f''(x) = 2*A is a nonzero CONSTANT -- the invariant
  holds trivially. Classification is decided by the sign of A (A < 0 ->
  local maximum, A > 0 -> local minimum), drawn 50/50 so both labels sit at
  ~50% of draws, comfortably inside the 35-65% band.
  - easy: x0 drawn from a small POSITIVE pool ({1..5}) and |A| in {1, 2} --
    every arithmetic step is small and single-sign.
  - medium: x0 drawn from a wider pool that includes negative values
    ({-9..9}\\{0}) and |A| in {1..9} -- computing B = -2*A*x0 and solving
    2*A*x + B = 0 now routinely mixes signs (a negative x0 times a negative
    A, etc.), a genuine trap-structure escalation over easy's single-sign
    arithmetic, not merely a wider range.
- hard: f(x) = A*u(x)^2 + C, where u(x) = x^2 - 2*p*x + q, q = p*p + k,
  k > 0 an integer, so u(x) has NO real root by construction (its
  discriminant is 4*p*p - 4*q = -4*k < 0 for every k > 0 -- asserted
  explicitly with sympy, not just trusted). u(x) is rendered EXPANDED, as
  an ordinary quadratic-in-x (`x^2 - 2p·x + q`), never re-factored back
  into a bracketed `(x - p)` form, so p is never a bare literal in the
  printed text. f'(x) = 2*A*u(x)*u'(x) = 2*A*u(x)*(2x - 2p); since u(x) is
  never zero, the only real critical point is where u'(x) = 0, i.e. x = p
  -- a genuine "solve the derivative, argue the other factor is never
  zero" step, distinct from copying a visible bracket. f''(p) = 4*A*k
  (since u'(p) = 0 and u(p) = k), so the classification is decided by the
  sign of A (A > 0 -> local minimum, A < 0 -> local maximum), same role B
  played in the previous construction.
- exam: f(x) = A*(x-p)^4 + B*(x-p)^2 + C, A and B drawn to share a sign
  (both positive or both negative), both nonzero, p != 0. f'(x) =
  2*(x-p) * [2*A*(x-p)^2 + B]; the bracket has no real root when A, B
  share a sign (since (x-p)^2 = -B/(2A) < 0 would be required), so x = p
  is the ONLY real critical point on every draw. f''(p) = 2*B != 0, so the
  classification is decided by the sign of B (B > 0 -> local minimum, B < 0
  -> local maximum) -- NOT the sign of A, which is easy to get backwards.
  f(x) is `sympy.expand`ed and rendered as an ordinary quartic polynomial
  in x, with the shift structure fully hidden -- the learner must find the
  critical point by solving f'(x) = 0 as a plain cubic in x (no bracketed
  or factored hint that only one real root exists), a further genuinely
  different, harder decision than hard's expanded-quadratic-squared form
  (which still telegraphs "outer square of an inner expression -- find
  where the inner expression's derivative vanishes").

**The invariant proof, every draw, every tier alike:** solve f'(x) = 0
with sympy (a route independent of the closed-form x0/classification
derivation above) and assert it returns EXACTLY ONE real root, equal to the
intended x0, and assert f''(x0) != 0 with the sign matching the intended
classification.

**Misconceptions**, every one distinct from the true full answer by
construction: (1) correct x-value, WRONG classification (the flipped
max/min label -- the sharpest, most natural error for this skill);
(2) reports the critical point as "neither maximum nor minimum", as if
f'(x0) = 0 were itself evidence the test is inconclusive; (3) correct
classification, WRONG x-value (the negated x0 -- a sign error solving
f'(x) = 0).
"""

from __future__ import annotations

from sympy import diff, expand, solve, symbols

from ..core import check, check_distinct, generator, latex_answer, phrase, pick_distractors

x = symbols("x")

_EASY_X0_POOL = (1, 2, 3, 4, 5)
_MEDIUM_X0_POOL = tuple(v for v in range(-9, 10) if v != 0)
_SHIFT_POOL = (-3, -2, -1, 1, 2, 3)
_K_POOL = (1, 2, 3, 4)


def _verify_quadratic_invariant(f_sym, x0: int, is_min: bool) -> None:
    """The actual invariant proof for the quadratic construction: solve
    f'(x) = 0 with sympy -- a route independent of the closed-form
    `B = -2*A*x0` derivation above -- and assert exactly one real root,
    equal to x0, with f''(x0) carrying the intended sign."""
    roots = solve(diff(f_sym, x), x)
    real_roots = [r for r in roots if r.is_real]
    check(len(real_roots) == 1, f"expected exactly one real critical point, got {roots}")
    check(real_roots[0] == x0, f"critical point {real_roots[0]} != intended x0={x0}")
    second = diff(f_sym, x, 2).subs(x, x0)
    check(second != 0, f"f''(x0) must not be zero at x0={x0}")
    check((second > 0) == is_min, f"second-derivative sign does not match intended classification")


def _verify_quartic_invariant(f_sym, p: int, is_min: bool) -> None:
    """Same proof for the shifted-quartic construction: solve f'(x) = 0
    directly with sympy (never assuming the bracket-has-no-real-root
    argument the construction relies on) and assert exactly one real root."""
    roots = solve(diff(f_sym, x), x)
    real_roots = [r for r in roots if r.is_real]
    check(len(real_roots) == 1, f"expected exactly one real critical point, got {roots}")
    check(real_roots[0] == p, f"critical point {real_roots[0]} != intended p={p}")
    second = diff(f_sym, x, 2).subs(x, p)
    check(second != 0, f"f''(p) must not be zero at p={p}")
    check((second > 0) == is_min, f"second-derivative sign does not match intended classification")


def _verify_hard_invariant(f_sym, u_sym, p: int, k: int, is_min: bool) -> None:
    """The hard-tier invariant proof: solve f'(x) = 0 directly with sympy
    (an independent route from the "u(x) has no real root, so the only
    real critical point of f'(x) = 2*A*u(x)*u'(x) comes from u'(x) = 0"
    closed-form argument) and assert exactly one real root, equal to p.
    Also assert explicitly that u(x) = x^2 - 2*p*x + q has no real root
    (discriminant -4*k < 0 for k > 0, checked directly rather than trusted)."""
    check(k > 0, f"k must be positive so u(x) has no real root, got k={k}")
    u_roots = solve(u_sym, x)
    real_u_roots = [r for r in u_roots if r.is_real]
    check(len(real_u_roots) == 0, f"u(x) must have no real root, got {u_roots}")
    roots = solve(diff(f_sym, x), x)
    real_roots = [r for r in roots if r.is_real]
    check(len(real_roots) == 1, f"expected exactly one real critical point, got {roots}")
    check(real_roots[0] == p, f"critical point {real_roots[0]} != intended p={p}")
    second = diff(f_sym, x, 2).subs(x, p)
    check(second != 0, f"f''(p) must not be zero at p={p}")
    check((second > 0) == is_min, f"second-derivative sign does not match intended classification")


def _quadratic_text(expr_text: str) -> str:
    return f"f(x) = {expr_text}"


def _term(coef: int, var_text: str) -> str:
    if coef == 1 and var_text:
        return var_text
    if coef == -1 and var_text:
        return f"-{var_text}"
    if not var_text:
        return str(coef)
    return f"{coef}·{var_text}" if coef >= 0 else f"-{abs(coef)}·{var_text}"


def _join_signed(terms: list[str]) -> str:
    """Join already-signed term strings (each starting with its own sign)
    into "a - b + c" form, dropping any empty term."""
    parts = [t for t in terms if t]
    out = parts[0]
    for t in parts[1:]:
        if t.startswith("-"):
            out += f" - {t[1:]}"
        else:
            out += f" + {t}"
    return out


def _shifted_quadratic_factor(p: int, q: int, power: int) -> str:
    """Render the EXPANDED quadratic `x^2 - 2*p*x + q`, raised to `power`,
    without factoring it back into a bracketed `(x - p)` form -- the whole
    point is that p never appears as a bare literal in the printed text."""
    inner = _join_signed([_term(1, "x^2"), _term(-2 * p, "x"), _term(q, "")])
    return f"({inner})^{power}"


def _make_answer(x0: int, label: str) -> tuple[list, str]:
    answer = [x0, label]
    answer_text = f"{x0}, {label}"
    return answer, answer_text


@generator(
    "p19f_second_derivative_test",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19f_second_derivative_test(rng, tier="easy"):
    if tier in ("easy", "medium"):
        is_min = rng.choice([True, False])
        if tier == "easy":
            x0 = rng.choice(_EASY_X0_POOL)
            mag_A = rng.choice((1, 2))
        else:
            x0 = rng.choice(_MEDIUM_X0_POOL)
            mag_A = rng.randint(1, 9)
        A = mag_A if is_min else -mag_A
        B = -2 * A * x0
        C = rng.randint(-9, 9)
        f_sym = A * x**2 + B * x + C
        _verify_quadratic_invariant(f_sym, x0, is_min)

        expr_text = _join_signed([_term(A, "x^2"), _term(B, "x"), _term(C, "")])
        label = "local minimum" if is_min else "local maximum"
        flipped_label = "local maximum" if is_min else "local minimum"

        text = phrase(
            rng,
            (
                "Let f(x) = {expr}. This function has exactly one critical point, "
                "where f'(x) = 0. Find the x-value of that critical point and use "
                "the second derivative test to classify it as a local maximum or a "
                "local minimum. Give your answer as the x-value, then the "
                "classification (local maximum or local minimum), separated by a "
                "comma, e.g. 10, local maximum.",
                "The function f(x) = {expr} has exactly one point where f'(x) = 0. "
                "Find its x-value, then use the second derivative test to say "
                "whether it is a local maximum or a local minimum. Answer with the "
                "x-value first, then the classification, separated by a comma "
                "(e.g. 10, local maximum).",
                "Consider f(x) = {expr}, which has exactly one critical point. "
                "Determine that critical point's x-value and, using the second "
                "derivative test, classify it as a local maximum or a local "
                "minimum. Give the x-value first, then the classification, "
                "separated by a comma (e.g. 10, local maximum).",
            ),
            expr=expr_text,
        )
    elif tier == "hard":
        is_min = rng.choice([True, False])
        mag_A = rng.randint(1, 2)
        A = mag_A if is_min else -mag_A
        p = rng.choice(_SHIFT_POOL)
        k = rng.choice(_K_POOL)
        q = p * p + k
        C = rng.randint(-9, 9)
        u_sym = x**2 - 2 * p * x + q
        f_sym = A * u_sym**2 + C
        _verify_hard_invariant(f_sym, u_sym, p, k, is_min)

        x0 = p
        label = "local minimum" if is_min else "local maximum"
        flipped_label = "local maximum" if is_min else "local minimum"

        bracket_text = _shifted_quadratic_factor(p, q, 2)
        expr_text = _join_signed(
            [
                _term(A, bracket_text),
                _term(C, ""),
            ]
        )
        text = phrase(
            rng,
            (
                "Let f(x) = {expr}. This function has exactly one critical "
                "point, where f'(x) = 0. Find the x-value of that critical "
                "point and use the second derivative test to classify it as a "
                "local maximum or a local minimum. Give your answer as the "
                "x-value, then the classification (local maximum or local "
                "minimum), separated by a comma, e.g. 10, local maximum.",
                "The function f(x) = {expr} has exactly one point where "
                "f'(x) = 0. Find its x-value, then use the second derivative "
                "test to say whether it is a local maximum or a local minimum. "
                "Answer with the x-value first, then the classification, "
                "separated by a comma (e.g. 10, local maximum).",
                "Consider f(x) = {expr}, which has exactly one critical point. "
                "Determine that critical point's x-value and, using the second "
                "derivative test, classify it as a local maximum or a local "
                "minimum. Give the x-value first, then the classification, "
                "separated by a comma (e.g. 10, local maximum).",
            ),
            expr=expr_text,
        )
    elif tier == "exam":
        is_min = rng.choice([True, False])
        mag_A = rng.randint(1, 2)
        mag_B = rng.randint(1, 6)
        A = mag_A if is_min else -mag_A
        B = mag_B if is_min else -mag_B
        p = rng.choice(_SHIFT_POOL)
        C = rng.randint(-9, 9)
        f_sym = A * (x - p) ** 4 + B * (x - p) ** 2 + C
        _verify_quartic_invariant(f_sym, p, is_min)

        x0 = p
        label = "local minimum" if is_min else "local maximum"
        flipped_label = "local maximum" if is_min else "local minimum"

        expanded = expand(f_sym)
        expr_text = str(expanded).replace("**", "^").replace("*", "·")
        text = phrase(
            rng,
            (
                "Let f(x) = {expr}. This function has exactly one critical "
                "point, where f'(x) = 0. Find the x-value of that critical "
                "point and use the second derivative test to classify it as a "
                "local maximum or a local minimum. Give your answer as the "
                "x-value, then the classification (local maximum or local "
                "minimum), separated by a comma, e.g. 10, local maximum.",
                "The function f(x) = {expr}, written out in full, has exactly "
                "one point where f'(x) = 0. Find its x-value, then use the "
                "second derivative test to say whether it is a local maximum "
                "or a local minimum. Answer with the x-value first, then the "
                "classification, separated by a comma (e.g. 10, local maximum).",
                "Consider the function f(x) = {expr}, which has exactly one "
                "critical point. Determine that critical point's x-value and, "
                "using the second derivative test, classify it as a local "
                "maximum or a local minimum. Give the x-value first, then the "
                "classification, separated by a comma (e.g. 10, local maximum).",
            ),
            expr=expr_text,
        )
    else:
        raise ValueError(f"unknown tier {tier!r}")

    answer, answer_text = _make_answer(x0, label)

    # Misconceptions -- every one distinct from the true full answer by
    # construction (x0 != 0 for hard/exam via `_SHIFT_POOL`; the labels are
    # never equal to each other by construction).
    check(label != flipped_label, "flipped label must differ from the true label")
    neg_x0 = -x0
    check_distinct(x0, neg_x0, msg="sign-error x-value must differ from the true x-value")
    cand1 = f"{x0}, {flipped_label}"
    cand2 = f"{x0}, neither maximum nor minimum"
    cand3 = f"{neg_x0}, {label}"
    distractors = pick_distractors(answer_text, [cand1, cand2, cand3])

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            "Solve f'(x) = 0 to find the critical point's x-value.",
            "Evaluate f''(x) at that x-value: a positive second derivative means a "
            "local minimum, a negative second derivative means a local maximum.",
            f"Critical point: x = {x0}. f''({x0}) has the sign of a {label}.",
        ],
        distractors=distractors,
    )
