"""Generates `p19f_concavity_inflection` questions: find a cubic's single inflection point and how its concavity
changes there.

Concavity and points of inflection.

Skill: senior-understand-the-concepts-of-concavity-and-points-of
(ACMMM110, topic ``p19f_concavity_inflection``, answer: ordered_list).

Given a cubic with EXACTLY ONE point of inflection (by construction, on
every draw), find that point's x-value and state whether concavity changes
from concave-up-to-concave-down or from concave-down-to-concave-up there.

**Why ``ordered_list``, not ``pair``.** Same reasoning as the sibling
(``p19f_second_derivative_test_build.py``): the contract answer is a mixed
(x-value, direction phrase) pair, and `grade_pair`'s component extraction
(`extract_values`) parses numbers only, so a text direction label can never
grade correctly as a `pair` component. `grade_ordered_list` compares
item-by-item positionally instead (numeric item -> parsed value, string item
-> case-insensitive text compare), exactly what this (number, phrase) answer
needs.

**The construction that makes exactly-one-inflection-point trivially
exact.** f(x) = A*x^3 + B*x^2 + C*x + D, A != 0. f''(x) = 6*A*x + 2*B is
LINEAR with nonzero slope, so it has exactly one real root and (being
linear, hence monotonic) always changes sign there -- there is no
"f''(x) = 0 but no sign change" case to exclude, unlike an even-multiplicity
root would require. x0 is drawn FIRST from an integer pool, then
B = -3*A*x0, so f''(x) = 6*A*x + 2*B = 6*A*(x - x0) has its root exactly at
x0. Direction: A > 0 -> f'' runs negative-to-positive through x0 -> "concave
down to concave up"; A < 0 -> "concave up to concave down". A's sign is
drawn 50/50, putting each direction at ~50% of draws (comfortably inside the
35-65% band).

**Sharp neighbour, the one that matters most.**
`senior-understand-and-use-the-second-derivative-test-for` solves f'(x) = 0 to find a STATIONARY point and classifies it using f'' at
that point -- it never solves f''(x) = 0. This skill solves f''(x) = 0 directly for the point of INFLECTION and
never touches f'(x) = 0 at all. The two skills share the "(x-value, phrase) ordered_list" answer SHAPE but solve
different equations for a different kind of point; every prompt here says "second derivative equals zero" /
"concavity" and never "critical point", "stationary" or "local maximum/minimum", to keep the two skills visually
distinct on the page. `second-derivative` renders f''(x) as a symbolic EXPRESSION -- never solved, never a
concavity direction reported. `stationary-points` solves f'(x) = 0 as an unordered list with no classification
at all.

**Misconceptions**, checked for genuine difference from the true answer by construction (whole-pair
inequality, not a bare-number distinctness check -- the ordered pair (x0, direction) is the unit that must
differ): (1) solves f'(x) = 0 instead of f''(x) = 0 -- confuses this skill with the second-derivative-test skill,
the single sharpest distractor risk, documented here rather than coded as a distractor (f'(x) = 0 is generally
a quadratic with zero, one or two real roots for this cubic, so it has no single clean integer counterpart to
render); (2) correct x-value, WRONG direction (the flipped concavity-change label); (3) correct direction,
WRONG x-value (the negated x0, a sign error solving 6*A*x + 2*B = 0).
"""

from __future__ import annotations

from sympy import diff, solve, symbols

from ..core import check, check_distinct, generator, latex_answer, phrase, pick_distractors

x = symbols("x")

_EASY_X0_POOL = (1, 2, 3, 4, 5)
_WIDE_X0_POOL = tuple(v for v in range(-9, 10) if v != 0)
_SPLIT_K_POOL = (1, 2, 3)


def _direction(is_up_to_down: bool) -> str:
    return "concave up to concave down" if is_up_to_down else "concave down to concave up"


def _verify_invariant(f_sym, x0: int, is_up_to_down: bool) -> None:
    """Solve f''(x) = 0 with sympy -- a route independent of the closed-form
    B = -3*A*x0 derivation above -- and assert exactly one real root, equal
    to x0, and that the third derivative is a nonzero constant (the
    sign-change guarantee for a smooth function) whose sign matches the
    intended direction."""
    second = diff(f_sym, x, 2)
    roots = solve(second, x)
    real_roots = [r for r in roots if r.is_real]
    check(len(real_roots) == 1, f"expected exactly one real inflection point, got {roots}")
    check(real_roots[0] == x0, f"inflection point {real_roots[0]} != intended x0={x0}")
    third = diff(f_sym, x, 3)
    check(third != 0, f"f'''(x) must be a nonzero constant, got {third}")
    # third = 6*A: A > 0 -> f'' rising through its root -> down-to-up.
    check((third > 0) == (not is_up_to_down), "direction does not match sign of f'''")


def _split_cubic_coeff(rng, A: int) -> tuple[int, int]:
    """Split A into two cubic-term coefficients P, E with P + E == A and
    both nonzero, for the hard/exam unexpanded rendering (the learner must
    collect P*x^3 + E*x^3 before differentiating)."""
    k = rng.choice(_SPLIT_K_POOL)
    while A + k == 0:
        k += 1
    return A + k, -k


def _term(coef: int, var_text: str) -> str:
    if coef == 1 and var_text:
        return var_text
    if coef == -1 and var_text:
        return f"-{var_text}"
    if not var_text:
        return str(coef)
    return f"{coef}·{var_text}" if coef >= 0 else f"-{abs(coef)}·{var_text}"


def _join_signed(terms: list[str]) -> str:
    parts = [t for t in terms if t]
    out = parts[0]
    for t in parts[1:]:
        if t.startswith("-"):
            out += f" - {t[1:]}"
        else:
            out += f" + {t}"
    return out


def _make_answer(x0: int, direction: str) -> tuple[list, str]:
    return [x0, direction], f"{x0}, {direction}"


@generator(
    "p19f_concavity_inflection",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19f_concavity_inflection(rng, tier="easy"):
    is_up_to_down = rng.choice([True, False])
    if tier == "easy":
        x0 = rng.choice(_EASY_X0_POOL)
        mag_A = rng.choice((1, 2))
    else:
        x0 = rng.choice(_WIDE_X0_POOL)
        mag_A = rng.randint(1, 9)
    A = -mag_A if is_up_to_down else mag_A
    B = -3 * A * x0
    C = rng.randint(-9, 9)
    if C == 0:
        C = 3
    D = rng.randint(-9, 9)
    f_sym = A * x**3 + B * x**2 + C * x + D
    _verify_invariant(f_sym, x0, is_up_to_down)

    direction = _direction(is_up_to_down)
    flipped_direction = _direction(not is_up_to_down)

    if tier in ("easy", "medium"):
        expr_text = _join_signed([_term(A, "x^3"), _term(B, "x^2"), _term(C, "x"), _term(D, "")])
        text = phrase(
            rng,
            (
                "Let f(x) = {expr}. This function has exactly one point of "
                "inflection, where f''(x) = 0 and the second derivative "
                "changes sign. Find the x-value of that point and state "
                "whether the concavity changes from concave up to concave "
                "down, or from concave down to concave up. Give your answer "
                "as the x-value, then the concavity change, separated by a "
                "comma, e.g. 10, concave up to concave down.",
                "The function f(x) = {expr} has exactly one point where "
                "f''(x) = 0 and the second derivative changes sign. Find its "
                "x-value, then say whether the concavity changes from "
                "concave up to concave down or from concave down to concave "
                "up there. Answer with the x-value first, then the "
                "concavity change, separated by a comma (e.g. 10, concave up "
                "to concave down).",
                "Consider f(x) = {expr}, which has exactly one point of "
                "inflection. Determine that point's x-value and, using the "
                "second derivative, state whether the concavity changes "
                "from concave up to concave down or from concave down to "
                "concave up. Give the x-value first, then the concavity "
                "change, separated by a comma (e.g. 10, concave up to "
                "concave down).",
            ),
            expr=expr_text,
        )
    elif tier == "hard":
        P, E = _split_cubic_coeff(rng, A)
        expr_text = _join_signed(
            [
                _term(P, "x^3"),
                _term(C, "x"),
                _term(B, "x^2"),
                _term(E, "x^3"),
                _term(D, ""),
            ]
        )
        text = phrase(
            rng,
            (
                "Let f(x) = {expr}, written as a sum of terms before "
                "simplifying. This function has exactly one point of "
                "inflection, where f''(x) = 0 and the second derivative "
                "changes sign. Collect like terms, then find the x-value of "
                "that point and state whether the concavity changes from "
                "concave up to concave down, or from concave down to "
                "concave up. Give your answer as the x-value, then the "
                "concavity change, separated by a comma, e.g. 10, concave "
                "up to concave down.",
                "The function f(x) = {expr} (not yet simplified) has "
                "exactly one point where f''(x) = 0 and the second "
                "derivative changes sign. Combine like terms first, then "
                "find its x-value and say whether the concavity changes "
                "from concave up to concave down or from concave down to "
                "concave up there. Answer with the x-value first, then the "
                "concavity change, separated by a comma (e.g. 10, concave "
                "up to concave down).",
            ),
            expr=expr_text,
        )
    else:  # exam
        P, E = _split_cubic_coeff(rng, A)
        expr_text = _join_signed(
            [
                _term(P, "x^3"),
                _term(C, "x"),
                _term(B, "x^2"),
                _term(E, "x^3"),
                _term(D, ""),
            ]
        )
        text = phrase(
            rng,
            (
                "The rate at which a chemical reaction proceeds is modelled "
                "by R(x) = {expr}{unit}. Find the "
                "x-value at which the reaction's behaviour changes from "
                "bending one way to bending the other, and state whether it "
                "changes from concave up to concave down or from concave "
                "down to concave up there. Give your answer as the "
                "x-value, then the concavity change, separated by a comma, "
                "e.g. 10, concave up to concave down.",
                "A quantity is modelled by Q(x) = {expr}. Find the x-value "
                "at which the graph of Q stops bending one way and starts "
                "bending the other way, and say whether that change is from "
                "concave up to concave down or from concave down to "
                "concave up. Answer with the x-value first, then the "
                "concavity change, separated by a comma (e.g. 10, concave "
                "up to concave down).",
            ),
            expr=expr_text,
            # Minutes only when the inflection is at a positive time
            # ("-5 minutes" would read as nonsense).
            unit=", where x is measured in minutes" if x0 > 0 else "",
        )

    answer, answer_text = _make_answer(x0, direction)

    # Misconceptions -- every one distinct from the true full answer by
    # construction, checked as WHOLE-PAIR inequality (either component
    # differs), not just bare-number distinctness of the x-value alone.
    check(direction != flipped_direction, "flipped direction must differ from the true direction")
    neg_x0 = -x0
    check_distinct(x0, neg_x0, msg="sign-error x-value must differ from the true x-value")
    check((x0, direction) != (x0, flipped_direction), "flipped-direction distractor must differ from the answer pair")
    check((neg_x0, direction) != (x0, direction), "sign-error distractor must differ from the answer pair")
    cand1 = f"{x0}, {flipped_direction}"
    cand2 = f"{neg_x0}, {direction}"
    distractors = pick_distractors(answer_text, [cand1, cand2])

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            "Solve f''(x) = 0 to find the x-value of the point of inflection.",
            "Check the sign of f'' on either side of that x-value: a change "
            "from negative to positive is concave down to concave up; a "
            "change from positive to negative is concave up to concave "
            "down.",
            f"Point of inflection: x = {x0}. Concavity changes {direction} there.",
        ],
        distractors=distractors,
    )
