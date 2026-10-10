"""Variable rates of change of non-linear functions (ACMMM087), senior ``functions-sequences``, ``answer:
number``.

Average rate of change of f over ``[x1, x2]`` is ``(f(x2) - f(x1)) / (x2 -
x1)``. Every function family used at every tier below has integer
coefficients and integer x1/x2, and this quotient is always an EXACT
integer for every one of them -- see ``_verify_average_rate``'s algebraic
route and the module-level identities proven there.

Every question built here states f(x)'s complete formula with every coefficient a concrete drawn integer --
`a`/`b`/`c` (or the implicit leading 1) are all fixed BEFORE the question f-string is built; nothing is ever
left symbolic or "an initial condition" the learner must assume.
"""

from __future__ import annotations

from sympy import Symbol, cancel

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    pick_distractors,
)

_NONZERO_9 = tuple(v for v in range(-9, 10) if v != 0)
_EASY_WIDTHS = (2, 3)
_MEDIUM_WIDTHS = (2, 3, 4, 5)
_HARD_WIDTHS = (2, 3, 4, 5)
_MEDIUM_LEADING = (2, 3, -2, -3)

_XS1 = Symbol("x1")
_XS2 = Symbol("x2")


def _verify_average_rate(kind: str, a: int, b: int, c: int, x1: int, x2: int, rate: int) -> None:
    """Independent route: build f as a SYMBOLIC sympy expression in the
    general symbols x1/x2, then cancel the difference quotient
    ``(f(x2) - f(x1)) / (x2 - x1)`` ALGEBRAICALLY via ``sympy.cancel``
    (polynomial division over the rationals) BEFORE substituting the drawn
    integers -- a genuinely different code path from construction's plain
    Python integer subtraction and floor division (``//``) at the specific
    numbers. Only after the algebraic identity is reduced does this
    substitute x1/x2 and compare against the constructed `rate`.
    """
    if kind == "quadratic":
        f = lambda v: a * v**2 + b * v + c
    elif kind == "cubic":
        f = lambda v: v**3 + b * v + c
    else:
        raise ValueError(f"unknown kind {kind!r}")
    general_quotient = cancel((f(_XS2) - f(_XS1)) / (_XS2 - _XS1))
    quotient_at_point = general_quotient.subs({_XS1: x1, _XS2: x2})
    check_equivalent(
        quotient_at_point,
        rate,
        f"sympy algebraic cancellation disagrees with construction: kind={kind}, a={a}, b={b}, c={c}, x1={x1}, x2={x2}",
    )


@generator(
    "p11b_variable_rate_of_change",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p11b_variable_rate_of_change(rng, tier="easy"):
    """Average rate of change of a fully-specified polynomial f(x) over a
    stated interval [x1, x2] of width >= 2 (never a unit interval -- see the
    misconception-collision note below).

    easy:   f(x) = x^2 + b*x + c (leading coefficient exactly 1). One
            reasoning step: evaluate f at two points, subtract, divide.
    medium: f(x) = a*x^2 + b*x + c, a a non-unit leading coefficient the
            learner must carry through both evaluations.
    hard:   f(x) = x^3 + b*x + c (cubic; a genuinely different algebraic
            identity from the quadratic case -- x2**3 - x1**3 factors as
            (x2 - x1)*(x1**2 + x1*x2 + x2**2) -- not merely bigger numbers).

    Guards against a zero rate (would collide the forgets-to-divide and
    sign-flip misconceptions with the answer and with each other) and
    against the endpoint-value-confusion distractor coinciding with the
    answer or another distractor, by resampling `b` (keeping a/c/x1/width
    fixed) up to 50 times.
    """
    if tier == "easy":
        kind = "quadratic"
        a = 1
        widths = _EASY_WIDTHS
    elif tier == "medium":
        kind = "quadratic"
        a = rng.choice(_MEDIUM_LEADING)
        widths = _MEDIUM_WIDTHS
    elif tier == "hard":
        kind = "cubic"
        a = 1
        widths = _HARD_WIDTHS
    else:
        raise ValueError(f"unknown tier {tier!r}; expected 'easy', 'medium', or 'hard'")

    found = False
    x1 = width = x2 = c = b = rate = diff_total = endpoint_val = None
    for _outer_attempt in range(50):
        x1 = rng.randint(-6, 6)
        width = rng.choice(widths)
        x2 = x1 + width
        c = rng.choice(_NONZERO_9)

        if kind == "quadratic":
            f = lambda v, bv: a * v**2 + bv * v + c
        else:
            f = lambda v, bv: v**3 + bv * v + c

        # Inner loop: the zero-rate guard -- keep a/c/x1/width
        # fixed and resample only b, up to 50 times. This alone is enough
        # to escape rate == 0 (rate is a bijection of b for fixed a/x1/x2,
        # so at most one b value in the pool of 18 makes it zero). It is
        # NOT always enough to escape an endpoint-value-confusion collision:
        # for a handful of (a, c, x1, x2) combinations (e.g. x2 == 1 with
        # c == a*x1) f(x2) equals the rate for EVERY b, independent of the
        # resample -- a structural collision, not an incidental one. The
        # outer loop above exists for exactly that case: it redraws
        # x1/width/c (and so x2) when the inner loop cannot escape it.
        for _inner_attempt in range(50):
            b = rng.choice(_NONZERO_9)
            diff_total = f(x2, b) - f(x1, b)
            check(
                diff_total % width == 0,
                f"average rate of change did not divide exactly: tier={tier}, a={a}, b={b}, c={c}, x1={x1}, x2={x2}",
            )
            rate = diff_total // width
            endpoint_val = f(x2, b)
            # Guard: rate must be nonzero (else forgets-to-divide/sign-flip
            # collide with the answer and each other), and every distractor
            # candidate must be pairwise distinct from the answer and from
            # every other candidate.
            if rate != 0 and len({rate, diff_total, -rate, endpoint_val}) == 4:
                found = True
                break
        if found:
            break
    else:
        raise ValueError(f"could not draw a non-degenerate question after 50 outer resamples: tier={tier}, a={a}")

    _verify_average_rate(kind, a, b, c, x1, x2, rate)

    if kind == "quadratic":
        poly = join_terms([(a, "x^2"), (b, "x"), (c, "")])
    else:
        poly = join_terms([(1, "x^3"), (b, "x"), (c, "")])

    kind_word = "quadratic" if kind == "quadratic" else "cubic"
    question_text = (
        f"A {kind_word} function is given by f(x) = {poly}. Find the average rate of change "
        f"of f(x) over the interval from x = {x1} to x = {x2}."
    )

    answer_text = str(rate)

    candidates_text = [str(diff_total), str(-rate), str(endpoint_val)]
    check_pairwise_distinct(
        rate,
        diff_total,
        -rate,
        endpoint_val,
        msg=f"a distractor collides with the answer or another distractor: tier={tier}, "
        f"a={a}, b={b}, c={c}, x1={x1}, x2={x2}",
    )
    distractors = pick_distractors(answer_text, candidates_text, n=3)

    f_x1 = f(x1, b)
    return dict(
        text=question_text,
        latex=question_text,
        answer=rate,
        answer_text=answer_text,
        answer_latex=latex_answer(rate),
        working=[
            f"f(x) = {poly}.",
            f"f({x2}) = {endpoint_val}; f({x1}) = {f_x1}.",
            f"Average rate of change = (f({x2}) - f({x1})) / ({x2} - {paren_negative(x1)}) "
            f"= {diff_total} / {width} = {rate}.",
        ],
        distractors=distractors,
    )
