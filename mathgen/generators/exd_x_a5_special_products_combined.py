"""Exam-tier source for special products: two special products combined in one expression.

The practice tiers expand or factorise ONE recognised product (a difference of squares or a perfect square).
Here two of them sit in the same expression, joined by + or -, and the learner has to expand both and combine
like terms, taking care that a subtraction applies to the whole second product. Often the squares cancel and
what remains is a short expression, so recognising the pattern pays off.

    Expand and simplify (3x + 2)^2 - (3x - 2)^2        ->  24x
    Expand and simplify (x + 4y)^2 - (x - 4y)(x + 4y)  ->  8xy + 32y^2

Built backwards from the pair (u, v): ``u = px`` and ``v`` is a constant or ``qy``. Each of the two products is
one of (u + v)^2, (u - v)^2, (u + v)(u - v). The key is the sympy expansion of the whole expression; an
independent route sums the hand-written term lists ``[u^2, +-2uv, v^2]`` and ``[u^2, -v^2]`` and must agree.

Distractors (each a named misconception, filtered to be genuinely different from the key)
    dropped-middle-term   a square written as u^2 + v^2
    partial-negation      only the first term of the subtracted product is negated (subtraction only)
    forgot-to-double      the middle term of a square is uv rather than 2uv
    swapped-middle-sign   (u + v)^2 and (u - v)^2 given each other's middle sign

The answer is graded with ``answer_form="expanded"``, so restating the question is rejected.
"""

from __future__ import annotations

import random

from sympy import expand, symbols

from ..core import check_equivalent, generator, join_terms, latex_answer, phrase, pick_distractors, symbolically_equal

x, y = symbols("x y")

_FORMS = ("square_plus", "square_minus", "conjugate")
_PATTERNS = (
    "Expand and simplify {expr}.",
    "Expand and simplify {expr}, giving your answer in simplest form.",
)


def _txt(expr) -> str:
    """Student-style text for a polynomial: no '*', '^' for powers."""
    return str(expr).replace("**", "^").replace("*", "")


def _term_list(form: str, u, v, *, drop_middle=False, double=True, flip=False) -> list:
    """The expanded terms of one product, as a list the caller can sum or partly negate."""
    if form == "conjugate":
        return [u**2, -(v**2)]
    sign = 1 if form == "square_plus" else -1
    if flip:
        sign = -sign
    if drop_middle:
        return [u**2, v**2]
    middle = (2 if double else 1) * sign * u * v
    return [u**2, middle, v**2]


def _product_text(form: str, u_text: str, v_text: str) -> str:
    plus = f"{u_text} + {v_text}"
    minus = f"{u_text} - {v_text}"
    if form == "square_plus":
        return f"({plus})^2"
    if form == "square_minus":
        return f"({minus})^2"
    return f"({plus})({minus})"


def _build(rng: random.Random) -> dict:
    p = rng.randint(1, 4)
    two_var = rng.random() < 0.5
    if two_var:
        q = rng.randint(1, 5)
        u, v = p * x, q * y
        u_text = "x" if p == 1 else f"{p}x"
        v_text = "y" if q == 1 else f"{q}y"
    else:
        q = rng.randint(1, 7)
        u, v = p * x, q
        u_text = "x" if p == 1 else f"{p}x"
        v_text = str(q)

    f1, f2 = rng.sample(_FORMS, 2)
    op = rng.choice((1, -1))
    joiner = "+" if op == 1 else "-"
    expr_text = f"{_product_text(f1, u_text, v_text)} {joiner} {_product_text(f2, u_text, v_text)}"

    # Route 1: literal expansion of the whole expression.
    def literal(form):
        return {
            "square_plus": (u + v) ** 2,
            "square_minus": (u - v) ** 2,
            "conjugate": (u + v) * (u - v),
        }[form]

    key = expand(literal(f1) + op * literal(f2))
    # Route 2: the hand-written term lists must agree.
    by_terms = sum(_term_list(f1, u, v)) + op * sum(_term_list(f2, u, v))
    check_equivalent(key, by_terms, "term-list expansion must agree with the literal expansion")
    if key == 0:
        raise ValueError("pair cancels to zero")  # caught below; never reaches the learner

    def total(first, second):
        return expand(sum(first) + op * sum(second))

    candidates = []
    # dropped-middle-term: every square is u^2 + v^2.
    candidates.append(
        total(
            _term_list(f1, u, v, drop_middle=(f1 != "conjugate")),
            _term_list(f2, u, v, drop_middle=(f2 != "conjugate")),
        )
    )
    # partial-negation: a subtraction negates only the first term of the second product.
    if op == -1:
        second = _term_list(f2, u, v)
        candidates.append(expand(sum(_term_list(f1, u, v)) - second[0] + sum(second[1:])))
    # forgot-to-double: middle term uv, not 2uv.
    candidates.append(total(_term_list(f1, u, v, double=False), _term_list(f2, u, v, double=False)))
    # swapped-middle-sign: each square gets the other's middle sign.
    candidates.append(
        total(
            _term_list(f1, u, v, flip=(f1 != "conjugate")),
            _term_list(f2, u, v, flip=(f2 != "conjugate")),
        )
    )
    candidates = [c for c in candidates if not symbolically_equal(c, key)]

    if len(candidates) < 2:
        raise ValueError("too few distinct misconceptions")  # e.g. (u+v)^2 + (u-v)^2: slips cancel
    answer_text = _txt(key)
    text = phrase(rng, _PATTERNS, expr=expr_text)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(key),
        distractors=pick_distractors(answer_text, [_txt(c) for c in candidates], n=3),
        answer_form="expanded",
    )


def _exam(rng: random.Random) -> dict:
    # A pair can cancel to zero; redraw rather than ask for "0", and redraw a pair
    # whose slips collapse onto the key.
    for _ in range(50):
        try:
            return _build(rng)
        except ValueError:
            continue
    raise RuntimeError("could not draw a non-zero special-product pair")


@generator(
    "exd_x_a5_special_products_combined",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a5_special_products_combined(rng: random.Random, tier: str = "exam") -> dict:
    """Expand and simplify two special products joined by + or -."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
