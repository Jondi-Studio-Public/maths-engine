"""Exam chain for one-sided limits: a piecewise function with a parameter in both pieces.

``f(x) = s1 x^2 + k x + m`` for ``x < a`` and ``f(x) = s2 x + n - t k`` for ``x >= a``, where ``k`` is a constant.
Set the way a senior exam sets it:

(a) the left-hand limit at x = a, in terms of k                    expression, 2 marks
(b) the right-hand limit at x = a, in terms of k                   expression, 2 marks
(c) the value of k for which the limit at x = a exists             fraction,   2 marks, follow-through (a), (b)
(d) the value of that limit                                        fraction,   1 mark,  follow-through (a), (c)

## Construction: choose k first

The integers ``k``, ``a``, ``t``, ``s1``, ``s2`` and ``m`` are drawn, and then ``n`` is set so that the two one-sided
limits agree at that ``k``: ``n = k (a + t) + s1 a^2 + m - s2 a``. The equation the learner solves,
``s1 a^2 + a k + m = s2 a + n - t k``, then has the unique solution ``k`` because ``a + t`` is kept non-zero. Both
limits are re-derived by evaluating each piece beside ``a`` with exact fractions.

The follow-through for (c) solves ``part_a = part_b`` for k when both are linear in k, reading each learner
expression's constant term and k coefficient by substitution.

## Distractors

* (a), (b): the other piece used (the side ignored), the multiplier on k forgotten, a squared as doubled.
* (c): the sign slip moving the k terms across, the multiplier t forgotten, the value of the limit given
  instead of k.
* (d): k offered as the limit, and the limit worked out at a wrong k.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Symbol

from mathgen.answers import Distractor, exact_value
from mathgen.core import check, generator, join_terms, latex_answer, phrase, pick_distractors

_K = Symbol("k")

_STEMS = (
    "A function f is defined by f(x) = {left} for x < {a} and f(x) = {right} for x ≥ {a}, where k is a constant.",
    "Let f(x) = {left} when x < {a} and f(x) = {right} when x ≥ {a}, where k is a constant.",
    "The function f is given by f(x) = {left} for x < {a}, and by f(x) = {right} for x ≥ {a}, where k is a constant.",
)


def _number_part(prompt, value, marks, working, wrong, follow=None):
    text = str(value)
    cands = [Distractor(str(Fraction(v)), tag, exact_value(Fraction(v))) for v, tag in wrong if v is not None]
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(Fraction(value)),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, cands),
    )
    if follow:
        part["follow_through"] = follow
    return part


def _expression_part(prompt, slope, constant, marks, working, wrong):
    """A part whose answer is `slope*k + constant`; `wrong` lists (slope, constant, id)."""
    text = join_terms([(slope, "k"), (constant, "")])
    cands = [
        Distractor(join_terms([(s, "k"), (c, "")]), tag, None) for s, c, tag in wrong if (s, c) != (slope, constant)
    ]
    return dict(
        prompt=prompt,
        answer_kind="expression",
        answer=str(slope * _K + constant),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, cands),
    )


@generator("exd_x_r2c_one_sided_limits", level=16, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_one_sided_limits(rng, tier: str = "exam"):
    """A two-piece function with a parameter in both pieces: the two one-sided limits, then k, then the limit."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(1000):  # bounded rejection loop
        a = rng.choice([v for v in range(-4, 5) if v != 0])
        t = rng.choice([1, 2, 3])
        k = rng.choice([v for v in range(-7, 8) if v != 0])
        s1, s2 = rng.choice([1, 2, -1, 3]), rng.choice([1, 2, 3, -1, -2])
        m = rng.randint(-8, 8)
        if a + t != 0 and a * k + s1 * a * a + m not in (k, 0):  # the limit is neither zero nor equal to k
            break
    else:  # pragma: no cover
        check(False, "no usable draw")
    n = k * (a + t) + s1 * a * a + m - s2 * a

    # Exact one-sided limits, as linear functions of k: (slope, constant).
    left = (a, s1 * a * a + m)
    right = (-t, s2 * a + n)
    left_at, right_at = left[0] * k + left[1], right[0] * k + right[1]
    check(left_at == right_at, "the one-sided limits do not agree at k")
    check(Fraction(right[1] - left[1], left[0] - right[0]) == k, "the equation does not give k")
    limit = left_at

    # Independent check: evaluate each piece beside a with exact fractions.
    eps = Fraction(1, 10**9)
    below = s1 * (a - eps) ** 2 + k * (a - eps) + m
    above = s2 * (a + eps) + n - t * k
    check(abs(below - limit) < 1e-6 and abs(above - limit) < 1e-6, "the pieces disagree with the limit near a")

    left_text = join_terms([(s1, "x^2"), (1, "k x"), (m, "")]).replace("k x", "kx")
    right_text = join_terms([(s2, "x"), (n, ""), (-t, "k")])
    stem = phrase(rng, _STEMS, left=left_text, right=right_text, a=a)

    part_a = _expression_part(
        f"Find lim x→{a}⁻ f(x) in terms of k.",
        *left,
        2,
        [f"For x < {a} use {left_text}: at x = {a} this is {join_terms([(left[0], 'k'), (left[1], '')])}."],
        [
            (right[0], right[1], "ignored-side"),
            (1, left[1], "forgot-multiplier-on-parameter"),
            (a, s1 * 2 * a + m, "squared-as-doubled"),
        ],
    )
    part_b = _expression_part(
        f"Find lim x→{a}⁺ f(x) in terms of k.",
        *right,
        2,
        [f"For x ≥ {a} use {right_text}: at x = {a} this is {join_terms([(right[0], 'k'), (right[1], '')])}."],
        [
            (left[0], left[1], "ignored-side"),
            (-1, right[1], "forgot-multiplier-on-parameter"),
            (t, right[1], "sign-error-moving-term"),
        ],
    )
    # k from part_a = part_b when both are linear in k.
    c_a, c_b = "Subs(part_a, (k,), (0,))", "Subs(part_b, (k,), (0,))"
    m_a, m_b = f"(Subs(part_a, (k,), (1,)) - {c_a})", f"(Subs(part_b, (k,), (1,)) - {c_b})"
    part_c = _number_part(
        f"Find the value of k for which lim x→{a} f(x) exists.",
        k,
        2,
        [
            f"Equate the one-sided limits: {join_terms([(left[0], 'k'), (left[1], '')])} = {join_terms([(right[0], 'k'), (right[1], '')])}, so k = {k}."
        ],
        [
            (
                Fraction(right[1] - left[1], left[0] + right[0]) if left[0] + right[0] else None,
                "sign-error-moving-term",
            ),
            (Fraction(right[1] - left[1], left[0] + 1) if left[0] + 1 else None, "forgot-multiplier-on-parameter"),
            (limit, "confuses-k-with-the-limit"),
            (-k, "sign-error-moving-term"),
            (Fraction(right[1] - left[1], a), "forgot-multiplier-on-parameter"),
            (Fraction(right[1] + left[1], left[0] - right[0]), "sign-error-moving-term"),
        ],
        follow=f"({c_b} - {c_a})/({m_a} - {m_b})",
    )
    part_d = _number_part(
        f"For this value of k, find lim x→{a} f(x).",
        limit,
        1,
        [f"With k = {k}, the common value of the one-sided limits is {limit}."],
        [(k, "confuses-k-with-the-limit")]
        + [(left[0] * w + left[1], "wrong-parameter-substituted") for w in (k + 1, -k, k - 1)],
        follow="Subs(part_a, (k,), (part_c,))",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
