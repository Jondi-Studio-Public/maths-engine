"""Exam-tier source for `logarithm-laws`: use the product law to solve a logarithmic equation.

The equation is ``log_b(x + p) + log_b(x - q) = n``. The learner has to combine the two logarithms with the product
law, remove the logarithm using the definition, solve the resulting quadratic, and reject the root for which a
logarithm would be of a negative number:

* (a) the number N in ``(x + p)(x - q) = N`` (2 marks);
* (b) the solution for x (3 marks, follow-through from (a)): the positive root of ``x^2 + (p - q)x - pq - N = 0``;
* (c) the value of ``log_b(x + p)`` (1 mark, follow-through from (b)).

Built backwards from ``u = b^i`` and ``v = b^j``: ``x + p = u`` and ``x - q = v`` give ``N = uv = b^(i+j)``,
``p + q = u - v`` and ``x = u - p``. The other root of the quadratic is ``-v - p``, which makes ``x - q = -u``
negative, so it is always rejected. The solution is confirmed by solving the quadratic with sympy.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import GRADER_TOLERANCE, check, generator, join_terms, latex_answer, pick_distractors

_BASES = {2: 8, 3: 5, 4: 5, 5: 4, 6: 3, 10: 2}  # base -> largest exponent n used
_MAX_PQ = 25


def _all_cases() -> list[tuple[int, int, int, int, int]]:
    """Every ``(b, i, j, p, q)`` with ``b^(i+j) <= b^n_max``, ``i > j >= 0`` and ``1 <= p, q <= _MAX_PQ``."""
    cases = []
    for b, n_max in _BASES.items():
        for i in range(1, n_max + 1):
            for j in range(0, i):
                if i + j > n_max:
                    continue
                gap = b**i - b**j
                for p in range(1, _MAX_PQ + 1):
                    q = gap - p
                    if 1 <= q <= _MAX_PQ:
                        cases.append((b, i, j, p, q))
    return cases


_CASES = _all_cases()

# (opening words, closing words) around the equation, so the plain text and its LaTeX are built from one source.
_STEMS = [
    ("Consider the equation", "."),
    ("The number x satisfies", "."),
    ("A student is asked to solve", " for x."),
]


@generator("exd_x_sc5_log_equation", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc5_log_equation(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: combine logs, remove the logarithm, solve the quadratic and reject the invalid root."""
    check(tier == "exam", f"unknown tier {tier!r}")
    b, i, j, p, q = rng.choice(_CASES)
    n = i + j
    u, v = b**i, b**j
    big_n = b**n
    x0 = u - p
    check(x0 - q == v and (x0 + p) * (x0 - q) == big_n, "backwards construction failed")
    # Independent route: solve the quadratic with sympy and keep the root that keeps both logarithms defined.
    x = sympy.Symbol("x")
    roots = sympy.solve(sympy.Eq((x + p) * (x - q), big_n), x)
    valid = [r for r in roots if r + p > 0 and r - q > 0]
    check(valid == [x0] and len(roots) == 2, "valid root disagrees with sympy")
    other = [r for r in roots if r != x0][0]
    other_val = int(other)
    check(other_val == -v - p, "rejected root is not -v - p")
    check(sympy.log(x0 + p, b) == i or math.isclose(math.log(x0 + p, b), i), "log value disagrees")

    # The "log of a sum" slip: (x + p) + (x - q) = b^n.
    sum_x = Fraction(big_n - p + q, 2)
    a_cands = [
        Distractor(str(n), "forgot-to-exponentiate", n),
        Distractor(str(b * n), "wrong-exponential-conversion", b * n),
        Distractor(str(n**b), "wrong-exponential-conversion", n**b),
        Distractor(str(b + n), "wrong-exponential-conversion", b + n),
    ]
    b_cands = [
        Distractor(str(other_val), "kept-the-invalid-root", other_val),
    ]
    if sum_x.denominator == 1:
        b_cands.append(Distractor(str(sum_x), "log-sum-as-log-of-sum", int(sum_x)))
    b_cands += [
        Distractor(str(x0 + 1), "slip-solving-the-quadratic", x0 + 1),
        Distractor(str(x0 - 1), "slip-solving-the-quadratic", x0 - 1),
        Distractor(
            str(big_n // (p + q) if big_n % (p + q) == 0 else x0 + 2),
            "slip-solving-the-quadratic",
            big_n // (p + q) if big_n % (p + q) == 0 else x0 + 2,
        ),
    ]
    c_cands = [
        Distractor(str(n), "evaluated-wrong-log-expression", n),
        Distractor(str(j), "evaluated-wrong-log-expression", j),
        Distractor(str(u), "evaluated-wrong-log-expression", u),
        Distractor(str(i + 1), "off-by-one-endpoint", i + 1),
    ]
    a_d = pick_distractors(str(big_n), a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(str(x0), b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(str(i), c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors survive")

    disc = (p + q) ** 2 + 4 * big_n
    check(math.isqrt(disc) ** 2 == disc, "discriminant is not a perfect square")
    ft_b = f"((({q} - {p}) + sqrt(({p} + {q})**2 + 4*part_a))/2)"
    lead, tail = rng.choice(_STEMS)
    return dict(
        text=f"{lead} log_{{{b}}}(x + {p}) + log_{{{b}}}(x - {q}) = {n}{tail}",
        latex=(rf"\text{{{lead} }}\log_{{{b}}}(x+{p})+\log_{{{b}}}(x-{q})={n}\text{{{tail}}}"),
        parts=[
            dict(
                prompt=f"Use the log laws to remove the logarithms. The equation becomes (x + {p})(x - {q}) = N, "
                "where N is a whole number. Find N.",
                prompt_latex=(
                    r"\text{Use the log laws to remove the logarithms. The equation becomes }"
                    rf"(x+{p})(x-{q})=N\text{{, where N is a whole number. Find N.}}"
                ),
                answer_kind="number",
                answer=big_n,
                answer_text=str(big_n),
                answer_latex=latex_answer(str(big_n)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"log_{b}(x + {p}) + log_{b}(x - {q}) = log_{b}((x + {p})(x - {q})).",
                    f"So (x + {p})(x - {q}) = {b}^{n} = {big_n}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Solve for x. Only give the value that keeps both logarithms defined.",
                answer_kind="number",
                answer=x0,
                answer_text=str(x0),
                answer_latex=latex_answer(str(x0)),
                marks=3,
                sensible_range=(0, None),
                follow_through=ft_b,
                working=[
                    f"{join_terms([(1, 'x^2'), (p - q, 'x'), (-(p * q + big_n), '')])} = 0, so x = {x0} or x = {other_val}.",
                    f"x = {other_val} makes x - {q} negative, so it is rejected. x = {x0}.",
                ],
                distractors=b_d,
            ),
            dict(
                prompt=f"Hence find the value of log_{{{b}}}(x + {p}).",
                prompt_latex=rf"\text{{Hence find the value of }}\log_{{{b}}}(x+{p})\text{{.}}",
                answer_kind="number",
                answer=i,
                answer_text=str(i),
                answer_latex=latex_answer(str(i)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"log(part_b + {p})/log({b})",
                working=[f"x + {p} = {u} = {b}^{i}, so log_{b}({u}) = {i}."],
                distractors=c_d,
            ),
        ],
    )
