"""Generates `exd_x_r3f_quotient_ring_roots` questions: an exam chain in Z_p[x]/(m(x)) using a root of m.

The modulus m(x) = (x - r1)(x - r2) has two roots in Z_p. A quadratic a(x) and a linear b(x) are multiplied in the
quotient ring. The chain:

(a) a(x)b(x) reduced modulo m(x), coefficients read modulo p           expression, 3 marks
(b) the smaller root of m(x) in Z_p                                      number,     1 mark
(c) the reduced product evaluated at that root, modulo p                 number,     2 marks

Because m(r) = 0 in Z_p, evaluating at a root is a ring homomorphism, so the answer to (c) is also a(r) b(r)
modulo p. The key to (c) is derived that way (from the factors, not from the reduced polynomial) and the two must
agree. Part (c) follows through from (a) and (b), so a learner with a wrong reduction, or the other root, still
earns (c) for evaluating their own results correctly.
"""

from __future__ import annotations

import random

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, latex_answer, pick_distractors
from ._exd_r3f_parts import number_part
from ._exd_distinct_parts import distinct_part_answers

_PRIMES = (5, 7, 11, 13)


def _times(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _reduce(poly: list[int], x_squared: tuple[int, int]) -> list[int]:
    """Reduce `poly` (coefficients low to high) using x^2 = x_squared[0] x + x_squared[1]; the result has two terms."""
    poly = list(poly)
    while len(poly) > 2:
        top = poly.pop()
        degree = len(poly)  # the degree of the popped term
        # top * x^degree = top * x^(degree-2) * (s x + c)
        poly[degree - 1] += top * x_squared[0]
        poly[degree - 2] += top * x_squared[1]
    return poly + [0] * (2 - len(poly))


def _text(poly: list[int]) -> str:
    """Coefficients low to high as `3*x^2 + 2*x + 5` (the form the grader reads back)."""
    terms = []
    for degree in range(len(poly) - 1, -1, -1):
        c = poly[degree]
        if c == 0:
            continue
        var = "" if degree == 0 else ("x" if degree == 1 else f"x^{degree}")
        terms.append((c, var))
    if not terms:
        return "0"
    out = ""
    for i, (c, var) in enumerate(terms):
        mag = abs(c)
        body = str(mag) if not var else (var if mag == 1 else f"{mag}*{var}")
        if i == 0:
            out += ("-" if c < 0 else "") + body
        else:
            out += f" {'-' if c < 0 else '+'} {body}"
    return out


def _prompt_poly(poly: list[int]) -> str:
    return join_terms(
        [(c, "" if d == 0 else ("x" if d == 1 else f"x^{d}")) for d, c in reversed(list(enumerate(poly)))]
    )


_STEMS = (
    "Let m(x) = {m} in Z_{p}[x], and work in the quotient ring Z_{p}[x]/(m(x)). Let a(x) = {a} and b(x) = {b}, "
    "with every coefficient read modulo {p}.",
    "In the quotient ring Z_{p}[x]/(m(x)), where m(x) = {m}, let a(x) = {a} and b(x) = {b}. "
    "Coefficients are read modulo {p}.",
)


@generator("exd_x_r3f_quotient_ring_roots", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_quotient_ring_roots(rng: random.Random, tier: str = "exam") -> dict:
    """Multiply and reduce in Z_p[x]/(m), find a root of m, and evaluate the reduced product there."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    p = rng.choice(_PRIMES)
    for _ in attempts(2000, "a ring where every part is non-degenerate"):
        r1, r2 = sorted(rng.sample(range(1, p), 2))
        beta, gamma = (-(r1 + r2)) % p, (r1 * r2) % p
        a = [rng.randrange(p), rng.randrange(1, p), rng.randrange(1, p)]  # a0, a1, a2
        b = [rng.randrange(p), rng.randrange(1, p)]  # b0, b1
        product = _times(a, b)
        reduced = [c % p for c in _reduce(product, (-beta % p, -gamma % p))]
        root = r1
        value = (reduced[1] * root + reduced[0]) % p
        if reduced[0] and reduced[1] and value and beta and gamma:
            break

    # Independent checks: m really has the two roots, the reduced product agrees with the factors at both roots,
    # and the product evaluates to the same residue directly.
    def m_at(v):
        return (v * v + beta * v + gamma) % p

    def a_at(v):
        return (a[0] + a[1] * v + a[2] * v * v) % p

    def b_at(v):
        return (b[0] + b[1] * v) % p

    check(m_at(r1) == 0 and m_at(r2) == 0, "m does not vanish at its roots")
    check([v for v in range(p) if m_at(v) == 0] == [r1, r2], "m has other roots")
    for r in (r1, r2):
        check((reduced[1] * r + reduced[0]) % p == a_at(r) * b_at(r) % p, "reduced product disagrees at a root")
    value_direct = a_at(root) * b_at(root) % p
    check(value == value_direct, "evaluation disagrees with the factors")
    other_value = (reduced[1] * r2 + reduced[0]) % p

    modulus = [gamma, beta, 1]
    unreduced = [c % p for c in product]
    int_reduced = _reduce(product, (-beta, -gamma))  # polynomial reduction done, coefficients left unreduced
    wrong_sign = [c % p for c in _reduce(product, (beta, gamma))]
    dropped = [c % p for c in _reduce(product, (-beta % p, 0))]
    a_text = _text(reduced)
    a_candidates = [
        (unreduced, "forgets-to-reduce-mod-m"),
        (int_reduced, "forgets-mod-p-reduction"),
        (wrong_sign, "sign-error-in-reduction"),
        (dropped, "drops-a-term-of-modulus"),
    ]
    a_options = [Distractor(_text(c), tag, _text(c)) for c, tag in a_candidates if c != reduced]
    root_candidates = [
        (r2, "reports-larger-root-for-smaller"),
        ((-r1) % p, "sign-flip-both-roots"),
        (beta, "reports-linear-coefficient-as-root"),
        (gamma, "reports-constant-term-as-root"),
    ]
    value_candidates = [
        (other_value, "evaluated-at-the-other-root"),
        (reduced[1] * root + reduced[0], "forgets-mod-p-reduction"),
        ((a_at(root) + b_at(root)) % p, "evaluations-added-not-multiplied"),
        (reduced[0], "evaluated-at-zero-instead-of-the-root"),
    ]

    stem = rng.choice(_STEMS).format(m=_prompt_poly(modulus), a=_prompt_poly(a), b=_prompt_poly(b), p=p)
    residues = f"a residue modulo {p}, an integer from 0 to {p - 1}"
    ring = f"Z_{p}[x]/(m(x))"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=(
                    f"Compute a(x)b(x) in {ring} as a polynomial of degree at most 1, with each coefficient {residues}."
                ),
                answer_kind="expression",
                answer=a_text,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=3,
                working=[
                    f"a(x)b(x) = {_text(unreduced)} (mod {p}).",
                    f"Using x\u00b2 = {_text([(-gamma) % p, (-beta) % p])}, this reduces to {a_text}.",
                ],
                distractors=pick_distractors(a_text, a_options, 3),
            ),
            number_part(
                f"The polynomial m(x) has two roots in Z_{p}. Find the smaller root, as {residues}.",
                root,
                root_candidates,
                marks=1,
                working=[
                    f"m({r1}) = {m_at(r1)} (mod {p}), and m({r2}) = {m_at(r2)} (mod {p}); the smaller root is {root}."
                ],
            ),
            number_part(
                f"Evaluate your answer to (a) at x equal to your answer to (b), giving the result as {residues}.",
                value,
                value_candidates,
                marks=2,
                follow_through=(f"Subs(part_a, (x,), (part_b,)) - {p}*floor(Subs(part_a, (x,), (part_b,))/{p})"),
                working=[f"{a_text} at x = {root} gives {value} (mod {p})."],
            ),
        ],
    )
