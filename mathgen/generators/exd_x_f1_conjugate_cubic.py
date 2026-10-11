"""Exam-tier source for ``senior-consider-conjugate-roots-for-polynomials-with-real-coefficients``.

The practice tier works with a monic quadratic whose roots are a conjugate pair. Here the polynomial
is a monic CUBIC with real coefficients, and the learner must use the conjugate root theorem to
recover the missing roots before applying Vieta's formulas. Two forms:

* ``roots``: a real root and one complex root are stated; the question asks for one named
  coefficient. The conjugate is a third root that nobody states.
* ``partial``: one complex root and one coefficient are stated; the real root has to be found from
  the sum of the roots before the second coefficient can be asked for.

Built backwards from the roots ``r`` and ``p +- qi``:

    (x - r)(x^2 - 2p x + p^2 + q^2) = x^3 + a x^2 + b x + c
    a = -(r + 2p),  b = p^2 + q^2 + 2 p r,  c = -r (p^2 + q^2)

All answers are integers.
"""

from __future__ import annotations

import random

from sympy import I, Poly, Symbol, expand

from ..core import check, check_equivalent, generator, join_terms, phrase, pick_distractors

x = Symbol("x")


def _complex_text(p: int, q: int) -> str:
    return join_terms([(p, ""), (q, "i")])


def _cubic_text(a, b, c, shown: str) -> str:
    """``x^3 + ... = `` with the coefficients named in ``shown`` ("abc" letters) left as letters."""
    out = "x^3"
    for name, value, var in (("a", a, "x^2"), ("b", b, "x"), ("c", c, "")):
        if name in shown:
            out += f" + {name}{var}"
        elif value:
            sign = "+" if value > 0 else "-"
            mag = abs(value)
            out += f" {sign} {mag if (mag != 1 or not var) else ''}{var}"
    return out


_ROOTS_WORDINGS = (
    "The cubic {poly} = 0 has real coefficients a, b and c. Two of its roots are {real} and {cplx}. Find {target}.",
    "Given that {real} and {cplx} are roots of {poly} = 0, where a, b and c are real, find {target}.",
)
_PARTIAL_WORDINGS = (
    "The cubic {poly} = 0 has real coefficients b and c, and one root is {cplx}. Find {target}.",
    "{cplx} is a root of {poly} = 0, where b and c are real numbers. Find {target}.",
)


def _coeffs(r, p, q):
    a = -(r + 2 * p)
    b = p * p + q * q + 2 * p * r
    c = -r * (p * p + q * q)
    return a, b, c


def _build(rng: random.Random) -> dict:
    for _ in range(200):
        r = rng.choice([n for n in range(-5, 6) if n != 0])
        p = rng.randint(-3, 3)
        q = rng.randint(1, 4)
        a, b, c = _coeffs(r, p, q)
        form = rng.choice(["roots", "partial"])
        target = rng.choice(["a", "b", "c"] if form == "roots" else ["b", "c"])
        truth = {"a": a, "b": b, "c": c}
        answer = truth[target]
        cplx = _complex_text(p, q)
        if form == "roots":
            text = phrase(
                rng,
                _ROOTS_WORDINGS,
                poly=_cubic_text(a, b, c, "abc"),
                real=str(r),
                cplx=cplx,
                target=f"the value of {target}",
            )
        else:
            text = phrase(
                rng, _PARTIAL_WORDINGS, poly=_cubic_text(a, b, c, "bc"), cplx=cplx, target=f"the value of {target}"
            )

        # Independent route: expand the product over all three roots with sympy.
        z = p + q * I
        full = Poly(expand((x - r) * (x - z) * (x - z.conjugate())), x)
        check_equivalent(full.coeff_monomial(x**2), a)
        check_equivalent(full.coeff_monomial(x), b)
        check_equivalent(full.coeff_monomial(1), c)

        mod2 = p * p + q * q
        # Principled slips, by target coefficient:
        #   a: no Vieta sign; forgot to double p; ignored the conjugate root; used q for p.
        #   b: no Vieta sign; dropped the cross term; z^2 in place of z * conj(z); forgot to double p.
        #   c: no Vieta sign; forgot the real root; z^2 in place of z * conj(z); times the pair's sum, not product.
        pool = {
            "a": [-a, r + p, r, -(r + 2 * q)],
            "b": [-b, mod2, p * p - q * q + 2 * p * r, mod2 + p * r],
            "c": [-c, mod2, -r * (p * p - q * q), -r * 2 * p],
        }
        cands = pool[target]
        wrong = []
        for v in cands:
            if v != answer and v not in wrong:
                wrong.append(v)
        if len(wrong) < 3:
            continue
        wrong = wrong[:3]
        check(all(w != answer for w in wrong))
        return dict(
            text=text,
            answer=answer,
            answer_text=str(answer),
            answer_latex=str(answer),
            working=[
                f"The conjugate {_complex_text(p, -q)} is also a root. The three roots are {r}, "
                f"{_complex_text(p, q)} and {_complex_text(p, -q)}, so the cubic is "
                f"(x - {r})(x^2 - {2 * p}x + {mod2}) and {target} = {answer}."
            ],
            distractors=pick_distractors(str(answer), [str(w) for w in wrong]),
        )
    check(False, "could not build a conjugate cubic question")


@generator(
    "exd_x_f1_conjugate_cubic",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f1_conjugate_cubic(rng: random.Random, tier: str = "exam") -> dict:
    """Coefficient of a real cubic recovered from a complex root via the conjugate root theorem."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _build(rng)
