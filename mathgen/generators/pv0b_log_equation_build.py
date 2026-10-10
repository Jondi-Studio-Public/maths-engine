"""The unknown inside a logarithm
(``senior-solve-simple-equations-involving-logarithmic-functions-algebraically-and``, ACMMM157).

Scope: solve an
equation in which the unknown is inside a logarithm, of the form
log_b(ax + c) = k, giving the exact value of x. The graphical half is
dropped -- algebraic solving only.

Four tiers, each changing the DECISION the learner makes:

- easy -- log_b(ax + c) = k, k a positive integer, a != 1 and c != 0.
  Rewrite in exponential form ax + c = b^k, then solve the linear equation.
- medium -- the log is not isolated: m*log_b(ax + c) + n = p (m != 1 on most
  draws), and k (the isolated log value) may be 0 or negative. Isolate the
  log first, then handle b^k <= 1.
- hard -- a numeric log term must be combined first: log_b(ax+c) + log_b(d)
  = k or log_b(ax+c) - log_b(d) = k. Apply the product/quotient law to reach
  a single log of a linear argument before converting.
- exam -- the unknown is in TWO logs: log_b(ax+c) - log_b(dx+e) = k, or
  log_b(ax+c) = log_b(dx+e) + k. Combine by the quotient law, solve the
  resulting linear equation, and both arguments must be positive at x
  (checked every seed, by construction and by an explicit re-check).

Every tier is constructed so the argument(s) of every log are exactly a
positive (integer or simple-fraction) power of ``b`` at the solution --
never approximate, so the answer is always an exact `Fraction`
(`answer: fraction`, matching `f10-exponential-relations-equations`'s own
``{"p": ..., "q": ...}`` shape). Integer bases 2..10 only (no ln/e).

No tier may reduce to log_b(x) = k (the shipping neighbours
`senior-define-logarithms-as-indices-a-x-b-is` and
`senior-define-the-natural-logarithm-ln-x-log-ex` already own that): every
argument's coefficient `a`/`d` is never 1 and every constant `c`/`e` is
never 0, on every tier.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from ..core import (
    check,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    rational_text,
)

# -- shared construction pools -----------------------------------------------

_BASES = tuple(range(2, 11))  # integer bases 2..10
_A_CHOICES = (2, 3, 4, 5, 6, 7, 8, 9)  # never 1 -- argument is never bare x
_D_CHOICES = (2, 3, 4, 5, 6, 7, 8, 9)  # never 1, same reason (hard/exam)
_NONZERO_SMALL = tuple(v for v in range(-9, 10) if v != 0)  # c, e pools

_MAX_DISTRACTOR_MAGNITUDE = 10**7  # guards `power-wrong-way` against overflow noise

# A distractor a learner would never produce (e.g. `59043/4`, `-1007/241`) is dropped, falling
# through to the next candidate by fixed priority (the list order `_core_distractors`/tier code
# builds).
_MAX_DISTRACTOR_DENOMINATOR = 24
_MAX_DISTRACTOR_ABS_VALUE = 1000


def _arg_text(coef: int, const: int) -> str:
    """``ax + c`` the way a student would write it -- also valid bare LaTeX."""
    return join_terms([(coef, "x"), (const, "")])


# Every phrasing states how to type the answer. The instruction is repeated verbatim in `latex`.
_ANSWER_FORM = "Give x as an integer or a fraction in lowest terms."
_ANSWER_FORM_LATEX = r"\text{ Give } x \text{ as an integer or a fraction in lowest terms.}"

_OPENINGS = (
    (
        "Solve {eq} for x.",
        r"\text{{Solve }} {eqL} \text{{ for }} x.",
    ),
    (
        "Find the exact value of x such that {eq}.",
        r"\text{{Find the exact value of }} x \text{{ such that }} {eqL}.",
    ),
    (
        "Given that {eq}, find x.",
        r"\text{{Given that }} {eqL}\text{{, find }} x.",
    ),
)


def _phrase(rng, eq_text: str, eq_latex: str) -> tuple[str, str]:
    t, l = rng.choice(_OPENINGS)
    return (
        t.format(eq=eq_text) + " " + _ANSWER_FORM,
        l.format(eqL=eq_latex) + _ANSWER_FORM_LATEX,
    )


def _core_distractors(b: int, k: int, a: int, c: int, value: Fraction) -> list[Fraction]:
    """The four generic-misconception values, evaluated against a log value
    ``k`` and the target argument ``value`` (== b**k unless a caller passes a
    different target, e.g. hard/exam's post-combination value):

    - ``multiplied-not-powered``: ``b*k`` used in place of ``b**k``.
    - ``power-wrong-way``: ``k**b`` used in place of ``b**k`` (base/exponent
      swapped), skipped when it would be unreasonably large.
    - ``forgot-c``: divides the target value by ``a`` without subtracting c.
    - ``sign-of-c``: adds c instead of subtracting it.
    - ``forgot-to-exponentiate``: treats k itself as the argument's value
      (never raises b to the power at all).
    - ``off-by-one-exponent``: uses ``b**(k+1)`` in place of ``b**k``.
    """
    out = [
        (Fraction(b * k) - c) / a,
        Fraction(value) / a,
        (Fraction(value) + c) / a,
        (Fraction(k) - c) / a,
        (Fraction(b) ** (k + 1) - c) / a,
    ]
    power_wrong_way = Fraction(k) ** b
    if abs(power_wrong_way) < _MAX_DISTRACTOR_MAGNITUDE:
        out.insert(1, (power_wrong_way - c) / a)
    return out


def _dedupe(values: list[Fraction]) -> list[Fraction]:
    seen: set[Fraction] = set()
    out: list[Fraction] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _shape_matches(text: str, answer_text: str) -> bool:
    return ("/" in text) == ("/" in answer_text)


_DIGIT_RUN_RE = re.compile(r"\d+")


def _printed_magnitudes(equation_text: str) -> set[int]:
    """Every literal digit-run in the equation body, as a positive
    magnitude -- a distractor formatted as a bare integer must not equal any
    of these regardless of the SIGN context it was printed under (a
    generator that only excluded the signed value `c` itself still let a
    distractor equal the printed MAGNITUDE of `c`, e.g. a distractor "4"
    beside a prompt showing "... - 4)" -- proven red, see the build test's
    `test_no_distractor_equals_a_printed_number_proven_red_on_unguarded_draw`)."""
    return {int(tok) for tok in _DIGIT_RUN_RE.findall(equation_text)}


def _finalize(
    *,
    text: str,
    latex: str,
    x: Fraction,
    candidates: list[Fraction],
) -> dict:
    answer_text = rational_text(x)
    # Every number printed in the prompt, EXCLUDING the trailing "Give x as
    # an integer or a fraction in lowest terms." sentence (which carries no
    # digits of its own, but split defensively regardless).
    equation_part = text.split(" Give x as")[0]
    printed_magnitudes = _printed_magnitudes(equation_part)
    filtered = [
        v
        for v in _dedupe(candidates)
        if v != x
        and not (v.denominator == 1 and abs(v.numerator) in printed_magnitudes)
        and v.denominator <= _MAX_DISTRACTOR_DENOMINATOR
        and abs(v) <= _MAX_DISTRACTOR_ABS_VALUE
    ]
    candidate_texts = [rational_text(v) for v in filtered]
    distractors = pick_distractors(answer_text, candidate_texts, n=3)

    # Never let the answer be the only option of its shape (integer vs. fraction) -- the
    # `midpoint_coordinate_scalar` defect class. Guarantee at least one shape-matching option by
    # offering x +/- n (n = 1, 2, 3...), which always shares x's own denominator (Fraction reduces
    # (p + n*q)/q to the same q whenever gcd(p, q) == 1, which it is by construction). The search
    # widens until it finds an offset that collides with neither the answer, an existing distractor,
    # nor a number literally printed in the prompt -- a small, unbounded search over a finite
    # exclusion set always terminates.
    if not any(_shape_matches(d, answer_text) for d in distractors):
        existing = set(distractors)
        for magnitude in range(1, 50):
            for offset in (magnitude, -magnitude):
                candidate_value = x + offset
                shape_safe = rational_text(candidate_value)
                collides_printed = (
                    candidate_value.denominator == 1 and abs(candidate_value.numerator) in printed_magnitudes
                )
                if shape_safe != answer_text and shape_safe not in existing and not collides_printed:
                    distractors = distractors[:-1] + [shape_safe]
                    break
            else:
                continue
            break

    return dict(
        text=text,
        latex=latex,
        answer={"p": x.numerator, "q": x.denominator},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )


# =============================================================================
# easy: log_b(ax + c) = k, k a positive integer
# =============================================================================


def _build_easy(rng) -> dict:
    b = rng.choice(_BASES)
    k = rng.randint(1, 3)
    a = rng.choice(_A_CHOICES)
    c = rng.choice(_NONZERO_SMALL)

    value = Fraction(b) ** k  # the argument's value, always a positive integer
    x = (value - c) / Fraction(a)

    # Independent check: a genuinely different route (floating change-of-base)
    # confirms the printed base/argument/RHS really do satisfy b**k == value.
    check(abs(math.log(float(value), b) - k) < 1e-9, "easy: k does not recompute via change-of-base")
    check(Fraction(a) * x + c == value, "easy: linear reconstruction failed")

    arg = _arg_text(a, c)
    text, latex = _phrase(rng, f"log_{b}({arg}) = {k}", rf"\log_{{{b}}}({arg}) = {k}")

    candidates = _core_distractors(b, k, a, c, value)
    return _finalize(text=text, latex=latex, x=x, candidates=candidates)


# =============================================================================
# medium: m * log_b(ax + c) + n = p (log not isolated); k may be 0/negative
# =============================================================================

_MEDIUM_K = tuple(range(-3, 4))
_MEDIUM_M_POOL = (2, 2, 2, 3, 3, 4, 1)  # m == 1 on ~1/7 of draws only
_MEDIUM_N = tuple(range(-6, 7))

# When k < 0 the isolated log VALUE is 1/b**|k|; letting `a` range over `_A_CHOICES` too compounds
# that denominator with `a` (e.g. a=9, k=-2, b=9 keyed x = 244/405). Restricting b so b**|k| itself
# stays <= 12, and fixing a = 1 for these draws, makes x's denominator EXACTLY b**|k| (gcd(1 -
# c*b**|k|, b**|k|) == 1 always, since the numerator is congruent to 1 mod b**|k|) -- never
# compounded further.
_MEDIUM_NEG_K_BASES = {
    -1: _BASES,  # b**1 <= 10
    -2: (2, 3),  # b**2 <= 9
    -3: (2,),  # b**3 == 8
}


def _build_medium(rng) -> dict:
    k = rng.choice(_MEDIUM_K)
    if k < 0:
        b = rng.choice(_MEDIUM_NEG_K_BASES[k])
        a = 1
    else:
        b = rng.choice(_BASES)
        a = rng.choice(_A_CHOICES)
    c = rng.choice(_NONZERO_SMALL)
    m = rng.choice(_MEDIUM_M_POOL)
    n = rng.choice(_MEDIUM_N)
    p = m * k + n

    value = Fraction(b) ** k  # 1 when k == 0, a proper fraction when k < 0
    x = (value - c) / Fraction(a)

    check(abs(math.log(float(value), b) - k) < 1e-9, "medium: k does not recompute via change-of-base")
    check(m * k + n == p, "medium: p does not reconstruct from m, k, n")
    check(Fraction(a) * x + c == value, "medium: linear reconstruction failed")

    arg = _arg_text(a, c)
    m_text = "" if m == 1 else f"{m} "
    if n > 0:
        n_part, n_partL = f" + {n}", f" + {n}"
    elif n < 0:
        n_part, n_partL = f" - {-n}", f" - {-n}"
    else:
        n_part, n_partL = "", ""
    eq_text = f"{m_text}log_{b}({arg}){n_part} = {p}"
    eq_latex = rf"{m_text}\log_{{{b}}}({arg}){n_partL} = {p}"
    text, latex = _phrase(rng, eq_text, eq_latex)

    candidates = _core_distractors(b, k, a, c, value)
    # did-not-isolate: treats p as though it were already the isolated log value
    b_to_p = Fraction(b) ** p
    if abs(b_to_p) < _MAX_DISTRACTOR_MAGNITUDE:
        did_not_isolate = (b_to_p - c) / a
        candidates.append(did_not_isolate)

    return _finalize(
        text=text,
        latex=latex,
        x=x,
        candidates=candidates,
    )


# =============================================================================
# hard: log_b(ax + c) +/- log_b(d) = k -- combine by product/quotient law first
# =============================================================================

_HARD_K = (1, 2, 3)


def _build_hard(rng) -> dict:
    sign = rng.choice(("+", "-"))
    b = rng.choice(_BASES)
    k = rng.choice(_HARD_K)
    a = rng.choice(_A_CHOICES)
    c = rng.choice(_NONZERO_SMALL)

    base_value = Fraction(b) ** k  # k >= 1 always, so this is a positive integer
    base_value_int = base_value.numerator  # denominator is always 1 here

    # "+": log_b((ax+c)*d) = k => ax+c = b^k / d -- keeping the combined
    # argument a positive INTEGER needs d | b^k.
    # Restrict d to a divisor of b^k actually present in the pool; fall back
    # to the quotient form ("-": ax+c = b^k * d, always an integer) when b^k
    # has none in range.
    if sign == "+":
        divisors = [dd for dd in _D_CHOICES if base_value_int % dd == 0]
        if divisors:
            d = rng.choice(divisors)
        else:
            sign = "-"
            d = rng.choice(_D_CHOICES)
    else:
        d = rng.choice(_D_CHOICES)

    value = base_value / d if sign == "+" else base_value * d
    x = (value - c) / Fraction(a)

    check(abs(math.log(float(base_value), b) - k) < 1e-9, "hard: k does not recompute via change-of-base")
    check(Fraction(a) * x + c == value, "hard: linear reconstruction failed")
    check(value > 0, "hard: combined argument is not positive")

    arg = _arg_text(a, c)
    op = "+" if sign == "+" else "-"
    eq_text = f"log_{b}({arg}) {op} log_{b}({d}) = {k}"
    eq_latex = rf"\log_{{{b}}}({arg}) {op} \log_{{{b}}}({d}) = {k}"
    text, latex = _phrase(rng, eq_text, eq_latex)

    # generic misconceptions, computed as if the second log term did not
    # exist at all (forgot to combine it in first).
    candidates = _core_distractors(b, k, a, c, base_value)

    # law-wrong-way: product used where quotient needed (or vice versa) --
    # the SAME two named misconceptions from the contract, in one value.
    wrong_value = base_value * d if sign == "+" else base_value / d
    candidates.append((wrong_value - c) / a)
    # law-wrong-way (additive variant): d added/subtracted inside the
    # argument instead of multiplied/divided in.
    additive_value = base_value + d if sign == "+" else base_value - d
    candidates.append((additive_value - c) / a)

    return _finalize(
        text=text,
        latex=latex,
        x=x,
        candidates=candidates,
    )


# =============================================================================
# exam: log_b(ax+c) - log_b(dx+e) = k, or log_b(ax+c) = log_b(dx+e) + k
# (unknown inside two logs; both arguments must be positive at the solution)
# =============================================================================

_EXAM_K = (1, 2, 3)

# Solving the linear SYSTEM for x after drawing b/k/a/c/d/e independently gave denominators in the
# thousands (e.g. `883/1123`) because `denom = a - b**k * d` can be tiny relative to the numerator.
# Constructed forward instead: draw x (an integer) and a small positive "count" for the second
# argument FIRST, then derive c and e so both arguments land exactly on `arg1 = b**k * arg2` -- x's
# own denominator is then always 1 (<= 12), and c/e are bounded by construction (retried until |c|,
# |e| <= 20).
_EXAM_X = tuple(v for v in range(-15, 16) if v != 0)
_EXAM_BK_PAIRS = tuple((b, k) for b in _BASES for k in _EXAM_K if b**k <= 40)


def _build_exam(rng) -> dict:
    form = rng.choice(("difference", "equated"))

    for _ in range(500):
        x = Fraction(rng.choice(_EXAM_X))
        a = rng.choice(_A_CHOICES)
        d = rng.choice(_D_CHOICES)
        b, k = rng.choice(_EXAM_BK_PAIRS)
        big_b = Fraction(b) ** k  # always a positive integer, k >= 1, <= 40

        # a == b^k * d makes the equation degenerate: a*x+c = b^k*(d*x+e)
        # reduces to a constant identity (no x-coefficient left), never a
        # single well-posed solution.
        if Fraction(a) == big_b * d:
            continue

        # Search for a positive "count" arg2 that makes arg1 = b^k * arg2
        # land close to a*x -- this is what keeps |c| <= 20 achievable for
        # ANY drawn x (a random arg2 only matches by luck, which collapsed
        # the exam tier to 14 distinct keys over 200 seeds -- proven during
        # development, not shipped).
        target = a * x
        approx = round(target / big_b)
        arg2 = None
        for delta in range(0, 9):
            for cand in {approx + delta, approx - delta} if delta else {approx}:
                if cand < 1:
                    continue
                cand_arg1 = big_b * cand
                cand_c = cand_arg1 - target
                if cand_c == 0 or abs(cand_c) > 20:
                    continue
                cand_e = cand - d * x
                if cand_e == 0 or abs(cand_e) > 20:
                    continue
                arg2, arg1, c, e = cand, cand_arg1, cand_c, cand_e
                break
            if arg2 is not None:
                break
        if arg2 is None:
            continue
        break
    else:
        raise RuntimeError("could not draw a valid pv0b_log_equation exam question")

    check(arg1 > 0 and arg2 > 0, "exam: an argument is not positive at the solution")
    check(Fraction(a) * x + c == arg1, "exam: first argument reconstruction failed")
    check(Fraction(d) * x + e == arg2, "exam: second argument reconstruction failed")
    # Independent check: recompute k via floating change of base -- a route
    # that never touches the a/c/d/e construction above.
    check(abs(math.log(float(arg1 / arg2), b) - k) < 1e-9, "exam: k does not recompute via change-of-base")

    arg = _arg_text(a, int(c))
    arg2_text = _arg_text(d, int(e))
    if form == "difference":
        eq_text = f"log_{b}({arg}) - log_{b}({arg2_text}) = {k}"
        eq_latex = rf"\log_{{{b}}}({arg}) - \log_{{{b}}}({arg2_text}) = {k}"
    else:
        eq_text = f"log_{b}({arg}) = log_{b}({arg2_text}) + {k}"
        eq_latex = rf"\log_{{{b}}}({arg}) = \log_{{{b}}}({arg2_text}) + {k}"
    text, latex = _phrase(rng, eq_text, eq_latex)

    candidates: list[Fraction] = []
    # quotient-inverted: solves (dx+e) = b^k(ax+c) instead of the other way
    denom2 = Fraction(d) - big_b * a
    if denom2 != 0:
        candidates.append((big_b * c - e) / denom2)
    # argument-negative: sign-of-k error -- uses b^-k in the same equation
    inv_b = Fraction(1) / big_b
    denom3 = Fraction(a) - inv_b * d
    if denom3 != 0:
        candidates.append((inv_b * e - c) / denom3)
    # ignores-second-log: pretends the equation was log_b(ax+c) = k alone
    candidates.append((big_b - c) / a)
    # sign-of-e: flips the sign of e inside the correctly combined equation
    # (a*x + c = b^k * (d*x - e) instead of + e)
    denom4 = Fraction(a) - big_b * d
    if denom4 != 0:
        candidates.append((-big_b * e - c) / denom4)
    # off-by-one-count: misreads arg2's count by one (arg2 +/- 1) -- small
    # and bounded by construction (arg2 is drawn from a fixed small pool).
    candidates.append((big_b * (arg2 + 1) - c) / a)
    if arg2 > 1:
        candidates.append((big_b * (arg2 - 1) - c) / a)
    # combined-additively: treats b^k*(dx+e) as b^k + (dx+e) instead of a
    # product -- small and bounded (big_b <= 40, arg2 <= 8).
    candidates.append((big_b + arg2 - c) / a)
    # sign-of-c: uses ax - c = b^k*(dx+e) instead of ax + c
    candidates.append((arg1 + c) / a)

    return _finalize(
        text=text,
        latex=latex,
        x=x,
        candidates=candidates,
    )


@generator(
    "pv0b_log_equation",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def log_equation(rng, tier="easy"):
    """Solve an equation in which the unknown is inside a logarithm, of the
    form log_b(ax + c) = k (or a variant requiring one log law / two logs),
    giving the exact value of x."""
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier: {tier!r}")
