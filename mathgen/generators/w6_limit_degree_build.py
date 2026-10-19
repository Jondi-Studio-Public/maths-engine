"""Generates `w6_limit_degree` questions (easy, medium, hard) for the skill ``limit-degree-comparison``.

``limit-degree-comparison`` (``limits``, answer ``expression``): limits of rational (and square-root) expressions as x
tends to infinity or minus infinity, found by comparing the degrees of the numerator and denominator. A single mismatched
fraction can only give 0, infinity or minus infinity, so the tiers ask for more than that to keep the answers from being
guessable:

* easy: the limit of one fraction whose degrees differ (0, ∞ or -∞), or the extreme whole-number power p for which
  ``(a x^p + b) / (c x^q + d)`` still tends to 0 (the largest p) or already has no finite limit (the smallest p).
* medium: a sum of a mismatched fraction and an equal-degree fraction (the mismatched term vanishes, or it dominates), and a
  fraction taken as x → -∞, where an odd difference of degrees flips the sign of an infinite limit.
* hard: degrees hidden in factored form, ``(x^2 + a)^m (px + q)^k / (c x^s (x + d)^j)``, where the degree is a sum of
  factor degrees and the leading coefficient a product of powers; and a polynomial over a square root,
  ``(a x^m + ...) / √(s² x^(2n) + ...)``, where the root halves the degree and √(x²) is |x|, so the direction x → -∞
  matters.

Every answer is built from leading terms, then checked a second way: the expression is evaluated exactly (or in floating
point for the root) at x = ±10^6 and compared with the limit.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, phrase, rational_text
from ._w6a_kit import enough, retry_draw

_BIG = 10**6

# -- text -----------------------------------------------------------------------------------------------------------


def _xp(p: int) -> str:
    return "" if p == 0 else "x" if p == 1 else f"x^{p}"


def _text(terms) -> str:
    """``[(coef, power), ...]`` as ``3x^2 - x + 5``."""
    return join_terms([(c, _xp(p)) for c, p in terms])


def _evalp(terms, x) -> object:
    return sum(c * x**p for c, p in terms)


def _poly(rng, deg: int, lead: int | None = None, extra: int | None = None):
    """A polynomial of degree `deg` with leading coefficient `lead` and up to `extra` lower terms."""
    if lead is None:
        lead = rng.choice((-1, 1)) * rng.randint(1, 9)
    terms = [(lead, deg)]
    k = rng.randint(0, min(2, deg)) if extra is None else min(extra, deg)
    for p in sorted(rng.sample(range(deg), k), reverse=True):
        terms.append((rng.choice((-1, 1)) * rng.randint(1, 9), p))
    return terms


def _sign(v) -> int:
    return (v > 0) - (v < 0)


# -- the answer: a Fraction, or "inf" / "-inf" -----------------------------------------------------------------------


def _fields(value) -> dict:
    text = _value_text(value)
    return dict(answer=text, answer_text=text, answer_latex=text, answer_kind="expression")


def _value_text(value) -> str:
    if value == "inf":
        return "∞"
    if value == "-inf":
        return "-∞"
    return rational_text(Fraction(value))


def _dists(answer, cands, n: int = 3):
    """Tagged distractors from ``(value, tag)`` candidates, each a different answer from the key and from each other."""
    seen = {_value_text(answer)}
    out = []
    for value, tag in cands:
        if value is None:
            continue
        text = _value_text(value) if value not in ("dne",) else "does not exist"
        if text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, text))
        if len(out) == n:
            break
    enough(out)
    return out


def _flip(v):
    return {"inf": "-inf", "-inf": "inf"}[v]


def _infinite(sign: int) -> str:
    return "inf" if sign > 0 else "-inf"


def _rule(lead_n, m, lead_d, n, direction: int = 1):
    """The limit of ``lead_n x^m / (lead_d x^n)`` as x → direction·∞, from the degrees alone."""
    if m < n:
        return Fraction(0)
    ratio = Fraction(lead_n, lead_d) if isinstance(lead_n, int) and isinstance(lead_d, int) else lead_n / lead_d
    if m == n:
        return ratio
    sign = _sign(ratio) * (direction if (m - n) % 2 else 1)
    return _infinite(sign)


def _close(f_value: float, answer) -> bool:
    if answer == "inf":
        return f_value > 1e3
    if answer == "-inf":
        return f_value < -1e3
    return abs(f_value - float(answer)) < 2e-3 * max(1.0, abs(float(answer)))


def _verify(func, answer, direction: int = 1) -> None:
    """The second route: evaluate at x = ±10^6 and compare with the limit."""
    check(_close(float(func(direction * _BIG)), answer), f"the value at {direction * _BIG} disagrees with {answer}")


def _frac_eval(num, den):
    return lambda x: Fraction(_evalp(num, x), _evalp(den, x))


_ASK = (
    "Find the limit as x → {arrow} of {f}.",
    "Evaluate the limit of {f} as x → {arrow}.",
    "What is the limit of {f} as x → {arrow}?",
)
_NOTE = " Give the limit as a number, ∞ or -∞."


def _arrow(direction: int) -> str:
    return "∞" if direction > 0 else "-∞"


# -- easy -----------------------------------------------------------------------------------------------------------


def _easy_limit(rng):
    for _ in attempts(200, "a mismatched pair of degrees"):
        m, n = rng.randint(0, 4), rng.randint(0, 4)
        if m != n and max(m, n) >= 1:
            break
    num, den = _poly(rng, m), _poly(rng, n)
    (a, _), (b, _) = num[0], den[0]
    ans = _rule(a, m, b, n)
    _verify(_frac_eval(num, den), ans)
    ratio = Fraction(a, b)
    if ans == 0:
        cands = [
            (_infinite(_sign(ratio)), "assumed-infinite"),
            (ratio, "ignored-degree-mismatch"),
            ("dne", "no-limit-at-infinity"),
            (Fraction(b, a), "inverted-ratio"),
        ]
        reason = f"The numerator has degree {m} and the denominator degree {n}, so the denominator grows faster and the limit is 0."
    else:
        cands = [
            (Fraction(0), "inverted-degree-comparison"),
            (ratio, "ignored-degree-mismatch"),
            (_flip(ans), "wrong-infinite-sign"),
            (Fraction(b, a), "inverted-ratio"),
        ]
        reason = (
            f"The numerator has degree {m} and the denominator degree {n}, so the numerator grows faster. "
            f"The leading terms behave like ({a}/{b})x^{m - n}, which tends to {_value_text(ans)}."
        )
    f = f"({_text(num)}) / ({_text(den)})"
    return dict(
        text=phrase(rng, _ASK, f=f, arrow="∞") + _NOTE,
        working=[reason],
        distractors=_dists(ans, cands),
        **_fields(ans),
    )


def _easy_power(rng):
    """The largest p with a limit of 0, or the smallest p with no finite limit."""
    q = rng.randint(2, 11)
    lead_d = rng.choice((-1, 1)) * rng.randint(1, 9)
    den = _poly(rng, q, lead_d, 1)
    a = rng.choice((-1, 1)) * rng.randint(1, 9)
    c = rng.choice((-1, 1)) * rng.randint(1, 9)
    num_text = f"{a}x^p + {c}" if c > 0 else f"{a}x^p - {-c}"
    if rng.random() < 0.5:
        want = q - 1
        stem = (
            f"p is a whole number and the limit of ({num_text}) / ({_text(den)}) as x → ∞ is 0. "
            "What is the largest possible value of p?"
        )
        # second route: p = q - 1 gives a degree gap, p = q gives a non-zero limit
        check(q - 1 < q and Fraction(a, lead_d) != 0, "p = q does not give a non-zero limit")
        cands = [
            (q, "w6-limit-equal-degree-allowed"),
            (q - 2, "w6-limit-power-off-by-one"),
            (q + 1, "w6-limit-power-direction-reversed"),
            (1, "w6-limit-first-power-only"),
        ]
        working = [
            f"The limit is 0 only when the numerator has a smaller degree than the denominator, which has degree {q}.",
            f"So p < {q}, and the largest whole number is p = {q - 1}.",
        ]
    else:
        want = q + 1
        stem = (
            f"p is a whole number and ({num_text}) / ({_text(den)}) has no finite limit as x → ∞ "
            "(it tends to ∞ or to -∞). What is the smallest possible value of p?"
        )
        cands = [
            (q, "w6-limit-equal-degree-allowed"),
            (q + 2, "w6-limit-power-off-by-one"),
            (q - 1, "w6-limit-power-direction-reversed"),
            (q * 2, "w6-limit-degrees-multiplied"),
        ]
        working = [
            f"The limit is infinite only when the numerator has a larger degree than the denominator, which has degree {q}.",
            f"So p > {q}, and the smallest whole number is p = {q + 1}.",
        ]
    return dict(
        text=stem,
        working=working,
        distractors=_dists(Fraction(want), [(Fraction(v), t) for v, t in cands]),
        **_fields(Fraction(want)),
    )


def _easy(rng):
    return rng.choice((_easy_limit, _easy_limit, _easy_power, _easy_power))(rng)


# -- medium ---------------------------------------------------------------------------------------------------------


def _medium_sum(rng):
    """A mismatched fraction plus an equal-degree fraction."""
    vanishes = rng.random() < 0.65
    n1 = rng.randint(1, 4)
    m1 = rng.randint(0, n1 - 1) if vanishes else rng.randint(n1 + 1, n1 + 2)
    a1, b1 = rng.choice((-1, 1)) * rng.randint(1, 9), rng.choice((-1, 1)) * rng.randint(1, 9)
    d2 = rng.randint(1, 3)
    a2, b2 = rng.choice((-1, 1)) * rng.randint(1, 9), rng.choice((-1, 1)) * rng.randint(1, 9)
    num1, den1 = _poly(rng, m1, a1), _poly(rng, n1, b1)
    num2, den2 = _poly(rng, d2, a2), _poly(rng, d2, b2)
    r2 = Fraction(a2, b2)
    ans = r2 if vanishes else _rule(a1, m1, b1, n1)
    f1, f2 = _frac_eval(num1, den1), _frac_eval(num2, den2)
    _verify(lambda x: f1(x) + f2(x), ans)
    t1, t2 = f"({_text(num1)}) / ({_text(den1)})", f"({_text(num2)}) / ({_text(den2)})"
    if rng.random() < 0.5:
        shown, mismatched_first = f"{t1} + {t2}", True
    else:
        shown, mismatched_first = f"{t2} + {t1}", False
    del mismatched_first
    ratio1 = Fraction(a1, b1)
    if vanishes:
        cands = [
            (r2 + ratio1, "ignored-degree-mismatch"),
            (Fraction(0), "w6-limit-only-mismatched-term"),
            (Fraction(b2, a2), "inverted-ratio"),
            (_infinite(_sign(r2 + ratio1)) if r2 + ratio1 != 0 else None, "assumed-infinite"),
        ]
        working = [
            f"The first fraction has numerator degree {m1} and denominator degree {n1}, so it tends to 0.",
            f"The second has equal degrees, so it tends to {a2}/{b2} = {rational_text(r2)}. The sum tends to {rational_text(r2)}.",
        ]
    else:
        cands = [
            (r2, "w6-limit-infinite-term-ignored"),
            (Fraction(0), "inverted-degree-comparison"),
            (_flip(ans), "wrong-infinite-sign"),
            (r2 + ratio1, "ignored-degree-mismatch"),
        ]
        working = [
            f"The first fraction has numerator degree {m1} and denominator degree {n1}, so it tends to {_value_text(ans)}.",
            f"The second tends to the finite value {rational_text(r2)}, which cannot stop the first from dominating. The sum tends to {_value_text(ans)}.",
        ]
    return dict(
        text=phrase(rng, _ASK, f=shown, arrow="∞") + _NOTE,
        working=working,
        distractors=_dists(ans, cands),
        **_fields(ans),
    )


def _medium_minus_infinity(rng):
    """One fraction as x → -∞, where the sign of an infinite limit depends on the parity of the difference of degrees."""
    for _ in attempts(200, "a pair of degrees"):
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        if abs(m - n) in (0, 1, 2, 3) and (m != n or rng.random() < 0.3):
            break
    num, den = _poly(rng, m), _poly(rng, n)
    (a, _), (b, _) = num[0], den[0]
    ans = _rule(a, m, b, n, -1)
    plus = _rule(a, m, b, n, 1)
    _verify(_frac_eval(num, den), ans, -1)
    ratio = Fraction(a, b)
    if m == n:
        cands = [
            (-ratio, "w6-limit-negative-infinity-flips-ratio"),
            (Fraction(b, a), "inverted-ratio"),
            (Fraction(0), "denominator-grows-so-zero"),
            (_infinite(_sign(ratio)), "assumed-infinite"),
        ]
        working = [
            f"The degrees are equal, so the limit is the ratio of leading coefficients {a}/{b}, whichever way x goes."
        ]
    elif m < n:
        cands = [
            (_infinite(_sign(ratio)), "assumed-infinite"),
            (ratio, "ignored-degree-mismatch"),
            ("dne", "no-limit-at-infinity"),
            (_infinite(-_sign(ratio)), "wrong-infinite-sign"),
        ]
        working = [f"The denominator has the larger degree ({n} against {m}), so the limit is 0 at either end."]
    else:
        cands = [
            (plus, "w6-limit-direction-sign-ignored"),
            (Fraction(0), "inverted-degree-comparison"),
            (_flip(ans), "wrong-infinite-sign"),
            (ratio, "ignored-degree-mismatch"),
        ]
        parity = "odd" if (m - n) % 2 else "even"
        working = [
            f"The leading terms behave like ({a}/{b})x^{m - n}, and the power {m - n} is {parity}.",
            f"For x → -∞ a {parity} power of x is {'negative' if parity == 'odd' else 'positive'}, so the limit is {_value_text(ans)}.",
        ]
    f = f"({_text(num)}) / ({_text(den)})"
    return dict(
        text=phrase(rng, _ASK, f=f, arrow="-∞") + _NOTE,
        working=working,
        distractors=_dists(ans, cands),
        **_fields(ans),
    )


def _medium(rng):
    return rng.choice((_medium_sum, _medium_minus_infinity))(rng)


# -- hard -----------------------------------------------------------------------------------------------------------


def _factored(rng):
    """``(x^2 + a)^e1 (px + q)^e2 / (c x^s (x + d)^j)``: the degrees and the leading coefficient come from the factors."""
    equal = rng.random() < 0.55
    e1, e2 = rng.randint(1, 2), rng.randint(1, 2)
    p = rng.randint(1, 3)
    q = rng.choice((-1, 1)) * rng.randint(1, 6)
    a = rng.randint(1, 7)
    c = rng.choice((-1, 1)) * rng.randint(1, 6)
    j = rng.randint(1, 3)
    d = rng.choice((-1, 1)) * rng.randint(1, 6)
    deg_n = 2 * e1 + e2
    s = deg_n - j if equal else rng.choice([v for v in range(0, 8) if v + j != deg_n])
    if s < 0:
        raise ValueError("redraw")
    deg_d = s + j
    lead_n = p**e2
    ans = _rule(lead_n, deg_n, c, deg_d)

    def func(x):
        return Fraction((x * x + a) ** e1 * (p * x + q) ** e2, c * x**s * (x + d) ** j)

    _verify(func, ans)
    qs = f"{p}x + {q}" if q > 0 else f"{p}x - {-q}"
    qs = qs.replace("1x", "x")
    ds = f"x + {d}" if d > 0 else f"x - {-d}"
    num = f"(x^2 + {a})" + (f"^{e1}" if e1 > 1 else "") + f"({qs})" + (f"^{e2}" if e2 > 1 else "")
    coef = "" if c == 1 else "-" if c == -1 else str(c)
    den = coef + _xp(s) + f"({ds})" + (f"^{j}" if j > 1 else "")
    factors_n, factors_d = e1 + e2, (1 if s else 0) + j
    cands = []
    if ans in ("inf", "-inf") or ans == 0:
        cands += [
            (Fraction(lead_n, c), "ignored-degree-mismatch"),
            (_flip(ans) if ans in ("inf", "-inf") else _infinite(_sign(Fraction(lead_n, c))), "wrong-infinite-sign"),
            (Fraction(0) if ans != 0 else _infinite(_sign(Fraction(lead_n, c))), "inverted-degree-comparison"),
        ]
    else:
        cands += [
            (Fraction(p * e2, c), "w6-limit-power-of-coefficient-not-taken"),
            (Fraction(c, lead_n), "inverted-ratio"),
            (Fraction(1, c), "w6-limit-factor-coefficients-dropped"),
        ]
    # counting factors instead of degrees
    wrong_by_factors = _rule(lead_n, factors_n, c, factors_d)
    cands.append((wrong_by_factors, "w6-limit-factors-counted-not-degrees"))
    cands.append((Fraction(lead_n, c), "ignored-degree-mismatch"))
    working = [
        f"Numerator degree: 2 x {e1} + {e2} = {deg_n}, with leading coefficient {p}^{e2} = {lead_n}.",
        f"Denominator degree: {s} + {j} = {deg_d}, with leading coefficient {c}.",
        (
            f"The degrees are equal, so the limit is {lead_n}/{c} = {_value_text(ans)}."
            if deg_n == deg_d
            else f"The degrees differ, so the limit is {_value_text(ans)}."
        ),
    ]
    return dict(
        text=phrase(rng, _ASK, f=f"{num} / ({den})", arrow="∞") + _NOTE,
        working=working,
        distractors=_dists(ans, cands),
        **_fields(ans),
    )


def _rooted(rng):
    """``(a x^m + ...) / √(s² x^(2n) + ...)`` as x → ±∞."""
    direction = rng.choice((1, -1))
    kind = rng.choice(("equal", "equal", "more", "less"))
    n = rng.randint(1, 3)
    m = n if kind == "equal" else (rng.randint(n + 1, 4) if kind == "more" else rng.randint(0, n - 1))
    if m == 0 and n == 0:
        raise ValueError("redraw")
    s = rng.randint(1, 4)
    a = rng.choice((-1, 1)) * rng.randint(1, 9)
    num = _poly(rng, m, a, 2)
    inner = [(s * s, 2 * n)]
    lower = sorted(rng.sample(range(2 * n), rng.randint(1, min(2, 2 * n))), reverse=True)
    for pw in lower:
        inner.append((rng.choice((-1, 1)) * rng.randint(1, 9), pw))
    inner[-1] = (abs(inner[-1][0]) if inner[-1][1] == 0 else inner[-1][0], inner[-1][1])
    # a constant term must not make the radicand negative anywhere near the limit: the leading term wins for large |x|
    ans = _rule_root(a, m, s, n, direction)

    def func(x):
        radicand = _evalp(inner, float(x))
        return float(_evalp(num, float(x))) / math.sqrt(radicand)

    _verify(func, ans, direction)
    ans_naive = _rule_root(a, m, s, n, 1)
    t_num, t_root = _text(num), _text(inner)
    f = f"({t_num}) / √({t_root})"
    cands = []
    if isinstance(ans, Fraction) and ans != 0:
        cands += [
            (ans_naive, "w6-limit-root-of-x-squared-taken-as-x"),
            (Fraction(a, s * s), "w6-limit-root-not-taken-of-coefficient"),
            (Fraction(s, a), "inverted-ratio"),
            (-ans, "w6-limit-negative-infinity-flips-ratio"),
        ]
    elif ans == 0:
        cands += [
            (Fraction(a, s), "ignored-degree-mismatch"),
            (_infinite(_sign(Fraction(a))), "assumed-infinite"),
            ("dne", "no-limit-at-infinity"),
        ]
    else:
        cands += [
            (ans_naive, "w6-limit-direction-sign-ignored"),
            (_flip(ans), "wrong-infinite-sign"),
            (Fraction(0), "inverted-degree-comparison"),
        ]
    # the root's degree not halved: compare m with 2n
    cands.append((_rule(a, m, s, 2 * n, direction), "w6-limit-root-degree-not-halved"))
    working = [
        f"For large |x| the root behaves like √({s * s}x^{2 * n}) = {s}|x|^{n}, because √(x^2) = |x|, not x.",
        f"The numerator behaves like {a}x^{m}. With x → {_arrow(direction)}, x^{m} = {'(-1)^' + str(m) + ' ' if direction < 0 else ''}|x|^{m}.",
        f"So the limit is {_value_text(ans)}.",
    ]
    return dict(
        text=phrase(rng, _ASK, f=f, arrow=_arrow(direction)) + _NOTE,
        working=working,
        distractors=_dists(ans, cands),
        **_fields(ans),
    )


def _rule_root(a, m, s, n, direction):
    """Limit of ``a x^m / (s |x|^n)`` as x → direction·∞."""
    ratio = Fraction(a, s)
    if m < n:
        return Fraction(0)
    signed = ratio * (direction**m)
    if m == n:
        return signed
    return _infinite(_sign(signed))


def _hard(rng):
    for _ in attempts(100, "a hard draw"):
        try:
            return retry_draw(rng, rng.choice((_factored, _rooted, _rooted)))
        except ValueError:
            continue
    raise AssertionError


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("w6_limit_degree", level=14, presets={t: {"tier": t} for t in _TIERS})
def w6_limit_degree(rng, tier: str = "easy"):
    """Degree-comparison limits; the tiers are described in the module docstring."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    if tier == "hard":
        return _hard(rng)
    return retry_draw(rng, _TIERS[tier])
