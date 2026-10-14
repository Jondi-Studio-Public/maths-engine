"""Unknowns in a discrete distribution: `r3a_discrete_unknown` and its exam chain.

Skill `discrete-unknown-from-expectation`. A finite distribution has one or two probabilities (or one outcome value)
left as letters, and the sum of the probabilities, a stated expectation, or both, fix them.

Tiers (topic `r3a_discrete_unknown`), every answer an exact fraction:

* easy   -- one unknown probability: the probabilities are multiples k, 2k, 3k, ... of one letter, or three are
  stated and the fourth is the unknown; the sum is 1;
* medium -- the same set-up, but the question is E(X), which needs the unknown first;
* hard   -- two unknown probabilities with E(X) given (two equations, solved together), one unknown outcome value
  with E(X) given, or Var(X) after both probabilities have been found.

The exam chain (topic `exd_x_r3a_discrete_unknown`) is (a) one unknown probability a from the sum and E(X), (b) the
other unknown from (a), (c) Var(X) from (a) and (b), (d) E(2X^2 - 3) from (c). Parts (b) to (d) carry
`follow_through`.

Every key is rebuilt by solving the same equations with SymPy's `linsolve` and compared with the construction.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, rational_text
from ._r3a_kit import build_until, fraction_part, single

_DENOMINATORS = (10, 12, 15, 20)


def _fr(value: Fraction) -> str:
    return rational_text(Fraction(value))


def _draw_outcomes(rng: random.Random, low=-2, high=6):
    return sorted(rng.sample(range(low, high + 1), 4))


def _draw_numerators(rng: random.Random, total: int):
    for _ in attempts(500, "four positive numerators"):
        cuts = sorted(rng.sample(range(1, total), 3))
        nums = [cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], total - cuts[2]]
        if len(set(nums)) >= 3:
            return nums
    raise AssertionError("unreachable")  # pragma: no cover


def _table(values, cells) -> str:
    """`cells` is a list of strings: a fraction, or a letter expression."""
    return ", ".join(f"P(X = {v}) = {c}" for v, c in zip(values, cells))


def _mean(values, probs) -> Fraction:
    return sum((Fraction(v) * p for v, p in zip(values, probs)), Fraction(0))


def _var(values, probs) -> Fraction:
    m = _mean(values, probs)
    return sum((Fraction(v) ** 2 * p for v, p in zip(values, probs)), Fraction(0)) - m * m


# -- easy and medium share the one-unknown set-ups ---------------------------------------------------------------


def _one_unknown(rng: random.Random):
    """(stem fragment, values, probs, name of the unknown, its position) for a one-unknown distribution."""
    values = _draw_outcomes(rng)
    if rng.random() < 0.5:
        for _ in attempts(500, "multiples of k that are not all equal"):
            coeffs = [rng.randint(1, 5) for _ in values]
            total = sum(coeffs)
            if len(set(coeffs)) >= 3 and total != 10:
                break
        probs = [Fraction(c, total) for c in coeffs]
        cells = [("k" if c == 1 else f"{c}k") for c in coeffs]
        unknown = Fraction(1, total)
        k = sympy.Symbol("k")
        solved = sympy.solve(sum(c * k for c in coeffs) - 1, k)
        check(Fraction(int(solved[0].p), int(solved[0].q)) == unknown, "SymPy disagrees about k")
        return _table(values, cells), values, probs, "k", unknown, ("multiples", coeffs)
    D = rng.choice(_DENOMINATORS)
    nums = _draw_numerators(rng, D)
    spot = rng.randrange(4)
    probs = [Fraction(n, D) for n in nums]
    cells = [("p" if i == spot else _fr(probs[i])) for i in range(4)]
    p = sympy.Symbol("p")
    known = sum(probs[i] for i in range(4) if i != spot)
    solved = sympy.solve(sympy.Rational(known.numerator, known.denominator) + p - 1, p)
    check(Fraction(int(solved[0].p), int(solved[0].q)) == probs[spot], "SymPy disagrees about p")
    return _table(values, cells), values, probs, "p", probs[spot], ("single", spot)


def _easy(rng: random.Random) -> dict:
    table, values, probs, name, unknown, kind = _one_unknown(rng)
    stem = f"A random variable X has the distribution {table}."
    n = 4
    if kind[0] == "multiples":
        coeffs = kind[1]
        slips = [
            (Fraction(1, n), "r3a-unknown-assumes-equal-probabilities"),
            (Fraction(1, sum(coeffs) - min(coeffs)), "r3a-sum-misses-a-term"),
            (Fraction(sum(coeffs)), "r3a-unknown-inverted"),
            (Fraction(1, max(coeffs)), "r3a-unknown-uses-the-largest-coefficient"),
        ]
        working = [f"{' + '.join(str(c) + 'k' for c in coeffs)} = 1, so k = 1/{sum(coeffs)}."]
    else:
        spot = kind[1]
        others = sum(probs[i] for i in range(4) if i != spot)
        slips = [
            (others, "r3a-reports-the-sum-of-the-known-probabilities"),
            (Fraction(1, 4), "r3a-unknown-assumes-equal-probabilities"),
            ((1 - others) / 2, "r3a-unknown-split-equally"),
            (1 - others + min(probs[i] for i in range(4) if i != spot), "r3a-sum-misses-a-term"),
        ]
        working = [f"The probabilities sum to 1, so {name} = 1 - {_fr(others)} = {_fr(unknown)}."]
    part = fraction_part(f"Find {name}.", unknown, 2, working, slips)
    return single(f"{stem} {part['prompt']}", part)


def _medium(rng: random.Random) -> dict:
    table, values, probs, name, unknown, kind = _one_unknown(rng)
    mean = _mean(values, probs)
    known_terms = Fraction(0)
    if kind[0] == "multiples":
        known_terms = Fraction(0)
        wrong_unknown = Fraction(1, 4)
        guess_probs = [Fraction(1, 4)] * 4
    else:
        spot = kind[1]
        known_terms = sum((Fraction(values[i]) * probs[i] for i in range(4) if i != spot), Fraction(0))
        guess_probs = [probs[i] if i != spot else Fraction(1, 4) for i in range(4)]
    slips = [
        (
            known_terms if kind[0] == "single" else sum(Fraction(v) for v in values) / 4,
            "r3a-expectation-omits-the-unknown-term",
        ),
        (sum(Fraction(v) for v in values) / 4, "r3a-expectation-averages-the-outcomes"),
        (sum(Fraction(v) for v in values), "r3a-expectation-adds-the-outcomes"),
        (_mean(values, guess_probs), "r3a-unknown-assumes-equal-probabilities"),
        (
            _mean(values, [Fraction(c, sum(kind[1])) for c in kind[1]][::-1])
            if kind[0] == "multiples"
            else _mean(values, probs[::-1]),
            "r3a-expectation-pairs-probabilities-with-the-wrong-outcomes",
        ),
    ]
    working = [
        f"The probabilities sum to 1, so {name} = {_fr(unknown)}.",
        f"E(X) = sum of x P(X = x) = {_fr(mean)}.",
    ]
    part = fraction_part("Find E(X).", mean, 3, working, slips, keep=5)
    stem = f"A random variable X has the distribution {table}."
    return single(f"{stem} {part['prompt']}", part)


# -- hard ------------------------------------------------------------------------------------------------------


def _two_unknowns(rng: random.Random):
    for _ in attempts(1000, "two unknown probabilities that solve to positive values"):
        values = _draw_outcomes(rng, 0, 6)
        D = rng.choice(_DENOMINATORS)
        nums = _draw_numerators(rng, D)
        i, j = rng.sample(range(4), 2)
        probs = [Fraction(n, D) for n in nums]
        if values[i] != values[j]:
            return values, probs, i, j
    raise AssertionError("unreachable")  # pragma: no cover


def _solve_two(values, probs, i, j, mean):
    a, b = sympy.symbols("a b")
    known_p = sum(probs[t] for t in range(4) if t not in (i, j))
    known_ex = sum(Fraction(values[t]) * probs[t] for t in range(4) if t not in (i, j))
    sol = sympy.linsolve(
        [
            a + b - sympy.Rational((1 - known_p).numerator, (1 - known_p).denominator),
            values[i] * a + values[j] * b - sympy.Rational((mean - known_ex).numerator, (mean - known_ex).denominator),
        ],
        [a, b],
    )
    ((ra, rb),) = list(sol)
    return Fraction(int(ra.p), int(ra.q)), Fraction(int(rb.p), int(rb.q))


def _hard(rng: random.Random) -> dict:
    variant = rng.choice(("probability", "value", "variance"))
    if variant == "value":
        D = rng.choice(_DENOMINATORS)
        nums = _draw_numerators(rng, D)
        probs = [Fraction(n, D) for n in nums]
        values = sorted(rng.sample(range(0, 8), 4))
        spot = rng.randrange(4)
        mean = _mean(values, probs)
        cells = [_fr(p) for p in probs]
        shown_values = [("x" if t == spot else str(values[t])) for t in range(4)]
        table = ", ".join(f"P(X = {shown_values[t]}) = {cells[t]}" for t in range(4))
        x = sympy.Symbol("x")
        known = sum(Fraction(values[t]) * probs[t] for t in range(4) if t != spot)
        solved = sympy.solve(
            sympy.Rational(known.numerator, known.denominator)
            + x * sympy.Rational(probs[spot].numerator, probs[spot].denominator)
            - sympy.Rational(mean.numerator, mean.denominator),
            x,
        )
        check(Fraction(int(solved[0].p), int(solved[0].q)) == values[spot], "SymPy disagrees about x")
        stem = f"A random variable X has the distribution {table}, and E(X) = {_fr(mean)}."
        part = fraction_part(
            "Find x.",
            Fraction(values[spot]),
            3,
            [f"E(X) = {_fr(known)} + {_fr(probs[spot])}x = {_fr(mean)}, so x = {values[spot]}."],
            [
                (mean / probs[spot], "r3a-unknown-value-ignores-the-known-terms"),
                (mean - known, "r3a-unknown-value-forgets-to-divide-by-probability"),
                ((mean - known) * probs[spot], "r3a-unknown-value-multiplies-by-probability"),
                (Fraction(values[(spot + 1) % 4]), "r3a-unknown-value-copied-from-neighbouring-outcome"),
            ],
            keep=4,
        )
        return single(f"{stem} {part['prompt']}", part)
    values, probs, i, j = _two_unknowns(rng)
    mean = _mean(values, probs)
    sol_a, sol_b = _solve_two(values, probs, i, j, mean)
    check((sol_a, sol_b) == (probs[i], probs[j]), "SymPy disagrees about the unknown probabilities")
    cells = [("a" if t == i else "b" if t == j else _fr(probs[t])) for t in range(4)]
    stem = f"A random variable X has the distribution {_table(values, cells)}, and E(X) = {_fr(mean)}."
    if variant == "probability":
        ask = rng.choice(("a", "b"))
        want, other = (probs[i], probs[j]) if ask == "a" else (probs[j], probs[i])
        xi, xj = (values[i], values[j]) if ask == "a" else (values[j], values[i])
        known_p = sum(probs[t] for t in range(4) if t not in (i, j))
        known_ex = sum(Fraction(values[t]) * probs[t] for t in range(4) if t not in (i, j))
        part = fraction_part(
            f"Find {ask}.",
            want,
            4,
            [
                f"a + b = {_fr(1 - known_p)} from the sum, and {values[i]}a + {values[j]}b = {_fr(mean - known_ex)} from E(X).",
                f"Solving the pair gives a = {_fr(probs[i])} and b = {_fr(probs[j])}.",
            ],
            [
                (other, "r3a-unknowns-swapped"),
                ((mean - known_ex) / xi if xi else None, "r3a-unknown-ignores-the-sum-equation"),
                ((1 - known_p) / 2, "r3a-unknown-split-equally"),
            ],
        )
        return single(f"{stem} {part['prompt']}", part)
    var = _var(values, probs)
    second = var + mean * mean
    part = fraction_part(
        "Find Var(X).",
        var,
        5,
        [
            f"a = {_fr(probs[i])} and b = {_fr(probs[j])}, so E(X^2) = {_fr(second)}.",
            f"Var(X) = E(X^2) - (E(X))^2 = {_fr(second)} - ({_fr(mean)})^2 = {_fr(var)}.",
        ],
        [
            (second, "r3a-variance-without-subtracting-the-mean-squared"),
            (second - mean, "r3a-variance-subtracts-the-mean-not-its-square"),
            (_var(values, [Fraction(1, 4)] * 4), "r3a-unknown-assumes-equal-probabilities"),
        ],
    )
    return single(f"{stem} {part['prompt']}", part)


@generator(
    "r3a_discrete_unknown",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3a_discrete_unknown(rng: random.Random, tier: str = "easy") -> dict:
    """Unknowns in a distribution: one probability (easy), then E(X) (medium), two unknowns or a value (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------

_SCENES = (
    ("The number of goals X that a school team scores in a match has", "The match bonus in dollars is Y = 2X^2 - 3."),
    (
        "The number of stalls X that open at a market on a wet morning has",
        "The council fee in dollars is Y = 2X^2 - 3.",
    ),
    ("The number of orders X placed in a minute by an online shop has", "The server load is Y = 2X^2 - 3."),
)


def _exam(rng: random.Random) -> dict:
    """Two unknown probabilities from the sum and the mean, then Var(X), then E(2X^2 - 3)."""
    intro, y_rule = rng.choice(_SCENES)
    values, probs, i, j = _two_unknowns(rng)
    mean = _mean(values, probs)
    a_val, b_val = _solve_two(values, probs, i, j, mean)
    check((a_val, b_val) == (probs[i], probs[j]), "SymPy disagrees about the unknown probabilities")
    var = _var(values, probs)
    second = var + mean * mean
    y_mean = 2 * second - 3
    check(
        y_mean == sum((2 * Fraction(v) ** 2 - 3) * p for v, p in zip(values, probs)), "E(Y) disagrees with Y's own sum"
    )
    cells = [("a" if t == i else "b" if t == j else _fr(probs[t])) for t in range(4)]
    stem = f"{intro} the distribution {_table(values, cells)}. Its mean is E(X) = {_fr(mean)}. {y_rule}"
    known_p = sum(probs[t] for t in range(4) if t not in (i, j))
    known_ex = sum(Fraction(values[t]) * probs[t] for t in range(4) if t not in (i, j))
    known_sq = sum(Fraction(values[t]) ** 2 * probs[t] for t in range(4) if t not in (i, j))
    paren = lambda f: f"({f.numerator}/{f.denominator})"  # noqa: E731
    parts = [
        fraction_part(
            "Find a.",
            probs[i],
            3,
            [
                f"a + b = {_fr(1 - known_p)} and {values[i]}a + {values[j]}b = {_fr(mean - known_ex)}.",
                f"Solving the pair gives a = {_fr(probs[i])}.",
            ],
            [
                (probs[j], "r3a-unknowns-swapped"),
                ((mean - known_ex) / values[i] if values[i] else None, "r3a-unknown-ignores-the-sum-equation"),
                ((1 - known_p) / 2, "r3a-unknown-split-equally"),
            ],
        ),
        fraction_part(
            "Find b.",
            probs[j],
            1,
            [f"b = {_fr(1 - known_p)} - a = {_fr(probs[j])}."],
            [
                (probs[i], "r3a-unknowns-swapped"),
                (1 - known_p, "r3a-unknown-ignores-the-other-unknown"),
                (1 - probs[i], "r3a-sum-misses-a-term"),
            ],
            follow_through=f"{paren(1 - known_p)} - part_a",
        ),
        fraction_part(
            "Find Var(X).",
            var,
            3,
            [
                f"E(X^2) = {_fr(second)}, so Var(X) = {_fr(second)} - ({_fr(mean)})^2 = {_fr(var)}.",
            ],
            [
                (second, "r3a-variance-without-subtracting-the-mean-squared"),
                (second - mean, "r3a-variance-subtracts-the-mean-not-its-square"),
                (second - mean * mean + (probs[i] - probs[j]), "r3a-unknowns-swapped"),
            ],
            follow_through=(
                f"{paren(known_sq)} + {values[i] ** 2}*part_a + {values[j] ** 2}*part_b - {paren(mean * mean)}"
            ),
        ),
        fraction_part(
            "Find E(Y).",
            y_mean,
            2,
            [f"E(Y) = 2E(X^2) - 3 = 2 x {_fr(second)} - 3 = {_fr(y_mean)}."],
            [
                (2 * var - 3, "r3a-second-moment-is-the-variance"),
                (2 * mean * mean - 3, "r3a-second-moment-squares-the-mean"),
                (2 * second, "r3a-linear-mean-omits-the-shift"),
            ],
            follow_through=f"2*(part_c + {paren(mean * mean)}) - 3",
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3a_discrete_unknown", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r3a_discrete_unknown(rng: random.Random, tier: str = "exam") -> dict:
    """Two unknown probabilities from the sum and the mean, then Var(X), then E(2X^2 - 3)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
