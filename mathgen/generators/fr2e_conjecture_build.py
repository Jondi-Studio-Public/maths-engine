"""Generates `f10-algebra-conjecture-algorithm` questions and its exam chain. `answer: fraction` (whole-number answers
are read as fractions); the exam chain also has a multiple-choice part.

Scope: Years 7, 8 and 10 reasoning about algebraic statements. A formula is varied one input at a time and the effect
on its output is found; a conjecture about every whole number is tested and the smallest number that defeats it is
found; a conjecture about a table of values from a quadratic or exponential rule is tested and the entry that breaks
it is named; and a test of a conjecture is carried to its conclusion. The neighbouring skill
`f10-linear-conjecture-testing` covers tables from linear rules; proofs and symbolic derivations are out of scope.

Ladder (reasoning escalation, not size):
  easy:   vary one or two inputs of a formula that uses powers or a quotient (the area of a circle, the energy
          E = ½mv², gravity as an inverse square) and give the factor the output changes by.
  medium: the smallest whole number that defeats a conjecture that an expression is always prime, or that one
          quantity always exceeds another; a number that merely gives a composite value, or a later counterexample, is
          not the answer.
  hard:   a table from a quadratic or exponential rule, a conjecture that it follows a stated rule or ratio, and
          the x-value of the one entry that breaks it. Every entry is checked against the rule.
  exam:   a three-part chain on one table: what the conjecture predicts at a tested value, how far the table is from
          it (follows through), and the correct next step in the test.

Every table is built from the rule and then re-checked entry by entry, and every counterexample is re-derived by
direct evaluation.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from sympy import Integer, Rational, Symbol

from ..core import attempts, check, generator, join_terms
from ._fr2e_kit import Redraw, draw, exact_number_part, fraction_part, mcq_part, rule_text, single

x = Symbol("x")

E_POWER_IGNORED = "fr2e-conj-power-ignored"
E_CHANGES_ADDED = "fr2e-conj-changes-added"
E_INVERSE_IGNORED = "fr2e-conj-inverse-treated-as-direct"
E_ONE_VARIABLE = "fr2e-conj-only-one-input-changed"
E_POWER_ON_WRONG = "fr2e-conj-power-on-the-wrong-input"
E_LATER = "fr2e-conj-later-counterexample"
E_VALUE_AS_INPUT = "fr2e-conj-value-reported-as-input"
E_NEXT = "fr2e-conj-neighbouring-input-reported"
E_LAST_WORKS = "fr2e-conj-last-input-that-works-reported"
E_ROW = "fr2e-conj-entry-that-fits-reported"
E_NONE = "fr2e-conj-accepted-after-partial-check"
E_NEIGHBOUR = "fr2e-conj-neighbouring-entry-blamed"

# -- easy ------------------------------------------------------------------------------------------------------------

# (formula sentence, output name, {symbol: (name, exponent)})
_FORMULAS = (
    ("The area of a circle is A = πr².", "the area A", {"r": ("radius r", 2)}),
    ("The volume of a cylinder is V = πr²h.", "the volume V", {"r": ("radius r", 2), "h": ("height h", 1)}),
    ("The kinetic energy of a moving object is E = ½mv².", "the energy E", {"m": ("mass m", 1), "v": ("speed v", 2)}),
    ("The volume of a cube is V = s³.", "the volume V", {"s": ("side length s", 3)}),
    ("The power used by an appliance is P = I²R.", "the power P", {"I": ("current I", 2), "R": ("resistance R", 1)}),
    (
        "The time to fill a tank is T = V/R, where V is the volume and R is the rate of the pump.",
        "the time T",
        {"V": ("volume V", 1), "R": ("pump rate R", -1)},
    ),
    ("The speed of a runner is v = d/t.", "the speed v", {"d": ("distance d", 1), "t": ("time t", -1)}),
    (
        "The gravitational force between two masses is F = GMm/d².",
        "the force F",
        {"M": ("mass M", 1), "m": ("mass m", 1), "d": ("distance d", -2)},
    ),
    (
        "The light from a lamp reaching a wall is I = P/d².",
        "the brightness I",
        {"P": ("lamp power P", 1), "d": ("distance d", -2)},
    ),
)
_CHANGES = (
    (Fraction(2), "doubled"),
    (Fraction(3), "tripled"),
    (Fraction(4), "multiplied by 4"),
    (Fraction(1, 2), "halved"),
    (Fraction(1, 3), "divided by 3"),
)


def _factor_text(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _easy(rng):
    sentence, output, variables = rng.choice(_FORMULAS)
    symbols = list(variables)
    count = 1 if len(symbols) == 1 else rng.choice((1, 2, 2))
    chosen = rng.sample(symbols, count)
    picks = {s: rng.choice(_CHANGES) for s in chosen}
    factor = Fraction(1)
    for s, (c, _) in picks.items():
        factor *= c ** variables[s][1]
    if factor == 1:
        raise Redraw("the output must change")
    # second route: evaluate the formula at base values with the inputs changed
    base = {s: Fraction(rng.randint(2, 9)) for s in symbols}

    def out(vals):
        v = Fraction(1)
        for s in symbols:
            v *= vals[s] ** variables[s][1]
        return v

    changed = {s: base[s] * (picks[s][0] if s in picks else 1) for s in symbols}
    check(out(changed) / out(base) == factor, "evaluating the formula must give the same factor")
    wording = " and ".join(f"the {variables[s][0]} is {picks[s][1]}" for s in chosen)
    wrong = []
    ignored = Fraction(1)
    for s, (c, _) in picks.items():
        ignored *= c
    wrong.append((ignored, E_POWER_IGNORED))
    wrong.append((sum(c for c, _ in picks.values()) if count > 1 else None, E_CHANGES_ADDED))
    direct = Fraction(1)
    for s, (c, _) in picks.items():
        direct *= c ** abs(variables[s][1])
    wrong.append((direct, E_INVERSE_IGNORED))
    first = chosen[0]
    wrong.append(
        (picks[first][0] ** variables[first][1], E_ONE_VARIABLE)
        if count > 1
        else (picks[first][0] ** (variables[first][1] + 1), E_POWER_ON_WRONG)
    )
    if count > 1:
        second = chosen[1]
        wrong.append(
            (picks[first][0] ** variables[second][1] * picks[second][0] ** variables[first][1], E_POWER_ON_WRONG)
        )
    wrong.append((picks[first][0] ** (variables[first][1] + 1), "power-law-exponent-off-by-one"))
    wrong.append(
        (
            picks[first][0] * variables[first][1] if variables[first][1] > 0 else None,
            "fr2e-conj-exponent-treated-as-multiplier",
        )
    )
    wrong = [(v, t) for v, t in wrong if v is not None]
    stem = f"{sentence} {wording[0].upper()}{wording[1:]}, and nothing else changes."
    part = fraction_part(
        f"By what factor does {output} change? Give a whole number or a fraction.",
        factor,
        1,
        [
            f"Each input changes the output by its factor raised to its power: {', '.join(f'{_factor_text(picks[s][0])}^{variables[s][1]}' for s in chosen)}.",
            f"The output changes by a factor of {_factor_text(factor)}.",
        ],
        wrong,
    )
    if len(part["distractors"]) < 3:
        raise Redraw("fewer than three distinct wrong factors")
    return single(stem, part)


# -- medium ----------------------------------------------------------------------------------------------------------


def is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


def _poly_text(a: int, b: int, c: int) -> str:
    return join_terms([(a, "n^2"), (b, "n"), (c, "")])


def _prime_conjecture(rng):
    a = rng.choice((1, 1, 1, 2))
    b = rng.choice((-3, -2, -1, 1, 2, 3)) if a == 1 else 0
    c = rng.choice((3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41))
    f = lambda n: a * n * n + b * n + c  # noqa: E731
    return (
        f,
        f"the expression {_poly_text(a, b, c)} gives a prime number for every whole number n from 1 up",
        lambda n: is_prime(f(n)),
        f,
    )


def _mersenne_conjecture(rng):
    f = lambda n: 2**n - 1  # noqa: E731
    return (
        f,
        "2^n - 1 is a prime number for every prime number n",
        lambda n: is_prime(n) and is_prime(f(n)) or not is_prime(n),
        f,
    )


def _smallest_defeating(holds, lo=1, hi=60):
    return [n for n in range(lo, hi + 1) if not holds(n)]


def _medium(rng):
    if rng.random() < 0.95:
        for _ in attempts(400, "a prime conjecture with a counterexample that takes some finding"):
            f, statement, holds, value = _prime_conjecture(rng)
            failing = _smallest_defeating(holds)
            if len(failing) >= 2 and 4 <= failing[0] <= 45:
                break
        tested = "n = 1, 2, 3 and so on in order"
        what = "whole number n"
        smallest = failing[0]
        wrong = [
            (failing[1], E_LATER),
            (smallest + 1, E_NEXT),
            (smallest - 1, E_LAST_WORKS),
            (value(smallest), E_VALUE_AS_INPUT),
        ]
        steps = [f"n = {n}: {'true' if holds(n) else 'false'}" for n in range(1, smallest + 1)]
    else:
        f, statement, holds, value = _mersenne_conjecture(rng)
        tested = "the prime numbers n = 2, 3, 5, 7 and so on in order"
        what = "prime number n"
        failing = [n for n in _smallest_defeating(holds) if is_prime(n)]
        smallest = failing[0]
        wrong = [(failing[1], E_LATER), (13, E_NEXT), (7, E_LAST_WORKS), (value(smallest), E_VALUE_AS_INPUT)]
        steps = [f"n = {n}: {'true' if holds(n) else 'false'}" for n in range(2, smallest + 1) if is_prime(n)]
    check(not holds(smallest) and all(holds(n) for n in range(1, smallest)), "smallest must be the first failure")
    wrong = [(w, t) for w, t in wrong if w >= 1]
    name = rng.choice(("Priya", "Marcus", "Aiko", "Tom", "Leila", "Sam", "Jess", "Kofi"))
    stem = f"{name} conjectures that {statement}. {name} tests {tested}."
    part = exact_number_part(
        f"Find the smallest {what} that shows the conjecture is false.",
        smallest,
        1,
        ["Testing in order: " + "; ".join(steps) + ".", f"The first value that defeats it is n = {smallest}."],
        wrong,
        suffix="",
    )
    return single(stem, part)


# -- hard and the exam: tables from quadratic and exponential rules ---------------------------------------------------


def _quad_rule(rng):
    a = rng.choice((1, 1, 2, 3))
    b = rng.randint(-2, 5)
    c = rng.randint(1, 9)
    return a * x**2 + b * x + c, "quad"


def _exp_rule(rng):
    r = rng.choice((2, 3, 4))
    A = rng.choice((1, 2, 3, 5)) if r <= 3 else rng.choice((1, 2))
    return A * Integer(r) ** x, "exp"


def _rule_text(rule) -> str:
    return rule_text(rule)


def _value(rule, v: int) -> int:
    r = sympy.Rational(rule.subs(x, Integer(v)))
    check(r.q == 1, "table values must be whole numbers")
    return int(r.p)


def _table(rule, xs, tampered: dict[int, int] | None = None):
    ys = [_value(rule, v) for v in xs]
    if tampered:
        for i, d in tampered.items():
            ys[i] += d
    return ys


def _text_table(xs, ys) -> str:
    return "x: " + ", ".join(str(v) for v in xs) + "; y: " + ", ".join(str(v) for v in ys)


def _hard(rng):
    rule, kind = rng.choice((_quad_rule, _exp_rule))(rng)
    xs = rng.choice(
        ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        if kind == "exp"
        else ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [0, 1, 2, 4, 5, 7])
    )
    bad = rng.randrange(1 if kind == "exp" else 0, len(xs))
    ys = _table(rule, xs, {bad: rng.choice((-3, -2, -1, 1, 2, 3, 4))})
    expected = [_value(rule, v) for v in xs]
    broken = [i for i in range(len(xs)) if ys[i] != expected[i]]
    check(broken == [bad], "the table must break the rule at exactly the chosen entry")
    name = rng.choice(("Priya", "Marcus", "Aiko", "Tom", "Leila", "Sam", "Jess", "Kofi"))
    if kind == "exp" and xs[0] == 0:
        A = _value(rule, 0)
        r = _value(rule, 1) // A
        claim = (
            f"{name} conjectures that this table starts at y = {A} when x = 0 and that each y-value is {r} times "
            f"the one for the x-value before, so y = {_rule_text(rule)}."
        )
    else:
        claim = f"{name} conjectures that the rule y = {_rule_text(rule)} gives every y-value in this table."
    stem = f"{claim} {_text_table(xs, ys)}. Exactly one entry in the table breaks the conjecture."
    neighbours = [xs[i] for i in (bad - 1, bad + 1) if 0 <= i < len(xs)]
    part = exact_number_part(
        "Find the x-value of the entry that breaks the conjecture.",
        xs[bad],
        1,
        [
            f"The rule gives {', '.join(str(v) for v in expected)} at these x-values.",
            f"The table shows {', '.join(str(v) for v in ys)}, so the entry for x = {xs[bad]} is the only one that differs.",
        ],
        [(n, E_NEIGHBOUR) for n in neighbours]
        + [
            (ys[bad], "fr2e-conj-table-value-reported"),
            (expected[bad], "fr2e-conj-prediction-reported"),
            (xs[0], E_ROW),
        ],
        suffix="",
    )
    return single(stem, part)


# -- exam ------------------------------------------------------------------------------------------------------------


def _exam(rng):
    rule, kind = rng.choice((_quad_rule, _exp_rule))(rng)
    xs = [1, 2, 3, 4, 5, 6] if kind == "quad" else [0, 1, 2, 3, 4, 5]
    N = rng.choice(xs[2:5])
    differs = rng.random() < 0.65
    delta = rng.choice((-3, -2, -1, 1, 2, 3, 5)) if differs else 0
    idx = xs.index(N)
    ys = _table(rule, xs, {idx: delta} if delta else None)
    predicted = _value(rule, N)
    observed = ys[idx]
    name = rng.choice(("Priya", "Marcus", "Aiko", "Tom", "Leila", "Sam", "Jess", "Kofi"))
    stem = (
        f"{name} conjectures that the rule y = {_rule_text(rule)} gives every y-value in this table: {_text_table(xs, ys)}. "
        f"{name} tests the rule at x = {N}."
    )
    if kind == "quad":
        a_, b_, c_ = (int(c) for c in sympy.Poly(rule, x).all_coeffs())
        slip = (a_ * 2 * N + b_ * N + c_, "squared-as-doubled")
    else:
        A_ = _value(rule, 0)
        r_ = _value(rule, 1) // A_
        slip = (A_ * r_ * N, "fr2e-conj-exponent-treated-as-multiplier")
    part_a = exact_number_part(
        f"Find the value of y that the conjecture gives at x = {N}.",
        predicted,
        1,
        [f"y = {_rule_text(rule)} at x = {N} gives {predicted}."],
        [
            (_value(rule, N - 1), "fr2e-model-x-off-by-one"),
            (_value(rule, N + 1), "fr2e-model-x-off-by-one"),
            slip,
            (observed if observed != predicted else predicted + 2, "fr2e-conj-table-value-reported"),
        ],
        suffix="",
        sensible_range=(0, None),
    )
    part_b = exact_number_part(
        f"Find the table's value of y at x = {N} minus the value the conjecture gives. Use a minus sign if the table's value is smaller.",
        observed - predicted,
        1,
        [f"The table shows {observed} and the rule gives {predicted}, so the difference is {observed - predicted}."],
        [
            (predicted - observed, "fr2e-conj-difference-taken-the-wrong-way"),
            (observed, "fr2e-conj-table-value-reported"),
            (predicted, "fr2e-conj-prediction-reported"),
            (observed + predicted, "fr2e-conj-values-added"),
            (
                1 if observed == predicted else observed - ys[idx - 1],
                "fr2e-conj-agreement-counted-as-one"
                if observed == predicted
                else "fr2e-conj-difference-between-neighbouring-entries",
            ),
        ],
        suffix="",
        follow_through=f"{observed} - part_a",
        signed=True,
    )
    k = len(xs) - 1
    if delta:
        key = f"x = {N} is a counterexample, so the conjecture is false. One counterexample is enough."
        wrong = [
            (
                f"The conjecture is probably still true, because the rule fits the other {k} values in the table.",
                E_NONE,
            ),
            (
                f"x = {N} is a counterexample, but the conjecture is false only if a second counterexample is found.",
                "fr2e-conj-more-tests-needed-after-counterexample",
            ),
            (
                f"The conjecture is true for the x-values before x = {N} and false from x = {N} onwards.",
                "fr2e-conj-counterexample-treated-as-boundary",
            ),
            (
                f"The table must contain an error, because a rule that fits {k} values cannot be false.",
                "fr2e-conj-counterexample-needs-majority",
            ),
        ]
    else:
        key = f"The rule fits at x = {N}, but that does not prove it. Test more x-values, including ones outside the table."
        wrong = [
            (f"The rule fits at x = {N}, so it is proved to be true for every x.", E_NONE),
            (
                f"The rule fits at x = {N}, so the other x-values in the table do not need to be checked.",
                "fr2e-conj-table-size-treated-as-proof",
            ),
            (
                f"The rule fits at x = {N}, so x = {N} is a counterexample to the claim that the rule fails.",
                "fr2e-conj-agreement-treated-as-counterexample",
            ),
            (
                f"The rule fits at x = {N}, which shows that it must fail at the x-values outside the table.",
                "fr2e-conj-one-fit-treated-as-boundary",
            ),
        ]
    part_c = mcq_part(
        "Which is the correct conclusion and next step?",
        key,
        wrong,
        2,
        [f"At x = {N} the rule gives {predicted} and the table shows {observed}."],
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator(
    "fr2e_conjecture",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2e_conjecture(rng, tier: str = "easy"):
    """Vary a formula's inputs, find the smallest counterexample, or find the table entry that breaks a conjecture."""
    build = {"easy": _easy, "medium": _medium, "hard": _hard}.get(tier)
    check(build is not None, f"unknown tier {tier!r}")
    return draw(build, rng)


@generator("exd_x_fr2e_conjecture", level=12, presets={"exam": dict(tier="exam")})
def exd_x_fr2e_conjecture(rng, tier: str = "exam"):
    """A three-part chain: the conjecture's prediction, the table's difference from it, and the correct next step."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return draw(_exam, rng)
