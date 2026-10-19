"""Generates `leslie-population-matrix` questions (easy, medium, hard) and the exam chain `r3c_leslie_chain`.

An animal population is split into three age classes. Each year a share of each class survives into the next age
class, the oldest class dies out, and each class produces young. The Leslie matrix ``L`` has the birth rates in its
first row and the survival shares just below the diagonal, and ``P(next) = L P``. The prompt gives the rates in words,
never as a matrix.

* easy: the whole population next year;
* medium: the whole population after two years, as a number;
* hard: the age classes are in a fixed ratio from year to year (a stable age pattern); by what factor does the whole
  population grow each year? The ratio is built from a Leslie matrix whose growth factor is a short decimal;
* exam (``r3c_leslie_chain``): the number of young next year, the whole population next year, the percentage change in
  the whole population (carried forward) and the age classes after two years.

Every count in a question is a whole number.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._r3c_kit import col, mat_mul, mat_pow, matrix_part, matrix_single, number_part, number_single, transpose, vec

_SPECIES = (
    dict(
        who="quokkas on an island reserve",
        kind="quokkas",
        classes=("juveniles (under 1 year)", "young adults (1 to 2 years)", "older adults (2 to 3 years)"),
        short=("juveniles", "young adults", "older adults"),
        young="young",
        one="young",
        high=False,
    ),
    dict(
        who="wombats in a wildlife park",
        kind="wombats",
        classes=("pups (under 1 year)", "juveniles (1 to 2 years)", "adults (2 to 3 years)"),
        short=("pups", "juveniles", "adults"),
        young="pups",
        one="pup",
        high=False,
    ),
    dict(
        who="rainbow trout in a hatchery pond",
        kind="trout",
        classes=("fry (under 1 year)", "fingerlings (1 to 2 years)", "adults (2 to 3 years)"),
        short=("fry", "fingerlings", "adults"),
        young="fry",
        one="fry",
        high=True,
    ),
    dict(
        who="frogs around a wetland",
        kind="frogs",
        classes=("tadpoles and froglets (under 1 year)", "young frogs (1 to 2 years)", "adult frogs (2 to 3 years)"),
        short=("tadpoles and froglets", "young frogs", "adult frogs"),
        young="young",
        one="young",
        high=True,
    ),
)


def _rate(x: Fraction) -> str:
    return f"{float(x):g}"


def _young(sp, x: Fraction) -> str:
    """ "1 pup", "0.8 pups"."""
    return f"{_rate(x)} {sp['one'] if x == 1 else sp['young']}"


def _leslie(f, s):
    return [
        [Fraction(f[0]), Fraction(f[1]), Fraction(f[2])],
        [Fraction(s[0]), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(s[1]), Fraction(0)],
    ]


def _describe(sp, f, s):
    c = sp["short"]
    births = [f"each of the {c[k]} produces {_young(sp, f[k])}" for k in range(3) if f[k] > 0]
    first = "Each year, " + (births[0] if len(births) == 1 else ", ".join(births[:-1]) + " and " + births[-1]) + ". "
    return (
        first + f"Of the {c[0]}, {int(s[0] * 100)}% survive to become {c[1]}, and of the {c[1]}, "
        f"{int(s[1] * 100)}% survive to become {c[2]}. None of the {c[2]} survive another year."
    )


def _pop_text(sp, P):
    c = sp["short"]
    return f"{int(P[0])} {c[0]}, {int(P[1])} {c[1]} and {int(P[2])} {c[2]}"


def _draw_rates(rng, sp):
    lo2, hi2, lo3, hi3 = (8, 30, 5, 25) if sp["high"] else (3, 10, 3, 10)
    f = [Fraction(0), Fraction(rng.randint(lo2, hi2), 10), Fraction(rng.randint(lo3, hi3), 10)]
    s = [Fraction(rng.randint(3, 9), 10), Fraction(rng.randint(2, 8), 10)]
    return f, s


def _slips(rng, L, f, s, P, years):
    """Wrong population vectors after ``years`` years."""
    births = lambda Q: sum(f[k] * Q[k] for k in range(3))
    right = vec(mat_mul(mat_pow(L, years), col(P)))
    out = [(vec(mat_mul(mat_pow(transpose(L), years), col(P))), "r3c-leslie-matrix-transposed")]
    one = vec(mat_mul(L, col(P)))
    out.append(([Fraction(0), s[0] * P[0], s[1] * P[1]], "r3c-leslie-no-births"))
    out.append(([births(P), P[1], P[2]], "r3c-leslie-no-ageing"))
    out.append(([births(P), s[0] * P[0], s[1] * P[1] + P[2]], "r3c-leslie-oldest-survive"))
    out.append(([f[1] * P[1], s[0] * P[0], s[1] * P[1]], "r3c-leslie-births-one-class"))
    if years > 1:
        out.append((one, "r3c-leslie-one-year-only"))
    return [(col(v), tag) for v, tag in out if v != right]


def _easy(rng):
    for _ in attempts(500, "an easy Leslie question"):
        sp = rng.choice(_SPECIES)
        f, s = _draw_rates(rng, sp)
        L = _leslie(f, s)
        P = [Fraction(10 * rng.randint(10, 80)) for _ in range(3)]
        N = vec(mat_mul(L, col(P)))
        if any(v.denominator != 1 for v in N) or sum(N) == sum(P):
            continue
        text = (
            f"A population of {sp['who']} is counted in three age classes. {_describe(sp, f, s)} "
            f"This year there are {_pop_text(sp, P)}. What is the total population next year?"
        )
        working = [
            f"New {sp['young']} come from the births of each class this year.",
            "Each survivor moves up one age class, and the oldest class does not survive.",
            f"Next year: {[int(v) for v in N]}, a total of {int(sum(N))}.",
        ]
        wrong = [(sum(vec(v)), tag) for v, tag in _slips(rng, L, f, s, P, 1)]
        q = number_single(text, sum(N), working, wrong, lo=1)
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _medium(rng):
    for _ in attempts(500, "a medium Leslie question"):
        sp = rng.choice(_SPECIES)
        f, s = _draw_rates(rng, sp)
        L = _leslie(f, s)
        P = [Fraction(100 * rng.randint(2, 12)) for _ in range(3)]
        P1, P2 = vec(mat_mul(L, col(P))), vec(mat_mul(mat_pow(L, 2), col(P)))
        if any(v.denominator != 1 for v in P1 + P2):
            continue
        n0, n1, n2 = sum(P), sum(P1), sum(P2)
        if len({n0, n1, n2}) < 3:
            continue
        text = (
            f"A population of {sp['who']} is counted in three age classes. {_describe(sp, f, s)} "
            f"This year there are {_pop_text(sp, P)}. "
            f"What is the total population in two years' time?"
        )
        wrong = [
            (n1, "r3c-leslie-one-year-only"),
            (n1 * n1 / n0, "r3c-leslie-constant-growth"),
            (sum(vec(mat_mul(mat_pow(transpose(L), 2), col(P)))), "r3c-leslie-matrix-transposed"),
            (n2 - sum(f[k] * P1[k] for k in range(3)), "r3c-leslie-no-births"),
            (n2 + P[2], "r3c-leslie-oldest-survive"),
            (n0 + 2 * (n1 - n0), "r3c-leslie-double-the-change"),
        ]
        wrong = [(w, t) for w, t in wrong if Fraction(w).denominator == 1]
        working = [
            f"Next year: {[int(v) for v in P1]}, a total of {int(n1)}.",
            f"In two years: {[int(v) for v in P2]}, a total of {int(n2)}.",
        ]
        q = number_single(text, n2, working, wrong, lo=1)
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _stable_for(rng, lam):
    """Rates and a whole-number stable age pattern for growth factor ``lam``, or None after a bounded search."""
    for _ in range(150):
        s1 = Fraction(rng.randint(3, 9), 10)
        s2 = Fraction(rng.randint(2, 8), 10)
        f2 = Fraction(rng.randint(8, 30), 10)
        f3 = (lam**3 - f2 * s1 * lam) / (s1 * s2)
        if not (Fraction(1, 2) <= f3 <= Fraction(5, 2)) or (f3 * 20).denominator != 1:
            continue
        v = [Fraction(1), s1 / lam, s1 * s2 / lam**2]
        scale = 1
        for x in v:
            scale = scale * x.denominator // _gcd(scale, x.denominator)
        v = [x * scale for x in v]
        if any(x.denominator != 1 for x in v) or max(v) > 400 or min(v) < 2:
            continue
        return [Fraction(0), f2, f3], [s1, s2], v, lam
    return None


_GROWTH = (
    Fraction(9, 10),
    Fraction(11, 10),
    Fraction(6, 5),
    Fraction(13, 10),
    Fraction(7, 5),
    Fraction(5, 4),
    Fraction(3, 2),
    Fraction(8, 5),
    Fraction(9, 5),
    Fraction(2),
)


def _stable(rng):
    """Rates (f, s) and a whole-number stable age pattern v with L v = lam v, lam a short decimal."""
    for _ in attempts(200, "a Leslie matrix with a decimal growth factor"):
        found = _stable_for(rng, rng.choice(_GROWTH))
        if found:
            return found
    raise AssertionError  # pragma: no cover


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _hard(rng):
    for _ in attempts(500, "a hard Leslie question"):
        sp = rng.choice([x for x in _SPECIES if x["high"]])
        f, s, v, lam = _stable(rng)
        L = _leslie(f, s)
        Lv = vec(mat_mul(L, col(v)))
        check(all(Lv[k] == lam * v[k] for k in range(3)), "the stable pattern is not an eigenvector")
        births = Lv[0]
        key = lam
        ratio = f"{int(v[0])} : {int(v[1])} : {int(v[2])}"
        text = phrase(
            rng,
            [
                "A population of {who} is counted in three age classes. {describe} This year the {a}, {b} and {c} "
                "are in the ratio {ratio}, and the population keeps exactly this ratio from year to year. By what "
                "factor does the whole population multiply each year? Give your answer to 2 decimal places.",
                "A population of {who} is counted in three age classes. {describe} The {a}, {b} and {c} are "
                "currently in the ratio {ratio} and stay in this ratio every year. Find the yearly growth factor of "
                "the whole population, correct to 2 decimal places.",
                "A population of {who} is counted in three age classes. {describe} Counts of the {a}, {b} and {c} "
                "are in the ratio {ratio}, and this ratio never changes from one year to the next. Find the number "
                "by which the total population is multiplied each year, to 2 decimal places.",
            ],
            who=sp["who"],
            describe=_describe(sp, f, s),
            a=sp["short"][0],
            b=sp["short"][1],
            c=sp["short"][2],
            ratio=ratio,
        )
        wrong = [
            (f[0] + f[1] + f[2], "r3c-leslie-fertility-sum"),
            (1 / lam, "r3c-leslie-ratio-inverted"),
            (lam - 1, "r3c-leslie-rate-not-factor"),
            (births / v[2], "r3c-leslie-wrong-class-ratio"),
            (births / sum(v), "r3c-leslie-birth-share"),
            ((s[0] + s[1]) / 2, "r3c-leslie-survival-average"),
        ]
        working = [
            f"Next year's youngest class is {_rate(f[0])} x {int(v[0])} + {_rate(f[1])} x {int(v[1])} + {_rate(f[2])} x {int(v[2])} = {_rate(births)}.",
            f"The ratio is kept, so the factor is {_rate(births)} / {int(v[0])} = {_rate(lam)}.",
        ]
        q = number_single(text, key, working, wrong, dp=2, lo=Fraction(1, 100))
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "r3c_leslie",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3c_leslie(rng, tier="easy"):
    """An age-class population model; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


@generator("r3c_leslie_chain", level=12, presets={"exam": dict(tier="exam")})
def r3c_leslie_chain(rng, tier="exam"):
    """Exam chain: young next year, total next year, the percentage change, and the age classes after two years."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(800, "a Leslie chain"):
        sp = rng.choice(_SPECIES)
        f, s = _draw_rates(rng, sp)
        L = _leslie(f, s)
        P = [Fraction(100 * rng.randint(2, 12)) for _ in range(3)]
        P1, P2 = vec(mat_mul(L, col(P))), vec(mat_mul(mat_pow(L, 2), col(P)))
        n0, n1 = sum(P), sum(P1)
        if any(v.denominator != 1 for v in P1 + P2) or n1 == n0 or len(set(P2)) < 3:
            continue
        stem = (
            f"A population of {sp['who']} is counted in three age classes. {_describe(sp, f, s)} "
            f"This year there are {_pop_text(sp, P)}."
        )
        births = P1[0]
        part_a = number_part(
            f"How many {sp['young']} are born during the year and counted next year in the youngest class?",
            births,
            2,
            [
                f"{_rate(f[0])} x {int(P[0])} + {_rate(f[1])} x {int(P[1])} + {_rate(f[2])} x {int(P[2])} = {int(births)}."
            ],
            [
                (f[1] * P[1], "r3c-leslie-births-one-class"),
                (f[1] * P[1] + f[2] * P[2], "r3c-leslie-births-omits-class"),
                (f[0] * P[0] + f[1] * P[1] + f[2] * P[2] + s[0] * P[0], "r3c-leslie-births-plus-survivors"),
                (P[0] * (f[0] + f[1] + f[2]), "r3c-leslie-fertility-sum"),
            ],
            lo=1,
            sensible=(0, None),
        )
        part_b = number_part(
            "What is the total population next year?",
            n1,
            2,
            [f"{int(births)} + {int(s[0] * P[0])} + {int(s[1] * P[1])} = {int(n1)}."],
            [
                (n0 + births, "r3c-leslie-births-added-to-all"),
                (births + P[1] + P[2], "r3c-leslie-no-ageing"),
                (n1 - births, "r3c-leslie-no-births"),
                (n1 + P[2], "r3c-leslie-oldest-survive"),
            ],
            lo=1,
            sensible=(0, None),
        )
        change = (n1 - n0) / n0 * 100
        part_c = number_part(
            f"By what percentage does the total population {'increase' if change > 0 else 'decrease'} over the year? Give your answer to 1 decimal place.",
            abs(change),
            2,
            [f"({int(n1)} - {int(n0)}) / {int(n0)} x 100 = {float(change):.1f}%."],
            [
                (abs(n1 - n0) / n1 * 100, "r3c-leslie-percent-of-new"),
                (n1 / n0 * 100, "r3c-leslie-percent-not-change"),
                (abs(n1 - n0), "r3c-leslie-change-not-percent"),
                (abs(change) / 100, "r3c-leslie-percent-as-decimal"),
            ],
            follow=f"Abs(part_b - {int(n0)})/{int(n0)}*100",
            dp=1,
            lo=Fraction(1, 10),
            sensible=(0, None),
        )
        part_d = matrix_part(
            "How many are in each age class in two years' time? Write the numbers as a column, youngest class first.",
            col(P2),
            3,
            [f"Next year {[int(v) for v in P1]}; in two years {[int(v) for v in P2]}."],
            _slips(rng, L, f, s, P, 2),
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
