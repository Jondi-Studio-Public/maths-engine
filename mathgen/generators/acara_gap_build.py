"""Narrow deterministic generators for the first ACARA gap-build tranche.

These topics intentionally stay scalar and self-contained: no diagram parsing,
units reasoning, or proof/notation assessment is needed by either generator.
"""

from __future__ import annotations

from math import prod

from sympy import Rational, factorint

from ..core import check, generator, pick_distractors


@generator(
    "triangle_area_base_height",
    level=7,
    presets={
        "easy": dict(base_hi=8, height_hi=8),
        "medium": dict(base_hi=16, height_hi=16),
        "hard": dict(base_hi=30, height_hi=24),
    },
)
def triangle_area_base_height(rng, base_hi=16, height_hi=16):
    """Calculate a triangle's area from a supplied base and perpendicular height."""
    if base_hi < 2 or height_hi < 2:
        raise ValueError("base_hi and height_hi must be at least 2")
    base = rng.randint(2, base_hi)
    height = rng.randint(2, height_hi)
    # Keep the scalar answer integral, while still sampling both odd and even
    # measurements when the preset allows them.
    if base * height % 2:
        if height < height_hi:
            height += 1
        elif base < base_hi:
            base += 1
        else:
            height -= 1

    answer = base * height // 2
    check(Rational(base * height, 2) == Rational(answer, 1))
    answer_text = str(answer)
    candidates = [
        2 * answer,  # omitted the one-half
        base + height,  # added the two supplied measurements
        answer + base,
        answer + height,
        answer - 1,
        answer + 1,
        answer + 2,
    ]
    return dict(
        text=(
            f"A triangle has a base of {base} cm and a perpendicular height of {height} cm. Calculate its area in cm²."
        ),
        latex=rf"b={base}\,\mathrm{{cm}},\ h_\perp={height}\,\mathrm{{cm}}.\ "
        rf"\text{{Area}}=\square\,\mathrm{{cm}}^2",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Area = ½ × base × perpendicular height = ½ × {base} × {height}.",
            f"Area = {answer} cm².",
        ],
        distractors=pick_distractors(answer_text, [str(c) for c in candidates]),
    )


_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _prime_power_text(powers: list[tuple[int, int]]) -> str:
    terms = []
    for prime, exponent in powers:
        terms.append(str(prime) if exponent == 1 else f"{prime}{str(exponent).translate(_SUPERSCRIPTS)}")
    return " × ".join(terms)


def _prime_power_latex(powers: list[tuple[int, int]]) -> str:
    return r" \times ".join(str(prime) if exponent == 1 else rf"{prime}^{{{exponent}}}" for prime, exponent in powers)


def _prime_factorisation_composite_form(rng, primes, term_hi, exponent_hi):
    """Number given as a product of composite-base powers (e.g. 6² × 10)."""
    if len(primes) < 2 or term_hi < 2 or exponent_hi < 2:
        raise ValueError("composite_form needs >=2 primes, term_hi >= 2 and exponent_hi >= 2")
    from itertools import combinations

    composites = [a * b for a, b in combinations(sorted(primes), 2)]
    bases = composites + list(primes)
    count = rng.randint(2, min(term_hi, len(bases)))
    # The first factor is always a composite base raised to a power >= 2, so the
    # outer exponent must be distributed over the base's prime factors.
    lead = rng.choice(composites)
    factors = [(lead, rng.randint(2, exponent_hi))]
    rest = [b for b in bases if b != lead]
    for base in rng.sample(rest, count - 1):
        factors.append((base, rng.randint(1, exponent_hi)))
    rng.shuffle(factors)

    def _collect(pairs):
        counts: dict[int, int] = {}
        for base, exponent in pairs:
            for prime, k in factorint(base).items():
                counts[prime] = counts.get(prime, 0) + k * exponent
        return sorted(counts.items())

    powers = _collect(factors)
    number = prod(base**exponent for base, exponent in factors)
    check(factorint(number) == dict(powers))
    answer = _prime_power_text(powers)
    shown = " × ".join(
        str(base) if exponent == 1 else f"{base}{str(exponent).translate(_SUPERSCRIPTS)}" for base, exponent in factors
    )
    shown_latex = r" \times ".join(
        str(base) if exponent == 1 else f"{base}^{{{exponent}}}" for base, exponent in factors
    )
    # Misconceptions: outer exponents dropped; exponent not distributed to the
    # whole base; the evaluated integer; one exponent off.
    dropped = _prime_power_text(_collect([(base, 1) for base, _ in factors]))
    changed = powers.copy()
    changed[0] = (changed[0][0], changed[0][1] + 1)
    reduced = powers.copy()
    reduced[-1] = (reduced[-1][0], max(1, reduced[-1][1] - 1))
    candidates = [dropped, _prime_power_text(changed), str(number), _prime_power_text(reduced)]
    return dict(
        text=f"Write {shown} as a product of prime powers using exponent notation.",
        latex=rf"{shown_latex}=\square\quad\text{{(prime powers)}}",
        answer=answer,
        answer_text=answer,
        answer_latex=_prime_power_latex(powers),
        working=[f"{shown} = {answer}"],
        distractors=pick_distractors(answer, candidates),
    )


@generator(
    "prime_factorisation_exponent",
    level=7,
    presets={
        "easy": dict(primes=(2, 3, 5), term_hi=2, exponent_hi=2),
        "medium": dict(primes=(2, 3, 5, 7), term_hi=3, exponent_hi=3),
        "hard": dict(primes=(2, 3, 5, 7, 11), term_hi=4, exponent_hi=4),
    },
)
def prime_factorisation_exponent(rng, primes=(2, 3, 5, 7), term_hi=3, exponent_hi=3, composite_form=False):
    """Express a supplied natural number as a product of prime powers.

    ``composite_form=True`` supplies the number as a product whose bases are
    themselves composite, e.g. ``6² × 10``: the learner must break each base
    into primes and combine the exponents. The default path is unchanged.
    """
    primes = tuple(primes)
    if not primes or term_hi < 1 or exponent_hi < 1:
        raise ValueError("primes, term_hi and exponent_hi must be non-empty/positive")
    if composite_form:
        return _prime_factorisation_composite_form(rng, primes, term_hi, exponent_hi)
    term_count = rng.randint(1, min(term_hi, len(primes)))
    selected = sorted(rng.sample(primes, term_count))
    exponents = [rng.randint(1, exponent_hi) for _ in selected]
    # Always use a composite input and make exponent notation meaningful. A
    # single prime is upgraded to a square; an all-exponent-one product gets
    # one repeated prime power.
    if term_count == 1 and exponents[0] == 1:
        exponents[0] = 2
    elif all(exponent == 1 for exponent in exponents):
        exponents[0] = 2
    powers = list(zip(selected, exponents))
    number = prod(prime**exponent for prime, exponent in powers)
    factor_counts = {prime: exponent for prime, exponent in powers}
    check(factorint(number) == factor_counts)

    answer = _prime_power_text(powers)
    expanded = " × ".join(str(prime) for prime, exponent in powers for _ in range(exponent))
    # A nearby exponent is a useful distractor even for one-term powers. The
    # integer itself and the expanded product target common notation errors.
    changed = powers.copy()
    changed[0] = (changed[0][0], changed[0][1] + 1)
    wrong_exponent = _prime_power_text(changed)
    candidates = [expanded, wrong_exponent, str(number)]
    if powers[0][1] > 1:
        reduced = powers.copy()
        reduced[0] = (reduced[0][0], reduced[0][1] - 1)
        candidates.append(_prime_power_text(reduced))
    return dict(
        text=f"Write {number} as a product of prime powers using exponent notation.",
        latex=rf"{number}=\square\quad\text{{(prime powers)}}",
        answer=answer,
        answer_text=answer,
        answer_latex=_prime_power_latex(powers),
        working=[f"{number} = {answer}"],
        distractors=pick_distractors(answer, candidates),
    )
