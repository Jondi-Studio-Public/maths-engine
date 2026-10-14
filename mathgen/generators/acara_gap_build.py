"""Narrow deterministic generators for the first ACARA gap-build tranche.

These topics intentionally stay scalar and self-contained: no units reasoning or proof/notation assessment
is needed by either generator. ``triangle_area_base_height`` can also draw its triangle (``figure=True``, what
the skill's sources use): a labelled triangle with the perpendicular height dashed and square-marked, drawn
to scale, with the slanted side labelled too at the hard tier. The words-only form is the default.
"""

from __future__ import annotations

from math import prod

from sympy import Rational, factorint

from .. import figures
from ..core import attempts, check, generator, latex_prompt, pick_distractors
from ..figure_kinds import space
from . import _space_kit as kit


@generator(
    "triangle_area_base_height",
    level=7,
    presets={
        "easy": dict(base_hi=8, height_hi=8),
        "medium": dict(base_hi=16, height_hi=16),
        "hard": dict(base_hi=30, height_hi=24),
    },
)
def triangle_area_base_height(rng, base_hi=16, height_hi=16, slant=False, figure=False, units=False):
    """Calculate a triangle's area from a supplied base and perpendicular height.

    With ``slant=True`` the prompt also supplies a slanted side (the hypotenuse
    of a real Pythagorean triple with the perpendicular height), and the learner
    must pick the perpendicular height rather than the slant. With ``figure=True`` the triangle is drawn (to
    scale) with its sides labelled instead of described (``units=True`` gives the lengths cm, m or mm instead of
    always cm); a draw whose labels would show the answer is redone."""
    if base_hi < 2 or height_hi < 2:
        raise ValueError("base_hi and height_hi must be at least 2")
    if not figure:
        return _triangle_area(rng, base_hi, height_hi, slant, None, False)
    for _ in attempts(60, "a triangle picture that does not show its own answer"):
        d = _triangle_area(rng, base_hi, height_hi, slant, True, units)
        if d is not None:
            return d
    raise AssertionError("unreachable")


_NAMES = ("ABC", "PQR", "XYZ", "DEF", "LMN", "RST")
_AREA_UNITS = {"cm": "cm²", "m": "m²", "mm": "mm²"}
_PLAIN_PHRASES = (
    "Calculate the area of triangle {name}, in {a}.",
    "Find the area of triangle {name} shown, in {a}.",
    "The diagram shows triangle {name}. What is its area, in {a}?",
    "Work out the area of triangle {name}, in {a}.",
    "Determine the area of triangle {name} in the diagram, in {a}.",
)
_UNITS_PHRASES = (
    "Triangle {name} is drawn with its perpendicular height dashed. Find its area, in {a}.",
    "The dashed line in triangle {name} is the perpendicular height. Calculate the area, in {a}.",
    "Find the area of triangle {name}, in {a}. The dashed line is perpendicular to the base.",
    "What is the area of triangle {name}, in {a}? The perpendicular height is shown dashed.",
    "Calculate the area of triangle {name}, in {a}. The square marks a right angle.",
)
_SLANT_PHRASES = (
    "Find the area of triangle {name} in the diagram, in {a}.",
    "Triangle {name} has three lengths marked. Calculate its area, in {a}.",
    "Use the lengths marked on triangle {name} to find its area, in {a}.",
    "Work out the area of the triangle {name} drawn here, in {a}.",
    "What is the area of triangle {name}, in {a}?",
)


def _triangle_figure(rng, base, height, side=None, unit="cm", names="ABC"):
    """The triangle with base ``base`` (horizontal before it is turned), perpendicular height ``height`` dashed
    from the apex to the base with a right-angle mark, and ``side`` (the slanted side, when given) labelled.
    Drawn to scale, apex over the base; with a slanted side the apex is exactly ``sqrt(side**2 - height**2)``
    along the base, so the slant is the true length."""
    if side is not None:
        offset = round((side * side - height * height) ** 0.5)
        check(offset * offset + height * height == side * side, "the slanted side is not a whole-number slant")
    else:
        offset = rng.uniform(0.15, 0.85) * base
    scale = min(80 / base, 70 / height)
    raw = {"A": (0.0, 0.0), "B": (base * scale, 0.0), "C": (offset * scale, height * scale)}
    foot = "D"
    if abs(offset - base) < 1e-9:
        foot = "B"  # the apex is directly above B: the triangle itself is right-angled there
    else:
        raw["D"] = (offset * scale, 0.0)
    pos = kit.oriented(raw, rng.randrange(8))
    points = [{"id": k, "at": v, **({"label": names["ABC".index(k)]} if k in "ABC" else {})} for k, v in pos.items()]
    segments = [
        {"from": "A", "to": "B", "label": f"{base} {unit}"},
        {"from": "C", "to": foot, "label": f"{height} {unit}", "dashed": True},
    ]
    if side is not None:
        segments.append({"from": "A", "to": "C", "label": f"{side} {unit}"})
    frm, to = ("B", "C") if foot == "D" else ("A", "C")  # the rays out of the foot: along the base, up the height
    if space.sweep_degrees(pos[foot], pos[frm], pos[to]) > 180:
        frm, to = to, frm
    return space.diagram(
        alt="A triangle with its vertices named and its base and its perpendicular height marked.",
        points=points,
        segments=segments,
        polygons=[{"points": ["A", "B", "C"]}],
        angles=[{"at": foot, "from": frm, "to": to, "right": True}],
        not_to_scale=False,
    )


def _read_triangle_area(question) -> str:
    """Re-derive the area from the picture alone: the dashed segment is the perpendicular height, the base is
    the other labelled side through the foot of the right-angle mark."""
    fig = question.figure
    pos = {p["id"]: tuple(p["at"]) for p in fig["points"]}
    foot = fig["angles"][0]["at"]
    dashed = [s for s in fig["segments"] if s.get("dashed")]
    check(len(dashed) == 1, "expected one dashed height")
    base = [
        s
        for s in fig["segments"]
        if not s.get("dashed") and kit.point_on_segment(pos[foot], pos[s["from"]], pos[s["to"]])
    ]
    check(len(base) == 1, "expected one base through the foot of the height")
    height_cm, base_cm = int(dashed[0]["label"].split()[0]), int(base[0]["label"].split()[0])
    check(height_cm * base_cm % 2 == 0, "the area is not a whole number")
    return str(height_cm * base_cm // 2)


figures.register_reader("triangle_area_base_height", _read_triangle_area)


def _triangle_area(rng, base_hi, height_hi, slant, figure, units):
    if slant:
        return _triangle_area_slant(rng, base_hi, height_hi, figure, units)
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
    fig = None
    if figure:
        fig = _triangle_figure(rng, base, height, None, *_unit_and_names(rng, units))
        if kit.leaks(fig, answer_text):
            return None
    d = dict(
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
    return _with_figure(rng, d, fig, units, False)


def _unit_and_names(rng, units):
    """The length unit (always cm unless ``units``) and the three vertex names the figure and prompt use."""
    return (rng.choice(tuple(_AREA_UNITS)) if units else "cm"), rng.choice(_NAMES)


def _with_figure(rng, d, fig, units, slant):
    """Swap the words that describe the triangle for the drawn triangle."""
    if fig is not None:
        label = next(s["label"] for s in fig["segments"] if not s.get("dashed"))
        unit = label.split()[1]
        names = "".join(p["label"] for p in fig["points"] if "label" in p)
        pool = _SLANT_PHRASES if slant else _UNITS_PHRASES if units else _PLAIN_PHRASES
        d["figure"] = fig
        d["text"] = rng.choice(pool).format(name=names, a=_AREA_UNITS[unit])
        d["working"] = [line.replace("cm²", _AREA_UNITS[unit]) for line in d["working"]]  # the question's own unit
        d["latex"] = latex_prompt(d["text"])
    return d


_SLANT_TRIPLES = ((3, 4, 5), (4, 3, 5), (5, 12, 13), (12, 5, 13), (8, 15, 17), (15, 8, 17))


def _triangle_area_slant(rng, base_hi, height_hi, figure=False, units=False):
    """Base, slanted side and perpendicular height: area uses the height only."""
    for _attempt in range(200):
        offset, rise, side = rng.choice(_SLANT_TRIPLES)
        k_max = min(height_hi // rise, base_hi // offset)
        if k_max < 1:
            continue
        k = rng.randint(1, k_max)
        offset, height, slant_side = offset * k, rise * k, side * k
        base = rng.randint(max(2, offset), base_hi)
        if base * height % 2:
            if base < base_hi:
                base += 1
            elif base > offset:
                base -= 1
            else:
                continue
        if base < offset or base * height % 2:
            continue
        break
    else:
        raise ValueError("no slant triangle fits base_hi and height_hi")
    check(slant_side > height and offset**2 + height**2 == slant_side**2)

    answer = base * height // 2
    check(Rational(base * height, 2) == Rational(answer, 1))
    answer_text = str(answer)
    slant_area = base * slant_side // 2 if base * slant_side % 2 == 0 else base * slant_side
    candidates = [
        slant_area,  # used the slanted side as the height
        base * height,  # omitted the one-half
        base + slant_side + height,  # added the lengths
        base * slant_side,
        answer + base,
        answer + height,
    ]
    distractors = pick_distractors(answer_text, [str(c) for c in candidates])
    check(len(distractors) >= 3)
    fig = None
    if figure:
        fig = _triangle_figure(rng, base, height, slant_side, *_unit_and_names(rng, units))
        if kit.leaks(fig, answer_text):
            return None
    d = dict(
        text=(
            f"A triangle has a base of {base} cm, a slanted side of {slant_side} cm "
            f"and a perpendicular height of {height} cm. Calculate its area in cm²."
        ),
        latex=rf"b={base}\,\mathrm{{cm}},\ s={slant_side}\,\mathrm{{cm}},\ h_\perp={height}\,\mathrm{{cm}}.\ "
        rf"\text{{Area}}=\square\,\mathrm{{cm}}^2",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The slanted side is not used: area = ½ × base × perpendicular height = ½ × {base} × {height}.",
            f"Area = {answer} cm².",
        ],
        distractors=distractors,
    )
    return _with_figure(rng, d, fig, units, True)


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
