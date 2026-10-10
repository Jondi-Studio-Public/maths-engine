"""Senior similarity conditions and linear scale-factor skills.

Two generators:

- ``similarity_conditions_classify``
  (``senior-review-the-conditions-for-similarity-of-two-dimensional``, ACMGM021)
  -- a fixed-option MCQ classification of two triangles as similar by AAA, SSS,
  SAS, or not similar. Ground truth for "similar" is always the SORTED-side (or
  sorted-angle) proportionality test, never the positionally-labelled
  correspondence stated in the prose -- similarity is a property of the two
  triangles' shapes, not of which letter is written next to which number, and a
  positional-only check can accidentally mark a genuinely similar pair "not
  similar" (or vice versa) under a different vertex correspondence.

- ``similar_figure_scale_factor_length``
  (``senior-use-the-scale-factor-for-two-similar-figures``, ACMGM022) -- a plain
  linear (k^1) length/perimeter scaling computation. Never mentions area,
  surface area or volume at any tier -- that is the entire boundary against the
  neighbouring ``scale_factor_similar_areas``/``scale_factor_similar_solids``
  skills in ``p6f_scale_factor_build.py``.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors

# ============================================================================
# Skill 1: similarity_conditions_classify (ACMGM021)
# ============================================================================

OPTION_AAA = "Similar by AAA (all pairs of corresponding angles are equal)"
OPTION_SSS = "Similar by SSS (all three pairs of corresponding sides are in the same ratio)"
OPTION_SAS = (
    "Similar by SAS (one pair of corresponding angles is equal and the sides that include it are in the same ratio)"
)
OPTION_NOT_SIMILAR = "Not similar (the given information does not establish similarity)"
ALL_OPTIONS = (OPTION_AAA, OPTION_SSS, OPTION_SAS, OPTION_NOT_SIMILAR)

# Anti-sliver bounds.
AAA_ANGLE_MIN, AAA_ANGLE_MAX = 20, 130
SSS_SIDE_MIN, SSS_SIDE_MAX = 4, 24
SSS_RATIO_MAX = 3
SSS_MARGIN = Fraction(115, 100)  # sum of two shorter sides >= 1.15x the longest
SSS_ANGLE_MIN, SSS_ANGLE_MAX = 20, 140  # every SSS triangle's own angles, not just its sides
SAS_SIDE_MIN, SAS_SIDE_MAX = 4, 24
SAS_RATIO_MAX = 3
SAS_ANGLE_MIN, SAS_ANGLE_MAX = 20, 140

# Small scale-factor pool bounded away from 1 and above 3 (matching the
# anti-sliver ratio bound): {4/3, 3/2, 5/3, 2, 5/2, 8/3, 3} and reciprocals.
RATIO_POOL = (
    (4, 3),
    (3, 4),
    (3, 2),
    (2, 3),
    (5, 3),
    (3, 5),
    (2, 1),
    (1, 2),
    (5, 2),
    (2, 5),
    (8, 3),
    (3, 8),
    (3, 1),
    (1, 3),
)

_MAX_TRIES = 500


# --- AAA (easy) --------------------------------------------------------------


def _draw_aaa_triangle(rng):
    for _ in range(_MAX_TRIES):
        a1 = rng.randint(AAA_ANGLE_MIN + 5, AAA_ANGLE_MAX - 5)
        a2 = rng.randint(AAA_ANGLE_MIN + 5, AAA_ANGLE_MAX - 5)
        a3 = 180 - a1 - a2
        if AAA_ANGLE_MIN <= a3 <= AAA_ANGLE_MAX:
            return a1, a2, a3
    raise RuntimeError("could not draw a plausible AAA triangle")


def _draw_not_similar_aaa(rng, a1, a2, a3):
    base = sorted([a1, a2, a3])
    for _ in range(_MAX_TRIES):
        change_first = rng.choice([True, False])
        delta = rng.choice([-1, 1]) * rng.randint(10, 25)
        d1 = a1 + delta if change_first else a1
        d2 = a2 + delta if not change_first else a2
        if not (AAA_ANGLE_MIN <= d1 <= AAA_ANGLE_MAX and AAA_ANGLE_MIN <= d2 <= AAA_ANGLE_MAX):
            continue
        f3 = 180 - d1 - d2
        if not (AAA_ANGLE_MIN <= f3 <= AAA_ANGLE_MAX):
            continue
        if sorted([d1, d2, f3]) == base:
            continue  # accidentally still similar under some correspondence
        return d1, d2, f3
    raise RuntimeError("could not draw a not-similar AAA triangle")


def _is_aaa_similar(a1, a2, a3, d1, d2, f3) -> bool:
    """Ground truth: similar iff the two angle multisets match -- independent
    of which letter was written next to which drawn value."""
    return sorted([a1, a2, a3]) == sorted([d1, d2, f3])


def _check_aaa(a1, a2, a3, d1, d2, f3, expected_similar):
    check(_is_aaa_similar(a1, a2, a3, d1, d2, f3) == expected_similar, "AAA classification mismatch")


# --- SSS (medium) ------------------------------------------------------------


def _sss_triangle_angles(sides) -> tuple[float, float, float]:
    """All three angles (degrees) of a triangle from its three sides, via Law
    of Cosines."""
    a, b, c = sides

    def ang(x, y, z):
        cos_v = (x * x + y * y - z * z) / (2 * x * y)
        cos_v = max(-1.0, min(1.0, cos_v))
        return math.degrees(math.acos(cos_v))

    return ang(b, c, a), ang(a, c, b), ang(a, b, c)


def _sss_sides_ok(sides) -> bool:
    lo, hi = min(sides), max(sides)
    if lo < SSS_SIDE_MIN or hi > SSS_SIDE_MAX:
        return False
    if hi > SSS_RATIO_MAX * lo:
        return False
    s = sorted(sides)
    if Fraction(s[0] + s[1]) < SSS_MARGIN * s[2]:
        return False
    # The ratio bound and the triangle-inequality margin are NOT a proxy for
    # "no sliver angle": (4, 10, 12) clears both yet has an 18.2 degree angle.
    # Check every angle directly.
    if not all(SSS_ANGLE_MIN <= ang <= SSS_ANGLE_MAX for ang in _sss_triangle_angles(sides)):
        return False
    return True


def _draw_sss_similar(rng):
    for _ in range(_MAX_TRIES):
        p, q = rng.choice(RATIO_POOL)
        lo = -(-SSS_SIDE_MIN // q)
        hi = SSS_SIDE_MAX // q
        if hi < lo:
            continue
        us = tuple(rng.randint(lo, hi) for _ in range(3))
        abc = tuple(q * u for u in us)
        if not _sss_sides_ok(abc):
            continue
        defs = tuple(p * u for u in us)
        if not _sss_sides_ok(defs):
            continue
        return abc, defs
    raise RuntimeError("could not draw a plausible similar SSS pair")


def _draw_sss_not_similar(rng):
    for _ in range(_MAX_TRIES):
        p, q = rng.choice(RATIO_POOL)
        p2, q2 = rng.choice(RATIO_POOL)
        if (p, q) == (p2, q2):
            continue
        L = math.lcm(q, q2)
        hi = SSS_SIDE_MAX // L
        if hi < 1:
            continue
        us = tuple(rng.randint(1, hi) for _ in range(3))
        abc = tuple(L * u for u in us)
        if not _sss_sides_ok(abc):
            continue
        idx_diff = rng.randrange(3)
        defs = []
        for i, side in enumerate(abc):
            if i == idx_diff:
                defs.append((side // q2) * p2)
            else:
                defs.append((side // q) * p)
        defs = tuple(defs)
        if not _sss_sides_ok(defs):
            continue
        if _is_sss_similar(abc, defs):
            continue  # accidentally proportional under some correspondence
        return abc, defs
    raise RuntimeError("could not draw a not-similar SSS pair")


def _is_sss_similar(abc, defs) -> bool:
    """Ground truth: sort both triples ascending and require a single
    consistent ratio -- independent of the positional AB/BC/CA labelling."""
    abc_sorted = sorted(abc)
    def_sorted = sorted(defs)
    ratios = {Fraction(d, a) for d, a in zip(def_sorted, abc_sorted)}
    return len(ratios) == 1


def _check_sss(abc, defs, expected_similar):
    check(_is_sss_similar(abc, defs) == expected_similar, "SSS classification mismatch")


# --- SAS (hard) ---------------------------------------------------------------


def _law_of_cosines_third_side(a, b, angle_deg) -> float:
    c2 = a * a + b * b - 2 * a * b * math.cos(math.radians(angle_deg))
    return math.sqrt(max(c2, 0.0))


def _other_two_angles(a, b, angle_c_deg, c) -> tuple[float, float]:
    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    cos_a = max(-1.0, min(1.0, cos_a))
    angle_a = math.degrees(math.acos(cos_a))
    angle_b = 180.0 - angle_c_deg - angle_a
    return angle_a, angle_b


def _sas_triangle_ok(a, b, angle_c_deg) -> bool:
    if not (SAS_SIDE_MIN <= a <= SAS_SIDE_MAX and SAS_SIDE_MIN <= b <= SAS_SIDE_MAX):
        return False
    if max(a, b) > SAS_RATIO_MAX * min(a, b):
        return False
    if not (SAS_ANGLE_MIN <= angle_c_deg <= SAS_ANGLE_MAX):
        return False
    c = _law_of_cosines_third_side(a, b, angle_c_deg)
    sides = sorted([a, b, c])
    if sides[2] > SAS_RATIO_MAX * sides[0]:
        return False
    angle_a, angle_b = _other_two_angles(a, b, angle_c_deg, c)
    return all(SAS_ANGLE_MIN <= ang <= SAS_ANGLE_MAX for ang in (angle_a, angle_b, angle_c_deg))


def _draw_sas_similar(rng):
    for _ in range(_MAX_TRIES):
        p, q = rng.choice(RATIO_POOL)
        lo = -(-SAS_SIDE_MIN // min(p, q))
        hi = SAS_SIDE_MAX // max(p, q)
        if hi < lo:
            continue
        u1 = rng.randint(lo, hi)
        u2 = rng.randint(lo, hi)
        a, b = q * u1, q * u2
        angle_c = rng.randint(SAS_ANGLE_MIN, SAS_ANGLE_MAX)
        if not _sas_triangle_ok(a, b, angle_c):
            continue
        def_a, def_b = p * u1, p * u2
        if not _sas_triangle_ok(def_a, def_b, angle_c):
            continue
        return (a, b, angle_c), (def_a, def_b, angle_c)
    raise RuntimeError("could not draw a plausible similar SAS pair")


def _draw_sas_not_similar(rng):
    for _ in range(_MAX_TRIES):
        mode = rng.choice(["ratio", "angle"])
        p, q = rng.choice(RATIO_POOL)
        if mode == "ratio":
            p2, q2 = rng.choice(RATIO_POOL)
            if (p, q) == (p2, q2):
                continue
            hi1 = SAS_SIDE_MAX // q
            hi2 = SAS_SIDE_MAX // q2
            if hi1 < 1 or hi2 < 1:
                continue
            u1 = rng.randint(1, hi1)
            u2 = rng.randint(1, hi2)
            a, b = q * u1, q2 * u2
            angle_c = rng.randint(SAS_ANGLE_MIN, SAS_ANGLE_MAX)
            if not _sas_triangle_ok(a, b, angle_c):
                continue
            def_a, def_b = p * u1, p2 * u2
            def_angle = angle_c
        else:
            lo = -(-SAS_SIDE_MIN // min(p, q))
            hi = SAS_SIDE_MAX // max(p, q)
            if hi < lo:
                continue
            u1 = rng.randint(lo, hi)
            u2 = rng.randint(lo, hi)
            a, b = q * u1, q * u2
            angle_c = rng.randint(SAS_ANGLE_MIN, SAS_ANGLE_MAX)
            if not _sas_triangle_ok(a, b, angle_c):
                continue
            def_a, def_b = p * u1, p * u2
            delta = rng.choice([-1, 1]) * rng.randint(10, 40)
            def_angle = angle_c + delta
            if not (SAS_ANGLE_MIN <= def_angle <= SAS_ANGLE_MAX):
                continue

        if not _sas_triangle_ok(def_a, def_b, def_angle):
            continue
        if _is_sas_similar((a, b, angle_c), (def_a, def_b, def_angle)):
            continue  # accidentally still similar under some correspondence
        return (a, b, angle_c), (def_a, def_b, def_angle)
    raise RuntimeError("could not draw a not-similar SAS pair")


def _is_sas_similar(abc, defs, tol: float = 1e-6) -> bool:
    """Ground truth via Law of Cosines: derive both triangles' third side
    independently of construction's own bookkeeping, sort each triangle's
    three sides ascending, and require a single consistent ratio."""
    a, b, angle_c = abc
    d, e, angle_f = defs
    c = _law_of_cosines_third_side(a, b, angle_c)
    f = _law_of_cosines_third_side(d, e, angle_f)
    abc_sorted = sorted([a, b, c])
    def_sorted = sorted([d, e, f])
    ratios = [x / y for x, y in zip(def_sorted, abc_sorted)]
    return (max(ratios) - min(ratios)) <= tol * max(ratios)


def _check_sas(abc, defs, expected_similar):
    check(_is_sas_similar(abc, defs) == expected_similar, "SAS classification mismatch")


@generator(
    "similarity_conditions_classify",
    level=12,
    presets={
        "easy": {"mode": "aaa"},
        "medium": {"mode": "sss"},
        "hard": {"mode": "sas"},
    },
)
def similarity_conditions_classify(rng, mode: str = "aaa") -> dict:
    """Classify two triangles as similar by AAA/SSS/SAS or not similar, from a
    fixed pool of four verbatim MCQ options."""
    if mode not in ("aaa", "sss", "sas"):
        raise ValueError(f"mode must be 'aaa', 'sss' or 'sas'; got {mode!r}")

    # Construct-to-target the SIMILAR/NOT_SIMILAR class FIRST,
    # then build the numbers to realise it.
    is_similar = rng.choice([True, False])

    if mode == "aaa":
        a1, a2, a3 = _draw_aaa_triangle(rng)
        if is_similar:
            d1, d2, f3 = a1, a2, a3
        else:
            d1, d2, f3 = _draw_not_similar_aaa(rng, a1, a2, a3)
        _check_aaa(a1, a2, a3, d1, d2, f3, is_similar)
        text = (
            f"Triangle ABC has angle A = {a1} degrees and angle B = {a2} degrees. "
            f"Triangle DEF has angle D = {d1} degrees and angle E = {d2} degrees. "
            f"Which of these is correct?"
        )
        criterion_option = OPTION_AAA

    elif mode == "sss":
        if is_similar:
            abc, defs = _draw_sss_similar(rng)
        else:
            abc, defs = _draw_sss_not_similar(rng)
        _check_sss(abc, defs, is_similar)
        text = (
            f"Triangle ABC has sides AB = {abc[0]} cm, BC = {abc[1]} cm and CA = {abc[2]} cm. "
            f"Triangle DEF has sides DE = {defs[0]} cm, EF = {defs[1]} cm and FD = {defs[2]} cm. "
            f"Which of these is correct?"
        )
        criterion_option = OPTION_SSS

    else:  # mode == "sas"
        if is_similar:
            abc, defs = _draw_sas_similar(rng)
        else:
            abc, defs = _draw_sas_not_similar(rng)
        _check_sas(abc, defs, is_similar)
        text = (
            f"Triangle ABC has AB = {abc[0]} cm, BC = {abc[1]} cm and the included angle "
            f"ABC = {abc[2]} degrees. Triangle DEF has DE = {defs[0]} cm, EF = {defs[1]} cm and "
            f"the included angle DEF = {defs[2]} degrees. Which of these is correct?"
        )
        criterion_option = OPTION_SAS

    answer = criterion_option if is_similar else OPTION_NOT_SIMILAR
    check(answer in (criterion_option, OPTION_NOT_SIMILAR), "answer must be the tier's own criterion or not-similar")
    distractors = [opt for opt in ALL_OPTIONS if opt != answer]
    check(len(distractors) == 3, "expected exactly three padding options")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        distractors=distractors,
    )


# ============================================================================
# Skill 2: similar_figure_scale_factor_length (ACMGM022)
# ============================================================================

# Local literal pool of small curriculum-style ratio pairs.
RATIO_PAIRS_LEN = (
    (2, 3),
    (3, 2),
    (2, 5),
    (5, 2),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
    (5, 6),
    (6, 5),
)

# Pool for medium/hard, restricted so the derived-scale-factor arithmetic
# (p*q**2*w and p**2*q*w) stays within [3, 60] for w in {1, 2} -- keeps every
# stated/derived length "sensible" and every distractor an exact integer.
DERIVED_RATIO_PAIRS = ((2, 3), (3, 2), (2, 5), (5, 2), (3, 4), (4, 3))

LEN_MIN, LEN_MAX = 3, 60
PERIMETER_MIN_MULTIPLE = 3  # perimeter >= 3x the stated side (a real polygon)
MAX_ANSWER = 999  # keep every number under four digits


def _valid_w_choices(p: int, q: int, cap: int) -> list[int]:
    return [w for w in (1, 2, 3) if p * q * q * w <= cap and p * p * q * w <= cap]


def _draw_easy_scale_length(rng):
    for _ in range(_MAX_TRIES):
        p, q = rng.choice(RATIO_PAIRS_LEN)
        s = rng.choice([1, 2, 3])
        side_a = p * q * q * s
        answer = p * p * q * s
        uses_squared_factor = p**3 * s
        swaps_p_and_q = q**3 * s
        if max(side_a, answer, uses_squared_factor, swaps_p_and_q) > MAX_ANSWER:
            continue
        return p, q, s, side_a, answer, uses_squared_factor, swaps_p_and_q
    raise RuntimeError("could not draw a plausible easy scale-length question")


def _draw_medium_scale_length(rng):
    for _ in range(_MAX_TRIES):
        p, q = rng.choice(DERIVED_RATIO_PAIRS)
        choices = _valid_w_choices(p, q, LEN_MAX)
        if not choices:
            continue
        w = rng.choice(choices)
        length_a2 = p * q * q * w
        answer = p * p * q * w
        uses_squared_factor = p**3 * w
        swaps_p_and_q = q**3 * w

        t1 = rng.randint(1, max(1, min(60 // q, 60 // p)))
        length_a1 = q * t1
        length_b1 = p * t1
        if length_a1 == length_a2:
            continue
        if not (LEN_MIN <= length_a1 <= LEN_MAX and LEN_MIN <= length_b1 <= LEN_MAX):
            continue
        if not (LEN_MIN <= length_a2 <= LEN_MAX and LEN_MIN <= answer <= LEN_MAX):
            continue
        if max(uses_squared_factor, swaps_p_and_q) > MAX_ANSWER:
            continue
        return dict(
            p=p,
            q=q,
            w=w,
            length_a1=length_a1,
            length_b1=length_b1,
            length_a2=length_a2,
            answer=answer,
            uses_squared_factor=uses_squared_factor,
            swaps_p_and_q=swaps_p_and_q,
        )
    raise RuntimeError("could not draw a plausible medium scale-length question")


def _draw_hard_scale_length(rng):
    for _ in range(_MAX_TRIES):
        p, q = rng.choice(DERIVED_RATIO_PAIRS)
        t1 = rng.randint(1, max(1, min(30 // q, 30 // p)))
        length_a1 = q * t1
        length_b1 = p * t1
        if not (LEN_MIN <= length_a1 <= LEN_MAX and LEN_MIN <= length_b1 <= LEN_MAX):
            continue

        cap = max(p**3, q**3, p * q * q, p * p * q)
        w2_max = MAX_ANSWER // cap
        w2_min_needed = -(-(PERIMETER_MIN_MULTIPLE * length_a1) // (p * q * q))
        if w2_max < max(1, w2_min_needed):
            continue
        w2 = rng.randint(max(1, w2_min_needed), w2_max)

        perimeter_a = p * q * q * w2
        answer = p * p * q * w2
        uses_squared_factor = p**3 * w2
        swaps_p_and_q = q**3 * w2
        if perimeter_a < PERIMETER_MIN_MULTIPLE * length_a1:
            continue
        if max(perimeter_a, answer, uses_squared_factor, swaps_p_and_q) > MAX_ANSWER:
            continue
        return dict(
            p=p,
            q=q,
            w2=w2,
            length_a1=length_a1,
            length_b1=length_b1,
            perimeter_a=perimeter_a,
            answer=answer,
            uses_squared_factor=uses_squared_factor,
            swaps_p_and_q=swaps_p_and_q,
        )
    raise RuntimeError("could not draw a plausible hard scale-length question")


@generator(
    "similar_figure_scale_factor_length",
    level=12,
    presets={
        "easy": {"mode": "direct_ratio"},
        "medium": {"mode": "derive_from_pair"},
        "hard": {"mode": "derive_perimeter"},
    },
)
def similar_figure_scale_factor_length(rng, mode: str = "direct_ratio") -> dict:
    """Scale factor applied to a plain LENGTH or PERIMETER of similar figures --
    never an area, surface area or volume (that boundary belongs to the
    neighbouring skills ACMGM024/ACMGM025 in p6f_scale_factor_build.py)."""
    if mode not in ("direct_ratio", "derive_from_pair", "derive_perimeter"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "direct_ratio":
        p, q, s, side_a, answer, uses_squared_factor, swaps_p_and_q = _draw_easy_scale_length(rng)

        text = (
            f"Figure A and Figure B are similar. The scale factor from Figure A "
            f"to Figure B is {p}/{q}. A side of Figure A measures {side_a} cm. "
            f"What is the length of the corresponding side of Figure B, in cm?"
        )

        # Independent check: exact cross-multiplication identity, a route
        # distinct from however construction built the integer via *s.
        check_equivalent(answer * q, side_a * p, "easy tier cross-multiplication failed")

        forgets_to_scale = side_a
        check_distinct(answer, uses_squared_factor, swaps_p_and_q, forgets_to_scale)
        candidates = [uses_squared_factor, swaps_p_and_q, forgets_to_scale]

    elif mode == "derive_from_pair":
        d = _draw_medium_scale_length(rng)
        text = (
            f"Figure A and Figure B are similar. A side of Figure A measures "
            f"{d['length_a1']} cm; the corresponding side of Figure B measures "
            f"{d['length_b1']} cm. A second side of Figure A measures "
            f"{d['length_a2']} cm. What is the length of the corresponding side "
            f"of Figure B, in cm?"
        )
        answer = d["answer"]

        # Independent check: cross-multiplication using the DERIVED pair only
        # (never the pool values p, q used to construct the draw).
        check_equivalent(
            answer * d["length_a1"], d["length_a2"] * d["length_b1"], "medium tier cross-multiplication failed"
        )

        forgets_to_scale = d["length_a2"]
        check_distinct(answer, d["uses_squared_factor"], d["swaps_p_and_q"], forgets_to_scale)
        candidates = [d["uses_squared_factor"], d["swaps_p_and_q"], forgets_to_scale]

    else:  # mode == "derive_perimeter"
        d = _draw_hard_scale_length(rng)
        text = (
            f"Figure A and Figure B are similar. A side of Figure A measures "
            f"{d['length_a1']} cm; the corresponding side of Figure B measures "
            f"{d['length_b1']} cm. Figure A has a perimeter of {d['perimeter_a']} cm. "
            f"What is the perimeter of Figure B, in cm?"
        )
        answer = d["answer"]

        check_equivalent(
            answer * d["length_a1"], d["perimeter_a"] * d["length_b1"], "hard tier cross-multiplication failed"
        )

        forgets_to_scale = d["perimeter_a"]
        check_distinct(answer, d["uses_squared_factor"], d["swaps_p_and_q"], forgets_to_scale)
        candidates = [d["uses_squared_factor"], d["swaps_p_and_q"], forgets_to_scale]

    answer_text = str(answer)
    return {
        "text": text,
        "latex": text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer),
        "distractors": pick_distractors(answer_text, candidates),
    }
