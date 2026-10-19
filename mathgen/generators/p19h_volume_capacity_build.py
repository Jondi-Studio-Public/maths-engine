"""Generates `p19h_volcap_general_relationship` and `p19h_volcap_named_equivalence` questions converting between
volume and capacity units.

The two rows are kept genuinely distinct by TARGET UNIT and DIRECTION, and that separation is built into the
rendered question, not just the contract prose:

- ``p19h_volcap_general_relationship`` (skill
  ``senior-understand-the-relationship-between-volume-and-capacity``,
  ACMEM028) -- target units are **mL or L only, never kL**; direction is
  **volume -> capacity ONLY** (forward). The m^3 -> L factor of 1000 is
  NEVER stated in the prompt and is not a lookup constant in this module --
  every occurrence is DERIVED, in both construction and the independent
  check, from the cm^3 = mL fact: ``1 m^3 = 1,000,000 cm^3 = 1,000,000 mL =
  1000 L`` (see ``_m3_to_L`` below).
- ``p19h_volcap_named_equivalence`` (skill
  ``senior-recognise-relations-between-volume-and-capacity-recognising-that``,
  ACMEM102) -- target units are **mL or kL only, never L**; **either
  direction**. Both named facts (1 cm^3 = 1 mL, 1 m^3 = 1 kL) are handed to
  the learner directly, restated in the prompt where useful (permitted for
  this skill -- the skill's own curriculum name IS the lookup fact) -- the
  reasoning content is applying/chaining a GIVEN fact; from ``medium`` tier
  up, combining TWO separately-stated quantities (never a single restated
numeral -- see ``_build_vc2_medium``, fixed: the m^3/kL
  pair is numeral-preserving by construction, so a single-quantity medium
  tier can never avoid the key being the prompt's own number; instead
  medium states one quantity in m^3 and a second, separate quantity in kL,
  and the answer is their sum); from ``hard`` tier up, additionally the
  ordinary metric ladder between mL and kL (x1,000,000), never a derivation
  of the facts themselves.

Why the two rows cannot collapse into "the same maths, two unit labels": 1 m^3 = 1 kL is a numeral-preserving
equivalence (both sides are literally the same number), while 1 m^3 = 1000 L is NOT -- so skill 1's m^3 -> L
tiers (medium/exam) always require a genuine x1000 multiplication the learner must know/derive, while skill
2's m^3 -> kL tiers (easy/medium) never do. Skill 2's hard/exam tiers introduce their OWN genuine arithmetic
instead: the ordinary mL<->kL metric-ladder scaling (x1,000,000), applied to an ALREADY-GIVEN volume, never to
a volume computed from dimensions -- while skill 1's hard/exam tiers do the opposite: dimensions must be
multiplied into a volume FIRST (a fresh reasoning step neither skill 2 tier ever asks for), and the resulting
capacity conversion is either trivial (hard, cm^3 -> mL, 1:1) or the derived x1000 (exam, m^3 -> L).

Every displayed answer is an exact whole number (both named facts and the
mL<->kL metric-ladder factor are exact powers of ten; multiplying whole-cm/
whole-m dimensions together stays a whole number) -- no rounding, no stated
decimal precision anywhere in this module, so `answer_text` is always a bare
integer string and the real grader (`mathgen/grading/typed.py`'s typed-number path)
requires an EXACT match.

Distractor collisions are avoided by construction with a bounded
redraw-until-distinct loop (`pairwise_distinct` guard) wherever more than one
candidate depends on the same drawn values, then `check_pairwise_distinct` as
a hard final invariant -- the same discipline `p19h_energy_build.py` uses,
because some of these particular catalogues/ranges DO coincide algebraically
(e.g. the hard tier's "sum of dimensions" and "product of two dimensions"
misconceptions collide whenever all three drawn sides are equal).
"""

from __future__ import annotations

from ..core import (
    check,
    check_combine,
    check_conversion,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    numeric_answer,
    pairwise_distinct,
    phrase,
    pick_distractors,
    stated_precision_tolerance,
)

_MAX_ATTEMPTS = 50
_DP = 0  # every answer in this module is an exact whole number

# ---------------------------------------------------------------------------
# Shared metric facts (module-level so both topics' independent checks pivot
# through the SAME literal constants, never re-declared per function).
# ---------------------------------------------------------------------------

_CM3_PER_M3 = 1_000_000  # 1 m^3 = 1,000,000 cm^3 (the linear ladder, cubed: 100^3)
_ML_PER_CM3 = 1  # the base fact: 1 cm^3 = 1 mL, exactly
_ML_PER_KL = 1_000_000  # the ordinary metric ladder: 1 kL = 1,000,000 mL


def _m3_to_L(volume_m3: int) -> int:
    """Skill 1's ONLY route to an m^3 -> L figure: derive it, every time,
    from the cm^3 = mL fact -- 1 m^3 = 1,000,000 cm^3 = 1,000,000 mL = 1000
    L -- never a bare ``x1000`` lookup constant. Used by BOTH construction
    and (via a second, independent call in each ``_build_*`` function's own
    ``check_equivalent``) the in-generator verification."""
    volume_cm3 = volume_m3 * _CM3_PER_M3
    volume_mL = volume_cm3 * _ML_PER_CM3
    check(volume_mL % 1000 == 0, "_m3_to_L: mL total is not an exact multiple of 1000")
    return volume_mL // 1000


def _fmt(value: int) -> str:
    return str(int(value))


def _distractor_texts(*values: int) -> list[str]:
    return [_fmt(v) for v in values]


# The key's rank among the sorted options must not be constant, and every distractor is a named
# misconception. Each builder hands ``_ranked_options`` a pool of wrong values BELOW the key and a
# pool ABOVE it; the helper draws the number of below-key options uniformly from the ranks that are
# feasible for the tier, so the key sits at a varying position (the review measured rank 2 in
# 200/200, or 'largest' in 200/200, for six of the eight tier sets).


def _ranked_options(rng, answer: int, below, above, ranks) -> list[int]:
    """Three wrong values: ``r`` from ``below`` (all < answer) and ``3 - r``
    from ``above`` (all > answer), with ``r`` drawn uniformly from the
    ``ranks`` the (deduplicated) pools can actually supply."""
    low: list[int] = []
    for v in below:
        v = int(v)
        if 0 < v < answer and v not in low:
            low.append(v)
    high: list[int] = []
    for v in above:
        v = int(v)
        if v > answer and v not in high and v not in low:
            high.append(v)
    feasible = [r for r in ranks if r <= len(low) and 3 - r <= len(high)]
    check(bool(feasible), "ranked options: pools cannot supply three distractors")
    r = rng.choice(feasible)
    chosen = rng.sample(low, r) + rng.sample(high, 3 - r)
    check_pairwise_distinct(answer, *chosen, msg="ranked options collide")
    return chosen


def _pack(text: str, answer: int, working: list[str], options: list[int]) -> dict:
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=numeric_answer(answer),
        answer_text=_fmt(answer),
        answer_latex=_fmt(answer),
        working=working,
        distractors=pick_distractors(
            _fmt(answer),
            _distractor_texts(*options),
            tolerance=stated_precision_tolerance(_DP),
        ),
    )


def _cap(container: str) -> str:
    return container[0].upper() + container[1:]


# Liquid containers only (a block, crate, ingot mould or lunchbox is not one),
# each with the range of capacities the named object can really have.
# {name: (min, max)} in LITRES for the small (cm^3) contexts...
_SMALL_LIQUID_CONTAINERS = {
    "a jug": (1, 3),
    "a kettle": (1, 2),
    "a bucket": (5, 15),
    "a watering can": (5, 12),
    "a drink dispenser": (5, 20),
    "a fish tank": (20, 120),
}
# ...and in CUBIC METRES for the large (m^3) contexts.
_LARGE_LIQUID_CONTAINERS_M3 = {
    "a water tank": (2, 30),
    "a rainwater tank": (2, 25),
    "a garden pond": (2, 20),
    "a swimming pool": (25, 90),
    "an irrigation tank": (5, 50),
    "an aquarium display tank": (20, 200),
}


# ===========================================================================
# Skill 1: senior-understand-the-relationship-between-volume-and-capacity
# GENERAL relationship, mL/L targets ONLY, forward (volume -> capacity) ONLY.
# ===========================================================================

_VC1_EASY_ML_TEMPLATES = (
    "{Container} has a volume of {v} cm³. What is its capacity in millilitres (mL)?",
    "{Container} holds {v} cm³ of liquid. What is its capacity in millilitres (mL)?",
)
_VC1_EASY_L_TEMPLATES = (
    "{Container} has a volume of {v} cm³. What is its capacity in litres (L)?",
    "{Container} holds {v} cm³ of liquid. What is its capacity in litres (L)?",
)
_VC1_MEDIUM_TEMPLATES = (
    "{Container} has a volume of {v} m³. What is its capacity in litres (L)?",
    "{Container} holds {v} m³ of water. What is its capacity in litres (L)?",
)
_VC1_HARD_TEMPLATES = (
    "A rectangular {noun} measures {a} cm by {b} cm by {c} cm. What is its capacity in millilitres (mL)?",
    "A rectangular {noun} measures {a} cm by {b} cm by {c} cm; find its capacity in millilitres (mL).",
)
_VC1_EXAM_TEMPLATES = (
    "{Container} measures {l} m by {w} m by {h} m. What is its capacity in litres (L)?",
    "{Container} is {l} m long, {w} m wide and {h} m deep. What is its capacity in litres (L)?",
)

# hard: (dimension range in cm, capacity band in litres)
_VC1_HARD_NOUNS = {
    "fish tank": ((25, 100), (20, 200)),
    "aquarium": ((30, 100), (30, 200)),
    "plastic tub": ((20, 70), (10, 60)),
}
# exam: (l range, w range, h range, volume band in m^3)
_VC1_EXAM_CONTEXTS = {
    "a swimming pool": ((8, 25), (4, 12), (1, 3), (25, 150)),
    "a rectangular water tank": ((2, 6), (1, 4), (1, 3), (2, 40)),
    "a garden pond": ((2, 8), (2, 5), (1, 2), (2, 20)),
    "an aquarium display tank": ((4, 15), (2, 6), (2, 4), (20, 200)),
}


def _build_vc1_easy(rng) -> dict:
    """easy: cm3 -> capacity for a container that really holds that much.
    38% of draws ask for LITRES (the real factor, 1000 cm3 = 1 L, so the key
    is never the prompt's own number and is the smallest option), 62% ask for
    mL (the 1 cm3 = 1 mL kernel, with the litre-number and divide-by-100
    slips BELOW the key and the x10/x100/x1000 slips above), so the key's rank
    among the sorted options varies (1, 2 or 3 in the mL draws)."""
    container = rng.choice(list(_SMALL_LIQUID_CONTAINERS))
    lo, hi = _SMALL_LIQUID_CONTAINERS[container]
    litres = rng.randint(lo, hi)
    volume_cm3 = litres * 1000

    if rng.random() < 0.38:
        answer = litres
        check_conversion(volume_cm3, "cm3", "L", answer, {"cm3": 1, "L": 1000}, label="vc1_easy_L")
        options = _ranked_options(
            rng,
            answer,
            [],
            [volume_cm3, litres * 100, litres * 10],
            (0,),
        )
        text = phrase(rng, _VC1_EASY_L_TEMPLATES, Container=_cap(container), v=volume_cm3)
        working = [f"1000 cm³ = 1 L, so {volume_cm3} cm³ = {volume_cm3} ÷ 1000 = {answer} L."]
    else:
        answer = volume_cm3
        check_conversion(volume_cm3, "cm3", "mL", answer, {"cm3": 1, "mL": 1}, label="vc1_easy_mL")
        check_equivalent(answer, volume_cm3 * _ML_PER_CM3, "vc1_easy independent recompute failed")
        options = _ranked_options(
            rng,
            answer,
            [volume_cm3 // 1000, volume_cm3 // 100, volume_cm3 // 10],
            [volume_cm3 * 10, volume_cm3 * 100, volume_cm3 * 1000],
            (1, 2, 3),
        )
        text = phrase(rng, _VC1_EASY_ML_TEMPLATES, Container=_cap(container), v=volume_cm3)
        working = [f"1 cm³ = 1 mL, so {volume_cm3} cm³ = {answer} mL."]
    return _pack(text, answer, working, options)


def _build_vc1_medium(rng) -> dict:
    """medium: m3 -> L, whole-number volume of a liquid container -- the
    decision this tier adds is knowing/deriving the x1000 factor, which is
    NEVER stated in the prompt (derived in-generator via the cm3=mL chain, see
    _m3_to_L). The misconceptions include 1 m3 = 1 L / 10 L / 100 L (below the
    key) and the cm3-style x10^4 / x10^6 slips (above it)."""
    container = rng.choice(list(_LARGE_LIQUID_CONTAINERS_M3))
    lo, hi = _LARGE_LIQUID_CONTAINERS_M3[container]
    volume_m3 = rng.randint(lo, hi)

    answer = _m3_to_L(volume_m3)
    check_equivalent(answer, volume_m3 * 1000, "vc1_medium independent recompute failed")
    options = _ranked_options(
        rng,
        answer,
        [volume_m3, volume_m3 * 10, volume_m3 * 100],
        [volume_m3 * 10_000, volume_m3 * _CM3_PER_M3],
        (1, 2, 3),
    )
    text = phrase(rng, _VC1_MEDIUM_TEMPLATES, Container=_cap(container), v=volume_m3)
    return _pack(
        text,
        answer,
        [
            f"1 m³ = 1000000 cm³ = 1000000 mL = 1000 L, so {volume_m3} m³ = {answer} L.",
        ],
        options,
    )


def _build_vc1_hard(rng) -> dict:
    """hard: cm3 -> mL where the volume is itself a COMPUTED quantity
    (three given dimensions of a tank/tub) -- the decision this tier adds is
    computing the volume first; the capacity conversion itself stays the
    1:1 easy's mL draws already cover. Misconceptions: summed dimensions,
    a two-dimension product, the surface area, and the x10/x100/x1000 slips;
    the key's rank varies because the pools straddle it."""
    noun = rng.choice(list(_VC1_HARD_NOUNS))
    (dlo, dhi), (blo, bhi) = _VC1_HARD_NOUNS[noun]
    for _ in range(200):
        a = rng.randint(dlo, dhi)
        b = rng.randint(dlo, dhi)
        c = rng.randint(dlo, dhi)
        volume_cm3 = a * b * c
        if blo * 1000 <= volume_cm3 <= bhi * 1000 and len({a, b, c}) > 1:
            break
    else:
        check(False, "vc1_hard: no dimensions inside the container's capacity band")

    answer = volume_cm3
    check_equivalent(answer, (a * b * c) * _ML_PER_CM3, "vc1_hard independent recompute failed")
    two_dims = rng.choice((a * b, b * c, a * c))
    surface_area = 2 * (a * b + b * c + a * c)
    options = _ranked_options(
        rng,
        answer,
        [a + b + c, two_dims, surface_area],
        [volume_cm3 * 10, volume_cm3 * 100, volume_cm3 * 1000, surface_area],
        (0, 1, 2, 3),
    )
    text = phrase(rng, _VC1_HARD_TEMPLATES, noun=noun, a=a, b=b, c=c)
    return _pack(
        text,
        answer,
        [
            f"Volume = {a} x {b} x {c} = {volume_cm3} cm³; 1 cm³ = 1 mL, so capacity = {answer} mL.",
        ],
        options,
    )


def _build_vc1_exam(rng) -> dict:
    """exam: m3 -> L where the volume is a computed quantity (three metre
    dimensions of a liquid container of realistic size: a pool is 25 m3+ and
    never a cube). Combines medium's x1000 derivation with hard's
    compute-the-volume-first step."""
    container = rng.choice(list(_VC1_EXAM_CONTEXTS))
    (llo, lhi), (wlo, whi), (hlo, hhi), (vlo, vhi) = _VC1_EXAM_CONTEXTS[container]
    for _ in range(500):
        l = rng.randint(llo, lhi)
        w = rng.randint(wlo, whi)
        h = rng.randint(hlo, hhi)
        volume_m3 = l * w * h
        if vlo <= volume_m3 <= vhi and len({l, w, h}) > 1:
            break
    else:
        check(False, "vc1_exam: no dimensions inside the container's volume band")

    answer = _m3_to_L(volume_m3)
    check_equivalent(answer, (l * w * h) * 1000, "vc1_exam independent recompute failed")
    options = _ranked_options(
        rng,
        answer,
        [volume_m3, volume_m3 * 100, l * w * 1000, (l + w + h) * 1000],
        [volume_m3 * 10_000, volume_m3 * _CM3_PER_M3, (l + w + h) * 1000],
        (0, 1, 2, 3),
    )
    text = phrase(rng, _VC1_EXAM_TEMPLATES, Container=_cap(container), l=l, w=w, h=h)
    return _pack(
        text,
        answer,
        [
            f"Volume = {l} × {w} × {h} = {volume_m3} m³; 1 m³ = 1000000 cm³ = 1000000 mL = 1000 L, "
            f"so capacity = {answer} L.",
        ],
        options,
    )


@generator(
    "p19h_volcap_general_relationship",
    level=19,
    presets={
        "easy": dict(mode="vc_general_easy"),
        "medium": dict(mode="vc_general_medium"),
        "hard": dict(mode="vc_general_hard"),
        "exam": dict(mode="vc_general_exam"),
    },
)
def p19h_volcap_general_relationship(rng, mode="vc_general_easy"):
    if mode == "vc_general_easy":
        return _build_vc1_easy(rng)
    if mode == "vc_general_medium":
        return _build_vc1_medium(rng)
    if mode == "vc_general_hard":
        return _build_vc1_hard(rng)
    if mode == "vc_general_exam":
        return _build_vc1_exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# ===========================================================================
# Skill 2: senior-recognise-relations-between-volume-and-capacity-...
# NAMED equivalences (1 cm3 = 1 mL, 1 m3 = 1 kL), mL/kL targets ONLY,
# EITHER direction.
# ===========================================================================

# m^3 contexts for skill 2 (volume band in m^3 == kL).
_VC2_TANK_CONTEXTS_M3 = {
    "a water tank": (2, 30),
    "a rainwater tank": (2, 25),
    "a header tank": (1, 8),
    "a swimming pool": (25, 90),
    "a fuel storage tank": (20, 150),
}
# medium: total (kL) the named container ends up holding.
_VC2_MEDIUM_TOTALS = {
    "a water tank": (6, 40),
    "a rainwater tank": (4, 30),
    "a header tank": (2, 12),
    "a swimming pool": (30, 90),
    "a town water reservoir": (100, 400),
}
# hard: tanks with a volume in kL / m^3.
_VC2_HARD_TANKS = {
    "a water tank": (2, 30),
    "a rainwater tank": (2, 25),
    "a header tank": (1, 8),
}

_VC2_EASY_ML_TEMPLATES = (
    "{Container} has a volume of {v} cm³. Using the fact that 1 cm³ = 1 mL, what is its capacity in millilitres (mL)?",
    "{Container} holds {v} cm³ of liquid. Using the fact that 1 cm³ = 1 mL, what is its capacity in millilitres (mL)?",
)
_VC2_EASY_KL_TEMPLATES = (
    "{Container} has a volume of {v} m³. Using the fact that 1 m³ = 1 kL, what is its capacity in kilolitres (kL)?",
    "{Container} holds {v} m³ of water. Using the fact that 1 m³ = 1 kL, what is its capacity in kilolitres (kL)?",
)
_VC2_MEDIUM_FORWARD_TEMPLATES = (
    "{Container} already holds {v1} m³ of water. A further {v2} kL is pumped in. "
    "Using the fact that 1 m³ = 1 kL, what is its capacity in kilolitres (kL)?",
    "{Container} already has a volume of {v1} m³. Another {v2} kL is added. "
    "Using the fact that 1 m³ = 1 kL, what is its capacity in kilolitres (kL)?",
)
_VC2_MEDIUM_REVERSE_TEMPLATES = (
    "{Container} already has a capacity of {v1} kL. A further {v2} m³ is poured in. "
    "Using the fact that 1 m³ = 1 kL, what is its volume in cubic metres (m³)?",
    "{Container} already holds {v1} kL of water. Another {v2} m³ is added. "
    "Using the fact that 1 m³ = 1 kL, what is its volume in cubic metres (m³)?",
)
_VC2_HARD_FORWARD_TEMPLATES = (
    "{Container} holds {v} cm³ of liquid. Using the facts that 1 cm³ = 1 mL and "
    "1 m³ = 1 kL, what is its capacity in kilolitres (kL)?",
)
_VC2_HARD_REVERSE_TEMPLATES = (
    "{Container} has a capacity of {v} kL. Using the facts that 1 m³ = 1 kL and "
    "1 cm³ = 1 mL, what is its volume in cubic centimetres (cm³)?",
)
_VC2_HARD_TO_ML_TEMPLATES = (
    "{Container} holds {v} m³ of water. Using the facts that 1 cm³ = 1 mL and "
    "1 m³ = 1 kL, what is its capacity in millilitres (mL)?",
)
_VC2_EXAM_TEMPLATES_ML = (
    "Container A holds {a} cm³ of liquid. Container B holds {b} m³ of liquid. "
    "Using the facts that 1 cm³ = 1 mL and 1 m³ = 1 kL, what is their combined "
    "capacity in millilitres (mL)?",
    "Container A holds {a} cm³ of liquid and Container B holds {b} m³ of "
    "liquid. Given that 1 cm³ = 1 mL and 1 m³ = 1 kL, find their combined "
    "capacity in millilitres (mL).",
)
_VC2_EXAM_TEMPLATES_KL = (
    "Container A holds {a} cm³ of liquid. Container B holds {b} m³ of liquid. "
    "Using the facts that 1 cm³ = 1 mL and 1 m³ = 1 kL, what is their combined "
    "capacity in kilolitres (kL)?",
    "Container A holds {a} cm³ of liquid and Container B holds {b} m³ of "
    "liquid. Given that 1 cm³ = 1 mL and 1 m³ = 1 kL, find their combined "
    "capacity in kilolitres (kL).",
)


def _build_vc2_easy(rng) -> dict:
    """easy: ONE named equivalence applied once, the fact stated as a given
    (the kernel: 1 cm3 = 1 mL, 1 m3 = 1 kL -- the key may equal the prompt's
    number here). 62% cm3 -> mL, 38% m3 -> kL. Distractors are named slips:
    the litre-number (key/1000) and divide-by-100 slips below the cm3 -> mL
    key, and the 1 m3 = 10 / 100 / 1000 kL slips above the m3 -> kL key."""
    if rng.random() < 0.62:
        container = rng.choice(list(_SMALL_LIQUID_CONTAINERS))
        lo, hi = _SMALL_LIQUID_CONTAINERS[container]
        volume_cm3 = rng.randint(lo, hi) * 1000
        answer = volume_cm3
        check_conversion(volume_cm3, "cm3", "mL", answer, {"cm3": 1, "mL": 1}, label="vc2_easy")
        options = _ranked_options(
            rng,
            answer,
            [volume_cm3 // 1000, volume_cm3 // 100, volume_cm3 // 10],
            [volume_cm3 * 10, volume_cm3 * 100, volume_cm3 * 1000],
            (1, 2, 3),
        )
        text = phrase(rng, _VC2_EASY_ML_TEMPLATES, Container=_cap(container), v=volume_cm3)
        working = [f"1 cm³ = 1 mL, so {volume_cm3} cm³ = {answer} mL."]
    else:
        container = rng.choice(list(_VC2_TANK_CONTEXTS_M3))
        lo, hi = _VC2_TANK_CONTEXTS_M3[container]
        volume_m3 = rng.randint(lo, hi)
        answer = volume_m3
        check_conversion(volume_m3, "m3", "kL", answer, {"m3": _ML_PER_KL, "kL": _ML_PER_KL}, label="vc2_easy_kL")
        options = _ranked_options(
            rng,
            answer,
            [],
            [volume_m3 * 10, volume_m3 * 100, volume_m3 * 1000],
            (0,),
        )
        text = phrase(rng, _VC2_EASY_KL_TEMPLATES, Container=_cap(container), v=volume_m3)
        working = [f"1 m³ = 1 kL, so {volume_m3} m³ = {answer} kL."]
    return _pack(text, answer, working, options)


def _build_vc2_medium(rng) -> dict:
    """medium: TWO separately-stated quantities, one in m3 and one in kL,
    combined to a single total. The total is a plausible
    content for the named container. Slips: only one addend, the difference
    (both below the key) and converting ONE addend by a x1000 (m3 read as L)
    before adding, or the whole total x1000 (above the key) -- so the key is
    not always the largest option."""
    container = rng.choice(list(_VC2_MEDIUM_TOTALS))
    lo, hi = _VC2_MEDIUM_TOTALS[container]
    reverse = rng.random() < 0.5
    unit_a, unit_b = ("kL", "m3") if reverse else ("m3", "kL")

    total = rng.randint(lo, hi)
    v1 = rng.randint(1, total - 1)
    v2 = total - v1
    answer = v1 + v2

    # Independent route: pivot BOTH given quantities through mL (the OTHER
    # named fact's unit, 1 m3 = 1 kL = 1,000,000 mL) and sum there.
    values = {unit_a: v1, unit_b: v2}
    check_combine(
        values,
        unit_a,
        unit_b,
        "sum",
        answer,
        {"m3": _ML_PER_KL, "kL": _ML_PER_KL},
        label="vc2_medium",
    )
    options = _ranked_options(
        rng,
        answer,
        [v1, v2, abs(v1 - v2)],
        [1000 * v1 + v2, v1 + 1000 * v2, 1000 * answer],
        (0, 1, 2, 3),
    )
    templates = _VC2_MEDIUM_REVERSE_TEMPLATES if reverse else _VC2_MEDIUM_FORWARD_TEMPLATES
    text = phrase(rng, templates, Container=_cap(container), v1=v1, v2=v2)
    working = (
        f"1 m³ = 1 kL, so a capacity of {v1} kL and a further {v2} m³ poured in give a total volume of "
        f"{v1} m³ + {v2} m³ = {answer} m³."
        if reverse
        else f"1 m³ = 1 kL, so a volume of {v1} m³ and a further {v2} kL pumped in give a total capacity of "
        f"{v1} kL + {v2} kL = {answer} kL."
    )
    return _pack(text, answer, [working], options)


def _build_vc2_hard(rng) -> dict:
    """hard: a chained conversion crossing BOTH pairs, three forms -- cm3 ->
    kL (25%), kL -> cm3 (37.5%) and m3 -> mL (37.5%). Both named facts are
    given; the decision this tier adds is the ordinary x/÷ 1,000,000 metric
    ladder step (never a dimension computation). Slips include the cm3<->m3
    linear-factor errors and the mm3 x10^9 confusion, straddling the key so
    its rank varies (cm3 -> kL has no whole-number slip below the key)."""
    container = rng.choice(list(_VC2_HARD_TANKS))
    lo, hi = _VC2_HARD_TANKS[container]
    k = rng.randint(lo, hi)  # kL == m3, given fact
    form = rng.random()

    if form < 0.25:
        cm3_value = k * _CM3_PER_M3
        mL_value = cm3_value  # 1 cm3 = 1 mL, given
        check(mL_value % _ML_PER_KL == 0, "vc2_hard: mL total not an exact multiple of 1,000,000")
        answer = mL_value // _ML_PER_KL  # ordinary ladder: 1 kL = 1,000,000 mL
        # Independent route: cm3 -> m3 (linear-cubed ladder) -> kL (given fact).
        check_equivalent(answer, cm3_value // _CM3_PER_M3, "vc2_hard (cm3 -> kL) independent recompute failed")
        options = _ranked_options(
            rng,
            answer,
            [],
            [mL_value, k * 1000, k * 100, k * 10_000],
            (0,),
        )
        text = phrase(rng, _VC2_HARD_FORWARD_TEMPLATES, Container=_cap(container), v=cm3_value)
        working = [
            f"1 cm³ = 1 mL, so {cm3_value} cm³ = {mL_value} mL; 1 kL = 1000000 mL, so {mL_value} mL = {answer} kL.",
        ]
    elif form < 0.625:
        m3_value = k  # 1 m3 = 1 kL, given
        answer = m3_value * _CM3_PER_M3  # ordinary ladder: 1 m3 = 1,000,000 cm3
        # Independent route: kL -> mL (ordinary ladder) -> cm3 (base fact).
        check_equivalent(answer, (k * _ML_PER_KL) // _ML_PER_CM3, "vc2_hard (kL -> cm3) independent recompute failed")
        options = _ranked_options(
            rng,
            answer,
            [k, k * 100, k * 1000, k * 10_000],
            [k * 10_000_000, k * 1_000_000_000],
            (1, 2, 3),
        )
        text = phrase(rng, _VC2_HARD_REVERSE_TEMPLATES, Container=_cap(container), v=k)
        working = [
            f"1 m³ = 1 kL, so {k} kL = {m3_value} m³; 1 m³ = 1000000 cm³, so {m3_value} m³ = {answer} cm³.",
        ]
    else:
        answer = k * _ML_PER_KL  # 1 m3 = 1 kL = 1,000,000 mL
        # Independent route: m3 -> cm3 (linear-cubed ladder) -> mL (base fact).
        check_equivalent(answer, k * _CM3_PER_M3 * _ML_PER_CM3, "vc2_hard (m3 -> mL) independent recompute failed")
        options = _ranked_options(
            rng,
            answer,
            [k, k * 100, k * 1000, k * 10_000],
            [k * 10_000_000, k * 1_000_000_000],
            (1, 2, 3),
        )
        text = phrase(rng, _VC2_HARD_TO_ML_TEMPLATES, Container=_cap(container), v=k)
        working = [
            f"1 m³ = 1 kL, and 1 kL = 1000000 mL (1 cm³ = 1 mL, 1 m³ = 1000000 cm³), so {k} m³ = {answer} mL.",
        ]
    return _pack(text, answer, working, options)


def _build_vc2_exam(rng) -> dict:
    """exam: TWO containers, one in each pair (one cm3, one m3), summed to a
    single total capacity in a STATED target unit (mL or kL) -- the decision
    this tier adds over hard is aggregation across items. Slips straddle the
    key (one container only and the raw-numbers sum below it; unit swaps and
    x1000 slips above it), so its rank varies."""
    target = rng.choice(("mL", "kL"))
    if target == "mL":
        a = rng.randint(50, 3000)  # container A: cm3, contributes a mL directly (1:1)
        b = rng.randint(1, 50)  # container B: m3, contributes b*1,000,000 mL
        contribution_a = a
        contribution_b = b * _ML_PER_KL
        answer = contribution_a + contribution_b
        check_equivalent(answer, a * _ML_PER_CM3 + b * _CM3_PER_M3, "vc2_exam (mL) independent recompute failed")
        options = _ranked_options(
            rng,
            answer,
            [contribution_a, contribution_b, a + b],
            [b + a * _CM3_PER_M3, (a + b) * _ML_PER_KL, a * 1000 + contribution_b],
            (0, 1, 2, 3),
        )
        text = phrase(rng, _VC2_EXAM_TEMPLATES_ML, a=a, b=b)
        working = [
            f"Container A: 1 cm³ = 1 mL, so {a} cm³ = {contribution_a} mL. "
            f"Container B: 1 m³ = 1 kL = 1000000 mL, so {b} m³ = {contribution_b} mL. "
            f"Total = {answer} mL.",
        ]
    else:
        k = rng.randint(1, 25)
        a = k * _CM3_PER_M3  # container A: cm3, a clean multiple of a million
        b = rng.randint(1, 60)  # container B: m3, contributes b kL directly (1:1)
        check(a % _ML_PER_KL == 0, "vc2_exam: container A's mL total is not an exact multiple of 1,000,000")
        contribution_a = a // _ML_PER_KL
        contribution_b = b
        answer = contribution_a + contribution_b
        # Independent route for container A: cm3 -> m3 (linear-cubed ladder)
        # -> kL (the given fact 1 m3 = 1 kL).
        check_equivalent(answer, a // _CM3_PER_M3 + b, "vc2_exam (kL) independent recompute failed")
        options = _ranked_options(
            rng,
            answer,
            [contribution_a, contribution_b, abs(contribution_a - contribution_b)],
            [a + b, 1000 * contribution_a + b, contribution_a + 1000 * b, 1000 * answer],
            (0, 1, 2, 3),
        )
        text = phrase(rng, _VC2_EXAM_TEMPLATES_KL, a=a, b=b)
        working = [
            f"Container A: 1 cm³ = 1 mL and 1 kL = 1,000,000 mL, so {a} cm³ = {contribution_a} kL. "
            f"Container B: 1 m³ = 1 kL, so {b} m³ = {contribution_b} kL. Total = {answer} kL.",
        ]
    return _pack(text, answer, working, options)


@generator(
    "p19h_volcap_named_equivalence",
    level=19,
    presets={
        "easy": dict(mode="vc_named_easy"),
        "medium": dict(mode="vc_named_medium"),
        "hard": dict(mode="vc_named_hard"),
        "exam": dict(mode="vc_named_exam"),
    },
)
def p19h_volcap_named_equivalence(rng, mode="vc_named_easy"):
    if mode == "vc_named_easy":
        return _build_vc2_easy(rng)
    if mode == "vc_named_medium":
        return _build_vc2_medium(rng)
    if mode == "vc_named_hard":
        return _build_vc2_hard(rng)
    if mode == "vc_named_exam":
        return _build_vc2_exam(rng)
    raise ValueError(f"unknown mode: {mode!r}")
