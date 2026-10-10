"""Generates `rectangular_frustum_volume` questions (skill `rectangular-frustum-volume`): the volume of a
rectangular frustum.

A rectangular frustum is a pyramid with its apex sliced off by a plane
parallel to the base. For the cut to actually be a pyramid frustum (single
apex, straight lateral edges) the top rectangle must be a UNIFORM linear
scale-down of the bottom rectangle: ``l2/l1 == w2/w1`` (the same ratio in
both directions) -- a top rectangle with a different aspect ratio from the
base is a different solid (an "obelisk"), not a frustum, and the closed-form
volume formula below would not be its true volume.

Parameterisation (the ratio constraint holds BY CONSTRUCTION, never by a
rejection loop): pick a scale ratio in lowest terms ``p/q``
(``2 <= q <= 6``, ``1 <= p <= q-1``, ``gcd(p, q) == 1``, EXCLUDING
``(p, q) == (2, 3)`` -- see the pairwise distractor collision noted below --
10 coprime pairs remain), and base-unit multipliers ``a, b``. Then::

    l1, w1 = q*a, q*b        # bottom
    l2, w2 = p*a, p*b        # top -- automatically l2 < l1, w2 < w1,
                              # l2/l1 == w2/w1 == p/q exactly (cross
                              # multiplication, never a float division)
    A1, A2 = l1*w1, l2*w2 = q**2*a*b, p**2*a*b
    G = p*q*a*b              # == sqrt(A1*A2) EXACTLY, since
                              # A1*A2 == (p*q*a*b)**2 by construction --
                              # verified with math.isqrt, not just algebra.

The removed-apex derivation (this is what ``working`` shows, every tier):
extend the four lateral edges until they meet at the pyramid's original
apex. By similar triangles, ``H_small / H_total == p/q`` (``H_small`` =
apex-to-top-plane distance, ``H_total`` = apex-to-base distance), and
``H_total = H_small + h`` (``h`` = frustum height). Solving,
``H_total = h*q/(q-p)``, ``H_small = h*p/(q-p)``. Choosing
``h = (q-p) * 6 * k`` (a positive integer ``k``) on the three exact tiers
makes ``H_total = 6*k*q`` and ``H_small = 6*k*p`` land on exact integers,
which is what keeps every distractor below (not just the answer) an exact
integer too. On the exam tier the ``*6`` is dropped (``h = (q-p) * k``), so
the volume is generically not a clean integer.

``V = (1/3)*A1*H_total - (1/3)*A2*H_small`` is the primary construction
(two pyramid volumes, subtracted -- shown in ``working``). It is
independently re-checked via the closed polynomial form
``V_closed = (h/3) * (A1 + A2 + G)`` with ``check_equivalent`` -- a
different route, mirroring ``t5_annulus_frustum_build.py::
_frustum_raw_and_checks``'s "big cone minus small cone" check against its
own closed form (opposite primary/check roles: here apex-removal is
primary because it is required to appear in ``working``).

Tiers (genuine reasoning escalation, never a wider number range):

- **easy** (``mode="direct"``): bottom, top and height all given directly
  as numbers. No derivation before applying the volume relation.
- **medium** (``mode="scale_factor"``): bottom and height given directly;
  the top is described only by the ``p/q`` ratio, so the learner derives
  ``l2 = l1*p/q``, ``w2 = w1*p/q`` first.
- **hard** (``mode="apex_height"``): bottom and top given directly (like
  easy); instead of the frustum height ``h``, the TOTAL height to the
  pyramid's apex (``H_total``) is given, so the learner derives
  ``H_small = H_total * (l2/l1)`` then ``h = H_total - H_small`` via
  similar triangles before applying the volume relation.
- **exam** (``mode="apex_height_decimal"``): bottom given directly; the
  top is described only by the ratio (as medium) AND the height only via
  the apex height (as hard) -- both derivations stacked. ``h = (q-p)*k``
  (the ``*6`` dropped), so ``V = k*a*b*(q**3-p**3)/3`` is generically not
  an integer -- BUT this alone does not reliably make it non-integer:
  ``q**3 - p**3 ≡ q - p (mod 3)`` (``x**3 ≡ x (mod 3)`` for every integer
  ``x``), so ``V`` is already an exact integer whenever ``3 | (q-p)``,
  regardless of ``k, a, b``. The ratio pool used on this tier
  (``_EXAM_RATIO_POOL``) excludes every pair with ``3 | (q-p)`` by
  construction (8 of the 10 pairs remain), and the draw resamples ``a, b,
  k`` (mirroring ``p7e_pyramid_sphere_volume_build.py``'s "dimension pool
  excludes multiples of 6" idiom) until ``(k*a*b) % 3 != 0`` too -- 3 is
  prime, so once neither ``(q-p)`` nor ``k*a*b`` is a multiple of 3, their
  product is not either, and ``V`` is always exactly ``...333`` or
  ``...667`` repeating, never a rounding-boundary risk. Rounded to 2 d.p.
  (``stated_precision_tolerance(2)``).

Distractors (misconceptions), every tier, checked by construction:

- ``omits-apex-removal``: uses the WHOLE big pyramid,
  ``(1/3) * A1 * H_total`` -- always strictly greater than ``V``.
- ``treats-as-prism``: ``A1 * h`` (ignores the taper) -- always strictly
  greater than ``V``.
- ``uses-mean-area-slab``: ``h * (A1 + A2) / 2`` -- the AM of the two face
  areas times height; ``AM(A1, A2) > GM(A1, A2) == sqrt(A1*A2) == G``
  strictly whenever ``A1 != A2`` (true here since ``p < q``), so this is
  always strictly greater than ``V`` too, but by a much smaller margin.

A real pairwise collision, found by re-deriving the three distractor values: ``prism
== omits-apex-removal`` exactly when ``H_total == 3h``, i.e.
``q/(q-p) == 3``, i.e. ``p/q == 2/3``. At ``q=3, p=2`` both distractors
equal ``54*k*a*b``, silently dropping one of the three misconception
options (``pick_distractors`` dedupes by value). ``(p, q) == (2, 3)`` is
excluded from the ratio pool for exactly this reason, and every seed
additionally re-checks the three distractor values pairwise distinct with
``check_pairwise_distinct`` -- so a future range change that reopens the
collision fails loudly rather than shipping a two-option question.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

#: Coprime scale-ratio pool p/q, 2 <= q <= 6, 1 <= p <= q-1, gcd(p,q) == 1,
#: excluding (p, q) == (2, 3) -- the prism/omits-apex-removal pairwise
#: collision (module docstring). 10 pairs.
_RATIO_POOL: tuple[tuple[int, int], ...] = tuple(
    (p, q) for q in range(2, 7) for p in range(1, q) if math.gcd(p, q) == 1 and (p, q) != (2, 3)
)

#: Exam-tier ratio pool: `_RATIO_POOL` further excludes every pair with
#: `3 | (q - p)`, since q**3 - p**3 ≡ q - p (mod 3) would otherwise make V
#: an exact integer regardless of k, a, b (module docstring). 8 pairs.
_EXAM_RATIO_POOL: tuple[tuple[int, int], ...] = tuple((p, q) for (p, q) in _RATIO_POOL if (q - p) % 3 != 0)

_EXAM_RESAMPLE_LIMIT = 200


def _geometry(p: int, q: int, a: int, b: int, mult: int) -> dict:
    """Every geometric quantity a (p, q, a, b, mult) draw determines, by
    construction: the top-rectangle/bottom-rectangle ratio is built into the
    draw (l2 = p*a, w2 = p*b against l1 = q*a, w1 = q*b), never
    checked-and-rejected after the fact."""
    l1, w1 = q * a, q * b
    l2, w2 = p * a, p * b
    h = (q - p) * mult
    H_total = q * mult
    H_small = p * mult

    A1 = l1 * w1
    A2 = l2 * w2
    G_sq = A1 * A2
    G_root = math.isqrt(G_sq)
    check(G_root * G_root == G_sq, "rectangular_frustum_volume: A1*A2 is not a perfect square")
    G = p * q * a * b
    check(G == G_root, "rectangular_frustum_volume: p*q*a*b != sqrt(A1*A2)")

    vol_big = Fraction(A1 * H_total, 3)
    vol_small = Fraction(A2 * H_small, 3)
    V = vol_big - vol_small

    return dict(
        p=p,
        q=q,
        a=a,
        b=b,
        l1=l1,
        w1=w1,
        l2=l2,
        w2=w2,
        h=h,
        H_total=H_total,
        H_small=H_small,
        A1=A1,
        A2=A2,
        G=G,
        vol_big=vol_big,
        vol_small=vol_small,
        V=V,
    )


def _construct(rng, *, exam: bool) -> dict:
    """Draw (p, q, a, b, mult) and compute every geometric quantity the
    tiers share. On the exam tier this is a genuine resample loop (mirroring
    `_build_sphere_decimal`'s idiom): the ratio side of the mod-3 guard is
    already excluded by construction (`_EXAM_RATIO_POOL`), but `(k*a*b) % 3`
    and a rounding-boundary check on V itself (cheap insurance -- provably
    unreachable here since every exam-tier V lands on a repeating .333/.667
    decimal, never a .xx5 tie, but kept for the same reason
    `p7e_pyramid_sphere_volume_build.py::_build_sphere_decimal` keeps its own
    `rounds_unambiguously` guard: a consistent idiom costs nothing) both
    resample rather than reject the question outright."""
    if exam:
        for _ in range(_EXAM_RESAMPLE_LIMIT):
            p, q = rng.choice(_EXAM_RATIO_POOL)
            a = rng.randint(2, 9)
            b = rng.randint(2, 9)
            k = rng.randint(1, 6)
            if (k * a * b) % 3 == 0:
                continue
            vals = _geometry(p, q, a, b, k)
            if not rounds_unambiguously(float(vals["V"]), 2):
                continue
            break
        else:
            check(False, "rectangular_frustum_volume/exam: no draw cleared the mod-3/rounding guard")
    else:
        p, q = rng.choice(_RATIO_POOL)
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        k = rng.randint(1, 5)
        vals = _geometry(p, q, a, b, 6 * k)

    l1, w1, l2, w2, h = vals["l1"], vals["w1"], vals["l2"], vals["w2"], vals["h"]
    check(l2 < l1, "rectangular_frustum_volume: top length not strictly less than base")
    check(w2 < w1, "rectangular_frustum_volume: top width not strictly less than base")
    check(l2 * w1 == l1 * w2, "rectangular_frustum_volume: top/bottom scale ratio not uniform")
    check(h > 0, "rectangular_frustum_volume: height not strictly positive")

    V_closed = Fraction(h, 3) * (vals["A1"] + vals["A2"] + vals["G"])
    check_equivalent(
        vals["V"],
        V_closed,
        "rectangular frustum: apex-removal and closed form disagree",
    )

    if not exam:
        check(vals["V"].denominator == 1, "rectangular frustum: exact-tier volume is not an integer")
        check(vals["vol_big"].denominator == 1, "rectangular frustum: big-pyramid volume is not an integer")
        check(vals["vol_small"].denominator == 1, "rectangular frustum: small-pyramid volume is not an integer")

    return vals


def _round_for_tier(value: Fraction, exam: bool) -> Fraction:
    """Round to the tier's own stated precision: exact integer at the exact
    tiers (raises if the value is somehow not exact -- a real invariant,
    never a display convenience), 2 d.p. half-up at the exam tier."""
    if exam:
        return round_half_up(value, 2)
    check(value.denominator == 1, "rectangular frustum: exact-tier value is not an integer")
    return value


def _display(value: Fraction, exam: bool) -> str:
    """The tier-appropriate rendering of an already-computed quantity: a
    plain digit string at the exact tiers (`_typed_number_tolerance` grades
    a no-decimal-point `answer_text` with an EXACT match, so no trailing
    `.0`), a fixed-2dp string at the exam tier."""
    rounded = _round_for_tier(value, exam)
    if exam:
        return stated_precision_string(rounded, 2)
    return str(int(rounded))


def _answer_and_distractors(vals: dict, exam: bool) -> tuple[object, str, list[str]]:
    """The canonical (answer, answer_text) pair plus the three misconception
    distractors, every one checked distinct from the answer AND pairwise
    distinct from each other (the (2, 3)-ratio collision guard)."""
    A1, A2, h = vals["A1"], vals["A2"], vals["h"]
    H_total, V = vals["H_total"], vals["V"]

    omits_apex = Fraction(A1 * H_total, 3)  # == vol_big: the whole big pyramid, apex not removed
    treats_as_prism = Fraction(A1 * h)
    uses_mean_area_slab = Fraction(h * (A1 + A2), 2)

    check_distinct(
        V,
        omits_apex,
        treats_as_prism,
        uses_mean_area_slab,
        msg="rectangular frustum: a misconception distractor equals the true volume",
    )
    check_pairwise_distinct(
        omits_apex,
        treats_as_prism,
        uses_mean_area_slab,
        msg="rectangular frustum: two misconception distractors collide with each other",
    )
    check(omits_apex > V, "rectangular frustum: omits-apex-removal must exceed the true volume")
    check(treats_as_prism > V, "rectangular frustum: treats-as-prism must exceed the true volume")
    check(uses_mean_area_slab > V, "rectangular frustum: uses-mean-area-slab must exceed the true volume")

    rounded_V = _round_for_tier(V, exam)
    answer_value = numeric_answer(rounded_V)
    answer_text = _display(V, exam)

    candidates = [
        _display(omits_apex, exam),
        _display(treats_as_prism, exam),
        _display(uses_mean_area_slab, exam),
    ]
    tolerance = stated_precision_tolerance(2) if exam else 0
    distractors = pick_distractors(answer_text, candidates, tolerance=tolerance)

    return answer_value, answer_text, distractors


def _derivation_working(vals: dict, exam: bool) -> list[str]:
    """The removed-apex narration every tier shares, REGARDLESS of what the
    question gave the learner directly: the base/top areas, the similarity
    ratio that recovers H_total/H_small (with the actual numbers
    substituted, not the closed form applied blind), the two pyramid
    volumes, and the subtraction giving V. On the hard/exam tiers this
    restates H_total/H_small from the mode-specific derivation already
    shown (there they were derived the other way round, from a GIVEN
    H_total down to h) -- recomputing them here from h and the ratio is a
    genuine second, independent-looking route through the same numbers,
    the same "state every intermediate" discipline as every other tier."""
    l1, w1, l2, w2 = vals["l1"], vals["w1"], vals["l2"], vals["w2"]
    p, q, h = vals["p"], vals["q"], vals["h"]
    A1, A2 = vals["A1"], vals["A2"]
    H_total, H_small = vals["H_total"], vals["H_small"]
    vol_big_text = _display(vals["vol_big"], exam)
    vol_small_text = _display(vals["vol_small"], exam)
    answer_text = _display(vals["V"], exam)
    return [
        f"Base area A1 = {l1} x {w1} = {A1} cm^2. Top area A2 = {l2} x {w2} = {A2} cm^2.",
        f"The top is a {p}/{q} scale copy of the base (l2/l1 = {l2}/{l1}), so extending the "
        f"four sloped edges to the pyramid's original apex, similar triangles give "
        f"H_small/H_total = {p}/{q}; since H_total = H_small + h, H_total = h x "
        f"{q}/({q} - {p}) = {h} x {q}/{q - p} = {H_total} cm, and H_small = H_total - h = "
        f"{H_total} - {h} = {H_small} cm.",
        f"Volume of the full pyramid = (1/3) x {A1} x {H_total} = {vol_big_text} cm^3.",
        f"Volume of the removed apex pyramid = (1/3) x {A2} x {H_small} = {vol_small_text} cm^3.",
        f"Frustum volume = {vol_big_text} - {vol_small_text} = {answer_text} cm^3.",
    ]


# ---------------------------------------------------------------------------
# Tier builders
# ---------------------------------------------------------------------------


def _build_direct(rng) -> dict:
    """Easy: bottom, top and height all given directly. No derivation
    before applying the removed-apex relation."""
    vals = _construct(rng, exam=False)
    l1, w1, l2, w2, h = vals["l1"], vals["w1"], vals["l2"], vals["w2"], vals["h"]

    text = phrase(
        rng,
        [
            "A frustum-shaped container has a rectangular base measuring {l1} cm by "
            "{w1} cm and a rectangular top measuring {l2} cm by {w2} cm, with a height "
            "of {h} cm between the two faces. Find its volume, in cm^3.",
            "A rectangular frustum has a base {l1} cm by {w1} cm, narrowing to a top "
            "{l2} cm by {w2} cm over a height of {h} cm. What is its volume, in cm^3?",
            "The base of a rectangular frustum measures {l1} cm by {w1} cm and its top "
            "measures {l2} cm by {w2} cm; the height between the two faces is {h} cm. "
            "Calculate the volume, in cm^3.",
            "A storage bin shaped like a rectangular frustum is {h} cm tall, with a "
            "base of {l1} cm by {w1} cm and a top of {l2} cm by {w2} cm. Find its "
            "volume, in cm^3.",
        ],
        l1=l1,
        w1=w1,
        l2=l2,
        w2=w2,
        h=h,
    )

    working = [
        f"Bottom rectangle {l1} cm x {w1} cm, top rectangle {l2} cm x {w2} cm, height h = {h} cm (all given directly).",
        *_derivation_working(vals, exam=False),
    ]
    answer, answer_text, distractors = _answer_and_distractors(vals, exam=False)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
        working=working,
    )


def _build_scale_factor(rng) -> dict:
    """Medium: bottom and height given directly; the top is described only
    by the p/q ratio, so the learner derives l2, w2 first."""
    vals = _construct(rng, exam=False)
    l1, w1, h = vals["l1"], vals["w1"], vals["h"]
    p, q, l2, w2 = vals["p"], vals["q"], vals["l2"], vals["w2"]

    text = phrase(
        rng,
        [
            "A rectangular frustum has a base {l1} cm by {w1} cm and a height of "
            "{h} cm. Each side of the top rectangle is {p}/{q} the length of the "
            "corresponding side of the base. Find its volume, in cm^3.",
            "A frustum of a rectangular pyramid is {h} cm tall, with a base of "
            "{l1} cm by {w1} cm. The top rectangle is scaled down from the base by "
            "a factor of {p}/{q} in both directions. What is its volume, in cm^3?",
            "A container shaped like a rectangular frustum narrows from a base of "
            "{l1} cm by {w1} cm to a top whose sides are each {p} out of every {q} "
            "of the base's corresponding side, over a height of {h} cm. Calculate "
            "its volume, in cm^3.",
            "The base of a rectangular frustum measures {l1} cm by {w1} cm; the top "
            "rectangle is similar to the base with a linear scale factor of {p}/{q}, "
            "and the height between the two faces is {h} cm. Find its volume, "
            "in cm^3.",
        ],
        l1=l1,
        w1=w1,
        h=h,
        p=p,
        q=q,
    )

    working = [
        f"Top rectangle sides: l2 = {l1} x {p}/{q} = {l2} cm, w2 = {w1} x {p}/{q} = {w2} cm.",
        *_derivation_working(vals, exam=False),
    ]
    answer, answer_text, distractors = _answer_and_distractors(vals, exam=False)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
        working=working,
    )


def _build_apex_height(rng) -> dict:
    """Hard: bottom and top given directly (like easy); instead of h, the
    TOTAL height to the pyramid's apex is given, so the learner derives
    H_small then h = H_total - H_small via similar triangles first."""
    vals = _construct(rng, exam=False)
    l1, w1, l2, w2 = vals["l1"], vals["w1"], vals["l2"], vals["w2"]
    H_total, H_small, h = vals["H_total"], vals["H_small"], vals["h"]

    text = phrase(
        rng,
        [
            "A rectangular frustum has a base {l1} cm by {w1} cm and a top {l2} cm "
            "by {w2} cm. If the four sloped edges were extended to meet at the "
            "pyramid's apex, the apex would be {H_total} cm above the base. Find "
            "the frustum's volume, in cm^3.",
            "A frustum of a rectangular pyramid has a base measuring {l1} cm by "
            "{w1} cm and a top measuring {l2} cm by {w2} cm. The full pyramid this "
            "frustum was cut from has a total height (base to apex) of {H_total} cm. "
            "What is its volume, in cm^3?",
            "The base of a rectangular frustum is {l1} cm by {w1} cm and its top is "
            "{l2} cm by {w2} cm; extending its sloped sides to the original apex "
            "gives a total height of {H_total} cm. Calculate the volume, in cm^3.",
            "A rectangular frustum's base measures {l1} cm by {w1} cm and its top "
            "measures {l2} cm by {w2} cm. Extended to a full pyramid, the apex sits "
            "{H_total} cm above the base plane. Find its volume, in cm^3.",
        ],
        l1=l1,
        w1=w1,
        l2=l2,
        w2=w2,
        H_total=H_total,
    )

    working = [
        f"By similar triangles, H_small / H_total = l2/l1, so H_small = "
        f"{H_total} x {l2}/{l1} = {H_small} cm, and h = H_total - H_small = "
        f"{H_total} - {H_small} = {h} cm.",
        *_derivation_working(vals, exam=False),
    ]
    answer, answer_text, distractors = _answer_and_distractors(vals, exam=False)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
        working=working,
    )


def _build_apex_height_decimal(rng) -> dict:
    """Exam: bottom given directly; both the top (via the p/q ratio, as
    medium) and the height (via the apex height, as hard) are derived --
    the two derivations stacked. Rounded to 2 d.p."""
    vals = _construct(rng, exam=True)
    l1, w1 = vals["l1"], vals["w1"]
    p, q, l2, w2 = vals["p"], vals["q"], vals["l2"], vals["w2"]
    H_total, H_small, h = vals["H_total"], vals["H_small"], vals["h"]

    text = phrase(
        rng,
        [
            "A rectangular frustum has a base {l1} cm by {w1} cm. Each side of the "
            "top rectangle is {p}/{q} the length of the corresponding side of the "
            "base. Extending the sloped edges to the pyramid's apex gives a total "
            "height of {H_total} cm. Find the frustum's volume, in cm^3, correct to "
            "2 decimal places.",
            "A frustum of a rectangular pyramid is cut from a base measuring {l1} "
            "cm by {w1} cm; the top rectangle is scaled down by a factor of {p}/{q} "
            "in both directions, and the full pyramid's apex is {H_total} cm above "
            "the base. Calculate its volume, in cm^3, correct to 2 decimal places.",
            "A container shaped like a rectangular frustum has a base of {l1} cm by "
            "{w1} cm, with the top's sides each {p} out of every {q} of the base's "
            "corresponding side. The apex of the original pyramid sits {H_total} cm "
            "above the base. What is its volume, in cm^3, correct to 2 decimal "
            "places?",
            "The base of a rectangular frustum measures {l1} cm by {w1} cm and its "
            "top is similar to the base with a linear scale factor of {p}/{q}. "
            "Extended to the full pyramid, the apex is {H_total} cm above the base "
            "plane. Find its volume, in cm^3, correct to 2 decimal places.",
        ],
        l1=l1,
        w1=w1,
        p=p,
        q=q,
        H_total=H_total,
    )

    working = [
        f"Top rectangle sides: l2 = {l1} x {p}/{q} = {l2} cm, w2 = {w1} x {p}/{q} = {w2} cm.",
        f"By similar triangles, H_small / H_total = l2/l1, so H_small = "
        f"{H_total} x {l2}/{l1} = {H_small} cm, and h = H_total - H_small = "
        f"{H_total} - {H_small} = {h} cm.",
        *_derivation_working(vals, exam=True),
    ]
    answer, answer_text, distractors = _answer_and_distractors(vals, exam=True)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
        working=working,
    )


@generator(
    "rectangular_frustum_volume",
    level=12,
    presets={
        "easy": dict(mode="direct"),
        "medium": dict(mode="scale_factor"),
        "hard": dict(mode="apex_height"),
        "exam": dict(mode="apex_height_decimal"),
    },
)
def rectangular_frustum_volume(rng, mode="direct"):
    """Volume of a rectangular frustum via the removed-apex construction
    (two pyramid volumes, subtracted), independently checked against the
    closed polynomial form. See the module docstring for the full geometry,
    tier escalation and distractor design."""
    if mode == "direct":
        return _build_direct(rng)
    if mode == "scale_factor":
        return _build_scale_factor(rng)
    if mode == "apex_height":
        return _build_apex_height(rng)
    if mode == "apex_height_decimal":
        return _build_apex_height_decimal(rng)
    raise ValueError(f"unknown mode: {mode!r}")
