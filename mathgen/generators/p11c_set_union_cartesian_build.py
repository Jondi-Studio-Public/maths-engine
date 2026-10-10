"""Set-union cardinality + Cartesian-form equating.

Two generators, both senior ``acara-senior-v8-4``, domain ``senior-mathematics``:

- ``p11c_set_union_cardinality`` (``senior-determine-and-use-the-formulas-for-
  finding-the``, ACMSM005) -- inclusion-exclusion for the number of elements
  in the union of two and the union of three sets, ``answer: number``. Every
  question is framed with an explicit finite universe (a survey/count word
  problem), never a probability -- the load-bearing separation from the
  shipping ``senior-review-the-rules-p-left-overline-a-right`` skill, which
  grades the structurally identical two-set identity but always as a
  probability in [0, 1] with no universe count anywhere in its own contract.
  easy: two-set union, find the "neither" count via N - |A union B|. medium:
  the two-set inverse -- given N, the two set sizes and the "neither" count,
  find the overlap |A intersect B| (a rearrangement of the two-set identity
  for the subtracted term). hard: three-set union, all seven region counts
  given save one -- the union total is now stated and exactly one of
  A, B, AB, AC, BC, ABC, C is missing (which one varies per draw), and the
  learner rearranges the seven-term identity for whichever term that is --
  a different algebraic move from either lower tier, not "one more set".

- ``p11c_cartesian_form_equating`` (``senior-review-cartesian-form``,
  ACMSM078) -- equating real and imaginary parts of a complex equation to
  solve for the two real unknowns x, y in z = x + yi, ``answer: pair``. This
  is NOT conversion to/from polar (senior-convert-between-cartesian-and-polar-
  form owns that) and NOT bare part-extraction (complex-real-part/complex-
  imaginary-part own that); every equation stays in the z, z-bar (conjugate),
  Re(z), Im(z) family, never a complex number multiplying z (ACMSM070's
  separate, unbuilt arithmetic-division row -- no tier here ever produces a
  ``(a+bi)z = c+di``-shaped equation). easy: (ax+b) + (cy+d)i = e+fi, already
  separated into real/imaginary components. medium: z + (p+qi) = r+si, one
  extra translation step (recognise z's own parts) before equating. hard:
  z + k*z-bar = r+si, requiring the conjugate's sign flip on y before
  equating.

Both generators verify their canonical answer against the ORIGINAL stated
equation via ``sympy``'s complex arithmetic (``sympy.I``, ``sympy.conjugate``)
-- a genuinely different route from the plain-integer arithmetic construction
uses to derive x0/y0 (or the answer count) in the first place.
"""

from __future__ import annotations

import sympy

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    xy_solution,
)

_RESAMPLE_ATTEMPTS = 200

# =============================================================================
# Skill 1: p11c_set_union_cardinality (ACMSM005)
# =============================================================================

_TWO_SUBJECTS = ["Math", "Science", "Art", "Music", "History", "Geography", "Drama", "French"]
_THREE_ACTIVITIES = ["football", "basketball", "tennis", "swimming", "chess", "reading", "painting", "cycling"]


def _verify_union_easy(N: int, a: int, b: int, c: int, answer: int):
    """Independent check: partition N into its four disjoint regions
    (A-only, B-only, both, neither) and confirm they sum to N -- a genuinely
    different route from the N - a - b + c rearrangement construction uses.
    Factored out so a test can call it directly with a corrupted answer."""
    a_only, b_only = a - c, b - c
    check_equivalent(
        a_only + b_only + c + answer, N, "the four disjoint regions (A-only, B-only, both, neither) must sum to N"
    )


def _verify_union_medium(A: int, B: int, C: int, AB: int, AC: int, BC: int, ABC: int, answer: int):
    """Independent check: the inclusion-exclusion formula applied to the
    STATED totals must reproduce the direct sum of the seven disjoint
    regions construction actually drew -- a genuinely different route from
    simply re-running the same accumulation. Factored out so a test can call
    it directly with a corrupted answer."""
    formula_union = A + B + C - AB - AC - BC + ABC
    check_equivalent(
        formula_union, answer, "inclusion-exclusion applied to the stated totals must match the direct partition sum"
    )


def _verify_union_hard(N: int, a: int, b: int, c: int, neither: int):
    """Independent check: the four disjoint regions (A-only, B-only, both,
    neither) built from the STATED quantities (a, b, N, neither) must sum to
    N -- a genuinely different route (region-partition reasoning) from the
    union-formula rearrangement a + b - (N - neither) used to derive c.
    Factored out so a test can call it directly with a corrupted c."""
    check_equivalent(
        (a - c) + (b - c) + c + neither, N, "the four disjoint regions (A-only, B-only, both, neither) must sum to N"
    )


def _verify_union_three_set_unknown(
    unknown: str, A: int, B: int, C: int, AB: int, AC: int, BC: int, ABC: int, union_total: int, answer: int
):
    """Independent check: for whichever term is missing, re-applies the
    seven-term inclusion-exclusion identity to the STATED quantities plus
    the candidate `answer` and confirms it reproduces `union_total` -- the
    same identity `_verify_union_medium` checks, but rearranged for a
    different unknown each time. One branch per unknown kind, factored so a
    test can call each directly with a corrupted answer."""
    if unknown == "ABC":
        check_equivalent(
            A + B + C - AB - AC - BC + answer,
            union_total,
            "inclusion-exclusion rearranged for the triple intersection must reproduce the stated union total",
        )
    elif unknown == "C":
        check_equivalent(
            A + B + answer - AB - AC - BC + ABC,
            union_total,
            "inclusion-exclusion rearranged for the third set's total must reproduce the stated union total",
        )
    elif unknown == "AB":
        check_equivalent(
            A + B + C - answer - AC - BC + ABC,
            union_total,
            "inclusion-exclusion rearranged for the AB pairwise total must reproduce the stated union total",
        )
    else:
        raise ValueError(f"unknown unknown kind: {unknown!r}")


def _union_easy(rng):
    """Two-set union framed as a survey/count word problem with an explicit
    finite universe N. The answer is always the "neither" count,
    N - |A union B| = N - a - b + c, which requires N itself -- a quantity
    the shipping probability skill's own contract never carries at all.

    Constraints (asserted over 60 seeds by the test): a, b > c > 0 and
    a + b <= N -- the latter guarantees the answer, and every one of its
    three distractors, is a non-negative integer strictly less than N (no
    "obviously wrong because negative" giveaway option).
    """
    for _ in range(_RESAMPLE_ATTEMPTS):
        s1, s2 = rng.sample(_TWO_SUBJECTS, 2)
        N = rng.randint(20, 50)
        c = rng.randint(2, 8)
        a = c + rng.randint(1, 15)
        b = c + rng.randint(1, 15)
        if a + b > N:
            continue
        answer = N - a - b + c
        forgot_overlap = N - a - b  # never re-added the overlap -> undercounts "neither" by c
        double_subtracted = N - a - b + 2 * c  # subtracted the overlap twice -> overcounts "neither" by c
        confuses_union = a + b - c  # reports the union total itself, not "neither"
        try:
            check_pairwise_distinct(answer, forgot_overlap, double_subtracted, confuses_union)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p11c_set_union_cardinality (easy): no collision-free draw found")

    check(0 <= answer < N, "easy tier: neither count must be a plausible count under N")
    check(
        0 <= forgot_overlap < N and 0 <= double_subtracted < N and 0 <= confuses_union < N,
        "easy tier: every distractor must be a plausible non-negative count under N",
    )

    _verify_union_easy(N, a, b, c, answer)

    text = (
        f"In a class of {N} students, {a} study {s1}, {b} study {s2}, and {c} study both "
        f"{s1} and {s2}. How many study neither?"
    )
    latex = (
        rf"\text{{In a class of }} {N} \text{{ students, }} {a} \text{{ study {s1}, }} "
        rf"{b} \text{{ study {s2}, and }} {c} \text{{ study both. How many study neither?}}"
    )
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(forgot_overlap), str(double_subtracted), str(confuses_union)], n=3)
    check(len(distractors) == 3, "easy tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"|A union B| = {a} + {b} - {c} = {a + b - c}.",
            f"Neither = N - |A union B| = {N} - {a + b - c} = {answer_text}.",
        ],
        distractors=distractors,
    )


def _union_medium(rng):
    """Three-set union |A union B union C|, all seven Venn-diagram regions
    either given directly (the three totals) or derivable from the stated
    pairwise/triple counts. Every pairwise and triple count is disambiguated
    with "(whether or not they also like ...)" -- the classic second reading
    of a three-set survey question (a stated "like both X and Y" figure could
    otherwise be misread as EXCLUDING the third activity).

    No shipping skill has any three-event/three-set analogue at all.
    """
    for _ in range(_RESAMPLE_ATTEMPTS):
        act1, act2, act3 = rng.sample(_THREE_ACTIVITIES, 3)
        # Every region >= 2 so no stated count is ever a bare "1 like ..." --
        # avoids the singular/plural wording defect class entirely.
        r_a, r_b, r_c, r_ab, r_ac, r_bc, r_abc = (rng.randint(2, 8) for _ in range(7))

        A = r_a + r_ab + r_ac + r_abc
        B = r_b + r_ab + r_bc + r_abc
        C = r_c + r_ac + r_bc + r_abc
        AB = r_ab + r_abc
        AC = r_ac + r_abc
        BC = r_bc + r_abc
        ABC = r_abc

        answer = r_a + r_b + r_c + r_ab + r_ac + r_bc + r_abc  # direct partition sum
        no_correction = A + B + C  # forgets to subtract any overlap
        omitted_readdition = A + B + C - AB - AC - BC  # forgets +|A intersect B intersect C|
        double_subtracted = A + B + C - AB - AC - BC - ABC  # subtracts the triple again instead of re-adding it
        try:
            check_pairwise_distinct(answer, no_correction, omitted_readdition, double_subtracted)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p11c_set_union_cardinality (medium): no collision-free draw found")

    _verify_union_medium(A, B, C, AB, AC, BC, ABC, answer)

    text = (
        f"In a survey of students, {A} like {act1}, {B} like {act2} and {C} like {act3}. "
        f"{AB} like both {act1} and {act2} (whether or not they also like {act3}), "
        f"{AC} like both {act1} and {act3} (whether or not they also like {act2}), "
        f"{BC} like both {act2} and {act3} (whether or not they also like {act1}), and "
        f"{ABC} like all three. How many students like at least one of {act1}, {act2} "
        f"or {act3}?"
    )
    latex = (
        rf"\text{{{A} like {act1}, {B} like {act2}, {C} like {act3}. }} {AB} "
        rf"\text{{ like both {act1} and {act2}, }} {AC} \text{{ like both {act1} and "
        rf"{act3}, }} {BC} \text{{ like both {act2} and {act3}, and }} {ABC} "
        rf"\text{{ like all three. How many like at least one?}}"
    )
    answer_text = str(answer)
    distractors = pick_distractors(
        answer_text, [str(no_correction), str(omitted_readdition), str(double_subtracted)], n=3
    )
    check(len(distractors) == 3, "medium tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"|A union B union C| = |A|+|B|+|C|-|A∩B|-|A∩C|-|B∩C|+|A∩B∩C| "
            f"= {A}+{B}+{C}-{AB}-{AC}-{BC}+{ABC} = {answer_text}."
        ],
        distractors=distractors,
    )


def _union_hard(rng):
    """Inverse direction: given the total universe N, the two individual set
    sizes and the "neither" count, find the overlap |A intersect B| by
    rearranging the two-set formula -- genuinely different reasoning (solve
    for the subtracted term) from either easier tier.
    """
    for _ in range(_RESAMPLE_ATTEMPTS):
        s1, s2 = rng.sample(_TWO_SUBJECTS, 2)
        a = rng.randint(15, 35)
        b = rng.randint(15, 35)
        if min(a, b) < 6:
            continue
        c = rng.randint(4, min(a, b) - 1)
        # neither >= 2 so "{neither} study neither ..." is never a bare "1
        # study" (the singular/plural wording defect class), and neither <
        # c so used_N_as_union (= c - neither) stays a positive distractor.
        neither = rng.randint(2, c - 1)
        union = a + b - c
        N = union + neither

        used_N_as_union = a + b - N  # treated the universe N as if it were the union total
        b_only_region = union - a  # the B-only region, mistaken for the overlap
        confuses_with_neither = neither  # reports "neither" itself as the overlap
        if not (0 <= used_N_as_union < N and 0 <= b_only_region < N):
            continue  # every distractor must be a plausible non-negative count under N
        try:
            check_pairwise_distinct(c, used_N_as_union, b_only_region, confuses_with_neither)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p11c_set_union_cardinality (hard): no collision-free draw found")

    _verify_union_hard(N, a, b, c, neither)

    text = (
        f"In a class of {N} students, {a} study {s1} and {b} study {s2}. {neither} study "
        f"neither {s1} nor {s2}. How many study both {s1} and {s2}?"
    )
    latex = (
        rf"\text{{In a class of }} {N} \text{{ students, }} {a} \text{{ study {s1} and }} "
        rf"{b} \text{{ study {s2}. }} {neither} \text{{ study neither. How many study "
        rf"both?}}"
    )
    answer_text = str(c)
    distractors = pick_distractors(
        answer_text, [str(used_N_as_union), str(b_only_region), str(confuses_with_neither)], n=3
    )
    check(len(distractors) == 3, "hard tier expects 3 distinct distractors")

    return dict(
        text=text,
        latex=latex,
        answer=c,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"|A union B| = N - neither = {N} - {neither} = {union}.",
            f"|A intersect B| = |A| + |B| - |A union B| = {a} + {b} - {union} = {answer_text}.",
        ],
        distractors=distractors,
    )


def _union_three_set_unknown(rng):
    """Three-set union identity, hard tier: the union
    total is now a STATED quantity and exactly one of the seven
    region-derived totals (A, B, C, AB, AC, BC, ABC) is withheld -- which one
    varies across draws. The learner rearranges the seven-term identity for
    whichever term is missing, a structurally different move each time this
    tier is drawn (three distinct prompt shapes), not "one more set" than
    `_union_medium`'s always-find-the-union question.

    Reuses `_union_medium`'s region draw and derived quantities exactly (same
    `rng.randint(2, 8)` per region, three distinct activities, same
    A/B/C/AB/AC/BC/ABC/union_total definitions) so the only thing that
    changes is which quantity is omitted from the prompt and asked for.
    """
    for _ in range(_RESAMPLE_ATTEMPTS):
        unknown = rng.choice(["ABC", "C", "AB"])
        act1, act2, act3 = rng.sample(_THREE_ACTIVITIES, 3)
        # Every region >= 2, same grammar-defect avoidance as _union_medium.
        r_a, r_b, r_c, r_ab, r_ac, r_bc, r_abc = (rng.randint(2, 8) for _ in range(7))

        A = r_a + r_ab + r_ac + r_abc
        B = r_b + r_ab + r_bc + r_abc
        C = r_c + r_ac + r_bc + r_abc
        AB = r_ab + r_abc
        AC = r_ac + r_abc
        BC = r_bc + r_abc
        ABC = r_abc
        union_total = r_a + r_b + r_c + r_ab + r_ac + r_bc + r_abc

        if unknown == "ABC":
            answer = union_total - A - B - C + AB + AC + BC
            # Distractors, every one computable from only the STATED
            # quantities (never from ABC itself, which is what is being
            # solved for) and non-negative by construction on these draw
            # ranges (A, B, C, AB+AC+BC and union_total are all structurally
            # larger than ABC <= 8):
            d1 = A + B + C - union_total  # dropped the pairwise correction terms entirely
            d2 = AB + AC + BC  # summed the stated pairwise totals, forgetting they double-count ABC
            d3 = union_total  # reported the stated union total itself
            distractor_vals = [d1, d2, d3]
        elif unknown == "C":
            answer = union_total - A - B + AB + AC + BC - ABC
            d1 = union_total - A - B + AB + AC + BC  # forgot the -ABC term (omits-triple-readdition)
            d2 = AC + BC - ABC  # dropped the A/B/union contribution entirely
            d3 = union_total  # reported the stated union total itself
            distractor_vals = [d1, d2, d3]
        else:  # unknown == "AB"
            answer = A + B + C - AC - BC + ABC - union_total
            d1 = A + B + C - AC - BC - union_total  # forgot the +ABC readdition (omits-triple-readdition)
            d2 = A + B + C - union_total  # dropped AC, BC and ABC entirely
            d3 = union_total  # reported the stated union total itself
            distractor_vals = [d1, d2, d3]

        if answer < 0 or any(v < 0 for v in distractor_vals):
            continue
        try:
            check_pairwise_distinct(answer, *distractor_vals)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("p11c_set_union_cardinality (three_set_unknown): no collision-free draw found")

    _verify_union_three_set_unknown(unknown, A, B, C, AB, AC, BC, ABC, union_total, answer)

    stated = f"{union_total} students like at least one of {act1}, {act2} or {act3}."
    if unknown == "ABC":
        prompt = (
            f"In a survey of students, {A} like {act1}, {B} like {act2} and {C} like "
            f"{act3}. {AB} like both {act1} and {act2} (whether or not they also like "
            f"{act3}), {AC} like both {act1} and {act3} (whether or not they also like "
            f"{act2}), and {BC} like both {act2} and {act3} (whether or not they also "
            f"like {act1}). {stated} How many students like all three?"
        )
        working = [f"ABC = union - A - B - C + AB + AC + BC = {union_total}-{A}-{B}-{C}+{AB}+{AC}+{BC} = {answer}."]
    elif unknown == "C":
        prompt = (
            f"In a survey of students, {A} like {act1} and {B} like {act2}. {AB} like "
            f"both {act1} and {act2} (whether or not they also like {act3}), {AC} like "
            f"both {act1} and {act3} (whether or not they also like {act2}), {BC} like "
            f"both {act2} and {act3} (whether or not they also like {act1}), and {ABC} "
            f"like all three. {stated} How many students like {act3} in total (whether "
            f"or not they also like {act1} or {act2})?"
        )
        working = [f"C = union - A - B + AB + AC + BC - ABC = {union_total}-{A}-{B}+{AB}+{AC}+{BC}-{ABC} = {answer}."]
    else:  # unknown == "AB"
        prompt = (
            f"In a survey of students, {A} like {act1}, {B} like {act2} and {C} like "
            f"{act3}. {AC} like both {act1} and {act3} (whether or not they also like "
            f"{act2}), {BC} like both {act2} and {act3} (whether or not they also like "
            f"{act1}), and {ABC} like all three. {stated} How many students like both "
            f"{act1} and {act2} (whether or not they also like {act3})?"
        )
        working = [f"AB = A + B + C - AC - BC + ABC - union = {A}+{B}+{C}-{AC}-{BC}+{ABC}-{union_total} = {answer}."]

    latex = rf"\text{{{prompt}}}"
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(v) for v in distractor_vals], n=3)
    check(len(distractors) == 3, "hard tier (three_set_unknown) expects 3 distinct distractors")

    return dict(
        text=prompt,
        latex=latex,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


@generator(
    "p11c_set_union_cardinality",
    level=12,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
        "two_set_overlap": dict(mode="two_set_overlap"),
        "three_set_unknown": dict(mode="three_set_unknown"),
    },
)
def p11c_set_union_cardinality(rng, mode="easy"):
    """Inclusion-exclusion for the number of elements in the union of two and
    the union of three sets. easy: two-set union framed with an explicit
    finite universe, find the "neither" count. medium (legacy preset) /
    two_set_overlap (shipping): inverse -- given the universe, the two set
    sizes and the "neither" count, find the overlap. hard (legacy preset):
    three-set union, all seven Venn-diagram regions given or derivable.
    three_set_unknown (shipping): three-set union with the union total now
    stated and exactly one of the seven region totals withheld -- which one
    varies across draws -- solve for the missing term."""
    if mode == "easy":
        return _union_easy(rng)
    if mode == "medium":
        return _union_medium(rng)
    if mode == "hard":
        return _union_hard(rng)
    if mode == "two_set_overlap":
        return _union_hard(rng)
    if mode == "three_set_unknown":
        return _union_three_set_unknown(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# =============================================================================
# Skill 2: p11c_cartesian_form_equating (ACMSM078)
# =============================================================================

_NONZERO_COEFFS = (-4, -3, -2, -1, 1, 2, 3, 4)
_NONZERO_SHIFTS = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
_CONJ_K_VALUES = (-4, -3, -2, 2, 3, 4)
_CONJ_M_VALUES = (-3, -2, -1, 1, 2, 3)


def _complex_text(re_part: int, im_part: int) -> str:
    """``re + im*i`` the way a student would write it: "3 - 2i", "i", "-6i",
    "-7" -- reuses ``join_terms`` (never hand-rolled) so a zero part is
    dropped and a unit imaginary coefficient renders as "i", not "1i"."""
    return join_terms([(re_part, ""), (im_part, "i")])


def _verify_cartesian_easy(a: int, b: int, c: int, d: int, e: int, f: int, x0: int, y0: int):
    """Independent check: substitute the solved (x, y) back into the
    ORIGINAL stated equation as a full complex expression (``sympy.I``), not
    into the separated x/y equations construction itself used to derive
    e, f. Factored out so a test can call it directly with a corrupted (x, y)."""
    lhs = (a * x0 + b) + (c * y0 + d) * sympy.I
    rhs = sympy.Integer(e) + sympy.Integer(f) * sympy.I
    check_equivalent(lhs, rhs, "solved (x, y) must satisfy the original stated equation")


def _verify_cartesian_medium(p: int, q: int, r: int, s: int, x0: int, y0: int):
    """Independent check: substitute the solved (x, y) back into the
    ORIGINAL stated equation z + (p + qi) = r + si as a single complex
    expression. Factored out so a test can call it directly with a
    corrupted (x, y)."""
    lhs = (x0 + y0 * sympy.I) + (sympy.Integer(p) + sympy.Integer(q) * sympy.I)
    rhs = sympy.Integer(r) + sympy.Integer(s) * sympy.I
    check_equivalent(lhs, rhs, "solved (x, y) must satisfy the original stated equation")


def _verify_cartesian_hard(k: int, r: int, s: int, x0: int, y0: int):
    """Independent check: build z and its conjugate via ``sympy.conjugate``
    (not the hand-built x - yi construction uses), substitute into the
    ORIGINAL stated equation z + k*z-bar = r + si as a single complex
    expression. Factored out so a test can call it directly with a
    corrupted (x, y)."""
    z = sympy.Integer(x0) + sympy.Integer(y0) * sympy.I
    zbar = sympy.conjugate(z)
    lhs = sympy.expand(z + k * zbar)
    rhs = sympy.Integer(r) + sympy.Integer(s) * sympy.I
    check_equivalent(lhs, rhs, "solved (x, y) must satisfy the original stated equation")


def _cartesian_easy(rng):
    """(ax + b) + (cy + d)i = e + fi -- a linear equation in each of the real
    and imaginary parts separately, already separated (one-step solve per
    part)."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        x0, y0 = rng.randint(-9, 9), rng.randint(-9, 9)
        if x0 == 0 or y0 == 0 or x0 == y0:
            continue
        a, c = rng.choice(_NONZERO_COEFFS), rng.choice(_NONZERO_COEFFS)
        b, d = rng.randint(-9, 9), rng.randint(-9, 9)
        e, f = a * x0 + b, c * y0 + d
        break
    else:
        raise ValueError("p11c_cartesian_form_equating (easy): no valid draw found")

    _verify_cartesian_easy(a, b, c, d, e, f, x0, y0)

    real_expr = join_terms([(a, "x"), (b, "")])
    imag_expr = join_terms([(c, "y"), (d, "")])
    rhs_text = _complex_text(e, f)

    text = f"Let z = x + yi. If ({real_expr}) + ({imag_expr})i = {rhs_text}, find x and y."
    latex = (
        rf"\text{{Let }} z = x + yi. \text{{ If }} ({real_expr}) + ({imag_expr})i = "
        rf"{rhs_text}, \text{{ find }} x \text{{ and }} y."
    )

    swap, sign_x, copy_both = (y0, x0), (-x0, y0), (x0, x0)
    check_pairwise_distinct((x0, y0), swap, sign_x, copy_both)
    result = xy_solution(x0, y0)
    dist_strs = [f"x = {cx}, y = {cy}" for cx, cy in (swap, sign_x, copy_both)]
    result["distractors"] = pick_distractors(result["answer_text"], dist_strs, n=3)
    check(len(result["distractors"]) == 3, "easy tier expects 3 distinct distractors")
    result.update(
        text=text,
        latex=latex,
        working=[
            f"Real parts: {real_expr} = {e}, so x = {x0}.",
            f"Imaginary parts: {imag_expr} = {f}, so y = {y0}.",
        ],
    )
    return result


def _cartesian_medium(rng):
    """z + (p + qi) = r + si, i.e. (x + p) + (y + q)i = r + si -- the learner
    must recognise z's own real/imaginary parts before equating, one small
    extra translation step vs. easy."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        x0, y0 = rng.randint(-9, 9), rng.randint(-9, 9)
        if x0 == 0 or y0 == 0 or x0 == y0:
            continue
        p, q = rng.choice(_NONZERO_SHIFTS), rng.choice(_NONZERO_SHIFTS)
        r, s = x0 + p, y0 + q

        swap, direct_copy, copy_both = (y0, x0), (r, s), (x0, x0)
        candidates = [swap, direct_copy, copy_both]
        filtered = [pair for pair in candidates if pair != (x0, y0)]
        if len(set(filtered)) < 3:
            continue
        break
    else:
        raise ValueError("p11c_cartesian_form_equating (medium): no collision-free draw found")

    _verify_cartesian_medium(p, q, r, s, x0, y0)

    pq_text = _complex_text(p, q)
    rhs_text = _complex_text(r, s)
    text = f"Let z = x + yi. If z + ({pq_text}) = {rhs_text}, find x and y."
    latex = (
        rf"\text{{Let }} z = x + yi. \text{{ If }} z + ({pq_text}) = {rhs_text}, "
        rf"\text{{ find }} x \text{{ and }} y."
    )

    real_expr = join_terms([(1, "x"), (p, "")])
    imag_expr = join_terms([(1, "y"), (q, "")])

    result = xy_solution(x0, y0)
    dist_strs = [f"x = {cx}, y = {cy}" for cx, cy in filtered]
    result["distractors"] = pick_distractors(result["answer_text"], dist_strs, n=3)
    check(len(result["distractors"]) == 3, "medium tier expects 3 distinct distractors")
    result.update(
        text=text,
        latex=latex,
        working=[
            f"z + ({pq_text}) = ({real_expr}) + ({imag_expr})i.",
            f"Real parts: {real_expr} = {r}, so x = {x0}. Imaginary parts: {imag_expr} = {s}, so y = {y0}.",
        ],
    )
    return result


def _cartesian_hard(rng):
    """z + k*z-bar = r + si, where z-bar = x - yi -- the learner must expand
    the conjugate correctly (sign flip on y) before equating, giving
    (1+k)x = r and (1-k)y = s, which the earlier tiers never require.

    y0 = (1+k)*m is drawn deliberately (not x0, y0 independently) so the
    "forgot the conjugate's sign flip" distractor -- treating z-bar as z
    again, giving (1+k)y = s -- solves to y' = s/(1+k) = (1-k)*m, an EXACT
    integer every time, never a fraction identifiable by shape among integer
    options.
    """
    for _ in range(_RESAMPLE_ATTEMPTS):
        k = rng.choice(_CONJ_K_VALUES)
        m = rng.choice(_CONJ_M_VALUES)
        y0 = (1 + k) * m
        x0 = rng.choice([v for v in range(-6, 7) if v != 0])
        if x0 == y0 or x0 == 0:
            continue
        r = (1 + k) * x0
        s = (1 - k) * y0
        conj_mistake_y = (1 - k) * m  # = s / (1 + k), the "forgot the sign flip" y-value

        swap, conj_mistake, copy_both = (y0, x0), (x0, conj_mistake_y), (x0, x0)
        candidates = [swap, conj_mistake, copy_both]
        filtered = [pair for pair in candidates if pair != (x0, y0)]
        if len(set(filtered)) < 3:
            continue
        break
    else:
        raise ValueError("p11c_cartesian_form_equating (hard): no collision-free draw found")

    _verify_cartesian_hard(k, r, s, x0, y0)

    sign_word = "+" if k > 0 else "-"
    rhs_text = _complex_text(r, s)
    text = (
        f"Let z = x + yi. The conjugate of z is x - yi. If z {sign_word} {abs(k)} times "
        f"the conjugate of z equals {rhs_text}, find x and y."
    )
    latex = (
        rf"\text{{Let }} z = x + yi. \text{{ The conjugate of }} z \text{{ is }} x - yi. "
        rf"\text{{ If }} z {sign_word} {abs(k)} \overline{{z}} = {rhs_text}, "
        rf"\text{{ find }} x \text{{ and }} y."
    )
    check(
        ")z" not in text and "iz" not in text,
        "the equation must never take the (a+bi)z multiplication form (ACMSM070's row)",
    )

    result = xy_solution(x0, y0)
    dist_strs = [f"x = {cx}, y = {cy}" for cx, cy in filtered]
    result["distractors"] = pick_distractors(result["answer_text"], dist_strs, n=3)
    check(len(result["distractors"]) == 3, "hard tier expects 3 distinct distractors")
    result.update(
        text=text,
        latex=latex,
        working=[
            f"z {sign_word} {abs(k)}(conjugate of z) = {join_terms([(1 + k, 'x'), (1 - k, 'yi')])}.",
            f"Real parts: {1 + k}x = {r}, so x = {x0}. Imaginary parts: {1 - k}y = {s}, so y = {y0}.",
        ],
    )
    return result


@generator(
    "p11c_cartesian_form_equating",
    level=12,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def p11c_cartesian_form_equating(rng, mode="easy"):
    """Equate real and imaginary parts of a complex equation to solve for the
    two real unknowns x, y in z = x + yi. easy: (ax+b)+(cy+d)i = e+fi, already
    separated. medium: z + (p+qi) = r+si, one extra translation step. hard:
    z + k*z-bar = r+si, requiring the conjugate's sign flip on y."""
    if mode == "easy":
        return _cartesian_easy(rng)
    if mode == "medium":
        return _cartesian_medium(rng)
    if mode == "hard":
        return _cartesian_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")
