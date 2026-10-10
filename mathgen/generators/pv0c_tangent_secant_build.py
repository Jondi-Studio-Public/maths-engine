"""Generates `pv0c_tangent_secant` questions: find a missing length from AM x BM = TM^2 for a secant and a
tangent.

The tangent-secant length relation.

``pv0c_tangent_secant`` -- given two of the lengths AM, BM and TM for a secant through A and B and a tangent
at T drawn from an external point M, find the third using AM x BM = TM^2.

Convention stated every time: M is outside the circle; the secant from M
meets the circle first at B, then at A, so B lies between A and M (AM > BM);
the tangent from M touches at T. Every length is a positive integer under
1000, in cm.

Tiers:
- easy: AM and BM stated, find TM.
- medium: TM and one of AM/BM stated (draw which), find the other.
- hard: BM and the chord AB stated, find TM; or TM and BM stated, find AB
  (draw which) -- the relation uses the WHOLE secant AM = AB + BM.
- exam: TM and the chord AB stated, find BM or AM (draw which), solving
  x(x + AB) = TM^2 and rejecting the negative root; or the secant passes
  through the circle's centre O (radius r stated) and TM is stated, find MO.

Every (AM, BM, TM) triple is built from AM = d*q^2, BM = d*p^2 (q > p >= 1),
so TM = d*p*q is always an integer and BM < TM < AM strictly.

Fix: distractors are COMPUTED misconceptions (chord-for-secant,
no-square-root, arithmetic-mean, whole-secant-when-part-asked, tm-not-squared,
other-root-or-other-segment, pythagoras-slip), never a number already printed
in the prompt -- an echoed given is identifiable without any geometry in
multiple-choice mode, not a misconception. Every candidate is filtered
against the numbers actually printed in ``text`` (a regex over the rendered
string, not the generator's own variable names), which also catches stray
literals such as the "2" in "TM^2".
"""

from __future__ import annotations

import math
import random
import re

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

# ---------------------------------------------------------------------------
# The (d, p, q) triple pool: AM = d*q^2, BM = d*p^2, TM = d*p*q, AB = AM - BM.
# Plausibility (proven red on the unguarded generator first -- see the
# oracle/build tests): AM/BM = (q/p)^2 in [1.3, 9] (no near-zero chord, no
# hair-thin secant); no length of 1 beside a length over 50; every length
# under 1000.
# ---------------------------------------------------------------------------


def _build_triple_pool():
    pool = []
    for d in range(1, 41):
        for p in range(1, 9):
            for q in range(p + 1, 13):
                ratio = (q / p) ** 2
                if not (1.3 <= ratio <= 9.0):
                    continue
                am, bm, tm = d * q * q, d * p * p, d * p * q
                ab = am - bm
                if am >= 1000 or bm >= 1000 or tm >= 1000 or ab >= 1000:
                    continue
                small = min(am, bm, tm, ab)
                large = max(am, bm, tm, ab)
                if small == 1 and large > 50:
                    continue
                if len({am, bm, tm, ab}) < 4:
                    continue
                pool.append((d, p, q, am, bm, tm, ab))
    check(len(pool) >= 50, "tangent_secant: triple pool too small")
    return pool


_TRIPLE_POOL = _build_triple_pool()


def _pythagorean_pool():
    """Scaled Pythagorean triples (r, TM, MO) for the exam-tier centre
    variant, filtered so BM = MO - r and AM = MO + r stay within the same
    plausibility ratio bound as every other tier."""
    base = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41)]
    pool = []
    for a, b, c in base:
        for k in range(1, 15):
            for r, tm in ((a * k, b * k), (b * k, a * k)):
                mo = c * k
                if mo >= 1000 or r >= 1000 or tm >= 1000:
                    continue
                bm, am = mo - r, mo + r
                if bm <= 0:
                    continue
                ratio = am / bm
                if not (1.3 <= ratio <= 9.0):
                    continue
                if bm == 1 and max(am, tm, r, mo) > 50:
                    continue
                if len({r, tm, mo, bm, am}) < 5:
                    continue
                pool.append((r, tm, mo, bm, am))
    check(len(pool) >= 15, "tangent_secant: pythagorean pool too small")
    return pool


_PYTHAG_POOL = _pythagorean_pool()

# ---------------------------------------------------------------------------
# Wording banks
# ---------------------------------------------------------------------------

_CONVENTION = (
    "M is a point outside a circle. A secant from M meets the circle first at B and then at A, "
    "so B lies between A and M. A tangent from M touches the circle at T."
)

_EASY_TEMPLATES = (
    _CONVENTION + " AM is {am} cm and BM is {bm} cm. Find the length of TM, in cm.",
    _CONVENTION + " The length AM is {am} cm and the length BM is {bm} cm. Using AM x BM = TM^2, find TM, in cm.",
    "From an external point M, a secant meets a circle at B then A (with B between A and M), and a "
    "tangent from M touches the circle at T. Given AM = {am} cm and BM = {bm} cm, find TM, in cm.",
)

_MEDIUM_TEMPLATES_AM = (
    _CONVENTION + " TM is {tm} cm and BM is {bm} cm. Find the length of AM, in cm.",
    _CONVENTION + " The tangent length TM is {tm} cm and BM is {bm} cm. Using AM x BM = TM^2, find AM, in cm.",
)

_MEDIUM_TEMPLATES_BM = (
    _CONVENTION + " TM is {tm} cm and AM is {am} cm. Find the length of BM, in cm.",
    _CONVENTION + " The tangent length TM is {tm} cm and AM is {am} cm. Using AM x BM = TM^2, find BM, in cm.",
)

_HARD_TEMPLATES_TM = (
    _CONVENTION + " BM is {bm} cm and the chord AB is {ab} cm. Find the length of TM, in cm.",
    _CONVENTION + " The chord AB is {ab} cm and BM is {bm} cm. Using the whole secant AM = AB + BM "
    "and AM x BM = TM^2, find TM, in cm.",
)

_HARD_TEMPLATES_AB = (
    _CONVENTION + " TM is {tm} cm and BM is {bm} cm. Find the length of the chord AB, in cm.",
    _CONVENTION + " The tangent length TM is {tm} cm and BM is {bm} cm. Using AM x BM = TM^2 and "
    "AM = AB + BM, find the length of the chord AB, in cm.",
)

_EXAM_TEMPLATES_BM = (
    _CONVENTION + " TM is {tm} cm and the chord AB is {ab} cm. Given that all lengths are positive, "
    "find the length of BM, in cm.",
)

_EXAM_TEMPLATES_AM = (
    _CONVENTION + " TM is {tm} cm and the chord AB is {ab} cm. Given that all lengths are positive, "
    "find the length of AM, in cm.",
)

_EXAM_TEMPLATES_CENTRE = (
    "M is a point outside a circle of radius {r} cm, centre O. A secant from M passes through O, "
    "meeting the circle first at B then at A (so AB is a diameter), and a tangent from M touches the "
    "circle at T. TM is {tm} cm. Find the length of MO, in cm.",
)


def _computed_candidates(text: str, answer: int, raw: list) -> list[str]:
    """Filter a priority-ordered list of computed misconception values down to
    positive integers that are neither the answer nor a number printed
    anywhere in ``text`` (a regex over the rendered string, not the
    generator's own variable names -- this is what also catches a stray
    literal such as the "2" in "TM^2")."""
    printed = set(re.findall(r"\d+", text))
    out = []
    for v in raw:
        if v is None or v <= 0 or v == answer:
            continue
        s = str(v)
        if s in printed:
            continue
        out.append(s)
    return out


@generator(
    "pv0c_tangent_secant",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def pv0c_tangent_secant(rng: random.Random, tier: str = "easy"):
    """AM x BM = TM^2 for a secant through A, B and a tangent at T from an
    external point M, solving for whichever length is not given."""

    if tier == "easy":
        d, p, q, am, bm, tm, ab = rng.choice(_TRIPLE_POOL)
        text = phrase(rng, _EASY_TEMPLATES, am=am, bm=bm)
        answer = tm
        check(bm * am == tm * tm, f"tangent_secant/easy: {bm}*{am} != {tm}^2")

        raw = [
            ab,  # chord-for-secant: reports the chord, not TM
            am * bm if am * bm < 1000 else None,  # no-square-root: AM x BM = TM^2, not sqrt'd
            (am + bm) // 2,  # arithmetic-mean (floor when odd)
        ]
        candidates = _computed_candidates(text, answer, raw)

    elif tier == "medium":
        d, p, q, am, bm, tm, ab = rng.choice(_TRIPLE_POOL)
        unknown = rng.choice(["AM", "BM"])
        if unknown == "AM":
            text = phrase(rng, _MEDIUM_TEMPLATES_AM, tm=tm, bm=bm)
            answer = am
            check(bm * am == tm * tm, f"tangent_secant/medium(AM): {bm}*{am} != {tm}^2")
            raw = [
                ab,  # chord-for-secant
                tm * bm if tm * bm < 1000 else None,  # no-square-root
                tm // bm if bm and tm % bm == 0 else None,  # tm-not-squared
                (tm + bm) // 2,  # arithmetic-mean (floor when odd)
            ]
            candidates = _computed_candidates(text, answer, raw)
        else:
            text = phrase(rng, _MEDIUM_TEMPLATES_BM, tm=tm, am=am)
            answer = bm
            check(bm * am == tm * tm, f"tangent_secant/medium(BM): {bm}*{am} != {tm}^2")
            raw = [
                ab,  # chord-for-secant
                tm * am if tm * am < 1000 else None,  # no-square-root
                tm // am if am and tm % am == 0 else None,  # tm-not-squared
                (tm + am) // 2,  # arithmetic-mean (floor when odd)
            ]
            candidates = _computed_candidates(text, answer, raw)

    elif tier == "hard":
        d, p, q, am, bm, tm, ab = rng.choice(_TRIPLE_POOL)
        variant = rng.choice(["find_tm", "find_ab"])
        if variant == "find_tm":
            text = phrase(rng, _HARD_TEMPLATES_TM, bm=bm, ab=ab)
            answer = tm
            check(am == ab + bm, f"tangent_secant/hard: am != ab+bm ({am} vs {ab}+{bm})")
            check(bm * am == tm * tm, "tangent_secant/hard(find_tm): relation failed")
            chord_for_secant_sq = ab * bm
            root = math.isqrt(chord_for_secant_sq)
            chord_for_secant = root if root * root == chord_for_secant_sq else None
            raw = [
                chord_for_secant,  # chord-for-secant: TM^2 = AB x BM
                am,  # whole-secant-when-part-asked: stops at AM
                am * bm if am * bm < 1000 else None,  # no-square-root
                (am + bm) // 2,  # arithmetic-mean (floor when odd)
            ]
            candidates = _computed_candidates(text, answer, raw)
        else:
            text = phrase(rng, _HARD_TEMPLATES_AB, tm=tm, bm=bm)
            answer = ab
            check(am == ab + bm, f"tangent_secant/hard: am != ab+bm ({am} vs {ab}+{bm})")
            check(bm * am == tm * tm, "tangent_secant/hard(find_ab): relation failed")
            raw = [
                am,  # whole-secant-when-part-asked: forgot -BM
                am + bm,  # sign slip: added instead of subtracted
                tm - bm if tm > bm else None,  # tm-not-squared: linear, not TM^2/BM
                (tm + bm) // 2,  # arithmetic-mean (floor when odd)
            ]
            candidates = _computed_candidates(text, answer, raw)

    elif tier == "exam":
        variant = rng.choice(["find_bm", "find_am", "centre"])
        if variant == "find_bm":
            d, p, q, am, bm, tm, ab = rng.choice(_TRIPLE_POOL)
            text = phrase(rng, _EXAM_TEMPLATES_BM, tm=tm, ab=ab)
            answer = bm
            check(bm * (bm + ab) == tm * tm, "tangent_secant/exam(find_bm): quadratic failed")
            raw = [
                am,  # other-root-or-other-segment
                2 * bm,  # forgot to halve the root sum
                am + bm,  # took sqrt(discriminant) and stopped (that value is AM + BM here)
            ]
            candidates = _computed_candidates(text, answer, raw)
        elif variant == "find_am":
            d, p, q, am, bm, tm, ab = rng.choice(_TRIPLE_POOL)
            text = phrase(rng, _EXAM_TEMPLATES_AM, tm=tm, ab=ab)
            answer = am
            check(am * (am - ab) == tm * tm, "tangent_secant/exam(find_am): quadratic failed")
            raw = [
                bm,  # other-root-or-other-segment
                2 * am,  # forgot to halve the root sum
                am + bm,  # took sqrt(discriminant) and stopped
            ]
            candidates = _computed_candidates(text, answer, raw)
        else:
            r, tm, mo, bm, am = rng.choice(_PYTHAG_POOL)
            text = phrase(rng, _EXAM_TEMPLATES_CENTRE, r=r, tm=tm)
            answer = mo
            check(tm * tm + r * r == mo * mo, "tangent_secant/exam(centre): Pythagorean check failed")
            check(bm * am == tm * tm, "tangent_secant/exam(centre): power-of-a-point check failed")
            raw = [
                bm,  # pythagoras-slip: BM given as MO
                am,  # pythagoras-slip: AM given as MO
                tm + r,  # linear slip: TM + r instead of sqrt(TM^2 + r^2)
            ]
            candidates = _computed_candidates(text, answer, raw)

    else:
        raise ValueError(f"unknown tier {tier!r}")

    answer_text = str(answer)
    distractors = pick_distractors(answer_text, candidates)
    check_pairwise_distinct(
        answer_text, *distractors, msg=f"tangent_secant/{tier}: final options collided for answer={answer}"
    )
    printed = set(re.findall(r"\d+", text))
    check(
        not (set(distractors) & printed),
        f"tangent_secant/{tier}: a distractor echoed a printed number ({distractors} vs {text!r})",
    )

    working = [
        "By the tangent-secant relation, AM x BM = TM^2.",
        f"So the requested length is {answer} cm.",
    ]

    return {
        "text": text,
        "answer": answer,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
        "working": working,
    }
