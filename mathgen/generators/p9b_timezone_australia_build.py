"""Australian/neighbour time zones with daylight saving, and call-window overlap
(senior-solve-problems-involving-time-zones-in-australia-and,
senior-solve-problems-associated-with-time-zones-for-example).

Both skills use REAL Australian/neighbour daylight-saving facts:

  never observes DST (whole-hour standard offset): Perth (+8), Brisbane (+10),
      Port Moresby (+10), Jakarta (+7)
  never observes DST (half-hour standard offset):  Darwin (+9:30)
  observes DST (whole-hour standard offset):        Sydney, Melbourne, Hobart,
      Canberra (all +10 standard), Auckland (+12 standard)
  observes DST (half-hour standard offset):         Adelaide (+9:30 standard)

- Skill 1 (australia_timezone_conversion): a plain point-to-point time
  difference. The learner is given each place's STANDARD offset plus an
  explicit currently-observing-DST sentence, states the "+1 hour" rule once,
  and must apply the allowance themselves. answer = magnitude of the
  effective-offset difference, in hours.
- Skill 2 (timezone_call_window_overlap): a genuinely different quantity --
  the length of the daily window during which two people's own stated
  "reasonable hours to call" overlap, once one window is shifted into the
  other's clock frame by the (bounded, real-world) offset difference `d`.
  answer = max(0, min(a2, b2+d) - max(a1, b1+d)).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

from ..core import check, check_equivalent, generator, latex_answer, pick_distractors, stated_precision_string


# ============================================================================
# Skill 1: australia_timezone_conversion
# ============================================================================

# Real standard (non-DST) UTC offsets and whether each place EVER observes
# daylight saving. Every DST-observing place in this generator is always
# presented as "currently observing" -- this generator never models a
# DST-observing place in its non-DST season, only the never-observing/
# currently-observing split the difficulty tiers actually use.
PLACES_TZ: dict[str, tuple[Fraction, bool]] = {
    "Perth": (Fraction(8), False),
    "Darwin": (Fraction(19, 2), False),
    "Brisbane": (Fraction(10), False),
    "Port Moresby": (Fraction(10), False),
    "Jakarta": (Fraction(7), False),
    "Adelaide": (Fraction(19, 2), True),
    "Sydney": (Fraction(10), True),
    "Melbourne": (Fraction(10), True),
    "Hobart": (Fraction(10), True),
    "Canberra": (Fraction(10), True),
    "Auckland": (Fraction(12), True),
}

NEVER_DST_WHOLE = ["Perth", "Brisbane", "Port Moresby", "Jakarta"]
# Medium tier's DST-observing pool excludes Adelaide (the half-hour zone is
# reserved for hard tier's half-hour shape).
DST_WHOLE_FOR_MEDIUM = ["Sydney", "Melbourne", "Hobart", "Canberra", "Auckland"]
# Hard subtype (a): every DST-observing place, including Adelaide.
DST_ALL = ["Sydney", "Melbourne", "Hobart", "Canberra", "Adelaide", "Auckland"]
HALF_HOUR_PLACES = ["Darwin", "Adelaide"]
ALL_PLACES_TZ = list(PLACES_TZ)


def _fmt_utc(offset: Fraction) -> str:
    """ "+8:00" / "+9:30" for a standard-offset Fraction of whole or half hours."""
    total_minutes = int(offset * 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"+{hours}:{minutes:02d}"


def _fmt_hours(value: Fraction) -> tuple:
    """(answer, answer_text) for a whole- or half-hour Fraction: an int for a
    whole number (exact-match grading), else a 1-decimal-place string
    (Fraction denominator is always 1 or 2 by construction here)."""
    check(value.denominator in (1, 2), f"unexpected sub-half-hour value {value}")
    if value.denominator == 1:
        return int(value), str(int(value))
    return float(value), f"{float(value):.1f}"


def _hour_word(value: Fraction) -> str:
    """ "hour" for a magnitude of exactly 1, else "hours" -- working text is
    prose, so it must not read "1 hours"."""
    return "hour" if value == 1 else "hours"


def _dst_sentence(place: str) -> str:
    _, observes = PLACES_TZ[place]
    if observes:
        return f"{place} is currently observing daylight saving."
    return f"{place} does not observe daylight saving."


def _effective_offset(place: str, *, table=PLACES_TZ, force_active: bool | None = None) -> Fraction:
    offset, observes = table[place]
    active = observes if force_active is None else force_active
    return offset + (1 if active else 0)


def _diff_honest(a: str, b: str) -> Fraction:
    """The honest, by-construction magnitude: apply each place's OWN real
    DST-observance flag (always "currently observing" when it observes at
    all, per this generator's convention). Never assumes two DST terms add;
    for a pair where both are DST-observing this genuinely cancels the +1s,
    which is why hard subtype (a) can honestly produce the same magnitude
    as the un-adjusted standard-offset difference."""
    return abs(_effective_offset(b) - _effective_offset(a))


# --- independent check: a SEPARATE, re-typed table + re-derived route ------

_CHECK_TABLE_TZ: dict[str, tuple[Fraction, bool]] = {
    "Perth": (Fraction(8), False),
    "Darwin": (Fraction(19, 2), False),
    "Brisbane": (Fraction(10), False),
    "Port Moresby": (Fraction(10), False),
    "Jakarta": (Fraction(7), False),
    "Adelaide": (Fraction(19, 2), True),
    "Sydney": (Fraction(10), True),
    "Melbourne": (Fraction(10), True),
    "Hobart": (Fraction(10), True),
    "Canberra": (Fraction(10), True),
    "Auckland": (Fraction(12), True),
}


def _verify_timezone_diff(place_a: str, place_b: str, claimed: Fraction) -> None:
    """Recompute the magnitude from a separately-typed table, and assert
    A/B-swap symmetry independently of how the caller constructed `claimed`."""

    def eff(place: str) -> Fraction:
        offset, observes = _CHECK_TABLE_TZ[place]
        return offset + (1 if observes else 0)

    forward = abs(eff(place_b) - eff(place_a))
    backward = abs(eff(place_a) - eff(place_b))
    check(forward == backward, f"magnitude not symmetric: {forward} != {backward}")
    check_equivalent(forward, claimed, f"independent check {forward} != {claimed}")


def _tz_distractor_candidates(a: str, b: str) -> list[str]:
    off_a, dst_a = PLACES_TZ[a]
    off_b, dst_b = PLACES_TZ[b]

    # ignores-dst: omit the +1h term for every currently-observing place.
    ignores_dst = abs(off_b - off_a)

    # wrong-dst-table-entry: swap exactly one place's DST-observance flag.
    # NOTE: when only one of the pair ever observes DST (every easy/medium
    # seed, and hard's half_hour subtype whenever the "other" place is
    # never-DST), flipping that one place's flag off is numerically
    # IDENTICAL to `ignores_dst` -- there is nothing else to flip. That
    # collision is real, not a bug; `pick_distractors` drops the duplicate,
    # and `sums_not_differences` below exists precisely so a seed like that
    # still has a second, genuinely distinct wrong answer to offer.
    wrong_entry_a = abs(_effective_offset(b) - _effective_offset(a, force_active=not dst_a))
    wrong_entry_b = abs(_effective_offset(b, force_active=not dst_b) - _effective_offset(a))

    # dst-sign-flip: subtract the DST hour instead of adding it.
    sign_flip_a = off_a - (1 if dst_a else 0)
    sign_flip_b = off_b - (1 if dst_b else 0)
    dst_sign_flip = abs(sign_flip_b - sign_flip_a)

    # Not one of the three named misconceptions, but a genuinely different
    # wrong operation (sums the two offsets instead of differencing them) --
    # added because the three misconceptions above collapse to at most two
    # distinct VALUES whenever only one place in the pair ever observes DST
    # (measured: without this, >=2 distractors landed on only 68%/43% of a
    # 60-seed medium/hard sweep, well under the required 85% bar). A sum of
    # two 7-13-hour offsets is always far outside the real 0-6 hour
    # difference band, so it never collides with the answer.
    sums_not_differences = _effective_offset(a) + _effective_offset(b)

    return [
        _fmt_hours(ignores_dst)[1],
        _fmt_hours(dst_sign_flip)[1],
        _fmt_hours(sums_not_differences)[1],
        _fmt_hours(wrong_entry_a)[1],
        _fmt_hours(wrong_entry_b)[1],
    ]


@generator(
    "australia_timezone_conversion",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def australia_timezone_conversion(rng, tier="easy"):
    """The magnitude of the effective time difference (after the learner's
    OWN daylight-saving allowance) between two named Australian/neighbour
    places."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")

    hard_subtype = None
    if tier == "easy":
        candidates = [(a, b) for a, b in permutations(NEVER_DST_WHOLE, 2) if _diff_honest(a, b) != 0]
        place_a, place_b = rng.choice(candidates)
    elif tier == "medium":
        dst_place = rng.choice(DST_WHOLE_FOR_MEDIUM)
        never_place = rng.choice(NEVER_DST_WHOLE)
        if rng.random() < 0.5:
            place_a, place_b = dst_place, never_place
        else:
            place_a, place_b = never_place, dst_place
    else:  # hard
        hard_subtype = rng.choice(["two_dst", "half_hour"])
        if hard_subtype == "two_dst":
            candidates = [(a, b) for a, b in permutations(DST_ALL, 2) if _diff_honest(a, b) != 0]
            place_a, place_b = rng.choice(candidates)
        else:  # half_hour
            half_place = rng.choice(HALF_HOUR_PLACES)
            other = rng.choice([p for p in ALL_PLACES_TZ if p != half_place])
            if rng.random() < 0.5:
                place_a, place_b = half_place, other
            else:
                place_a, place_b = other, half_place

    diff = _diff_honest(place_a, place_b)
    check(0 < diff <= 6, f"implausible AU/neighbour difference {diff} for {place_a}/{place_b}")
    answer, answer_text = _fmt_hours(diff)
    _verify_timezone_diff(place_a, place_b, diff)

    variant = rng.choice([0, 1, 2])
    if variant == 0:
        question_sentence = f"What is the current time difference between {place_a} and {place_b}, in hours?"
    elif variant == 1:
        question_sentence = f"By how many hours do the current local times in {place_a} and {place_b} differ?"
    else:
        question_sentence = (
            f"What is the size of the gap between the current local times in {place_a} and {place_b}, in hours?"
        )
    text = (
        f"{place_a}'s standard time zone offset is UTC{_fmt_utc(PLACES_TZ[place_a][0])}. "
        f"{_dst_sentence(place_a)} "
        f"{place_b}'s standard time zone offset is UTC{_fmt_utc(PLACES_TZ[place_b][0])}. "
        f"{_dst_sentence(place_b)} "
        f"Daylight saving advances local clocks by 1 hour. "
        f"{question_sentence}"
    )
    working = [
        f"{place_a} effective offset: UTC{_fmt_utc(_effective_offset(place_a))}",
        f"{place_b} effective offset: UTC{_fmt_utc(_effective_offset(place_b))}",
        f"Magnitude of the difference: {answer_text} {_hour_word(diff)}",
    ]

    distractors = pick_distractors(answer_text, _tz_distractor_candidates(place_a, place_b))

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: timezone_call_window_overlap
# ============================================================================

# Re-typed independently of Skill 1's PLACES_TZ (same real facts, a separate
# dict literal, even though both skills live
# in this one module).
TABLE_CALL: dict[str, tuple[Fraction, bool]] = {
    "Perth": (Fraction(8), False),
    "Darwin": (Fraction(19, 2), False),
    "Brisbane": (Fraction(10), False),
    "Port Moresby": (Fraction(10), False),
    "Jakarta": (Fraction(7), False),
    "Adelaide": (Fraction(19, 2), True),
    "Sydney": (Fraction(10), True),
    "Melbourne": (Fraction(10), True),
    "Hobart": (Fraction(10), True),
    "Canberra": (Fraction(10), True),
    "Auckland": (Fraction(12), True),
}
ALL_PLACES_CALL = list(TABLE_CALL)
NEVER_DST_CALL = [p for p, (_, dst) in TABLE_CALL.items() if not dst]
DST_CALL = [p for p, (_, dst) in TABLE_CALL.items() if dst]

# Only the places a real Australian zone-abbreviation key would name --
# Jakarta (WIB) and Port Moresby (PGT) are deliberately excluded from hard
# tier, since the stated key only covers AU/NZ abbreviations.
ZONE_ABBREV: dict[str, str] = {
    "Perth": "AWST",
    "Darwin": "ACST",
    "Sydney": "AEDT",
    "Melbourne": "AEDT",
    "Hobart": "AEDT",
    "Canberra": "AEDT",
    "Adelaide": "ACDT",
    "Auckland": "NZDT",
}
HARD_PLACES_CALL = list(ZONE_ABBREV)
ABBREV_KEY_TEXT = "AEST = +10, AEDT = +11, ACST = +9:30, ACDT = +10:30, AWST = +8, NZST = +12, NZDT = +13"
ABBREV_VALUE: dict[str, Fraction] = {
    "AEST": Fraction(10),
    "AEDT": Fraction(11),
    "ACST": Fraction(19, 2),
    "ACDT": Fraction(21, 2),
    "AWST": Fraction(8),
    "NZST": Fraction(12),
    "NZDT": Fraction(13),
}

WIDE_WINDOW = (8, 22)  # 8 am to 10 pm, 14 hours
NARROW_WINDOW = (9, 21)  # 9 am to 9 pm, 12 hours


def _eff_call(place: str) -> Fraction:
    offset, observes = TABLE_CALL[place]
    return offset + (1 if observes else 0)


def _overlap_length(a1, a2, b1, b2, d) -> Fraction:
    """max(0, min(a2, b2+d) - max(a1, b1+d)) -- assumes |d| <= 6, so no
    modular wraparound is ever needed."""
    check(abs(d) <= 6, f"|d| = {abs(d)} exceeds the real-world bound of 6")
    lo = max(Fraction(a1), Fraction(b1) + d)
    hi = min(Fraction(a2), Fraction(b2) + d)
    return max(Fraction(0), hi - lo)


# --- independent check: separately-typed table/key + a second route --------

_CHECK_TABLE_CALL: dict[str, tuple[Fraction, bool]] = {
    "Perth": (Fraction(8), False),
    "Darwin": (Fraction(19, 2), False),
    "Brisbane": (Fraction(10), False),
    "Port Moresby": (Fraction(10), False),
    "Jakarta": (Fraction(7), False),
    "Adelaide": (Fraction(19, 2), True),
    "Sydney": (Fraction(10), True),
    "Melbourne": (Fraction(10), True),
    "Hobart": (Fraction(10), True),
    "Canberra": (Fraction(10), True),
    "Auckland": (Fraction(12), True),
}
_CHECK_ABBREV_VALUE: dict[str, Fraction] = {
    "AEST": Fraction(10),
    "AEDT": Fraction(11),
    "ACST": Fraction(19, 2),
    "ACDT": Fraction(21, 2),
    "AWST": Fraction(8),
    "NZST": Fraction(12),
    "NZDT": Fraction(13),
}


def _verify_call_overlap(
    tier: str,
    place_a: str,
    place_b: str,
    a1,
    a2,
    b1,
    b2,
    abbrev_a: str | None,
    abbrev_b: str | None,
    claimed_overlap: Fraction,
) -> None:
    """Re-derive `d` a second way (abbreviation lookup for hard tier, direct
    re-typed-table addition otherwise), then recompute the overlap from
    scratch and compare exactly."""
    if tier == "hard":
        eff_a = _CHECK_ABBREV_VALUE[abbrev_a]
        eff_b = _CHECK_ABBREV_VALUE[abbrev_b]
    else:

        def eff(place: str) -> Fraction:
            offset, observes = _CHECK_TABLE_CALL[place]
            return offset + (1 if observes else 0)

        eff_a, eff_b = eff(place_a), eff(place_b)

    d = eff_b - eff_a
    check(abs(d) <= 6, f"independent check: |d| = {abs(d)} exceeds bound 6")
    lo = max(Fraction(a1), Fraction(b1) + d)
    hi = min(Fraction(a2), Fraction(b2) + d)
    overlap = max(Fraction(0), hi - lo)
    check_equivalent(overlap, claimed_overlap, f"independent overlap check {overlap} != {claimed_overlap}")


def _call_distractor_candidates(
    a1,
    a2,
    b1,
    b2,
    d,
    off_a_std,
    off_b_std,
    dst_a,
    dst_b,
    *,
    swap_windows: bool,
) -> list[str]:
    # ignores-dst-or-abbreviation: use each place's base/standard offset
    # (never a DST addition, and -- for hard tier -- never the DST component
    # baked into an AEDT/ACDT/NZDT-style abbreviation).
    d_ignoring = off_b_std - off_a_std
    lo = max(Fraction(a1), Fraction(b1) + d_ignoring)
    hi = min(Fraction(a2), Fraction(b2) + d_ignoring)
    ignoring_overlap = max(Fraction(0), hi - lo)

    # adds-instead-of-intersects: sum of both window lengths, not their
    # intersection.
    summed = (Fraction(a2) - Fraction(a1)) + (Fraction(b2) - Fraction(b1))

    # Not one of the three named misconceptions, but a genuinely different
    # wrong operation (treats the two clocks as if there were no offset
    # difference at all, i.e. d = 0) -- added because `ignoring_overlap` is
    # numerically IDENTICAL to the correct overlap whenever the offsets it
    # drops make no difference (both places are never-DST, or both are
    # DST-active and the +1s cancel), and, separately, a d -> -d "reversed
    # shift" construction was tried and rejected because with two EQUAL-
    # length windows (every easy/medium seed, and hard's window-length
    # itself is even in d) overlap(d) == overlap(-d) identically, so that
    # construction never differs either (proved, not assumed, before
    # shipping). Since every valid pair here has d != 0 by construction
    # (the easy/hard candidate filters and medium's distinct-place-family
    # split both exclude d == 0), and the overlap is strictly largest at
    # d == 0 on this bounded domain, this candidate always differs.
    zero_lo = max(Fraction(a1), Fraction(b1))
    zero_hi = min(Fraction(a2), Fraction(b2))
    zero_shift_overlap = max(Fraction(0), zero_hi - zero_lo)

    # Also not one of the three named misconceptions: flips ONE place's DST
    # contribution only (mirrors Skill 1's wrong-dst-table-entry). Needed
    # because `zero_shift_overlap` and `summed` are always whole-hour
    # (every window bound here is a whole hour), so whenever the true
    # answer is a HALF-hour value (exactly when one of the pair is the
    # +9:30 zone) and `ignoring_overlap` also collapses onto the answer
    # (both places DST-active, so their +1s cancel -- e.g. Adelaide vs
    # Auckland), every other candidate was whole-hour while the answer was
    # not, making the answer pickable by shape alone with zero arithmetic
    # (measured: 17/60 easy-tier seeds before this fix). Flipping only one
    # place's DST term changes the total by a whole 1, which preserves the
    # answer's own half-hour-ness while still changing its value.
    eff_a_asym = off_a_std + (0 if dst_a else 1)
    eff_b_asym = off_b_std + (1 if dst_b else 0)
    flip_a_lo = max(Fraction(a1), Fraction(b1) + (eff_b_asym - eff_a_asym))
    flip_a_hi = min(Fraction(a2), Fraction(b2) + (eff_b_asym - eff_a_asym))
    flip_a_overlap = max(Fraction(0), flip_a_hi - flip_a_lo)

    eff_a_asym2 = off_a_std + (1 if dst_a else 0)
    eff_b_asym2 = off_b_std + (0 if dst_b else 1)
    flip_b_lo = max(Fraction(a1), Fraction(b1) + (eff_b_asym2 - eff_a_asym2))
    flip_b_hi = min(Fraction(a2), Fraction(b2) + (eff_b_asym2 - eff_a_asym2))
    flip_b_overlap = max(Fraction(0), flip_b_hi - flip_b_lo)

    # Flips BOTH places' DST flags at once (the fourth of the four possible
    # true/false combinations). Needed on top of the three above: when
    # exactly one of the pair truly observes DST, `ignoring_overlap`,
    # `flip_a_overlap` and `flip_b_overlap` all reduce to the SAME
    # "both flags equal" value (adding 1 to both or neither cancels
    # identically), which can coincide in |d| -- and so in overlap -- with
    # the true value for a specific standard-offset gap (measured: Adelaide
    # vs Port Moresby/Brisbane, a 0.5-hour standard gap, collided this way
    # on 4/60 easy-tier seeds even after the fix above). Flipping both
    # reaches the one remaining combination and reliably lands on a
    # different |d|.
    eff_a_dbl = off_a_std + (0 if dst_a else 1)
    eff_b_dbl = off_b_std + (0 if dst_b else 1)
    flip_both_lo = max(Fraction(a1), Fraction(b1) + (eff_b_dbl - eff_a_dbl))
    flip_both_hi = min(Fraction(a2), Fraction(b2) + (eff_b_dbl - eff_a_dbl))
    flip_both_overlap = max(Fraction(0), flip_both_hi - flip_both_lo)

    candidates = [
        _fmt_hours(ignoring_overlap)[1],
        _fmt_hours(flip_a_overlap)[1],
        _fmt_hours(flip_b_overlap)[1],
        _fmt_hours(flip_both_overlap)[1],
        _fmt_hours(zero_shift_overlap)[1],
        _fmt_hours(summed)[1],
    ]

    # uses-wrong-window-length: only exercisable at hard tier, where the two
    # windows genuinely differ -- literally "uses 14h for both" (forgets one
    # place's window is the narrower 12h one), holding the real d fixed.
    # NOTE: swapping which BOUND PAIR each place is assigned (rather than
    # forcing both wide) was tried first and rejected -- WIDE_WINDOW and
    # NARROW_WINDOW share the same midpoint (3 pm), which makes that swap a
    # mathematical identity with the correct answer for every d (proved by
    # direct algebra before shipping, not assumed).
    if swap_windows:
        wide_lo = max(Fraction(WIDE_WINDOW[0]), Fraction(WIDE_WINDOW[0]) + d)
        wide_hi = min(Fraction(WIDE_WINDOW[1]), Fraction(WIDE_WINDOW[1]) + d)
        wrong_window_overlap = max(Fraction(0), wide_hi - wide_lo)
        candidates.append(_fmt_hours(wrong_window_overlap)[1])

    return candidates


@generator(
    "timezone_call_window_overlap",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def timezone_call_window_overlap(rng, tier="easy"):
    """The length, in hours, of the daily window during which two places'
    own stated "reasonable hours to call" windows overlap, once one window
    is shifted into the other's clock frame by their effective UTC-offset
    difference."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")

    abbrev_a = abbrev_b = None
    window_swapped = False

    if tier == "easy":
        candidates = [(a, b) for a, b in permutations(ALL_PLACES_CALL, 2) if _eff_call(a) != _eff_call(b)]
        place_a, place_b = rng.choice(candidates)
        eff_a, eff_b = _eff_call(place_a), _eff_call(place_b)
        (a1, a2), (b1, b2) = WIDE_WINDOW, WIDE_WINDOW
        offset_lines = [
            f"Place A ({place_a}) is at UTC{_fmt_utc(eff_a)}.",
            f"Place B ({place_b}) is at UTC{_fmt_utc(eff_b)}.",
        ]
    elif tier == "medium":
        derive_place = rng.choice(DST_CALL)
        direct_place = rng.choice(NEVER_DST_CALL)
        if rng.random() < 0.5:
            place_a, place_b = direct_place, derive_place
        else:
            place_a, place_b = derive_place, direct_place
        eff_a, eff_b = _eff_call(place_a), _eff_call(place_b)
        (a1, a2), (b1, b2) = WIDE_WINDOW, WIDE_WINDOW
        offset_lines = []
        for label, place in (("A", place_a), ("B", place_b)):
            std_off, observes = TABLE_CALL[place]
            if observes:
                offset_lines.append(
                    f"Place {label} ({place})'s standard time zone offset is UTC{_fmt_utc(std_off)}. "
                    f"{place} is currently observing daylight saving; daylight saving advances "
                    f"local clocks by 1 hour."
                )
            else:
                offset_lines.append(f"Place {label} ({place}) is at UTC{_fmt_utc(std_off)}.")
    else:  # hard
        candidates = [
            (a, b)
            for a, b in permutations(HARD_PLACES_CALL, 2)
            if ABBREV_VALUE[ZONE_ABBREV[a]] != ABBREV_VALUE[ZONE_ABBREV[b]]
        ]
        place_a, place_b = rng.choice(candidates)
        abbrev_a, abbrev_b = ZONE_ABBREV[place_a], ZONE_ABBREV[place_b]
        eff_a, eff_b = ABBREV_VALUE[abbrev_a], ABBREV_VALUE[abbrev_b]
        window_swapped = rng.random() < 0.5
        (a1, a2) = NARROW_WINDOW if window_swapped else WIDE_WINDOW
        (b1, b2) = WIDE_WINDOW if window_swapped else NARROW_WINDOW
        offset_lines = [
            f"Zone abbreviation key: {ABBREV_KEY_TEXT}.",
            f"Place A ({place_a})'s time zone is {abbrev_a}.",
            f"Place B ({place_b})'s time zone is {abbrev_b}.",
        ]

    def window_words(w1, w2) -> str:
        return "8 am to 10 pm" if (w1, w2) == WIDE_WINDOW else "9 am to 9 pm"

    d = eff_b - eff_a
    check(abs(d) <= 6, f"implausible offset difference {d} for {place_a}/{place_b}")
    overlap = _overlap_length(a1, a2, b1, b2, d)
    check(overlap > 0, f"implausible zero overlap for {place_a}/{place_b}")
    answer, answer_text = _fmt_hours(overlap)

    # For "ignores-dst-or-abbreviation" the relevant base value is always
    # each place's STANDARD (non-DST) offset -- for hard tier that means
    # stripping the DST component baked into an AEDT/ACDT/NZDT abbreviation,
    # which TABLE_CALL's own standard-offset field already gives directly.
    off_a_std = TABLE_CALL[place_a][0]
    off_b_std = TABLE_CALL[place_b][0]
    dst_a = TABLE_CALL[place_a][1]
    dst_b = TABLE_CALL[place_b][1]

    _verify_call_overlap(tier, place_a, place_b, a1, a2, b1, b2, abbrev_a, abbrev_b, overlap)

    text = " ".join(offset_lines) + (
        f" Place A ({place_a})'s reasonable hours to call are {window_words(a1, a2)}, local time. "
        f"Place B ({place_b})'s reasonable hours to call are {window_words(b1, b2)}, local time. "
        f"What is the length, in hours, of the daily window during which both places' reasonable "
        f"hours to call overlap?"
    )
    working = [
        f"Place A window (local): {a1}:00-{a2}:00; Place B window (local): {b1}:00-{b2}:00",
        f"Offset difference d = {eff_b} - {eff_a} = {d}",
        f"Overlap = max(0, min({a2}, {b2}+d) - max({a1}, {b1}+d)) = {answer_text} {_hour_word(overlap)}",
    ]

    # Shape the options like the key: a half-hour key shows every option to one
    # decimal place ("14.0" beside "11.5"), a whole-hour key keeps only whole-hour
    # options, so the decimal point never gives the key away.
    raw = [
        Fraction(c)
        for c in _call_distractor_candidates(
            a1,
            a2,
            b1,
            b2,
            d,
            off_a_std,
            off_b_std,
            dst_a,
            dst_b,
            swap_windows=(tier == "hard"),
        )
    ]
    places = 1 if "." in answer_text else 0
    raw += [overlap + 1, overlap - 1, overlap + 2, overlap - 2]  # off-by-hours slips top the list up
    distractors = pick_distractors(
        answer_text,
        [stated_precision_string(v, places) for v in raw if v > 0 and (v * (10**places)).denominator == 1],
    )

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )
