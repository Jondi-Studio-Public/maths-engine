"""Generates `p19a_regroup_place_value` questions for `f10-regroup-zero-place-value` (AC9M2N02): restate a two-
or three-digit number from a non-standard grouping of hundreds, tens and ones.

Scope: "Given a two- or three-digit number expressed in a
non-standard grouping (e.g. as a stated count of hundreds, tens and ones that
is not already in standard form, including a grouping that implies a zero
digit in one place), regroup or rename it into its standard hundreds, tens
and ones counts."

## Answer shape

`answer: pair`, a LABELLED dict ``{"hundreds": H, "tens": T, "ones": O}``,
``answer_text`` ``"hundreds = H, tens = T, ones = O"``. Because every label
appears exactly once and the labels match on both sides,
`mathgen/grading/typed.py::_labelled_pair_verdict` accepts the three segments in ANY
order, plus a bare positional triple ("H, T, O") once
`_split_labelled_segments` fails to find a leading label on the learner's
side and the grader falls back to its positional comparison (same order as
`answer`'s own dict, which is always hundreds/tens/ones) -- and a run-on
numeral ("HTO" concatenated, e.g. "304") is rejected because it extracts to
a single value, never three. See the module's own build test for the direct
probe against `mathgen.grading.GRADERS["pair"]`.

## The three tiers (genuine reasoning escalation, never a bigger number)

- **easy**: the stated grouping names ONLY tens and ones (hundreds never
  mentioned) and exactly ONE of the two overflows past 9 -- a single carry.
  DECISION: which one place overflows, and carry it once.
- **medium**: the stated grouping still never names hundreds, but the stated
  tens count is now itself always >= 10 -- the hundreds place is introduced
  from nothing, never named in the prompt at all, and the stated ones may
  independently overflow too. DECISION: recognise an UNSTATED place must be
  created (medium's own new decision, unavailable to easy, where the total
  never exceeded 99 unless the single tens-overflow branch happened to fire).
- **hard**: hundreds, tens AND ones are all named explicitly, sized so a
  ones-place carry can itself push the stated tens over 9 and cascade into
  the hundreds place (a genuine second carry step no lower tier requires),
  and the three places are sometimes listed OUT OF ORDER in the prompt text
  -- the learner must resolve place value from the WORD attached to each
  count, not its position in the sentence.

Both a "zero-digit" result (the regrouped tens or ones digit lands on exactly
0) and a three-digit result appear at every tier -- see the build test's
200-seed instruments for the measured shares; construction deliberately
biases at least 35% of draws toward the zero-digit case (`want_zero` below),
which only ever RAISES the realised rate above the 25% floor, never
below it, since the un-forced branch is free to land on zero anyway.

## Independent check

`_check_regroup` reconstructs the ORIGINAL stated total two ways: once from
the literal stated groupings (built during the draw, `total`) and once from
the candidate regrouped digits (`H*100 + T*10 + O`) -- a genuinely different
route from how `H`, `T`, `O` were computed (`total // 100`, `(total // 10) %
10`, `total % 10`), via `check_equivalent`. `check_pairwise_distinct` runs
over the (H, T, O) tuples of the answer and all three distractors, never the
free-form display text.
"""

from __future__ import annotations

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
)

TOPIC = "p19a_regroup_place_value"
_TIERS = ("easy", "medium", "hard")
_RETRY_LIMIT = 500

_NAMES = ("Mia", "Sam", "Ali", "Zoe", "Ben", "Leo", "Ivy", "Tom")

_FORMAT_NOTE = " Give your answer as hundreds = _, tens = _, ones = _."


def _word(n: int, sg: str, pl: str) -> str:
    return sg if n == 1 else pl


_EASY_PATTERNS = (
    "A number is made of {tens} {tens_w} and {ones} {ones_w}. What are its hundreds, tens and ones?" + _FORMAT_NOTE,
    "{name} has {tens} {tens_w} and {ones} {ones_w}. Regroup this into hundreds, tens and ones." + _FORMAT_NOTE,
    "{tens} {tens_w} and {ones} {ones_w} make a number. Write its hundreds, tens and ones." + _FORMAT_NOTE,
)
_MEDIUM_PATTERNS = (
    "A number is made of {tens} {tens_w} and {ones} {ones_w}, with no "
    "hundreds stated. What are its hundreds, tens and ones?" + _FORMAT_NOTE,
    "{name} counted {tens} {tens_w} and {ones} {ones_w}, and no hundreds at "
    "all. Regroup this into hundreds, tens and ones." + _FORMAT_NOTE,
    "{tens} {tens_w} and {ones} {ones_w} make a number, with no hundreds "
    "given. Write its hundreds, tens and ones." + _FORMAT_NOTE,
)
_HARD_PATTERNS_ORDERED = (
    "A number is made of {hundreds} {hundreds_w}, {tens} {tens_w} and {ones} "
    "{ones_w}. What are its hundreds, tens and ones after regrouping?" + _FORMAT_NOTE,
    "{name} counted {hundreds} {hundreds_w}, {tens} {tens_w} and {ones} "
    "{ones_w}. Regroup this into hundreds, tens and ones." + _FORMAT_NOTE,
)
_HARD_PATTERNS_SHUFFLED = (
    "A number is made of {ones} {ones_w}, {hundreds} {hundreds_w} and {tens} "
    "{tens_w}. What are its hundreds, tens and ones after regrouping?" + _FORMAT_NOTE,
    "{name} counted {tens} {tens_w}, {ones} {ones_w} and {hundreds} "
    "{hundreds_w}. Regroup this into hundreds, tens and ones." + _FORMAT_NOTE,
)


def _decompose(total: int) -> tuple[int, int, int]:
    return total // 100, (total // 10) % 10, total % 10


def _check_regroup(total: int, H: int, T: int, O: int) -> None:
    """Independent route: reconstruct the total from the CANDIDATE digits
    (H*100 + T*10 + O) and confirm it matches the total built from the
    literal stated groupings -- the two never share an arithmetic step."""
    check(H * 100 + T * 10 + O == total, "p19a_regroup_place_value: regroup mismatch")
    check(0 <= T <= 9 and 0 <= O <= 9, "p19a_regroup_place_value: digit out of range")


def _carry_dropped(H: int, T: int, O: int) -> tuple[int, int, int]:
    """Misconception: performs the regroup but forgets to add the final
    carry into the highest place it reached -- decrement whichever place is
    highest and nonzero."""
    if H > 0:
        return (H - 1, T, O)
    if T > 0:
        return (H, T - 1, O)
    return (H, T, max(O - 1, 0))


def _carry_wrong_place(H: int, T: int, O: int) -> tuple[int, int, int]:
    """Misconception: a tens-to-hundreds carry is applied an extra time (an
    extra hundred appears while only one unit leaves the tens place)."""
    return (H + 1, max(T - 1, 0), O)


def _answer_text(H: int, T: int, O: int) -> str:
    return f"hundreds = {H}, tens = {T}, ones = {O}"


def _finalise(text: str, H: int, T: int, O: int, raw_candidates: list[tuple[int, int, int]]) -> dict:
    seen_text = {_answer_text(H, T, O)}
    distractors = []
    for cand in raw_candidates:
        txt = _answer_text(*cand)
        if txt in seen_text:
            continue
        seen_text.add(txt)
        distractors.append(txt)
    check(len(distractors) >= 2, "p19a_regroup_place_value: not enough surviving distractors")
    answer_text = _answer_text(H, T, O)
    return dict(
        text=text,
        answer={"hundreds": H, "tens": T, "ones": O},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=distractors,
        working=[f"Total: {H * 100 + T * 10 + O}.", f"Answer: {answer_text}."],
    )


# --- easy: single-place overflow, hundreds never named --------------------


def _draw_easy(rng):
    want_zero = rng.random() < 0.35
    for _ in range(_RETRY_LIMIT):
        kind = rng.choice(("ones_overflow", "tens_overflow"))
        if kind == "ones_overflow":
            stated_tens = rng.randint(1, 8)
            stated_ones = rng.randint(10, 18)
        else:
            stated_tens = rng.randint(10, 18)
            stated_ones = rng.randint(0, 9)
        total = stated_tens * 10 + stated_ones
        H, T, O = _decompose(total)
        if want_zero and T != 0 and O != 0:
            continue
        return dict(stated_tens=stated_tens, stated_ones=stated_ones, total=total, H=H, T=T, O=O, kind=kind)
    check(False, "p19a_regroup_place_value easy: could not draw a valid case")


def _build_easy(rng):
    for _ in range(_RETRY_LIMIT):
        d = _draw_easy(rng)
        total, H, T, O = d["total"], d["H"], d["T"], d["O"]
        _check_regroup(total, H, T, O)
        m1 = (0, d["stated_tens"], d["stated_ones"])
        m2 = _carry_dropped(H, T, O)
        m3 = _carry_wrong_place(H, T, O)
        try:
            check_pairwise_distinct(
                (H, T, O), m1, m2, m3, msg="easy: a distractor collides with the answer or another option"
            )
        except Exception:
            continue
        name = rng.choice(_NAMES)
        text = phrase(
            rng,
            _EASY_PATTERNS,
            tens=d["stated_tens"],
            ones=d["stated_ones"],
            name=name,
            tens_w=_word(d["stated_tens"], "ten", "tens"),
            ones_w=_word(d["stated_ones"], "one", "ones"),
        )
        return _finalise(text, H, T, O, [m1, m2, m3])
    check(False, "p19a_regroup_place_value easy: could not draw distinct distractors")


# --- medium: hundreds always introduced from nothing -----------------------


def _draw_medium(rng):
    want_zero = rng.random() < 0.35
    for _ in range(_RETRY_LIMIT):
        stated_tens = rng.randint(10, 30)
        stated_ones = rng.randint(0, 19)
        total = stated_tens * 10 + stated_ones
        if total > 999:
            continue
        H, T, O = _decompose(total)
        if want_zero and T != 0 and O != 0:
            continue
        return dict(stated_tens=stated_tens, stated_ones=stated_ones, total=total, H=H, T=T, O=O)
    check(False, "p19a_regroup_place_value medium: could not draw a valid case")


def _build_medium(rng):
    for _ in range(_RETRY_LIMIT):
        d = _draw_medium(rng)
        total, H, T, O = d["total"], d["H"], d["T"], d["O"]
        _check_regroup(total, H, T, O)
        m1 = (0, d["stated_tens"], d["stated_ones"])
        m2 = _carry_dropped(H, T, O)
        m3 = _carry_wrong_place(H, T, O)
        try:
            check_pairwise_distinct(
                (H, T, O), m1, m2, m3, msg="medium: a distractor collides with the answer or another option"
            )
        except Exception:
            continue
        name = rng.choice(_NAMES)
        text = phrase(
            rng,
            _MEDIUM_PATTERNS,
            tens=d["stated_tens"],
            ones=d["stated_ones"],
            name=name,
            tens_w=_word(d["stated_tens"], "ten", "tens"),
            ones_w=_word(d["stated_ones"], "one", "ones"),
        )
        return _finalise(text, H, T, O, [m1, m2, m3])
    check(False, "p19a_regroup_place_value medium: could not draw distinct distractors")


# --- hard: hundreds explicit, cascading carry, places sometimes shuffled ---


def _draw_hard(rng):
    want_zero = rng.random() < 0.35
    for _ in range(_RETRY_LIMIT):
        stated_hundreds = rng.randint(0, 6)
        stated_tens = rng.randint(7, 9)
        stated_ones = rng.randint(10, 19)
        total = stated_hundreds * 100 + stated_tens * 10 + stated_ones
        if total > 999:
            continue
        H, T, O = _decompose(total)
        if want_zero and T != 0 and O != 0:
            continue
        return dict(
            stated_hundreds=stated_hundreds,
            stated_tens=stated_tens,
            stated_ones=stated_ones,
            total=total,
            H=H,
            T=T,
            O=O,
        )
    check(False, "p19a_regroup_place_value hard: could not draw a valid case")


def _build_hard(rng):
    for _ in range(_RETRY_LIMIT):
        d = _draw_hard(rng)
        total, H, T, O = d["total"], d["H"], d["T"], d["O"]
        _check_regroup(total, H, T, O)
        m1 = (d["stated_hundreds"], d["stated_tens"], d["stated_ones"])
        m2 = _carry_dropped(H, T, O)
        m3 = _carry_wrong_place(H, T, O)
        try:
            check_pairwise_distinct(
                (H, T, O), m1, m2, m3, msg="hard: a distractor collides with the answer or another option"
            )
        except Exception:
            continue
        name = rng.choice(_NAMES)
        shuffled = rng.random() < 0.5
        patterns = _HARD_PATTERNS_SHUFFLED if shuffled else _HARD_PATTERNS_ORDERED
        text = phrase(
            rng,
            patterns,
            hundreds=d["stated_hundreds"],
            tens=d["stated_tens"],
            ones=d["stated_ones"],
            name=name,
            hundreds_w=_word(d["stated_hundreds"], "hundred", "hundreds"),
            tens_w=_word(d["stated_tens"], "ten", "tens"),
            ones_w=_word(d["stated_ones"], "one", "ones"),
        )
        return _finalise(text, H, T, O, [m1, m2, m3])
    check(False, "p19a_regroup_place_value hard: could not draw distinct distractors")


_BUILDERS = {
    "easy": _build_easy,
    "medium": _build_medium,
    "hard": _build_hard,
}


@generator(
    TOPIC,
    level=2,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p19a_regroup_place_value(rng, tier: str = "easy"):
    """Regroup a two- or three-digit number stated in a non-standard grouping
    (a count of hundreds/tens/ones not already in standard form) into its
    standard hundreds, tens and ones counts. `tier` selects one of the three
    genuinely different decisions (see module docstring)."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {_TIERS}")
    return _BUILDERS[tier](rng)
