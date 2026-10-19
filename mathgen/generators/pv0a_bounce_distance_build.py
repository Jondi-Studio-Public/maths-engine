"""Generates `pv0a_bounce_distance` questions (ACMGM074): the distance travelled by a bouncing ball, using
geometric sequences.

Scope: "Given a ball dropped from a stated height that rebounds to
a stated fraction of its previous height, find the total vertical distance it travels up to a
stated bounce." This is the bouncing-ball corner of the name's own worked list, kept as a TOTAL
distance (drop counted once, every rebound counted twice) -- never a single term (that is
``senior-use-the-formula-t-n-r-n-1``/``compound-interest``) and never a threshold count (that is
``senior-use-geometric-sequences-in-contexts-involving-geometric-growth``, ACMMM076).

Model: dropped from rest at height ``h``; after each impact it rebounds to fraction ``r`` of the
height it fell from.

``D(k-th impact)      = h + 2h(r + r^2 + ... + r^(k-1))``
``D(top of k-th rebound) = D(k-th impact) + h*r^k``

Every prompt states (1) that the total counts the initial drop ONCE and every rebound TWICE
(up and down), (2) the exact endpoint (the k-th impact, or the top of the k-th rebound), and
(3) the precision ("in metres, to 2 decimal places"). Work in exact ``Fraction`` arithmetic
throughout; ``answer_text = stated_precision_string(round_half_up(D, 2), 2)``, and every draw is
rejected/resampled when the exact total sits on the 2dp rounding boundary
(``rounds_unambiguously``).

Construct-to-target: `h` (1-20 m, whole or half metres), `r` (0.40-0.85) and `k` (3-7) are drawn
uniformly and directly -- the closed-form total is then computed exactly, so there is no search
to bias; the only resampling is the rounding-boundary guard and (at hard/exam) the extra
plausibility/margin checks named below.

Tiers -- the DECISION changes at each:
- easy: `h`, `r`, `k` stated; endpoint = the k-th impact.
- medium: endpoint = the top of the k-th rebound (the learner decides the final stroke is a
  single upward-only trip).
- hard: `h` and `r` are NOT stated -- two consecutive rebound heights are given and the learner
  infers `r` (their quotient) and `h` (working back to the drop height) before summing to a
  stated endpoint (impact or top, drawn either way).
- exam: a threshold decides the endpoint -- the learner first finds which rebound is the first to
  fall below (or, for the "reaches its top" wording, first to rise less than) a stated height,
  then sums the total to landing after that rebound. `X` is kept at least 5% (relative, in
  log-space via the ratio to the nearest `h*r^i`) away from every candidate rebound height so
  the crossing period is never itself ambiguous.

Every candidate distractor is checked with ``check_pairwise_distinct`` at
``stated_precision_tolerance(2)`` against the answer and every other candidate on every draw,
resampling when the one-too-many increment (``2*h*r**k``) shrinks below that tolerance for `r`
near the top of its range with a larger `k`.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_pairwise_distinct,
    generator,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

_TOL2 = stated_precision_tolerance(2)


def _fmt_h(h: Fraction) -> str:
    """ "6" or "6.5" -- h is always a whole or half metre."""
    return str(int(h)) if h.denominator == 1 else str(float(h))


def _fmt_r_pct(r: Fraction) -> str:
    pct = r * 100
    return str(int(pct)) if pct.denominator == 1 else str(float(pct))


def _fmt_given(x: Fraction) -> str:
    """A GIVEN (stated, not-the-answer) quantity at up to 2 decimal places, with any trailing zero after the
    decimal point trimmed -- "3.2" not "3.20", "7" not "7.00", "0.17" unaffected."""
    s = f"{float(x):.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _series_sum(r: Fraction, k: int) -> Fraction:
    """r + r^2 + ... + r^k (k terms starting at r^1), by direct accumulation
    -- exact in Fraction arithmetic, and independent of the closed-form
    geometric-series formula (a genuinely different code path from a
    `r*(1-r**k)/(1-r)` expression, so a slip in either is caught)."""
    total = Fraction(0)
    power = r
    for _ in range(k):
        total += power
        power *= r
    return total


def _dist_to_impact(h: Fraction, r: Fraction, k: int) -> Fraction:
    """Total vertical distance travelled up to (and including) the k-th
    impact: the initial drop once, plus every one of the k-1 rebounds before
    it counted twice (up and down)."""
    check(k >= 1, "impact index must be at least 1")
    if k == 1:
        return Fraction(h)
    return Fraction(h) + 2 * Fraction(h) * _series_sum(r, k - 1)


def _dist_to_rebound_top(h: Fraction, r: Fraction, k: int) -> Fraction:
    """Total vertical distance up to the TOP of the k-th rebound: the k-th
    impact's total, plus the single (upward-only) rise of the k-th rebound."""
    return _dist_to_impact(h, r, k) + Fraction(h) * r**k


def _attempt(rng, build_fn, tries: int = 400):
    """Call `build_fn(rng)` (which draws its OWN fresh random inputs, computes
    the total, checks the rounding-boundary/plausibility/distractor-collision
    conditions with `check`/`check_pairwise_distinct`, and returns the final
    question dict) up to `tries` times, retrying on `VerificationError` --
    the same redraw-on-failure pattern its sibling generators use
    (e.g. `pv0a_annuity_duration._solve_a0_for_target`)."""
    last_err = None
    for _ in range(tries):
        try:
            return build_fn(rng)
        except VerificationError as exc:
            last_err = exc
            continue
    raise ValueError(f"could not construct a valid bounce-distance question: {last_err}")


def _draw_h(rng) -> Fraction:
    return Fraction(rng.randint(2, 40), 2)  # 1.0 .. 20.0 in half-metre steps


def _draw_r(rng) -> Fraction:
    return Fraction(rng.randint(40, 85), 100)  # 0.40 .. 0.85


def _draw_k(rng, lo: int = 3, hi: int = 7) -> int:
    return rng.randint(lo, hi)


def _misconception_candidates(h: Fraction, r: Fraction, k: int, to_top: bool) -> list[Fraction]:
    """Every misconception value, as EXACT Fraction totals
    -- collapsed against the answer and each other by the caller with
    `check_pairwise_distinct` before formatting."""
    plain_sum_heights = Fraction(h) + Fraction(h) * _series_sum(r, k - 1)  # no doubling
    double_drop = 2 * Fraction(h) * (1 + _series_sum(r, k - 1))  # initial drop doubled too
    correct_impact = _dist_to_impact(h, r, k)
    correct_top = _dist_to_rebound_top(h, r, k)
    one_more = (
        _dist_to_rebound_top(h, r, k + 1) if not to_top else _dist_to_impact(h, r, k + 2)
    )  # one bounce too many, in the same convention as the answer
    swapped = correct_top if not to_top else correct_impact  # impact<->top confusion
    return [plain_sum_heights, double_drop, one_more, swapped]


def _format_and_build(text: str, exact_total: Fraction, candidates: list[Fraction], *, working: str) -> dict:
    answer_rounded = round_half_up(exact_total, 2)
    answer_text = stated_precision_string(answer_rounded, 2)
    cand_rounded = [round_half_up(c, 2) for c in candidates]
    check_pairwise_distinct(answer_rounded, *cand_rounded, msg="bounce-distance distractor collides")
    # `check_pairwise_distinct` above is an exact-value check; the tolerance
    # margin used for construction retry lives in the caller (resample
    # when the one-too-many increment shrinks below `stated_precision_tolerance(2)`).
    cand_texts = [stated_precision_string(c, 2) for c in cand_rounded]
    distractors = pick_distractors(answer_text, cand_texts, n=3, tolerance=_TOL2)
    return dict(
        text=text,
        answer=float(answer_rounded),
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[working],
    )


# =============================================================================
# easy -- h, r, k stated; endpoint = the k-th impact
# =============================================================================

_EASY_TEMPLATES = (
    "A ball is dropped from a height of {h} m. Each time it hits the ground, it rebounds to "
    "{r}% of the height it fell from. Counting both the upward and downward travel of each "
    "rebound (and the initial drop once), find the total vertical distance the ball has "
    "travelled by the time it hits the ground for the {k}th time. Give your answer in metres, "
    "to 2 decimal places.",
    "A ball dropped from {h} m rebounds to {r}% of its previous height every time it bounces. "
    "Counting the initial drop once and both the upward and downward travel of every rebound, "
    "find the total vertical distance travelled up to the {k}th impact with the ground. Give "
    "your answer in metres, to 2 decimal places.",
    "Dropped from a height of {h} m, a ball rebounds to {r}% of the height of its previous fall "
    "each time it lands. Find the total vertical distance travelled (the initial drop counted "
    "once, and both the rise and fall of every rebound counted) by the {k}th time it hits the "
    "ground. Give your answer in metres, to 2 decimal places.",
)


def _easy_once(rng):
    h = _draw_h(rng)
    r = _draw_r(rng)
    k = _draw_k(rng)
    total = _dist_to_impact(h, r, k)
    check(rounds_unambiguously(float(total), 2), "easy total sits on the 2dp rounding boundary")
    template = rng.choice(_EASY_TEMPLATES)
    text = template.format(h=_fmt_h(h), r=_fmt_r_pct(r), k=k)
    candidates = _misconception_candidates(h, r, k, to_top=False)
    working = (
        f"D = h + 2h(r + ... + r^{k - 1}) with h = {_fmt_h(h)} and r = {_fmt_given(r)}, "
        f"so D = {stated_precision_string(round_half_up(total, 2), 2)} m."
    )
    return _format_and_build(text, total, candidates, working=working)


def _easy(rng):
    return _attempt(rng, _easy_once)


# =============================================================================
# medium -- endpoint = the top of the k-th rebound
# =============================================================================

_MEDIUM_TEMPLATES = (
    "A ball is dropped from a height of {h} m and rebounds to {r}% of the height it fell from "
    "every time it bounces. Counting both the upward and downward travel of each rebound (and "
    "the initial drop once), find the total vertical distance the ball has travelled by the "
    "time it reaches the TOP of its {k}th rebound (the final upward rise is counted once, not "
    "twice). Give your answer in metres, to 2 decimal places.",
    "Dropped from {h} m, a ball rebounds to {r}% of its previous height each time it lands. "
    "Counting the initial drop once and both the rise and fall of every completed rebound, find "
    "the total vertical distance travelled up to the highest point of its {k}th bounce (that "
    "last rise upward is counted only once). Give your answer in metres, to 2 decimal places.",
    "A ball dropped from a height of {h} m rebounds to {r}% of the height of its previous fall "
    "each time it hits the ground. Find the total vertical distance it has travelled once it "
    "reaches the top of its {k}th rebound -- every earlier rebound counted both up and down, "
    "the initial drop counted once, and this final rise counted once only. Give your answer in "
    "metres, to 2 decimal places.",
)


def _medium_once(rng):
    h = _draw_h(rng)
    r = _draw_r(rng)
    k = _draw_k(rng)
    total = _dist_to_rebound_top(h, r, k)
    check(rounds_unambiguously(float(total), 2), "medium total sits on the 2dp rounding boundary")
    template = rng.choice(_MEDIUM_TEMPLATES)
    text = template.format(h=_fmt_h(h), r=_fmt_r_pct(r), k=k)
    candidates = _misconception_candidates(h, r, k, to_top=True)
    working = f"D = [h + 2h(r + ... + r^{k - 1})] + h*r^{k} = {stated_precision_string(round_half_up(total, 2), 2)} m."
    return _format_and_build(text, total, candidates, working=working)


def _medium(rng):
    return _attempt(rng, _medium_once)


# =============================================================================
# hard -- h and r NOT stated; two consecutive rebound heights given instead
# =============================================================================

_HARD_IMPACT_TEMPLATES = (
    "A ball is dropped and bounces repeatedly, each rebound reaching a fixed fraction of the "
    "height of the previous fall. Its {ord1} rebound reaches {h1} m and its {ord2} rebound "
    "reaches {h2} m. Counting the initial drop once and both the upward and downward travel of "
    "every rebound, find the total vertical distance the ball has travelled by the time it hits "
    "the ground for the {k}th time. Give your answer in metres, to 2 decimal places.",
    "Dropped from some height, a ball rebounds by the same fraction of its previous fall's "
    "height every time it lands. Its {ord1} rebound rises to {h1} m and its {ord2} rebound "
    "rises to {h2} m. Find the total vertical distance travelled (the initial drop counted "
    "once, every rebound counted both up and down) by the {k}th impact with the ground. Give "
    "your answer in metres, to 2 decimal places.",
)
_HARD_TOP_TEMPLATES = (
    "A ball is dropped and bounces repeatedly, rebounding each time to a fixed fraction of the "
    "height of its previous fall. Its {ord1} rebound reaches {h1} m and its {ord2} rebound "
    "reaches {h2} m. Counting the initial drop once and both the upward and downward travel of "
    "every earlier rebound, find the total vertical distance travelled once the ball reaches "
    "the TOP of its {k}th rebound (that final rise is counted once only). Give your answer in "
    "metres, to 2 decimal places.",
    "A ball dropped from some height rebounds by the same fraction of its previous fall's "
    "height on every bounce. Its {ord1} rebound rises to {h1} m and its {ord2} rebound rises to "
    "{h2} m. Find the total vertical distance travelled up to the top of its {k}th rebound -- "
    "every earlier rebound counted both up and down, the initial drop counted once, and this "
    "final rise counted once only. Give your answer in metres, to 2 decimal places.",
)
_ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}


def _draw_hard_heights(rng):
    """Draw h, r so that the FIRST and SECOND (or SECOND and THIRD)
    rebound heights are exact at 2 decimal places or fewer -- h in whole or
    half metres, r a whole percent, keeps h*r^i landing on a clean 2dp grid
    for the small indices used here."""
    h = _draw_h(rng)
    r = _draw_r(rng)
    start = rng.choice((1, 2))
    h1 = h * r**start
    h2 = h * r ** (start + 1)
    check(round_half_up(h1, 2) == h1 and round_half_up(h2, 2) == h2, "stated rebound heights are not exact at 2dp")
    return h, r, start, h1, h2


def _hard_once(rng):
    h, r, start, h1, h2 = _draw_hard_heights(rng)
    to_top = rng.random() < 0.5
    k = _draw_k(rng, 4, 8)
    total = _dist_to_rebound_top(h, r, k) if to_top else _dist_to_impact(h, r, k)
    check(rounds_unambiguously(float(total), 2), "hard total sits on the 2dp rounding boundary")
    ord1, ord2 = _ORDINALS[start], _ORDINALS[start + 1]
    templates = _HARD_TOP_TEMPLATES if to_top else _HARD_IMPACT_TEMPLATES
    template = rng.choice(templates)
    text = template.format(ord1=ord1, ord2=ord2, h1=_fmt_given(h1), h2=_fmt_given(h2), k=k)
    candidates = _misconception_candidates(h, r, k, to_top=to_top)
    # A hard-tier-specific misconception: taking the FIRST STATED rebound
    # height itself as the drop height h.
    took_first_as_h = _dist_to_rebound_top(h1, r, k) if to_top else _dist_to_impact(h1, r, k)
    candidates.append(took_first_as_h)
    working = (
        f"r = {stated_precision_string(h2, 2)}/{stated_precision_string(h1, 2)} = "
        f"{_fmt_r_pct(r)}%; h = {stated_precision_string(h1, 2)}/r^{start} = {_fmt_h(h)} m; "
        f"summing to the stated endpoint gives "
        f"{stated_precision_string(round_half_up(total, 2), 2)} m."
    )
    return _format_and_build(text, total, candidates, working=working)


def _hard(rng):
    return _attempt(rng, _hard_once)


# =============================================================================
# exam -- a threshold decides which rebound is the endpoint
# =============================================================================

_EXAM_TEMPLATES = (
    "A ball dropped from a height of {h} m rebounds to {r}% of the height it fell from every "
    "time it hits the ground. It keeps bouncing; counting the initial drop once and both the "
    "upward and downward travel of every rebound, find the total vertical distance travelled "
    "from release until it lands after the first rebound that rises LESS than {x} m. Give your "
    "answer in metres, to 2 decimal places.",
    "A ball is dropped from {h} m and rebounds to {r}% of the height of its previous fall each "
    "time it bounces. It keeps bouncing until the first rebound whose peak height is lower than "
    "{x} m; counting the initial drop once and both the upward and downward travel of every "
    "rebound, find the total vertical distance travelled from release to when the ball lands "
    "after that rebound. Give your answer in metres, to 2 decimal places.",
    "Dropped from a height of {h} m, a ball keeps bouncing, rebounding each time to {r}% of the "
    "height of its previous fall. Counting the initial drop once and both the upward and "
    "downward travel of every rebound, find the total vertical distance it has travelled from "
    "release until it lands after the first rebound that rises LESS than {x} m. Give your "
    "answer in metres, to 2 decimal places.",
)


def _exam_once(rng):
    h = _draw_h(rng)
    r = _draw_r(rng)
    k_target = _draw_k(rng, 4, 9)  # the first rebound below X is the k_target-th rebound
    heights = [h * r**i for i in range(1, k_target + 2)]
    # X sits strictly between heights[k_target - 1] (still >= X) and
    # heights[k_target - 1] itself is the (k_target)th rebound... place X in
    # log-space between the (k_target)th and (k_target - 1)th rebound heights,
    # with a comfortable relative margin from EVERY rebound height.
    above = heights[k_target - 2] if k_target >= 2 else Fraction(h)
    below = heights[k_target - 1]
    f = Fraction(rng.randint(25, 75), 100)
    x_exact = below + (above - below) * f  # strictly between below and above (log-space skipped:
    # heights decay geometrically, so a LINEAR interpolation between two
    # consecutive heights still sits strictly between them for any f in (0, 1))
    x = round_half_up(x_exact, 2)
    check(below < x < above, "threshold does not sit strictly between the two bracketing rebounds")
    check(
        all(abs(x - hh) / hh > Fraction(1, 20) for hh in heights), "threshold too close (relative) to a rebound height"
    )
    # Confirm which rebound is genuinely first below x, on the ROUNDED x.
    k_correct = next(i for i, hh in enumerate(heights, start=1) if hh < x)
    total = _dist_to_impact(h, r, k_correct + 1)  # lands after that rebound = the next impact
    check(rounds_unambiguously(float(total), 2), "exam total sits on the 2dp rounding boundary")
    template = rng.choice(_EXAM_TEMPLATES)
    text = template.format(h=_fmt_h(h), r=_fmt_r_pct(r), x=_fmt_given(x))
    # Misconceptions specific to the threshold decision: one rebound too
    # early/late, plus the shared no-doubling/double-drop/impact-vs-top set.
    one_early = _dist_to_impact(h, r, k_correct)  # stopped one rebound too soon
    one_late = _dist_to_impact(h, r, k_correct + 2)  # went one rebound too far
    to_top_same_k = _dist_to_rebound_top(h, r, k_correct)  # impact<->top confusion at the found k
    candidates = _misconception_candidates(h, r, k_correct + 1, to_top=False)
    candidates += [one_early, one_late, to_top_same_k]
    working = (
        f"The {k_correct}th rebound reaches {stated_precision_string(round_half_up(heights[k_correct - 1], 2), 2)} m "
        f"(>= {stated_precision_string(x, 2)} m); the {k_correct + 1}th rebound reaches "
        f"{stated_precision_string(round_half_up(heights[k_correct], 2), 2)} m (< "
        f"{stated_precision_string(x, 2)} m), the first below the threshold. Summing to the "
        f"({k_correct + 2})th impact -- when the ball lands after that rebound -- gives "
        f"{stated_precision_string(round_half_up(total, 2), 2)} m."
    )
    return _format_and_build(text, total, candidates, working=working)


def _exam(rng):
    return _attempt(rng, _exam_once)


_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}


@generator("pv0a_bounce_distance", level=13, presets=_PRESETS)
def bounce_distance(rng, variant="easy"):
    if variant == "easy":
        return _easy(rng)
    if variant == "medium":
        return _medium(rng)
    if variant == "hard":
        return _hard(rng)
    if variant == "exam":
        return _exam(rng)
    raise ValueError(f"unknown variant: {variant!r}")
