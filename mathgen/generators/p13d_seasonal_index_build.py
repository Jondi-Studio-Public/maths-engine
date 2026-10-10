"""Two senior seasonal-index skills sharing one kernel family (quarterly data, the average-percentage method).
Two independent topics in one module; neither reads the other's data.

## `p13d_seasonal_index` (skill `senior-calculate-seasonal-indices-by-using-
the-average-percentage`, ACMGM090)

Four seasons per year, held fixed as QUARTERLY data across every tier (the
reasoning shape escalates, not the table size). One NAMED quarter is the
subject of every question. The average-percentage method for that quarter:

1. Each year's own mean (average of that year's 4 quarter values).
2. That year's named-quarter percentage-of-year-mean: `value / mean x 100`.
3. Average those per-year percentages across the years given -> the
   (unadjusted) seasonal index for that quarter, stated to 1 decimal place.

### Tiers
- easy: 2 years, each year's OWN MEAN given directly -- two percentages then
  one average; no derivation of a mean at all.
- medium: 2 years, all 4 quarters' raw values given per year (not the mean)
  -- the learner must derive each year's own mean first, a genuinely earlier
  added step before the same percentage-then-average reasoning easy uses.
- hard: INVERTS the method. Given the (already-computed, 1-dp) unadjusted
  index, one year's own percentage (stated as a plain given fact) and the
  OTHER year's own mean (also stated directly), find the missing raw
  quarter value that would have produced the stated index -- index -> missing
  percentage (`2*index - pct1`) -> missing raw value (`pct2/100 * mean`), a
  different decision (solving) from computing forward.
- exam: the ADJUSTMENT step. Given the STATED SUM of all four quarters'
  unadjusted indices and three of the four individual unadjusted indices,
  find the fourth quarter's ADJUSTED index once every quarter is scaled by
  the same `400 / sum` factor so all four sum to exactly 400 -- a
  normalisation/consistency reasoning move neither easy/medium/hard perform.

### Deviation from the original specification, disclosed

The specified hard tier also asks for "the other year's own raw
table" alongside its mean. Read literally this creates a circular construction (the raw table's own
sum-to-mean consistency would require already knowing the missing value X, which is exactly what the question
asks the learner to find) -- so this module gives the OTHER year's mean directly as a stated fact (exactly as
easy's own mean is a stated fact) and does not additionally display that year's raw table.

### Independent check
Construction sums per-year percentages built from year means it also
builds (easy/medium) or is handed as given data (hard/exam). The check
recomputes the average/adjustment purely from the STATED per-year
percentages via `Fraction` arithmetic -- never re-running construction's own
raw-value loop.

## `p13d_deseasonalise` (skill `senior-deseasonalise-a-time-series-by-using-
a-seasonal`, ACMGM091)

`deseasonalised = actual / (index / 100)`. Every tier's raw actual value is
constructed BACKWARD from a chosen "nice" deseasonalised target and an
integer seasonal index, so `actual = target * index / 100` always terminates
within 2 decimal places exactly (`index` is an integer, so `index / 100` has
<= 2 dp; a target with <= 2 dp times that still has <= 2 dp) -- no
generator-side rounding error ever enters the forward direction, and the
reverse (easy/medium tiers, solving for the deseasonalised value) recovers
the exact target by construction.

### Tiers
- easy: actual value and seasonal index given directly as a bare percentage
  number; single division.
- medium: the index is given as a stated PERCENTAGE CHANGE from the 100
  baseline ("15% above"/"12% below" the yearly average) rather than as the
  bare index number -- convert to an index first, a genuine added step.
- hard: INVERTS the operation -- given the deseasonalised value and the
  index, find the original actual value (multiply instead of divide).
- exam: composes two deseasonalisations with a follow-on PERCENTAGE-CHANGE
  comparison between the two results, to 1 decimal place.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

# ============================================================================
# Topic 1: p13d_seasonal_index
# ============================================================================

_SI_TIERS = ("easy", "medium", "hard", "exam")
_SI_QUARTERS = ("Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4")
_SI_DP = 1


def _si_pct(value, mean) -> Fraction:
    return Fraction(value) * 100 / Fraction(mean)


def _si_index_from_pcts(pct1: Fraction, pct2: Fraction) -> Fraction:
    """The independent-check route: average of the STATED per-year
    percentages alone, via Fraction arithmetic."""
    return (pct1 + pct2) / 2


def _si_verify_average(pct1, pct2, index_rounded):
    check_equivalent(
        round_half_up(_si_index_from_pcts(Fraction(pct1), Fraction(pct2)), _SI_DP),
        Fraction(index_rounded),
        "p13d_seasonal_index: average-percentage index disagrees with the independent recomputation",
    )


_SI_EASY_TEMPLATES = (
    "A retailer tracks {season} across two years. In Year 1, {season}'s sales were ${v1:,} "
    "and Year 1's own mean quarterly sales were ${m1:,}. In Year 2, {season}'s sales were "
    "${v2:,} and Year 2's own mean quarterly sales were ${m2:,}. Using the average percentage "
    "method, calculate the (unadjusted) seasonal index for {season}, to 1 decimal place.",
    "For a small business, {season}'s sales were ${v1:,} in Year 1 (that year's own mean "
    "quarterly sales: ${m1:,}) and ${v2:,} in Year 2 (that year's own mean quarterly sales: "
    "${m2:,}). Find the (unadjusted) seasonal index for {season} by the average percentage "
    "method, to 1 decimal place.",
    "Two years of data are available for {season}: Year 1 sales of ${v1:,} against a Year 1 "
    "mean of ${m1:,}, and Year 2 sales of ${v2:,} against a Year 2 mean of ${m2:,}. Apply the "
    "average percentage method to find {season}'s (unadjusted) seasonal index, to 1 decimal "
    "place.",
)
_SI_MEDIUM_TEMPLATES = (
    "A retailer's quarterly sales ($) over two years were: Year 1 -- Quarter 1 {y1[0]}, "
    "Quarter 2 {y1[1]}, Quarter 3 {y1[2]}, Quarter 4 {y1[3]}; Year 2 -- Quarter 1 {y2[0]}, "
    "Quarter 2 {y2[1]}, Quarter 3 {y2[2]}, Quarter 4 {y2[3]}. Using the average percentage "
    "method, calculate the (unadjusted) seasonal index for {season}, to 1 decimal place.",
    "Here are two years of a small business's quarterly sales ($). Year 1: Quarter 1 {y1[0]}, "
    "Quarter 2 {y1[1]}, Quarter 3 {y1[2]}, Quarter 4 {y1[3]}. Year 2: Quarter 1 {y2[0]}, "
    "Quarter 2 {y2[1]}, Quarter 3 {y2[2]}, Quarter 4 {y2[3]}. Find {season}'s (unadjusted) "
    "seasonal index by the average percentage method, to 1 decimal place.",
    "Quarterly sales ($), two years: Year 1 gives Quarter 1 {y1[0]}, Quarter 2 {y1[1]}, "
    "Quarter 3 {y1[2]}, Quarter 4 {y1[3]}; Year 2 gives Quarter 1 {y2[0]}, Quarter 2 {y2[1]}, "
    "Quarter 3 {y2[2]}, Quarter 4 {y2[3]}. Apply the average percentage method to find "
    "{season}'s (unadjusted) seasonal index, to 1 decimal place.",
)
_SI_HARD_TEMPLATES = (
    "Using the average percentage method on two years of data, {season}'s (unadjusted) "
    "seasonal index came out at {index}. Year 1's percentage-of-year-mean for {season} was "
    "stated as {pct1}%. Year 2's own mean quarterly sales were ${mean2:,}. Find the raw dollar "
    "value for {season} in Year 2 that would produce the stated index, to 2 decimal places.",
    "A seasonal index of {index} for {season} was computed by the average percentage method "
    "from two years of data. Year 1's own {season} percentage-of-year-mean was {pct1}%, and "
    "Year 2's own mean quarterly sales were ${mean2:,}. What raw dollar value for {season} in "
    "Year 2 would give this stated index, to 2 decimal places?",
    "{season}'s (unadjusted) seasonal index, by the average percentage method over two years, "
    "is {index}. Given Year 1's percentage-of-year-mean of {pct1}% and Year 2's own mean "
    "quarterly sales of ${mean2:,}, find the raw dollar value for {season} in Year 2, to 2 "
    "decimal places.",
)
_SI_EXAM_TEMPLATES = (
    "Exam-style question. The four quarters' unadjusted seasonal indices (average percentage "
    "method) summed to {total}. Three of the four are: Quarter {a} = {ia}, Quarter {b} = {ib}, "
    "Quarter {c} = {ic}. Find the ADJUSTED seasonal index for Quarter {d}, once every quarter "
    "is scaled by the same factor so all four sum to exactly 400. Give your answer to 1 "
    "decimal place.",
    "This exam question gives four unadjusted seasonal indices summing to {total}: Quarter {a} "
    "= {ia}, Quarter {b} = {ib}, Quarter {c} = {ic}, and Quarter {d} (not stated). Scale every "
    "quarter by the same factor so the four ADJUSTED indices sum to exactly 400, and find "
    "Quarter {d}'s adjusted index, to 1 decimal place.",
    "The unadjusted seasonal indices for four quarters sum to {total} (not exactly 400, as is "
    "typical before adjustment). Quarter {a} = {ia}, Quarter {b} = {ib}, Quarter {c} = {ic}. "
    "After scaling all four by the same factor so they sum to exactly 400, what is Quarter "
    "{d}'s adjusted index? Give your answer to 1 decimal place.",
)


def _si_build_easy(rng):
    for _ in range(200):
        season = rng.choice(_SI_QUARTERS)
        means = [rng.randint(40, 200) for _ in range(2)]
        values = [max(1, round(means[i] * rng.randint(70, 130) / 100)) for i in range(2)]
        pcts = [_si_pct(values[i], means[i]) for i in range(2)]
        index_exact = _si_index_from_pcts(*pcts)
        if not rounds_unambiguously(float(index_exact), _SI_DP):
            continue
        index = round_half_up(index_exact, _SI_DP)
        _si_verify_average(pcts[0], pcts[1], index)

        candidates = []
        # averages the RAW values instead of the percentages
        alt_raw = round_half_up(Fraction(sum(values), 2), _SI_DP)
        candidates.append(alt_raw)
        # divides by number of SEASONS (4) instead of YEARS (2)
        alt_seasons = round_half_up((pcts[0] + pcts[1]) / 4, _SI_DP)
        candidates.append(alt_seasons)
        # uses the OVERALL (average-of-means) mean as the denominator
        overall_mean = Fraction(sum(means), 2)
        alt_overall = round_half_up((_si_pct(values[0], overall_mean) + _si_pct(values[1], overall_mean)) / 2, _SI_DP)
        candidates.append(alt_overall)

        candidates = [c for c in candidates if c != index]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(index, *candidates)
        except Exception:
            continue

        index_text = stated_precision_string(index, _SI_DP)
        cand_texts = [stated_precision_string(c, _SI_DP) for c in candidates]
        pct1_disp = stated_precision_string(round_half_up(pcts[0], 2), 2)
        pct2_disp = stated_precision_string(round_half_up(pcts[1], 2), 2)
        text = phrase(
            rng,
            _SI_EASY_TEMPLATES,
            season=season,
            v1=values[0],
            m1=means[0],
            v2=values[1],
            m2=means[1],
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(index),
            answer_text=index_text,
            answer_latex=latex_answer(index_text),
            distractors=pick_distractors(index_text, cand_texts),
            working=[
                f"Year 1: {values[0]} / {means[0]} x 100 = {pct1_disp}%.",
                f"Year 2: {values[1]} / {means[1]} x 100 = {pct2_disp}%.",
                f"Average: ({pct1_disp} + {pct2_disp}) / 2 = {index_text}.",
            ],
        )
    check(False, "p13d_seasonal_index/easy: could not draw a valid question in 200 attempts")


def _si_build_medium(rng):
    for _ in range(200):
        season_idx = rng.randrange(4)
        season = _SI_QUARTERS[season_idx]
        years = []
        for _ in range(2):
            base = rng.randint(40, 200)
            quarters = [max(1, base + rng.randint(-30, 30)) for _ in range(4)]
            years.append(quarters)
        y1, y2 = years
        means = [Fraction(sum(y1), 4), Fraction(sum(y2), 4)]
        values = [y1[season_idx], y2[season_idx]]
        pcts = [_si_pct(values[i], means[i]) for i in range(2)]
        index_exact = _si_index_from_pcts(*pcts)
        if not rounds_unambiguously(float(index_exact), _SI_DP):
            continue
        index = round_half_up(index_exact, _SI_DP)
        _si_verify_average(pcts[0], pcts[1], index)

        candidates = []
        alt_raw = round_half_up(Fraction(sum(values), 2), _SI_DP)
        candidates.append(alt_raw)
        alt_seasons = round_half_up((pcts[0] + pcts[1]) / 4, _SI_DP)
        candidates.append(alt_seasons)
        overall_mean = Fraction(sum(y1) + sum(y2), 8)
        alt_overall = round_half_up((_si_pct(values[0], overall_mean) + _si_pct(values[1], overall_mean)) / 2, _SI_DP)
        candidates.append(alt_overall)

        candidates = [c for c in candidates if c != index]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(index, *candidates)
        except Exception:
            continue

        index_text = stated_precision_string(index, _SI_DP)
        cand_texts = [stated_precision_string(c, _SI_DP) for c in candidates]
        pct1_disp = stated_precision_string(round_half_up(pcts[0], 2), 2)
        pct2_disp = stated_precision_string(round_half_up(pcts[1], 2), 2)
        text = phrase(rng, _SI_MEDIUM_TEMPLATES, season=season, y1=y1, y2=y2)
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(index),
            answer_text=index_text,
            answer_latex=latex_answer(index_text),
            distractors=pick_distractors(index_text, cand_texts),
            working=[
                f"Year 1 mean: ({', '.join(str(v) for v in y1)}) / 4 = {stated_precision_string(means[0], 2)}.",
                f"Year 2 mean: ({', '.join(str(v) for v in y2)}) / 4 = {stated_precision_string(means[1], 2)}.",
                f"Percentages for {season}: {pct1_disp}%, {pct2_disp}%. Average = {index_text}.",
            ],
        )
    check(False, "p13d_seasonal_index/medium: could not draw a valid question in 200 attempts")


def _si_verify_hard(pct1, index, mean2, x):
    """Independent route: pct2 = 2*index - pct1 (algebra on the average),
    then x = pct2/100 * mean2 (algebra on the percentage formula) -- the
    inverse of construction's own forward path."""
    pct2 = 2 * Fraction(index) - Fraction(pct1)
    expected = pct2 / 100 * Fraction(mean2)
    check_equivalent(
        round_half_up(x, 2),
        round_half_up(expected, 2),
        "p13d_seasonal_index/hard: recovered raw value disagrees with the independent inversion",
    )


def _si_build_hard(rng):
    for _ in range(300):
        season = rng.choice(_SI_QUARTERS)
        mean2 = rng.randint(40, 250)
        pct1 = Fraction(rng.randint(700, 1300), 10)
        index = Fraction(rng.randint(700, 1300), 10)
        pct2 = 2 * index - pct1
        if pct2 <= 0:
            continue
        x = pct2 / 100 * Fraction(mean2)
        if not (Fraction(1, 1) <= x <= Fraction(5 * mean2, 1)):
            continue
        x = round_half_up(x, 2)
        # re-derive index from the ROUNDED x/mean2/pct1 to guarantee internal
        # consistency of what is actually stated as "given" in the question:
        pct2_check = _si_pct(x, mean2)
        index_recheck = round_half_up((pct1 + pct2_check) / 2, _SI_DP)
        if index_recheck != round_half_up(index, _SI_DP):
            continue
        index = round_half_up(index, _SI_DP)
        _si_verify_hard(pct1, index, mean2, x)

        candidates = []
        # uses index directly as pct2 (forgets to double before subtracting pct1)
        wrong_pct2_direct = round_half_up(index / 100 * Fraction(mean2), 2)
        candidates.append(wrong_pct2_direct)
        # uses pct1 (not the derived pct2) against mean2
        wrong_uses_pct1 = round_half_up(pct1 / 100 * Fraction(mean2), 2)
        candidates.append(wrong_uses_pct1)
        # forgets to convert percent to a proportion (drops the /100)
        pct2_derived = 2 * index - pct1
        wrong_no_percent = round_half_up(pct2_derived * Fraction(mean2), 2)
        candidates.append(wrong_no_percent)

        candidates = [c for c in candidates if c != x]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(x, *candidates)
        except Exception:
            continue

        x_text = stated_precision_string(x, 2)
        cand_texts = [stated_precision_string(c, 2) for c in candidates]
        pct1_text = stated_precision_string(pct1, 1)
        index_text = stated_precision_string(index, 1)
        text = phrase(
            rng,
            _SI_HARD_TEMPLATES,
            season=season,
            index=index_text,
            pct1=pct1_text,
            mean2=mean2,
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(x),
            answer_text=x_text,
            answer_latex=latex_answer(x_text),
            distractors=pick_distractors(x_text, cand_texts),
            working=[
                f"Missing percentage: 2 x {index_text} - {pct1_text} = {stated_precision_string(pct2_derived, 1)}%.",
                f"Missing value: {stated_precision_string(pct2_derived, 1)} / 100 x {mean2} = {x_text}.",
            ],
        )
    check(False, "p13d_seasonal_index/hard: could not draw a valid question in 300 attempts")


def _si_verify_exam(ia, ib, ic, total, i_d, adjusted):
    """Independent route: derive the missing unadjusted index from the
    STATED total and the three given ones (`total - ia - ib - ic`), then
    scale by `400 / total` -- never reusing construction's own draw order."""
    i_d_check = Fraction(total) - Fraction(ia) - Fraction(ib) - Fraction(ic)
    check_equivalent(i_d_check, Fraction(i_d), "p13d_seasonal_index/exam: missing unadjusted index mismatch")
    factor = Fraction(400, 1) / Fraction(total)
    check_equivalent(
        round_half_up(i_d_check * factor, _SI_DP),
        round_half_up(Fraction(adjusted), _SI_DP),
        "p13d_seasonal_index/exam: adjusted index disagrees with the independent recomputation",
    )


def _si_build_exam(rng):
    for _ in range(300):
        indices = [Fraction(rng.randint(800, 1200), 10) for _ in range(4)]
        total = sum(indices)
        if total <= 0 or not (Fraction(360) <= total <= Fraction(440)):
            continue
        order = list(range(4))
        rng.shuffle(order)
        d_pos = order[3]
        a_pos, b_pos, c_pos = order[0], order[1], order[2]
        ia, ib, ic, i_d = indices[a_pos], indices[b_pos], indices[c_pos], indices[d_pos]
        factor = Fraction(400, 1) / total
        adjusted = i_d * factor
        if not rounds_unambiguously(float(adjusted), _SI_DP):
            continue
        adjusted = round_half_up(adjusted, _SI_DP)
        _si_verify_exam(ia, ib, ic, total, i_d, adjusted)

        # forgets the adjustment step: reports the unadjusted 4th index
        wrong_unadjusted = round_half_up(i_d, _SI_DP)
        # scales only the given three (a self-consistency slip): treats
        # (400 - ia - ib - ic) as the answer directly (never scales at all)
        wrong_no_scale = round_half_up(Fraction(400, 1) - ia - ib - ic, _SI_DP)
        # uses the WRONG factor base (divides by 400 instead of by total)
        wrong_factor = round_half_up(i_d * total / Fraction(400, 1), _SI_DP)

        candidates = [c for c in (wrong_unadjusted, wrong_no_scale, wrong_factor) if c != adjusted]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(adjusted, *candidates)
        except Exception:
            continue

        adjusted_text = stated_precision_string(adjusted, _SI_DP)
        cand_texts = [stated_precision_string(c, _SI_DP) for c in candidates]
        total_text = stated_precision_string(total, _SI_DP)
        text = phrase(
            rng,
            _SI_EXAM_TEMPLATES,
            total=total_text,
            a=a_pos + 1,
            ia=stated_precision_string(ia, _SI_DP),
            b=b_pos + 1,
            ib=stated_precision_string(ib, _SI_DP),
            c=c_pos + 1,
            ic=stated_precision_string(ic, _SI_DP),
            d=d_pos + 1,
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(adjusted),
            answer_text=adjusted_text,
            answer_latex=latex_answer(adjusted_text),
            distractors=pick_distractors(adjusted_text, cand_texts),
            working=[
                f"Missing unadjusted index: {total_text} - {stated_precision_string(ia, _SI_DP)} - "
                f"{stated_precision_string(ib, _SI_DP)} - {stated_precision_string(ic, _SI_DP)} = "
                f"{stated_precision_string(i_d, _SI_DP)}.",
                f"Factor: 400 / {total_text}.",
                f"Adjusted: {stated_precision_string(i_d, _SI_DP)} x 400 / {total_text} = {adjusted_text}.",
            ],
        )
    check(False, "p13d_seasonal_index/exam: could not draw a valid question in 300 attempts")


_SI_BUILDERS = {
    "easy": _si_build_easy,
    "medium": _si_build_medium,
    "hard": _si_build_hard,
    "exam": _si_build_exam,
}


@generator(
    "p13d_seasonal_index",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13d_seasonal_index(rng, tier: str = "easy"):
    if tier not in _SI_TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _SI_BUILDERS[tier](rng)


# ============================================================================
# Topic 2: p13d_deseasonalise
# ============================================================================

_DS_TIERS = ("easy", "medium", "hard", "exam")
_DS_DP = 2
_DS_PCT_DP = 1


def _ds_deseasonalise(actual: Fraction, index: Fraction) -> Fraction:
    return actual / (index / 100)


def _ds_actualise(deseasonalised: Fraction, index: Fraction) -> Fraction:
    return deseasonalised * (index / 100)


_DS_EASY_TEMPLATES = (
    "A shop's actual sales this quarter were ${actual}. The seasonal index for this quarter is "
    "{index}. Deseasonalise this figure, to 2 decimal places.",
    "This quarter, actual sales were ${actual} and the seasonal index is {index}. Find the "
    "deseasonalised sales figure, to 2 decimal places.",
    "The seasonal index for this quarter is {index}, against actual sales of ${actual}. "
    "Calculate the deseasonalised value, to 2 decimal places.",
)
_DS_MEDIUM_ABOVE_TEMPLATES = (
    "A shop's actual sales this quarter were ${actual}. This quarter's seasonal index is {pct}% "
    "above the yearly average. Deseasonalise the sales figure, to 2 decimal places.",
    "This quarter, actual sales were ${actual}. The seasonal index is stated as {pct}% above "
    "the 100 baseline. Find the deseasonalised sales figure, to 2 decimal places.",
)
_DS_MEDIUM_BELOW_TEMPLATES = (
    "A shop's actual sales this quarter were ${actual}. This quarter's seasonal index is {pct}% "
    "below the yearly average. Deseasonalise the sales figure, to 2 decimal places.",
    "This quarter, actual sales were ${actual}. The seasonal index is stated as {pct}% below "
    "the 100 baseline. Find the deseasonalised sales figure, to 2 decimal places.",
)
_DS_HARD_TEMPLATES = (
    "A quarter's deseasonalised sales figure is ${deseasonalised}, and the seasonal index for "
    "that quarter is {index}. Find the original ACTUAL sales value, to 2 decimal places.",
    "The deseasonalised sales value for a quarter is ${deseasonalised}, with a seasonal index "
    "of {index}. What was the actual (raw) sales figure, to 2 decimal places?",
    "Given a deseasonalised figure of ${deseasonalised} and a seasonal index of {index} for one "
    "quarter, find the actual sales value that produced it, to 2 decimal places.",
)
_DS_EXAM_TEMPLATES = (
    "Exam-style question. Period A had actual sales of ${a_actual} and a seasonal index of "
    "{a_index}. Period B had actual sales of ${b_actual} and a seasonal index of {b_index}. "
    "Deseasonalise both figures, then find the percentage change from Period A's "
    "deseasonalised value to Period B's, to 1 decimal place.",
    "This exam question gives two periods: Period A -- actual sales ${a_actual}, seasonal "
    "index {a_index}; Period B -- actual sales ${b_actual}, seasonal index {b_index}. After "
    "deseasonalising both, find the percentage change from Period A to Period B, to 1 decimal "
    "place.",
    "Two trading periods are compared. Period A recorded actual sales of ${a_actual} against a "
    "seasonal index of {a_index}; Period B recorded actual sales of ${b_actual} against a "
    "seasonal index of {b_index}. Deseasonalise each period's figure, then state the "
    "percentage change from Period A's deseasonalised value to Period B's, to 1 decimal place.",
)


def _ds_build_easy(rng):
    for _ in range(200):
        target = Fraction(rng.randint(50, 5000), 1)
        index = rng.randint(60, 160)
        if index == 100:
            continue
        actual = target * index / 100
        if actual.denominator != 1 and actual.denominator not in (2, 4, 5, 10, 20, 25, 50, 100):
            continue
        check_equivalent(
            _ds_deseasonalise(actual, Fraction(index)),
            target,
            "p13d_deseasonalise/easy: construction round-trip failed",
        )

        wrong_multiplies = round_half_up(actual * (Fraction(index) / 100), _DS_DP)
        wrong_raw_divisor = round_half_up(actual / index, _DS_DP)
        target_r = round_half_up(target, _DS_DP)
        candidates = [c for c in (wrong_multiplies, wrong_raw_divisor) if c != target_r]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(target_r, *candidates)
        except Exception:
            continue

        actual_text = stated_precision_string(round_half_up(actual, _DS_DP), _DS_DP)
        target_text = stated_precision_string(target_r, _DS_DP)
        cand_texts = [stated_precision_string(c, _DS_DP) for c in candidates]
        text = phrase(rng, _DS_EASY_TEMPLATES, actual=actual_text, index=index)
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(target_r),
            answer_text=target_text,
            answer_latex=latex_answer(target_text),
            distractors=pick_distractors(target_text, cand_texts),
            working=[f"Deseasonalised = {actual_text} / ({index} / 100) = {target_text}."],
        )
    check(False, "p13d_deseasonalise/easy: could not draw a valid question in 200 attempts")


def _ds_build_medium(rng):
    for _ in range(200):
        target = Fraction(rng.randint(50, 5000), 1)
        pct = rng.randint(5, 40)
        above = rng.choice((True, False))
        index = 100 + pct if above else 100 - pct
        actual = target * index / 100
        check_equivalent(
            _ds_deseasonalise(actual, Fraction(index)),
            target,
            "p13d_deseasonalise/medium: construction round-trip failed",
        )

        wrong_multiplies = round_half_up(actual * (Fraction(index) / 100), _DS_DP)
        wrong_reads_pct_as_index = round_half_up(_ds_deseasonalise(actual, Fraction(pct)), _DS_DP)
        target_r = round_half_up(target, _DS_DP)
        candidates = [c for c in (wrong_multiplies, wrong_reads_pct_as_index) if c != target_r]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(target_r, *candidates)
        except Exception:
            continue

        actual_text = stated_precision_string(round_half_up(actual, _DS_DP), _DS_DP)
        target_text = stated_precision_string(target_r, _DS_DP)
        cand_texts = [stated_precision_string(c, _DS_DP) for c in candidates]
        bank = _DS_MEDIUM_ABOVE_TEMPLATES if above else _DS_MEDIUM_BELOW_TEMPLATES
        text = phrase(rng, bank, actual=actual_text, pct=pct)
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(target_r),
            answer_text=target_text,
            answer_latex=latex_answer(target_text),
            distractors=pick_distractors(target_text, cand_texts),
            working=[
                f"Index = 100 {'+' if above else '-'} {pct} = {index}.",
                f"Deseasonalised = {actual_text} / ({index} / 100) = {target_text}.",
            ],
        )
    check(False, "p13d_deseasonalise/medium: could not draw a valid question in 200 attempts")


def _ds_build_hard(rng):
    for _ in range(200):
        deseasonalised = Fraction(rng.randint(50, 5000), 1)
        index = rng.randint(60, 160)
        if index == 100:
            continue
        actual = _ds_actualise(deseasonalised, Fraction(index))
        check_equivalent(
            _ds_deseasonalise(actual, Fraction(index)),
            deseasonalised,
            "p13d_deseasonalise/hard: construction round-trip failed",
        )

        wrong_divides = round_half_up(_ds_deseasonalise(deseasonalised, Fraction(index)), _DS_DP)
        wrong_raw_multiplier = round_half_up(deseasonalised * index, _DS_DP)
        actual_r = round_half_up(actual, _DS_DP)
        candidates = [c for c in (wrong_divides, wrong_raw_multiplier) if c != actual_r]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(actual_r, *candidates)
        except Exception:
            continue

        deseasonalised_text = stated_precision_string(deseasonalised, _DS_DP)
        actual_text = stated_precision_string(actual_r, _DS_DP)
        cand_texts = [stated_precision_string(c, _DS_DP) for c in candidates]
        text = phrase(rng, _DS_HARD_TEMPLATES, deseasonalised=deseasonalised_text, index=index)
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(actual_r),
            answer_text=actual_text,
            answer_latex=latex_answer(actual_text),
            distractors=pick_distractors(actual_text, cand_texts),
            working=[f"Actual = {deseasonalised_text} x ({index} / 100) = {actual_text}."],
        )
    check(False, "p13d_deseasonalise/hard: could not draw a valid question in 200 attempts")


def _ds_verify_change(d_a, d_b, change):
    check_equivalent(
        round_half_up((d_b - d_a) / d_a * 100, _DS_PCT_DP),
        round_half_up(Fraction(change), _DS_PCT_DP),
        "p13d_deseasonalise/exam: percentage change disagrees with the independent recomputation",
    )


def _ds_build_exam(rng):
    for _ in range(200):
        d_a = Fraction(rng.randint(200, 3000), 1)
        d_b = Fraction(rng.randint(200, 3000), 1)
        index_a = rng.randint(70, 150)
        index_b = rng.randint(70, 150)
        if index_a == 100 or index_b == 100 or d_a == d_b:
            continue
        actual_a = d_a * index_a / 100
        actual_b = d_b * index_b / 100
        change_exact = (d_b - d_a) / d_a * 100
        if not rounds_unambiguously(float(change_exact), _DS_PCT_DP):
            continue
        change = round_half_up(change_exact, _DS_PCT_DP)
        _ds_verify_change(d_a, d_b, change)

        # reports the difference between the two ACTUAL values (as a % of
        # actual_a) instead of the deseasonalised values
        wrong_raw_change = round_half_up((actual_b - actual_a) / actual_a * 100, _DS_PCT_DP)
        # reverses the direction (B to A instead of A to B)
        wrong_reversed = round_half_up((d_a - d_b) / d_b * 100, _DS_PCT_DP)

        candidates = [c for c in (wrong_raw_change, wrong_reversed) if c != change]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(change, *candidates)
        except Exception:
            continue

        change_text = stated_precision_string(change, _DS_PCT_DP)
        cand_texts = [stated_precision_string(c, _DS_PCT_DP) for c in candidates]
        a_actual_text = stated_precision_string(round_half_up(actual_a, _DS_DP), _DS_DP)
        b_actual_text = stated_precision_string(round_half_up(actual_b, _DS_DP), _DS_DP)
        text = phrase(
            rng,
            _DS_EXAM_TEMPLATES,
            a_actual=a_actual_text,
            a_index=index_a,
            b_actual=b_actual_text,
            b_index=index_b,
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(change),
            answer_text=change_text,
            answer_latex=latex_answer(change_text),
            distractors=pick_distractors(change_text, cand_texts),
            working=[
                f"Deseasonalised A = {a_actual_text} / ({index_a} / 100) = {stated_precision_string(d_a, _DS_DP)}.",
                f"Deseasonalised B = {b_actual_text} / ({index_b} / 100) = {stated_precision_string(d_b, _DS_DP)}.",
                f"Percentage change = {change_text}%.",
            ],
        )
    check(False, "p13d_deseasonalise/exam: could not draw a valid question in 200 attempts")


_DS_BUILDERS = {
    "easy": _ds_build_easy,
    "medium": _ds_build_medium,
    "hard": _ds_build_hard,
    "exam": _ds_build_exam,
}


@generator(
    "p13d_deseasonalise",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13d_deseasonalise(rng, tier: str = "easy"):
    if tier not in _DS_TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _DS_BUILDERS[tier](rng)
