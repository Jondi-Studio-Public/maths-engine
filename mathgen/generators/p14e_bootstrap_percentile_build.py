"""Generates `p14e_bootstrap_percentile` questions (`ucv1-bootstrap-summary`): a bootstrap summary of stated
resamples, answered as a pair.

``ucv1-bootstrap-summary``, built by this module (topic ``p14e_bootstrap_percentile``, answer: pair).

Kernel: a
supplied finite bootstrap resample of ``n`` integers is graded as a pair --
(mean, one STATED percentile endpoint) -- the prompt always names which
percentile is wanted (varying low/high across seeds/tiers so both ends of a
bootstrap interval are exercised across the corpus), and the mean is graded
alongside it every single question.

Percentile convention, stated verbatim in every prompt: "the k-th smallest of
these n values is the (100k/n)-th percentile" -- an exact order-statistic
convention, never interpolated. `n` and the requested percentile are always
chosen so `k = percentile * n / 100` is an exact integer.

Boundary against `calculate-mean`: that skill computes ONLY the mean of a raw list -- no percentile, no
resampling framing, no sorting requirement, a bare `number` answer. This skill ALWAYS also asks for a stated
percentile endpoint (a `pair` answer) of a list explicitly framed as a bootstrap resample, never a plain data
set -- a learner could never mistake one question for the other, since `calculate-mean`'s prompt never
mentions a percentile or the word "resample" at all.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import GRADER_TOLERANCE, check, generator, latex_answer, phrase, rational_text

N_EASY = 10
N_STD = 20

_CONVENTION = (
    "Convention: the k-th smallest of these {n} values is its (100k/{n})-th "
    "percentile, with no interpolation between values."
)

_EASY_TEMPLATES = (
    "A bootstrap resample of n={n} values, already sorted from smallest to "
    "largest, is: {resample}. " + _CONVENTION + " Find the mean of the resample and its {p}th percentile.",
    "The following n={n} values are a bootstrap resample, listed already in "
    "sorted order: {resample}. " + _CONVENTION + " Report the resample's mean and its {p}th percentile.",
    "A statistician draws a bootstrap resample of n={n} values from the "
    "original data; sorted from smallest to largest, it reads: {resample}. "
    + _CONVENTION
    + " Compute the mean and the {p}th percentile.",
)

_UNSORTED_TEMPLATES = (
    "A bootstrap resample of n={n} values (not sorted) is: {resample}. "
    + _CONVENTION
    + " Find the mean of the resample and its {p}th percentile.",
    "The following n={n} values form a bootstrap resample, listed in the "
    "order they were drawn (not sorted): {resample}. "
    + _CONVENTION
    + " Report the resample's mean and its {p}th percentile.",
    "A statistician draws a bootstrap resample of n={n} values, recorded here "
    "in draw order rather than sorted order: {resample}. "
    + _CONVENTION
    + " Compute the mean and the {p}th percentile.",
)

_EXAM_TEMPLATES = (
    "A bootstrap resample of n={n} values (not sorted) is: {resample}. "
    + _CONVENTION
    + " For a {coverage}% bootstrap interval, find the resample mean and the "
    "{which} endpoint of the interval.",
    "The following n={n} values form a bootstrap resample, listed in draw "
    "order (not sorted): {resample}. "
    + _CONVENTION
    + " A {coverage}% bootstrap interval is required; report the resample's "
    "mean and the interval's {which} endpoint.",
    "A statistician draws a bootstrap resample of n={n} values, recorded here "
    "in draw order rather than sorted order: {resample}. "
    + _CONVENTION
    + " Compute the mean and the {which} endpoint of a {coverage}% bootstrap "
    "interval.",
)

# --- shared order-statistic mechanics ---------------------------------


def _kth_smallest(sorted_values: list[int], k: int) -> int:
    """1-indexed order statistic: the k-th smallest of `sorted_values`."""
    return sorted_values[k - 1]


def _toward_middle(k: int, n: int) -> int:
    """Off-by-one index toward the middle of [1, n] -- never out of range."""
    return k + 1 if k <= n // 2 else k - 1


def _label(p: int) -> str:
    return f"p{p}"


def _mean_text(values: list[int]) -> tuple[Fraction, str]:
    mean = Fraction(sum(values), len(values))
    return mean, rational_text(mean)


def _pair_text(mean_text: str, label: str, endpoint) -> str:
    return f"mean = {mean_text}, {label} = {endpoint}"


def _pair_latex(mean: Fraction, label: str, endpoint) -> str:
    """Real LaTeX for the (mean, endpoint) pair -- `answer_text`'s own
    `"mean = X, {label} = Y"` shape fed through `latex_answer` unmodified
    mangles; both
    words are wrapped in `\\text{}` so they render upright instead, and the
    numeric mean still goes through `latex_answer` for a real `\\frac{}{}`."""
    return rf"\text{{mean}} = {latex_answer(str(mean))},\; \text{{{label}}} = {endpoint}"


def _components_differ(mean_a: Fraction, end_a, mean_b: Fraction, end_b) -> bool:
    """Whether two (mean, endpoint) pairs are a genuinely different answer as
    the real `grade_pair` sees it: the endpoint compared exactly, the mean
    compared with the pair grader's own tolerance (`GRADER_TOLERANCE`).

    Candidate shared helper: pair_components_differ(a, b, tolerances) -- a general
    per-component tolerance-aware distinctness check for `answer: pair`
    skills, mirroring `check_pairwise_distinct` but for a pair's own
    (possibly mixed exact/tolerant) components -- would serve every `pair`
    skill built after this one, not just this generator. Inlined here as the
    minimum needed.
    """
    if int(end_a) != int(end_b):
        return True
    return abs(mean_a - mean_b) > GRADER_TOLERANCE


def _verify_and_build(
    original_values: list[int],
    displayed_sorted: bool,
    p: int,
    n: int,
    label: str,
) -> tuple[Fraction, str, int, list[int]]:
    """Independent in-generator check: recompute the mean
    via a direct `Fraction(sum, n)` (never reusing any running total the
    caller used) and recompute the requested order statistic by fully
    re-sorting the ORIGINAL (pre-shuffle) list with Python's own `sorted()`,
    re-deriving `k` from the stated percentile -- never reusing the
    caller's own sorted copy or `k` value directly."""
    n_actual = len(original_values)
    check(n_actual == n, "resample size mismatch")
    resorted = sorted(original_values)
    k = p * n // 100
    check(p * n % 100 == 0, "percentile does not land on an exact order statistic")
    endpoint = _kth_smallest(resorted, k)
    mean = Fraction(sum(original_values), n)
    return mean, resorted, endpoint, k


# --- easy: n=10, already sorted, extreme-ish percentile ----------------


def _easy(rng) -> dict:
    n = N_EASY
    values = sorted(rng.sample(range(0, 60), n))
    p = rng.choice([10, 90])
    mean, resorted, endpoint, k = _verify_and_build(values, True, p, n, _label(p))
    check(resorted == values, "easy resample was not already sorted")
    mean_check = Fraction(sum(values), n)
    check(mean == mean_check, "easy mean independent recheck mismatch")

    mean_text = rational_text(mean)
    label = _label(p)
    answer_text = _pair_text(mean_text, label, endpoint)

    candidates = []
    # off-by-one-index: same mean, wrong endpoint.
    off_k = _toward_middle(k, n)
    off_endpoint = _kth_smallest(resorted, off_k)
    if _components_differ(mean, endpoint, mean, off_endpoint):
        candidates.append(_pair_text(mean_text, label, off_endpoint))
    # wrong-tail: reports the (100-p)-th percentile's value instead.
    tail_k = n - k
    if 1 <= tail_k <= n:
        tail_endpoint = _kth_smallest(resorted, tail_k)
        if _components_differ(mean, endpoint, mean, tail_endpoint):
            candidates.append(_pair_text(mean_text, label, tail_endpoint))
    # mean-off-by-a-value: drops the value farthest from the mean, same endpoint.
    dropped = max(values, key=lambda v: abs(Fraction(v) - mean))
    wrong_mean = Fraction(sum(values) - dropped, n - 1)
    if _components_differ(mean, endpoint, wrong_mean, endpoint):
        candidates.append(_pair_text(rational_text(wrong_mean), label, endpoint))

    distractors = _dedupe(answer_text, candidates, mean, endpoint)
    resample_str = ", ".join(str(v) for v in values)
    text = phrase(rng, _EASY_TEMPLATES, n=n, resample=resample_str, p=p)
    return dict(
        text=text,
        answer={"mean": mean, "endpoint": endpoint},
        answer_text=answer_text,
        answer_latex=_pair_latex(mean, label, endpoint),
        working=[
            f"Mean = ({' + '.join(str(v) for v in values)}) / {n} = {mean_text}.",
            f"k = {p} × {n} / 100 = {k}; the {k}-th smallest value is {endpoint}, so {label} = {endpoint}.",
        ],
        distractors=distractors,
    )


# --- medium/hard: n=20, unsorted -----------------------------------------


def _shuffled_breaking_position(rng, values: list[int], k: int) -> list[int]:
    """A shuffle of `values` such that position k-1 (0-indexed) differs from
    the sorted list's own value there -- so `forgot-to-sort` is always
    genuinely exercisable, never left to chance."""
    sorted_values = sorted(values)
    target = sorted_values[k - 1]
    for _ in range(200):
        shuffled = list(values)
        rng.shuffle(shuffled)
        if shuffled[k - 1] != target:
            return shuffled
    # Fall back: force it directly by swapping with a differing position.
    shuffled = list(values)
    rng.shuffle(shuffled)
    for i, v in enumerate(shuffled):
        if v != target:
            shuffled[k - 1], shuffled[i] = shuffled[i], shuffled[k - 1]
            break
    check(shuffled[k - 1] != target, "could not construct a sort-breaking shuffle")
    return shuffled


def _medium_or_hard(rng, force_duplicate: bool) -> dict:
    n = N_STD
    p = rng.choice([25, 75])
    k = p * n // 100
    if force_duplicate:
        base = sorted(rng.sample(range(0, 90), n - 1))
        # Force a duplicate pair away from k-1's immediate neighbourhood so
        # off-by-one/wrong-tail stay live.
        forbidden = {k - 1, max(0, k - 2), min(n - 1, k)}
        candidates_idx = [i for i in range(len(base)) if i not in forbidden and i + 1 not in forbidden]
        dup_source_idx = rng.choice(candidates_idx) if candidates_idx else 0
        values = base[: dup_source_idx + 1] + [base[dup_source_idx]] + base[dup_source_idx + 1 :]
    else:
        values = sorted(rng.sample(range(0, 90), n))

    unsorted_values = _shuffled_breaking_position(rng, values, k)

    if force_duplicate:
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        check(max(counts.values()) >= 2, "hard resample missing a forced duplicate")

    mean, resorted, endpoint, k_recheck = _verify_and_build(unsorted_values, False, p, n, _label(p))
    check(k_recheck == k, "hard/medium recomputed k mismatch")
    check(resorted == sorted(unsorted_values), "resorted mismatch")

    mean_text = rational_text(mean)
    label = _label(p)
    answer_text = _pair_text(mean_text, label, endpoint)

    candidates = []
    off_k = _toward_middle(k, n)
    off_endpoint = _kth_smallest(resorted, off_k)
    if _components_differ(mean, endpoint, mean, off_endpoint):
        candidates.append(_pair_text(mean_text, label, off_endpoint))

    tail_k = n - k
    if 1 <= tail_k <= n:
        tail_endpoint = _kth_smallest(resorted, tail_k)
        if _components_differ(mean, endpoint, mean, tail_endpoint):
            candidates.append(_pair_text(mean_text, label, tail_endpoint))

    # forgot-to-sort: index into the ORIGINAL unsorted list at position k.
    forgot_endpoint = unsorted_values[k - 1]
    if _components_differ(mean, endpoint, mean, forgot_endpoint):
        candidates.append(_pair_text(mean_text, label, forgot_endpoint))

    dropped = max(unsorted_values, key=lambda v: abs(Fraction(v) - mean))
    wrong_mean = Fraction(sum(unsorted_values) - dropped, n - 1)
    if _components_differ(mean, endpoint, wrong_mean, endpoint):
        candidates.append(_pair_text(rational_text(wrong_mean), label, endpoint))

    distractors = _dedupe(answer_text, candidates, mean, endpoint)
    resample_str = ", ".join(str(v) for v in unsorted_values)
    text = phrase(rng, _UNSORTED_TEMPLATES, n=n, resample=resample_str, p=p)
    return dict(
        text=text,
        answer={"mean": mean, "endpoint": endpoint},
        answer_text=answer_text,
        answer_latex=_pair_latex(mean, label, endpoint),
        working=[
            f"Sorted resample: {', '.join(str(v) for v in resorted)}.",
            f"Mean = (sum of all {n} values) / {n} = {mean_text}.",
            f"k = {p} × {n} / 100 = {k}; the {k}-th smallest sorted value is {endpoint}, so {label} = {endpoint}.",
        ],
        distractors=distractors,
    )


def _medium(rng) -> dict:
    return _medium_or_hard(rng, force_duplicate=False)


def _hard(rng) -> dict:
    return _medium_or_hard(rng, force_duplicate=True)


# --- exam: coverage stated, percentile derived, then hard-tier lookup ----

_COVERAGES = (50, 60, 70, 80)


def _exam(rng) -> dict:
    n = N_STD
    coverage = rng.choice(_COVERAGES)
    lower_p = (100 - coverage) // 2
    upper_p = 100 - lower_p
    which = rng.choice(["lower", "upper"])
    p = lower_p if which == "lower" else upper_p
    k = p * n // 100
    check(p * n % 100 == 0, "exam derived percentile does not land on an exact order statistic")

    base = sorted(rng.sample(range(0, 90), n - 1))
    forbidden = {k - 1, max(0, k - 2), min(n - 1, k)}
    candidates_idx = [i for i in range(len(base)) if i not in forbidden and i + 1 not in forbidden]
    dup_source_idx = rng.choice(candidates_idx) if candidates_idx else 0
    values = base[: dup_source_idx + 1] + [base[dup_source_idx]] + base[dup_source_idx + 1 :]
    unsorted_values = _shuffled_breaking_position(rng, values, k)

    mean, resorted, endpoint, k_recheck = _verify_and_build(unsorted_values, False, p, n, which)
    check(k_recheck == k, "exam recomputed k mismatch")

    mean_text = rational_text(mean)
    answer_text = _pair_text(mean_text, which, endpoint)

    candidates = []
    off_k = _toward_middle(k, n)
    off_endpoint = _kth_smallest(resorted, off_k)
    if _components_differ(mean, endpoint, mean, off_endpoint):
        candidates.append(_pair_text(mean_text, which, off_endpoint))

    tail_k = n - k
    if 1 <= tail_k <= n:
        tail_endpoint = _kth_smallest(resorted, tail_k)
        if _components_differ(mean, endpoint, mean, tail_endpoint):
            candidates.append(_pair_text(mean_text, which, tail_endpoint))

    forgot_endpoint = unsorted_values[k - 1]
    if _components_differ(mean, endpoint, mean, forgot_endpoint):
        candidates.append(_pair_text(mean_text, which, forgot_endpoint))

    dropped = max(unsorted_values, key=lambda v: abs(Fraction(v) - mean))
    wrong_mean = Fraction(sum(unsorted_values) - dropped, n - 1)
    if _components_differ(mean, endpoint, wrong_mean, endpoint):
        candidates.append(_pair_text(rational_text(wrong_mean), which, endpoint))

    # coverage-inversion: treats the coverage percentage itself as the
    # percentile level (e.g. "60% interval" -> 60th percentile).
    if coverage * n % 100 == 0:
        wrong_k = coverage * n // 100
        wrong_endpoint = _kth_smallest(resorted, wrong_k)
        if _components_differ(mean, endpoint, mean, wrong_endpoint):
            candidates.append(_pair_text(mean_text, which, wrong_endpoint))

    distractors = _dedupe(answer_text, candidates, mean, endpoint)
    resample_str = ", ".join(str(v) for v in unsorted_values)
    text = phrase(rng, _EXAM_TEMPLATES, n=n, resample=resample_str, coverage=coverage, which=which)
    return dict(
        text=text,
        answer={"mean": mean, "endpoint": endpoint},
        answer_text=answer_text,
        answer_latex=_pair_latex(mean, which, endpoint),
        working=[
            f"A {coverage}% interval leaves {100 - coverage}% split between the two "
            f"tails, {(100 - coverage) // 2}% each; the {which} endpoint is the "
            f"{p}th percentile.",
            f"k = {p} × {n} / 100 = {k}; the {k}-th smallest sorted value is {endpoint}.",
            f"Mean = (sum of all {n} values) / {n} = {mean_text}.",
        ],
        distractors=distractors,
    )


def _dedupe(answer_text: str, candidates: list[str], mean: Fraction, endpoint: int) -> list[str]:
    """Drop any candidate string that is not a genuinely different (mean,
    endpoint) pair from the answer -- and de-duplicate candidates against
    each other -- using the same tolerance-aware component comparison
    `_components_differ` uses, never a bare string comparison."""
    out: list[str] = []
    seen_pairs: list[tuple[Fraction, int]] = [(mean, endpoint)]
    for c in candidates:
        c_mean, c_label, c_endpoint = _parse_pair_text(c)
        if not _components_differ(mean, endpoint, c_mean, c_endpoint):
            continue
        if any(not _components_differ(sm, se, c_mean, c_endpoint) for sm, se in seen_pairs):
            continue
        seen_pairs.append((c_mean, c_endpoint))
        out.append(c)
        if len(out) == 3:
            break
    return out


def _parse_pair_text(text: str) -> tuple[Fraction, str, int]:
    left, right = text.split(",")
    left_label, left_val = left.split("=")
    right_label, right_val = right.split("=")
    return Fraction(left_val.strip()), right_label.strip(), int(right_val.strip())


@generator(
    "p14e_bootstrap_percentile",
    level=14,
    presets={
        "easy": dict(kind="easy"),
        "medium": dict(kind="medium"),
        "hard": dict(kind="hard"),
        "exam": dict(kind="exam"),
    },
)
def p14e_bootstrap_percentile(rng, kind: str = "easy"):
    """Compute a fixed bootstrap summary: the mean of a supplied finite
    resample, paired with one stated percentile order statistic. See the
    module docstring for the kernel, the tier design and the boundary
    against `calculate-mean`."""
    builder = {"easy": _easy, "medium": _medium, "hard": _hard, "exam": _exam}[kind]
    return builder(rng)
