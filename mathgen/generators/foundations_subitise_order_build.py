"""Early-foundations pilot pair: the
first two generators for two genuinely new cognitive operations in this
domain, both framework ``acara-f10-v9``.

- ``subitise_to_5`` (AC9MFN02, prerequisite ``count-to-5``) -- state the
  cardinality of a collection of at most five objects *without* counting
  each object individually. This is a different cognitive operation from
  the existing ``count_objects`` generator (``mathgen/generators/
  early_math.py``), which counts by literal one-to-one enumeration; here the
  layout itself is the difficulty lever, following the real perceptual/
  conceptual subitising distinction: ``easy`` renders a canonical dice-face
  dot pattern (instantly recognisable, no counting needed); ``medium``
  scatters the same range of counts with no recognisable pattern, so a
  learner cannot rely on pattern-matching alone; ``hard`` splits the count
  into two visually separated clusters that must be perceived individually
  and then combined -- genuine conceptual subitising, not just a bigger or
  messier scatter. The grader (``grade_number``) only ever sees the final
  count, so the "without counting" requirement is enforced by item design
  (canonical/scattered/clustered layouts), not by anything the grader can
  verify about how the learner arrived at their answer -- recorded explicitly

- ``order_to_twenty`` (AC9MFN01, prerequisites ``recognise-numerals-to-10``,
  ``order-numbers-to-10``) -- order a stated set of distinct whole numbers
  0-20 inclusive. Zero is a genuine possible sampled
  member at every difficulty (the pool is always ``range(0, 21)``, never
  filtered to exclude it) -- the skill's own new content beyond
  the ``order-numbers-to-10`` prerequisite is precisely "including zero".
  Since the full range is fixed and small, difficulty is driven by (a) how
  many numbers are ordered, (b) how close together the sampled values are
  (closer values need finer-grained comparison), and (c) at ``hard``, zero
  is always included as a genuine trap value alongside multiple other close
  pairs -- a well-documented early-years misconception is to treat zero as
  if it were not a "real" quantity and place it last regardless of the other
  values, which this generator's own ``zero-placed-last`` distractor
  encodes.
"""

from __future__ import annotations

from .. import figures
from ..core import check, generator, pick_distractors, reversed_and_adjacent_swap
from ..figure_kinds.early import counters, counters_total, scatter_dots
from ..slots import NAMES
from .early_math import DOTS

# -- subitise_to_5 ----------------------------------------------------------

# Counter shapes the scattered and clustered layouts draw, keyed by the glyph an earlier text-only
# version of this generator printed (kept so the shape draw consumes the same random numbers).
_SHAPE_OF = {"●": "circle", "◆": "diamond", "★": "star", "■": "square"}

_TEXT = {
    "canonical": (
        "{n} rolls a die. How many dots are on top?",
        "{n} looks at this die. How many dots are on it?",
        "{n} looks at a die. How many dots can you see?",
    ),
    "scattered": (
        "{n} sees some shapes spread out. How many are there?",
        "{n} glances at a group of shapes. How many are there?",
        "{n} takes a quick look at some shapes. How many are there?",
    ),
    "clustered": (
        "{n} sees two groups of shapes. How many are there altogether?",
        "{n} sees two groups of shapes. How many are there in all?",
        "{n} looks at both groups of shapes. How many altogether?",
    ),
}


def _verify_symbol_count(layout: str, symbol: str, count: int) -> None:
    """Independent check: count actual occurrences of the chosen symbol
    character in the assembled display string and confirm it matches the
    intended count. This never re-reads the `count` variable used to build
    the layout -- it re-derives the count from the rendered text itself, so
    a layout-construction bug that silently dropped or duplicated a symbol
    while `count` stayed correct would still be caught."""
    rendered = layout.count(symbol)
    check(rendered == count, f"rendered display shows {rendered} {symbol!r} symbols, expected {count}")


@generator(
    "subitise_to_5",
    level=1,
    presets={
        "easy": dict(mode="canonical", count_range=(1, 5)),
        "medium": dict(mode="scattered", count_range=(1, 5)),
        "hard": dict(mode="clustered", count_range=(3, 5)),
    },
)
def subitise_to_5(rng, mode="canonical", count_range=(1, 5)):
    """State the cardinality of a collection of at most five objects. See
    the module docstring for the perceptual/conceptual subitising rationale
    behind the three layout modes."""
    if not (isinstance(count_range, (tuple, list)) and len(count_range) == 2):
        raise ValueError(f"count_range must be a (lo, hi) pair; got {count_range!r}")
    lo, hi = count_range
    if not (1 <= lo <= hi <= 5):
        raise ValueError(f"count_range must be within 1-5 inclusive; got {count_range!r}")
    if mode == "clustered" and hi < 2:
        raise ValueError("clustered mode needs count_range allowing at least 2")

    count = rng.randint(lo, hi)
    symbol = rng.choice(DOTS)
    shape = _SHAPE_OF[symbol]
    name = rng.choice(NAMES)

    cluster_sizes: tuple[int, int] | None = None
    if mode == "canonical":
        fig = counters(layout="dice", alt="a die face", faces=[{"n": count, "colour": 1}])
        working = [
            "This is a dot pattern you may already recognise (like a dice face) -- "
            "there is no need to count one by one.",
            f"There are {count} altogether.",
        ]
    elif mode == "scattered":
        fig = counters(
            layout="scatter",
            alt="some shapes spread out on a plain background",
            shape=shape,
            dots=scatter_dots(rng, count, arrangement="free"),
        )
        working = [
            "These are scattered, not in a familiar pattern, so look at the whole group rather than a single shape.",
            f"There are {count} altogether.",
        ]
    elif mode == "clustered":
        dots = scatter_dots(rng, count, arrangement="clusters")
        fig = counters(layout="scatter", alt="two separate groups of shapes", shape=shape, dots=dots)
        # re-read the two groups from the drawn positions: the left and the right half of the picture
        cluster_a = sum(1 for d in dots if d["at"][0] < 0.5)
        cluster_sizes = (cluster_a, count - cluster_a)
        working = [
            f"See the two groups without counting each shape: {cluster_sizes[0]} and {cluster_sizes[1]}.",
            f"Combine the groups: {cluster_sizes[0]} + {cluster_sizes[1]} = {count}.",
        ]
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'canonical', 'scattered' or 'clustered'")

    _verify_symbol_count(symbol * counters_total(fig), symbol, count)
    figures.assert_no_leak(fig, [str(count)])
    text = rng.choice(_TEXT[mode]).format(n=name)

    # Misconception distractors:
    #  - undercount-by-one: one fewer than the true count.
    #  - overcount-by-one: one more than the true count.
    #  - miscounts-one-cluster (clustered/hard only): notices only one of
    #    the two visible sub-groups and reports its size alone, forgetting
    #    to combine it with the other -- a genuine failure-to-combine
    #    conceptual-subitising error, distinct from a blanket off-by-one on
    #    the final total.
    candidates = [count - 1, count + 1]
    if cluster_sizes is not None:
        # Provably distinct here: for count in {3,4,5} with each cluster >= 1, the smaller
        # cluster's size is always <= count-2, which can never equal count-1, count+1 or count.
        candidates.append(min(cluster_sizes))
    candidates += [count + 2, count - 2, count + 3]  # further miscounts, so a count of 1 still has three options
    # a picture of objects is never counted as none, so 0 is not a plausible option
    candidates = [str(c) for c in candidates if c > 0]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=count,
        answer_text=str(count),
        answer_latex=str(count),
        working=working,
        distractors=pick_distractors(str(count), candidates),
        figure=fig,
    )


figures.register_reader("subitise_to_5", lambda q: str(counters_total(q.figure)))


# -- order_to_twenty ----------------------------------------------------

_MAX_WHOLE_NUMBER = 20  # inclusive; the narrowed kernel's own upper bound


def _min_gap(numbers: list[int]) -> int:
    ordered = sorted(numbers)
    return min(b - a for a, b in zip(ordered, ordered[1:]))


def _count_close_pairs(numbers: list[int], threshold: int) -> int:
    return sum(
        1 for i in range(len(numbers)) for j in range(i + 1, len(numbers)) if abs(numbers[i] - numbers[j]) <= threshold
    )


def _sample_ordering_numbers(
    rng,
    count: int,
    *,
    force_zero: bool,
    min_gap: int | None,
    min_close_pairs: int,
    close_gap: int = 2,
    max_attempts: int = 5000,
) -> list[int]:
    """Rejection-sample `count` distinct whole numbers from 0-20 inclusive
    satisfying the requested difficulty shape. Zero is always drawable (the
    population is always the full 0-20 range) -- `force_zero` only ever adds
    zero as a *guaranteed* member on top of that, it never removes zero as a
    possibility when False."""
    if count < 2:
        raise ValueError("count must be at least 2")
    population_size = _MAX_WHOLE_NUMBER + 1
    if count > population_size:
        raise ValueError(
            f"cannot sample {count} distinct whole numbers from only "
            f"{population_size} (0-{_MAX_WHOLE_NUMBER} inclusive)"
        )
    non_zero_pool = list(range(1, _MAX_WHOLE_NUMBER + 1))
    if force_zero and count - 1 > len(non_zero_pool):
        raise ValueError(
            f"cannot sample {count - 1} distinct non-zero whole numbers from only "
            f"{len(non_zero_pool)} (1-{_MAX_WHOLE_NUMBER} inclusive)"
        )

    for _ in range(max_attempts):
        if force_zero:
            rest = rng.sample(non_zero_pool, count - 1)
            numbers = [0] + rest
        else:
            numbers = rng.sample(range(0, _MAX_WHOLE_NUMBER + 1), count)

        if min_gap is not None and _min_gap(numbers) < min_gap:
            continue
        if min_close_pairs:
            checked_pool = [n for n in numbers if not (force_zero and n == 0)]
            if _count_close_pairs(checked_pool, close_gap) < min_close_pairs:
                continue
        return numbers

    raise RuntimeError(
        "could not sample numbers satisfying the requested difficulty shape "
        f"(count={count}, force_zero={force_zero}, min_gap={min_gap}, "
        f"min_close_pairs={min_close_pairs}) in {max_attempts} attempts"
    )


def _verify_ascending_order(sampled: list[int], ordered: list[int]) -> None:
    """Independent check, same discipline as
    ``ordered_sequence_build._verify_chronological_order``: verify every
    pair (i, j) with i < j -- not just adjacent pairs, and never by re-running
    the `sorted(...)` call that built `ordered` -- via direct integer
    comparison of the two numbers' actual values."""
    check(sorted(sampled) == sorted(ordered), "ordered list is not a permutation of the sampled numbers")
    check(len(ordered) == len(set(ordered)), "sampled numbers must be distinct")
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            check(int(ordered[i]) < int(ordered[j]), f"{ordered[i]} does not precede {ordered[j]}")


@generator(
    "order_to_twenty",
    level=1,
    presets={
        "easy": dict(count=(3, 4), min_gap=4, force_zero=False, min_close_pairs=0),
        "medium": dict(count=4, min_gap=None, force_zero=False, min_close_pairs=1),
        "hard": dict(count=5, min_gap=None, force_zero=True, min_close_pairs=2),
    },
)
def order_to_twenty(rng, count=4, min_gap=None, force_zero=False, min_close_pairs=0):
    """Order a stated set of distinct whole numbers 0-20 inclusive,
    ascending. See the module docstring for the difficulty rationale."""
    if isinstance(count, (tuple, list)):
        if not count:
            raise ValueError("count sequence must not be empty")
        count = rng.choice(tuple(count))
    if count < 2:
        raise ValueError("count must be at least 2")
    if count > _MAX_WHOLE_NUMBER + 1:
        raise ValueError(
            f"cannot sample more than {_MAX_WHOLE_NUMBER + 1} distinct whole numbers 0-{_MAX_WHOLE_NUMBER}"
        )

    numbers = _sample_ordering_numbers(
        rng,
        count,
        force_zero=force_zero,
        min_gap=min_gap,
        min_close_pairs=min_close_pairs,
    )
    ordered = sorted(numbers)
    _verify_ascending_order(numbers, ordered)

    display_order = list(numbers)
    rng.shuffle(display_order)
    display_text = ", ".join(str(n) for n in display_order)
    answer_text = ", ".join(str(n) for n in ordered)

    working = (
        [
            "Zero counts as a real number and always comes before every positive number.",
            "Compare the numbers by value, smallest first.",
            f"In order: {answer_text}.",
        ]
        if 0 in numbers
        else [
            "Compare the numbers by value, smallest first.",
            f"In order: {answer_text}.",
        ]
    )

    # Misconception distractors:
    #  - reversed-order: largest to smallest instead of smallest to largest.
    #  - adjacent-swap: correctly ordered overall, but the two middle values
    #    are swapped, as if two neighbouring values did not need comparing.
    #  - zero-placed-last (only when zero was sampled): treats zero as if it
    #    were not a real quantity and places it at the end regardless of its
    #    actual value -- a real, well-documented early-years misconception.
    reversed_order, adjacent_swap = reversed_and_adjacent_swap(ordered)
    candidates = [
        ", ".join(str(n) for n in reversed_order),
        ", ".join(str(n) for n in adjacent_swap),
    ]
    if 0 in ordered:
        zero_placed_last = [n for n in ordered if n != 0] + [0]
        candidates.append(", ".join(str(n) for n in zero_placed_last))

    return dict(
        text=f"Arrange these numbers from smallest to largest: {display_text}.",
        latex=rf"\text{{Order: }} {display_text}",
        answer=list(ordered),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )
