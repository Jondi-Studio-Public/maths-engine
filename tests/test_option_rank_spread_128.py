"""Where the key sits among the wrong options, for every topic of the data-display, early-years, graph-transformation
and senior-statistics builds (`dg_*`, `eyz_*`, `gt_*`, `sc_*`), and whether the wrong options are possible at all.

A generator that lists its slips "likeliest first" gives the key the same rank on every seed (always the smallest
option, always second-largest), so a learner can pick it without the maths. `mathgen/generators/_spread_kit.py` draws a
random below/above split. This file fails when

* a question kind, or a whole part, still has one rank for too many seeds (rank = how many wrong options are below the
  key; options that are not plain numbers are ordered by their value when they evaluate, else by their text);
* a wrong option repeats the key, a "nearest 10" question offers a value that is not a multiple of 10, or a question has
  fewer than three options;
* a count-type option is impossible for the figure (a fourth turning point of a parabola squared, a pyramid with two
  vertices, a rotational order of 0) -- impossible options give the key away by elimination;
* a round key is too often the lone multiple of 10 among its options.
"""

from __future__ import annotations

import functools
import glob
import inspect
import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from mathgen import make
from mathgen.core import REGISTRY

SEEDS = 400
MIN_GROUP = 30  # a question kind needs this many samples before its rank shares mean anything
CAP = {4: 0.40, 3: 0.55}  # the largest share one rank may have, by number of options (key and wrong ones)

_GEN = Path(__file__).resolve().parent.parent / "mathgen" / "generators"
_FILES = {Path(f).name for f in glob.glob(str(_GEN / "*_build.py")) if re.match(r"(dg|eyz|gt|sc)_", Path(f).name)}
TOPICS = sorted(
    t
    for t, (_, fn) in REGISTRY.items()
    if (src := inspect.getsourcefile(inspect.unwrap(fn))) and Path(src).name in _FILES
)
TIERS = ("easy", "medium", "hard", "exam")

# (topic, tier, part, kind) -> why the key cannot be spread. Every entry is a place where the feasible options are so few
# that the key is at the edge of them whatever is drawn; the kind "" stands for the whole part.
INHERENT = {
    (
        "gt_f_squared_exam",
        "exam",
        "p2",
        "",
    ): "turning points of a parabola squared: 0 to 3 are possible and the key is 3",
}

# (topic, tier, part) -> (fewest, most) a count option can be and still be possible for what the part asks.
FEASIBLE = {
    ("gt_f_squared_exam", "exam", "p2"): (0, 3),  # turning points of a parabola squared
    ("gt_f_squared_exam", "exam", "p3"): (0, 4),  # solutions of (f(x))^2 = c
    ("gt_modulus_exam", "exam", "p3"): (0, 4),  # solutions of |f(x)| = c
    ("eyz_prism_pyramid_counts_exam", "exam", "p0"): (4, None),  # faces: a solid has at least 4
    ("eyz_prism_pyramid_counts_exam", "exam", "p1"): (6, None),  # edges: at least 6
    ("eyz_prism_pyramid_counts_exam", "exam", "p2"): (4, None),  # vertices: at least 4
    ("eyz_symmetry_count", "medium", "q"): (1, None),  # order of rotational symmetry: at least 1
    ("eyz_symmetry_count_exam", "exam", "p1"): (1, None),
}
# "Round to the nearest 10" is in the stem of the exam estimate chain, but its last part multiplies by a per-item count.
NOT_TENS = {("eyz_estimate_collection_exam", "exam", "p3")}


@functools.lru_cache(maxsize=None)
def _number(text):
    """The number an option stands for (a plain number, a fraction, or an expression such as 3*pi/4 or 1 - pi/4), else None."""
    text = str(text).strip().replace("$", "").replace(",", "")
    try:
        return float(Fraction(text)) if re.fullmatch(r"-?\d+/\d+", text) else float(text)
    except ValueError:
        pass
    if not re.fullmatch(r"[0-9pisqrt()*/+\-. ]{1,60}", text.replace("π", "pi")):
        return None
    try:
        expr = sp.sympify(text.replace("π", "pi").replace("^", "**"), locals={"pi": sp.pi})
        return float(expr) if getattr(expr, "is_number", False) else None
    except (sp.SympifyError, TypeError, ValueError, SyntaxError):
        return None


def _rank(answer, options):
    """How many wrong options are below the key in value, and every option's value (None where it has none).

    Options that do not evaluate (a table, a pair of coordinates, an expression in x) have no magnitude to rank by;
    for those `rank` is None and only the displayed slot (`_slot`) is checked."""
    texts = [str(answer), *(getattr(o, "text", str(o)) for o in options)]
    nums = [_number(t) for t in texts]
    if any(n is None for n in nums):
        return None, nums
    return sum(n < nums[0] for n in nums[1:]), nums


def _slot(seed, answer, options) -> int:
    """Where the key lands in the option list the presenter shows (`mathgen.core.mcq_options` shuffles by the seed)."""
    shown = [str(answer), *(getattr(o, "text", str(o)) for o in options)]
    random.Random(seed).shuffle(shown)
    return shown.index(str(answer))


def _kind(text: str) -> str:
    words = re.findall(r"[a-z#']+", re.sub(r"\d+(?:\.\d+)?", "#", text.lower()))
    return " ".join(words[:4])


def _units(q):
    if q.parts:
        return [(f"p{i}", p.answer_text, p.distractors, p.prompt) for i, p in enumerate(q.parts)]
    return [("q", q.answer_text, q.distractors, q.text)]


def _limit(n_options: int, n: int, sigmas: float = 3.5) -> float:
    """The most share one rank may take: the cap for a big group, a little more for a small one (chance can reach it)."""
    chance = 1 / n_options
    return max(CAP[n_options], chance + sigmas * math.sqrt(chance * (1 - chance) / n))


def survey(topic: str, tier: str, seeds=range(SEEDS)):
    """Per (part, kind) and per part: Counter of ranks (and option counts), plus every problem found on the way."""
    kinds: dict = defaultdict(Counter)
    parts: dict = defaultdict(Counter)
    samples: dict = defaultdict(list)  # (part, kind, options) -> [(key value, wrong values)] for whole-number groups
    slots: dict = defaultdict(Counter)
    problems: list[str] = []
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        for label, answer, options, text in _units(q):
            shown = [getattr(o, "text", str(o)) for o in options or []]
            if len(shown) < 2:
                problems.append(f"{topic} {tier} {label} seed {seed}: only {len(shown)} wrong options")
                continue
            if str(answer) in shown:
                problems.append(f"{topic} {tier} {label} seed {seed}: an option repeats the key {answer}")
            rank, nums = _rank(answer, options)
            if all(n is not None for n in nums):
                if "nearest 10" in text and (topic, tier, label) not in NOT_TENS and any(v % 10 for v in nums):
                    problems.append(f"{topic} {tier} {label} seed {seed}: not a multiple of 10: {answer} {shown}")
                if (bounds := FEASIBLE.get((topic, tier, label))) is not None:
                    lo, hi = bounds
                    bad = [v for v in nums if (lo is not None and v < lo) or (hi is not None and v > hi)]
                    if bad:
                        problems.append(f"{topic} {tier} {label} seed {seed}: impossible option {bad} for {answer}")
            slots[(label, len(shown) + 1)][_slot(q.seed, answer, options)] += 1
            if rank is None:
                continue
            kinds[(label, _kind(text), len(shown) + 1)][rank] += 1
            parts[(label, "", len(shown) + 1)][rank] += 1
            if all(n is not None and n == int(n) for n in nums):
                samples[(label, _kind(text), len(shown) + 1)].append((nums[0], nums[1:]))
                samples[(label, "", len(shown) + 1)].append((nums[0], nums[1:]))
    return kinds, parts, samples, slots, problems


def forced_share(topic: str, tier: str, label: str, group: list, wrong: int) -> float:
    """The smallest largest-rank-share any option sets could reach for these keys.

    A whole-number key that is the least value the question ever shows has nothing to be above, so it is the smallest
    option whatever is drawn; one a step from the least has at most one option below. Each seed can reach only the
    ranks its key leaves room for (`wrong` options, `key - least` steps below it, `most - key` above); the answer is the
    best split of the seeds over those ranks, by Hall's condition on runs of ranks."""
    values = [v for key, opts in group for v in (key, *opts)]
    lo, hi = min(values), max(values)
    if (bounds := FEASIBLE.get((topic, tier, label))) is not None:
        lo = bounds[0] if bounds[0] is not None else lo
        hi = bounds[1] if bounds[1] is not None else hi
    step = functools.reduce(math.gcd, [int(abs(v - lo)) for v in values if v != lo], 0) or 1
    reach = []
    for key, _ in group:
        below, above = int((key - lo) // step), int((hi - key) // step)
        reach.append((max(0, wrong - above), min(wrong, below)))
    worst = 0.0
    for x in range(wrong + 1):
        for y in range(x, wrong + 1):
            inside = sum(1 for a, b in reach if x <= a and b <= y)
            worst = max(worst, inside / (len(reach) * (y - x + 1)))
    return worst


def lopsided(topic: str, tier: str, seeds=range(SEEDS)) -> tuple[list[str], list[str]]:
    kinds, parts, samples, slots, problems = survey(topic, tier, seeds)
    found = []
    for (label, n_options), where in slots.items():
        total = sum(where.values())
        if total >= MIN_GROUP and n_options in CAP and max(where.values()) / total > _limit(n_options, total):
            found.append(f"{tier} {label} the key sits in one slot of the shown list {dict(sorted(where.items()))}")
    for table in (kinds, parts):
        for (label, kind, n_options), ranks in table.items():
            total = sum(ranks.values())
            if total < MIN_GROUP or n_options not in CAP:
                continue
            if (topic, tier, label, kind) in INHERENT or (topic, tier, label, "") in INHERENT:
                continue
            share = max(ranks.values()) / total
            limit = _limit(n_options, total, 3.5 if not kind else 4.0)
            note = ""
            if (
                len(samples.get((label, kind, n_options), ())) == total
            ):  # whole-number options: allow what the keys force
                forced = forced_share(topic, tier, label, samples[(label, kind, n_options)], n_options - 1)
                limit = max(limit, forced + 0.15)
                note = f" (keys force {forced:.2f})"
            if share > limit:
                found.append(f"{tier} {label} [{kind or 'whole part'}] {dict(sorted(ranks.items()))}{note}")
    return found, problems


@pytest.mark.parametrize("topic", TOPICS)
def test_key_has_no_fixed_rank_among_the_options_and_no_option_is_impossible(topic):
    unfair, broken = [], []
    for tier in TIERS:
        try:
            make(topic, seed=0, difficulty=tier)
        except (KeyError, ValueError):  # the topic has no such tier
            continue
        found, problems = lopsided(topic, tier)
        unfair += found
        broken += problems
    assert not broken, broken[:3]
    assert not unfair, unfair


def test_a_round_key_is_not_usually_the_lone_round_option():
    """With one multiple of 10 among the options the learner picks it. Over every question kind that has such a set the
    key must be the lone multiple of 10 no more than half the time (a chance draw gives about a quarter)."""
    lone = Counter()
    seen = Counter()
    for topic in TOPICS:
        for tier in TIERS:
            try:
                make(topic, seed=0, difficulty=tier)
            except (KeyError, ValueError):
                continue
            for seed in range(300):
                for label, answer, options, text in _units(make(topic, seed=seed, difficulty=tier)):
                    shown = [getattr(o, "text", str(o)) for o in options or []]
                    nums = [_number(x) for x in [answer, *shown]]
                    if "nearest 10" in text or any(n is None or n != int(n) for n in nums) or len(nums) < 3:
                        continue
                    rounds = [n for n in nums if n != 0 and n % 10 == 0]
                    if len(rounds) == 1:
                        group = (topic, tier, label, _kind(text))
                        seen[group] += 1
                        lone[group] += rounds[0] == nums[0]
    worst = [(g, lone[g], n) for g, n in seen.items() if n >= MIN_GROUP and lone[g] / n > 0.5]
    assert not worst, worst
    assert sum(lone.values()) / sum(seen.values()) < 0.35


def test_a_graph_read_against_round_gridlines_pads_its_options_with_no_neighbouring_integers():
    """150 km beside 151 and 152 km, with a gridline every 50, is no reading of the graph."""
    checked = 0
    for tier in ("easy", "medium", "hard"):
        for seed in range(300):
            q = make("dg_real_data_graphs", seed=seed, difficulty=tier)
            key = _number(q.answer_text)
            if key is None or key < 20 or key % 10:
                continue
            checked += 1
            plain = [_number(o) for o, tag in q.option_notes.items() if tag == "dg-near-miss-value"]
            assert all(abs(o - key) > 3 for o in plain), (tier, seed, q.answer_text, list(q.distractors))
    assert checked >= 40


def test_the_fixes_this_file_was_written_for_hold():
    """The question kinds reviewers measured as having a fixed key rank, named so a regression names them."""
    cases = [
        ("dg_real_data_graphs", "hard"),
        ("dg_real_data_graphs", "medium"),
        ("eyz_prism_pyramid_counts", "easy"),
        ("sc_simulated_proportions", "hard"),
        ("sc_simulated_proportions_exam", "exam"),
        ("gt_sqrt_function_exam", "exam"),
        ("gt_inverse_trig_exam", "exam"),
        ("gt_inverse_trig", "hard"),
        ("sc_conditional_counts", "easy"),
        ("sc_conditional_counts_exam", "exam"),
        ("sc_integrate_tan_sec_exam", "exam"),
    ]
    for topic, tier in cases:
        found, problems = lopsided(topic, tier)
        assert not problems and not found, (topic, tier, found, problems[:3])


# --------------------------------------------------------------------------------------------- robustness


@pytest.mark.parametrize(
    "topic",
    [
        "sc_conditional_counts_exam",
        "sc_integrate_tan_sec_exam",
        "sc_simulated_proportions_exam",
        "sc_complex_euler_exam",
    ],
)
def test_an_exam_generator_called_directly_with_an_unknown_tier_raises_value_error(topic):
    """The registry's presets name only the exam tier; a direct call with another name is a ValueError, not a
    TypeError or KeyError from deep inside the build."""
    fn = inspect.unwrap(REGISTRY[topic][1])
    with pytest.raises(ValueError):
        fn(random.Random(0), tier="nosuch")


@pytest.mark.parametrize(
    ("topic", "tier", "seeds"),
    [
        ("dg_graph_terms_chain", "exam", [625, 1726, *range(0, 400)]),  # IndexError in the drawn chain at 625 and 1726
        ("gt_graph_transform_exam", "exam", [1967, *range(0, 400)]),  # too few distractors for part (d) at 1967
        ("dg_real_data_graphs_chain", "exam", range(0, 600)),  # part (c) once had two options only
    ],
)
def test_seeds_that_once_crashed_or_starved_a_part_of_options_now_make(topic, tier, seeds):
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        for part in q.parts:
            assert len(part.distractors) >= 2, (topic, seed, part.label)
        if topic == "dg_real_data_graphs_chain":
            assert len(q.parts[2].distractors) >= 3, (topic, seed)
