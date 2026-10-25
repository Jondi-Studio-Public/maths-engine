"""Where the key sits among the wrong options, for every topic of the data-display, early-years, graph-transformation
and senior-statistics builds (`dg_*`, `eyz_*`, `gt_*`, `sc_*`, `nv_*`, `nx_*`, `vm_*`), and whether the wrong options are possible at all.

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
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.core import REGISTRY

SEEDS = 600
MIN_GROUP = 30  # a question kind needs this many samples before its rank shares mean anything


def _cap(n_options: int) -> float:
    """The largest share one rank may have, by number of options (key and wrong ones): about 1.6 times chance."""
    return 0.55 if n_options <= 3 else round(1.6 / n_options, 3)


CAP = {n: _cap(n) for n in range(3, 13)}

_GEN = Path(__file__).resolve().parent.parent / "mathgen" / "generators"
_FILES = {
    Path(f).name for f in glob.glob(str(_GEN / "*_build.py")) if re.match(r"(dg|eyz|gt|sc|nv|nx|vm)_", Path(f).name)
}
TOPICS = sorted(
    t
    for t, (_, fn) in REGISTRY.items()
    if (src := inspect.getsourcefile(inspect.unwrap(fn))) and Path(src).name in _FILES
)
TIERS = ("easy", "medium", "hard", "exam")

# (topic, tier, part, kind) -> why the key cannot be spread. Nothing is exempt: a redesign gives the key room instead
# (the turning-point count of a parabola squared asks about an interval now, so 0 to 3 are all live answers).
# (topic, tier, part) -> (fewest, most) a count option can be and still be possible for what the part asks.
FEASIBLE = {
    ("gt_f_squared_exam", "exam", "p2"): (0, 3),  # turning points of a parabola squared
    ("gt_f_squared_exam", "exam", "p3"): (0, 4),  # solutions of (f(x))^2 = c
    ("eyz_estimate_collection_exam", "exam", "p0"): (10, 66),  # a side of 6 x 11 cells holds at most 66
    ("gt_modulus_exam", "exam", "p3"): (0, 4),  # solutions of |f(x)| = c
    ("eyz_prism_pyramid_counts_exam", "exam", "p0"): (4, None),  # faces: a solid has at least 4
    ("eyz_prism_pyramid_counts_exam", "exam", "p1"): (6, None),  # edges: at least 6
    ("eyz_prism_pyramid_counts_exam", "exam", "p2"): (4, None),  # vertices: at least 4
    ("eyz_symmetry_count", "medium", "q"): (1, None),  # order of rotational symmetry: at least 1
    ("eyz_symmetry_count_exam", "exam", "p1"): (1, None),
}
# "Round to the nearest 10" is in the stem of the exam estimate chain, but its last part multiplies by a per-item count.
NOT_TENS = {("eyz_estimate_collection_exam", "exam", "p3")}


_WORDS = {"pi": sp.pi, "sqrt": sp.sqrt, "ln": sp.log, "log": sp.log}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


@functools.lru_cache(maxsize=None)
def _number(text):
    """The number an option stands for (a plain number, a fraction, or an expression such as 2pi/3, 3*pi/4, ln(2)/2,
    1 - pi/4 or 3sqrt(2)), else None. Implicit multiplication is read ("2pi" is 2*pi)."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(text).strip().replace("$", "")).replace("\u2212", "-")
    try:
        return float(Fraction(text)) if re.fullmatch(r"-?\d+/\d+", text) else float(text)
    except ValueError:
        pass
    text = text.replace("\u03c0", "pi").replace("\u221a", "sqrt")
    if not re.fullmatch(r"[0-9a-z()*/+\-.^ ]{1,60}", text) or set(re.findall(r"[a-z]+", text)) - set(_WORDS):
        return None
    try:
        expr = parse_expr(text, local_dict=dict(_WORDS), transformations=_TRANSFORMS)
        return float(expr) if getattr(expr, "is_number", False) else None
    except (sp.SympifyError, TypeError, ValueError, SyntaxError, ZeroDivisionError, AttributeError):
        return None


_LIKELIHOOD = (("impossible", 0), ("unlikely", 1), ("an even chance", 2), ("likely", 3), ("certain", 4))
_INF = 1e9


def _frac(text):
    try:
        return float(Fraction(text.strip()))
    except (ValueError, ZeroDivisionError):
        return _number(text)


def _hull(text):
    """(least, greatest) of the real values an inequality (or `all real numbers`) allows, infinities as +-1e9."""
    low = text.strip()
    if low == "all real numbers":
        return -_INF, _INF
    if low == "no solution":
        return _INF, -_INF  # the empty set: nothing is allowed
    if " or " in low:
        return -_INF, _INF
    m = re.fullmatch(r"(-?[\d./]+) (≤|<) [xy] (≤|<) (-?[\d./]+)", low)
    if m:
        return _frac(m.group(1)), _frac(m.group(4))
    m = re.fullmatch(r"[xy] (≤|<|≥|>) (-?[\d./]+)", low)
    if m:
        v = _frac(m.group(2))
        return (-_INF, v) if m.group(1) in "≤<" else (v, _INF)
    return None


@functools.lru_cache(maxsize=None)
def _expression(text):
    """(value at 0.37, value at 1.41) of an expression in x as the grader reads it (`3*exp(2*x)/2`, `-cos(2*(x + pi/4))`,
    a trailing `+ C` dropped), or None when it cannot be read: the rank is by what the option IS, not how long it is."""
    body = re.sub(r"\s*\+\s*C$", "", text.strip()).replace("^", "**").replace("ln", "log")
    try:
        expr = parse_expr(
            body,
            local_dict={"x": sp.Symbol("x"), "e": sp.E},
            transformations=standard_transformations + (implicit_multiplication_application,),
        )
        out = []
        for point in (0.37, 1.41):
            value = complex(expr.subs(sp.Symbol("x"), point))
            out.append(value.real if abs(value.imag) < 1e-9 else _INF)
        return tuple(out)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, sp.SympifyError, AttributeError):
        return None


def _complex(text):
    """The value of a complex number written ``a + bi`` (with fractions and surds), else None."""
    t = re.sub(r"(?<=[\d)])i\b", "*I", text.replace("\u221a", "sqrt"))
    t = re.sub(r"(?<![\w*])i\b", "I", t)
    try:
        value = parse_expr(t, local_dict={"I": sp.I, "sqrt": sp.sqrt}, transformations=_TRANSFORMS)
        return complex(value)
    except (sp.SympifyError, TypeError, ValueError, SyntaxError, ZeroDivisionError):
        return None


def features(text) -> dict | None:
    """What an option stands for as numbers, so that the key's place among the options can be ranked for ANY answer kind.

    A number is its value; a coordinate pair is its y and its x; grid squares are their count and their summed column and
    row; a set, a list, a bar chart and a stem plot are their sizes and totals; a chance sentence is its likelihood
    word; an inequality is its first and last bound and its length; an expression in x is its value at two points. None
    means the option is none of these and the question must be given a kind here (a gap in this test is a failure)."""
    text = str(text).strip()
    if (v := _number(text)) is not None:
        return {"value": v}
    if re.fullmatch(r"[0-9A-F]*[A-F][0-9A-F]*", text):  # a hexadecimal numeral with a letter: ordered by its value
        return {"value": float(int(text, 16))}
    if m := re.fullmatch(r"x = (\S+), y = (\S+)", text):
        x, y = _frac(m.group(1)), _frac(m.group(2))
        return {"y": y, "x": x} if x is not None and y is not None else None
    if cells := re.findall(r"\((-?\d+), (-?\d+)\)", text):
        if re.fullmatch(r"(\(-?\d+, -?\d+\) ?)+", text):
            cols, rows = [int(c) for c, _ in cells], [int(r) for _, r in cells]
            # how many squares, and where they are: the mean row and the mean column, each ranked on its own
            return {
                "count": float(len(cells)),
                "row": sum(rows) / len(rows),
                "column": sum(cols) / len(cols),
            }
    if m := re.fullmatch(r"\{(.*)\}", text):
        items = sorted(i.strip() for i in m.group(1).split(","))
        return {"size": float(len(items)), "_items": frozenset(items)}
    if "Leaf =" in text:
        rows = tuple(tuple(int(d) for d in part.split("=")[1].split()) for part in text.split(", "))
        return {"rows": rows}
    if re.fullmatch(r"[^,=]+ = -?[\d./]+(, [^,=]+ = -?[\d./]+)*", text):
        return {"bars": tuple(_frac(part.split(" = ")[1]) for part in text.split(", "))}
    for word, score in _LIKELIHOOD:
        if re.search(rf"\b{word}\b", text):
            return {"likelihood": float(score)}
    if hull := _hull(text):  # an inequality: its bounds as written (first, last) and its length
        nums = [_frac(b) or 0.0 for b in re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)]
        return {
            "first": nums[0] if nums else 0.0,
            "last": nums[-1] if nums else 0.0,
            "length": float(len(text)),
        }
    if re.fullmatch(r"-?[\d./]+(, -?[\d./]+)+", text):
        nums = [_frac(i) for i in text.split(", ")]
        return {"size": float(len(nums)), "greatest": max(nums), "least": min(nums)}
    if re.fullmatch(r"[0-9a-z()*/+\-.^ ]*\d*i\)?|-?\d+", text) and re.search(
        r"(?<![a-z])\d*i\b|\bi\b", text
    ):  # a complex number
        z = _complex(text)
        if z is not None:
            return {"re": z.real, "im": z.imag}
    if re.search(r"\bx\b", text):  # an expression in x: its values at two points, and its length
        if (at := _expression(text)) is None:
            return None
        return {"at 0.37": at[0], "at 1.41": at[1], "length": float(len(text))}
    return None


def ranks(answer, options):
    """{feature: key rank} for one question, and the option texts that stand for nothing.

    The rank is the number of wrong options below the key, plus half of those equal to it: a tie is never skipped, it counts as half a place (the
    key's mean place if the tie were broken by a coin). A feature the options do not
    all have, or on which they all agree (the x of three coordinates in one column), says nothing and is left out."""
    texts = [str(answer), *(getattr(o, "text", str(o)) for o in options)]
    feats = [features(t) for t in texts]
    if any(f is None for f in feats):
        return {}, [t for t, f in zip(texts, feats) if f is None]
    if all("_items" in f for f in feats):
        # how much each list shares with the others: the key is the list the wrong ones are a slip away from, so it
        # tends to share most (or, with wrong lists far from each other, least). It reads the SET of options, so it
        # is the options that decide it, not which entries the key happens to hold.
        lists = [f["_items"] for f in feats]
        for i, f in enumerate(feats):
            f["agreement"] = sum(
                len(lists[i] & o) / len(lists[i] | o) for j, o in enumerate(lists) if j != i and lists[i] | o
            )
    for f in feats:
        f.pop("_items", None)
    out = {}
    for name in set.intersection(*(set(f) for f in feats)):
        vals = [round(f[name], 9) if isinstance(f[name], float) else f[name] for f in feats]
        if len(set(vals)) == 1:
            continue
        key = vals[0]
        out[name] = sum(v < key for v in vals[1:]) + sum(v == key for v in vals[1:]) / 2
    return out, []


def impossible(text: str, answer: str, nums: list) -> list:
    """The options the stem rules out, whatever the figure: a count below 0 or not whole, a probability or proportion
    above 1 (unless the answer is a percentage), a difference asked for as "greater than" that is not positive."""
    low = text.lower()
    key, wrong = nums[0], nums[1:]
    bad = []
    if re.search(r"\bhow many\b|\bnumber of\b", low) and key == int(key) and key >= 0:
        bad += [v for v in wrong if v < 0 or v != int(v)]
    if re.search(r"by how much is .{0,80}(greater|more|larger|higher) than", low) and key > 0:
        bad += [v for v in wrong if v <= 0]
    if re.search(r"probability|proportion", low) and 0 <= key <= 1 and "%" not in answer and "percent" not in low:
        bad += [v for v in wrong if v > 1 or v < 0 and "minus" not in low]
    return bad


def _kind(text: str) -> str:
    words = re.findall(r"[a-z#']+", re.sub(r"\d+(?:\.\d+)?", "#", text.lower()))
    return " ".join(words[:4])


def _units(q):
    if q.parts:
        return [(f"p{i}", p.answer_text, p.distractors, p.prompt) for i, p in enumerate(q.parts)]
    return [("q", q.answer_text, q.distractors, q.text)]


def _tags(q, options) -> list:
    """The misconception tag of each option, in order (None for an untagged filler)."""
    notes = {**q.option_notes, **{k: v for p in q.parts or () for k, v in p.option_notes.items()}}
    return [notes.get(getattr(o, "text", str(o))) for o in options]


def _limit(n_options: int, n: int, sigmas: float = 3.5) -> float:
    """The most share one rank may take: the cap for a big group, a little more for a small one (chance can reach it)."""
    chance = 1 / n_options
    return max(CAP[min(n_options, 12)], chance + sigmas * math.sqrt(chance * (1 - chance) / n))


def survey(topic: str, tier: str, seeds=range(SEEDS)):
    """Per (part, kind, feature) and per (part, feature): Counter of key ranks, plus every problem found on the way."""
    kinds: dict = defaultdict(Counter)
    parts: dict = defaultdict(Counter)
    problems: list[str] = []
    unranked: Counter = Counter()
    seen: Counter = Counter()
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        for label, answer, options, text in _units(q):
            shown = [getattr(o, "text", str(o)) for o in options or []]
            if len(shown) < 2:
                problems.append(f"{topic} {tier} {label} seed {seed}: only {len(shown)} wrong options")
                continue
            if str(answer) in shown:
                problems.append(f"{topic} {tier} {label} seed {seed}: an option repeats the key {answer}")
            nums = [_number(t) for t in [answer, *shown]]
            if all(n is not None for n in nums):
                if "nearest 10" in text and (topic, tier, label) not in NOT_TENS and any(v % 10 for v in nums):
                    problems.append(f"{topic} {tier} {label} seed {seed}: not a multiple of 10: {answer} {shown}")
                if bad := impossible(text, str(answer), nums):
                    problems.append(
                        f"{topic} {tier} {label} seed {seed}: impossible option {bad} for {answer}: {text[:60]}"
                    )
                if (bounds := FEASIBLE.get((topic, tier, label))) is not None:
                    lo, hi = bounds
                    bad = [v for v in nums if (lo is not None and v < lo) or (hi is not None and v > hi)]
                    if bad:
                        problems.append(f"{topic} {tier} {label} seed {seed}: impossible option {bad} for {answer}")
            found, nothing = ranks(answer, options)
            seen[label] += 1
            if nothing:
                unranked[label] += 1
                if unranked[label] <= 2:
                    problems.append(f"{topic} {tier} {label} seed {seed}: options with no rank to check: {nothing[:2]}")
            # plain near misses fill the side of the key that the named slips leave short, so a question that HAS such
            # fillers can put the key at one end every time (sc_complex_euler: +1/2, +1 and 6 beside a key of 3 sqrt 2)
            filled = any(tag and tag.endswith("near-miss-value") for tag in _tags(q, options))
            for name, rank in found.items():
                kinds[(label, _kind(text), len(shown) + 1, name)][rank] += 1
                parts[(label, "", len(shown) + 1, name)][rank] += 1
                if filled:
                    kinds[(label, _kind(text) + " +fillers", len(shown) + 1, name)][rank] += 1
                    parts[(label, " +fillers", len(shown) + 1, name)][rank] += 1
    for label, count in unranked.items():
        if count > seen[label] // 10:
            problems.append(
                f"{topic} {tier} {label}: {count} of {seen[label]} questions have options no rank can be read from"
            )
    return kinds, parts, problems


def lopsided(topic: str, tier: str, seeds=range(SEEDS)) -> tuple[list[str], list[str]]:
    kinds, parts, problems = survey(topic, tier, seeds)
    found = []
    rare: dict = defaultdict(Counter)  # groups too small to judge alone are judged together, never skipped
    for table in (kinds, parts):
        for (label, kind, n_options, name), ranks_ in table.items():
            total = sum(ranks_.values())
            if n_options < 3:
                continue
            if total < MIN_GROUP:
                rare[(label, n_options, name)].update(ranks_)
                continue
            share = max(ranks_.values()) / total
            if share > _limit(n_options, total, 3.5 if not kind.strip() else 4.5):
                found.append(f"{tier} {label} [{kind.strip() or 'whole part'} / {name}] {dict(sorted(ranks_.items()))}")
    for (label, n_options, name), ranks_ in rare.items():
        total = sum(ranks_.values())
        if total >= MIN_GROUP and max(ranks_.values()) / total > _limit(n_options, total):
            found.append(f"{tier} {label} [small groups together / {name}] {dict(sorted(ranks_.items()))}")
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


# --------------------------------------------------------------------------------------- bounds on every option


def test_every_call_that_offers_options_must_state_how_far_they_may_reach():
    """`lo=` and `hi=` are mandatory (None for an open end); both open needs `unbounded="why"`. A call that says nothing
    cannot offer a filler the question cannot have."""
    from mathgen.generators import _spread_kit as kit

    for call in (
        lambda: kit.mixed(5, [(3, "x")]),
        lambda: kit.mixed(5, [(3, "x")], lo=0),
        lambda: kit.mixed(5, [(3, "x")], lo=None, hi=None),
        lambda: kit.pick("5", [], lo=0),
        lambda: kit.whole_part("p", 5, 1, [(3, "x")], working=[]),
    ):
        with pytest.raises(TypeError):
            call()
    assert kit.mixed(5, [(3, "x")], lo=0, hi=None)  # one open end is a statement
    assert kit.mixed(5, [(3, "x")], lo=None, hi=None, unbounded="open")  # so is a reason (one of the kinds)


def test_a_value_outside_the_bounds_is_dropped_not_offered():
    from mathgen.generators import _spread_kit as kit

    out = kit.mixed(
        5,
        [(-3, "slip"), (4, "slip"), (9, "slip")],
        fillers=[(6, "near"), (-1, "near"), (12, "near"), (3, "near")],
        lo=0,
        hi=10,
    )
    assert all(0 <= v <= 10 for v, _ in out), out
    assert [v for v, _ in out][:3] and set(v for v, _ in out) <= {4, 9, 6, 3}
    wide = kit.mixed(
        5, [(-3, "slip")], fillers=[(6, "near"), (4, "near"), (7, "near")], lo=0, hi=10, slip_range=(-5, 10)
    )
    assert (-3, "slip") in wide  # a slip that IS the mistake of leaving the range may stray, a filler may not


def test_the_integrals_over_zero_to_pi_over_4_are_never_offered_negative():
    for seed in range(300):
        q = make("sc_integrate_tan_sec_exam", seed=seed, difficulty="exam")
        for part in q.parts[:3]:
            for option in part.distractors:
                value = _number(getattr(option, "text", option))
                assert value is not None and 0 < value <= math.pi / 4 + 1e-9, (seed, part.label, option)


def test_a_multigraph_never_offers_more_pairs_or_extra_copies_than_its_edges_allow():
    checked = 0
    for seed in range(400):
        q = make("dg_graph_terms", seed=seed, difficulty="hard")
        listed = re.search(r"are (.+?)\. An edge written twice", q.text)
        if not listed:
            continue
        edges = [e.strip() for e in listed.group(1).split(",")]
        nonloop = sum(1 for e in edges if e[0] != e[1])
        if "How many pairs of vertices are joined by multiple edges" in q.text:
            cap = nonloop // 2  # a pair that is repeated has at least two of the non-loop edges
        elif "extra" in q.text and "multiple" in q.text:
            cap = nonloop - 1
        else:
            continue
        checked += 1
        values = [_number(x) for x in (q.answer_text, *q.distractors)]
        assert all(v is not None and 0 <= v <= cap for v in values), (seed, q.answer_text, list(q.distractors), cap)
    assert checked >= 20


def test_part_a_of_the_symmetry_chain_and_the_estimate_chain_stay_inside_their_figures():
    for seed in range(300):
        q = make("eyz_estimate_collection_exam", seed=seed, difficulty="exam")
        for part in q.parts[:2]:
            assert all(_number(x) <= 66 for x in (part.answer_text, *part.distractors)), (seed, part.label)


# ------------------------------------------------------------------- non-scalar answers: coordinates, lists, sets, bars


def _y_and_x_ranks(topic, tier, seeds=range(400)):
    """{(y rank, x rank): n} for every coordinate-pair part of a topic, ties counted half."""
    out = Counter()
    for seed in seeds:
        for _label, answer, options, _text in _units(make(topic, seed=seed, difficulty=tier)):
            shown = [getattr(o, "text", str(o)) for o in options or []]
            feats = [features(t) for t in (answer, *shown)]
            if not all(f and "y" in f for f in feats):
                continue
            key = feats[0]
            for name in ("y", "x"):
                below = sum(f[name] < key[name] for f in feats[1:]) + sum(f[name] == key[name] for f in feats[1:]) / 2
                out[(name, below)] += 1
    return out


@pytest.mark.parametrize(
    ("topic", "tier"),
    [("gt_f_squared", "medium"), ("gt_f_squared", "hard"), ("gt_modulus", "hard"), ("gt_modulus", "medium")],
)
def test_the_key_coordinate_is_not_always_the_highest_or_the_lowest_option(topic, tier):
    """Measured on 37bc137: the correct coordinate of gt_f_squared medium had the greatest y on 3000 of 3000 seeds, and
    that of gt_modulus hard sat second from the top on 92%. Each coordinate is ranked on its own: y, then x."""
    ranks = _y_and_x_ranks(topic, tier)
    for name in ("y", "x"):
        mine = {r: n for (k, r), n in ranks.items() if k == name}
        total = sum(mine.values())
        assert total >= 100, (topic, tier, name, mine)
        assert max(mine.values()) / total <= 0.5, (topic, tier, name, mine)
        assert len(mine) >= 3, (topic, tier, name, mine)


def test_pick_pairs_draws_the_rank_of_the_key_in_y_and_in_x():

    from mathgen.generators import _spread_kit as kit
    from mathgen.generators._sr2b_kit import Pair

    seen = {"y": Counter(), "x": Counter()}
    for seed in range(300):
        token = kit._RNG.set(random.Random(seed))
        try:
            key = Pair("x", Fraction(3), "y", Fraction(4))
            slips = [("gtf-reflected", -3, 4), ("gtf-swapped", 4, 3), ("gtf-doubled", 3, 8), ("gtf-negated", 3, -4)]
            options = kit.pick_pairs(key, slips, x_lo=-8, x_hi=8, y_lo=-8, y_hi=8)
        finally:
            kit._RNG.reset(token)
        feats = [features(o.text) for o in options]
        assert len(feats) == 3 and all(feats), options
        seen["y"][sum(f["y"] < 4 for f in feats) + sum(f["y"] == 4 for f in feats) / 2] += 1
        seen["x"][sum(f["x"] < 3 for f in feats) + sum(f["x"] == 3 for f in feats) / 2] += 1
    for name, counts in seen.items():
        assert len(counts) >= 4 and max(counts.values()) <= 0.4 * 300, (name, counts)


def test_a_free_text_opt_out_of_the_bound_check_is_refused():
    """`unbounded=` takes one of KINDS (the shape of the answer, so the rank test knows how to read it), never a sentence:
    the old free-text "coordinate pairs ..." let 15 call sites out of the test."""
    from mathgen.generators import _spread_kit as kit

    assert "coordinate" not in "".join(kit.KINDS) and "list" in kit.KINDS and "set" in kit.KINDS
    with pytest.raises(ValueError):
        kit.check_bounds(None, None, "coordinate pairs have no one-dimensional bound")
    with pytest.raises(ValueError):
        kit.pick_kind("[1, 2]", [], "coordinates")
    kit.check_bounds(None, None, "open")  # a scalar with an open range is still a stated reason


def test_no_generator_leaves_the_rank_test_with_a_free_text_unbounded_reason():
    """Every `unbounded=` in the generators is one of the kinds (no coordinate pair or list among them)."""
    from mathgen.generators import _spread_kit as kit

    root = Path(__file__).resolve().parents[1] / "mathgen" / "generators"
    bad = []
    for path in root.glob("*.py"):
        for m in re.finditer(r"unbounded\s*=\s*([\"'])(.*?)\1", path.read_text(encoding="utf-8")):
            if m.group(2) not in kit.KINDS and path.name not in {"_spread_kit.py"}:
                bad.append((path.name, m.group(2)))
    assert not bad, bad[:5]


def test_the_negative_rate_of_change_is_no_longer_offered_as_an_option():
    """dg_real_data_graphs medium offered "-12 litres per 100 km" on 76 seeds; a rate of consumption is not negative,
    so the option was a giveaway and the rank test exempted its tag."""
    for tier in ("easy", "medium", "hard"):
        for seed in range(300):
            q = make("dg_real_data_graphs", seed=seed, difficulty=tier)
            assert "dg-rd-rate-sign" not in set(q.option_notes.values()), (tier, seed)
            assert all(not str(getattr(o, "text", o)).startswith("-") for o in q.distractors), (tier, seed)


def test_bars_stem_and_cell_options_are_drawn_on_both_sides_of_the_key():
    """Bars, stem plots, shaded squares and vertex sets are ranked from their content (lexicographic bars and rows, the
    mean row and column of shaded squares, the size and letters of a set): none has a rank in more than half of them."""
    for topic, tier in (
        ("dg_construct_columns", "medium"),
        ("dg_construct_dotplot", "easy"),
        ("dg_construct_stem", "medium"),
        ("dg_grid_map", "hard"),
        ("dg_graph_terms_chain", "exam"),
    ):
        found, problems = lopsided(topic, tier)
        assert not found and not problems, (topic, tier, found, problems[:2])
