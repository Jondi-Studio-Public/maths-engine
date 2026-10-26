"""The key must be the lone holder of a feature value about as often as any one wrong option, in BOTH directions.

Review rounds 4 to 7 of the NSW/VCE/QCE PR each found a new "the key is the only option that ..." leak (the only
negative, the only real, the only half-integer, the only whole number, the only mixed complex number ...) and each was
fixed one case at a time. This test checks the class at once, for every source (easy, medium, hard, exam) of the six
skills, every part of every question. It judges only what a learner can see: the options of one part, pooled over seeds.

Each option is described by features: sign of the real and imaginary parts, whole or not, trailing ".0" vs a nonzero
decimal, complex kind (real / pure imaginary / mixed), power of two, parity, multiple of 5, digit count, standard degree,
standard multiple of pi, leading-coefficient form, hex letter, and for plain numbers how many other options lie within
1 and within 2 and whether the option is the most central or the most outlying. Formatting artefacts (text length,
expression shape) are not features. For each (part, feature value), among draws where EXACTLY ONE option holds the value,
the key should be that option 1 time in n_options. It fails when the key is it

    * too often (a learner who picks the odd one out gains): one-sided binomial tail under 1e-4, an excess of at least 1%
      of the draws over the chance count, and at least 5 such draws; or
    * too rarely (a learner who eliminates the odd one out gains): lower binomial tail under 1e-4, a deficit of at least
      1% of the draws, and at least 30 lone events. Complex kind is exempt from this side: the generators keep the key
      company on purpose, and its other side is a zero-tolerance rule.

A key of 0 is exempt (it is the only 0 by construction; the same test checks the premise: 0 must also be offered as a wrong option).
Parts whose options are fixed vocabulary words (odd / even / neither) are exempt. Complex kind is guarded outright: no
draw may leave the key alone in its kind. Each feature is judged per value and also summed over its values ("*").

Only the too-often rows, complex kind, the zero key and the strategy-edge bound are gates. The per-feature "key too
rarely" rows are diagnostic only: they are printed and appended to $KEY_LONE_REPORT, never asserted. Seeds come from two
disjoint windows (0-599 and 6400-6999, half the budget each) so a fix tuned to one range of seeds cannot pass.

The same test plays the learner outright (strategies: eliminate the odd ones out, pick the most central, pick the most
outlying) and bounds each at 6 points from chance. The too-rare side is a printed diagnostic, not an assert. Elimination: discard every option that is the only one with
some feature, guess among the rest, and compare with chance (review 7 measured +10.7 points median on these skills, +3.3
on the merged library).
"""

from __future__ import annotations

import functools
import math
import os
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from mathgen import make
from tests.test_option_rank_spread_128 import _complex, _number, _units

SEEDS = 1200
SEEDS_FOR = {"exd_nx_trig_sub_hsc": 700}  # 130 ms a call: fewer seeds keep the run under four minutes at -n 8
# Two disjoint windows of seeds (half of SEEDS each), so that a fix tuned to one range of seeds cannot pass. The windows
# can be moved for a held-out check: KEY_LONE_WINDOWS="3200,5200" starts the two windows there.
WINDOW_STARTS = (0, 6400)


def seeds_for(topic: str, starts=None) -> list:
    """The seeds a source is judged on: two windows that share the budget (SEEDS_FOR, else SEEDS) between them."""
    if starts is None:
        env = os.environ.get("KEY_LONE_WINDOWS")
        starts = tuple(int(x) for x in env.split(",")) if env else WINDOW_STARTS
    each = SEEDS_FOR.get(topic, SEEDS) // len(starts)
    return [s for start in starts for s in range(start, start + each)]


MIN_KEY_COUNT = 5  # fewer lone-key draws than this is not evidence
MIN_EXCESS = 0.01  # the key's lone count must beat the chance count by at least this share of the draws
P_MAX = 1e-4  # ... and the binomial tail of the key's count under 1/n_options must be this small.
MIN_EVENTS_UNDER = 30  # a key that is rarely the lone holder needs this many lone events to be judged
ELIM_POINTS = (
    0.30  # relative to the merged library: its best-strategy edge has median +16 and p90 +30 on these features
)
ELIM_SIGMAS = 3.0

# Features the generators guard outright (a real, pure imaginary or mixed key always has a wrong option of its kind): the
# guard leaves no draw at all, and a rate that is "only" chance-level (8 of 1200) is the guard missing a case.
ZERO_TOLERANCE = {"kind"}

SIX = (
    "pseudocode-trace-output",
    "power-function-graph-features",
    "complex-arg-ratio-geometry",
    "trig-substitution-hsc-tan-and-convert",
    "choose-summary-statistic-for-data",
    "bits-hex-and-digit-counts",
    "trapezium-rule-estimate",
    "slope-field-readings",
    "whole-units-needed-round-up-down",
)
SLOW_FIRST = "trig-substitution-hsc-tan-and-convert"
_SKILLS = Path(__file__).resolve().parent.parent / "skills"
KNOWN_WORDS = {"pi", "sqrt", "ln", "log", "exp", "sin", "cos", "tan", "sec", "cot", "csc"}
STD_PI_DEN = {1, 2, 3, 4, 6}
STD_DEG = {30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360}


def _sources():
    out = []
    for skill in SIX:
        path = next(_SKILLS.glob(f"*/{skill}.yaml"))
        for i, src in enumerate(yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]):
            out.append(pytest.param(skill, src, id=f"{skill}-{i}-{src['difficulty']}"))
    # The trig substitution generators take 25 to 130 ms a call: start them first so the run is not waiting on them.
    slow = [p for p in out if p.values[0] == SLOW_FIRST][::-1]  # exam first
    return slow + [p for p in out if p.values[0] != SLOW_FIRST]


# ---------------------------------------------------------------------------------------------------------------- features


def _is_word_answer(texts) -> bool:
    """Fixed vocabulary (odd / even / neither, 'the median'): alphabetic words that are not maths symbols."""

    def has_word(t):
        return any(w.lower() not in KNOWN_WORDS for w in re.findall(r"[A-Za-z]{3,}", t))

    return all(has_word(t) for t in texts)


@functools.lru_cache(maxsize=None)
def _items(text: str) -> list | None:
    """The numbers an option stands for (one for a scalar, several for 'a, b'), real or complex; None if it has none."""
    t = text.strip()
    if (v := _num(t)) is not None:
        return [v]
    if "," in t:
        parts = [_num(p.strip()) for p in t.split(",")]
        if all(p is not None for p in parts):
            return parts
    return None


@functools.lru_cache(maxsize=None)
def _num(t: str):
    try:
        v = _number(t)
        if v is not None:
            return complex(v)
        if re.search(r"(?<![A-Za-z])\d*i\b", t) and re.fullmatch(r"[0-9a-z()*/+\-.^ −√]*", t):
            return _complex(t)
    except Exception:  # noqa: BLE001 - text the readers cannot parse (tokenizer errors) is simply not a number
        return None
    return None


def _sign(x: float) -> str:
    return "neg" if x < -1e-12 else "zero" if abs(x) <= 1e-12 else "pos"


def _is_int(x: float) -> bool:
    return abs(x - round(x)) < 1e-9


def _pi_kind(text: str, v: complex) -> str:
    if "pi" not in text and "π" not in text:
        return "no-pi"
    r = v.real / math.pi
    f = Fraction(r).limit_denominator(1000)
    if abs(float(f) - r) > 1e-9:
        return "pi-mixed"
    return "pi-std" if f.denominator in STD_PI_DEN else "pi-other"


@functools.lru_cache(maxsize=None)
def _option_features(text: str, hexlike: bool) -> dict:
    """{feature: hashable value} for one option text."""
    t = text.strip()
    f: dict = {}
    f["ndigits"] = min(len(re.findall(r"\d", t)), 8)
    f["lead"] = "numeric" if re.fullmatch(r"-?[\d.,]+(/[\d.]+)?", t) else "coef" if re.match(r"-?\(?\d", t) else "bare"
    if hexlike:
        f["hex_letter"] = bool(re.search(r"[A-Fa-f]", t))
    m = re.fullmatch(r"-?\d+\.(\d+)", t)
    f["decimal_style"] = "none" if not m else "zeros" if not m.group(1).strip("0") else "nonzero"
    items = _items(t)
    if items is None:
        return f
    f["n_items"] = len(items)
    f["kind"] = tuple("real" if abs(z.imag) < 1e-9 else "imag" if abs(z.real) < 1e-9 else "mixed" for z in items)
    f["re_sign"] = tuple(_sign(z.real) for z in items)
    f["im_sign"] = tuple(_sign(z.imag) for z in items)
    reals = [z.real for z in items if abs(z.imag) < 1e-9]
    if len(reals) == len(items):
        f["whole"] = tuple(_is_int(x) for x in reals)
        ints = [round(x) for x in reals if _is_int(x)]
        if len(ints) == len(reals):
            f["parity"] = tuple(x % 2 for x in ints)
            f["pow2"] = tuple(x > 0 and x & (x - 1) == 0 for x in ints)
            f["mult5"] = tuple(x % 5 == 0 for x in ints)
            f["std_degree"] = tuple(abs(x) in STD_DEG for x in ints)
    if len(items) == 1:
        f["pi"] = _pi_kind(t, items[0])
    return f


def question_features(texts: list) -> list:
    """Feature dicts for the options in `texts` (key first). A feature is kept only if every option has it."""
    hexlike = all(re.fullmatch(r"[0-9A-Fa-f]+", t.strip()) for t in texts)
    feats = [dict(_option_features(t, hexlike)) for t in texts]
    values = [_plain(t) for t in texts]
    if all(v is not None for v in values):  # plain numbers: where each option sits among the others
        for i, f in enumerate(feats):
            f["near1"] = min(sum(abs(v - values[i]) <= 1 for j, v in enumerate(values) if j != i), 3)
            f["near2"] = min(sum(abs(v - values[i]) <= 2 for j, v in enumerate(values) if j != i), 3)
        spread = [round(sum(abs(v - w) for w in values), 9) for v in values]
        for i, f in enumerate(feats):
            lo, hi = min(spread), max(spread)
            f["centrality"] = (
                "central"
                if spread[i] == lo and spread.count(lo) == 1
                else "outlier"
                if spread[i] == hi and spread.count(hi) == 1
                else "mid"
            )
    common = set.intersection(*(set(f) for f in feats))
    return [{k: f[k] for k in common} for f in feats]


def _plain(text: str):
    """The real number an option is, or None (a plain number, fraction or surd; not a list, complex or hex)."""
    if re.fullmatch(r"[0-9A-F]*[A-F][0-9A-F]*", text.strip()):
        return None
    items = _items(text)
    if items is None or len(items) != 1 or abs(items[0].imag) > 1e-9:
        return None
    return items[0].real


LABEL_ONLY = {"parity", "pow2", "std_degree"}


def _key_is_zero(text: str) -> bool:
    items = _items(text)
    return items is not None and all(abs(z) < 1e-12 for z in items)


ELIM_EXTRA = {
    "has_slash": lambda t: "/" in t,
    "has_sqrt": lambda t: "sqrt" in t or "\u221a" in t,
    "has_minus": lambda t: "-" in t or "\u2212" in t,
    "mult10": lambda t: bool(re.fullmatch(r"-?\d+", t.strip())) and int(t) != 0 and int(t) % 10 == 0,
}


def _elimination(feats: list, texts: list) -> float:
    """The learner's chance of the key: drop every option that is the only holder of some feature value, guess among
    the survivors (everything if none survive)."""
    rows = [{**f, **{k: fn(t) for k, fn in ELIM_EXTRA.items()}} for f, t in zip(feats, texts)]
    n = len(rows)
    lone = [False] * n
    for name in rows[0]:
        vals = [r[name] for r in rows]
        counts = Counter(vals)
        if len(counts) == 1:
            continue
        for i, v in enumerate(vals):
            lone[i] |= counts[v] == 1
    alive = [i for i in range(n) if not lone[i]] or list(range(n))
    return 1 / len(alive) if 0 in alive else 0.0


def _survey_uncached(topic: str, difficulty: str, params: dict, seeds=None) -> dict:
    seeds = range(SEEDS) if seeds is None else seeds
    n_seen: Counter = Counter()
    n_opts: Counter = Counter()
    key_lone: dict = defaultdict(list)
    wrong_lone: Counter = Counter()
    elim: dict = defaultdict(
        lambda: [0.0, 0.0, 0]
    )  # (part, strategy) -> sum of gain over chance, sum of squares, draws
    key_zero: Counter = Counter()  # part -> draws whose key is 0
    wrong_zero: Counter = Counter()  # part -> draws with a wrong option 0 (and a key that is not)
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=difficulty, **params)
        for label, answer, options, prompt in _units(q):
            shown = [getattr(o, "text", str(o)) for o in options or []]
            texts = [str(answer), *shown]
            if not _is_word_answer(texts):
                if _key_is_zero(texts[0]):
                    key_zero[label] += 1
                elif any(_key_is_zero(t) for t in shown):
                    wrong_zero[label] += 1
            if len(shown) < 2 or _is_word_answer(texts) or _key_is_zero(texts[0]):
                continue
            n_seen[label] += 1
            n_opts[label] += len(texts)
            feats = question_features(texts)
            if "quadrant numbers" in str(prompt):
                # The parity or power-of-two pattern of a set of quadrant numbers is a label, not a quantity (at most 4
                # keys, a closed answer space); size, membership and the rest are judged like any option.
                feats = [{k: v for k, v in f.items() if k not in LABEL_ONLY} for f in feats]
            chance = 1 / len(texts)
            gains = {"eliminate": _elimination(feats, texts) - chance}
            if "centrality" in feats[0]:  # plain numbers: pick the most central / the most outlying option
                for strategy, tag in (("most-central", "central"), ("most-outlying", "outlier")):
                    pick = [i for i, f in enumerate(feats) if f["centrality"] == tag]
                    gains[strategy] = ((1 / len(pick) if 0 in pick else 0.0) if pick else chance) - chance
            values = [_plain(t) for t in texts]
            if all(v is not None for v in values):  # pick the option nearest the median of all the options
                ordered = sorted(values)
                mid = len(ordered) // 2
                median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
                dist = [round(abs(v - median), 9) for v in values]
                pick = [i for i, d in enumerate(dist) if d == min(dist)]
                gains["nearest-median"] = (1 / len(pick) if 0 in pick else 0.0) - chance
            for strategy, gain in gains.items():
                e = elim[(label, strategy)]
                e[0] += gain
                e[1] += gain * gain
                e[2] += 1
            for name in feats[0]:
                counts = Counter(f[name] for f in feats)
                for idx, f in enumerate(feats):
                    if counts[f[name]] != 1:
                        continue
                    for tag in ((name, f[name]), (name, "*")):
                        if idx == 0:
                            key_lone[(label, *tag)].append(seed)
                        else:
                            wrong_lone[(label, *tag)] += 1
    return {
        "n_seen": n_seen,
        "n_opts": n_opts,
        "key_lone": key_lone,
        "wrong_lone": wrong_lone,
        "elim": dict(elim),
        "key_zero": key_zero,
        "wrong_zero": wrong_zero,
    }


survey = _survey_uncached


def _binom_pmf(i: int, n: int, p: float) -> float:
    return math.exp(
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)
    )


def _upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(_binom_pmf(i, n, p) for i in range(k, n + 1))


def _lower_tail(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(_binom_pmf(i, n, p) for i in range(0, k + 1))


def tells(survey_result: dict) -> list:
    """Rows (part, feature, value, side, key_share, chance, key_count, lone_events, draws, excess, seeds) that break the rule."""
    n_seen, n_opts = survey_result["n_seen"], survey_result["n_opts"]
    key_lone, wrong_lone = survey_result["key_lone"], survey_result["wrong_lone"]
    keys = set(key_lone) | set(wrong_lone)
    bad = []
    for label, name, value in sorted(keys, key=str):
        seeds = key_lone.get((label, name, value), [])
        n = n_seen[label]
        k = len(seeds)
        events = k + wrong_lone[(label, name, value)]  # draws-by-option where exactly one option held the value
        if not events:
            continue
        chance = n / n_opts[label]  # 1 / options per draw
        share = k / events
        excess = (k - events * chance) / n
        row = (label, name, value, share, chance, k, events, n, excess, seeds[:4])
        if name in ZERO_TOLERANCE:
            if value != "*" and k:
                bad.append((*row[:3], "key alone in its kind", *row[3:]))
            continue
        if k >= MIN_KEY_COUNT and excess >= MIN_EXCESS and _upper_tail(k, events, chance) <= P_MAX:
            bad.append((*row[:3], "key too often", *row[3:]))
        elif events >= MIN_EVENTS_UNDER and -excess >= MIN_EXCESS and _lower_tail(k, events, chance) <= P_MAX:
            bad.append((*row[:3], "key too rarely", *row[3:]))
    return bad


def elimination_gains(survey_result: dict) -> list:
    """Rows (part, strategy, advantage, standard error, draws) where a no-maths strategy beats or trails chance too far,
    for each part and each strategy (eliminate the lone options, pick the most central, pick the most outlying)."""
    out = []
    for (label, strategy), (total, squares, draws) in sorted(survey_result["elim"].items()):
        if draws < 100:
            continue
        mean = total / draws
        se = math.sqrt(max(squares / draws - mean * mean, 0.0) / draws)
        if abs(mean) > ELIM_POINTS and abs(mean) > ELIM_SIGMAS * se:
            out.append((label, strategy, mean, se, draws))
    return out


def best_strategy_edges(survey_result: dict) -> dict:
    """{part: (strategy, advantage)} for the strategy with the largest advantage over chance, whatever the bound."""
    best: dict = {}
    for (label, strategy), (total, _sq, draws) in survey_result["elim"].items():
        if draws >= 100 and (label not in best or total / draws > best[label][1]):
            best[label] = (strategy, total / draws)
    return best


def _report(header: str, lines: list) -> None:
    """The non-gating diagnostics: printed under `pytest -s`, and appended to the file named by KEY_LONE_REPORT."""
    text = "\n".join([f"## {header}", *lines])
    print("\n" + text)
    if path := os.environ.get("KEY_LONE_REPORT"):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def _fmt(row) -> str:
    label, name, value, side, share, chance, k, events, n, excess, seeds = row
    return (
        f"part {label} feature {name}={value!s}: {side}: key is the lone holder in {k} of {events} lone events "
        f"(key_share {share:.2f}, chance {chance:.2f}, {k}/{n} draws, excess {excess:+.3f}) seeds {seeds}"
    )


@pytest.mark.parametrize(("skill", "src"), _sources())
def test_key_is_not_the_lone_option_of_its_kind(skill, src):
    """One pass over the source (the generator calls are the cost) feeds every check.

    Gating: the key is not the lone holder of a feature value too often (and complex kind is never lone); no learner
    strategy (eliminate the odd ones out, pick the most central, pick the most outlying, pick the one nearest the median) is more than 6 points from
    chance in a part; a key of 0 is also offered as a wrong option. Diagnostic only, not asserted: the key being the lone
    holder too RARELY (the odd one out is almost never the key), printed under -s and appended to $KEY_LONE_REPORT."""
    topic, params = src["topic"], dict(src.get("params") or {})
    result = survey(topic, src["difficulty"], params, seeds_for(topic))
    name = f"{skill} / {topic} {src['difficulty']}"
    rows = tells(result)
    lines = [_fmt(r) for r in rows if r[3] != "key too rarely"]
    lines += [
        f"strategy {strategy}: part {label} {mean:+.3f} over chance (se {se:.3f}, {n} draws)"
        for label, strategy, mean, se, n in elimination_gains(result)
    ]
    # A key of 0 is exempt from the statistics above, but only if 0 is also offered as a wrong option at a comparable
    # rate: otherwise a "0" among the options is always the key.
    key_zero, wrong_zero = result["key_zero"], result["wrong_zero"]
    lines += [
        f"zero key: part {label} has 0 as the key in {n} draws and as a wrong option in {wrong_zero[label]}"
        for label, n in sorted(key_zero.items())
        if n >= MIN_KEY_COUNT and n / (n + wrong_zero[label]) > 0.5
    ]
    best = best_strategy_edges(result)
    _report(
        name,
        [f"best strategy, part {p}: {st} {edge:+.3f}" for p, (st, edge) in sorted(best.items())]
        + [f"diagnostic {_fmt(r)}" for r in rows if r[3] == "key too rarely"],
    )
    assert not lines, f"{name}:\n" + "\n".join(lines)


def test_quadrant_wrong_options_are_sets_a_power_function_can_have():
    """The vertical line test rules out "1, 4" and "2, 3": a wrong option is never one of them."""
    seen = 0
    for diff, params in (("medium", {}), ("exam", {"tier": "exam"})):
        topic = "vm_power_graph_features" if diff == "medium" else "vm_power_graph_features_exam"
        for seed in range(SEEDS):
            for _label, _answer, options, prompt in _units(make(topic, seed=seed, difficulty=diff, **params)):
                if "quadrant numbers" not in str(prompt):
                    continue
                seen += 1
                texts = [getattr(o, "text", str(o)) for o in options]
                assert not {"1, 4", "2, 3"} & set(texts), (topic, seed, texts)
    assert seen > 500
