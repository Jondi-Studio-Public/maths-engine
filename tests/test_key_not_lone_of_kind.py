"""The key must never be the ONLY option of its kind more often than a wrong option is.

Review rounds 4 to 6 of the NSW/VCE/QCE PR each found a new "the key is the only option that ..." leak (the only
negative, the only real, the only half-integer, the only whole number, the only mixed complex number ...) and each was
fixed one case at a time. This test checks the whole class at once, for every source (easy, medium, hard, exam) of the
six skills, every part of every question.

For each option the text is described by a set of features (sign, whole or not, trailing ".0", complex kind, power of
two, parity, digit count, standard angle, hex letter, leading coefficient, expression shape, length extreme ...). A
feature value is a tell when the key is the ONLY option that has it. Per (source, part, feature) over SEEDS seeds:

    key_lone_rate  = share of seeds where the key has the value and no wrong option does
    baseline       = mean, over the wrong options, of the share of seeds where that option is the lone holder

The test fails when key_lone_rate beats the baseline by the margin below AND the excess is too large to be chance
(a one-sided binomial tail under 1e-4, so the few hundred comparisons made do not give false alarms). Parts whose
options are fixed vocabulary words (odd / even / neither) are exempt: they are the whole answer space and cannot leak.
Complex kind is a guarded feature: no draw may leave the key alone in its kind. Each feature is judged per value ("whole-number"), and also summed over its values ("*") for features with many values.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from mathgen import make
from tests.test_option_rank_spread_128 import _complex, _number, _units

SEEDS = 1200
MIN_STRATUM = 40  # a recipe of slips seen in fewer questions than this is judged only inside the whole part
MIN_KEY_COUNT = 5  # fewer lone-key draws than this is not evidence
MIN_EXCESS = 0.005  # the key's lone rate must beat the baseline by at least this much ...
RATIO = 1.5  # ... and by this factor ...
P_MAX = 1e-4  # ... and the binomial tail of the key's count under the baseline must be this small.

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
)
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
    return out


# ---------------------------------------------------------------------------------------------------------------- features


def _is_word_answer(texts) -> bool:
    """Fixed vocabulary (odd / even / neither, 'the median'): alphabetic words that are not maths symbols."""

    def has_word(t):
        return any(w.lower() not in KNOWN_WORDS for w in re.findall(r"[A-Za-z]{3,}", t))

    return all(has_word(t) for t in texts)


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


def _num(t: str):
    v = _number(t)
    if v is not None:
        return complex(v)
    if re.search(r"(?<![A-Za-z])\d*i\b", t) and re.fullmatch(r"[0-9a-z()*/+\-.^ −√]*", t):
        return _complex(t)
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


def option_features(text: str, hexlike: bool) -> dict:
    """{feature: hashable value} for one option text."""
    t = text.strip()
    f: dict = {}
    digits = re.findall(r"\d", t)
    f["ndigits"] = min(len(digits), 8)
    f["no_digit"] = not digits
    f["shape"] = re.sub(r"\d+", "#", t)
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
            f["mult10"] = tuple(x != 0 and x % 10 == 0 for x in ints)
            f["std_degree"] = tuple(abs(x) in STD_DEG for x in ints)
    if len(items) == 1:
        f["pi"] = _pi_kind(t, items[0])
    return f


def _length_extreme(texts: list) -> list:
    lens = [len(t) for t in texts]
    return [
        "longest"
        if n == max(lens) and lens.count(n) == 1
        else "shortest"
        if n == min(lens) and lens.count(n) == 1
        else "mid"
        for n in lens
    ]


def question_features(texts: list) -> list:
    """Feature dicts for the options in `texts` (key first). A feature is kept only if every option has it."""
    hexlike = all(re.fullmatch(r"[0-9A-Fa-f]+", t.strip()) for t in texts)
    feats = [option_features(t, hexlike) for t in texts]
    for f, ext in zip(feats, _length_extreme(texts)):
        f["length"] = ext
    common = set.intersection(*(set(f) for f in feats))
    return [{k: f[k] for k in common} for f in feats]


# ---------------------------------------------------------------------------------------------------------------- survey


def _strata(q, label: str, shown: list) -> list:
    """The groups a question is judged in: the whole part ("*"), each slip that made a wrong option ("has T": a key
    class is often tied to one slip, as an even-count median is to its mis-averaging slip), and the exact recipe of
    slips ("recipe T1 + T2 + T3"). Misconception tags name the slips; an untagged filler is "untagged"."""
    part = q.parts[int(label[1:])] if q.parts else None
    notes = {**q.option_notes, **(part.option_notes if part is not None else {})}
    tags = sorted(str(notes.get(t) or "untagged") for t in shown)
    return ["*", *(f"has {t}" for t in sorted(set(tags))), "recipe " + " + ".join(tags)]


def survey(topic: str, difficulty: str, params: dict, seeds=range(SEEDS)):
    """Counts per (part, stratum) where stratum is "*" (the whole part) or the recipe of slips behind the wrong options.

    Strata matter: a key that is the only half-integer in every draw where the slips are all whole numbers is lost in the
    average over the whole part, because other recipes put a lone non-whole option on the wrong side."""
    n_seen: Counter = Counter()  # (part, stratum) -> questions inspected
    n_wrong: Counter = Counter()  # (part, stratum) -> wrong options inspected
    key_lone: dict = defaultdict(list)  # (part, stratum, feature, value) -> seeds where the key alone has it
    wrong_lone: Counter = Counter()  # same key -> wrong options that were alone, summed
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=difficulty, **params)
        for label, answer, options, _prompt in _units(q):
            shown = [getattr(o, "text", str(o)) for o in options or []]
            texts = [str(answer), *shown]
            if len(shown) < 2 or _is_word_answer(texts):
                continue
            strata = _strata(q, label, shown)
            for st in strata:
                n_seen[(label, st)] += 1
                n_wrong[(label, st)] += len(shown)
            feats = question_features(texts)
            for name in feats[0]:
                counts = Counter(f[name] for f in feats)
                for idx, f in enumerate(feats):
                    if counts[f[name]] != 1:
                        continue
                    for st in strata:
                        for tag in ((name, f[name]), (name, "*")):
                            if idx == 0:
                                key_lone[(label, st, *tag)].append(seed)
                            else:
                                wrong_lone[(label, st, *tag)] += 1
    return n_seen, n_wrong, key_lone, wrong_lone


def _tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact for the small k used here."""
    if p <= 0:
        return 0.0 if k > 0 else 1.0
    if p >= 1:
        return 1.0
    below = 0.0
    for i in range(k):
        below += math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * math.log(p)
            + (n - i) * math.log1p(-p)
        )
    return max(0.0, 1.0 - below)


def tells(n_seen, n_wrong, key_lone, wrong_lone) -> list:
    """Rows (part, stratum, feature, value, key_lone_rate, baseline, count, n, example seeds) that break the rule."""
    bad = []
    for (label, st, name, value), seeds in sorted(key_lone.items(), key=lambda kv: str(kv[0])):
        n = n_seen[(label, st)]
        if n < MIN_STRATUM:
            continue
        k = len(seeds)
        baseline = wrong_lone[(label, st, name, value)] / n_wrong[(label, st)]  # mean over wrong options
        rate = k / n
        if name in ZERO_TOLERANCE and st == "*" and value != "*":
            bad.append((label, st, name, value, rate, baseline, k, n, seeds[:4]))  # one draw is one too many
            continue
        if k < MIN_KEY_COUNT or rate <= max(baseline * RATIO, baseline + MIN_EXCESS):
            continue
        if _tail(k, n, baseline) > P_MAX:
            continue
        bad.append((label, st, name, value, rate, baseline, k, n, seeds[:4]))
    return bad


def _fmt(row) -> str:
    label, st, name, value, rate, baseline, k, n, seeds = row
    where = "whole part" if st == "*" else f"[{st}]"
    return (
        f"part {label} {where}, feature {name}={value!s}: key_lone_rate {rate:.3f} ({k}/{n}) "
        f"baseline {baseline:.3f} seeds {seeds}"
    )


@pytest.mark.parametrize(("skill", "src"), _sources())
def test_key_is_not_the_lone_option_of_its_kind(skill, src):
    topic, params = src["topic"], dict(src.get("params") or {})
    result = survey(topic, src["difficulty"], params)
    bad = tells(*result)
    assert not bad, f"{skill} / {topic} {src['difficulty']}:\n" + "\n".join(_fmt(r) for r in bad)
