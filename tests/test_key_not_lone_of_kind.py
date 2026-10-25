"""The key must not be the lone holder of a feature value more often than chance, as a learner would see it.

Review rounds 4 to 6 of the NSW/VCE/QCE PR each found a new "the key is the only option that ..." leak (the only
negative, the only real, the only half-integer, the only whole number, the only mixed complex number ...) and each was
fixed one case at a time. This test checks the class at once, for every source (easy, medium, hard, exam) of the six
skills, every part of every question. It judges only what a learner can see: the options of one part, pooled over seeds.

Each option is described by features: sign of the real and imaginary parts, whole or not, trailing ".0" vs a nonzero
decimal, complex kind (real / pure imaginary / mixed), power of two, parity, digit count, standard degree, standard
multiple of pi, leading-coefficient form and hex letter. Formatting artefacts (text length, expression shape) are not
features. For each (part, feature value), among draws where EXACTLY ONE option holds the value, the key should be that
option 1 time in n_options. The test fails when the key is it more often, by all of

    * a one-sided binomial tail under 1e-4 against 1/n_options (the few hundred comparisons give no false alarms),
    * an excess of at least 1% of the draws over the chance count, and
    * at least 5 draws where the key is the lone holder.

A key of 0 is exempt (it is the only 0 by construction). Parts whose options are fixed vocabulary words (odd / even /
neither) are exempt: they are the whole answer space. Complex kind is guarded outright: no draw may leave the key alone
in its kind, because the generators promise a same-kind wrong option. Each feature is judged per value and also summed
over its values ("*").
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
MIN_KEY_COUNT = 5  # fewer lone-key draws than this is not evidence
MIN_EXCESS = 0.01  # the key's lone count must beat the chance count by at least this share of the draws
P_MAX = 1e-4  # ... and the binomial tail of the key's count under 1/n_options must be this small.

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
            f["std_degree"] = tuple(abs(x) in STD_DEG for x in ints)
    if len(items) == 1:
        f["pi"] = _pi_kind(t, items[0])
    return f


def question_features(texts: list) -> list:
    """Feature dicts for the options in `texts` (key first). A feature is kept only if every option has it."""
    hexlike = all(re.fullmatch(r"[0-9A-Fa-f]+", t.strip()) for t in texts)
    feats = [option_features(t, hexlike) for t in texts]
    common = set.intersection(*(set(f) for f in feats))
    return [{k: f[k] for k in common} for f in feats]


def _key_is_zero(text: str) -> bool:
    items = _items(text)
    return items is not None and all(abs(z) < 1e-12 for z in items)


def survey(topic: str, difficulty: str, params: dict, seeds=range(SEEDS)):
    """Per part: draws inspected, options per draw, and per (feature, value) the seeds where the key is the lone holder
    and the count of draws where a wrong option is."""
    n_seen: Counter = Counter()
    n_opts: Counter = Counter()
    key_lone: dict = defaultdict(list)
    wrong_lone: Counter = Counter()
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=difficulty, **params)
        for label, answer, options, _prompt in _units(q):
            shown = [getattr(o, "text", str(o)) for o in options or []]
            texts = [str(answer), *shown]
            if len(shown) < 2 or _is_word_answer(texts) or _key_is_zero(texts[0]):
                continue
            n_seen[label] += 1
            n_opts[label] += len(texts)
            feats = question_features(texts)
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
    return n_seen, n_opts, key_lone, wrong_lone


def _tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
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


def tells(n_seen, n_opts, key_lone, wrong_lone) -> list:
    """Rows (part, feature, value, key_share, chance, key_count, lone_events, draws, excess, seeds) that break the rule."""
    bad = []
    for (label, name, value), seeds in sorted(key_lone.items(), key=lambda kv: str(kv[0])):
        n = n_seen[label]
        k = len(seeds)
        events = k + wrong_lone[(label, name, value)]  # draws-by-option where exactly one option held the value
        chance = n / n_opts[label]  # 1 / options per draw
        share = k / events
        excess = (k - events * chance) / n
        if name in ZERO_TOLERANCE and value != "*":
            bad.append((label, name, value, share, chance, k, events, n, excess, seeds[:4]))
            continue
        if k < MIN_KEY_COUNT or excess < MIN_EXCESS or _tail(k, events, chance) > P_MAX:
            continue
        bad.append((label, name, value, share, chance, k, events, n, excess, seeds[:4]))
    return bad


def _fmt(row) -> str:
    label, name, value, share, chance, k, events, n, excess, seeds = row
    return (
        f"part {label} feature {name}={value!s}: key is the lone holder in {k} of {events} lone events "
        f"(key_share {share:.2f}, chance {chance:.2f}, {k}/{n} draws, excess {excess:.3f}) seeds {seeds}"
    )


@pytest.mark.parametrize(("skill", "src"), _sources())
def test_key_is_not_the_lone_option_of_its_kind(skill, src):
    topic, params = src["topic"], dict(src.get("params") or {})
    bad = tells(*survey(topic, src["difficulty"], params))
    assert not bad, f"{skill} / {topic} {src['difficulty']}:\n" + "\n".join(_fmt(r) for r in bad)
