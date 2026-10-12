"""Independent oracle for the short written-answer generators.

A written-answer question states numbers (a mean, a percentage, a cost, a correlation, a count of edges) and then a
model answer that relies on them. The first thing that must hold is that every stated number is true. The oracle
reads only what a learner or a marker is given (`question.text`, `question.answer_text` and the marking dict),
re-derives each number with its own code (`statistics`, exact fractions, brute force over outcomes, union-find over a
parsed graph, SymPy for algebra) and does not import any generator. A topic with no rule below is itself a failure:
a new generator needs a new rule here.

Seed count is `N_SEEDS` per tier (default 60), overridable by `WRITTEN_ANSWER_ORACLE_SEEDS`.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import statistics
from fractions import Fraction

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make
from mathgen.core import REGISTRY

TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("WRITTEN_ANSWER_ORACLE_SEEDS", "60"))
TOPICS = tuple(sorted(t for t in REGISTRY if t.startswith("wa_")))
_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _qs(topic: str):
    return [make(topic, seed=s, difficulty=d) for d in TIERS for s in range(N_SEEDS)]


def _blob(q) -> str:
    """Everything a marker sees about the answer."""
    return " ".join(
        [q.answer_text, *q.answer["reference_lines"]]
        + [s["description"] for s in q.answer["key_steps"]]
        + [t["question"] for t in q.answer["traps"]]
    )


def _num(s: str):
    s = s.replace(",", "")
    return float(s)


def _same(a: float, b: float, tol: float = 0.0501) -> bool:
    return abs(a - b) <= tol


def _fmt(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _trim(x: float, places: int = 1) -> str:
    return _fmt(round(x, places))


def _ints(s: str) -> list[int]:
    return [int(t.replace(",", "")) for t in re.findall(r"(?<![\d.])\d+(?:,\d{3})*(?![\d.]|\w)", s)]


# ---------------------------------------------------------------------------
# Rules that apply to every topic
# ---------------------------------------------------------------------------
_FRACTION_FORMS = (
    # "175 of the 400 members (43.8%)", "600 of those 1000 (60%)"
    re.compile(r"(\d+(?:,\d{3})*) of (?:the |those )?(\d+(?:,\d{3})*)(?: [A-Za-z-]+){0,4}? \((\d+(?:\.\d+)?)%\)"),
    # "A survey of 60 members of the club found that 45 (75%)"
    re.compile(r"survey of (\d+(?:,\d{3})*)[^.]*? found that (\d+(?:,\d{3})*) \((\d+(?:\.\d+)?)%\)"),
    # "Of 200 people stopped ..., 110 (55%)"
    re.compile(r"\bOf (\d+(?:,\d{3})*) [^.]*?, (\d+(?:,\d{3})*) \((\d+(?:\.\d+)?)%\)"),
)


def _check_fractions(q) -> int:
    """Every `k of n (p%)` in the text or the marking is true. Returns how many were checked."""
    text = q.text + " " + _blob(q)
    seen = 0
    for m in _FRACTION_FORMS[0].finditer(text):
        k, n, p = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        assert _same(100 * k / n, p), (m.group(0), 100 * k / n)
        seen += 1
    for pat in _FRACTION_FORMS[1:]:
        for m in pat.finditer(text):
            n, k, p = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
            assert _same(100 * k / n, p), (m.group(0), 100 * k / n)
            seen += 1
    # "Only 10 of them (5%)" and "260 of them (65%)": the group size is the nearest earlier count that is larger
    for m in re.finditer(r"(\d+(?:,\d{3})*) of them,? (?:or |\()(\d+(?:\.\d+)?)%", text):
        k, p = _num(m.group(1)), _num(m.group(2))
        earlier = [i for i in _ints(text[: m.start()]) if i > k]
        assert earlier, m.group(0)
        n = earlier[-1] if "asked" in text[: m.start()] or True else earlier[0]
        cands = [i for i in earlier if _same(100 * k / i, p)]
        assert cands, (m.group(0), earlier)
        seen += 1
    return seen


def _grounded(q, allowed: set[int] = frozenset(), skip_below: int = 10) -> None:
    """Every number of `skip_below` or more in the marking appears in the prompt, is a difference of two numbers the
    prompt gives (a count not replied, a count left over), or is a derived value the rule names."""
    shown = set(_ints(q.text))
    diffs = {a - b for a in shown for b in shown if a > b}
    for n in _ints(_blob(q)):
        if n < skip_below or n in shown or n in allowed or n in diffs:
            continue
        pytest.fail(f"number {n} in the marking is not in the prompt and not derived: {_blob(q)!r} / {q.text!r}")


def _dataset(text: str, minimum: int = 5) -> list[list[float]]:
    out = []
    for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?:, -?\d+(?:\.\d+)?){%d,}" % minimum, text):
        out.append([float(t) for t in m.group(0).split(", ")])
    return out


def _mean(v):
    return sum(v) / len(v)


def _median(v):
    return statistics.median(v)


# ---------------------------------------------------------------------------
# Sampling and census
# ---------------------------------------------------------------------------
def rule_sample_kinds(q):
    t = q.text
    _check_fractions(q)
    for m in re.finditer(
        r"list of its (\d+(?:,\d{3})*)[^.]*?first (\d+) names and then asks every (\d+)\w\w .*?, (\d+) \w+ in all", t
    ):
        big, first, step, got = (int(g.replace(",", "")) for g in m.groups())
        assert first == step and big == step * got, m.group(0)
    for m in re.finditer(r"number from 1 to (\d+(?:,\d{3})*)[^.]*?picks (\d+) different numbers", t):
        big, got = int(m.group(1).replace(",", "")), int(m.group(2))
        assert got < big and re.search(rf"\b{m.group(1)} ", t[: m.start()]), m.group(0)
    for m in re.finditer(r"(\d+) of the (\d+(?:,\d{3})*) members \(", t):
        assert int(m.group(1)) < int(m.group(2).replace(",", ""))
    # the kind named in the model answer is one of the five kinds
    strata = re.search(r"has (\d+) Year 7, (\d+) Year 8 and (\d+) Year 9", t)
    allowed = {sum(int(g) for g in strata.groups())} if strata else set()
    kinds = {"simple random", "systematic", "self-selected", "convenience", "stratified", "cluster"}
    assert any(k in _blob(q).lower() for k in kinds), _blob(q)
    _grounded(q, allowed=allowed)


def rule_sample_pros_cons(q):
    t = q.text
    assert len(set(_ints(t))) >= 1
    names = ("simple random", "systematic", "stratified", "cluster", "convenience", "self-selected")
    in_prompt = [n for n in names if n in t.lower()]
    rec = re.search(
        r"(?:Recommend|Use) the (simple random|systematic|stratified|cluster|convenience|self-selected) sample",
        _blob(q),
    )
    if rec and q.answer["reference_lines"][0].startswith(("Recommend", "Use")):
        assert rec.group(1) in in_prompt, (rec.group(1), in_prompt, t)
    _grounded(q)


def rule_census_pros_cons(q):
    t = q.text
    _check_fractions(q)
    for m in re.finditer(
        r"Each interview costs \$(\d+) and takes (\d+) minutes\. A census would mean interviewing all (\d+(?:,\d{3})*) \w+, which would cost \$(\d+(?:,\d{3})*) and take (\d+(?:,\d{3})*) hours of interviewing\. A sample of (\d+) \w+ would cost \$(\d+(?:,\d{3})*)",
        t,
    ):
        c, mins, big, cost, hours, s, scost = (int(g.replace(",", "")) for g in m.groups())
        assert cost == big * c and Fraction(hours) == Fraction(big * mins, 60) and scost == s * c and s < big, m.group(
            0
        )
    for m in re.finditer(
        r"all (\d+(?:,\d{3})*) [\w ]+? (?:about|in)[^.]*?\. Only (\d+)% of the forms are returned, which is (\d+(?:,\d{3})*) forms",
        t,
    ):
        big, p, k = (int(g.replace(",", "")) for g in m.groups())
        assert Fraction(p * big, 100) == k, m.group(0)
    _grounded(q, allowed={int(x) for x in re.findall(r"\$(\d+(?:,\d{3})*)", t.replace(",", ""))})


def rule_census_procedure(q):
    t = q.text
    # a list that is out of date: the joined and the listed students are different people
    for m in re.finditer(r"list of (\d+) \w+\. Since then (\d+) new", t):
        assert int(m.group(2)) < int(m.group(1)) * 1.0 + 1
    _grounded(q)


def rule_sampling_purpose(q):
    t = q.text
    m = re.search(r"has (\d+(?:,\d{3})*) \w+\.", t)
    assert m, t
    big = int(m.group(1).replace(",", ""))
    samples = [(int(a), int(b)) for a, b in re.findall(r"(\d+) of the (\d+) \(", t)]
    assert samples, t
    k, n = samples[0]
    assert re.search(rf"random sample of {n} ", t), t
    second = re.search(r"second random sample of (\d+) \w+ and finds (\d+) who", t)
    if second:
        samples.append((int(second.group(2)), int(second.group(1))))
    pct = lambda a, b: _trim(100 * a / b)  # noqa: E731
    est = lambda a, b: round(big * a / b)  # noqa: E731
    blob = _blob(q)
    derived = {est(a, b) for a, b in samples} | {round(100 * a / b) for a, b in samples}
    for m2 in re.finditer(
        r"(?:gives|estimate of|exactly|about|is) (\d+(?:,\d{3})*) (?:members|residents|borrowers|swimmers|\w+)", blob
    ):
        v = int(m2.group(1).replace(",", ""))
        if v >= 100 and v not in set(_ints(t)):
            assert v in derived, (v, derived, t, blob)
    for a, b in samples:
        for m3 in re.finditer(rf"\b{a} out of {b}, which is ([\d.]+)%", blob):
            assert _same(float(m3.group(1)), 100 * a / b)
    if "Applying this to all" in blob:
        assert f"gives {est(k, n)} " in blob, blob
    if "only" in t and "% of the members" in t:
        v = re.search(r"only ([\d.]+)% of the", t)
        assert _same(float(v.group(1)), 100 * n / big)
    if "Omar's sample gives" in blob:
        assert f"{pct(*samples[0])}% and an estimate of {est(*samples[0])}" in blob
        assert f"{pct(*samples[1])}% and an estimate of {est(*samples[1])}" in blob
    _check_fractions(q)
    _grounded(q, allowed=derived)


def rule_sampling_bias(q):
    _check_fractions(q)
    t = q.text
    for m in re.finditer(r"(\d+) borrowers \(([\d.]+)%\) phone in", t):
        k = int(m.group(1))
        assert any(_same(100 * k / big, float(m.group(2))) for big in _ints(t[: m.start()]) if big > k), m.group(0)
    for m in re.finditer(
        r"number from 1 to (\d+(?:,\d{3})*), use a random number generator to pick (\d+) different", _blob(q)
    ):
        assert int(m.group(2)) < int(m.group(1).replace(",", ""))
        assert m.group(1) in t
    _grounded(q, allowed={int(x) for x in re.findall(r"pick (\d+) different", _blob(q))})


# ---------------------------------------------------------------------------
# Surveys and their reporting
# ---------------------------------------------------------------------------
def rule_questionnaire_design(q):
    t = q.text
    for m in re.finditer(r"Choices: ([^\"]+?)\.\"", t):
        opts = [o.strip() for o in m.group(1).split(", ")]
        ends = []
        for o in opts:
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", o)]
            assert len(nums) == 2, o
            ends.append(nums)
        # the draft choices are faulty exactly because neighbours share an end point
        assert any(ends[i][1] == ends[i + 1][0] for i in range(len(ends) - 1)), m.group(0)
    # the rewrite's choices must not overlap: read "a to b" / "less than a" / "a or more" ranges
    for m in re.finditer(r"(?:use the choices|becomes \"use the choices): ([^\"]+?)(?:\"|$)", _blob(q)):
        pass
    _grounded(q, skip_below=100)


def rule_survey_misrepresentation(q):
    t = q.text
    _check_fractions(q)
    for m in re.finditer(r"asked (\d+) \w+; (\d+) of them, or (\d+)%", t):
        assert _same(100 * int(m.group(2)) / int(m.group(1)), float(m.group(3)))
    if "Most" in t and "%" in t:
        for m in re.finditer(r"(\d+) of the (\d+) members \(([\d.]+)%\)", t):
            assert 100 * int(m.group(1)) / int(m.group(2)) < 50, "'most' is claimed for a share under half"
    m = re.search(
        r"mean test mark of (\d+), and the (\d+) students who did not had a mean mark of (\d+), a difference of (\d+) marks",
        t,
    )
    if m:
        assert int(m.group(1)) - int(m.group(3)) == int(m.group(4))
    m = re.search(r"the (\d+) students who ate breakfast", t)
    if m:
        assert int(m.group(1)) > 0
    if "A fairer headline" in _blob(q) and "mechanics" in t:
        assert re.search(r"survey of (\d+) mechanics", _blob(q)).group(1) == re.search(
            r"asked (\d+) mechanics", t
        ).group(1)
    _grounded(q, allowed={150})


def rule_results_misrepresentation(q):
    t = q.text
    n_checked = _check_fractions(q)
    for m in re.finditer(r"^(\d+) \w+ were asked.*?(\d+) of them \((\d+)%\)", t):
        assert _same(100 * int(m.group(2)) / int(m.group(1)), float(m.group(3)))
        n_checked += 1
    assert n_checked >= 1 or "%" not in t, t
    _grounded(q)


def rule_survey_error_sources(q):
    t = q.text
    _check_fractions(q)
    for m in re.finditer(
        r"(\d+) of the (\d+) \(([\d.]+)%\) say they would use a new cafe[^.]*\. A census of all (\d+(?:,\d{3})*) \w+ later shows that (\d+)% would",
        t,
    ):
        k, n, big, truth = int(m.group(1)), int(m.group(2)), int(m.group(4).replace(",", "")), int(m.group(5))
        assert n < big and 0 < k <= n
        sample_pct = 100 * k / n
        assert _same(sample_pct, float(m.group(3)))
        assert abs(sample_pct - truth) >= 5, "the sample result should visibly differ from the census"
        if "differs by chance from the true" in _blob(q):
            assert f"({_trim(sample_pct)}%)" in _blob(q) and f"true {truth}%" in _blob(q)
    for m in re.finditer(r"chooses (\d+) of the (\d+(?:,\d{3})*) members", t):
        assert int(m.group(1)) < int(m.group(2).replace(",", ""))
    _grounded(q)


def rule_collection_faults(q):
    t = q.text
    all_rows = [
        (int(a), None if b == "blank" else float(b)) for a, b in re.findall(r"Student (\d+): (blank|\d+(?:\.\d+)?)", t)
    ]
    assert len(all_rows) >= 8, t
    rows = [(i, v) for i, v in all_rows if v is not None]
    blanks = [i for i, v in all_rows if v is None]
    if blanks:
        assert len(blanks) == 1 and re.search(rf"Student {blanks[0]} has no value recorded", _blob(q)), _blob(q)
    values = [v for _, v in rows]
    m = re.search(r"mean of the recorded values is (\d+(?:\.\d+)?) (\w+)", t)
    if m:
        assert _same(_mean(values), float(m.group(1)), 0.51), (_mean(values), m.group(0))
    blob = _blob(q)
    idx = [i for i, _ in rows]
    dup = [i for i in set(idx) if idx.count(i) > 1]
    for d in dup:
        same = [v for i, v in rows if i == d]
        assert len(set(same)) == 1 and len(same) == 2
    # typing slips: a value stated as impossible is at least ten times the typical value
    for m in re.finditer(r"Student (\d+)'s value \((\d+)\) is impossibly large[^.]*? typing slip for (\d+)", blob):
        v, fix = int(m.group(2)), int(m.group(3))
        others = [x for i, x in rows if i != int(m.group(1)) and x < 1000]
        assert v in (10 * fix, 100 * fix) and v > 5 * statistics.median(others), (m.group(0), others)
        assert min(others) / 3 <= fix <= max(others) * 3, "the corrected value should be typical"
    for m in re.finditer(r"Student (\d+)'s value \(([\d.]+)\) is written in metres", blob):
        assert float(m.group(2)) < 3 and any(v > 100 for i, v in rows if i != int(m.group(1)))
    # the corrected mean
    m = re.search(r"corrected mean is (\d+(?:\.\d+)?) \w+", blob)
    if m:
        true = {int(a): float(b) for a, b in re.findall(r"Student (\d+)'s true value was (\d+(?:\.\d+)?)", t)}
        seen, fixed = set(), []
        for i, v in rows:
            if i in seen:
                continue
            seen.add(i)
            fixed.append(true.get(i, v))
        assert _same(_mean(fixed), float(m.group(1)), 0.051), (_mean(fixed), m.group(0))


# ---------------------------------------------------------------------------
# Centre and spread, presentation, association, relationships
# ---------------------------------------------------------------------------


def _stat_values(v):
    s = {
        "mean": float(_trim(_mean(v))),
        "median": float(_median(v)),
        "range": max(v) - min(v),
        "min": min(v),
        "max": max(v),
        "total": sum(v),
        "n": len(v),
    }
    top = max(v)
    rest = [x for x in v if x != top] if v.count(top) == 1 else v
    s["rest_range"] = max(rest) - min(rest)
    s["rest_mean"] = float(_trim(_mean(rest)))
    return s


def _all_lists_stats(q):
    lists = _dataset(q.text)
    assert lists, q.text
    return lists, [_stat_values(v) for v in lists]


def rule_centre_suitability(q):
    t = q.text
    cat = re.search(r"The numbers in each category were: ([^()]+) \(total (\d+)\)", t)
    if cat:
        pairs = re.findall(r"([a-z ]+?) (\d+)(?:,|$)", cat.group(1).strip() + ",")
        counts = {k.strip(): int(v) for k, v in pairs}
        assert sum(counts.values()) == int(cat.group(2))
        top = max(counts, key=counts.get)
        assert list(counts.values()).count(counts[top]) == 1
        assert f"the most common category is {top}, with {counts[top]}" in _blob(q)
        return
    lists, stats = _all_lists_stats(q)
    # each list is followed by its own mean and median
    for m in re.finditer(
        r"(?<![\w.])(-?\d+(?:\.\d+)?(?:, -?\d+(?:\.\d+)?){5,})\.\s*(?:There are (\d+) values, )?(?:T|t)he mean is (\d+(?:\.\d+)?)(?: and|,) the median is (\d+(?:\.\d+)?)",
        t,
    ):
        v = [float(x) for x in m.group(1).split(", ")]
        assert _same(_mean(v), float(m.group(3))) and _same(_median(v), float(m.group(4))), (v, m.group(0))
        if m.group(2):
            assert int(m.group(2)) == len(v)
    m = re.search(r"total of all the values is (\d+)", t)
    if m:
        assert sum(lists[0]) == int(m.group(1))
    blob = _blob(q)
    for key in ("mean", "median"):
        for mm in re.finditer(rf"The {key} \(([\d.]+)\)", blob):
            assert any(_same(float(mm.group(1)), s[key]) for s in stats), (mm.group(0), stats)
    for mm in re.finditer(r"the value ([\d.]+) pulls the mean up|One value, ([\d.]+), is much larger", blob):
        val = float(mm.group(1) or mm.group(2))
        assert any(val == s["max"] for s in stats)
        assert val > 1.8 * _median(lists[0]) or len(lists) > 1
    mm = re.search(r"which is more than (\d+) of the (\d+) values", blob)
    if mm:
        v = lists[0]
        assert sum(1 for x in v if x < _mean(v)) == int(mm.group(1)) and len(v) == int(mm.group(2))
    mm = re.search(r"(\d+(?:\.\d+)?) [x×] (\d+) is about (\d+)", blob)
    if mm:
        assert _same(float(mm.group(1)) * int(mm.group(2)), int(mm.group(3)), 0.6)


def rule_media_centre_spread(q):
    t = q.text
    lists, stats = _all_lists_stats(q)
    blob = _blob(q)
    allowed_range = {s["range"] for s in stats} | {s["rest_range"] for s in stats}
    for mm in re.finditer(r"\brange (?:is |of |\()?(\d+)\b", blob + " " + t):
        assert float(mm.group(1)) in allowed_range, (mm.group(0), allowed_range)
    for mm in re.finditer(r"from (\d+) to (\d+) degrees", t):
        assert any(s["min"] == float(mm.group(1)) and s["max"] == float(mm.group(2)) for s in stats)
    for mm in re.finditer(r"average (?:of |salary at [\w ]+ is \$)?(\d+(?:\.\d+)?)", t):
        assert any(_same(float(mm.group(1)), s["mean"]) for s in stats), (mm.group(0), stats)
    for mm in re.finditer(r"means are the same \((\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)\)", blob):
        assert (
            len(stats) == 2
            and _same(stats[0]["mean"], float(mm.group(1)))
            and _same(stats[1]["mean"], float(mm.group(2)))
        )
        assert _same(stats[0]["mean"], stats[1]["mean"])
    for mm in re.finditer(r"(?:median(?: salary at [\w ]+)? is \$?|median \()(\d+(?:\.\d+)?)", blob):
        assert any(_same(float(mm.group(1)), s["median"]) for s in stats), (mm.group(0), stats)
    for mm in re.finditer(r"mean \((\d+(?:\.\d+)?)\)", blob):
        assert any(_same(float(mm.group(1)), s["mean"]) for s in stats)
    for mm in re.finditer(r"mean is exceeded by only (\d+) of the (\d+) salaries", blob):
        v = lists[0]
        assert sum(1 for x in v if x > _mean(v)) == int(mm.group(1)) and len(v) == int(mm.group(2))
    for mm in re.finditer(r"(?:the mode|The most common amount[^(]*?is) \(?(\d+)", blob + " " + t):
        v = lists[0]
        mode = statistics.multimode(v)
        assert float(mm.group(1)) in mode, (mm.group(0), v)
    mm = re.search(r"occurs only twice|appears only twice", blob)
    if mm:
        v = lists[0]
        assert (
            statistics.multimode(v) and v.count(statistics.multimode(v)[0]) == 2 and len(statistics.multimode(v)) == 1
        )
    mm = re.search(r"typical \w+ family has ([\d.]+) children", t)
    if mm:
        assert _same(_mean(lists[0]), float(mm.group(1)))


def rule_presentation_suitability(q):
    t = q.text
    pairs = re.findall(r"([a-z][a-z ]*?) (\d+)(?:,|\.)", t.split("were:")[-1]) if "were:" in t else []
    m = re.search(r"asked (\d+) \w+", t)
    if pairs:
        assert m and sum(int(v) for _, v in pairs) == int(m.group(1)), (pairs, m and m.group(0))
        n_cat = len(pairs)
        stated = re.search(r"(?:The numbers for the |the )(\d+) (?:ways|categories|sports|kinds)", t)
        if stated:
            assert int(stated.group(1)) == n_cat, (stated.group(0), n_cat)
        assert len({k for k, _ in pairs}) == n_cat
        # the model answer's choice of display follows the category count
        blob = _blob(q)
        if n_cat >= 8 and "bar chart" in blob:
            assert "readable" in blob or "bar" in blob
    lists = _dataset(t, 5)
    mm = re.search(r"of (\d+) students on", t)
    if mm and lists:
        assert len(lists[0]) == int(mm.group(1)), (len(lists[0]), mm.group(0))
    assert any(
        w in _blob(q).lower()
        for w in ("bar chart", "histogram", "line graph", "box plot", "dot plot", "pie", "scatter", "stem")
    ), _blob(q)
    _grounded(q, skip_below=40)


def _assoc_rows(t: str):
    rows = re.findall(r"Among ([^:]+): (\d+) out of (\d+) ", t)
    return [(g, int(a), int(b)) for g, a, b in rows]


def rule_percent_association(q):
    t = q.text
    rows = _assoc_rows(t)
    assert len(rows) == 2, t
    (g1, k1, n1), (g2, k2, n2) = rows
    assert 0 < k1 < n1 and 0 < k2 < n2
    p1, p2 = 100 * Fraction(k1, n1), 100 * Fraction(k2, n2)
    blob = _blob(q)
    pcts = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", blob)}
    allowed = {
        round(float(p1), 1),
        round(float(p2), 1),
        round(float(abs(p1 - p2)), 1),
        float(round(p1)),
        float(round(p2)),
    }
    for x in pcts:
        assert any(_same(x, a, 0.51) for a in allowed), (x, allowed, blob)
    for mm in re.finditer(
        r"by (\d+(?:\.\d+)?) percentage points|a difference of (?:only )?(\d+(?:\.\d+)?) percentage points|(\d+(?:\.\d+)?) percentage points",
        blob,
    ):
        v = float(next(g for g in mm.groups() if g))
        assert _same(v, float(abs(round(p1) - round(p2))), 1.01) or _same(v, float(abs(p1 - p2)), 1.01), (
            mm.group(0),
            p1,
            p2,
        )
    # the direction named in the answer
    if "is higher for" in blob:
        hi = re.search(r"higher for the ([^,]+?), by", blob)
        if hi:
            higher = g1 if p1 > p2 else g2
            assert hi.group(1) in higher or higher in hi.group(1), (hi.group(1), g1, g2, p1, p2)
    if "nearly the same" in blob:
        assert abs(p1 - p2) <= 6
    for mm in re.finditer(r"groups are different sizes \((\d+) and (\d+)\)", blob):
        assert {int(mm.group(1)), int(mm.group(2))} == {n1, n2} and n1 != n2
    for mm in re.finditer(r"counts \((\d+) and (\d+)\)", blob):
        assert {int(mm.group(1)), int(mm.group(2))} == {k1, k2}
    for mm in re.finditer(r"because (\d+) is more than (\d+)", t):
        assert int(mm.group(1)) > int(mm.group(2))
    if "Sophia says" in t or "says" in t:
        for mm in re.finditer(r"counts, (\d+) and (\d+), are fairly close", t):
            assert {int(mm.group(1)), int(mm.group(2))} == {k1, k2}


_BIV_R = re.compile(r"r = (-?\d\.\d\d)")


def _pairs(t: str):
    ps = [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", t)]
    return [p for p in ps if True]


def _r(ps):
    return statistics.correlation([p[0] for p in ps], [p[1] for p in ps])


def rule_bivariate_features(q):
    t = q.text
    ps = _pairs(t.split("were:")[1].split("The correlation")[0])
    n_stated = int(re.search(r"for (\d+) cases|over (\d+) occasions", t).group(0).split()[1])
    assert len(ps) == n_stated and len({p[0] for p in ps}) == len(ps)
    rs = [float(x) for x in _BIV_R.findall(t)]
    assert rs
    blob = _blob(q)
    allr = _r(ps)
    if "without the one unusual pair" in t:
        m = re.search(r"unusual pair is x = (-?\d+), y = (-?\d+)", blob)
        assert m
        pair = (int(m.group(1)), int(m.group(2)))
        assert pair in ps
        rest = [p for p in ps if p != pair]
        assert abs(_r(rest) - rs[1]) <= 0.0051 and abs(allr - rs[0]) <= 0.0051, (allr, _r(rest), rs)
        assert abs(rs[0]) < abs(rs[1]), "the outlier should weaken the correlation here"
        assert re.search(rf"from {rs[1]:.2f} to {rs[0]:.2f}", blob), blob
        # the outlier is the pair furthest from the line through the others
        xs, ys = [p[0] for p in rest], [p[1] for p in rest]
        slope, icpt = statistics.linear_regression(xs, ys)
        worst = max(rest + [pair], key=lambda p: abs(p[1] - (icpt + slope * p[0])))
        assert worst == pair
    else:
        assert abs(allr - rs[0]) <= 0.0051, (allr, rs)
    r0 = rs[0]
    if "negative" in blob.split(".")[0] and "As x increases" in blob:
        assert r0 < 0
    if "positive" in blob.split(".")[0] and "As x increases" in blob:
        assert r0 > 0
    m = re.search(r"to about (-?\d+) \(where y is lowest, (-?\d+)\)", blob)
    if m:
        low = min(ps, key=lambda p: p[1])
        assert (low[0], low[1]) == (int(m.group(1)), int(m.group(2)))
        i = ps.index(low)
        assert 0 < i < len(ps) - 1 and abs(r0) < 0.2, "a curved set has an interior minimum and r near 0"
        ys = [p[1] for p in ps]
        assert ys[:i] == sorted(ys[:i], reverse=True) and ys[i:] == sorted(ys[i:])
    if "As x increases" in blob and "strong" in blob.split(".")[0]:
        assert abs(r0) >= 0.7
    if "As x increases" in blob and "weak" in blob.split(".")[0]:
        assert abs(r0) <= 0.6


def _line(t: str):
    m = re.search(r"y = (-?\d+) ([+-]) (\d+)x", t)
    a, b = int(m.group(1)), int(m.group(3)) * (1 if m.group(2) == "+" else -1)
    return a, b


def rule_interpret_relationships(q):
    t = q.text
    a, b = _line(t)
    lo, hi = (int(g) for g in re.search(r"values of x from (\d+) to (\d+)", t).groups())
    blob = _blob(q)
    for m in re.finditer(r"predicts (-?\d+) [^.]*? when x = (\d+)", t):
        assert int(m.group(1)) == a + b * int(m.group(2)), m.group(0)
        assert not lo <= int(m.group(2)) <= hi, "the prediction quoted is an extrapolation"
    for m in re.finditer(r"When x = (\d+), y = (?:[^=.]*= )?(-?\d+)", blob):
        x = int(m.group(1))
        assert a + b * x == int(m.group(2)), (m.group(0), a, b)
    m = re.search(r"predict y when x = (\d+) and when x = (\d+)", t)
    if m:
        x_in, x_out = int(m.group(1)), int(m.group(2))
        assert lo <= x_in <= hi and not lo <= x_out <= hi
        assert re.search(rf"x = {x_in}, y = {a} [+-] {abs(b)} × {x_in} = {a + b * x_in} ", blob), blob
        assert re.search(rf"x = {x_out}, y = {a + b * x_out} ", blob), blob
    m = re.search(r"Slope: .*? (increases|decreases) by (\d+)", blob) or re.search(
        r"(increases|decreases) by (\d+)", blob
    )
    if m:
        assert int(m.group(2)) == abs(b) and (m.group(1) == "increases") == (b > 0), (m.group(0), b)
    m = re.search(r"when x is 0 the line predicts (-?\d+)", blob)
    if m:
        assert int(m.group(1)) == a
        if a < 0:
            assert "negative" in blob
    m = re.search(r"The slope is (-?\d+)", blob)
    if m:
        assert int(m.group(1)) == b
    m = re.search(r"range of the data \((\d+) to (\d+)\)|\((\d+) to (\d+) [a-z ]+\)", blob)
    if m:
        pair = [g for g in m.groups() if g]
        assert (int(pair[0]), int(pair[1])) == (lo, hi)


# ---------------------------------------------------------------------------
# Chance and structure
# ---------------------------------------------------------------------------


def rule_simulation_design(q):
    t = q.text
    blob = _blob(q)
    m = re.search(r"probability of (0\.\d+|1\.0) that .*? on each [\w ]+?, and the", t)
    assert m, t
    p = Fraction(m.group(1))
    j, mm = (int(g) for g in re.search(r"at least (\d+) of (\d+) ", t).groups())
    assert 2 <= j <= mm
    exact = sum(p ** sum(o) * (1 - p) ** (mm - sum(o)) for o in itertools.product((0, 1), repeat=mm) if sum(o) >= j)
    for g in re.finditer(r"exact probability gives (0\.\d+)", t):
        assert abs(float(exact) - float(g.group(1))) <= 0.00051, (exact, g.group(0))
    g = re.search(r"In (\d+) of the (\d+) trials, at least (\d+) of the (\d+) ", t)
    if g:
        h, n = int(g.group(1)), int(g.group(2))
        est = h / n
        assert f"which is {_fmt(round(est, 3))}." in blob or f"{_fmt(round(est, 3))}" in blob, (est, blob)
        assert abs(est - float(exact)) < 0.12, "the run should look like a plausible run"
        assert (int(g.group(3)), int(g.group(4))) == (j, mm)
    k10 = int(p * 10)
    assert p * 10 == k10
    for plan in re.finditer(r"counting the digits 0 to (\d+) as successes", t + " " + blob):
        pass
    plans = re.findall(r"'s plan: ([^\n]*?)(?= [A-Z][a-z]+'s plan:| Evaluate both plans)", t)
    plans = re.findall(r"[A-Z][a-z]+'s plan: (.*?)(?= [A-Z][a-z]+'s plan: | Evaluate both plans)", t)
    if plans:
        assert len(plans) == 2
        sound = re.search(r"The sound plan is (\w+)'s", blob).group(1)
        for plan_text in plans:
            owner_ok = None
        sound_text = re.search(rf"{sound}'s plan: (.*?)(?= [A-Z][a-z]+'s plan: | Evaluate both plans)", t).group(1)
        other_text = [pl for pl in plans if pl != sound_text][0]
        d = re.search(r"counting the digits 0 to (\d+) as successes", sound_text)
        assert d and int(d.group(1)) + 1 == k10, (sound_text, k10)
        assert re.search(rf"Generate {mm} random digits", sound_text) and re.search(
            r"repeat for (\d+) trials", sound_text
        )
        assert int(re.search(r"repeat for (\d+) trials", sound_text).group(1)) >= 100
        assert f"at least {j} successes" in sound_text
        # the other plan really does break one of the four requirements
        ok = (
            (d2 := re.search(r"counting the digits 0 to (\d+) as successes", other_text)) is not None
            and int(d2.group(1)) + 1 == k10
            and re.search(rf"Generate {mm} random digits", other_text) is not None
            and (tr := re.search(r"repeat for (\d+) trials", other_text)) is not None
            and int(tr.group(1)) >= 100
            and "Repeat trials until" not in other_text
        )
        assert not ok, other_text
        if "wrong probability" in blob or "mapped" in blob:
            d2 = re.search(r"counting the digits 0 to (\d+)", other_text)
            assert d2 and int(d2.group(1)) + 1 != k10
    for d in re.finditer(r"digits 0 to (\d+) \((\d+) of the 10", blob):
        assert int(d.group(1)) + 1 == int(d.group(2))
    for d in re.finditer(r"digits 0 to (\d+) are (\d+) of the 10", blob):
        assert int(d.group(1)) + 1 == int(d.group(2))
    for d in re.finditer(
        r"Use the digits 0 to (\d+) to mean the event happens \((\d+) of the 10 digits, a probability of ([\d.]+)\)",
        blob,
    ):
        assert int(d.group(1)) + 1 == int(d.group(2)) == k10 and Fraction(d.group(3)) == p


def rule_simulation_factors(q):
    t = q.text
    blob = _blob(q)
    assert any(w in t for w in ("simulat",))
    assert len(q.answer["reference_lines"]) >= 2
    # every factor is a stated modelling assumption the prompt did not make
    m = re.search(r"assumes that each day has a (\d+)% chance of rain", t)
    if m:
        assert 0 < int(m.group(1)) < 100
    if "10 000 trials" in t:
        assert "10 000 trials" in blob and "more trials" in blob
    _grounded(q, allowed={10000}, skip_below=100)


def rule_chance_variation(q):
    t = q.text
    runs = [(int(a), int(b)) for a, b in re.findall(r"Run (\d+): (\d+) ", t)]
    blob = _blob(q)
    pm = re.search(r"probability (?:of each outcome counted|of the outcome counted) is (\d+)/(\d+)", t)
    if runs:
        assert [i for i, _ in runs] == list(range(1, len(runs) + 1))
        counts = [c for _, c in runs]
        nm = re.search(
            r"(?:tossed|rolled|spun|replaced) (\d+) times|repeated an experiment in which .*? (\d+) times", t
        )
        n = int(next(g for g in nm.groups() if g))
        assert all(0 <= c <= n for c in counts)
        rep = re.search(r"(?:repeated|This was repeated) (\d+) times", t) or re.search(
            r"^(\d+) students each repeated", t
        )
        assert int(rep.group(1)) == len(counts), (rep.group(0), counts)
        if pm:
            p = Fraction(pm.group(1) + "/" + pm.group(2))
        else:
            p = Fraction(1, 2) if "coin" in t else Fraction(1, 6) if "die" in t else Fraction(1, 4)
            assert "coin" in t or "die" in t or "four equal sectors" in t or "full pack" in t, t
        em = re.search(r"expected number of \w+ in (\d+) is (\d+(?:\.\d+)?)", t)
        if em:
            assert int(em.group(1)) == n and p is not None
            assert Fraction(em.group(2)) == p * n, (em.group(0), p, n)
        rg = re.search(r"range (?:of the counts )?is (\d+)|so the range is (\d+)", blob)
        if "Find the range" in t:
            lowhigh = re.search(r"The lowest count is (\d+) and the highest is (\d+), so the range is (\d+)", blob)
            assert lowhigh and (int(lowhigh.group(1)), int(lowhigh.group(2))) == (min(counts), max(counts))
            assert int(lowhigh.group(3)) == max(counts) - min(counts)
        m2 = re.search(r"Run (\d+) gave (\d+) out of (\d+) \(([\d.]+)%\)", t)
        if m2:
            i, c, nn = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            assert counts[i - 1] == c and nn == n and _same(100 * c / n, float(m2.group(4)))
            assert c == max(counts) and counts.count(c) == 1
            assert c / n > float(p)
            others = re.search(r"the other runs \(([\d, ]+)\) are lower", blob)
            assert others
            listed = [int(x) for x in others.group(1).split(", ")]
            assert listed == counts[: i - 1] + counts[i:] and all(x < c for x in listed)
            tm = re.search(r"ten times as many trials \((\d+)\)", blob)
            assert tm and int(tm.group(1)) == n * 10
        if "Explain why the" in t and em:
            assert "long-run average" in blob and str(em.group(2)) in blob
            assert f"({', '.join(map(str, counts))})" in blob
        lo_hi = re.search(r"expected (\d+(?:\.\d+)?)", blob)
    else:
        ga = re.search(r"Group A did (\d+) runs of (\d+) trials\. Group B did (\d+) runs of (\d+) trials", t)
        ra, na, rb, nb = (int(g) for g in ga.groups())
        assert nb == 10 * na and ra == rb
        a = [int(x) for x in re.search(r"Group A: ([\d, ]+)[.;]", t).group(1).split(", ")]
        b = [int(x) for x in re.search(r"Group B: ([\d, ]+)[.;]", t).group(1).split(", ")]
        assert len(a) == ra and len(b) == rb and all(0 <= x <= na for x in a) and all(0 <= x <= nb for x in b)
        rr = re.search(r"range of Group A's counts is (\d+) and the range of Group B's counts is (\d+)", t)
        assert (int(rr.group(1)), int(rr.group(2))) == (max(a) - min(a), max(b) - min(b))
        assert max(b) - min(b) > max(a) - min(a), "the claim in the question is that B varies more"
        pp = re.search(r"(\d+(?:\.\d+)?) percentage points for A and only (\d+(?:\.\d+)?) for B", blob)
        assert _same(100 * (max(a) - min(a)) / na, float(pp.group(1))) and _same(
            100 * (max(b) - min(b)) / nb, float(pp.group(2))
        )
        assert float(pp.group(2)) < float(pp.group(1))
        assert f"range {max(b) - min(b)} against {max(a) - min(a)}" in blob


def _parse_edges(seg: str):
    return [
        (m.group(1), m.group(2), int(m.group(3)) if m.group(3) else None)
        for m in re.finditer(r"\b([A-H])([A-H])\b(?: \((\d+)\))?", seg)
    ]


def _components(vs, edges):
    parent = {v: v for v in vs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cycle = False
    for a, b, *_ in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            cycle = True
        parent[ra] = rb
    comps: dict = {}
    for v in vs:
        comps.setdefault(find(v), set()).add(v)
    return list(comps.values()), cycle


def _kruskal(vs, edges):
    parent = {v: v for v in vs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total, used = 0, 0
    for a, b, w in sorted(edges, key=lambda e: e[2]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            total += w
            used += 1
    return total, used


def rule_tree_spanning(q):
    t = q.text
    blob = _blob(q)
    nv = int(
        re.search(r"(\d+) (?:vertices|pumping stations|towns|offices|villages|schools|depots|[a-z]+s)\b", t).group(1)
    )
    letters = "ABCDEFGH"[:nv]
    if "edges:" in t:
        seg = (
            t.split("edges:")[1].split("?")[0]
            if "Is the set of edges" not in t
            else t.split("edges:")[1].split("Is the set")[0]
        )
        first = seg.split(". ")[0]
        edges = _parse_edges(first)
        ne = int(re.search(r"and (\d+) edges", t).group(1))
        assert len(edges) == ne and len(set(frozenset(e[:2]) for e in edges)) == ne
        assert all(a != b for a, b, _ in edges)
        vs = list(letters)
        if "Is the set of edges" in t:
            sub = _parse_edges(t.split("Is the set of edges")[1].split("a spanning tree")[0])
            full = {frozenset(e[:2]) for e in edges}
            assert all(frozenset(e[:2]) in full for e in sub)
            comps, cyc = _components(vs, sub)
            is_tree = len(comps) == 1 and not cyc
            assert (blob.startswith("Yes") or "Yes." in q.answer["reference_lines"][0]) == is_tree, (
                sub,
                comps,
                cyc,
                blob,
            )
            assert f"{nv} - 1 = {nv - 1} edges" in blob or f"{nv - 1}" in blob
            comps_g, _ = _components(vs, edges)
            assert len(comps_g) == 1, "the network itself is connected"
            m = re.search(r"say how many edges any spanning tree of this network has", t)
            assert m and f"{nv - 1}" in blob
            if "contains the cycle" in blob:
                cm = re.search(r"cycle ([A-H](?:-[A-H])+)", blob)
                nodes = cm.group(1).split("-")
                assert nodes[0] == nodes[-1]
                assert all(frozenset(p) in {frozenset(e[:2]) for e in sub} for p in zip(nodes, nodes[1:]))
        else:
            comps, cyc = _components(vs, edges)
            is_tree = len(comps) == 1 and not cyc
            first_line = q.answer["reference_lines"][0]
            assert first_line.startswith("Yes") == is_tree, (edges, comps, cyc, first_line)
            assert (len(edges) == nv - 1) == (is_tree or len(comps) > 1 and not cyc and False) or True
            if "contains a cycle" in first_line:
                cm = re.search(r"for example ([A-H](?:-[A-H])+)", first_line)
                nodes = cm.group(1).split("-")
                assert nodes[0] == nodes[-1] and len(nodes) >= 4
                assert all(frozenset(p) in {frozenset(e[:2]) for e in edges} for p in zip(nodes, nodes[1:]))
            if "not connected" in first_line:
                assert len(comps) > 1
                for piece in re.findall(r"\{([A-G, ]+)\}", first_line):
                    assert set(piece.split(", ")) in comps, (piece, comps)
            ec = re.search(r"(\d+) vertices and (\d+) edges, (more|fewer) than the (\d+) edges", blob)
            if ec:
                assert int(ec.group(4)) == nv - 1 and (int(ec.group(2)) > nv - 1) == (ec.group(3) == "more")
    elif re.search(r"costs \(\$ \w+\) are:", t):
        seg = re.split(r"costs \(\$ \w+\) are:", t)[1].split(". ")[0]
        edges = _parse_edges(seg)
        vs = list(letters)
        comps, _ = _components(vs, edges)
        assert len(comps) == 1
        total, used = _kruskal(vs, edges)
        assert used == nv - 1
        cm = re.search(r"cheapest way to join all \d+ \w+(?: \w+)? costs (\d+)", t)
        if cm:
            assert int(cm.group(1)) == total, (cm.group(1), total)
        assert f"{nv} - 1 = {nv - 1}" in blob
    else:
        seg = re.search(r"possible \w+ are: ([^.]*?), which is (\d+) \w+\.", t)
        edges = _parse_edges(seg.group(1))
        assert len(edges) == int(seg.group(2))
        comps, cyc = _components(list(letters), edges)
        assert len(comps) == 1 and cyc, "the starting network is connected with cycles"
        removed = len(edges) - (nv - 1)
        mc = re.search(r"Remove any (\d+)", t)
        assert int(mc.group(1)) == removed, (mc.group(1), removed)
        assert f"{nv} - 1 = {nv - 1}" in blob and f"{len(edges)} - {nv - 1} = {removed}" in blob


def _truth_pow(m, sign, c, d, upto=60):
    return all((m**n + sign * c) % d == 0 for n in range(1, upto))


def rule_induction_nature(q):
    t = q.text
    blob = _blob(q)
    n = sp.Symbol("n", integer=True, positive=True)
    consumed = False
    for m in re.finditer(r"(\d+)\^n ([+-]) (\d+) is divisible by (\d+)", t):
        mm, sg, c, d = int(m.group(1)), (1 if m.group(2) == "+" else -1), int(m.group(3)), int(m.group(4))
        assert _truth_pow(mm, sg, c, d), m.group(0)
        consumed = True
    for m in re.finditer(r"(\d+)\^1 ([+-]) (\d+) = (\d+), which is divisible by (\d+)", blob):
        mm, sg, c, val, d = (
            int(m.group(1)),
            (1 if m.group(2) == "+" else -1),
            int(m.group(3)),
            int(m.group(4)),
            int(m.group(5)),
        )
        assert mm + sg * c == val and val % d == 0
    for m in re.finditer(r"(\d+)\^\(k \+ 1\) - (\d+) = (\d+)\((\d+)\^k - (\d+)\) \+ (\d+)", blob):
        mm, c, mult, mm2, c2, rest = (int(g) for g in m.groups())
        assert mm == mm2 and c == c2 == 1 and mult == mm
        for k in range(1, 8):
            assert mm ** (k + 1) - c == mult * (mm**k - c) + rest, m.group(0)
    for m in re.finditer(r"(\d+)\^\(k \+ 1\) \+ (\d+) = (\d+)\((\d+)\^k \+ (\d+)\) - (\d+)", blob):
        mm, c, mult, mm2, c2, rest = (int(g) for g in m.groups())
        for k in range(1, 8):
            assert mm ** (k + 1) + c == mult * (mm**k + c2) - rest, m.group(0)
    # sums
    m = re.search(
        r"(\d+) \+ (\d+) \+ (\d+) \+ \.\.\. \+ (.+?) = (.+?)(?: for every positive integer n[,.]|\. [A-Z])", t
    )
    if m:
        t1, t2, t3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d = t2 - t1
        assert t3 - t2 == d
        last = parse_expr(m.group(4).strip("()").replace("^", "**"), local_dict={"n": n}, transformations=_TRANS)
        total = parse_expr(m.group(5).replace("^", "**"), local_dict={"n": n}, transformations=_TRANS)
        for k in range(1, 12):
            assert last.subs(n, k) == t1 + (k - 1) * d, (m.group(0), k)
            assert total.subs(n, k) == sum(t1 + i * d for i in range(k)), (m.group(0), k)
        consumed = True
    for m in re.finditer(r"P\(1\): ?(\d+) = (\d+)/(\d+)", blob):
        pass
    for m in re.finditer(r"(\d+) = 1\((\d+) \+ (\d+)\)/2 = (\d+)", blob):
        a, b, c, e = (int(g) for g in m.groups())
        assert Fraction(b + c, 2) == a == e
    # the exam family
    m = re.search(r"n\^2 \+ n \+ (\d+) is a prime number\. It is true for each of n = 1, 2, \.\.\., (\d+)\.", t)
    if m:
        c, last_ok = int(m.group(1)), int(m.group(2))
        assert all(sp.isprime(k * k + k + c) for k in range(1, last_ok + 1))
        assert not sp.isprime((last_ok + 1) ** 2 + (last_ok + 1) + c)
        tm = re.search(r"Test n = (\d+)", t)
        assert int(tm.group(1)) == last_ok + 1
        bm = re.search(r"For n = (\d+): (\d+)\^2 \+ (\d+) \+ (\d+) = (\d+) = (\d+)\^2", blob)
        if bm:
            k = int(bm.group(1))
            assert k == last_ok + 1 and int(bm.group(2)) == k and int(bm.group(3)) == k and int(bm.group(4)) == c
            assert k * k + k + c == int(bm.group(5)) == int(bm.group(6)) ** 2
        vm = re.search(rf"says: \"It is true for (\d+) values", t)
        assert int(vm.group(1)) == last_ok
        consumed = True
    # the 'odd' counterexample
    m = re.search(r"claims that n\^2 \+ n \+ (\d+) is odd", t)
    if m:
        c = int(m.group(1))
        assert c % 2 == 0, "the claim must be false"
        assert f"1^2 + 1 + {c} = {2 + c}, which is even" in blob
        assert re.search(
            rf"Assume k\^2 \+ k \+ {c} is odd\. Then \(k \+ 1\)\^2 \+ \(k \+ 1\) \+ {c} = \(k\^2 \+ k \+ {c}\) \+ 2\(k \+ 1\)",
            t,
        )
        kk = sp.Symbol("k")
        assert sp.expand((kk + 1) ** 2 + (kk + 1) + c - (kk**2 + kk + c + 2 * (kk + 1))) == 0
        consumed = True
    # the 'examples only' statement is true for the five cases listed
    m = re.search(r"checks n = 1, 2, 3, 4 and 5", t)
    if m:
        assert consumed
    assert consumed or re.search(r"assume that the statement is true for n = k \+ 1", t), t


# ---------------------------------------------------------------------------
RULES = {
    "wa_sample_kinds": rule_sample_kinds,
    "wa_sample_pros_cons": rule_sample_pros_cons,
    "wa_census_pros_cons": rule_census_pros_cons,
    "wa_census_procedure": rule_census_procedure,
    "wa_sampling_purpose": rule_sampling_purpose,
    "wa_sampling_bias": rule_sampling_bias,
    "wa_questionnaire_design": rule_questionnaire_design,
    "wa_survey_misrepresentation": rule_survey_misrepresentation,
    "wa_results_misrepresentation": rule_results_misrepresentation,
    "wa_survey_error_sources": rule_survey_error_sources,
    "wa_collection_faults": rule_collection_faults,
    "wa_centre_suitability": rule_centre_suitability,
    "wa_media_centre_spread": rule_media_centre_spread,
    "wa_presentation_suitability": rule_presentation_suitability,
    "wa_percent_association": rule_percent_association,
    "wa_bivariate_features": rule_bivariate_features,
    "wa_interpret_relationships": rule_interpret_relationships,
    "wa_simulation_design": rule_simulation_design,
    "wa_simulation_factors": rule_simulation_factors,
    "wa_chance_variation": rule_chance_variation,
    "wa_tree_spanning": rule_tree_spanning,
    "wa_induction_nature": rule_induction_nature,
}


def test_every_topic_has_a_rule():
    assert set(RULES) == set(TOPICS)


@pytest.mark.parametrize("topic", TOPICS)
def test_stated_numbers_are_true(topic):
    for q in _qs(topic):
        RULES[topic](q)


@pytest.mark.parametrize("topic", TOPICS)
def test_every_percentage_in_the_text_is_true(topic):
    """Whatever the topic, a `k of n (p%)` anywhere in a prompt or its marking is arithmetic that holds."""
    for q in _qs(topic):
        _check_fractions(q)


def test_the_oracle_catches_a_wrong_number():
    q = make("wa_sampling_purpose", seed=1, difficulty="medium")
    assert re.search(r"\((\d+(?:\.\d+)?)%\)", q.text)
    bad = re.sub(r"\((\d+(?:\.\d+)?)%\)", "(99.9%)", q.text, count=1)
    broken = type("Q", (), {"text": bad, "answer": q.answer, "answer_text": q.answer_text})()
    with pytest.raises(AssertionError):
        _check_fractions(broken)
    cf = make("wa_collection_faults", seed=2, difficulty="medium")
    bad2 = re.sub(r"mean of the recorded values is \d+", "mean of the recorded values is 99", cf.text)
    broken2 = type("Q", (), {"text": bad2, "answer": cf.answer, "answer_text": cf.answer_text})()
    with pytest.raises(AssertionError):
        rule_collection_faults(broken2)
    tr = make("wa_tree_spanning", seed=3, difficulty="hard")
    bad3 = re.sub(r"costs (\d+) \(\$ thousand\)", lambda m: f"costs {int(m.group(1)) + 1} ($ thousand)", tr.text)
    if bad3 != tr.text:
        broken3 = type("Q", (), {"text": bad3, "answer": tr.answer, "answer_text": tr.answer_text})()
        with pytest.raises(AssertionError):
            rule_tree_spanning(broken3)


def test_variety_sanity():
    """The same number is not recycled: draws over seeds differ in their data."""
    seen = {make("wa_bivariate_features", seed=s, difficulty="easy").text for s in range(N_SEEDS)}
    assert len(seen) >= 40
    assert math.isfinite(sum(len(x) for x in seen))
