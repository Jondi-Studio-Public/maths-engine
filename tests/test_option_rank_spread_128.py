"""Where the key sits among the wrong options, for every topic of the data-display, early-years, graph-transformation
and senior-statistics builds (`dg_*`, `eyz_*`, `gt_*`, `sc_*`).

A generator that lists its slips "likeliest first" gives the key the same rank on every seed (always the smallest
option, always second-largest), so a learner can pick it without the maths. `mathgen/generators/_spread_kit.py` draws a
random below/above split; this file fails when a question kind still has one rank for more than 60% of its seeds, when
a wrong option repeats the key, when a "nearest 10" question offers a value that is not a multiple of 10, and when a
question has fewer than three options.
"""

from __future__ import annotations

import glob
import inspect
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen import make
from mathgen.core import REGISTRY

SEEDS = 300
MIN_GROUP = 40  # a question kind needs this many numeric samples before its rank shares mean anything
MAX_SHARE = 0.6

_GEN = Path(__file__).resolve().parent.parent / "mathgen" / "generators"
_FILES = {Path(f).name for f in glob.glob(str(_GEN / "*_build.py")) if re.match(r"(dg|eyz|gt|sc)_", Path(f).name)}
TOPICS = sorted(
    t
    for t, (_, fn) in REGISTRY.items()
    if (src := inspect.getsourcefile(inspect.unwrap(fn))) and Path(src).name in _FILES
)

# (tier, part, question kind) -> why the key cannot be spread: the natural option set is tiny or sits at the bottom of
# the counting numbers, so the key is the smallest option whatever is drawn.
INHERENT = {
    ("dg_graph_terms", "medium", "q", "the drawn network is"): "edges to join the pieces: 1 or 2, nothing below 1",
    ("dg_graph_terms_chain", "exam", "p1", "how many extra edges"): "a count of 1 to 3",
    ("eyz_estimate_collection", "hard", "q", "about how many more"): "a difference of 10 to 30, nothing below 10",
    ("eyz_estimate_collection", "hard", "q", "estimate how many more"): "a difference of 10 to 30, nothing below 10",
    ("eyz_named_solids", "easy", "q", "count the curved surfaces"): "0 or 1 curved surfaces",
    ("eyz_named_solids", "easy", "q", "how many curved surfaces"): "0 or 1 curved surfaces",
    ("eyz_symmetry_count", "hard", "q", "count the lines of"): "0 to 6 lines, mostly 0 and 1",
    ("eyz_symmetry_count", "hard", "q", "how many lines of"): "0 to 6 lines, mostly 0 and 1",
    ("gt_graph_transform_exam", "exam", "p1", "the graph of g"): "two wrong options only (a sign slip and a swap)",
}
# "Round to the nearest 10" is in the stem of the exam estimate chain, but its last part multiplies by a per-item count.
NOT_TENS = {("eyz_estimate_collection_exam", "exam", "p3")}


def _number(text):
    text = str(text).strip().replace("$", "").replace(",", "")
    try:
        return float(Fraction(text)) if re.fullmatch(r"-?\d+/\d+", text) else float(text)
    except ValueError:
        return None


def _kind(text: str) -> str:
    words = re.findall(r"[a-z#']+", re.sub(r"\d+(?:\.\d+)?", "#", text.lower()))
    return " ".join(words[:4])


def rank_table(topic: str, tier: str, seeds=range(SEEDS), parts=None):
    """{(part label, question kind): Counter of how many wrong options lie below the key}, plus problems found."""
    table: dict = defaultdict(Counter)
    problems: list[str] = []
    for seed in seeds:
        q = make(topic, seed=seed, difficulty=tier)
        units = [(f"p{i}", p.answer_text, p.distractors, p.prompt + " " + q.text) for i, p in enumerate(q.parts)]
        if not units:
            units = [("q", q.answer_text, q.distractors, q.text)]
        for label, answer, options, text in units:
            if parts is not None and label not in parts:
                continue
            shown = [getattr(o, "text", str(o)) for o in options or []]
            if len(shown) < 2:
                problems.append(f"{topic} {tier} {label} seed {seed}: only {len(shown)} wrong options")
                continue
            if str(answer) in shown:
                problems.append(f"{topic} {tier} {label} seed {seed}: an option repeats the key {answer}")
            key, values = _number(answer), [_number(o) for o in shown]
            if key is None or any(v is None for v in values):
                continue
            if "nearest 10" in text and (topic, tier, label) not in NOT_TENS and any(v % 10 for v in [key, *values]):
                problems.append(f"{topic} {tier} {label} seed {seed}: not a multiple of 10: {answer} {shown}")
            table[(label, _kind(text))][sum(v < key for v in values)] += 1
    return table, problems


@pytest.mark.parametrize("topic", TOPICS)
def test_key_has_no_fixed_rank_among_the_options(topic):
    lopsided = []
    for tier in ("easy", "medium", "hard", "exam"):
        try:
            make(topic, seed=0, difficulty=tier)
        except Exception:  # the topic has no such tier
            continue
        table, problems = rank_table(topic, tier)
        assert not problems, problems[:3]
        for (label, kind), ranks in table.items():
            total = sum(ranks.values())
            if total < MIN_GROUP or (topic, tier, label, kind) in INHERENT:
                continue
            if max(ranks.values()) / total > MAX_SHARE:
                lopsided.append(f"{tier} {label} [{kind}] {dict(sorted(ranks.items()))}")
    assert not lopsided, lopsided


def test_the_fixes_this_file_was_written_for_hold():
    """The question kinds reviewers measured as having a fixed key rank, named so a regression names them."""
    cases = [
        ("dg_real_data_graphs", "hard", None),
        ("dg_real_data_graphs", "medium", None),
        ("eyz_prism_pyramid_counts", "easy", None),
        ("sc_simulated_proportions", "hard", None),
        ("sc_simulated_proportions_exam", "exam", {"p3", "p4"}),
        ("gt_sqrt_function_exam", "exam", {"p1"}),
        ("gt_inverse_trig_exam", "exam", {"p0"}),
        ("sc_conditional_counts_exam", "exam", {"p0"}),
    ]
    for topic, tier, parts in cases:
        table, problems = rank_table(topic, tier, parts=parts)
        assert not problems, problems[:3]
        merged = Counter()
        for (_, kind), ranks in table.items():
            merged.update(ranks)
        total = sum(merged.values())
        assert total >= MIN_GROUP and max(merged.values()) / total <= MAX_SHARE and len(merged) >= 3, (
            topic,
            tier,
            dict(sorted(merged.items())),
        )
