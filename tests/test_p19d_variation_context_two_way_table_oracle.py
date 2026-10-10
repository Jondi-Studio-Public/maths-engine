"""Re-derives the correct answer for both skills from the rendered question alone, using a technique that never
imports ``mathgen.generators.p19d_variation_context_two_way_table_build``.

- ``p19d_variation_context``: a keyword/regex classifier over each rendered
  option's own text (continuous-measurement language vs discrete-count,
  strictly-non-negative/skewed, bounded/clustered, or uniform-"near-miss"
  language), independently coded from generic statistical vocabulary, never
  reading the generator's own class-tag pools.
- ``p19d_two_way_table``: a plain-text table parser that extracts the Rows/
  Columns lists, every stated cell, and any stated row total, then
  independently recomputes the requested cell/total/proportion by parsing
  the rendered question sentence's own wording to decide what was asked and
  which axis is which -- never importing this module's own 2D-array
  construction.
"""

import math
import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make

TOPIC_VC = "p19d_variation_context"
TOPIC_TW = "p19d_two_way_table"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
WIDE_SEEDS = range(1000)


# ============================================================================
# Oracle 1: p19d_variation_context
# ============================================================================


def _oracle_vc_classify(phrase: str) -> str:
    """Independently-coded rule: is this described quantity a naturally
    occurring continuous measurement (normal), a discrete count over an
    open-ended window (count), a strictly non-negative/skewed quantity
    (skewed), a bounded/clustered-at-a-boundary count from a small fixed
    number of trials (bounded), or a uniform "arrives on a fixed schedule"
    near-miss (near_miss)?"""
    p = phrase.lower()
    # A restriction/truncation clause on a measured quantity ("only ... are
    # included", paired with a comparative threshold) changes the shape
    # away from normal, regardless of how normal the un-truncated base
    # measurement would otherwise be -- checked before every other rule so
    # a truncated normal-base phrase is never misfiled as "normal".
    if re.search(r"only .*(taller|heavier|wider|shorter|older|younger|larger|smaller) than", p) and "included" in p:
        return "truncated"
    if re.search(r"tossed \d+ times|multiple-choice quiz|batch of \d+|proportion of students", p):
        return "bounded"
    if re.search(r"arrives exactly every|circular dial|spinner", p):
        return "near_miss"
    if "number of" in p:
        return "count"
    if re.search(
        r"\btime\b.*\buntil\b|income|on hold|battery"
        r"|masses?,? in kg,? of fish|distances?,? in km,? .*travel",
        p,
    ):
        return "skewed"
    if re.search(r"combined group of .+ and .+", p):
        return "bimodal"
    return "normal"


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc_oracle_matches_answer_text(difficulty):
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        options = [q.answer_text, *q.distractors]
        classes = [_oracle_vc_classify(o) for o in options]
        normal_options = [o for o, c in zip(options, classes) if c == "normal"]
        if normal_options != [q.answer_text]:
            challenged.append((seed, normal_options, q.answer_text, options))
    assert not challenged, f"oracle challenges: {challenged}"


def test_vc_oracle_confirms_all_five_categories_occur():
    counts = Counter()
    for difficulty in TIERS:
        for seed in WIDE_SEEDS[:300]:
            q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
            for o in [q.answer_text, *q.distractors]:
                counts[_oracle_vc_classify(o)] += 1
    assert set(counts) >= {"normal", "count", "skewed", "bounded", "near_miss"}, dict(counts)
    print(f"p19d_variation_context per-category option counts (1200 draws): {dict(counts)}")


def test_vc_red_proof_oracle_catches_a_flipped_key():
    """RED first: swap the answer for a distractor and confirm the oracle
    disagrees, over >=1000 seeds combined across tiers."""
    caught = 0
    total = 0
    for difficulty in TIERS:
        for seed in range(260):
            q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
            bogus = q.distractors[0]
            total += 1
            if _oracle_vc_classify(bogus) != "normal":
                caught += 1
    assert total >= 1000
    assert caught == total, f"oracle failed to catch a flipped key on {total - caught}/{total} draws"


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc_real_key_passes_oracle_over_1000_seeds(difficulty):
    for seed in range(1000):
        q = make(TOPIC_VC, seed=seed, difficulty=difficulty)
        options = [q.answer_text, *q.distractors]
        normal_options = [o for o in options if _oracle_vc_classify(o) == "normal"]
        assert normal_options == [q.answer_text], (seed, normal_options, q.answer_text)


# ============================================================================
# Oracle 2: p19d_two_way_table
# ============================================================================

_ROWS_RE = re.compile(r"Rows: (.+?)\.")
_COLS_RE = re.compile(r"Columns: (.+?)\.")

# Category names may contain a hyphen ("In-store order") or lead with a
# digit ("40 or older"), so every generic capture group below allows
# `[\w\s-]` (word chars, spaces, hyphens), not merely `\w`.
_PAT_PROP_GRAND = re.compile(r"What proportion of all respondents are both ([\w\s-]+?) and ([\w\s-]+?)\?")
_PAT_PROP_COL = re.compile(r"Of those in the ([\w\s-]+?) category, what proportion are ([\w\s-]+?)\?")
_PAT_PROP_ROW = re.compile(r"[Ww]hat proportion of ([\w\s-]+?) respondents also fall into the ([\w\s-]+?) category")
_PAT_CELL_BOTH = re.compile(r"How many are both ([\w\s-]+?) and ([\w\s-]+?)\?")
_PAT_IN_TOTAL = re.compile(r"How many are ([\w\s-]+?) in total\?")
_PAT_MISSING_CELL = re.compile(r"How many ([\w\s-]+?) are ([\w\s-]+?)\?")


def _round_half_up(value: Fraction, places: int) -> Fraction:
    scale = 10**places
    scaled = value * scale
    sign = -1 if scaled < 0 else 1
    magnitude = math.floor(abs(scaled) + Fraction(1, 2))
    return Fraction(sign * magnitude, scale)


def _pct_str(frac: Fraction) -> str:
    rounded = _round_half_up(frac * 100, 1)
    scaled = rounded * 10
    assert scaled.denominator == 1
    n = scaled.numerator
    sign = "-" if n < 0 else ""
    n = abs(n)
    whole, tenth = divmod(n, 10)
    return f"{sign}{whole}.{tenth}%"


class _Table:
    def __init__(self, text):
        self.rows = [s.strip() for s in _ROWS_RE.search(text).group(1).split(",")]
        self.cols = [s.strip() for s in _COLS_RE.search(text).group(1).split(",")]
        # Direct substring search keyed off the ALREADY-EXTRACTED row/col
        # name lists, rather than a generic word-character capture group --
        # category names can contain a hyphen or a leading digit, and a
        # generic `[\w\s]+?` capture cannot see either reliably. This still
        # never imports the generator: it only uses the row/col names the
        # rendered text itself declared in its own "Rows:"/"Columns:" lines.
        self.totals = {}
        for r in self.rows:
            m = re.search(re.escape(r) + r" total: (\d+)\.", text)
            if m:
                self.totals[r] = int(m.group(1))
        self.cells = {}
        for r in self.rows:
            for c in self.cols:
                m = re.search(re.escape(f"{r}/{c}") + r": (\d+)", text)
                if m:
                    self.cells[(r, c)] = int(m.group(1))

    def row_total(self, r):
        if r in self.totals:
            return self.totals[r]
        return sum(v for (rr, cc), v in self.cells.items() if rr == r)

    def cell(self, r, c):
        if (r, c) in self.cells:
            return self.cells[(r, c)]
        assert r in self.totals, f"cell ({r}, {c}) missing and row {r} has no stated total"
        others = sum(v for (rr, cc), v in self.cells.items() if rr == r)
        return self.totals[r] - others

    def col_total(self, c):
        return sum(self.cell(r, c) for r in self.rows)

    def grand_total(self):
        return sum(self.row_total(r) for r in self.rows)


def _axis(table, name):
    if name in table.rows:
        return "row"
    if name in table.cols:
        return "col"
    raise AssertionError(f"{name!r} is neither a stated row nor a stated column")


def _oracle_tw(q) -> str:
    text = q.text
    table = _Table(text)

    m = _PAT_PROP_GRAND.search(text)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        r, c = (a, b) if _axis(table, a) == "row" else (b, a)
        return _pct_str(Fraction(table.cell(r, c), table.grand_total()))

    m = _PAT_PROP_COL.search(text)
    if m:
        c, r = m.group(1).strip(), m.group(2).strip()
        return _pct_str(Fraction(table.cell(r, c), table.col_total(c)))

    m = _PAT_PROP_ROW.search(text)
    if m:
        r, c = m.group(1).strip(), m.group(2).strip()
        return _pct_str(Fraction(table.cell(r, c), table.row_total(r)))

    m = _PAT_CELL_BOTH.search(text)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        r, c = (a, b) if _axis(table, a) == "row" else (b, a)
        return str(table.cell(r, c))

    m = _PAT_IN_TOTAL.search(text)
    if m:
        name = m.group(1).strip()
        return str(table.row_total(name) if _axis(table, name) == "row" else table.col_total(name))

    m = _PAT_MISSING_CELL.search(text)
    if m:
        r, c = m.group(1).strip(), m.group(2).strip()
        return str(table.cell(r, c))

    raise AssertionError(f"oracle recognised no question pattern in: {text!r}")


@pytest.mark.parametrize("difficulty", TIERS)
def test_tw_oracle_matches_answer_text(difficulty):
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
        oracle = _oracle_tw(q)
        if oracle != q.answer_text:
            challenged.append((seed, oracle, q.answer_text, q.text))
    assert not challenged, f"oracle challenges: {challenged}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_tw_real_key_passes_oracle_over_1000_seeds(difficulty):
    for seed in range(1000):
        q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
        assert _oracle_tw(q) == q.answer_text, (seed, q.text, q.answer_text)


def test_tw_red_proof_oracle_catches_a_flipped_key():
    caught = 0
    total = 0
    for difficulty in TIERS:
        for seed in range(260):
            q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
            bogus = q.distractors[0]
            total += 1
            if _oracle_tw(q) != bogus:
                caught += 1
    assert total >= 1000
    assert caught == total, f"oracle failed to catch a flipped key on {total - caught}/{total} draws"


def test_tw_oracle_enumerates_question_shapes():
    shapes = Counter()
    for difficulty in TIERS:
        for seed in range(300):
            q = make(TOPIC_TW, seed=seed, difficulty=difficulty)
            if _PAT_PROP_GRAND.search(q.text):
                shapes["grand"] += 1
            elif _PAT_PROP_COL.search(q.text):
                shapes["col-conditional"] += 1
            elif _PAT_PROP_ROW.search(q.text):
                shapes["row-conditional"] += 1
            elif _PAT_CELL_BOTH.search(q.text):
                shapes["cell"] += 1
            elif _PAT_IN_TOTAL.search(q.text):
                shapes["margin-total"] += 1
            elif _PAT_MISSING_CELL.search(q.text):
                shapes["derived-cell"] += 1
    print(f"p19d_two_way_table: question-shape counts over 1200 draws: {dict(shapes)}")
    assert len(shapes) >= 4
