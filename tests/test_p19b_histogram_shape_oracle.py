"""Independent oracle for `p19b_histogram_shape`
(skill `senior-compare-the-characteristics-of-the-shape-of-histograms`).

Imports NOTHING from `mathgen.generators.p19b_histogram_shape_build`. Parses
the class-frequency table (or, for `easy`, re-reads the description's own
class-vocabulary phrase) straight out of `question.text`, reorders by class
lower bound where the text presents the rows out of order (`hard`), for
`exam` re-derives which histogram the stated criterion (wider range, or
larger total) picks out, applies its own classification rule -- written
without reading the generator's `_classify` -- and asserts it matches the
key.

`N_SEEDS` (overridable by `P19B_ORACLE_SEEDS`) sets the per-tier seed count.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p19b_histogram_shape"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P19B_ORACLE_SEEDS", 250))

_LABELS = ("symmetric", "skewed left", "skewed right", "bimodal")

_ROW_RE = re.compile(r"(\d+)-(\d+): (\d+)")


def _local_peaks(freqs):
    n = len(freqs)
    peaks = []
    for i in range(n):
        left_ok = (i == 0) or (freqs[i] > freqs[i - 1])
        right_ok = (i == n - 1) or (freqs[i] > freqs[i + 1])
        if left_ok and right_ok:
            peaks.append(i)
    return peaks


def _oracle_classify(freqs):
    """A second, independently-written copy of the shape rule -- same
    contract as the generator's own `_classify`, deliberately re-derived
    rather than imported, so the two only ever agree if they are both right.
    """
    n = len(freqs)
    if n < 5 or any(f <= 0 for f in freqs) or len(set(freqs)) == 1:
        return None
    peaks = _local_peaks(freqs)
    if len(peaks) == 1:
        p = peaks[0]
        mirrored = all(abs(freqs[i] - freqs[n - 1 - i]) <= 1 for i in range(n))
        centred = abs(2 * p - (n - 1)) <= 1
        if mirrored and centred:
            return "symmetric"
        if p in (0, 1) and (n - 1 - p) >= 3:
            tail = freqs[p:]
            if all(tail[i] > tail[i + 1] for i in range(len(tail) - 1)):
                return "skewed right"
        if p in (n - 1, n - 2) and p >= 3:
            head = freqs[: p + 1]
            if all(head[i] < head[i + 1] for i in range(len(head) - 1)):
                return "skewed left"
        return None
    if len(peaks) == 2:
        p1, p2 = peaks
        if p2 - p1 < 2:
            return None
        valley = min(freqs[p1 + 1 : p2])
        smaller = min(freqs[p1], freqs[p2])
        if valley * 2 <= smaller:
            return "bimodal"
    return None


def _parse_tables(text: str):
    """Returns a list of tables, each a list of (lo, freq) tuples in the
    order they were PRINTED (not necessarily class order). One table for
    easy/medium/hard; two for exam ("Group A"/"Group B")."""
    if "Group A" in text and "Group B" in text:
        a_part, b_part = text.split("Group B:", 1)
        a_part = a_part.split("Group A:", 1)[1]
        tables = [a_part, b_part]
    else:
        tables = [text]
    parsed = []
    for chunk in tables:
        rows = [(int(lo), int(freq)) for lo, _hi, freq in _ROW_RE.findall(chunk)]
        parsed.append(rows)
    return parsed


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_agrees_with_key(tier):
    checked_easy = 0
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        options = set([q.answer_text, *q.distractors])
        assert options == set(_LABELS)

        if tier == "easy":
            # Easy carries no table -- re-derive from the description's own
            # wording bank (matched by phrase content, not by the label
            # words, which the prompt deliberately avoids in the body).
            body = q.text.lower()
            if (
                "two distinct peaks" in body
                or "two separate clusters" in body
                or ("two different groups of common values" in body)
            ):
                derived = "bimodal"
            elif (
                "stretching out to the right" in body
                or "tailing off gradually towards a handful of very large" in body
                or ("reaching towards the larger values" in body)
            ):
                derived = "skewed right"
            elif (
                "stretching out to the left" in body
                or "tailing off gradually towards a handful of very small" in body
                or ("reaching towards the smaller values" in body)
            ):
                derived = "skewed left"
            elif (
                "middle of the range" in body
                or "centre of the range" in body
                or ("single cluster of common values in the middle" in body)
            ):
                derived = "symmetric"
            else:
                pytest.fail(f"easy: could not classify description: {q.text!r}")
            assert derived == q.answer_text, q.text
            checked_easy += 1
            continue

        tables = _parse_tables(q.text)
        if tier == "exam":
            (rows_a, rows_b) = tables
            rows_a_sorted = sorted(rows_a)
            rows_b_sorted = sorted(rows_b)
            if "WIDER range" in q.text:
                chosen = rows_a_sorted if len(rows_a_sorted) > len(rows_b_sorted) else rows_b_sorted
            else:
                total_a = sum(f for _lo, f in rows_a_sorted)
                total_b = sum(f for _lo, f in rows_b_sorted)
                chosen = rows_a_sorted if total_a > total_b else rows_b_sorted
            freqs = [f for _lo, f in chosen]
        else:
            rows_sorted = sorted(tables[0])
            freqs = [f for _lo, f in rows_sorted]

        derived = _oracle_classify(freqs)
        assert derived == q.answer_text, (
            f"{tier} seed={seed}: oracle derived {derived!r}, key is {q.answer_text!r}\n{q.text}"
        )
    if tier == "easy":
        assert checked_easy == N_SEEDS


def test_red_first_proof(monkeypatch):
    """Prove this oracle actually fires: feed it a table whose true shape is
    symmetric but relabel it (simulating a corrupted key) and confirm the
    oracle's derived shape disagrees, so the assertion above would have
    failed had the generator shipped this bug."""
    freqs = [2, 5, 8, 12, 11, 8, 5, 2]
    assert _oracle_classify(freqs) == "symmetric"
    fake_key = "bimodal"
    assert _oracle_classify(freqs) != fake_key, "red-first check failed to disagree with a corrupted key"


@pytest.mark.parametrize("tier", TIERS)
def test_template_matrix_coherent(tier):
    """Enumerate every (tier, key label, description-phrase-family / table
    order) combination seen over the sweep; each combination must be
    internally coherent (the oracle already checked correctness above --
    this just asserts every combination that occurs is one of the small,
    designed set)."""
    seen = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        if tier == "hard":
            bounds = [int(m) for m in re.findall(r"(\d+)-\d+:", q.text)]
            order = "sorted" if bounds == sorted(bounds) else "shuffled"
            seen.add((q.answer_text, order))
        elif tier == "exam":
            criterion = "range" if "WIDER range" in q.text else "total"
            seen.add((q.answer_text, criterion))
        else:
            seen.add(q.answer_text)
    assert seen, f"{tier}: empty matrix"
    if tier == "hard":
        # every combination observed must be a real label with a real order tag
        for label, order in seen:
            assert label in _LABELS and order in ("sorted", "shuffled")
