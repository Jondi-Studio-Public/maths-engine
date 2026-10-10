"""Shared plumbing for generators whose answer kind is `table` (a grid with blank cells).

The `table` grader (`mathgen/grading/typed.py`, "-- table" section) reads `Question.answer` as

    {"columns": [corner, header, ...],
     "rows": [[row label, cell, ...], ...],      # None marks a blank
     "blanks": {"r,c": value, ...}}              # 0-based into "rows"

and grades every blank. The canonical `answer_text` is `table_answer_text(answer)`, i.e.
"<row label> <column header> = <value>" per blank, in reading order, which is also the typed form the
grader reads back. Counts here are plain ints, so a blank is right only when it is exactly that integer.

This module is not a generator (its name starts with an underscore, so the generators package skips it).
It holds the pieces both table generators need so neither hand-rolls them: building the answer dict,
rendering the grid as one line of plain text, and turning a "what a learner with one specific error would fill in"
edit into a distractor that the real grader is proven to reject.
"""

from __future__ import annotations

from types import SimpleNamespace

from mathgen.core import check
from mathgen.grading.typed import grade_table, table_answer_text

BLANK_MARK = "?"
SEPARATOR = " — "


def build_answer(columns: list[str], full_rows: list[list], blank_cells: set[tuple[int, int]]) -> dict:
    """The `table` answer dict: `full_rows` with the cells in `blank_cells` (row, column) replaced by None
    and their true values moved to `blanks`. A blank's column index counts the label column as 0."""
    check(blank_cells, "a table question needs at least one blank")
    rows, blanks = [], {}
    for r, row in enumerate(full_rows):
        out = list(row)
        for c in range(1, len(row)):
            if (r, c) in blank_cells:
                blanks[f"{r},{c}"] = row[c]
                out[c] = None
        rows.append(out)
    return {"columns": list(columns), "rows": rows, "blanks": blanks}


def grid_text(answer: dict) -> str:
    """The grid as one line of plain text (question text may not contain line breaks): the column headers,
    then each row, rows separated by semicolons, blanks shown as "?". Reads only the masked `rows`, so the
    answer cannot leak. The client draws the real grid from `answer` with `table_prompt`."""
    header = SEPARATOR.join(str(h) for h in answer["columns"])
    rows = (SEPARATOR.join(BLANK_MARK if cell is None else str(cell) for cell in row) for row in answer["rows"])
    return f"Table ({header}): " + "; ".join(rows) + "."


def _probe(answer: dict, answer_text: str):
    return SimpleNamespace(answer=answer, answer_text=answer_text)


def with_blanks(answer: dict, changes: dict[str, int]) -> dict:
    """A copy of `answer` whose blanks are overridden by `changes` ("r,c" -> value)."""
    blanks = dict(answer["blanks"])
    for key, value in changes.items():
        check(key in blanks, "an edit must target a blank")
        blanks[key] = value
    return {"columns": answer["columns"], "rows": answer["rows"], "blanks": blanks}


def finish(answer: dict, wrong_edits: list[tuple[str, dict[str, int]]], max_distractors: int = 3):
    """(answer_text, distractors, option_notes) for a table question.

    `wrong_edits` is an ordered list of (misconception id, {"r,c": wrong value}). Each is turned into a full
    wrong table in the canonical text format. Edits that change nothing, or that duplicate an earlier
    distractor, are dropped. Every kept distractor is graded by the real `table` grader and must be
    rejected, and the canonical answer must be accepted: both are checked here, on every candidate."""
    answer_text = table_answer_text(answer)
    probe = _probe(answer, answer_text)
    check(grade_table(probe, answer_text).correct, "table grader rejects its own canonical answer")
    distractors: list[str] = []
    notes: dict[str, str] = {}
    for note_id, changes in wrong_edits:
        if len(distractors) >= max_distractors:
            break
        if all(answer["blanks"][k] == v for k, v in changes.items()):
            continue
        if any(v < 0 for v in changes.values()):
            continue
        text = table_answer_text(with_blanks(answer, changes))
        if text == answer_text or text in distractors:
            continue
        check(not grade_table(probe, text).correct, "table grader accepts a deliberately wrong table")
        distractors.append(text)
        notes[text] = note_id
    return answer_text, distractors, notes


def pick_blanks(rng, candidates: list[tuple[int, int]], lo: int, hi: int, must: list[tuple[int, int]] | None = None):
    """A random set of between `lo` and `hi` blank positions from `candidates`, always including `must`."""
    must = list(must or [])
    rest = [c for c in candidates if c not in must]
    hi = min(hi, len(must) + len(rest))
    lo = min(max(lo, len(must)), hi)
    n = rng.randint(lo, hi)
    chosen = must + rng.sample(rest, n - len(must))
    return set(chosen)
