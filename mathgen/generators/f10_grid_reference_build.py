"""``f10_grid_reference_pathway``: follow a pathway on a lettered-column, numbered-row grid.

Skill ``f10-grid-reference-pathway`` (AC9M4SP02). The learner is told the grid's size, how the rows are
numbered, a starting square such as ``C2`` and a sequence of directional moves, and states the finishing
square. The answer is a *labelled pair* of numbers, ``column = 5, row = 3``, with the column given as its
position in the alphabet (A = 1, B = 2, ...): the pair grader reads numbers, not letters, and the
conversion from letter to number is itself part of the reasoning. The drawn-grid version of this skill
needs a figure and is not built here.

Tiers:

- easy:   two or three moves in plain words (left, right, up, down); row 1 is the bottom row.
- medium: three to five compass moves (north is up the page); row 1 is still the bottom row.
- hard:   rows are numbered from the TOP, so moving south increases the row number. Alternates a
          longer forward path (five or six moves) with an inverse question: given the start and the
          finish, how many squares were travelled in each of two named directions?

The grader is ``grade_pair``. The answer is labelled, so the labels carry the meaning and a reply may
list them in either order. A distractor is only kept when it differs from the answer in at least one
labelled value.
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# direction word -> (dx, dy) in SCREEN terms: x grows to the right, y grows down the page.
_PLAIN = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}
_COMPASS = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}


def _squares(n: int) -> str:
    return f"{n} square" if n == 1 else f"{n} squares"


def _ref(col: int, row: int) -> str:
    return f"{_LETTERS[col - 1]}{row}"


def _screen_row(row: int, rows: int, top_down: bool) -> int:
    """Convert a row number to a screen y (0 at the top of the page)."""
    return row - 1 if top_down else rows - row


def _row_from_screen(y: int, rows: int, top_down: bool) -> int:
    return y + 1 if top_down else rows - y


def _random_walk(rng, words, cols, rows, top_down, start, n_moves, max_step):
    """Pick `n_moves` in-bounds moves from `start`, with both axes used and a different finish.

    Returns (moves, finish) where moves is a list of (word, count); or None if the draw failed.
    Positions here are (col, row) as the learner sees them.
    """
    col, row = start
    moves = []
    previous_axis = None
    for _ in range(n_moves):
        options = []
        for word, (dx, dy) in words.items():
            axis = "h" if dx else "v"
            if axis == previous_axis:
                continue  # alternate axes so adjacent legs never merge into one longer leg
            for count in range(1, max_step + 1):
                ncol = col + dx * count
                nrow = row + (dy * count if top_down else -dy * count)
                if 1 <= ncol <= cols and 1 <= nrow <= rows:
                    options.append((word, count, ncol, nrow, axis))
        if not options:
            return None
        word, count, col, row, previous_axis = rng.choice(options)
        moves.append((word, count))
    if (col, row) == start:
        return None
    return moves, (col, row)


def _finish_by_screen(words, moves, start, rows, top_down):
    """Second route to the finish: work in screen coordinates, then convert back."""
    x = start[0]
    y = _screen_row(start[1], rows, top_down)
    for word, count in moves:
        dx, dy = words[word]
        x += dx * count
        y += dy * count
    return x, _row_from_screen(y, rows, top_down)


def _forward(rng, cols, rows, top_down, words, n_moves, max_step):
    for _ in range(300):
        start = (rng.randint(1, cols), rng.randint(1, rows))
        walk = _random_walk(rng, words, cols, rows, top_down, start, n_moves, max_step)
        if walk is None:
            continue
        moves, finish = walk
        axes = {("h" if words[w][0] else "v") for w, _ in moves}
        if axes != {"h", "v"}:
            continue
        keyed = _forward_distractors(words, moves, start, finish, top_down, cols, rows)
        if len(keyed) >= 3:
            break
    else:  # pragma: no cover - defensive, the loop converges within a handful of draws
        raise RuntimeError("could not draw a grid pathway")

    check(
        _finish_by_screen(words, moves, start, rows, top_down) == finish,
        "grid finish differs between the learner-view walk and the screen-coordinate walk",
    )
    col, row = finish
    steps = [_squares_phrase(w, c) for w, c in moves]
    row_rule = (
        f"Rows are numbered from the top, so row 1 is the top row and row {rows} is the bottom row."
        if top_down
        else f"Row 1 is the bottom row and row {rows} is the top row."
    )
    compass_note = " North is towards the top of the page." if "north" in words else ""
    text = (
        f"A map grid has columns {_LETTERS[0]} to {_LETTERS[cols - 1]} from left to right. {row_rule}{compass_note} "
        f"A walker starts in square {_ref(*start)}. The walker moves {', then '.join(steps)}. "
        f"Which square does the walker finish in? Give it as column = _, row = _, "
        f"where the column is its number from the left ({_LETTERS[0]} = 1, {_LETTERS[1]} = 2, and so on)."
    )
    answer = {"column": col, "row": row}
    answer_text = f"column = {col}, row = {row}"

    rng.shuffle(keyed)
    notes = dict(keyed)
    picked_texts = pick_distractors(answer_text, [t for t, _ in keyed])[:3]
    for t in picked_texts:
        check(t != answer_text, "a grid distractor equals the answer")

    working = [f"Start at {_ref(*start)}: column {start[0]}, row {start[1]}."]
    c, r = start
    for w, n in moves:
        dx, dy = words[w]
        c += dx * n
        r += dy * n if top_down else -dy * n
        working.append(f"Move {_squares_phrase(w, n)}: column {c}, row {r}.")
    working.append(f"Finish at {_ref(col, row)}, so {answer_text}.")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=picked_texts,
        option_notes={t: notes[t] for t in picked_texts},
    )


def _forward_distractors(words, moves, start, finish, top_down, cols, rows):
    """(option text, misconception id) pairs, each from repeating the walk with one specific slip.

    Only squares that lie on the grid are kept, and nothing equal to the answer.
    """
    col, row = finish

    def land(slip_moves, vertical_sign=1, horizontal_sign=1):
        c, r = start
        for w, n in slip_moves:
            dx, dy = words[w]
            c += dx * n * horizontal_sign
            r += (dy * n if top_down else -dy * n) * vertical_sign
        return c, r

    last_word, last_count = moves[-1]
    candidates = {
        "swapped-column-row": (row, col),
        "vertical-direction-reversed": land(moves, vertical_sign=-1),
        "horizontal-direction-reversed": land(moves, horizontal_sign=-1),
        "drops-a-leg": land(moves[:-1]),
        "start-square-counted": land(moves[:-1] + [(last_word, last_count + 1)]),
        "letter-numbered-from-zero": (col - 1, row),
    }
    keyed = []
    for note, (c, r) in candidates.items():
        if 1 <= c <= cols and 1 <= r <= rows and (c, r) != (col, row):
            keyed.append((f"column = {c}, row = {r}", note))
    return keyed


def _squares_phrase(word, count):
    return f"{_squares(count)} {word}"


def _inverse(rng, cols, rows):
    """Hard inverse: start and finish given (rows numbered from the top); count the squares per direction."""
    for _ in range(200):
        h_word = rng.choice(["east", "west"])
        v_word = rng.choice(["north", "south"])
        dx = 1 if h_word == "east" else -1
        dy = 1 if v_word == "south" else -1  # row number grows southwards
        across = rng.randint(1, cols - 1)
        vertical = rng.randint(1, rows - 1)
        cols_ok = [c for c in range(1, cols + 1) if 1 <= c + dx * across <= cols]
        rows_ok = [r for r in range(1, rows + 1) if 1 <= r + dy * vertical <= rows]
        if cols_ok and rows_ok:
            c0 = rng.choice(cols_ok)
            r0 = rng.choice(rows_ok)
            break
    else:  # pragma: no cover
        raise RuntimeError("could not draw an inverse grid pathway")
    c1, r1 = c0 + dx * across, r0 + dy * vertical
    # Check by reading the finish references back from the answer.
    check(
        abs(c1 - c0) == across and abs(r1 - r0) == vertical,
        "inverse grid counts do not match the two references",
    )
    answer = {h_word: across, v_word: vertical}
    answer_text = f"{h_word} = {across}, {v_word} = {vertical}"
    text = (
        f"A map grid has columns {_LETTERS[0]} to {_LETTERS[cols - 1]} from left to right. "
        f"Rows are numbered from the top, so row 1 is the top row and row {rows} is the bottom row. "
        f"North is towards the top of the page. A walker goes from square {_ref(c0, r0)} to square {_ref(c1, r1)} "
        f"moving only {h_word} and {v_word}. How many squares did the walker travel {h_word} and how many "
        f"{v_word}? Give it as {h_word} = _, {v_word} = _."
    )
    candidates = [
        (f"{h_word} = {vertical}, {v_word} = {across}", "swapped-column-row"),
        (f"{h_word} = {across + 1}, {v_word} = {vertical + 1}", "start-square-counted"),
        (f"{h_word} = {c1}, {v_word} = {r1}", "reports-finish-not-difference"),
        (f"{h_word} = {across}, {v_word} = {r1}", "reports-finish-not-difference"),
    ]
    rng.shuffle(candidates)
    picked = pick_distractors(answer_text, [t for t, _ in candidates])[:3]
    notes = dict(candidates)
    working = [
        f"Columns: {_LETTERS[c0 - 1]} is column {c0} and {_LETTERS[c1 - 1]} is column {c1}, so {_squares(abs(c1 - c0))} {h_word}.",
        f"Rows: row {r0} to row {r1} is {_squares(abs(r1 - r0))} {v_word}.",
        f"So {answer_text}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=picked,
        option_notes={t: notes[t] for t in picked},
    )


@generator(
    "f10_grid_reference_pathway",
    level=6,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def f10_grid_reference_pathway(rng, tier="easy"):
    """Follow a stated pathway on a lettered-column, numbered-row grid and state the finishing square."""
    if tier == "easy":
        return _forward(rng, cols=6, rows=6, top_down=False, words=_PLAIN, n_moves=rng.randint(2, 3), max_step=3)
    if tier == "medium":
        return _forward(rng, cols=8, rows=8, top_down=False, words=_COMPASS, n_moves=rng.randint(3, 5), max_step=4)
    if tier == "hard":
        if rng.random() < 0.5:
            return _inverse(rng, cols=10, rows=9)
        return _forward(rng, cols=10, rows=9, top_down=True, words=_COMPASS, n_moves=rng.randint(5, 6), max_step=5)
    raise ValueError("tier must be easy, medium or hard")
