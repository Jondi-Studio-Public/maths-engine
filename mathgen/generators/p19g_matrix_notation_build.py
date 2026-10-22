"""Generates `p19g_matrix_type_and_size` and `p19g_matrix_entry_lookup` questions: classify a matrix's type and
size, and read an entry by subscript.

Matrix type/size classification and subscript entry lookup, two senior skills sharing the ``matrices`` domain:

- ``senior-recognise-different-types-of-matrices-row-column-square``
  (ACMGM014, topic ``p19g_matrix_type_and_size``, ``answer: ordered_list``):
  given a stated matrix, classify its type as row, column, square, zero,
  identity, or "none of these", and state its number of rows and its number
  of columns.
- ``senior-understand-the-matrix-definition-and-notation`` (ACMSM051, topic
  ``p19g_matrix_entry_lookup``, ``answer: number``): given a stated matrix
  and a subscript reference a_{i,j} using row-column notation, state the
  value of the entry in row i, column j.

**GRADER-KIND, load-bearing:** skill 1 is ``answer: ordered_list`` with value ``[type_label: str, rows: int,
cols: int]`` -- NOT ``answer: pair``, because `grade_pair` routes every component through `extract_values`
(numbers only), so the string type label could never grade correctly through it. `grade_ordered_list`'s
per-item rule (`mathgen/grading/typed.py`) compares a string item case-insensitively/whitespace-stripped with
no abbreviation table, and a numeric item by parsed value -- confirmed by direct probe.

**BOUNDARY, quoted in both generators below:** skill 1 never reports a
single entry value as its final answer -- only type + size. Skill 2 never
classifies the matrix's type or reports its size -- only a single entry's
numeric value, addressed by subscript notation a_{i,j}.

**Text-only constraint:** no figure. Every matrix is rendered as TEXT in
both ``question.text`` (a plain Python-literal-style nested bracket list,
e.g. ``[[1, 2], [3, 4]]``, chosen for exact machine-parseability by the
independent oracle) and ``question.latex`` (a real LaTeX ``pmatrix`` block).

**Precedence rule (load-bearing, stated verbatim in every prompt):**
"classify using the first of these that applies: identity, zero, row,
column, square, none of these." ``_classify_matrix`` below is the single
function implementing that precedence order exactly, used both to VERIFY a
matrix built to realise a chosen label (construction decides the label
first, then builds a matrix, then re-derives the label from the raw matrix
values alone -- never trusting the construction-time intent) and by the
test file's independently-written oracle, which re-implements the same rule
from scratch after parsing the rendered text, never importing this
function.

**1x1 matrices are excluded by construction** (a 1x1 matrix is
simultaneously row, column, square and -- depending on its one entry -- zero
or identity, making the precedence rule meaningless at that size): every
draw below asserts ``rows > 1 or cols > 1``.

**Answer-space contract (skill 1):** the type label is chosen first (from the tier's own label set; identity
at 0.6 weight, since it is one fixed matrix per size), then a matrix is built to realise it -- never the
reverse -- so no label keys more than ~18% of a six-label tier (~25% of easy, which draws four labels), and
"none of these" is a real share by construction.

**Answer-space contract (skill 2):** row/column indices vary across the
matrix's own bounds every draw, never fixed at a_{1,1}; every draw's matrix
entries are sampled WITHOUT REPLACEMENT from a range wide enough to hold
every cell, so every entry is pairwise distinct by construction -- this
alone guarantees a_{i,j} != a_{j,i} at any two distinct positions, checked
directly with ``check_distinct`` on every seed. The medium tier's
non-square matrix restricts the asked-for (i, j) to
``1 <= i, j <= min(rows, cols)`` (with i != j) so the transpose position
(j, i) always exists inside the matrix's own bounds -- the "swapped index"
distractor stays a live, well-defined wrong option even when rows != cols
(a rectangular matrix's a_{rows,cols} entry has no defined transpose
position when rows != cols, so asking outside the square sub-block would
make that particular distractor meaningless).
"""

from __future__ import annotations

from ..core import check, check_distinct, check_pairwise_distinct, generator, latex_answer

# ---------------------------------------------------------------------------
# Shared matrix rendering
# ---------------------------------------------------------------------------


def _matrix_text(matrix) -> str:
    """Plain Python-literal-style rendering, e.g. "[[1, 2], [3, 4]]" -- chosen
    so the independent oracle can recover it with `ast.literal_eval` from
    `question.text` alone."""
    return "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in matrix) + "]"


def _matrix_latex(matrix) -> str:
    rows_tex = " \\\\ ".join(" & ".join(str(v) for v in row) for row in matrix)
    return r"\begin{pmatrix}" + rows_tex + r"\end{pmatrix}"


# ---------------------------------------------------------------------------
# Skill 1: p19g_matrix_type_and_size -- answer: ordered_list
# ---------------------------------------------------------------------------

_LABELS = ["identity", "zero", "row", "column", "square", "none of these"]
_EASY_LABELS = ["row", "column", "square", "none of these"]

#: Identity is one fixed matrix per size, so it is drawn at 0.6 the weight of
#: the other labels: keeps the prompt pool wide without hiding the label.
_LABEL_WEIGHTS = {"identity": 0.6, "zero": 1.0, "row": 1.0, "column": 1.0, "square": 1.0, "none of these": 1.0}

_ANSWER_FORMAT_TAIL = (
    'Classify A\'s type as row, column, square, zero, identity, or "none of these", using the '
    "FIRST of these that applies, in this order: identity, zero, row, column, square, none of "
    "these. Then state its number of rows and its number of columns. Give your answer as "
    "type, rows, columns -- for example, diagonal, 2, 2 (this example is illustrative only: "
    '"diagonal" is not one of the six real type labels).'
)


def _classify_matrix(matrix) -> str:
    """The one function implementing the closed precedence order, reading
    only the raw matrix values -- never a construction-time intent tag.
    Re-implemented independently (not imported) by the test file's oracle."""
    rows, cols = len(matrix), len(matrix[0])
    if rows == cols and all(matrix[i][j] == (1 if i == j else 0) for i in range(rows) for j in range(cols)):
        return "identity"
    if all(v == 0 for row in matrix for v in row):
        return "zero"
    if rows == 1:
        return "row"
    if cols == 1:
        return "column"
    if rows == cols:
        return "square"
    return "none of these"


def _pick_label(rng, labels):
    return rng.choices(labels, weights=[_LABEL_WEIGHTS[x] for x in labels])[0]


def _nonzero_row(rng, cols, lo, hi):
    while True:
        row = [rng.randint(lo, hi) for _ in range(cols)]
        if any(v != 0 for v in row):
            return row


def _nonzero_matrix(rng, rows, cols, lo, hi):
    while True:
        m = [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]
        if any(v != 0 for row in m for v in row):
            return m


def _nonzero_value(rng, lo, hi):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _zero_matrix(rows, cols):
    return [[0] * cols for _ in range(rows)]


def _identity_matrix(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _rect_pair(rng, sizes):
    """A (rows, cols) pair with rows != cols drawn from `sizes`."""
    while True:
        rows, cols = rng.choice(sizes), rng.choice(sizes)
        if rows != cols:
            return rows, cols


def _plain_square(rng, n, lo, hi):
    while True:
        m = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]
        if _classify_matrix(m) == "square":
            return m


def _draw_easy(rng) -> tuple[str, list]:
    """Easy: row / column / square / none of these only, sizes 2-3, small
    positive entries (so identity and zero cannot occur, even by accident)."""
    label = rng.choice(_EASY_LABELS)
    if label == "row":
        matrix = [_nonzero_row(rng, rng.choice([2, 3]), 1, 9)]
    elif label == "column":
        matrix = [[v] for v in _nonzero_row(rng, rng.choice([2, 3]), 1, 9)]
    elif label == "square":
        n = rng.choice([2, 3])
        matrix = [[rng.randint(1, 9) for _ in range(n)] for _ in range(n)]
    else:
        rows, cols = rng.choice([(2, 3), (3, 2)])
        matrix = [[rng.randint(1, 9) for _ in range(cols)] for _ in range(rows)]
    return label, matrix


def _draw_medium(rng) -> tuple[str, list]:
    """Medium: all six labels, plain matrices (no traps), signed entries,
    sizes up to 4x4 including rectangles."""
    label = _pick_label(rng, _LABELS)
    dims = [2, 3, 4]
    lo, hi = -9, 9
    if label == "identity":
        matrix = _identity_matrix(rng.choice(dims))
    elif label == "zero":
        kind = rng.choice(["row", "column", "square", "rect"])
        if kind == "row":
            matrix = _zero_matrix(1, rng.choice(dims))
        elif kind == "column":
            matrix = _zero_matrix(rng.choice(dims), 1)
        elif kind == "square":
            n = rng.choice(dims)
            matrix = _zero_matrix(n, n)
        else:
            matrix = _zero_matrix(*_rect_pair(rng, dims))
    elif label == "row":
        matrix = [_nonzero_row(rng, rng.choice(dims), lo, hi)]
    elif label == "column":
        matrix = [[v] for v in _nonzero_row(rng, rng.choice(dims), lo, hi)]
    elif label == "square":
        matrix = _plain_square(rng, rng.choice(dims), lo, hi)
    else:
        rows, cols = _rect_pair(rng, dims)
        matrix = _nonzero_matrix(rng, rows, cols, lo, hi)
    return label, matrix


_HARD_BIG = [2, 3, 4, 5]
_HARD_RECT = [(2, 5), (5, 2), (3, 5), (5, 3), (4, 5), (5, 4)]


def _square_trap(rng, n, lo, hi):
    """A square matrix that looks like identity/zero but is genuinely neither:
    one entry off identity, one nonzero entry in a zero matrix, a diagonal
    matrix that is not identity, or a zero row/column inside an otherwise
    nonzero matrix. Falls through the precedence chain to "square"."""
    kind = rng.choice(["near_identity", "near_zero", "diagonal", "zero_line"])
    if kind == "near_identity":
        m = _identity_matrix(n)
        while True:
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j:
                break
        m[i][j] = _nonzero_value(rng, lo, hi)
    elif kind == "near_zero":
        m = _zero_matrix(n, n)
        m[rng.randrange(n)][rng.randrange(n)] = _nonzero_value(rng, lo, hi)
    elif kind == "diagonal":
        while True:
            d = [rng.choice([v for v in range(lo, hi + 1) if v != 0]) for _ in range(n)]
            if any(v != 1 for v in d):
                break
        m = [[d[i] if i == j else 0 for j in range(n)] for i in range(n)]
    else:
        m = _nonzero_matrix(rng, n, n, lo, hi)
        line = rng.randrange(n)
        if rng.random() < 0.5:
            m[line] = [0] * n
        else:
            for row in m:
                row[line] = 0
    return m


def _rect_trap(rng, rows, cols, lo, hi):
    """A non-square matrix ("none of these"): an identity-looking block
    (ones down the main diagonal, zeros elsewhere), a rectangle with a zero
    row or column, or a plain wide/tall rectangle."""
    kind = rng.choice(["identity_block", "zero_line", "plain"])
    if kind == "identity_block":
        return [[1 if i == j else 0 for j in range(cols)] for i in range(rows)]
    m = _nonzero_matrix(rng, rows, cols, lo, hi)
    if kind == "zero_line":
        if rng.random() < 0.5:
            m[rng.randrange(rows)] = [0] * cols
        else:
            line = rng.randrange(cols)
            for row in m:
                row[line] = 0
        if all(v == 0 for row in m for v in row):
            return _rect_trap(rng, rows, cols, lo, hi)
    return m


def _draw_hard(rng) -> tuple[str, list]:
    """Hard: identity/zero TRAPS and shapes medium never draws (5 wide/high,
    1xn up to n = 5, 2x5, 5x2, 3x5 ...)."""
    label = _pick_label(rng, _LABELS)
    lo, hi = -9, 9
    if label == "identity":
        matrix = _identity_matrix(rng.choice(_HARD_BIG))
    elif label == "zero":
        kind = rng.choice(["row", "column", "square", "rect", "rect"])
        if kind == "row":
            matrix = _zero_matrix(1, rng.choice(_HARD_BIG))
        elif kind == "column":
            matrix = _zero_matrix(rng.choice(_HARD_BIG), 1)
        elif kind == "square":
            n = rng.choice(_HARD_BIG)
            matrix = _zero_matrix(n, n)
        else:
            matrix = _zero_matrix(*rng.choice(_HARD_RECT))
    elif label in ("row", "column"):
        n = rng.choice(_HARD_BIG)
        if rng.random() < 0.5:  # identity-row look-alike: only 0 and 1
            line = [rng.randint(0, 1) for _ in range(n)]
            if not any(line):
                line[rng.randrange(n)] = 1
        else:
            line = _nonzero_row(rng, n, lo, hi)
        matrix = [line] if label == "row" else [[v] for v in line]
    elif label == "square":
        matrix = _square_trap(rng, rng.choice(_HARD_BIG), lo, hi)
    else:
        matrix = _rect_trap(rng, *rng.choice(_HARD_RECT), lo, hi)
    return label, matrix


# --- Exam: the matrix is never printed; the learner expands a_{i,j} ------------

#: (text, latex, fn). `text` uses "x" for the multiplication sign (written as
#: the real multiplication sign) and implied products ("2i"); the test file
#: evaluates the TEXT with its own parser, never these lambdas. The label is
#: always re-derived from the expanded matrix with `_classify_matrix`.
_TIMES = "×"
_EXAM_RULES = (
    ("i + j", "i + j", lambda i, j: i + j),
    ("i - j", "i - j", lambda i, j: i - j),
    (f"i {_TIMES} j", r"i \times j", lambda i, j: i * j),
    ("2i - j", "2i - j", lambda i, j: 2 * i - j),
    ("i^2 + j", "i^{2} + j", lambda i, j: i * i + j),
    ("i + 2j - 3", "i + 2j - 3", lambda i, j: i + 2 * j - 3),
    ("3i - 2j + 1", "3i - 2j + 1", lambda i, j: 3 * i - 2 * j + 1),
    (f"i {_TIMES} j - 2", r"i \times j - 2", lambda i, j: i * j - 2),
    ("i^2 - j^2", "i^{2} - j^{2}", lambda i, j: i * i - j * j),
    ("j - i^2", "j - i^{2}", lambda i, j: j - i * i),
    ("(i - 2)(j - 2)", "(i - 2)(j - 2)", lambda i, j: (i - 2) * (j - 2)),
    ("2i + j - 4", "2i + j - 4", lambda i, j: 2 * i + j - 4),
    ("1 - (i - j)^2", r"1 - (i - j)^{2}", lambda i, j: 1 - (i - j) ** 2),
    (
        "1 if i = j, and 0 otherwise",
        r"1\text{ if }i = j\text{, and }0\text{ otherwise}",
        lambda i, j: 1 if i == j else 0,
    ),
    (
        "3 if i = j, and 0 otherwise",
        r"3\text{ if }i = j\text{, and }0\text{ otherwise}",
        lambda i, j: 3 if i == j else 0,
    ),
    ("0", "0", lambda i, j: 0),
    (f"i - i {_TIMES} j", r"i - i \times j", lambda i, j: i - i * j),
    (f"j - i {_TIMES} j", r"j - i \times j", lambda i, j: j - i * j),
    ("(i - 1)(j - 1)", "(i - 1)(j - 1)", lambda i, j: (i - 1) * (j - 1)),
)

_EXAM_SIZES = [2, 3, 4, 5]


def _exam_shape(rng, label):
    if label in ("identity", "square"):
        n = rng.choice(_EXAM_SIZES)
        return n, n
    if label == "row":
        return 1, rng.choice(_EXAM_SIZES)
    if label == "column":
        return rng.choice(_EXAM_SIZES), 1
    if label == "none of these":
        return _rect_pair(rng, _EXAM_SIZES)
    while True:  # zero: any shape but 1x1
        rows, cols = rng.randint(1, 5), rng.randint(1, 5)
        if rows > 1 or cols > 1:
            return rows, cols


def _draw_exam(rng) -> tuple[str, list, tuple]:
    """Exam: label first, then a shape for it, then any rule whose expansion
    RE-CLASSIFIES (via `_classify_matrix`) as that label -- never trusted from
    intent. Returns (label, matrix, rule)."""
    label = _pick_label(rng, _LABELS)
    for _ in range(60):
        rows, cols = _exam_shape(rng, label)
        fits = []
        for rule in _EXAM_RULES:
            m = [[rule[2](i, j) for j in range(1, cols + 1)] for i in range(1, rows + 1)]
            if _classify_matrix(m) == label:
                fits.append((rule, m))
        if fits:
            rule, matrix = rng.choice(fits)
            check(rows > 1 or cols > 1, "matrix_type_and_size exam: a 1x1 matrix must never be drawn")
            return label, matrix, rule
    check(False, f"matrix_type_and_size exam: no rule realises {label!r}")
    raise AssertionError  # unreachable: check raised


def _reasoning_steps(label: str, rows: int, cols: int, expanded: str | None = None) -> list[str]:
    order = ["identity", "zero", "row", "column", "square"]
    steps = []
    if expanded is not None:
        steps.append(f"Expand the rule over the stated ranges: A = {expanded}.")
    if label == "none of these":
        steps += [f"{c.capitalize()}: no." for c in order]
        steps.append('None of the first five labels apply, so A is "none of these".')
    else:
        for c in order:
            if c == label:
                steps.append(f"{c.capitalize()}: yes -- classification stops here.")
                break
            steps.append(f"{c.capitalize()}: no.")
    steps.append(f"A has {rows} {'row' if rows == 1 else 'rows'} and {cols} {'column' if cols == 1 else 'columns'}.")
    return steps


# --- Distractors ----------------------------------------------------------------


def _types_at(rows: int, cols: int) -> list[str]:
    """Every type the precedence rule can give a matrix of this size: 1xn ->
    row/zero, nx1 -> column/zero, nxn -> square/identity/zero, rectangle ->
    none of these/zero. (1x1 is never drawn.)"""
    if rows == 1:
        return ["row", "zero"]
    if cols == 1:
        return ["column", "zero"]
    if rows == cols:
        return ["square", "identity", "zero"]
    return ["none of these", "zero"]


def _size_consistent(t: str, rows: int, cols: int) -> bool:
    """A type possible for its own size (see `_types_at`)."""
    return not (rows == 1 and cols == 1) and t in _types_at(rows, cols)


def _type_plausible(t: str, matrix) -> bool:
    """A type is only offered when THIS matrix does not contradict it at a
    glance: identity needs a square diagonal-or-one-entry-off matrix, row/column
    need a 1-2 wide dimension (a real miscount). ``zero`` is NOT filtered here:
    it is the only wrong type possible for most sizes, so it is offered at the
    key's own size and in the second size group."""
    rows, cols = len(matrix), len(matrix[0])
    if t == "identity":
        if rows != cols:
            return False
        off = sum(1 for i in range(rows) for j in range(cols) if i != j and matrix[i][j] != 0)
        diff = sum(1 for i in range(rows) for j in range(cols) if matrix[i][j] != (1 if i == j else 0))
        return off == 0 or diff == 1
    if t == "zero":
        # Only when at most one entry is nonzero or >= 75% are zero.
        zeros = sum(1 for row in matrix for v in row if v == 0)
        return rows * cols - zeros <= 1 or 4 * zeros >= 3 * rows * cols
    if t == "row":
        return rows <= 2
    if t == "column":
        return cols <= 2
    return True


def _distractor_triples(label: str, matrix, rng) -> list[tuple[str, int, int]]:
    """Three (type, rows, cols) triples pairwise distinct from the answer and
    each other. Every option's type is POSSIBLE for its own size (`_types_at`).
    The key's size is shared by one distractor (right size, wrong type) so the
    size cannot be read off the matrix alone, and the other two options share a
    second size where they can, so no dimension group singles the key out:

    - R1: key size + (natural type, zero) at a second size.
    - R2: key size + (key type, zero/other) at a second size.
    - R10: key size + the key's type at two further sizes.
    """
    rows, cols = len(matrix), len(matrix[0])
    d = (rows, cols)
    key = (label, rows, cols)

    def allowed(dims):
        return [t for t in _types_at(*dims) if _type_plausible(t, matrix)]

    sizes = {(r, c) for r in range(max(1, rows - 1), rows + 2) for c in range(max(1, cols - 1), cols + 2)}
    sizes.add((cols, rows))
    sizes = sorted(s for s in sizes if s != (1, 1) and s != d)
    # A swapped-dimensions slip is the commonest real error: weight it double.
    weighted = sizes + [s for s in sizes if s == (cols, rows) and rows != cols]

    same = [(t, rows, cols) for t in allowed(d) if t != label]

    def natural_pair():
        opts = [s for s in weighted if any(t not in (label, "zero") for t in allowed(s))]
        if not opts:
            return None
        s = rng.choice(opts)
        x = rng.choice([t for t in allowed(s) if t not in (label, "zero")])
        # The partner is "zero" only when THIS matrix could pass for zero.
        partners = [t for t in allowed(s) if t not in (label, x)]
        if not partners:
            return None
        return [(x, *s), ("zero" if "zero" in partners else rng.choice(partners), *s)]

    def keytype_pair():
        opts = [s for s in weighted if label in allowed(s) and any(t != label for t in allowed(s))]
        if not opts:
            return None
        s = rng.choice(opts)
        others = [t for t in allowed(s) if t != label]
        other = "zero" if "zero" in others else rng.choice(others)
        return [(label, *s), (other, *s)]

    def keytype_two_sizes():
        opts = [s for s in sizes if label in allowed(s)]
        if len(opts) < 2:
            return None
        s1, s2 = rng.sample(opts, 2)
        return [(label, *s1), (label, *s2)]

    builders = {"R1": natural_pair, "R2": keytype_pair, "R10": keytype_two_sizes}
    x = rng.random()
    order = ["R1", "R2", "R10"] if x < 0.62 else (["R2", "R1", "R10"] if x < 0.82 else ["R10", "R1", "R2"])
    rest = None
    if same:
        for name in order:
            rest = builders[name]()
            if rest:
                break
    if not rest:
        # No same-size wrong type (or no second size): fall back to any three
        # distinct possible options (only reachable at tiny sizes).
        pool = [(t, r, c) for r, c in sizes + [d] for t in allowed((r, c)) if (t, r, c) != key]
        picked = rng.sample(pool, 3)
    else:
        picked = [rng.choice(same)] + rest
    check(len(picked) == 3, "matrix_type_and_size: could not build three distractors")
    rng.shuffle(picked)
    return picked


PATTERN_A_SHARE = 0.70
KEY_TYPE_ABSENT_SHARE = 0.5


def _easy_triples(label: str, matrix, rng) -> list[tuple[str, int, int]]:
    """Easy: a mixture that leaves no single size-plus-consistency
    reading of the options that finds the key. Never identity or zero, the four
    options pairwise distinct, and at most one option at the TRUE size.

    - Pattern A (70%): "2 + 2". The key and one wrong-type option (row vs
      column, "none of these" vs square; impossible for its own size) at the true
      size; at ONE other size s2 the natural type for s2 and a different easy
      label that is impossible at s2. The modal sizes tie and the consistent
      options are the key and the natural one.
    - Pattern F (30%): the key alone at the true size; two options at one other
      size s2 (the natural type and an impossible easy label) and a fourth,
      natural, at a third size s3. The modal size is s2, whose consistent option
      is a distractor.
    """
    rows, cols = len(matrix), len(matrix[0])
    wrong_at_true = {"row": ["column"], "column": ["row"], "square": ["none of these"], "none of these": ["square"]}[
        label
    ]
    dims = [(r, c) for r in range(1, 5) for c in range(1, 5) if (r, c) not in ((1, 1), (rows, cols))]
    weighted = dims + [(cols, rows)] * 6 * (rows != cols)

    def nat(d):
        return _classify_matrix([[2] * d[1] for _ in range(d[0])])

    def impossible_at(d):
        return [x for x in _EASY_LABELS if not _size_consistent(x, *d) and not (x == "none of these" and 1 in d)]

    pattern_a = rng.random() < PATTERN_A_SHARE
    s2, x = None, None
    if pattern_a and rng.random() < KEY_TYPE_ABSENT_SHARE:
        # Neither option at s2 repeats the key's type, so "exactly one distractor
        # shares the key's type and one shares its size" is not a constant tell.
        pool = [d for d in weighted if nat(d) != label and [v for v in impossible_at(d) if v != label]]
        if pool:
            s2 = rng.choice(pool)
            x = rng.choice([v for v in impossible_at(s2) if v != label])
    if s2 is None:
        s2 = rng.choice(weighted)
        x = rng.choice(impossible_at(s2))
    if pattern_a:
        t = rng.choice(wrong_at_true)
        picked = [(t, rows, cols), (nat(s2), *s2), (x, *s2)]
    else:
        s3 = rng.choice([d for d in dims if d != s2])
        picked = [(nat(s2), *s2), (x, *s2), (nat(s3), *s3)]
    rng.shuffle(picked)
    return picked


_PROMPT_STEMS = (
    ("A matrix A is defined as A = ", "A matrix A is defined as A = "),
    ("Consider the matrix A = ", "Consider the matrix A = "),
    ("Let A = ", "Let A = "),
    ("The matrix A = ", "The matrix A = "),
)


def _range_phrase(var: str, n: int) -> str:
    return f"{var} = 1" if n == 1 else f"{var} = 1, ..., {n}"


def _type_and_size_question(rng, label: str, matrix, rule=None, easy: bool = False) -> dict:
    rows, cols = len(matrix), len(matrix[0])
    mtext = _matrix_text(matrix)
    mlatex = _matrix_latex(matrix)
    if rule is None:
        stem = rng.choice(_PROMPT_STEMS)[0]
        if stem == "The matrix A = ":
            text = f"{stem}{mtext} is given. {_ANSWER_FORMAT_TAIL}"
            latex = r"\text{The matrix }A = " + mlatex + r"\text{ is given. " + _ANSWER_FORMAT_TAIL + "}"
        else:
            text = f"{stem}{mtext}. {_ANSWER_FORMAT_TAIL}"
            latex = (
                r"\text{"
                + stem.split("A = ")[0].rstrip()
                + r" }A = "
                + mlatex
                + r"\text{. "
                + _ANSWER_FORMAT_TAIL
                + "}"
            )
        expanded = None
    else:
        rtext, rtex, _fn = rule
        entry = f"a_{{i,j}} = {rtext}"
        entry_tex = f"a_{{i,j}} = {rtex}"
        ri, cj = _range_phrase("i", rows), _range_phrase("j", cols)
        variant = rng.randrange(3)
        if variant == 0:
            text = (
                f"The entries of a matrix A are given by the rule {entry}, where {ri} and {cj}. {_ANSWER_FORMAT_TAIL}"
            )
            latex = (
                r"\text{The entries of a matrix }A\text{ are given by the rule }"
                + entry_tex
                + rf"\text{{, where }}{ri}\text{{ and }}{cj}"
                + r"\text{. "
                + _ANSWER_FORMAT_TAIL
                + "}"
            )
        elif variant == 1:
            text = (
                f"Matrix A is built from the rule {entry} (row number {ri}; column number {cj}). {_ANSWER_FORMAT_TAIL}"
            )
            latex = (
                r"\text{Matrix }A\text{ is built from the rule }"
                + entry_tex
                + rf"\text{{ (row number }}{ri}\text{{; column number }}"
                + rf"{cj}\text{{). "
                + _ANSWER_FORMAT_TAIL
                + "}"
            )
        else:
            text = f"Write out the matrix A defined by {entry} for {ri} and {cj}. {_ANSWER_FORMAT_TAIL}"
            latex = (
                r"\text{Write out the matrix }A\text{ defined by }"
                + entry_tex
                + rf"\text{{ for }}{ri}\text{{ and }}{cj}"
                + r"\text{. "
                + _ANSWER_FORMAT_TAIL
                + "}"
            )
        expanded = mtext

    answer = [label, rows, cols]
    answer_text = f"{label}, {rows}, {cols}"
    triples = _easy_triples(label, matrix, rng) if easy else _distractor_triples(label, matrix, rng)
    # A structural (type-word, int, int) triple is not a sympy expression --
    # `check_pairwise_distinct`/`symbolically_equal` would try to `sympify`
    # the multi-word label "none of these" and raise. Plain Python-tuple
    # equality is the correct, exact distinctness check here (no numeric
    # tolerance is meaningful for a type label anyway).
    all_triples = [tuple(answer), *triples]
    check(
        len(all_triples) == len(set(all_triples)),
        "matrix_type_and_size: a distractor collides with the answer or another option",
    )
    check(
        easy or all(_size_consistent(t, r, c) for t, r, c in all_triples),
        "matrix_type_and_size: an option is inconsistent with its own size",
    )
    if easy:
        check(
            sum(1 for _t, r, c in triples if (r, c) == (rows, cols)) <= 1
            and all(t not in ("identity", "zero") for t, _r, _c in triples),
            "matrix_type_and_size easy: at most one true-size distractor, no zero/identity",
        )
    distractors = [f"{t}, {r}, {c}" for t, r, c in triples]
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=_reasoning_steps(label, rows, cols, expanded),
        distractors=distractors,
    )


@generator(
    "p19g_matrix_type_and_size",
    level=19,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p19g_matrix_type_and_size(rng, variant="easy"):
    """Given a stated matrix, classify its type as row, column, square, zero,
    identity, or none of these, and state its number of rows and its number
    of columns (ACMGM014). Never reports a single entry value -- only type
    and size (contrast the sibling ``p19g_matrix_entry_lookup``, which never
    classifies type or reports size).

    Tiers: easy = row/column/square/none only, sizes 2-3, small
    positive entries; medium = all six labels, plain, signed, up to 4x4
    including rectangles; hard = identity/zero traps and 5-wide/high shapes;
    exam = the matrix is NOT printed, only the rule a_{i,j} and the ranges."""
    if variant == "easy":
        label, matrix = _draw_easy(rng)
    elif variant == "medium":
        label, matrix = _draw_medium(rng)
    elif variant == "hard":
        label, matrix = _draw_hard(rng)
    elif variant == "exam":
        label, matrix, rule = _draw_exam(rng)
        return _type_and_size_question(rng, label, matrix, rule)
    else:
        raise ValueError(f"unknown variant: {variant}")
    rows, cols = len(matrix), len(matrix[0])
    check(rows > 1 or cols > 1, "matrix_type_and_size: a 1x1 matrix must never be drawn")
    check(
        _classify_matrix(matrix) == label,
        f"matrix_type_and_size: constructed for {label!r} but reclassifies as {_classify_matrix(matrix)!r}",
    )
    return _type_and_size_question(rng, label, matrix, easy=(variant == "easy"))


# ---------------------------------------------------------------------------
# Skill 2: p19g_matrix_entry_lookup -- answer: number
# ---------------------------------------------------------------------------

_ENTRY_TEMPLATES = (
    "A matrix A is defined as A = {mtext}. Using row-column notation a_{{i,j}}, state the "
    "value of a_{{{i},{j}}} -- the entry in row {i}, column {j}.",
    "Given the matrix A = {mtext}, state the value of the entry a_{{{i},{j}}} (row {i}, "
    "column {j}), using row-column notation.",
    "A = {mtext}. Find the value of a_{{{i},{j}}}, the entry in row {i} and column {j} of A.",
)

#: Each rule's coefficients are chosen so it is injective (no two entries
#: collide) for every square size this generator ever draws (n up to 5,
#: i.e. i, j in 1..5): `6i - j` is safe because the j-coefficient's whole
#: range of differences (at most 4) can never match a nonzero multiple of
#: the i-coefficient (6); `i + 6j` and `9i - 2j` are safe by the same
#: argument with the dominant coefficient on the other index. Verified
#: directly at generator build time by `check(len(set(...)) == n * n, ...)`
#: below, not merely asserted here.
_ENTRY_RULE_POOL = (
    ("a_{i,j} = 6i - j", "6i - j", lambda i, j: 6 * i - j),
    ("a_{i,j} = i + 6j", "i + 6j", lambda i, j: i + 6 * j),
    ("a_{i,j} = 9i - 2j", "9i - 2j", lambda i, j: 9 * i - 2 * j),
)


def _distinct_matrix(rng, rows, cols, lo, hi):
    values = rng.sample(range(lo, hi + 1), rows * cols)
    return [values[r * cols : (r + 1) * cols] for r in range(rows)]


def _verify_entry_lookup(matrix, i: int, j: int, claimed) -> None:
    """Re-index the RAW matrix directly (1-indexed i, j), never a value the
    construction loop already held onto."""
    recomputed = matrix[i - 1][j - 1]
    check(recomputed == claimed, f"matrix_entry_lookup: entry mismatch at ({i}, {j}): {recomputed} vs {claimed}")


def _entry_distractors(matrix, i: int, j: int, answer) -> list[int]:
    rows, cols = len(matrix), len(matrix[0])
    swapped = matrix[j - 1][i - 1]
    alt_row = ((i - 1) + 1) % rows
    alt_col = ((j - 1) + 1) % cols
    cands = [swapped, matrix[alt_row][j - 1], matrix[i - 1][alt_col]]
    check_distinct(
        answer,
        *cands,
        msg="matrix_entry_lookup: a distractor equals the answer (entries must be pairwise distinct by construction)",
    )
    check_pairwise_distinct(answer, *cands, msg="matrix_entry_lookup: distractors collide with each other")
    return cands


def _entry_question(matrix, i: int, j: int) -> dict:
    """Builds an easy/medium-tier entry-lookup question: a directly listed
    matrix (no algebraic rule), numeric i/j stated inline. `_pick_entry_text`
    selects one of three structurally distinct wordings deterministically
    from (i, j) itself (not from an rng draw, since this helper is a plain
    function of the already-drawn matrix/indices) -- the wording variety
    across a real sweep comes from i/j varying across seeds."""
    answer = matrix[i - 1][j - 1]
    _verify_entry_lookup(matrix, i, j, answer)
    mtext = _matrix_text(matrix)
    mlatex = _matrix_latex(matrix)

    text = _pick_entry_text(i, j, mtext)
    latex = _pick_entry_latex(i, j, mlatex)
    cands = _entry_distractors(matrix, i, j, answer)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"Read row {i}, column {j} of A.", f"a_{{{i},{j}}} = {answer}."],
        distractors=[str(c) for c in cands],
    )


def _pick_entry_text(i, j, mtext) -> str:
    idx = (i * 7 + j) % len(_ENTRY_TEMPLATES)
    return _ENTRY_TEMPLATES[idx].format(mtext=mtext, i=i, j=j)


def _pick_entry_latex(i, j, mlatex) -> str:
    return (
        r"\text{A matrix A is defined as }A = "
        + mlatex
        + r"\text{. Using row-column notation }a_{i,j}"
        + r"\text{, state the value of }a_{"
        + f"{i},{j}"
        + "}"
        + rf"\text{{ -- the entry in row {i}, column {j}.}}"
    )


def _draw_indices_square(rng, n) -> tuple[int, int]:
    i = rng.randint(1, n)
    j = rng.randint(1, n)
    while j == i:
        j = rng.randint(1, n)
    return i, j


def _entry_easy(rng) -> dict:
    n = rng.choice([2, 3])
    matrix = _distinct_matrix(rng, n, n, 1, 40)
    i, j = _draw_indices_square(rng, n)
    return _entry_question(matrix, i, j)


def _entry_medium(rng) -> dict:
    rows, cols = rng.choice([(4, 5), (5, 4)])
    matrix = _distinct_matrix(rng, rows, cols, -30, 30)
    m = min(rows, cols)
    i = rng.randint(1, m)
    j = rng.randint(1, m)
    while j == i:
        j = rng.randint(1, m)
    return _entry_question(matrix, i, j)


_ROW_PHRASES = ("last", "second_last_relative")
_COL_PHRASES = ("second", "second_last_relative")


def _entry_hard(rng) -> dict:
    n = rng.choice([3, 4, 5])
    matrix = _distinct_matrix(rng, n, n, -40, 40)
    for _ in range(50):
        row_kind = rng.choice(_ROW_PHRASES)
        col_kind = rng.choice(_COL_PHRASES)
        i = n if row_kind == "last" else n - 1
        j = 2 if col_kind == "second" else n - 1
        if i != j and 1 <= i <= n and 1 <= j <= n:
            break
    else:
        # Every combination collided (only possible when n == 3 and both
        # relative phrasings land on the same index) -- fall back to a
        # guaranteed-distinct pair, keeping row_kind/col_kind consistent
        # with the i/j actually used so the rendered phrase never lies.
        row_kind, col_kind = "last", "second"
        i, j = n, 2
    row_text = "the last row" if row_kind == "last" else f"row m - 1 (the matrix has m = {n} rows)"
    col_text = "the second column" if col_kind == "second" else f"column n - 1 (the matrix has n = {n} columns)"

    answer = matrix[i - 1][j - 1]
    _verify_entry_lookup(matrix, i, j, answer)
    mtext = _matrix_text(matrix)
    mlatex = _matrix_latex(matrix)
    text = (
        f"A matrix A (with {n} rows and {n} columns) is given by A = {mtext}. State the value "
        f"of the entry a_{{i,j}} where i is {row_text} and j is {col_text}."
    )
    latex = (
        rf"\text{{A matrix A (with {n} rows and {n} columns) is given by }}A = "
        + mlatex
        + r"\text{. State the value of the entry }a_{i,j}"
        + rf"\text{{ where i is {row_text} and j is {col_text}.}}"
    )
    cands = _entry_distractors(matrix, i, j, answer)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"Resolve the index reference: i = {i}, j = {j}.",
            f"Read row {i}, column {j} of A.",
            f"a_{{{i},{j}}} = {answer}.",
        ],
        distractors=[str(c) for c in cands],
    )


def _entry_exam(rng) -> dict:
    n = rng.choice([3, 4, 5])
    rule_text, rule_bare, fn = rng.choice(_ENTRY_RULE_POOL)
    matrix = [[fn(i + 1, j + 1) for j in range(n)] for i in range(n)]
    check(
        len({v for row in matrix for v in row}) == n * n,
        f"matrix_entry_lookup exam: rule {rule_text!r} did not give distinct entries at n={n}",
    )
    i, j = _draw_indices_square(rng, n)
    answer = matrix[i - 1][j - 1]
    _verify_entry_lookup(matrix, i, j, answer)
    # Rule-only prompt: the expanded matrix stays internal (distractors are
    # read from it) so the learner must substitute i and j into the rule.
    text = (
        f"A matrix A has entries given by the rule {rule_text}, for i = 1, ..., {n} (rows) and "
        f"j = 1, ..., {n} (columns). State the value of a_{{{i},{j}}} -- "
        f"the entry in row {i}, column {j}."
    )
    latex = (
        r"\text{A matrix A has entries given by the rule }"
        + rule_text
        + rf"\text{{, for }} i = 1, ..., {n} \text{{ (rows) and }} j = 1, ..., {n} "
        + rf"\text{{ (columns). State the value of }}a_{{{i},{j}}}\text{{ -- the entry in row {i}, column {j}.}}"
    )
    cands = _entry_distractors(matrix, i, j, answer)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"Substitute i = {i}, j = {j} into a_{{i,j}} = {rule_bare} to get a_{{{i},{j}}} = {answer}.",
        ],
        distractors=[str(c) for c in cands],
    )


@generator(
    "p19g_matrix_entry_lookup",
    level=19,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def p19g_matrix_entry_lookup(rng, variant="easy"):
    """Given a stated matrix and a subscript reference a_{i,j} using
    row-column notation, state the value of the entry in row i, column j
    (ACMSM051). Never classifies the matrix's type or reports its size --
    only a single entry's numeric value (contrast the sibling
    ``p19g_matrix_type_and_size``, which never reports a single entry)."""
    if variant == "easy":
        return _entry_easy(rng)
    if variant == "medium":
        return _entry_medium(rng)
    if variant == "hard":
        return _entry_hard(rng)
    if variant == "exam":
        return _entry_exam(rng)
    raise ValueError(f"unknown variant: {variant}")
