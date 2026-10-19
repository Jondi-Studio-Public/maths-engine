"""Generates `r4a_group_table_check` questions (easy, medium, hard) and the exam chain `exd_x_r4a_group_table_check`.

``uc2-group-table-check`` (``abstract-algebra``, answer ``table``): complete missing entries of the Cayley table of a
finite group whose elements are labelled by letters. The tiers differ in what forces an entry.

* easy: a group of order 6 or 8 with the identity labelled e and 3 to 5 blanks, every one of which is forced by the
  Latin square rule alone (each element once in each row and column), applied again and again.
* medium: a group of order 6 or 8 with four blanks in a 2×2 block that the Latin rule cannot settle (swapping the two
  symbols is also a Latin square); only the group axioms (associativity, with the entries already known) decide it.
* hard: a non-abelian group of order 8 (the dihedral group of the square or the quaternion group) with the identity not
  named, the same 2×2 block of blanks and two more blanks.
* exam: a group of order 8 with the identity not named and a 2×2 block of blanks: (a) the missing entries, (b) how many
  elements x satisfy x∘x = identity, (c) the order of a named element, (d) how many elements have order 2, which follows
  through from (b).

Each table is built from a real group, relabelled at random. Every key is checked by counting the completions of the
blanks that satisfy the group axioms (identity, associativity, inverses): exactly one, and it is the key. The tier rules
are checked the same way, by repeated application of the Latin rule, and by exhaustive search for the medium and hard
tiers.
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import attempts, check, generator
from ..grading.typed import grade_table, table_answer_text
from ._f10_table_kit import build_answer, grid_text, with_blanks
from ._r4a_alg_kit import number_part, with_latex

# -- small groups -------------------------------------------------------------------------------------------------


def _cyclic(n):
    els = list(range(n))
    return els, lambda a, b: (a + b) % n


def _dihedral(n):
    els = [(a, b) for a in range(n) for b in range(2)]
    return els, lambda x, y: ((x[0] + (-1) ** x[1] * y[0]) % n, (x[1] + y[1]) % 2)


_UNIT_PRODUCT = {
    ("1", "1"): (1, "1"), ("1", "i"): (1, "i"), ("1", "j"): (1, "j"), ("1", "k"): (1, "k"),
    ("i", "1"): (1, "i"), ("i", "i"): (-1, "1"), ("i", "j"): (1, "k"), ("i", "k"): (-1, "j"),
    ("j", "1"): (1, "j"), ("j", "i"): (-1, "k"), ("j", "j"): (-1, "1"), ("j", "k"): (1, "i"),
    ("k", "1"): (1, "k"), ("k", "i"): (1, "j"), ("k", "j"): (-1, "i"), ("k", "k"): (-1, "1"),
}  # fmt: skip


def _quaternion():
    els = [(s, u) for s in (1, -1) for u in "1ijk"]

    def mul(x, y):
        sign, unit = _UNIT_PRODUCT[(x[1], y[1])]
        return (x[0] * y[0] * sign, unit)

    return els, mul


def _product(m, n):
    els = [(a, b) for a in range(m) for b in range(n)]
    return els, lambda x, y: ((x[0] + y[0]) % m, (x[1] + y[1]) % n)


GROUPS = {
    "Z6": (_cyclic, (6,)),
    "S3": (_dihedral, (3,)),
    "Z8": (_cyclic, (8,)),
    "Z4xZ2": (_product, (4, 2)),
    "D4": (_dihedral, (4,)),
    "Q8": (_quaternion, ()),
    "Z2^3": (None, ()),
}


def _index_table(name):
    """(table, identity) for a named group: table[x][y] is the index of the product of elements x and y."""
    if name == "Z2^3":
        els = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]

        def mul(x, y):
            return tuple((p + q) % 2 for p, q in zip(x, y))
    else:
        build, args = GROUPS[name]
        els, mul = build(*args)
    where = {e: i for i, e in enumerate(els)}
    table = [[where[mul(x, y)] for y in els] for x in els]
    identity = next(i for i in range(len(els)) if all(table[i][j] == j for j in range(len(els))))
    return table, identity


# -- group-axiom checks --------------------------------------------------------------------------------------------


def is_group_table(t):
    """Whether the square table t (entries are indices) is the table of a group: a Latin square with an identity that
    is associative."""
    m = len(t)
    if any(sorted(row) != list(range(m)) for row in t) or any(sorted(col) != list(range(m)) for col in zip(*t)):
        return False
    if not any(all(t[e][j] == j and t[j][e] == j for j in range(m)) for e in range(m)):
        return False
    return all(t[t[a][b]][c] == t[a][t[b][c]] for a in range(m) for b in range(m) for c in range(m))


def _latin_candidates(t, r, c):
    m = len(t)
    used = {t[r][j] for j in range(m) if t[r][j] is not None} | {t[i][c] for i in range(m) if t[i][c] is not None}
    return [v for v in range(m) if v not in used]


def latin_propagation_solves(full, blanks):
    """Whether repeating "a cell with only one symbol left by its row and column takes it" fills every blank."""
    t = [list(row) for row in full]
    for r, c in blanks:
        t[r][c] = None
    todo = set(blanks)
    progress = True
    while todo and progress:
        progress = False
        for r, c in sorted(todo):
            options = _latin_candidates(t, r, c)
            if len(options) == 1:
                t[r][c] = options[0]
                todo.discard((r, c))
                progress = True
    return not todo and all(t[r][c] == full[r][c] for r, c in blanks)


def group_completions(full, blanks, limit=2):
    """How many ways (counted up to `limit`) the blanks can be filled so the table is that of a group."""
    t = [list(row) for row in full]
    for r, c in blanks:
        t[r][c] = None
    order = sorted(blanks)
    found = 0

    def fill(k):
        nonlocal found
        if found >= limit:
            return
        if k == len(order):
            if is_group_table(t):
                found += 1
            return
        r, c = order[k]
        for v in _latin_candidates(t, r, c):
            t[r][c] = v
            fill(k + 1)
        t[r][c] = None

    fill(0)
    return found


def _intercalates(t, identity):
    """Pairs of rows and pairs of columns (all avoiding the identity) where the 2×2 block is x y / y x."""
    m = len(t)
    out = []
    for r1 in range(m):
        for r2 in range(r1 + 1, m):
            for c1 in range(m):
                for c2 in range(c1 + 1, m):
                    if identity in (r1, r2, c1, c2):
                        continue
                    if t[r1][c1] == t[r2][c2] and t[r1][c2] == t[r2][c1] and t[r1][c1] != t[r1][c2]:
                        out.append((r1, r2, c1, c2))
    return out


# -- question building -------------------------------------------------------------------------------------------


def _labels(rng, m, named_identity, identity):
    """Display names for the m elements: 'e' for the identity (when named), distinct letters for the rest."""
    letters = [ch for ch in "fghkmnpqrstuvwxyz"]  # never a-d: those are the exam part labels
    rng.shuffle(letters)
    names = letters[:m]
    if named_identity:
        names[identity] = "e"
    return names


def _table_question(rng, name, tier, extra_blanks):
    table, identity = _index_table(name)
    m = len(table)
    named = tier in ("easy", "medium")
    block = None
    for _ in attempts(400, "a labelling and blank pattern for the tier"):
        names = _labels(rng, m, named, identity)
        order = sorted(range(m), key=lambda i: "" if names[i] == "e" else names[i])
        if tier == "easy":
            nonid = [i for i in range(m) if i != identity]
            cells = [(r, c) for r in nonid for c in nonid]
            blanks = set(rng.sample(cells, rng.randint(3, 5)))
            if not latin_propagation_solves(table, blanks):
                continue
        else:
            blocks = _intercalates(table, identity)
            if not blocks:
                raise AssertionError(f"{name} has no 2x2 block")
            r1, r2, c1, c2 = rng.choice(blocks)
            blanks = {(r1, c1), (r1, c2), (r2, c1), (r2, c2)}
            block = set(blanks)
            nonid = [i for i in range(m) if i != identity]
            extras = [(r, c) for r in nonid for c in nonid if (r, c) not in blanks]
            blanks |= set(rng.sample(extras, extra_blanks))
            if latin_propagation_solves(table, blanks) or group_completions(table, blanks) != 1:
                continue
        break
    position = {el: k for k, el in enumerate(order)}
    columns = ["∘"] + [names[i] for i in order]
    rows = [[names[r]] + [names[table[r][c]] for c in order] for r in order]
    blank_cells = {(position[r], position[c] + 1) for r, c in blanks}
    answer = build_answer(columns, rows, blank_cells)
    return table, identity, names, order, position, blanks, answer, block


def _wrong_tables(table, names, order, position, blanks, block):
    """Wrong fillings: [(table text, misconception id)], each built from one named error."""
    out = []
    keyed = {(position[r], position[c] + 1): (r, c) for r, c in blanks}

    def edit(fn):
        changes = {}
        for (row, col), (r, c) in keyed.items():
            v = fn(r, c)
            changes[f"{row},{col}"] = names[v]
        return changes

    out.append((edit(lambda r, c: table[c][r]), "cayley-table-row-column-swapped"))
    # the other Latin-square filling of the 2x2 block: the two symbols swapped
    swapped = {}
    for r, c in block or ():
        partner = next((r2, c2) for r2, c2 in block if r2 == r and c2 != c)
        swapped[f"{position[r]},{position[c] + 1}"] = names[table[partner[0]][partner[1]]]
    if swapped:
        out.append((swapped, "cayley-latin-fill-without-associativity"))
    ordered = sorted(keyed.items())
    (row, col), (r, c) = ordered[0]
    known_row = [table[r][j] for j in range(len(table)) if (r, j) not in blanks]
    if known_row:
        out.append(({f"{row},{col}": names[known_row[0]]}, "cayley-row-repeats-element"))
    (row, col), (r, c) = ordered[-1]
    known_col = [table[i][c] for i in range(len(table)) if (i, c) not in blanks]
    if known_col:
        out.append(({f"{row},{col}": names[known_col[0]]}, "cayley-column-repeats-element"))
    (row, col), (r, c) = ordered[len(ordered) // 2]
    out.append(({f"{row},{col}": names[c]}, "cayley-entry-copies-column-label"))
    out.append(({f"{row},{col}": names[r]}, "cayley-entry-copies-row-label"))
    return out


def _distractors(answer, wrong):
    text = table_answer_text(answer)
    probe = type("Probe", (), {"answer": answer, "answer_text": text})()
    check(grade_table(probe, text).correct, "the table grader rejects its own answer")
    out, seen = [], {text}
    for changes, tag in wrong:
        wrong_text = table_answer_text(with_blanks(answer, changes))
        if wrong_text in seen:
            continue
        check(not grade_table(probe, wrong_text).correct, "the table grader accepts a deliberately wrong table")
        seen.add(wrong_text)
        out.append(Distractor(wrong_text, tag))
    return text, out


_REPLY = "Give each missing entry as “x y = z”, meaning x∘y = z, with the row element first."
_STEMS = {
    "easy": "The table below is the Cayley table of a group under the operation ∘; the entry in row x and column y is x∘y. "
    "Some entries are missing. {grid} Fill in the missing entries. " + _REPLY,
    "medium": "The table below is the Cayley table of a group under the operation ∘; the entry in row x and column y is x∘y. "
    "Four entries are missing. {grid} Fill in the missing entries. " + _REPLY,
    "hard": "The table below is the Cayley table of a non-abelian group under the operation ∘; the entry in row x and "
    "column y is x∘y. Some entries are missing. {grid} Fill in the missing entries. " + _REPLY,
}


def _single(rng, tier):
    if tier == "easy":
        name = rng.choice(["Z6", "S3", "Z8", "Z4xZ2", "D4", "Q8"])
        extra = 0
    elif tier == "medium":
        name = rng.choice(["Z6", "S3", "Z8", "Z4xZ2", "D4", "Q8"])
        extra = 0
    else:
        name = rng.choice(["D4", "Q8"])
        extra = 2
    table, identity, names, order, position, blanks, answer, block = _table_question(rng, name, tier, extra)
    keyed = {(r, c) for r, c in blanks}
    check(group_completions(table, keyed) == 1, "the blanks do not have exactly one group completion")
    text, options = _distractors(answer, _wrong_tables(table, names, order, position, blanks, block))
    check(len(options) >= 2, "too few distractors")
    if tier == "easy":
        working = [
            "Each missing entry is the only symbol its row and column do not already contain; fill those first and repeat."
        ]
    else:
        working = [
            "The Latin square rule leaves two ways to fill the block, so use associativity: write a row element as a "
            "product of two known entries and move the bracket.",
        ]
    working.append("The completed entries are: " + text + ".")
    return dict(
        text=_STEMS[tier].format(grid=grid_text(answer)),
        answer_kind="table",
        answer=answer,
        answer_text=text,
        working=working,
        distractors=options,
    )


@generator(
    "r4a_group_table_check",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_latex
def r4a_group_table_check(rng, tier="easy"):
    """Complete a Cayley table; the tiers are described in the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return _single(rng, tier)


# -- exam -------------------------------------------------------------------------------------------------------


def _element_order(table, identity, x):
    k, y = 1, x
    while y != identity:
        y = table[y][x]
        k += 1
    return k


@generator("exd_x_r4a_group_table_check", level=19, presets={"exam": dict(tier="exam")})
@with_latex
def exd_x_r4a_group_table_check(rng, tier="exam"):
    """Exam chain: a Cayley table of a group of order 8 with a 2×2 block missing, then square roots of the identity."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(100, "a group with an element of order 4 or more"):
        name = rng.choice(["D4", "Q8", "Z4xZ2", "Z8"])
        table, identity, names, order, position, blanks, answer, block = _table_question(rng, name, "exam", 0)
        m = len(table)
        orders = [_element_order(table, identity, x) for x in range(m)]
        squares = sum(1 for x in range(m) if table[x][x] == identity)
        if max(orders) >= 4:
            break
    check(group_completions(table, blanks) == 1, "the blanks do not have exactly one group completion")
    g = rng.choice([x for x in range(m) if orders[x] >= 4])
    og = orders[g]
    involutions = sum(1 for x in range(m) if orders[x] == 2)
    check(squares == involutions + 1, "square roots of the identity differ from 1 + elements of order 2")
    check(all(is_group_table(table) for _ in (0,)), "the source table is not a group table")
    text, options = _distractors(answer, _wrong_tables(table, names, order, position, blanks, block))
    check(len(options) >= 2, "too few distractors")
    ident_name = names[identity]
    part_a = dict(
        prompt="Fill in the missing entries. " + _REPLY,
        answer_kind="table",
        answer=answer,
        answer_text=text,
        marks=3,
        working=[
            "The Latin square rule leaves two fillings of the block; associativity picks one.",
            "Entries: " + text + ".",
        ],
        distractors=options,
    )
    part_b = number_part(
        "How many elements x of the group satisfy x∘x = the identity? (The identity itself counts.)",
        squares,
        2,
        [
            f"The identity is {ident_name}, the element whose row repeats the column headings.",
            f"The diagonal entries equal {ident_name} for {squares} elements.",
        ],
        [
            (squares - 1, "square-root-count-omits-identity"),
            (m // 2, None),
            (m, "every-element-squares-to-identity"),
            (max(orders), None),
        ],
        sensible_range=(0, None),
    )
    part_c = number_part(
        f"Find the order of the element {names[g]}.",
        og,
        2,
        [f"The powers of {names[g]} first return to {ident_name} after {og} steps."],
        [
            (m, "element-order-taken-as-group-order"),
            (og * 2, None),
            (og // 2, None),
            (sorted(set(orders))[-1] + 1, None),
        ],
        sensible_range=(0, None),
    )
    part_d = number_part(
        "How many elements of the group have order 2?",
        involutions,
        1,
        [f"Remove the identity from the {squares} square roots of the identity: {squares} - 1 = {involutions}."],
        [(squares, "involution-count-includes-identity"), (m // 2, None), (squares + 1, None), (squares - 2, None)],
        follow_through="part_b - 1",
        sensible_range=(0, None),
    )
    return dict(
        text=(
            "The table below is the Cayley table of a group G of order 8 under the operation ∘; the entry in row x "
            f"and column y is x∘y. Four entries are missing. {grid_text(answer)}"
        ),
        parts=[part_a, part_b, part_c, part_d],
    )
