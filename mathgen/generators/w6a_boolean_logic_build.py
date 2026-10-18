"""Generates `w6a_boolean_logic` questions (easy, medium, hard) and the exam chain `exd_x_w6a_boolean_logic`.

``boolean-algebra-logic-gates`` (``digital-logic``-style content in ``logic-computability-and-information``, answer
``number``): evaluate Boolean expressions and gate circuits described in words, read truth tables, and simplify with a
Karnaugh map. Every answer is a count or a value read off a truth table, so none can be had by guessing 0 or 1.

* easy: a three-variable expression (``A·B + ¬C``) whose rows with output 1 are counted; a two-variable expression whose
  output column, read as a binary number, is the answer; or the same count for a two-gate circuit described in words.
* medium: the minimal sum-of-products of a three-variable function from its Karnaugh map (total literals, or product
  terms), the number of rows on which two expressions differ, or a three-gate circuit with NAND, NOR and XOR gates.
* hard: a four-variable Karnaugh map (groups wrap round the edges), a map with don't-care cells, or a four-input,
  four-gate circuit counted over sixteen rows.
* exam (``exd_x_w6a_boolean_logic``): a three-gate circuit in words. (a) the number of rows with output 1, (b) the row
  numbers, (c) the literals in the minimal sum-of-products, (d) the literals saved against the full minterm form.

Every answer is computed by evaluating the circuit or expression row by row, and every simplification count by an
exhaustive search over the prime implicants (the cheapest cover by product terms, then by literals), checked
against a search over all covers for one more term.
"""

from __future__ import annotations

from itertools import combinations, product

from ..core import attempts, check, generator, phrase
from ._w6a_kit import enough, num_dist, num_fields, part, retry_draw

TOPIC = "w6a_boolean_logic"
EXAM_TOPIC = "exd_x_w6a_boolean_logic"

AREA = "boolean-logic"
MISCONCEPTIONS = [
    ("bool-reads-zero-rows", "Counts or lists the rows where the output is 0 instead of the rows where it is 1.", AREA),
    (
        "bool-precedence-or-before-and",
        "Applies the operations left to right, or the lower-priority operation first, instead of doing NOT, then AND, "
        "then OR or XOR.",
        AREA,
    ),
    ("bool-ignores-negation", "Evaluates a negated input as if the NOT were not there.", AREA),
    (
        "bool-gate-type-confused",
        "Evaluates a NAND, NOR or XOR gate as the AND or OR gate it is built from, forgetting the inversion or the exclusion.",
        AREA,
    ),
    (
        "bool-bit-order-reversed",
        "Reads the input bits with the last variable as the most significant, so every row number is the wrong one.",
        AREA,
    ),
    (
        "bool-row-numbering-from-one",
        "Numbers the truth table rows from 1 instead of 0, so every row number is one too large.",
        AREA,
    ),
    (
        "bool-compares-totals-not-rows",
        "Compares how many ones each function has instead of comparing them row by row.",
        AREA,
    ),
    ("bool-counts-agreeing-rows", "Counts the rows on which two functions agree where the rows on which they differ were asked for.", AREA),
    ("kmap-no-grouping", "Writes one full product term for every cell with a 1, without grouping any cells.", AREA),
    ("kmap-terms-not-literals", "Gives the number of groups where the total number of literals was asked for.", AREA),
    ("kmap-groups-too-small", "Stops at groups of two cells and misses a larger group of four or eight.", AREA),
    (
        "kmap-literals-as-variables",
        "Counts every variable in every group, forgetting that a variable which changes across a group drops out.",
        AREA,
    ),
    ("kmap-dont-care-as-zero", "Treats a don't-care cell as a 0, so the groups cannot use it.", AREA),
    ("kmap-miscount-literals", "Miscounts the literals in the simplified expression by one.", AREA),
    ("bool-saving-uses-terms", "Subtracts the product terms instead of the literals when finding how many literals are saved.", AREA),
]

_VARS = "ABCD"
_ARTICLE = {"AND": "an AND", "OR": "an OR", "NOT": "a NOT", "NAND": "a NAND", "NOR": "a NOR", "XOR": "an XOR"}


# -- evaluating -------------------------------------------------------------------------------------------------------


def _gate(op: str, a: int, b: int = 0) -> int:
    return {
        "AND": a & b,
        "OR": a | b,
        "NOT": 1 - a,
        "NAND": 1 - (a & b),
        "NOR": 1 - (a | b),
        "XOR": a ^ b,
    }[op]


def _rows(n: int):
    """Every input row as a tuple of bits, row number 0 first, the first variable the most significant bit."""
    return [tuple((r >> (n - 1 - i)) & 1 for i in range(n)) for r in range(2**n)]


def _ones(f, n: int) -> list[int]:
    return [r for r, bits in enumerate(_rows(n)) if f(*bits)]


# -- three-variable expressions with a known wrong reading ---------------------------------------------------------------

# (text pattern, true function, function when the order of operations is misapplied)
_TEMPLATES = (
    ("{X}·{Y} + {Z}", lambda x, y, z: (x & y) | z, lambda x, y, z: x & (y | z)),
    ("({X} + {Y})·{Z}", lambda x, y, z: (x | y) & z, lambda x, y, z: x | (y & z)),
    ("{X} ⊕ {Y}·{Z}", lambda x, y, z: x ^ (y & z), lambda x, y, z: (x ^ y) & z),
    ("{X} + {Y}·{Z}", lambda x, y, z: x | (y & z), lambda x, y, z: (x | y) & z),
)


def _literals(rng):
    """Three literals of A, B, C in random order with at least one negation: [(variable index, negated)]."""
    order = [0, 1, 2]
    rng.shuffle(order)
    flags = [rng.random() < 0.5 for _ in range(3)]
    if not any(flags):
        flags[rng.randrange(3)] = True
    return list(zip(order, flags))


def _lit_text(var: int, neg: bool) -> str:
    return ("¬" if neg else "") + _VARS[var]


def _expression(rng):
    """(text, f, wrong_order_f, f_ignoring_negation) for a random three-variable expression."""
    pattern, true_f, wrong_f = rng.choice(_TEMPLATES)
    lits = _literals(rng)
    text = pattern.format(X=_lit_text(*lits[0]), Y=_lit_text(*lits[1]), Z=_lit_text(*lits[2]))

    def lit_values(bits, drop_not=False):
        return [bits[v] ^ (0 if (drop_not or not neg) else 1) for v, neg in lits]

    def f(a, b, c):
        return true_f(*lit_values((a, b, c)))

    def g(a, b, c):
        return wrong_f(*lit_values((a, b, c)))

    def h(a, b, c):
        return true_f(*lit_values((a, b, c), drop_not=True))

    return text, f, g, h


_ORDER_NOTE = "Apply ¬ first, then ·, then + or ⊕."


# -- circuits in words ------------------------------------------------------------------------------------------------


class _Circuit:
    def __init__(self, inputs, gates):
        self.inputs = list(inputs)
        self.gates = gates  # [(name, op, [signal names])]

    def evaluate(self, bits, swap=None):
        """Output for the input bits; `swap` maps an operation to the one a careless reader uses in its place."""
        signals = dict(zip(self.inputs, bits))
        for name, op, args in self.gates:
            used = (swap or {}).get(op, op)
            signals[name] = _gate(used, *(signals[a] for a in args))
        return signals[self.gates[-1][0]]

    def function(self, swap=None):
        return lambda *bits: self.evaluate(bits, swap)

    def words(self) -> str:
        lead = "A logic circuit has inputs " + _join_names(self.inputs) + "."
        sentences = [lead]
        for name, op, args in self.gates:
            described = [self._signal(a) for a in args]
            kind = _ARTICLE[op]
            if op == "NOT":
                sentences.append(f"Gate {name} is {kind} gate whose input is {described[0]}.")
            else:
                sentences.append(f"Gate {name} is {kind} gate whose inputs are {described[0]} and {described[1]}.")
        sentences.append(f"The output F is the output of gate {self.gates[-1][0]}.")
        return " ".join(sentences)

    def _signal(self, signal: str) -> str:
        return signal if signal in self.inputs else f"the output of gate {signal}"

    def uses(self, ops) -> bool:
        return any(op in ops for _, op, _ in self.gates)


def _join_names(names) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _circuit(rng, n_inputs: int, n_gates: int, ops, want_special: bool = False) -> _Circuit:
    inputs = list(_VARS[:n_inputs])
    gate_names = "PQRS"
    for _ in attempts(2000, f"a {n_gates}-gate circuit on {n_inputs} inputs"):
        signals = list(inputs)
        gates = []
        for i in range(n_gates):
            op = rng.choice(ops)
            arity = 1 if op == "NOT" else 2
            if len(signals) < arity:
                break
            args = rng.sample(signals, arity)
            gates.append((gate_names[i], op, args))
            signals.append(gate_names[i])
        else:
            used = {a for _, _, args in gates for a in args}
            if not all(name in used for name in inputs + [g[0] for g in gates[:-1]]):
                continue
            if sum(1 for _, op, _ in gates if op == "NOT") > 1:
                continue
            circuit = _Circuit(inputs, gates)
            if want_special and not circuit.uses({"NAND", "NOR", "XOR"}):
                continue
            n_ones = len(_ones(circuit.function(), n_inputs))
            if 2 <= n_ones <= 2**n_inputs - 2:
                return circuit
    raise AssertionError  # pragma: no cover


# -- Karnaugh maps: prime implicants and the cheapest cover ---------------------------------------------------------------


def _cubes(n: int):
    """Every product term over n variables as (pattern, mask of the rows it covers): pattern entries 0, 1 or None."""
    out = []
    for pattern in product((0, 1, None), repeat=n):
        mask = 0
        for r, bits in enumerate(_rows(n)):
            if all(p is None or p == b for p, b in zip(pattern, bits)):
                mask |= 1 << r
        out.append((pattern, mask))
    return out


def _mask(rows) -> int:
    m = 0
    for r in rows:
        m |= 1 << r
    return m


def _primes(n: int, care_mask: int, max_dashes: int | None = None):
    valid = [(p, m) for p, m in _cubes(n) if m and m & ~care_mask == 0]
    primes = [(p, m) for p, m in valid if not any(m != m2 and m & m2 == m for _, m2 in valid)]
    if max_dashes is not None:
        primes = [(p, m) for p, m in primes if p.count(None) <= max_dashes]
    return primes


def _literals_of(pattern) -> int:
    return sum(1 for p in pattern if p is not None)


def _covers(primes, ones_mask: int, size: int):
    for combo in combinations(primes, size):
        covered = 0
        for _, m in combo:
            covered |= m
        if covered & ones_mask == ones_mask:
            yield combo


def _cheapest(n: int, ones, dcs=(), max_dashes=None):
    """(product terms, literals) of the cheapest cover of `ones` by prime implicants of `ones` + `dcs`, by terms then
    literals; None when `max_dashes` makes a cover impossible. Also checks no cover with one more term has fewer
    literals, so the answer does not depend on how "minimal" is read."""
    ones_mask = _mask(ones)
    primes = _primes(n, ones_mask | _mask(dcs), max_dashes)
    for size in range(1, len(primes) + 1):
        best = min((sum(_literals_of(p) for p, _ in combo) for combo in _covers(primes, ones_mask, size)), default=None)
        if best is not None:
            larger = min(
                (sum(_literals_of(p) for p, _ in combo) for combo in _covers(primes, ones_mask, size + 1)), default=None
            )
            return size, best, larger
    return None


def _kmap_function(rng, n: int, lo: int, hi: int, with_dc: bool = False, min_terms: int = 2, min_literals: int = 3):
    """(ones, dcs, terms, literals) for a random function whose cheapest cover is unambiguous and not trivial."""
    rows = list(range(2**n))
    for _ in attempts(3000, "a Karnaugh-map function"):
        count = rng.randint(lo, hi)
        chosen = rng.sample(rows, count + (rng.randint(1, 3) if with_dc else 0))
        ones = sorted(chosen[:count])
        dcs = sorted(chosen[count:])
        result = _cheapest(n, ones, dcs)
        terms, literals, larger = result
        if terms < min_terms or (larger is not None and larger < literals):
            continue
        if literals < min_literals:
            continue
        if literals >= n * len(ones):  # no grouping at all: not a Karnaugh-map question
            continue
        if with_dc:
            plain = _cheapest(n, ones)
            if plain[1] == literals:  # the don't-cares change nothing
                continue
        return ones, dcs, terms, literals
    raise AssertionError  # pragma: no cover


def _kmap_distractors(n, ones, dcs, terms, literals, with_dc=False):
    cands = [(len(ones) * n, "kmap-no-grouping"), (terms, "kmap-terms-not-literals")]
    pairs = _cheapest(n, ones, dcs, max_dashes=1)
    if pairs is not None:
        cands.append((pairs[1], "kmap-groups-too-small"))
    cands.append((terms * n, "kmap-literals-as-variables"))
    if with_dc:
        cands.insert(0, (_cheapest(n, ones)[1], "kmap-dont-care-as-zero"))
    cands.extend([(literals + 1, "kmap-miscount-literals"), (literals - 1, "kmap-miscount-literals")])
    return num_dist(literals, cands)


_GRAY = (0, 1, 3, 2)


def _kmap_grid(n: int, ones, dcs) -> str:
    """The map as rows of cell values: 3 variables A | BC, 4 variables AB | CD, both in Gray-code order."""
    ones, dcs = set(ones), set(dcs)

    def cell(r):
        return "d" if r in dcs else "1" if r in ones else "0"

    if n == 3:
        rows = [" ".join(cell(a * 4 + _GRAY[c] ) for c in range(4)) for a in range(2)]
        labels = ["A = 0", "A = 1"]
        cols = "BC = 00, 01, 11, 10"
    else:
        rows = [" ".join(cell(_GRAY[a] * 4 + _GRAY[c]) for c in range(4)) for a in range(4)]
        labels = [f"AB = {g:02b}" for g in _GRAY]
        cols = "CD = 00, 01, 11, 10"
    body = "; ".join(f"{label}: {row}" for label, row in zip(labels, rows))
    return f"the Karnaugh map of F has columns {cols} and reads {body}"


def _kmap_listing(n: int, ones, dcs) -> str:
    names = ", ".join(_VARS[:n])
    text = f"F({names}) is 1 on the row{'s' if len(ones) != 1 else ''} numbered {', '.join(map(str, ones))}"
    if dcs:
        text += f", and its output is a don't care on row{'s' if len(dcs) != 1 else ''} {', '.join(map(str, dcs))}"
    return text + f" (the inputs read as a binary number with {_VARS[0]} the leftmost digit)"


# -- the tiers --------------------------------------------------------------------------------------------------------------


def _answer(value: int) -> dict:
    return num_fields(value)


def _count_stem(rng, text, kind):
    return phrase(
        rng,
        (
            "Consider the Boolean function F = {e}, with inputs A, B and C. " + _ORDER_NOTE + " For how many of the "
            "8 combinations of inputs does F equal 1?",
            "A logic circuit has output F = {e}, where A, B and C are inputs. " + _ORDER_NOTE + " How many rows of "
            "its truth table show an output of 1?",
            "Write out the truth table of F = {e} for the inputs A, B and C. " + _ORDER_NOTE + " On how many of the "
            "8 rows is the output 1?",
        ),
        e=text,
    )


def _easy_expression(rng):
    for _ in attempts(500, "an expression with a non-trivial count"):
        text, f, g, h = _expression(rng)
        n = len(_ones(f, 3))
        if 2 <= n <= 6:
            break
    ones = n
    check(ones == sum(f(*bits) for bits in _rows(3)), "row count differs")
    cands = [
        (8 - ones, "bool-reads-zero-rows"),
        (len(_ones(g, 3)), "bool-precedence-or-before-and"),
        (len(_ones(h, 3)), "bool-ignores-negation"),
        (ones + 1, "kmap-miscount-literals"),
        (ones - 1, "kmap-miscount-literals"),
    ]
    dist = num_dist(ones, cands)
    enough(dist)
    working = [
        f"Evaluate F = {text} on each of the 8 rows of A, B, C (000 to 111).",
        f"F = 1 on rows {', '.join(map(str, _ones(f, 3)))}, which is {ones} rows.",
    ]
    return dict(text=_count_stem(rng, text, None), working=working, distractors=dist, **_answer(ones), answer_kind="number")


_TWO_VAR = (
    ("A ⊕ B", lambda a, b: a ^ b),
    ("¬(A·B)", lambda a, b: 1 - (a & b)),
    ("¬(A + B)", lambda a, b: 1 - (a | b)),
    ("A + ¬B", lambda a, b: a | (1 - b)),
    ("¬A·B", lambda a, b: (1 - a) & b),
    ("A·B + ¬A·¬B", lambda a, b: 1 - (a ^ b)),
    ("¬A + B", lambda a, b: (1 - a) | b),
    ("A·¬B", lambda a, b: a & (1 - b)),
)


def _easy_function_number(rng):
    text, f = rng.choice(_TWO_VAR)
    column = [f(*bits) for bits in _rows(2)]
    value = int("".join(map(str, column)), 2)
    check(0 < value < 15, "a constant column")
    swapped = [column[0], column[2], column[1], column[3]]  # rows ordered 00, 10, 01, 11 (A least significant)
    cands = [
        (int("".join(map(str, column[::-1])), 2), "bool-bit-order-reversed"),
        (15 - value, "bool-reads-zero-rows"),
        (int("".join(map(str, swapped)), 2), "bool-bit-order-reversed"),
        (int("".join(str(1 - b) for b in column[::-1]), 2), "bool-reads-zero-rows"),
        (value + 1, "kmap-miscount-literals"),
    ]
    dist = num_dist(value, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "A gate network has output F = {e}. List F for the inputs AB = 00, 01, 10, 11 (in that order, top row first) "
            "and read the four outputs as a binary number, the top output on the left. What is that number in decimal?",
            "Find the truth table of F = {e} with rows AB = 00, 01, 10, 11 from the top. Writing the output column from "
            "top to bottom as a four-digit binary number gives what whole number in decimal?",
            "For F = {e}, write the output for each of AB = 00, 01, 10, 11, top to bottom. Read that column as a "
            "binary number with the top output first. Give its decimal value.",
        ),
        e=text,
    )
    working = [
        f"The outputs for AB = 00, 01, 10, 11 are {', '.join(map(str, column))}.",
        f"As a binary number that is {''.join(map(str, column))}, which is {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _easy_circuit(rng):
    circuit = _circuit(rng, 3, 2, ("AND", "OR", "NOT", "NAND", "NOR", "XOR"))
    return _circuit_count(rng, circuit, 3, "easy")


def _circuit_distractors(circuit, n_inputs, ones):
    total = 2**n_inputs
    swap = {"NAND": "AND", "NOR": "OR", "XOR": "OR"}
    cands = [(total - ones, "bool-reads-zero-rows")]
    if circuit.uses(swap):
        cands.append((len(_ones(circuit.function(swap), n_inputs)), "bool-gate-type-confused"))
    cands.extend([(ones + 1, "kmap-miscount-literals"), (ones - 1, "kmap-miscount-literals"), (total // 2, "kmap-miscount-literals")])
    return num_dist(ones, cands)


def _circuit_count(rng, circuit, n_inputs, tier):
    ones = len(_ones(circuit.function(), n_inputs))
    dist = _circuit_distractors(circuit, n_inputs, ones)
    enough(dist)
    total = 2**n_inputs
    ask = phrase(
        rng,
        (
            "For how many of the {t} combinations of inputs is F equal to 1?",
            "Complete the truth table of this circuit. How many of its {t} rows have an output of 1?",
            "How many input rows out of {t} make the output F equal 1?",
        ),
        t=total,
    )
    working = [
        f"Evaluate each gate in turn on all {total} rows.",
        f"F = 1 on rows {', '.join(map(str, _ones(circuit.function(), n_inputs)))}: {ones} rows.",
    ]
    return dict(text=f"{circuit.words()} {ask}", working=working, distractors=dist, **_answer(ones), answer_kind="number")


def _medium_kmap(rng):
    ones, dcs, terms, literals = _kmap_function(rng, 3, 3, 6)
    ask_terms = rng.random() < 0.3
    value = terms if ask_terms else literals
    cands = (
        [(literals, "kmap-terms-not-literals"), (len(ones), "kmap-no-grouping"), (terms + 1, "kmap-miscount-literals"), (terms * 3, "kmap-literals-as-variables")]
        if ask_terms
        else None
    )
    dist = num_dist(terms, cands) if ask_terms else _kmap_distractors(3, ones, dcs, terms, literals)
    enough(dist)
    show = _kmap_grid(3, ones, dcs) if rng.random() < 0.5 else _kmap_listing(3, ones, dcs)
    if ask_terms:
        ask = "Using a Karnaugh map, how many product terms are in the simplest sum-of-products expression for F?"
    else:
        ask = phrase(
            rng,
            (
                "Group the 1s on a Karnaugh map. How many literals (a variable or its complement counts as one) are in the "
                "simplest sum-of-products expression for F?",
                "Simplify F to a minimal sum-of-products using a Karnaugh map. How many literals does it contain in total?",
            ),
        )
    stem = f"For three inputs A, B and C, {show}. {ask}"
    working = [
        f"Group the 1s into the largest possible groups; the cheapest cover has {terms} product term{'s' if terms != 1 else ''}.",
        f"Its terms use {literals} literals in total." if not ask_terms else f"So there are {terms} product terms.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _medium_differ(rng):
    for _ in attempts(500, "two expressions that differ on some rows"):
        t1, f1, _g, _h = _expression(rng)
        t2, f2, _g2, _h2 = _expression(rng)
        if t1 == t2:
            continue
        differ = sum(f1(*bits) != f2(*bits) for bits in _rows(3))
        if 1 <= differ <= 6:
            break
    agree = 8 - differ
    n1, n2 = len(_ones(f1, 3)), len(_ones(f2, 3))
    both = sum(f1(*bits) & f2(*bits) for bits in _rows(3))
    cands = [
        (agree, "bool-counts-agreeing-rows"),
        (abs(n1 - n2), "bool-compares-totals-not-rows"),
        (both, "bool-counts-agreeing-rows"),
        (differ + 1, "kmap-miscount-literals"),
        (differ - 1, "kmap-miscount-literals"),
    ]
    dist = num_dist(differ, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "Let G = {a} and H = {b}, both with inputs A, B and C. " + _ORDER_NOTE + " On how many of the 8 rows of "
            "the truth table do G and H give different outputs?",
            "Two circuits have outputs G = {a} and H = {b} for the same inputs A, B and C. " + _ORDER_NOTE + " For how "
            "many of the 8 input rows is G not equal to H?",
        ),
        a=t1,
        b=t2,
    )
    working = [
        f"G = 1 on rows {', '.join(map(str, _ones(f1, 3)))}; H = 1 on rows {', '.join(map(str, _ones(f2, 3)))}.",
        f"They differ on {differ} rows.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(differ), answer_kind="number")


def _medium_circuit(rng):
    circuit = _circuit(rng, 3, 3, ("AND", "OR", "NOT", "NAND", "NOR", "XOR"), want_special=True)
    return _circuit_count(rng, circuit, 3, "medium")


def _hard_kmap4(rng):
    ones, dcs, terms, literals = _kmap_function(rng, 4, 6, 10, min_terms=3)
    dist = _kmap_distractors(4, ones, dcs, terms, literals)
    enough(dist)
    show = _kmap_grid(4, ones, dcs) if rng.random() < 0.5 else _kmap_listing(4, ones, dcs)
    ask = phrase(
        rng,
        (
            "Groups may wrap round the edges of the map. How many literals are in the simplest sum-of-products expression for F?",
            "Simplify F to a minimal sum-of-products with a Karnaugh map, allowing groups that wrap round the edges. "
            "What is the total number of literals?",
        ),
    )
    stem = f"For four inputs A, B, C and D, {show}. {ask}"
    working = [
        "Mark the largest groups of 1s (sizes 8, 4 or 2); a group may wrap from one edge of the map to the opposite edge.",
        f"The cheapest cover uses {terms} product terms and {literals} literals in total.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(literals), answer_kind="number")


def _hard_dontcare(rng):
    n = rng.choice((3, 4))
    ones, dcs, terms, literals = _kmap_function(rng, n, 3 if n == 3 else 5, 4 if n == 3 else 7, with_dc=True)
    dist = _kmap_distractors(n, ones, dcs, terms, literals, with_dc=True)
    enough(dist)
    names = ", ".join(_VARS[:n])
    show = _kmap_grid(n, ones, dcs) if rng.random() < 0.5 else _kmap_listing(n, ones, dcs)
    stem = (
        f"For {'three' if n == 3 else 'four'} inputs {names}, {show}. Each don't-care (shown as d on a map) may be taken as 0 "
        "or 1, whichever gives the simpler expression. What is the fewest literals in a sum-of-products expression for F?"
    )
    working = [
        "Use the don't-care cells in a group wherever that makes a group larger, and leave out any that help no group.",
        f"The cheapest cover has {terms} terms and {literals} literals.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(literals), answer_kind="number")


def _hard_circuit(rng):
    circuit = _circuit(rng, 4, 4, ("AND", "OR", "NOT", "NAND", "NOR", "XOR"), want_special=True)
    return _circuit_count(rng, circuit, 4, "hard")


_EASY = (_easy_expression, _easy_function_number, _easy_circuit)
_MEDIUM = (_medium_kmap, _medium_differ, _medium_circuit)
_HARD = (_hard_kmap4, _hard_dontcare, _hard_circuit)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6a_boolean_logic(rng, tier="easy"):
    """Truth tables, gate circuits and Karnaugh-map simplification; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    check(pool is not None, f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------------


def _set_variants(circuit):
    """Candidate wrong row-number sets for the circuit's output."""
    ones = _ones(circuit.function(), 3)
    zeros = [r for r in range(8) if r not in ones]
    reversed_bits = sorted(int(f"{r:03b}"[::-1], 2) for r in ones)
    swap = {"NAND": "AND", "NOR": "OR", "XOR": "OR"}
    out = [(zeros, "bool-reads-zero-rows"), (reversed_bits, "bool-bit-order-reversed")]
    if circuit.uses(swap):
        out.append((_ones(circuit.function(swap), 3), "bool-gate-type-confused"))
    out.append(([r + 1 for r in ones], "bool-row-numbering-from-one"))
    return out


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w6a_boolean_logic(rng, tier="exam"):
    """Exam chain on a three-gate circuit: rows with output 1, their numbers, the minimal literals, the saving."""
    from ._w6a_kit import set_dist, set_fields

    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a circuit with a real simplification"):
        circuit = _circuit(rng, 3, 3, ("AND", "OR", "NOT", "NAND", "NOR", "XOR"), want_special=True)
        ones = _ones(circuit.function(), 3)
        terms, literals, larger = _cheapest(3, ones)
        if terms >= 2 and literals < 3 * len(ones) and (larger is None or larger >= literals):
            break
    n = len(ones)
    # Second route for the count: the Karnaugh cover must account for every row with a 1 and no other row.
    covered = 0
    primes = _primes(3, _mask(ones))
    for combo in _covers(primes, _mask(ones), terms):
        if sum(_literals_of(p) for p, _ in combo) == literals:
            for _, m in combo:
                covered |= m
            break
    check(covered == _mask(ones), "the cheapest cover is not exactly the ones")
    saved = 3 * n - literals
    check(saved > 0, "nothing saved")

    a_dist = _circuit_distractors(circuit, 3, n)
    b_dist = set_dist(ones, _set_variants(circuit), 3)
    c_dist = _kmap_distractors(3, ones, [], terms, literals)
    d_dist = num_dist(
        saved,
        [
            (3 * n - terms, "bool-saving-uses-terms"),
            (n - literals if n - literals > 0 else None, "bool-saving-uses-terms"),
            (literals, "kmap-terms-not-literals"),
            (saved + 1, "kmap-miscount-literals"),
            (saved - 1, "kmap-miscount-literals"),
        ],
        3,
    )
    for d in (a_dist, b_dist, c_dist, d_dist):
        enough(d)

    return dict(
        text=circuit.words(),
        parts=[
            part(
                "For how many of the 8 combinations of inputs is F equal to 1?",
                "number",
                num_fields(n),
                1,
                [f"Evaluating the gates on each of the 8 rows gives F = 1 on {n} of them."],
                a_dist,
                sensible_range=(0, 8),
            ),
            part(
                "List the row numbers on which F = 1, where the inputs A, B, C read as a three-digit binary number "
                "(A the leftmost digit) give the row number from 0 to 7.",
                "set",
                set_fields(ones),
                2,
                [f"F = 1 on rows {', '.join(map(str, ones))}."],
                b_dist,
            ),
            part(
                "Use a Karnaugh map to write F as a minimal sum-of-products. How many literals does it have in total?",
                "number",
                num_fields(literals),
                2,
                [f"The cheapest cover has {terms} product terms using {literals} literals."],
                c_dist,
                sensible_range=(0, None),
            ),
            part(
                "Writing F as a sum of full minterms uses three literals for every row with a 1. How many fewer "
                "literals does the minimal expression use?",
                "number",
                num_fields(saved),
                1,
                [f"The full form needs 3 × {n} = {3 * n} literals; the minimal form needs {literals}, so {saved} fewer."],
                d_dist,
                follow_through="3*part_a - part_c",
                sensible_range=(0, None),
            ),
        ],
    )
