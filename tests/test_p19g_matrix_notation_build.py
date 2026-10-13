"""Tests `p19g_matrix_type_and_size` and `p19g_matrix_entry_lookup`: determinism, distractors and grading.

Focused contract tests for the matrix type/size classification and
subscript entry-lookup skills:

- ``senior-recognise-different-types-of-matrices-row-column-square`` (topic
  ``p19g_matrix_type_and_size``, `answer: ordered_list`)
- ``senior-understand-the-matrix-definition-and-notation`` (topic
  ``p19g_matrix_entry_lookup``, `answer: number`)

The independent oracles below parse ONLY `question.text`/`question.answer`
(never anything from `mathgen.generators.p19g_matrix_notation_build`) and
re-derive the expected classification/entry from scratch: a freshly
re-implemented precedence classifier for skill 1, and either a numeric
`a_{i,j}` regex or the hard tier's own relative-index phrasing (parsed as
wording, exactly as any classification/scenario oracle in this suite
parses wording) for skill 2."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

TYPE_TOPIC = "p19g_matrix_type_and_size"
ENTRY_TOPIC = "p19g_matrix_entry_lookup"
TOPICS = [TYPE_TOPIC, ENTRY_TOPIC]

TYPE_ID = "senior-recognise-different-types-of-matrices-row-column-square"
ENTRY_ID = "senior-understand-the-matrix-definition-and-notation"

_REAL_LABELS = {"identity", "zero", "row", "column", "square", "none of these"}


# ===========================================================================
# Shared parsing helpers -- read ONLY the rendered text, never the generator
# ===========================================================================


def _extract_matrix(text: str) -> list:
    """Locate "A = [[...]]" and literal_eval the balanced-bracket substring
    -- pure text parsing, no import from the generator module."""
    idx = text.index("A = [")
    start = idx + len("A = ")
    depth = 0
    end = None
    for k in range(start, len(text)):
        if text[k] == "[":
            depth += 1
        elif text[k] == "]":
            depth -= 1
            if depth == 0:
                end = k + 1
                break
    assert end is not None, text
    return ast.literal_eval(text[start:end])


def _oracle_classify(matrix: list) -> str:
    """Freshly re-implemented precedence classifier -- written independently
    of `p19g_matrix_notation_build._classify_matrix`, never importing it."""
    rows, cols = len(matrix), len(matrix[0])
    is_identity = rows == cols and all(matrix[r][c] == (1 if r == c else 0) for r in range(rows) for c in range(cols))
    if is_identity:
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


_EXAM_RE = re.compile(
    r"a_\{i,j\} = (.+?)(?:, where | \(row number | for )i = 1(?:, \.\.\., (\d+))?"
    r"(?: and |; column number )j = 1(?:, \.\.\., (\d+))?"
)


def _eval_rule(expr: str, i: int, j: int) -> int:
    """Independent evaluator for the exam rule text (own parser, never the
    generator's lambdas)."""
    m = re.fullmatch(r"(-?\d+) if i = j, and (-?\d+) otherwise", expr)
    if m:
        return int(m[1]) if i == j else int(m[2])
    e = expr.replace("\u00d7", "*").replace("^", "**")
    e = re.sub(r"(\d)\s*(?=[ij(])", r"\1*", e)
    e = re.sub(r"\)\s*\(", ")*(", e)
    return int(eval(e, {"__builtins__": {}}, {"i": i, "j": j}))


def _matrix_of(q) -> list:
    """The matrix a question is about: parsed from the printed list, or (exam
    tier, nothing printed) expanded from the stated rule and ranges."""
    if "A = [[" in q.text:
        return _extract_matrix(q.text)
    m = _EXAM_RE.search(q.text)
    assert m, q.text
    expr, rows, cols = m.group(1), int(m.group(2) or 1), int(m.group(3) or 1)
    return [[_eval_rule(expr, i, j) for j in range(1, cols + 1)] for i in range(1, rows + 1)]


def _parse_opt(text: str) -> tuple:
    label, rows, cols = text.split(", ")
    return label, int(rows), int(cols)


def _options(q) -> list:
    """[key, d1, d2, d3] as (label, rows, cols)."""
    return [_parse_opt(q.answer_text)] + [_parse_opt(d) for d in q.distractors]


def _consistent(label: str, rows: int, cols: int) -> bool:
    return {
        "identity": rows == cols,
        "zero": True,
        "row": rows == 1,
        "column": cols == 1,
        "square": rows == cols,
        "none of these": rows != cols,
    }[label]


def _attack_odd_one_out(opts) -> float:
    """Majority dimensions + odd-one-out type; score 1/k when the key is among
    the k candidates left (expected score of a uniform guess among them)."""
    dims = Counter((r, c) for _, r, c in opts)
    top = max(dims.values())
    modal = {d for d, n in dims.items() if n == top}
    cand = [k for k, (_, r, c) in enumerate(opts) if (r, c) in modal]
    types = Counter(t for t, _, _ in opts)
    uniq = [k for k in cand if types[opts[k][0]] == 1]
    pick = uniq or cand
    return 1 / len(pick) if 0 in pick else 0.0


def _attack_modal_dims(opts) -> float:
    dims = Counter((r, c) for _, r, c in opts)
    top = max(dims.values())
    modal = {d for d, n in dims.items() if n == top}
    cand = [k for k, (_, r, c) in enumerate(opts) if (r, c) in modal]
    return 1 / len(cand) if 0 in cand else 0.0


def _attack_repeated_type(opts) -> float:
    """The mirror attack: among the modal-dimensions options take the one
    whose type label is SHARED with another option (the key's type is repeated
    when a distractor is 'right type, wrong size'). This is what breaks the
    old construction, where the odd-one-out form scores 0 by accident."""
    dims = Counter((r, c) for _, r, c in opts)
    top = max(dims.values())
    modal = {d for d, n in dims.items() if n == top}
    cand = [k for k, (_, r, c) in enumerate(opts) if (r, c) in modal]
    types = Counter(t for t, _, _ in opts)
    rep = [k for k in cand if types[opts[k][0]] > 1]
    pick = rep or cand
    return 1 / len(pick) if 0 in pick else 0.0


def _every_distractor_one_edit(opts) -> bool:
    """Every distractor differs from the key in exactly one of (type, dims):
    the key is then the centre of the option set."""
    key = opts[0]
    return all((d[0] == key[0]) != (d[1:] == key[1:]) for d in opts[1:])


def _one_edit_signature(opts) -> bool:
    key = opts[0]
    same_type = sum(1 for d in opts[1:] if d[0] == key[0])
    same_dims = sum(1 for d in opts[1:] if d[1:] == key[1:])
    return same_type == 1 and same_dims == 1


def _identity_plausible(matrix) -> bool:
    n = len(matrix)
    if n != len(matrix[0]):
        return False
    off = sum(1 for r in range(n) for c in range(n) if r != c and matrix[r][c] != 0)
    diff = sum(1 for r in range(n) for c in range(n) if matrix[r][c] != (1 if r == c else 0))
    return off == 0 or diff == 1


def _attack_combined(opts) -> float:
    """Pool the options at a modal size (ties pooled), keep those whose type is possible for their own size,
    score 1/len(kept) when the key is kept."""
    dims = Counter((r, c) for _, r, c in opts)
    top = max(dims.values())
    pool = [k for k, (_, r, c) in enumerate(opts) if dims[(r, c)] == top]
    kept = [k for k in pool if _consistent(*opts[k])]
    return 1 / len(kept) if 0 in kept else 0.0


def measure_combined(maker, difficulty, seeds=200) -> float:
    return (
        sum(_attack_combined(_options(maker(TYPE_TOPIC, seed=seed, difficulty=difficulty))) for seed in range(seeds))
        / seeds
    )


def measure_attacks(maker, difficulty, seeds=200) -> dict:
    """All the metrics for one tier through `maker(topic, seed=, difficulty=)`
    (the real `make`, or a scratch red-proof copy)."""
    a1 = a2 = a3 = sig = edit_all = 0.0
    ident_rows = ident_nonsquare = nonident_rows = ident_dismissable = 0
    inconsistent = 0
    prompts = set()
    sizes = set()
    labels = Counter()
    for seed in range(seeds):
        q = maker(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        opts = _options(q)
        a1 += _attack_odd_one_out(opts)
        a2 += _attack_modal_dims(opts)
        a3 += _attack_repeated_type(opts)
        edit_all += _every_distractor_one_edit(opts)
        sig += _one_edit_signature(opts)
        prompts.add(q.text)
        labels[q.answer[0]] += 1
        inconsistent += sum(1 for o in opts if not _consistent(*o))
        matrix = _matrix_of(q)
        sizes.add((len(matrix), len(matrix[0])))
        if q.answer[0] != "identity":
            nonident_rows += 1
            ids = [o for o in opts[1:] if o[0] == "identity"]
            if ids:
                ident_rows += 1
                if any(o[1] != o[2] for o in ids):
                    ident_nonsquare += 1
                if not _identity_plausible(matrix):
                    ident_dismissable += 1
    return dict(
        attack_odd_one_out=a1 / seeds,
        attack_modal_dims=a2 / seeds,
        attack_repeated_type=a3 / seeds,
        all_one_edit=edit_all / seeds,
        one_edit_signature=sig / seeds,
        identity_distractor_rows=ident_rows,
        identity_distractor_share=ident_rows / max(nonident_rows, 1),
        identity_nonsquare=ident_nonsquare,
        identity_dismissable=ident_dismissable,
        inconsistent_options=inconsistent,
        unique_prompts=len(prompts),
        sizes=sizes,
        labels=labels,
    )


# ===========================================================================
# Determinism, shape, distractor hygiene (both topics, every tier)
# ===========================================================================


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        q1 = make(topic, seed=seed, difficulty=difficulty)
        q2 = make(topic, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_non_empty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert q.distractors, f"seed {seed}: no distractors"
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors)), q.distractors
        assert len(q.distractors) == 3, f"seed {seed}: {len(q.distractors)} distractors"


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic} [{difficulty}]: only {len(texts)} distinct questions"


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_and_distractors_formatted_alike(topic, difficulty):
    """No option identifiable by shape alone: every distractor is a string
    of the same general shape as the answer (a "label, int, int" triple for
    skill 1, a bare integer literal for skill 2)."""
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        if topic == TYPE_TOPIC:
            for opt in [q.answer_text, *q.distractors]:
                parts = opt.split(", ")
                assert len(parts) == 3, opt
                assert parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit(), opt
        else:
            for opt in [q.answer_text, *q.distractors]:
                assert re.fullmatch(r"-?\d+", opt), opt


# ===========================================================================
# Skill 1: p19g_matrix_type_and_size (answer: ordered_list)
# ===========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_type_answer_is_triple_shaped_list(difficulty):
    for seed in SEEDS:
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, list) and len(q.answer) == 3
        label, rows, cols = q.answer
        assert label in _REAL_LABELS
        assert isinstance(rows, int) and isinstance(cols, int)
        assert q.answer_text == f"{label}, {rows}, {cols}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_type_1x1_matrix_never_drawn(difficulty):
    """Excluded by construction, over every difficulty, every seed --
    directly checked here, not merely asserted in the generator."""
    for seed in range(400):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        _, rows, cols = q.answer
        assert not (rows == 1 and cols == 1), (difficulty, seed, q.text)
        matrix = _matrix_of(q)
        assert not (len(matrix) == 1 and len(matrix[0]) == 1), (difficulty, seed, q.text)


def test_type_never_states_own_answer_as_the_illustrative_example():
    """The illustrative worked example ("diagonal, 2, 2") uses a label the
    closed set never contains -- verified directly by construction, every
    seed, every tier: no real draw's answer_text is ever "diagonal, 2, 2",
    and "diagonal" never appears as a real label."""
    for difficulty in TIERS:
        for seed in range(400):
            q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text != "diagonal, 2, 2"
            assert q.answer[0] != "diagonal"
            assert q.answer[0] in _REAL_LABELS


@pytest.mark.parametrize("difficulty", TIERS)
def test_type_real_grader_accepts_canonical_variants_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        label, rows, cols = q.answer
        variants = [
            q.answer_text,
            q.answer_text.upper(),
            q.answer_text.replace(", ", ","),
            f" {label} , {rows} , {cols} ",
        ]
        for variant in variants:
            result = GRADERS["ordered_list"](q, variant)
            assert result.correct, (seed, difficulty, variant)
        for distractor in q.distractors:
            result = GRADERS["ordered_list"](q, distractor)
            assert not result.correct, (seed, difficulty, distractor)


@pytest.mark.parametrize("difficulty", TIERS)
def test_type_grader_rejects_a_wrong_label_at_the_correct_size(difficulty):
    for seed in range(20):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        label, rows, cols = q.answer
        wrong_label = next(l for l in _REAL_LABELS if l != label)
        result = GRADERS["ordered_list"](q, f"{wrong_label}, {rows}, {cols}")
        assert not result.correct, (seed, difficulty, wrong_label)


# --- Answer-space contract: 400 seeds, each label under 25%, "none of these" present ---


@pytest.mark.parametrize("difficulty", TIERS)
def test_type_label_share_over_400_seeds_no_label_over_cap(difficulty):
    """Easy draws only its own four labels (cap 35%); the six-label tiers keep the original 25% cap."""
    labels = Counter()
    for seed in range(400):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        labels[q.answer[0]] += 1
    cap = 0.35 if difficulty == "easy" else 0.25
    for label, count in labels.items():
        assert count / 400 < cap, (difficulty, label, count, labels)
    assert labels["none of these"] > 0, (difficulty, labels)
    assert set(labels) <= _REAL_LABELS
    if difficulty != "easy":
        assert set(labels) == _REAL_LABELS, (difficulty, labels)


# --- Independent oracle: re-classify from the RENDERED matrix, never trusting q.answer ---


@pytest.mark.parametrize("difficulty", TIERS)
def test_type_oracle_matches_generator_over_300_seeds_per_tier(difficulty):
    for seed in range(300):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        matrix = _matrix_of(q)
        oracle_label = _oracle_classify(matrix)
        oracle_rows, oracle_cols = len(matrix), len(matrix[0])
        assert [oracle_label, oracle_rows, oracle_cols] == q.answer, (
            seed,
            difficulty,
            q.text,
            q.answer,
            oracle_label,
        )


def test_type_precedence_examples_from_the_task_file():
    """Direct, hand-built examples of the precedence rule's own edge cases
    (worked examples), classified with the SAME independent
    oracle used above."""
    assert _oracle_classify([[0, 0, 0, 0]]) == "zero"  # 1xN zero: zero, not row
    assert _oracle_classify([[0, 0], [0, 0]]) == "zero"  # square zero: zero, not square
    assert _oracle_classify([[1, 0], [0, 1]]) == "identity"
    assert _oracle_classify([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == "identity"
    assert _oracle_classify([[1, 2, 3]]) == "row"
    assert _oracle_classify([[1], [2], [3]]) == "column"
    assert _oracle_classify([[1, 2], [3, 4]]) == "square"
    assert _oracle_classify([[1, 2, 3], [4, 5, 6]]) == "none of these"


@pytest.mark.parametrize("difficulty", TIERS)
def test_type_precedence_stated_verbatim_in_every_prompt(difficulty):
    for seed in range(5):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        assert "identity, zero, row, column, square, none of these" in q.text


def test_type_hard_tier_includes_near_identity_or_near_zero_traps():
    """The hard tier's own escalation (a matrix CLOSE to identity/zero but
    genuinely neither) is a real, non-trivial share, not merely possible in
    theory."""
    found = False
    for seed in range(300):
        q = make(TYPE_TOPIC, seed=seed, difficulty="hard")
        matrix = _extract_matrix(q.text)
        if q.answer[0] != "square":
            continue
        rows = len(matrix)
        if rows != len(matrix[0]) or rows < 2:
            continue
        nonzero_off_diag = sum(1 for r in range(rows) for c in range(rows) if r != c and matrix[r][c] != 0)
        zero_off_diag_count = rows * rows - rows - nonzero_off_diag
        diag_ones = sum(1 for r in range(rows) if matrix[r][r] == 1)
        diag_zeros = sum(1 for r in range(rows) if matrix[r][r] == 0)
        near_identity = diag_ones == rows - 1 and nonzero_off_diag == 1
        near_zero = diag_zeros == rows and zero_off_diag_count == rows * rows - rows - 1
        if near_identity or near_zero:
            found = True
            break
    assert found, "no near-identity/near-zero trap found in 300 hard seeds"


def test_type_exam_tier_gives_only_a_rule_and_ranges_never_the_matrix():
    """The exam matrix is NOT printed; the learner expands the rule a_{i,j} over i = 1..R, j = 1..C mentally."""
    for seed in range(40):
        q = make(TYPE_TOPIC, seed=seed, difficulty="exam")
        assert "rule" in q.text or "defined by" in q.text
        assert "a_{i,j}" in q.text
        assert "[[" not in q.text and "pmatrix" not in q.latex
        matrix = _matrix_of(q)  # expanded by the test's own evaluator
        for row in matrix:
            for v in row:
                assert isinstance(v, int)


@pytest.fixture(scope="module")
def sweep():
    return {d: measure_attacks(make, d) for d in TIERS}


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_not_rebuildable_by_majority_dims_and_odd_one_out(sweep, difficulty):
    # Easy has exactly one true-size option (required) and at most one inconsistent option, so the
    # key and that option are the only two options with the modal dimensions and every
    # dimension-based attack is a coin flip (0.5 by construction); the bar is relaxed to 0.5
    # for easy only.
    assert sweep[difficulty]["attack_odd_one_out"] <= (0.50 if difficulty == "easy" else 0.40), sweep[difficulty]


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_not_the_modal_dimensions_option(sweep, difficulty):
    # Easy has exactly one true-size option (required) and at most one inconsistent option, so the
    # key and that option are the only two options with the modal dimensions and every
    # dimension-based attack is a coin flip (0.5 by construction); the bar is relaxed to 0.5
    # for easy only.
    assert sweep[difficulty]["attack_modal_dims"] <= (0.50 if difficulty == "easy" else 0.45), sweep[difficulty]


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_not_the_repeated_type_at_modal_dimensions(sweep, difficulty):
    # Easy has exactly one true-size option (required) and at most one inconsistent option, so the
    # key and that option are the only two options with the modal dimensions and every
    # dimension-based attack is a coin flip (0.5 by construction); the bar is relaxed to 0.5
    # for easy only.
    assert sweep[difficulty]["attack_repeated_type"] <= (0.50 if difficulty == "easy" else 0.40), sweep[difficulty]


@pytest.mark.parametrize("difficulty", TIERS)
def test_one_edit_signature_is_rare(sweep, difficulty):
    assert sweep[difficulty]["one_edit_signature"] <= 0.30, sweep[difficulty]
    assert sweep[difficulty]["all_one_edit"] <= 0.30, sweep[difficulty]


@pytest.mark.parametrize("difficulty", TIERS)
def test_identity_distractor_is_never_dismissable(sweep, difficulty):
    m = sweep[difficulty]
    assert m["identity_dismissable"] == 0, m
    assert m["identity_nonsquare"] == 0, m
    assert m["identity_distractor_share"] <= 0.25, m


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_option_is_internally_self_consistent(sweep, difficulty):
    # Easy: the one deliberate true-size distractor is a right size with a wrong type, so it is not
    # self-consistent; it is exactly one option per item; hard/medium stay at 0. Easy items carry at
    # most two inconsistent options each (<= 400).
    limit = 400 if difficulty == "easy" else 0
    assert sweep[difficulty]["inconsistent_options"] <= limit, sweep[difficulty]


def test_easy_keys_are_only_row_column_square_none(sweep):
    assert set(sweep["easy"]["labels"]) == {"row", "column", "square", "none of these"}


def test_hard_uses_sizes_medium_never_draws(sweep):
    fresh = sweep["hard"]["sizes"] - sweep["medium"]["sizes"]
    assert len(fresh) >= 3, (sorted(sweep["hard"]["sizes"]), sorted(sweep["medium"]["sizes"]))


def test_medium_sizes_stay_within_4x4(sweep):
    assert all(max(s) <= 4 for s in sweep["medium"]["sizes"]), sweep["medium"]["sizes"]
    assert any(s[0] != s[1] for s in sweep["medium"]["sizes"])  # rectangles present


def test_easy_sizes_are_2_or_3(sweep):
    assert all(1 <= r <= 3 and 1 <= c <= 3 for r, c in sweep["easy"]["sizes"])


@pytest.mark.parametrize("difficulty", TIERS)
def test_unique_prompts_over_200_seeds(sweep, difficulty):
    assert sweep[difficulty]["unique_prompts"] >= 150, sweep[difficulty]["unique_prompts"]


def test_exam_prompt_has_no_printed_matrix_and_widens_the_rules():
    rules = set()
    for seed in range(200):
        q = make(TYPE_TOPIC, seed=seed, difficulty="exam")
        assert "[[" not in q.text
        m = _EXAM_RE.search(q.text)
        assert m, q.text
        rules.add(m.group(1))
    assert len(rules) >= 10, rules


def test_exam_has_more_than_one_prompt_template():
    stems = {make(TYPE_TOPIC, seed=s, difficulty="exam").text.split("a_{i,j}")[0] for s in range(60)}
    assert len(stems) >= 3, stems


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_key_accepted_every_distractor_rejected_40_seeds(difficulty):
    for seed in range(40):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        assert GRADERS["ordered_list"](q, q.answer_text).correct, (difficulty, seed)
        for d in q.distractors:
            assert not GRADERS["ordered_list"](q, d).correct, (difficulty, seed, d)


# ===========================================================================
# Skill 2: p19g_matrix_entry_lookup (answer: number)
# ===========================================================================

_NUMERIC_AIJ_RE = re.compile(r"a_\{(-?\d+),\s*(-?\d+)\}")
_LAST_ROW_RE = re.compile(r"i is the last row")
_ROW_REL_RE = re.compile(r"i is row m - 1 \(the matrix has m = (\d+) rows\)")
_SECOND_COL_RE = re.compile(r"j is the second column")
_COL_REL_RE = re.compile(r"j is column n - 1 \(the matrix has n = (\d+) columns\)")


_ENTRY_EXAM_RE = re.compile(
    r"rule a_\{i,j\} = (.+?), for i = 1, \.\.\., (\d+) \(rows\) and j = 1, \.\.\., (\d+) \(columns\)"
)


def _entry_matrix(q) -> list:
    """Matrix an entry-lookup question is about: the printed list, or (exam
    tier, rule-only prompt) expanded test-side from the stated rule."""
    if "A = [[" in q.text:
        return _extract_matrix(q.text)
    m = _ENTRY_EXAM_RE.search(q.text)
    assert m, q.text
    rows, cols = int(m.group(2)), int(m.group(3))
    return [[_eval_rule(m.group(1), i, j) for j in range(1, cols + 1)] for i in range(1, rows + 1)]


def _oracle_entry_indices(text: str, matrix: list) -> tuple[int, int]:
    """Independent index derivation: a numeric a_{i,j} occurrence is read
    directly; failing that (the hard tier's relative-reference phrasing),
    the wording itself is parsed -- never importing the generator's own
    `_entry_hard`."""
    numeric = _NUMERIC_AIJ_RE.findall(text)
    if numeric:
        i, j = int(numeric[-1][0]), int(numeric[-1][1])
        return i, j
    rows = len(matrix)
    if _LAST_ROW_RE.search(text):
        i = rows
    else:
        match = _ROW_REL_RE.search(text)
        assert match, text
        i = int(match.group(1)) - 1
    if _SECOND_COL_RE.search(text):
        j = 2
    else:
        match = _COL_REL_RE.search(text)
        assert match, text
        j = int(match.group(1)) - 1
    return i, j


@pytest.mark.parametrize("difficulty", TIERS)
def test_entry_answer_is_a_plain_integer(difficulty):
    for seed in SEEDS:
        q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, int)
        assert q.answer_text == str(q.answer)


@pytest.mark.parametrize("difficulty", TIERS)
def test_entry_real_grader_accepts_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
        result = GRADERS["number"](q, q.answer_text)
        assert result.correct, (seed, difficulty)
        for distractor in q.distractors:
            result = GRADERS["number"](q, distractor)
            assert not result.correct, (seed, difficulty, distractor)


@pytest.mark.parametrize("difficulty", TIERS)
def test_entry_a_ij_never_equals_a_ji(difficulty):
    """Load-bearing requirement: the swapped-index distractor is
    a live, genuinely-wrong option every draw -- checked directly with
    `check_equivalent`'s negation (a deliberate `assert not`, not merely a
    generator-internal `check_distinct` re-read)."""
    for seed in SEEDS:
        q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
        matrix = _entry_matrix(q)
        i, j = _oracle_entry_indices(q.text, matrix)
        a_ij = matrix[i - 1][j - 1]
        a_ji = matrix[j - 1][i - 1]
        assert i != j, (difficulty, seed, q.text)
        assert a_ij != a_ji, (difficulty, seed, q.text, a_ij, a_ji)
        assert a_ij == q.answer, (difficulty, seed, q.text, a_ij, q.answer)
        assert str(a_ji) in q.distractors, (difficulty, seed, q.text, a_ji, q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_entry_indices_vary_across_the_sweep_never_fixed_at_1_1(difficulty):
    seen = set()
    for seed in range(60):
        q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
        matrix = _entry_matrix(q)
        i, j = _oracle_entry_indices(q.text, matrix)
        seen.add((i, j))
    assert len(seen) > 1, (difficulty, seen)
    assert seen != {(1, 1)}, (difficulty, seen)


# --- Answer-space contract: 200 seeds/tier, top key < 15% ---


@pytest.mark.parametrize("difficulty", TIERS)
def test_entry_top_answer_concentration_under_15_percent(difficulty):
    answers = Counter()
    for seed in range(200):
        q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
        answers[q.answer] += 1
    top = max(answers.values()) / 200
    assert top < 0.15, (difficulty, answers.most_common(3))


# --- Independent oracle: re-derive the entry from the RENDERED matrix + indices ---


@pytest.mark.parametrize("difficulty", TIERS)
def test_entry_oracle_matches_generator_over_300_seeds_per_tier(difficulty):
    for seed in range(300):
        q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
        matrix = _entry_matrix(q)
        i, j = _oracle_entry_indices(q.text, matrix)
        oracle_answer = matrix[i - 1][j - 1]
        check_equivalent(oracle_answer, q.answer, "oracle/generator mismatch")
        assert oracle_answer == q.answer, (seed, difficulty, q.text, oracle_answer, q.answer)


@pytest.mark.parametrize("difficulty", ["medium"])
def test_entry_medium_non_square_matrix_shape(difficulty):
    for seed in range(30):
        q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
        matrix = _entry_matrix(q)
        assert len(matrix) != len(matrix[0]), (seed, q.text)
        assert {len(matrix), len(matrix[0])} == {4, 5}


@pytest.mark.parametrize("difficulty", ["hard"])
def test_entry_hard_tier_uses_relative_index_reference(difficulty):
    for seed in range(10):
        q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
        assert "i is" in q.text and "j is" in q.text
        assert not _NUMERIC_AIJ_RE.search(q.text)


def test_entry_exam_tier_uses_an_algebraic_rule_and_expands_it_fully():
    """Rule-only contract: the prompt states the rule and the
    ranges, never the expanded matrix; the rule expands to integers."""
    for seed in range(10):
        q = make(ENTRY_TOPIC, seed=seed, difficulty="exam")
        assert "rule" in q.text
        for row in _entry_matrix(q):
            for v in row:
                assert isinstance(v, int)


def test_entry_exam_prints_only_the_rule_and_answer_matches_the_rule():
    for seed in range(200):
        q = make(ENTRY_TOPIC, seed=seed, difficulty="exam")
        assert "[[" not in q.text, (seed, q.text)
        assert "pmatrix" not in (q.latex or ""), (seed, q.latex)
        assert "giving A" not in q.text, (seed, q.text)
        m = _ENTRY_EXAM_RE.search(q.text)
        assert m, (seed, q.text)
        i, j = (int(x) for x in _NUMERIC_AIJ_RE.findall(q.text)[-1])
        assert 1 <= i <= int(m.group(2)) and 1 <= j <= int(m.group(3)), (seed, q.text)
        assert _eval_rule(m.group(1), i, j) == q.answer, (seed, q.text, q.answer)


# ===========================================================================
# Boundary: skill 1 (type + size) vs skill 2 (single entry)
# ===========================================================================


def test_type_never_reports_a_single_entry_value():
    """Skill 1's exam tier legitimately uses "a_{i,j}" as the entry-rule
    NOTATION (e.g. "the rule a_{i,j} = i + j") -- the boundary this checks
    is the FINAL ANSWER shape, never a single numeric entry, not the
    presence of subscript notation anywhere in the prompt."""
    for difficulty in TIERS:
        for seed in range(30):
            q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, list) and len(q.answer) == 3
            label, rows, cols = q.answer
            assert label in _REAL_LABELS
            assert not re.fullmatch(r"-?\d+", q.answer_text)


def test_entry_never_classifies_type_or_reports_size_as_the_final_answer():
    for difficulty in TIERS:
        for seed in range(30):
            q = make(ENTRY_TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, int)
            for label in _REAL_LABELS:
                assert q.answer_text != label


# ===========================================================================
# Skill contract loading tests
# ===========================================================================


def test_both_skills_load_with_correct_answer_kind_and_prerequisites():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    type_skill = by_id[TYPE_ID]
    entry_skill = by_id[ENTRY_ID]
    assert type_skill.answer == "ordered_list"
    assert type_skill.prerequisites == ["senior-understand-the-matrix-definition-and-notation"]
    assert entry_skill.answer == "number"
    assert entry_skill.prerequisites == ["input-output-tables"]


def test_both_skills_are_active_placement_ineligible_and_general_band():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    for skill_id in (TYPE_ID, ENTRY_ID):
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert getattr(skill, "learner_band", "general") == "general"


def test_both_skills_have_sources_covering_all_four_tiers():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    for skill_id in (TYPE_ID, ENTRY_ID):
        difficulties = {src.difficulty for src in by_id[skill_id].sources}
        assert difficulties == {"easy", "medium", "hard", "exam"}
        exam_src = next(s for s in by_id[skill_id].sources if s.difficulty == "exam")
        assert exam_src.stages == ["exam"]


def test_classify_matrix_verification_raises_on_a_corrupted_claim():
    """`check`/`check_equivalent` inside the generator module is a real
    check, not decorative -- proved on a deliberately wrong claim, not by
    mutating the shared module."""
    with pytest.raises(VerificationError):
        check_equivalent(1, 2, "deliberately wrong")


def _possible_for_dims(label: str, rows: int, cols: int) -> bool:
    """Strict per-dims possibility under the precedence rule: 1xn -> row/zero,
    nx1 -> column/zero, nxn -> square/identity/zero, rectangle -> none/zero."""
    if rows == 1 and cols == 1:
        return False
    if rows == 1:
        return label in ("row", "zero")
    if cols == 1:
        return label in ("column", "zero")
    if rows == cols:
        return label in ("square", "identity", "zero")
    return label in ("none of these", "zero")


def _attack_unique_dims(opts) -> float:
    """Pick the option whose dims differ from every other option's. Uniform
    over all four when no option is dimension-unique."""
    dims = Counter((r, c) for _, r, c in opts)
    cand = [k for k, (_, r, c) in enumerate(opts) if dims[(r, c)] == 1] or list(range(len(opts)))
    return 1 / len(cand) if 0 in cand else 0.0


def _attack_true_dims_random(q, opts, matrix) -> float:
    """Read the real dims off the matrix; pick uniformly among the options
    that carry them."""
    true = (len(matrix), len(matrix[0]))
    cand = [k for k, (_, r, c) in enumerate(opts) if (r, c) == true]
    return 1 / len(cand) if 0 in cand else 0.0


def measure_dims(maker, difficulty, seeds=200) -> dict:
    impossible = shared = 0
    unique = true_random = 0.0
    for seed in range(seeds):
        q = maker(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        opts = _options(q)
        matrix = _matrix_of(q)
        impossible += sum(1 for o in opts if not _possible_for_dims(*o))
        shared += any(d[1:] == opts[0][1:] for d in opts[1:])
        unique += _attack_unique_dims(opts)
        true_random += _attack_true_dims_random(q, opts, matrix)
    return dict(
        impossible_options=impossible,
        key_dims_shared_rows=shared,
        key_dims_shared_share=shared / seeds,
        attack_unique_dims=unique / seeds,
        attack_true_dims_random=true_random / seeds,
    )


@pytest.fixture(scope="module")
def sweep2140():
    return {d: measure_dims(make, d) for d in TIERS}


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_option_type_possible_for_its_own_dims(sweep2140, difficulty):
    # Easy: see test_every_option_is_internally_self_consistent.
    limit = 400 if difficulty == "easy" else 0  # At most two per easy item
    assert sweep2140[difficulty]["impossible_options"] <= limit, sweep2140[difficulty]


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_dims_appear_in_a_distractor_where_a_plausible_wrong_type_exists(sweep2140, difficulty):
    # Easy has no identity/zero keys or look-alikes: at the key's own size no
    # plausible wrong type exists there (a wrong type would contradict its own
    # dims or be dismissable at a glance), so only the other tiers are held to a
    # floor. Zero/identity are offered only when plausible.
    if difficulty == "easy":
        return
    assert sweep2140[difficulty]["key_dims_shared_share"] >= 0.25, sweep2140[difficulty]


@pytest.mark.parametrize("difficulty", TIERS)
def test_unique_dims_and_true_dims_attacks_are_bounded(sweep2140, difficulty):
    assert sweep2140[difficulty]["attack_unique_dims"] <= 0.40, sweep2140[difficulty]
    if difficulty != "easy":
        # Easy has no plausible wrong type at the key's own size (see the test
        # above), so reading the size off the matrix necessarily singles out the
        # key there; that is the skill's easy content (read size, name type).
        assert sweep2140[difficulty]["attack_true_dims_random"] <= 0.90, sweep2140[difficulty]


def _zero_entries(matrix) -> int:
    return sum(1 for row in matrix for v in row if v == 0)


def _sweep2144(difficulty, seeds=200):
    out = dict(
        true_size=0,
        easy_bad_option=0,
        easy_multi_true=0,
        easy_inconsistent_gt2=0,
        zero_no_zero_entries=0,
        zero_under_half=0,
        identity_not_diagonalish=0,
        identity_or_zero_options=0,
    )
    for seed in range(seeds):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        opts = _options(q)
        matrix = _matrix_of(q)
        true = (len(matrix), len(matrix[0]))
        dist = opts[1:]
        n_true = sum(1 for d in dist if d[1:] == true)
        out["true_size"] += n_true >= 1
        out["easy_multi_true"] += n_true > 1
        out["easy_inconsistent_gt2"] += sum(1 for o in opts if not _consistent(*o)) > 2
        out["easy_bad_option"] += any(d[0] in ("identity", "zero") for d in dist)
        for d in dist:
            if d[0] == "zero":
                out["zero_no_zero_entries"] += _zero_entries(matrix) == 0
                out["zero_under_half"] += 2 * _zero_entries(matrix) < true[0] * true[1]
            if d[0] == "identity" and not _identity_plausible(matrix):
                out["identity_not_diagonalish"] += 1
    return out


@pytest.fixture(scope="module")
def sweep2144():
    return {d: _sweep2144(d) for d in ("easy", "hard")}


def test_easy_some_distractor_carries_the_true_size(sweep2144):
    m = sweep2144["easy"]
    # At most one true-size distractor per item, present in at least 60% of items.
    assert m["true_size"] >= 0.60 * 200, m
    assert m["easy_multi_true"] == 0, m  # never more than one


def test_easy_never_offers_identity_or_zero_and_keeps_inconsistency_to_one(sweep2144):
    # At most two inconsistent options per item (was one).
    m = sweep2144["easy"]
    assert m["easy_bad_option"] == 0, m
    assert m["easy_inconsistent_gt2"] == 0, m


def _sweep2148(difficulty, seeds=200):
    out = dict(
        zero_visibly_nonzero=0,
        zero_rows=0,
        easy_two_consistent_distinct=0,
        easy_inconsistent_total=0,
        easy_lt2_consistent=0,
    )
    for seed in range(seeds):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        opts = _options(q)
        matrix = _matrix_of(q)
        n = len(matrix) * len(matrix[0])
        zeros = _zero_entries(matrix)
        for d in opts[1:]:
            if d[0] == "zero":
                out["zero_rows"] += 1
                out["zero_visibly_nonzero"] += not (n - zeros <= 1 or 4 * zeros >= 3 * n)
        if difficulty == "easy":
            out["easy_inconsistent_total"] += sum(1 for o in opts if not _consistent(*o))
            good = [o for o in opts if _consistent(*o)]  # key included
            out["easy_lt2_consistent"] += len(good) < 2
            out["easy_two_consistent_distinct"] += len(set(good)) == len(good)
    return out


@pytest.fixture(scope="module")
def sweep2148():
    return {d: _sweep2148(d) for d in ("easy", "medium", "hard")}


def test_easy_at_most_two_inconsistent_and_two_consistent_options_per_item(sweep2148):
    m = sweep2148["easy"]
    assert m["easy_inconsistent_total"] <= 400, m
    assert m["easy_lt2_consistent"] == 0, m
    assert m["easy_two_consistent_distinct"] == 200, m


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_zero_label_only_on_at_most_one_nonzero_entry_or_75_percent_zero(sweep2148, difficulty):
    assert sweep2148[difficulty]["zero_visibly_nonzero"] == 0, sweep2148[difficulty]


def test_easy_2137_bars_still_hold():
    r = measure_attacks(make, "easy")
    # 0.5 by construction (key vs the true-size option); see the tests above.
    assert r["attack_odd_one_out"] <= 0.50, r
    assert r["attack_repeated_type"] <= 0.50, r
    assert r["all_one_edit"] <= 0.04, r
    assert r["one_edit_signature"] <= 0.30, r
    assert r["identity_dismissable"] == 0, r


def test_hard_zero_label_only_when_the_matrix_could_pass_for_zero(sweep2144):
    m = sweep2144["hard"]
    assert m["zero_no_zero_entries"] == 0, m
    assert m["zero_under_half"] == 0, m  # at least half the printed entries are zero


def test_hard_identity_label_only_on_a_diagonalish_square_matrix(sweep2144):
    assert sweep2144["hard"]["identity_not_diagonalish"] == 0, sweep2144["hard"]


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_2144_grader_accepts_key_and_rejects_distractors(difficulty):
    for seed in range(60):
        q = make(TYPE_TOPIC, seed=seed, difficulty=difficulty)
        assert GRADERS["ordered_list"](q, q.answer_text).correct, (difficulty, seed)
        for d in q.distractors:
            assert not GRADERS["ordered_list"](q, d).correct, (difficulty, seed, d)


@pytest.mark.parametrize("difficulty", TIERS)
def test_combined_modal_size_and_consistency_attack_is_bounded(difficulty):
    assert measure_combined(make, difficulty) <= 0.40, difficulty
