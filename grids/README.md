# Coverage grids

A skill list says *which topics* the library has. An exam question is a topic crossed with a function family, an
operation and an exactness mode, and the topic-level list is far coarser than that. A grid makes the finer
structure explicit, so "what are we missing?" becomes a query instead of an audit of one past paper.

## The idea

A **cell** is `family/operation/mode`, for example `logarithmic/implied_domain/hand`.

- **family**: `polynomial`, `power`, `rational`, `exponential`, `logarithmic`, `circular`, `hybrid`, `composite`,
  `inverse`, `pdf`, `discrete_pmf` (the same closed list the recipe subjects use).
- **operation**: what is asked (domain, solve, differentiate, stationary point, definite integral, probability ...).
  Each grid lists its own operations in `grid.yaml`.
- **mode**: `hand` (an exact answer, no technology: Exam 1 style) or `calc` (technology expected, decimal answers).

A cell is **possible** when a row of the study design makes it possible, **covered** when a skill tier or a recipe
produces it, and a **gap** when it is possible and not covered. `scripts/grid_gap_report.py` lists the gaps.

## Layout

```
grids/<id>/grid.yaml          closed lists, ask_operations, and the study-design rows
grids/<id>/evidence/*.yaml    which skill tier produces which cells
grids/<id>/papers/*.yaml      past papers, mapped to cells: the held-out test
```

**Rows** are the key skills of the study design, paraphrased: this repository carries short labels and a link,
not the design's text. Each row has `cells` (expanded as a cross product of `families x operations x modes`), or
`auto_markable: false` with a reason (sketches, proofs and written reasons have no deterministic grader). A row is
`provisional` until it has been checked line by line against the study design; then it becomes `verified`.

**Evidence** entries say `by: skill:<id>`, a `tier`, and the cells. A claim is checked: the skill must exist, be
active and have that tier, must not be `technology: required` when it claims `hand`, and must claim only cells some
row makes possible. Make a claim only after rendering that tier (`scripts/render_skill_questions.py`) and reading
the prompts and answers: a false claim hides a gap. **Recipes need no evidence entry**: every recipe under
`recipes/` counts for its subject's families x the operation each of its asks maps to (`ask_operations` in
`grid.yaml`) in the recipe's mode (`technology_active: true` is `calc`, otherwise `hand`). An ask with no
`ask_operations` entry is reported as an error, so adding an ask forces the decision of what it counts as.

**Papers** are the held-out test, not a source of the grid. Each part is mapped to cells, or marked `not_auto`.
Verdicts: `covered`; `gap` (cells possible, not yet produced); `grid-miss` (a cell the part needs is not possible,
so the grid is missing something and should be fixed first); `not-auto`. A new paper that produces a `grid-miss`
is the signal that the study-design rows are incomplete.

## Shapes: the finer dimension

Family, operation and mode are still coarse. A skill that asks for the domain of `ln(2x+1)` covers
`logarithmic/implied_domain/hand`, yet an exam asks for `ln(3+2x-x^2)`, which needs a quadratic inequality. A cell can
therefore carry **shapes** from the grid's closed `shapes` list (`quadratic_argument`, `quadratic_in_exponential`,
`unknown_constants_2`, `binomial_count` ...; names follow the recipe vocabulary where one exists). Rows say which shapes
are possible for a cell, evidence says which shapes its questions really have, and a paper part says which it needs.
A part whose cells are covered but whose shapes are not is a `shape-gap`, and the report's "Shape gaps" section lists
every covered cell still missing a shape. Evidence with no `shapes` proves only the plain cell.

## Commands

```
python scripts/grid_gap_report.py                     # text report for every grid
python scripts/grid_gap_report.py --format markdown --out /tmp/gaps.md
python scripts/grid_gap_report.py --check             # exit 1 if any claim fails verification
python scripts/grid_gap_report.py --format json       # for tooling
```

`tests/test_grid_gap_report.py` runs `--check` logic on the shipped grids, so a skill that is renamed, deactivated or
loses a tier fails the test that cites it.

## Extending

New family or operation: add it to the grid's closed list in the same PR as the row that needs it. New course
(Specialist, HSC, QCE): a new directory, reusing the same families and adding operations. 100% of a grid means every
cell the study design makes possible is produced; it does not mean every question VCAA could write.
