# Interactive graph answers

Status: built (task w9). The engine owns the **reply schema** and the **verification**. A front end builds the widget
later; nothing here draws one.

## How it works

1. The question's `figure` is a `cartesian` or `number_line` spec with an optional **`canvas`** key (built with
   `mathgen.figure_kinds.canvas.canvas`). The figure draws what is *given*; the canvas names the answer kind the widget
   sends back (`reply`), the enabled `tools`, the `snap` step, optional `vertex_labels` and `max_points`. The figure
   never holds the answer.
2. The widget sends the reply as a **string** (`engine.grade(skill, question, reply)`): canonical JSON, or the typed
   form below. The same string is what a text-only platform asks the learner to type.
3. Every grader reads JSON with `json.loads` only (floats kept as text, `NaN`/`Infinity` refused, 8,000 characters and
   60 items at most, `RecursionError` caught). Nothing is evaluated. Anything unreadable, off the canvas, off the snap
   lattice or too big is **wrong**, never an error.
4. Numbers are exact: an integer, a decimal (`0.5`) or a string (`"1/3"`), compared as `Fraction`. There is **no
   tolerance**; the snap grid is the tolerance. The widget must snap to `canvas.snap` and send exact values (a third
   step sends `"1/3"`, not `0.333`). `Question.answer` carries the same `x_range`/`y_range` (or `range`) and `snap`, and a
   reply outside the range or off the lattice is wrong. A generator test checks the figure and key agree.
5. In a multi-part exam question the canvas applies to the part whose `answer_kind` equals `canvas.reply`.

Coordinates are `[x, y]` in graph units. `answer_text` is always a valid typed reply for its kind.

## Catalogue

| Interaction | Kind | Status |
|---|---|---|
| Plot one or several points; move a point | `points` | existing |
| Line through two points; by gradient and intercept | `line` | existing |
| Polyline / piecewise path | `polyline` | existing |
| Colour grid cells | `grid_cells` | existing |
| Shade a half-plane or system | `half_planes` | new |
| Number-line circles and rays | `number_line_set` | new |
| Polygon; translate, reflect, rotate by vertices | `polygon` | new |
| Parabola by vertex and a point; exponential by points | `curve` | new |
| Histogram or bar heights | `bars` | new |
| Network edges | `edges` | new |
| Sinusoid by key points | `points` (key points), no curve kind | partial |

### Existing kinds

- **`points`** reply: typed points `(1, 2) (3, -4)`; answer `{"points": [...], "ordered": bool}` or a list. Move a point
  is plot the final position. Tool `plot_points`.
- **`line`** reply: any two distinct points on the line, `(0, 1) (2, 5)`. Verification: both satisfy the key's line, so
  gradient-and-intercept is sent as two points (`(0, c)` and `(1, m + c)`). Tool `line_two_points`.
- **`polyline`, `grid_cells`**: unchanged; `polyline` takes the tool `polyline`.

### `half_planes` (tool `half_plane`)

Reply: `{"half_planes": [{"through": [[0,1],[1,3]], "boundary": "solid", "shade": [0,5]}]}`. `through` is any two
distinct points on the boundary (snapped); `boundary` is `solid` (inequality includes the line) or `dashed`; `shade` is a
point strictly on the shaded side (on the line is wrong).
Rule: each entry becomes `ax + by + c` scaled so the first non-zero of `a`, `b` has absolute value 1, signed so the
shaded side is positive, with the boundary flag. The reply set must equal the key set: any two points on a line and any
shade point match; a missing, extra or flipped half-plane, or the wrong boundary, is wrong.
Figure: plane with `x_range`, `y_range`, a grid, the given lines if any; canvas `snap` 1, tool `half_plane`.
Typed: `y > 2x - 1`, several joined by `,` `;` `and` or `∧`.

### `number_line_set` (tool `number_line_marks`)

Reply: `{"pieces": [{"from": 2, "to": null, "from_closed": false, "to_closed": false}]}` (`null` is unbounded, a ray), or
`{"marks": [3, 5]}` for plotted points. Rule: pieces are merged and sorted, then compared as a set; open versus closed
matters; a single point is a piece with both ends closed. Figure: `number_line` with `range` and `snap`. Typed: `x > 2`,
`-1 <= x < 4`, `x < 1 or x >= 5`, `x = 7`.

### `polygon` (tool `polygon_points`)

Reply: `{"vertices": [[1,1],[4,1],[4,3]]}`, a bare list, or typed points. Labelled (transformations: the image of A, B, C
is A', B', C'): vertex *i* of the reply must be exactly vertex *i* of the key. Unlabelled: duplicates and collinear
points are removed, then the outline compares as a cycle in either direction from any start. Figure: plane plus the
given shape; canvas `vertex_labels`, `max_points`. Typed: `(1, 1) (4, 1) (4, 3)`, optionally in `[...]` or `{...}`.

### `curve` (tools `parabola_vertex_point`, `curve_points`)

Key: `family` `quadratic` (`a`, `h`, `k`: `y = a(x-h)² + k`) or `exponential` (`a`, `b`, `c`: `y = a·bˣ + c`).
Reply: `{"vertex": [h, k], "point": [x, y]}` (quadratic), or `{"points": [...]}`: at least 3 distinct x for a quadratic, 2
for an exponential (integer `|x| <= 60`), every point exactly on the key curve and in range. Any one correct
vertex-and-point pair, or any set of points on the curve, is accepted; a vertex alone or an off-curve point is wrong.
Typed: `vertex (2, -3) point (3, -1)` or a list of points. Sinusoids are not a kind: give key points to `points`.

### `bars` (planned canvas tool; kind built)

Reply: `{"heights": [3, 5, 2]}`, `{"bars": {"Mon": 3}}` or a bare list; heights are non-negative, at most `max`, on the
snap lattice, in category order. Typed: `3, 5, 2` or `Mon = 3, Tue = 5`. Canvas support on `bar_chart` figures is not
registered yet.

### `edges` (planned canvas tool; kind built)

Reply: `{"edges": [["A","B"],["B","C",4]]}` (weight third when weighted). Undirected edges are normalised, duplicates
collapse, a self-loop is wrong, and the edge set must equal the key. Typed: `A-B, B-C`, `A->B`, `A-B:4`. Canvas support
on `diagram` figures is not registered yet.

## Skills built on it

`plot-a-point-from-coordinates` (`points`), `line-draw-from-gradient-and-point` (`line`), `inequality-on-number-line`
(`number_line_set`), `shade-linear-inequality-region` (`half_planes`), `transform-a-shape-on-grid` (`polygon`),
`parabola-from-vertex-and-point` (`curve`). Each has easy, medium, hard and exam tiers; every distractor is checked
against the real grader at generation time.

## Still unreachable (the "sketch" scope call)

- **Free-hand sketches** (a curve drawn with the pen): no exact key exists; needs a tolerance or a rubric. Not built.
- **Sketching a function from its equation** with intercepts, turning points and asymptotes labelled: reachable only
  as separate key-point `points` parts or a `curve` quadratic/exponential; cubics, reciprocals, logarithms and
  trigonometric graphs have no `curve` family.
- **Sinusoids** by amplitude, period and phase: key points via `points` only.
- **Rubric-graded sketches and proofs**: need an open-response answer kind. Not mine.
- **Statistical plots** (box plots, scatter lines of best fit): only `bars` for histograms is built.
- **3D and vector diagrams**: no kind.
