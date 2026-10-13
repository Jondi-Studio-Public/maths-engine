# Figure specs

A question can carry a figure: a graph, a clock, a chart, a diagram. The engine describes it as a JSON object called a figure spec. This guide is for developers who show those figures in their own platform.

A spec holds every number and label that is drawn. You never need to do maths to draw it. The machine-readable description is a JSON Schema (draft 2020-12): `python -m mathgen --figure-schema > figure_schema.json` writes it, and from Python `mathgen.figures.schema()` returns it as a dict. Its source is [`mathgen/figure_schema.py`](../mathgen/figure_schema.py).

The schema checks structure, types and ranges. The engine also checks relations a schema cannot state, such as `min < max`, or hand angles that match the time. Call `mathgen.figures.validate(spec)` for the full check. It returns a list of problems, empty when the spec is valid.

## The envelope

Every spec is an object with two keys that all kinds share.

| Key | Type | Meaning |
|---|---|---|
| `kind` | string | Which picture this is. One of the kinds below. |
| `alt` | string, not empty | A description for screen readers. |

Some kinds also take an optional `title`, drawn above the figure.

**The no-answer-leak rule.** When a question asks the learner to read a value off the figure, `alt` and every label name the object only. They never state the value. For example, `alt` says "a straight line on a grid", not "a line with slope 2". Show `alt` and the labels exactly as given. Do not add text of your own that computes or repeats a value.

**The spec itself can hold the answer.** For a read-the-picture question the spec carries the value the learner must find: a clock's `hour` and `minute`, a protractor's `angle`, a scale's reading. Never send the spec to a learner's browser for such a question. Draw the figure on your server (with the SVG renderer, or your own code) and send only the picture and `alt`. `engine.present(skill, q, reveal=False)` gives that view: it leaves out the spec, the answer and the working.

All coordinates and sizes in a spec are plain JSON numbers.

## Full descriptions for blind learners

`alt` is vague on purpose, so it leaves a learner who cannot see the figure unable to answer. `mathgen.describe.describe(spec)` returns a full description instead. It states everything a sighted learner sees: every value, label, mark and position, in reading order. It never works anything out that the picture does not show. For example, a clock is described by where its hands point, not by the time it reads. A blind learner gets the same information as the picture and no more.

That means the description can state the answer when reading the figure is the whole question, as with a protractor's arm crossing the scale at 140. The picture shows that too, so the description is as safe to send to a learner as the SVG. Show it in an accessibility mode, and keep `alt` as the default.

- `engine.present(skill, q)` returns it as `figure.description`, which is `{"text", "reveals_answer"}`. `reveals_answer` is true when the text states the answer, so a platform can treat that attempt differently (for example, not count it towards mastery). The engine finds this by matching the answer's numbers by value, or its words, so it errs towards true: an axis numeral that equals the answer also sets it.
- `engine.present(skill, q, reveal=False)` keeps the description and drops the spec.
- `figure_svg(spec, describe=True)` puts it in the SVG as a `<desc>` that `aria-describedby` points at, so a screen reader reads it after the title.
- The description is computed from the spec, so it is not a key of the spec and the schema does not list it.

## Kinds

There are 21 kinds. Each example below is a valid spec.

### cartesian

A coordinate plane. It can hold dots, segments, infinite lines and sampled curves. The y axis points up.

Keys: `x_range` and `y_range` (`[min, max]`), `grid`, `x_grid`, `y_grid` (tick and grid steps), `equal_scale` (same drawn length per unit on both axes, or fill the box), `x_tick_format` (`"number"` or `"pi"`; with `"pi"`, x values are in units of pi), `x_label`, `y_label`, `tick_labels`, and the lists `points`, `segments`, `lines` and `curves`.

- A point is `{"at": [x, y], "label"?, "open"?}`. `open` draws a hollow dot.
- A segment is `{"from": [x, y], "to": [x, y], "label"?, "open_start"?, "open_end"?}`.
- A line is `{"through": [[x, y], [x, y]], "label"?}`. It runs past both ends of the plane.
- A curve is `{"pieces": [[[x, y], ...], ...], "label"?}`. Draw each piece as a polyline. Do not join two pieces: the curve breaks there (an asymptote, or it leaves the view).

```json
{
  "kind": "cartesian",
  "alt": "A straight line on a grid.",
  "x_range": [-5, 5],
  "y_range": [-5, 5],
  "grid": 1,
  "x_grid": 1,
  "y_grid": 1,
  "equal_scale": true,
  "x_tick_format": "number",
  "x_label": "x",
  "y_label": "y",
  "tick_labels": true,
  "points": [{"at": [1, 2], "label": "P"}],
  "segments": [],
  "lines": [{"through": [[0, 0], [1, 2]]}],
  "curves": []
}
```

### number_line

A horizontal number line with dots, rays and intervals.

Keys: `range` (`[min, max]`), `tick` (step), `tick_labels`, and the lists `marks`, `rays` and `intervals`.

- A mark is `{"at": x, "open"?, "label"?}`.
- A ray is `{"from": x, "direction": "left" | "right", "open"?}`. It runs to the end of the line.
- An interval is `{"from": x, "to": x, "open_start"?, "open_end"?}`.

```json
{
  "kind": "number_line",
  "alt": "A number line with a hollow dot and a ray.",
  "range": [-5, 5],
  "tick": 1,
  "tick_labels": true,
  "marks": [],
  "rays": [{"from": 2, "direction": "right", "open": true}],
  "intervals": []
}
```

### money

Up to three labelled collections of Australian coins and notes. Draw generic discs and rectangles with the value printed. Do not copy real currency artwork.

Keys: `groups`, each `{"label", "items"}`. `label` is empty for a lone group, otherwise one or two characters such as `"A"`. Each item is `{"denom", "type", "cents"}`: `denom` is one of `5c 10c 20c 50c $1 $2 $5 $10 $20 $50 $100`, `type` is `coin` or `note`, and `cents` is the value in cents (not drawn). Items run left to right and wrap. At most 12 per group.

```json
{
  "kind": "money",
  "alt": "Some coins.",
  "groups": [
    {"label": "", "items": [
      {"denom": "50c", "type": "coin", "cents": 50},
      {"denom": "$5", "type": "note", "cents": 500}
    ]}
  ]
}
```

### hundred_chart

A 10 by 10 chart. Cells run left to right, then top to bottom.

Keys: `start` (0 or 1: the number in the top-left cell), `hidden` (cells drawn empty, the numeral not printed), `shaded` (cells filled), `marked` (cells with a heavy ring). Each list holds cell numbers. A hidden cell keeps its place in the grid. Mark shaded and ringed cells with a fill and a ring, not by colour alone.

```json
{
  "kind": "hundred_chart",
  "alt": "A hundred chart with some empty cells.",
  "start": 1,
  "hidden": [14, 37],
  "shaded": [50],
  "marked": []
}
```

### fraction_model

One to four wholes of equal drawn size, each cut into equal parts with some shaded. No fraction numeral is drawn.

Keys: `models`, each `{"shape", "parts", "shaded", "label"}`, plus `rows` and `cols` for a rectangle.

- `shape` is `bar` (vertical cuts, parts numbered 0.. from the left), `rectangle` (a `rows` by `cols` grid, numbered row by row from the top left) or `circle` (sectors, part 0 starts at 12 o'clock and numbering runs clockwise).
- `parts` is 2 to 12. A circle allows 2, 3, 4, 5, 6, 8, 10 or 12.
- `shaded` lists the filled part numbers. At least one part is shaded and at least one is not. Mark shaded parts with a hatch, not by colour alone.
- `label` is empty for one model. Several models share one shape and are labelled `"A"`, `"B"`, `"C"`, `"D"` in order.

```json
{
  "kind": "fraction_model",
  "alt": "A bar cut into equal parts with some shaded.",
  "models": [{"shape": "bar", "parts": 5, "shaded": [0, 1], "label": ""}]
}
```

### counters

Counters, dots and dice. The picture draws no text. `layout` picks the drawing. `shape` (`circle`, `square`, `star` or `diamond`) is the counter shape; dice dots are always round. Colour `1` is the main counter. Colour `2` is a second counter that must look different by more than hue (draw it ringed).

| `layout` | Extra keys | Draws |
|---|---|---|
| `ten_frame` | `frames`: 1 or 2 lists of 10 values | Ten frames of 2 rows by 5 boxes. Each box is `0` (empty), `1` or `2`, listed row by row from the top left. Two frames sit side by side. |
| `array` | `rows`, `cols` (1 to 10); optional `colours` | A full grid of counters. `colours` is a `rows` by `cols` grid of `1` or `2`. Default all `1`. |
| `groups` | `groups`: 1 to 9 integers | One ring per entry, left to right, wrapping. Each entry (0 to 9) is the count in that ring. |
| `dice` | `faces`: 1 to 3 of `{"n": 1-6, "colour": 1\|2}` | Dice faces with the standard pip patterns. |
| `scatter` | `dots`: 1 to 20 of `{"at": [x, y], "colour": 1\|2}` | Counters at free positions. `x` and `y` run 0 to 1 in a unit square; x goes right, y goes down. |

```json
{
  "kind": "counters",
  "alt": "A group of red counters.",
  "layout": "groups",
  "shape": "circle",
  "groups": [3, 4]
}
```

### base_ten

Base-ten blocks, drawn left to right: hundreds flats (10 by 10 squares), tens rods (columns of 10) and ones cubes. The counts are exactly what is drawn, so `ones` may be 10 or more.

Keys: `hundreds` (0 to 9), `tens` (0 to 20), `ones` (0 to 20). At least one block.

```json
{"kind": "base_ten", "alt": "Base-ten blocks.", "hundreds": 1, "tens": 2, "ones": 3}
```

### clock

An analogue clock face. The 12 is at the top and the hands turn clockwise. The hour hand is shorter and thicker than the minute hand.

Keys: `hour` (1 to 12), `minute` (0 to 59), `hour_angle` and `minute_angle` (hand directions in degrees clockwise from 12 o'clock; use these to draw, do not recompute), `numerals` (`all`: 1 to 12, `quarters`: 12, 3, 6, 9, `none`), `minute_ticks` (true: 60 ticks round the rim with every fifth longer; false: 12 hour ticks only), and an optional `caption` shown under the face.

```json
{
  "kind": "clock",
  "alt": "A clock face.",
  "hour": 3,
  "minute": 45,
  "hour_angle": 112.5,
  "minute_angle": 270,
  "numerals": "all",
  "minute_ticks": true
}
```

### clock_set

Two to four clocks of one size, drawn in order, for a comparison.

Keys: `clocks`, a list of 2 to 4 complete `clock` specs. Each has a distinct, non-empty `caption`.

```json
{
  "kind": "clock_set",
  "alt": "Two clock faces.",
  "clocks": [
    {"kind": "clock", "alt": "Clock A.", "hour": 3, "minute": 0, "hour_angle": 90, "minute_angle": 0,
     "numerals": "all", "minute_ticks": true, "caption": "A"},
    {"kind": "clock", "alt": "Clock B.", "hour": 6, "minute": 0, "hour_angle": 180, "minute_angle": 0,
     "numerals": "all", "minute_ticks": true, "caption": "B"}
  ]
}
```

### protractor

A semicircular protractor laid on an angle, with the angle's two rays.

In the protractor's own frame, the centre of the straight edge is the vertex. The semicircle bulges up. Two scales run 0 to 180. The outer scale has 0 at the right end of the baseline and counts anticlockwise. The inner scale has 0 at the left end and counts clockwise. Ticks every 5 degrees; every 10th is longer and labelled.

The whole protractor, rays included, is then turned anticlockwise by `rotation` degrees about the vertex. Directions below are screen directions in degrees anticlockwise from pointing right.

Keys: `angle` (the reading on the scale that starts at the base ray; a multiple of 5, 10 to 170, never 90), `zero_on` (`outer` or `inner`: the scale whose 0 is on the base ray), `rotation` (0 to 359), `base_direction`, `second_direction` (directions of the two rays; draw both from the vertex, longer than the protractor radius), `reflex` (false marks the small angle with an arc; true marks the reflex angle, the long way round, outside the protractor), and optional `names` (`vertex`, `base`, `second`, each one capital letter).

```json
{
  "kind": "protractor",
  "alt": "A protractor laid on an angle.",
  "angle": 40,
  "zero_on": "outer",
  "rotation": 0,
  "base_direction": 0,
  "second_direction": 40,
  "reflex": false,
  "names": {"vertex": "O", "base": "A", "second": "B"}
}
```

### unit_strip

Objects measured by identical informal units (paper clips, cubes) laid beside them. Distances are in unit lengths; 1 is the nominal width of one unit.

Keys: `unit` (singular name, written as a caption), `unit_style` (`paper_clip`, `cube`, `block`, `footprint` or `counter`: the picture to repeat) and `rows` (1 to 3, top to bottom). A row is `{"label", "length", "units"}`. The bar runs from 0 to `length` with an end line at each end. `units` are `{"start", "length"}` and are drawn under the bar. A gap between units shows as empty space. An overlap shows as units drawn over each other, the later one on top. A unit may hang past the end of the object.

```json
{
  "kind": "unit_strip",
  "alt": "A pencil measured with paper clips.",
  "unit": "paper clip",
  "unit_style": "paper_clip",
  "rows": [
    {"label": "pencil", "length": 3, "units": [
      {"start": 0, "length": 1}, {"start": 1, "length": 1}, {"start": 2, "length": 1}
    ]}
  ]
}
```

### scale

A measuring scale. Ticks run from `min` to `max`: a short tick every `minor`, a longer labelled tick every `major`. `unit` is written once beside the scale.

Keys: `variant`, `unit`, `min`, `max`, `major`, `minor`, `extra_ticks` (values halfway between two minor ticks that get an extra short tick), and what is indicated.

| `variant` | Looks like | Indicated by |
|---|---|---|
| `ruler` | Horizontal, `min` at the left. | `object`: `{"start", "end", "label"?}`, a bar above the ruler with a short line down to the scale at each end. |
| `thermometer` | Vertical, `min` at the bottom, with a bulb. May be negative. | `level`: liquid fills the tube up to that value. |
| `jug` | Vertical jug, starts at 0. | `level`: the liquid surface. |
| `dial` | Round face. Values run clockwise from `min` at `start_angle` through `sweep` degrees to `max`. Angles are degrees clockwise from 12 o'clock. | `level`: where the needle points. Needs `start_angle` and `sweep`. |

Instead of `level` or `object`, a scale may carry `marks`: up to two `{"value", "label"}` arrow markers, for questions that compare two readings. Give exactly one of `level`, `object` or `marks`. Every indicated value sits on a minor tick or an extra tick.

```json
{
  "kind": "scale",
  "alt": "A thermometer.",
  "variant": "thermometer",
  "unit": "°C",
  "min": 0,
  "max": 40,
  "major": 10,
  "minor": 2,
  "extra_ticks": [],
  "level": 22
}
```

### diagram

A labelled two-dimensional geometry diagram: triangles, crossing and parallel lines, angles at a point.

The drawing frame is an abstract box `[0, W] x [0, H]`. Its origin is at the bottom left and y points up. Scale both axes equally.

Keys: `frame` (`[W, H]`), `not_to_scale` (true: print "Not drawn to scale" under the figure), `points`, `segments`, `polygons`, `angles`.

- A point is `{"id", "at": [x, y], "label"?}`. Other items refer to points by `id`. A labelled point (up to 4 characters) is named on the drawing, just outside the figure, away from the segments that leave it. A point without a label is not drawn.
- A segment is `{"from": id, "to": id, "label"?, "ticks"?, "arrows"?, "dashed"?}`. The label (for example `"5 cm"`) goes at the midpoint, offset away from the centroid. `ticks` (1 to 3) are cross-marks: segments with the same count are equal in length. `arrows` (1 to 2) are chevrons pointing from `from` to `to`: segments with the same count are parallel. A segment has ticks or arrows, not both. `dashed` draws a dashed line.
- A polygon is `{"points": [id, ...], "fill"?}`: three or more points joined in order. If a segment names the same pair, it only adds a label or style to that edge. Draw one line.
- An angle is `{"at": id, "from": id, "to": id, "label"?, "right"?}`. It is measured anticlockwise from the ray `at -> from` to the ray `at -> to`, so a sweep over 180 degrees is a reflex angle. Draw an arc of constant radius and put the label at the middle of the sweep, just beyond the arc. With `right: true`, draw a small square in the corner instead. A right angle has no label.

```json
{
  "kind": "diagram",
  "alt": "A right-angled triangle.",
  "frame": [100, 100],
  "not_to_scale": false,
  "points": [
    {"id": "A", "at": [10, 10], "label": "A"},
    {"id": "B", "at": [70, 10], "label": "B"},
    {"id": "C", "at": [10, 55], "label": "C"}
  ],
  "segments": [{"from": "A", "to": "B", "label": "4 cm"}],
  "polygons": [{"points": ["A", "B", "C"]}],
  "angles": [{"at": "A", "from": "B", "to": "C", "right": true}]
}
```

### tree_diagram

A probability tree read left to right, two or three stages deep. All leaves are at the same depth, with at most 12 leaves.

Keys: `root`, `{"label"?, "children"}`. The label (for example "Start") sits beside the root. Each node is `{"branch", "label", "children"?, "outcome"?, "outcome_p"?}`.

- `branch` is the text on the branch leading into the node: a probability as a fraction (`"3/5"`) or decimal (`"0.4"`), or `"?"` for a value the learner works out (draw an empty box).
- `label` is the event at the node, drawn at the node.
- A node with `children` has 2 to 4 of them. Nodes are laid out top to bottom in the order given.
- A leaf has no `children` and must have `outcome`, the whole path as text, drawn to the right of the leaf. `outcome_p` optionally gives that outcome's probability, or `"?"`.

```json
{
  "kind": "tree_diagram",
  "alt": "A two-stage probability tree.",
  "root": {"label": "Start", "children": [
    {"branch": "1/2", "label": "Red", "children": [
      {"branch": "1/2", "label": "Red", "outcome": "Red, Red"},
      {"branch": "1/2", "label": "Blue", "outcome": "Red, Blue"}
    ]},
    {"branch": "1/2", "label": "Blue", "children": [
      {"branch": "1/2", "label": "Red", "outcome": "Blue, Red"},
      {"branch": "1/2", "label": "Blue", "outcome": "Blue, Blue"}
    ]}
  ]}
}
```

### venn

A Venn diagram of two or three sets inside a universal rectangle. With two sets, A is left and B is right, as overlapping circles. With three, A is top-left, B top-right and C below, each pair overlapping and all three overlapping in the middle.

Keys: `universe` (label in the rectangle corner, for example `"U"`), `sets` (2 or 3 of `{"id": "A", "name"?}`; `id` is one capital letter, and `name` is written beside the circle, default the id), and `regions`.

`regions` has one entry for every region. A key lists the sets the region lies in, in set order: `"A"` is inside A only, `"AB"` inside A and B only, `"ABC"` inside all three, and `"none"` is the rectangle outside every circle. The value is the text drawn inside: a count (`"12"`), a short comma-separated list of elements (`"2, 4, 6"`) or `"?"` for a region the learner works out. Counts are exclusive. Two sets have 4 regions, three sets have 8.

```json
{
  "kind": "venn",
  "alt": "A Venn diagram of two sets.",
  "universe": "U",
  "sets": [{"id": "A", "name": "Tennis"}, {"id": "B", "name": "Chess"}],
  "regions": {"A": "5", "B": "7", "AB": "3", "none": "?"}
}
```

### bar_chart

A bar (column) graph with a gridded value axis. One series, or two drawn side by side with a legend.

Keys: `orientation` (`"vertical"`: columns, first category at the left; `"horizontal"`: bars growing rightwards, first category at the top), `categories` (1 to 10 distinct names; at most 6 with two series), `series` (one or two `{"name", "values"}`, one value per category), `value_axis`, `x_label`, `y_label`.

`value_axis` is `{"min": 0, "max", "step", "minor"}`: a gridline and numeral at every `step` from 0 to `max`, and with `minor` an unlabelled half line between them. Every value sits on a gridline, or a half line when `minor` is true.

```json
{
  "kind": "bar_chart",
  "alt": "A bar graph of books read each day.",
  "orientation": "vertical",
  "categories": ["Mon", "Tue", "Wed"],
  "series": [{"name": "", "values": [2, 5, 3]}],
  "value_axis": {"min": 0, "max": 6, "step": 1, "minor": false},
  "x_label": "Day",
  "y_label": "Books"
}
```

### picture_graph

A pictograph: one row of repeated symbols per category, the first row at the top, and a key written as `Each symbol = k` followed by the unit.

Keys: `symbol` (`circle`, `square`, `triangle`, `star`, `heart` or `diamond`), `key` (`{"each": k, "unit"}`), `rows` (`{"label", "symbols"}`; `symbols` is a whole number or ends in `.5`, drawn as the left half of a symbol, and a half is allowed only when `k` is even; at most 12).

```json
{
  "kind": "picture_graph",
  "alt": "A picture graph of favourite fruit.",
  "symbol": "circle",
  "key": {"each": 2, "unit": "students"},
  "rows": [{"label": "Apple", "symbols": 3}, {"label": "Pear", "symbols": 1.5}]
}
```

### dot_plot

A number axis with a column of equal dots above each value, one dot per observation.

Keys: `range` and `step` (a tick and numeral at every step, at most 40), `dots` (`{"at", "count"}` in increasing order of `at`; each `at` is a tick, `count` is 1 to 12), `x_label`.

```json
{
  "kind": "dot_plot",
  "alt": "A dot plot of goals scored.",
  "range": [0, 5],
  "step": 1,
  "dots": [{"at": 1, "count": 2}, {"at": 2, "count": 4}, {"at": 4, "count": 1}],
  "x_label": "Goals"
}
```

### histogram

Equal-width classes with touching bars. Two panels are stacked, each with its own axes and a label above it, to compare two data sets.

Keys: `panels` (one or two of `{"label", "edges", "frequencies", "frequency_axis"}`), `x_label`, `y_label`. `edges` are the class boundaries (3 to 13, one width apart); a class includes its lower edge and excludes its upper edge. `frequency_axis` has the same shape as a bar chart's `value_axis`.

```json
{
  "kind": "histogram",
  "alt": "A histogram of heights.",
  "panels": [
    {"label": "", "edges": [150, 155, 160, 165], "frequencies": [3, 6, 2],
     "frequency_axis": {"min": 0, "max": 6, "step": 1, "minor": false}}
  ],
  "x_label": "Height (cm)",
  "y_label": "Frequency"
}
```

### box_plot

One or two box-and-whisker plots on a shared horizontal axis, the first at the top.

Keys: `range`, `step` and `minor` (as for `dot_plot`, with unlabelled half ticks when `minor`), `plots` (one or two of `{"label", "min", "q1", "median", "q3", "max", "outliers"}`), `x_label`. `min` and `max` are the whisker ends, the extreme values that are not outliers. Each outlier is drawn as a cross and lies beyond the 1.5 × IQR fences.

```json
{
  "kind": "box_plot",
  "alt": "A box plot of test scores.",
  "range": [0, 20],
  "step": 2,
  "minor": false,
  "plots": [{"label": "", "min": 4, "q1": 8, "median": 10, "q3": 12, "max": 16, "outliers": [20]}],
  "x_label": "Score"
}
```

### pie_chart

A circle cut into `parts` equal wedges, with a thin radius at every boundary, so each sector's share can be counted. The first sector starts at 12 o'clock and the rest follow clockwise.

Keys: `parts` (2, 3, 4, 5, 6, 8, 10 or 12), `sectors` (2 to 6 of `{"label", "parts"}`, whose parts sum to the chart's).

```json
{
  "kind": "pie_chart",
  "alt": "A pie chart of how students travel to school.",
  "parts": 8,
  "sectors": [{"label": "Walk", "parts": 3}, {"label": "Bus", "parts": 4}, {"label": "Car", "parts": 1}]
}
```

## Drawing figures

### Option A: use the SVG renderer

```python
from mathgen.render import figure_svg

svg = figure_svg(spec, theme="light", id_prefix="q1-fig")
```

`figure_svg` returns one self-contained SVG string. It needs no JavaScript, no web fonts and no network. Put it in a page as it is, save it as a `.svg` file, or convert it to PNG or PDF with any SVG tool. It raises `ValueError` for an invalid spec.

- **Accessibility.** The root element has `role="img"` and `aria-labelledby` pointing at a `<title>` that holds the spec's `alt`. A screen reader announces the figure as one image with that description.
- **Theming.** Every colour is a CSS custom property with a light default. Set any `--mgf-*` variable on the SVG or on an ancestor to restyle all figures on the page:

  ```css
  .question { --mgf-text: #0b1220; --mgf-accent: #7c3aed; }
  ```

  The variables are `--mgf-text`, `--mgf-text-muted`, `--mgf-surface`, `--mgf-surface-2`, `--mgf-surface-3`, `--mgf-border`, `--mgf-border-strong`, `--mgf-accent`, `--mgf-accent-soft`, `--mgf-tick`, and the data colours `--mgf-c1` to `--mgf-c5`. Their defaults are in `mathgen.render.THEME_VARIABLES`. Pass `theme="dark"` to bake in the dark defaults instead, for an SVG shown outside a page (a dark slide, a PNG).
- **Scoped style.** The `<style>` inside the SVG applies only to elements with the `mgf` class, so inlining the SVG does not restyle anything else on the page.
- **Ids.** Clip paths, arrowheads and the title get ids with a per-figure prefix. By default the prefix comes from a hash of the spec, so two different figures never clash. If you draw the same spec twice on one page, pass a different `id_prefix` for each.

### Option B: draw natively

Use the schema as the contract. Check the incoming spec against it, branch on `kind`, and draw each kind with your own components.

1. Validate with any draft 2020-12 validator, or call `mathgen.figures.validate(spec)`.
2. Branch on `kind`. Draw only what the spec lists. Do not infer a value the spec does not carry.
3. Keep text exactly as given. Do not add labels, and do not print `alt` on screen if it would give away what the learner must read off.
4. Give the figure an accessible name from `alt` (for example `role="img"` and `aria-label`, or a native control's own label).
5. Do not rely on colour alone. Where this guide says a part is filled, ringed, hatched or drawn differently, keep that second cue.
6. Keep the frame's scale where the kind says so. A `diagram` frame, and a `cartesian` plane with `equal_scale: true`, must not be stretched on one axis.

Guarantees you can rely on when `validate` returns no problems: points lie inside their plane or frame; tick, grid and part counts are within the limits above; clock hands match the time; protractor rays match the stated angle; sibling probabilities in a tree total 1. A `diagram` with `not_to_scale: false` has numeric angle labels within 2 degrees of the drawn angle and numeric length labels proportional to drawn length within 6%.
