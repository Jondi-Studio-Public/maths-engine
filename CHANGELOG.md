# Changelog

All notable changes to maths-engine (the `mathgen` package) are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release (0.x to 0.y) may change
the public API; a patch release does not.

A seed is not promised to give the same question in every release, because a generator may be fixed or improved.
Each question's id (below) tells you when the question behind it has changed.

## [Unreleased]

### Added

- Interactive graph answers. Seven answer kinds (`half_planes`, `line_graph`, `number_line_set`, `polygon`, `curve`, `bars`,
  `edges`) read what a learner draws on a graph, as JSON a front end sends or as a typed form. Points are compared as
  exact fractions on the figure's snap grid, with no tolerance, and unreadable or oversized replies are marked wrong.
  A `cartesian` or `number_line` figure may carry a `canvas` key naming the reply kind, the tools and the snap step.
  Six skills use them: plot a point, draw a line, an inequality on a number line, shade an inequality region, transform
  a shape on a grid and draw a parabola. The schemas are in `docs/design/interactive-graph-answers.md`.
- **43 more skills and drawn networks** (#120). The school-year coverage tail adds 43 skills. A `network` figure kind
  (labelled vertices at fixed positions, undirected or directed edges, edge labels, highlighted edges and vertices)
  has an SVG renderer, and the graph and network generators now draw the network their text describes; the text is
  unchanged. Exam chains were added or widened for scale drawings, floor-plan, site-plan and map conventions, and
  for the product-to-sum, De Moivre, sinusoid, simple-harmonic-motion and linear-combination skills.
- **26 more skills** (#128): graph transformations (equation, f(x) squared, square root of a function, modulus,
  inverse trigonometry), conditional counts, simulated sample proportions, tan and sec integrals, complex exponential
  form, seven data-display and graph skills (graph terms from a network, constructing column, picture, dot and
  stem-and-leaf displays, comparing displays, real-data graphs, a grid map) and ten early-years skills (twos facts,
  estimating a collection, angles, chance, solids, symmetry). Each has easy, medium and hard tiers and a
  follow-through exam chain.
- **6 more skills** (#133), closing gaps found against the NSW, Victorian and Queensland senior courses: tracing
  pseudocode (`pseudocode-trace-output`), features of power-function graphs (`power-function-graph-features`),
  complex-plane geometry with ratios, midpoints and w = az + b (`complex-arg-ratio-geometry`), trigonometric
  substitution integrals (`trig-substitution-hsc-tan-and-convert`), choosing mode, median or mean (`choose-summary-statistic-for-data`) and
  bits, hexadecimal and digit counts (`bits-hex-and-digit-counts`, hexadecimal typed as text). Each has an exam chain.
- **New exam chains** (#132, #135): reasonableness of an answer and causation versus correlation (General and
  Essential courses), and typed multi-part chains for proof by contradiction and for sin, cos and tan graphs on an
  extended domain (a second exam source on `senior-recognise-the-graphs-of-y-sin-x-y`).
- **Two grid skills now active** (#136): `f10-tessellation-transform-combinations` and
  `f10-two-dimensional-object-representations` were deferred and are now practised like the box-nets and
  turn-shapes skills. The plan-views exam chain now draws the key's rank at random among the options, and plan
  views of cube stacks are tagged AC9M3SP01, so their ids change.

### Fixed

- Seven rounds of a content bug hunt (#121, #124, #125, #126, #127, #130) changed some questions for the same seed, so
  their ids change. Fixed: a wrong key in the cosine and sine ratio angle question (arcsin was keyed for an adjacent
  leg); multiple-choice generators that offered a different number of options from draw to draw (they now always
  offer the same number); a divisibility generator that gave zero answers; a function-concept key that did not match
  its displayed square root; a quartile method the stem did not state; impossible triangles and quadrilaterals; and a
  time-zone working line with the wrong sign.
- Student-visible text: placeholder `answer: N` working, leaked tokens, raw braces, SymPy's `I`, Python notation and
  dicts in worked solutions now read as plain maths. Plurals and articles agree ("1 part", "an 8"), `^1`, `1x`,
  `1.0x`, `X = X` and unbracketed negative factors are gone from working and stems, and implausible contexts
  (a brick lighter than a banana, half-way prices, negative durations) are redrawn. (#121, #125, #126, #127, #130)
- About 25 skills raised an error on some seeds (empty distractor candidates, a distractor that collapsed onto the
  key, a costed graph draw whose total was one printed cost); they now redraw. (#126, #127, #130)
- The variance multiplier in the linear-transform working printed `60.0^2` instead of `60^2`. (#129)
- A fourth round of review fixes (#134) changes some questions and replies. The displayed stem of
  `ucv1-lhopital-rule` and `integrate-tan-sec-powers` showed the variable x as a times sign (`lim(x→∞) × ln(...)`),
  which changed the limit; it now reads as the variable. An answer holding `e` inside a tuple (`uc2-curl-of-field`)
  was marked wrong although it matched the displayed key. `trig-equations-radians-list` now accepts `π` as the stem
  asks. `f10-fraction-decimal-links` no longer accepts the decimal in the stem as the answer to "write it as a
  fraction". The bar-chart grader matches category names without regard to case. Two slow replies are fixed: a long
  reply with `over` in it took minutes to read, and a reply such as `cos((E-2)^6)` hung the factorising graders. A
  batch to the HTTP grading endpoint first stopped running replies after three had timed out (see the next entry for
  the full limit).
- Robustness and service fixes (#137):
  - A batch to `/v1/grade/batch` may spend at most `MATHS_API_BATCH_BUDGET` worker-seconds (default three times
    `MATHS_API_TIMEOUT`); every outcome of an item, success or not, is charged to the budget, and an item that would not get a full deadline is not run. Batches together use at most half
    the workers, and another batch gets `503` with `Retry-After`, so single grades keep a worker (this needs two or
    more workers). An opt-in per-client cool-down, `MATHS_API_CLIENT_STRIKES` (default `0`, off), names clients by API
    key or, when the server is open, by address.
  - A reply of repeated labels (`x=x=...`) is collapsed before it is tidied: 40,000 characters grade in about 10 ms.
  - `sympify_safely` and `core._sympify` read a dict or set item by item through the safe parser.
  - `grade_with_deadline` is exported from `mathgen.engine` and documented.
  - `year=university` (any case) now matches in the HTTP API, the MCP server and `mathgen.service`.
  - JSON `true`, `false` and `2.0` are no longer a multiple-choice option index over HTTP and MCP; send an integer.
  - `make_with_id` refuses a seed that is not an integer, and an id's seed must be written `-?[0-9]+`.
  - A figure renderer that overflows or divides by zero raises the documented `ValueError`.

## [0.2.0] - 2026-10-19

### Upgrading from 0.1.0

- Many questions differ from 0.1.0 for the same seed: 107 exam tiers became multi-part chains (#72), other exam
  tiers gained chains (#75, #83, #94), and marking and stem fixes changed some prompts (#82, #98). Store question ids
  (below), not just seeds, and use `engine.regenerate` to tell when a stored question has changed.
- `Part` gained a `sig_figs` field before `seed` (#82). Code that builds a `Part` positionally past
  `sensible_range` must pass `seed` by keyword.
- Marking is stricter where the question asks for it (#82): a decimal or unreduced fraction is wrong where lowest
  terms or simplest form is asked, a reply with more significant figures than asked is wrong, and a reply in a
  unit other than the one asked for is wrong. Every canonical answer in the library is still accepted.
- Marking is stricter again (#104). An expression around the value (`sqrt(9)`, `9!`, `9%`, `f(9)`, `2*9`) is wrong
  for an answer of 9, and so is a doubting reply ("not 9", "9 is wrong", "maybe 9", "no 9"). A multiple-choice
  reply takes only an option index, a label (`B`, `(b)`) or the whole option text. **Send an option index as an
  integer, not a digit string**: where the options are themselves numbers, a string equal to an option's text picks
  that option (text wins). The command line does this for you. "Point five" reads as 0.5, and "three and four" is
  two values, not 7. About 140 canonical answers that were wrongly refused (numeric-option multiple choice, 34
  skills such as `count-to-10`) are now right. Every other canonical answer is still accepted.
- Unit prefixes are read case-sensitively where it matters (#111): `2000 ML` (megalitres) no longer passes for
  `2000 mL`, and `mPa` or `mHz` (milli spelled with a small m on a mega unit) no longer pass for `MPa` or `MHz`.
  `2000 ml`, `5 KM` and all-capital `MM` are still read as before.
- Marking is stricter about what surrounds a number (#117). A stray letter or operator next to the value is wrong
  (`7x`, `7/x`, `x7` for 7), and so is a minus that belongs to nothing: `-(-6, 4)`, `-A$352.50`, `-F = 10.0 N` and
  `-x = 7` no longer lose their minus and pass. Labels (`x = 7`, `P(2, 3)`), unit symbols (`7 m`, `20 J`), dotted
  abbreviations (`3 p.m.`) and unit words (`7 dogs`, `0.064 cm`) are still read as before. An SI prefix on the
  unit a question asks for is a different unit (`10 mN`, `10 MN` for 10 N). Every canonical answer in the library
  is still accepted.
- Some questions changed text for the same seed, so their ids change: 11 worked solutions that printed `-3² - 4(6)`
  where `(-3)²` was meant (#104), and the early-years skills rebuilt in #103 (below). `compare-lengths` is now
  picture-based, `f10-compare-measurement-attributes` is renamed "Compare mass and capacity directly" and no longer
  asks about length or duration; `ey-compare-duration` (#108) now covers comparing durations.

### Added

- **Question ids.** Every question can carry a stable id, `<skill id>/<source key>/<seed>/<hash>`, where the
  hash covers the question's full content. New in `mathgen.engine`: `make_with_id`, `question_id`,
  `content_hash`, `parse_question_id`, `regenerate` and the `QuestionChanged` exception. `regenerate(id, skills)`
  rebuilds the question an id names and raises `QuestionChanged` if the library now generates different
  content, so an application never marks an old answer against a changed question. (#53)
- **Source keys.** A skill source may carry an optional `key` (lower-case words joined by hyphens). It defaults
  to the source's `difficulty` label and is needed only when two sources of one skill share a difficulty.
  `Source.effective_key` reports the name in use, and `engine.pick_source` accepts a key. (#53)
- **HTTP API and exports.** `python -m mathgen.server` (install the `server` extra) serves the engine over HTTP
  with `learner` and `author` keys; `mathgen.service` returns skills, questions, worksheets and grading as plain
  JSON-ready data; `mathgen.export` builds printable papers and Moodle XML and QTI 2.1 banks; `mathgen.svg` draws
  every figure kind as SVG. A `Dockerfile` and `compose.example.yaml` run it. See `docs/server.md`. (#71, in #81)
- **Ready-made LMS question banks.** `scripts/build_banks.py` writes a Moodle XML bank and a QTI 2.1 package (for
  Canvas and others) for every skill, ready to attach to a release. See `docs/banks.md`. In the exports, a QTI item
  marks each reply exactly and shows the right answer as feedback, a Moodle option explains itself in words, and the
  QTI export is marked experimental. (#87)
- **MCP server.** `mathgen-mcp` (install the `mcp` extra) serves the engine to AI agents over the Model Context
  Protocol: find skills, make and mark questions, build worksheets and LMS banks. In the `learner` role no tool
  returns an answer. See `docs/mcp.md`. (#81)
- **Agent Plugins.** `plugins/maths-engine-tutor` (learner role) and `plugins/maths-engine-teacher` (author role)
  package the server with skills for tutoring, worksheets and LMS export, following the Agent Plugins spec, and
  `.claude-plugin/marketplace.json` lists both for Claude Code. They install `maths-engine[mcp]==0.2.0` from PyPI.
  (#81)
- **About 210 new skills**, from early years to second-year university:
  - Years 5 to 10: nets, plan views, scale plans, maps, experimental probability and three modelling skills
    (#75), and 45 F-10 and early-years skills (#91).
  - Senior secondary: networks, mechanics, residual plots, binomial expansion, reciprocal trigonometry and polar
    form (#76), and 36 senior and first-year gap-closing skills (#94).
  - First-year university: calculus, 3D geometry, hypothesis tests, complex numbers and integration techniques
    (#77), and linear algebra, Taylor polynomials, series tests and ODEs (#79).
  - Second-year university: abstract algebra, numerical methods, probability and statistics, and discrete
    mathematics (#93).
- **174 more skills** (#115, #118), so the library now has about 1,250 active skills.
  - A second build train (155 skills, #115): round 3 (87 skills across probability, calculus, matrices, mechanics,
    finance, trigonometry, differential equations and discrete mathematics), second-year university (58), complex
    analysis (7), and z-tests, gradient vectors, e as a limit, critical path and more early-years variety. 47
    single-part exam generators gain a second exam source.
  - 19 skills for rows that had none (#118), each with easy, medium, hard and exam tiers: Boolean algebra and logic
    gates, a line meeting a parabola or circle, the auxiliary angle, complex polygon vertices and roots of unity,
    complex rotation, sinusoid features, 4x4 cofactor expansion, Hermitian and unitary matrices, SVD of small
    matrices, multivariable Taylor with the Hessian, complete bipartite and connected graphs, minimum cut and
    maximum flow, dynamic-programming tables, relative velocity, unit-cost depreciation, number bases, primes and
    divisors, floor and ceiling estimation, quadric surfaces and error-correcting codes.
- **Grid figures and shaded-cell answers.** A `grid_drawing` figure kind draws scale drawings, maps with grid
  references, stacks of cubes, nets and plan or elevation views. A `grid_cells` answer kind lets a learner answer by
  shading cells, with four match modes (exact, translate, congruent and cube net). 17 skills use them: 13 are
  active, and 4 shading skills are deferred until the visual-construction gate is benchmarked. (#90)
- **Early-years figures and read-aloud.** Five new figure kinds (`shape_scene`, `pattern_strip`, `tally`,
  `calendar`/`day_strip` and `compare_picture`) have SVG renderers, and a `read_aloud` skill flag marks skills whose
  prompts are at most 15 words. 22 new and 6 rewritten Foundation to Year 2 skills use them for shapes, patterns,
  data, calendars and measurement comparison, and seven new senior graph-reading skills reuse the cartesian,
  dot-plot and histogram figures. (#103) Ten more skills follow (#108): position words, directions, steps and
  quarter or half turns on a grid, simple maps, Years 3 to 6 plans, turns and nets on a grid, latitude and longitude,
  and `ey-compare-duration`, which uses a new `durations` layout of `compare_picture`.
- **Exam chains.** 13 graph-reading, box-plot and time skills (#83) and the tree-diagram, Venn and stem-and-leaf
  skills (#75) gain multi-part exam questions with follow-through, marks and tagged distractors.
- **Year levels for senior skills.** `year_level` is now filled for 269 senior skills from ACARA's unit pages
  (Units 1 and 2 are Year 11, Units 3 and 4 are Year 12). (#88)
- **Grader.** `|x - 1| < 4` is read as an inequality, a row or column vector may carry a transpose mark
  (`(2, 3)^T`), a fraction reply may carry the unit asked for or a leading `$`, and a labelled reply such as
  `P(red) = 1/8` is accepted. (#82)
- **A shared quality contract for exam generators.** Every exam generator is now checked the same way: at least 100
  distinct answers with none above 5%, no text defects, three or more prompt shapes, well-formed parts and
  determinism. Known gaps are pinned in a shrink-only `tests/generator_contract_baseline.yaml`. The check found
  `exd_x_sc3_apply_rule_chain` saying "1 seconds", now "1 second". (#89)
- An unknown topic or skill now suggests the closest names (`did you mean 'quadratic'?`), from `mathgen.make`
  and from the command line. (#86)
- Python 3.14 is listed as supported. The generators and skills were checked on it. (#86)
- `docs/architecture.md` explains how a skill, its generators and its grader fit together (#85), and
  `docs/api.md` lists the public API.

### Changed

- 107 exam tiers whose answer could be guessed, or that took a single step, are now follow-through chains: later
  parts use the learner's own earlier answers. (#72) Seven more skills whose answers could be guessed from the
  options were rebuilt (#91), and six thin exam tiers were rebuilt (#94).
- `skills.validate` reports an error when two sources of one skill share a key (by default, a difficulty
  label). Add an explicit `key` to one of them. (#53)
- `mathgen.make` with an unknown topic raises a `KeyError` that names close matches instead of listing every
  topic. (#86)
- `import mathgen` is about twice as fast: the Euler's-formula generator builds its pool of graphs on its first
  hard or exam question instead of at import. (#86)
- Stricter marking of stray letters, a dropped minus and SI prefixes: see "Upgrading from 0.1.0". (#117)
- `ucv1-definition-quantifier-check` no longer prints `None` in two of its answer options, and the text-quality
  check flags a Python `None` or `NaN` printed into learner text. (#117)
- Displayed notation (#116): `x*` and `z*` (conjugates and starred variables) now stay as written instead of
  turning into `x ×`, and the natural log shows as `ln` instead of sympy's `log` in integration by parts, partial
  fractions and the heat-rod answers. Division of decimals and the directional derivative now say "correct to 6
  significant figures" where the key is rounded.
- `README.md` and `SECURITY.md` described the engine as having no server; they now describe the optional HTTP API
  and MCP server, and `SECURITY.md` says what is in scope for them. (#116)

### Fixed

- Figures: tick, axis and point labels no longer have curves struck through them, a step graph draws the filled dot at
  its closed end, side-length labels sit clear of the edge, and curve labels that left the frame are kept inside.
  Only the picture changes, never a question, key or grader. (#102)
- A seed near 2**63 issued a question id the server then refused to grade when `MATHS_API_ID_SECRET` is set;
  `make_questions` now refuses such a seed with a clear `BadRequest`. `skills.validate` reported a crash instead of a
  problem for a `stages` value that is not a list or for nested or string `prerequisites`, and a `domains.yaml` entry
  without an `id` raised a bare `KeyError`; it now raises a `ValueError` naming the file and entry. (#107)
- Four marking misreads from the engine core bug hunt (#111). A mixed number typed as `1 1/2` was read as 1/2, so
  a key of 3/2 was rejected and a key of 1/2 accepted; it now reads as 3/2, and `-1 1/2` as -3/2. `e^2`, `log10(x)`
  and `log2(x)` could not be read. A zero vector never matched a zero key in a matrix scalar multiple, and
  `(a)5(b)6(c)7` with no spaces was not split into parts. A plain "2 3" is still 2 times 3.
- `compare-groups-evidence` raised an error on about 1 draw in 300 to 1,000 (every tier, including the exam chain)
  when its wrong answers for a difference in medians, or in an interquartile range, collapsed onto the right one,
  leaving fewer than two distractors; building the question banks stopped on it. Those draws now get spare wrong
  values, and every other draw is unchanged. (#114)
- A reply of about fifteen characters, such as `sqrt(acos(x)^2)`, held the grader for 40 seconds or more on about a
  hundred expression skills, and `1/0`, `tan(pi/2)`, `log(0)` and `nan` took 3 to 6 seconds on
  `ucv1-heat-equation-mode`; all now grade in under 0.1 seconds. (#104)
- A multi-part reply of `0` (the first option's index, or an answer of zero) was read as an empty box and earned
  no marks. (#88)
- Four exam generators failed on a few seeds when a slip coincided with the answer; they now draw again. (#97)
- `f10-rational-operation-strategy` (medium) gave a wrong answer key on 299 of 300 seeds, offering the true answer
  as a wrong option. The key now matches the question on every seed. Worked solutions in 13 other skills printed
  wrong or garbled text with right keys (a skipped division, negatives without brackets, `|r| <= -1`, "1 steps");
  they read correctly now. (#98)
- A prerequisite cycle made `gaps_before` loop forever; it now raises `ValueError`. A space between digits in a typed
  reply ("2 000", "3 4") was read as a product, so `x > 2 000` earned full marks against `x > 0`; it is no longer a
  product, and mixed numbers such as "2 1/2" still read. A lone surrogate in a JSON string made the HTTP API answer
  500; it now answers 422. A space after `author:` in `MATHS_API_KEYS` kept the key from ever matching. (#100)
- Marking and stem defects found by an audit of the exam chains and university skills: a correct reply with the
  unit written out (`100 s`) or as an exact fraction could score nothing, a wrong unit could score full marks,
  a reply that was not text raised instead of being marked wrong, and several stems used a rounded earlier
  answer without saying so. (#82)

## [0.1.0] - 2026-10-13

First public release: about 1,400 question generators with verified answers and distractors the grader
rejects, typed-answer graders, a skill library under CC BY 4.0, spaced-review scheduling and a command-line
interface.

[Unreleased]: https://github.com/Jondi-Studio-Public/maths-engine/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Jondi-Studio-Public/maths-engine/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Jondi-Studio-Public/maths-engine/releases/tag/v0.1.0
