# Changelog

All notable changes to maths-engine (the `mathgen` package) are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release (0.x to 0.y) may change
the public API; a patch release does not.

A seed is not promised to give the same question in every release, because a generator may be fixed or improved.
Each question's id (below) tells you when the question behind it has changed.

## [Unreleased]

## [0.2.0] - 2026-10-14

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
