# Changelog

All notable changes to maths-engine (the `mathgen` package) are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release (0.x to 0.y) may change
the public API; a patch release does not.

A seed is not promised to give the same question in every release, because a generator may be fixed or improved.
Each question's id (below) tells you when the question behind it has changed.

## [Unreleased]

### Added

- **Question ids.** Every question can carry a stable id, `<skill id>/<source key>/<seed>/<hash>`, where the
  hash covers the question's full content. New in `mathgen.engine`: `make_with_id`, `question_id`,
  `content_hash`, `parse_question_id`, `regenerate` and the `QuestionChanged` exception. `regenerate(id, skills)`
  rebuilds the question an id names and raises `QuestionChanged` if the library now generates different
  content, so an application never marks an old answer against a changed question. (#53)
- **Source keys.** A skill source may carry an optional `key` (lower-case words joined by hyphens). It defaults
  to the source's `difficulty` label and is needed only when two sources of one skill share a difficulty.
  `Source.effective_key` reports the name in use, and `engine.pick_source` accepts a key. (#53)
- An unknown topic or skill now suggests the closest names (`did you mean 'quadratic'?`), from `mathgen.make`
  and from the command line.
- Python 3.14 is listed as supported. The generators and skills were checked on it.
- **MCP server.** `mathgen-mcp` (install the `mcp` extra) serves the engine to AI agents over the Model Context
  Protocol: find skills, make and mark questions, build worksheets and LMS banks. In the `learner` role no tool
  returns an answer. See `docs/mcp.md`. (#81)
- **Agent Plugins.** `plugins/maths-engine-tutor` (learner role) and `plugins/maths-engine-teacher` (author role)
  package the server with skills for tutoring, worksheets and LMS export, following the Agent Plugins spec, and
  `.claude-plugin/marketplace.json` lists both for Claude Code. They install `maths-engine[mcp]==0.2.0` from PyPI.
  (#81)
- **HTTP API and exports.** `python -m mathgen.server` (install the `server` extra) serves the engine over HTTP
  with `learner` and `author` keys; `mathgen.export` builds printable papers and Moodle XML and QTI 2.1 banks;
  `mathgen.svg` draws every figure kind as SVG. See `docs/server.md`. (#71, in #81)

### Changed

- `skills.validate` reports an error when two sources of one skill share a key (by default, a difficulty
  label). Add an explicit `key` to one of them. (#53)
- `mathgen.make` with an unknown topic raises a `KeyError` that names close matches instead of listing every
  topic.
- `import mathgen` is about twice as fast: the Euler's-formula generator builds its pool of graphs on its first
  hard or exam question instead of at import.

## [0.1.0] - 2026-10-13

First public release: about 1,400 question generators with verified answers and distractors the grader
rejects, typed-answer graders, a skill library under CC BY 4.0, spaced-review scheduling and a command-line
interface.

[Unreleased]: https://github.com/Jondi-Studio-Public/maths-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Jondi-Studio-Public/maths-engine/releases/tag/v0.1.0
