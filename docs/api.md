# API reference

This page lists the public API of `maths-engine` (import name `mathgen`). For installing and a first run, see
[README.md](../README.md). For how the parts fit together, see [architecture.md](architecture.md). For writing a
generator or a skill, see [CONTRIBUTING.md](../CONTRIBUTING.md) and `mathgen/README.md`.

## What is public

Public means: the names `mathgen/__init__.py` exports (`__all__`), and the names listed on this page in
`mathgen.engine`, `mathgen.skills`, `mathgen.grading`, `mathgen.scheduling` (`stages`, `prerequisites`, `review`,
`fsrs`, `calibration`), `mathgen.core` (the helpers for writing generators), the entry points of `mathgen.service`,
`mathgen.export` and `mathgen.svg` and the few names in `mathgen.answers` and `mathgen.figures` that generator
authors use. Everything else, and every name starting with `_`, is internal and may change in any release. Before
1.0 a minor release may change the public API (see [CHANGELOG.md](../CHANGELOG.md)).

Importing `mathgen` imports every generator module, which takes about two seconds. Python 3.12 or newer.

## Typical use

```python
from mathgen import engine

domains, skills = engine.load_skills()
by_id = {s.id: s for s in skills}
skill = by_id["simplify-surds"]

q = engine.make_for_skill(skill, seed=7)  # same skill, source and seed: same question
print(q.text)
print(q.answer_text)

print(engine.grade(skill, q, "6√21").correct)  # a reply, as a learner would type it
print(engine.grade(skill, q, "2√3").correct)

q, qid = engine.make_with_id(skill, seed=7)  # the question and its stable id
print(qid)
again = engine.regenerate(qid, by_id)  # rebuild it later from the id alone
print(again.text == q.text)

try:  # an id whose hash no longer matches
    engine.regenerate(qid[:-12] + "000000000000", by_id)
except engine.QuestionChanged as e:
    print("changed:", e.new_id == qid)
```

Output:

```
Simplify √756. Give your answer in the form a√b.
6*sqrt(21)
True
False
simplify-surds/easy/7/598b8fd09043
True
changed: True
```

## `mathgen`

| Name | What it is |
|---|---|
| `make`, `topics`, `Question`, `REGISTRY` | Re-exported from `mathgen.core`; documented there. |
| `__version__` | The installed distribution's version, or `"0+unknown"` if the package was never installed. |

`mathgen` also attaches a `NullHandler` to the `"mathgen"` logger, so the library prints nothing unless your
application configures logging.

## `mathgen.engine`

The API for integrating: load skills, make a question for a skill, grade a reply, name and rebuild a question.
It touches no database or network. A `source` argument below is a `Source`, an index into `skill.sources`, a
source key, or a difficulty label (see `pick_source`).

| Signature | Description |
|---|---|
| `load_skills(root=None, extra_roots=None) -> (list[Domain], list[Skill])` | Re-exported from `mathgen.skills`. |
| `make_for_skill(skill, seed=None, *, source=None)` | Build one verified `Question` for the skill. With no `source`, uses the first. `seed=None` picks a random seed. Raises `ValueError` if the skill has no sources and `KeyError` for an unknown `source`. |
| `make_with_id(skill, seed, *, source=None)` | Returns `(question, question_id)`. The same question as `make_for_skill`. `seed` must be a whole number, or `None` for a random one (the id then carries the seed actually used); anything else raises `TypeError`. |
| `grade(skill, question, given: str \| dict) -> GradeResult` | Grades `given` with the grader the skill's `answer` kind names. A multi-part question is marked part by part and `given` may be a dict keyed by part label. For an `mcq` skill, `given` is the 0-based index (an `int` or a string of digits) into `mathgen.core.mcq_options(question)`. |
| `grade_with_deadline(skill, question, given, *, timeout=None) -> GradeResult` | `grade` in a worker process stopped after `timeout` seconds (2 by default); a reply that runs out of time is graded wrong. **Use this, not `grade`, for replies from people you do not trust.** `grade` runs in your process with no time limit, and a crafted reply can keep SymPy busy for minutes. Concurrent calls are queued behind the deadline's single worker, so budget for that on a busy service. The HTTP server and MCP tools already do this. |
| `question_id(skill, source, seed, question) -> str` | `<skill id>/<source key>/<seed>/<hash>`. The first three parts say how to rebuild the question; the hash pins its content. |
| `content_hash(question) -> str` | First 12 hex characters of SHA-256 over `question.to_dict()` (sorted keys, compact JSON). |
| `parse_question_id(qid) -> (skill_id, source_key, seed, hash)` | Splits an id. Raises `ValueError` if malformed. |
| `regenerate(qid, skills)` | Rebuilds the question an id names. `skills` is a list of `Skill` or a dict by id. Raises `KeyError` (unknown skill or source key), `ValueError` (malformed id) and `QuestionChanged` when the content no longer matches. |
| `pick_source(skill, source=None) -> Source` | Resolves a `source` argument. A difficulty label shared by several sources picks the first. Raises `ValueError` (no sources) or `KeyError` (no match; the message lists the valid labels and keys). |
| `QuestionChanged(old_id, new_id, question)` | Exception raised by `regenerate`. Attributes `old_id`, `new_id` and `question` (what the library now produces). Do not mark an old attempt against `question`. |

`engine` also re-exports `Skill`, `Source`, `Domain`, `GradeResult`, `display_question` and `paper_notation`.

`GradeResult` (a dataclass) has `correct: bool`, `canonical: str`, `given_normalised`, and for a wrong reply
that matches a known error `misconception`, `matched_distractor`, `note` and `note_kind` (`"misconception"`,
`"near_miss"` or `"unreadable_parts"`). Table answers set `cells`. Multi-part results set `marks` (a
`Fraction`), `max_marks`, `parts` (a result per part label) and, on a part, `follow_through`.

## `mathgen.skills`

Loads and validates the skill YAML files. The library ships inside the package. `MATHS_SKILLS_DIR` replaces
the folder; `MATHS_EXTRA_SKILLS_DIRS` (folders separated by `os.pathsep`) adds skills of your own.

| Signature | Description |
|---|---|
| `load_skills(root=None, extra_roots=None) -> (list[Domain], list[Skill])` | Loads every `<domain>/<id>.yaml` under `root` plus `domains.yaml`, then the skills in `extra_roots` (default `extra_skills_roots()`; pass `[]` for none). Cached, invalidated by any file change; each call returns its own deep copy. Raises `SkillsNotFoundError` for a missing or empty folder and `DuplicateSkillError` for an extra skill whose id is taken. |
| `validate(domains, skills, graders=None, gate_registry=None, rubric_loader=None) -> ValidationReport` | Checks skills against the live grader registry and capability gates. `rubric_loader(id)` resolves rubric ids for `answer: rubric` skills and raises `FileNotFoundError` or `ValueError` for a bad one; without it the id is required but not resolved. |
| `load_and_validate(root=None, rubric_loader=None, extra_roots=None) -> (domains, skills, report)` | `load_skills` then `validate`. |
| `extra_skills_roots() -> list[Path]` | The folders named by `MATHS_EXTRA_SKILLS_DIRS`, `~` expanded. |
| `skills_root() -> Path` | The main skills folder (`MATHS_SKILLS_DIR` or the default). |
| `SkillsNotFoundError(root)` | `FileNotFoundError` subclass; has `.root`. |
| `DuplicateSkillError` | `ValueError` subclass. |

Data classes (fields are read-only by convention; see `skills/README.md` for the YAML format):

- `Skill`: `id`, `name`, `domain`, `description`, `status` (`"active"` or `"deferred"`), `deferred_reason`,
  `prerequisites`, `answer` (the grader kind, a key of `GRADERS`), `sources`, `misconceptions`, `near_misses`,
  `tags`, and several optional fields for coverage and placement.
- `Source`: `topic`, `params`, `difficulty`, `stages`, `key`, and the property `effective_key`
  (`key` if set, else `difficulty`).
- `Domain`: `id`, `name`, `order`, `band`, `band_name`.
- `ValidationReport`: `errors`, `total`, `active`, `deferred`, the property `ok`, and `by_status()`.

## `mathgen.core`

The question type, the registry, and the helpers generators are built from.

### Making questions

| Signature | Description |
|---|---|
| `make(topic, seed=None, difficulty=None, **params) -> Question` | Builds one verified question. `difficulty` selects a named preset; explicit `params` override it. Same `(topic, seed, difficulty)` gives the same question on any machine. Raises `KeyError` for an unknown topic (with close matches) or a preset the topic lacks. |
| `topics() -> list[(level, topic)]` | Registered topics, sorted by level. |
| `REGISTRY` | `dict` of topic to `(level, function)`. Read it; add to it only with `@generator`. |
| `mcq_options(question) -> list[str]` | `[answer_text, *distractors]`, shuffled by the question's seed. The order `engine.grade` indexes for `mcq`. |
| `display_question(question) -> dict` | Keys `text`, `answer_text`, `options`, `distractors`, `working`, written as on paper; plus `parts` for a multi-part question. The question itself is unchanged. |
| `paper_notation(text)` | Rewrites Python spelling for display: `x**2` to `x^2`, `2*x` to `2x`, `sqrt(21)` to `√21`. Text with none of these is returned unchanged. |

`Question` is a dataclass. The fields you will use: `topic`, `level`, `seed`, `difficulty`, `text` and
`answer_text` (plain text, Python-style), `latex` and `answer_latex` (for rendering), `answer` (structured),
`working` (worked steps), `distractors` (wrong options, formatted like `answer_text`), `figure` (a picture
spec or `None`), `parts` and `stem` (multi-part questions). It has `to_dict()` and `to_json()`. `Part` is one
part of a multi-part question (`label`, `prompt`, `answer_kind`, `answer`, `answer_text`, `marks`,
`follow_through`, `sensible_range`, and the answer attributes `Question` has).

### Writing generators

| Signature | Description |
|---|---|
| `generator(topic, level, presets=None)` | Decorator. Registers a function `f(rng, **params) -> dict`. Raises `ValueError` on a duplicate topic. |
| `check(cond, msg="answer verification failed")` | Raises `VerificationError` if `cond` is false. |
| `check_equivalent(actual, expected, msg=...)` | Raises `VerificationError` unless the two expressions are equal. Use it to confirm an answer derived a second way. |
| `check_distinct(answer, *candidates, msg=...)` | Raises unless every candidate differs from `answer`. |
| `check_pairwise_distinct(*values, msg=...)` | Raises unless every pair differs. `pairwise_distinct(*values) -> bool` is the non-raising form. |
| `symbolically_equal(left, right) -> bool` | Exact equivalence of two SymPy-able values; sequences compare by element. |
| `pick_distractors(answer_text, candidates, n=3, *, unordered=False, tolerance=0) -> list[str]` | Drops candidates that repeat or equal the answer in value, not only in spelling; keeps order, returns at most `n`. Pass `GRADER_TOLERANCE` for numeric graders. |
| `GRADER_TOLERANCE` | `Fraction(51, 1000)`, the tolerance of the spoken-answer and fallback numeric paths. |
| `stated_precision_tolerance(dp) -> Fraction` | Half a unit in the `dp`-th decimal place. Raises `ValueError` if `dp < 0`. |
| `attempts(limit=1000, what=...)` | Yields `0..limit-1`, then raises `VerificationError`. A bounded replacement for `while True:`. |
| `draw_until(rng, draw, accept, *, tries=1000, what=...)` | Returns the first `draw(rng)` that `accept` allows; raises `VerificationError` after `tries`. |
| `phrase(rng, patterns, **slots) -> str` | Picks one `str.format` template with `rng` and fills it. |
| `compose(rng, parts, *, final) -> dict` | Chains sub-steps; each `part(rng, values) -> (value, working)`; `final(values, rng)` returns the generator dict. Every part must return a working line. |
| `fmt_signed(coef, var="")`, `paren_negative(value)`, `join_terms(terms)`, `rational_text(r)` | Text helpers for signs, brackets, sums of `(coef, var)` terms, and fractions. |
| `latex_answer(value, *, evaluate=True, plain=False)`, `latex_prompt(text)` | LaTeX for an answer or a prompt. `make()` fills `latex` from `latex_prompt` when a generator leaves it empty. |
| `numeric_answer(value)`, `stated_precision_string(value, d)`, `round_half_up(value, places=0)`, `exact_decimal_string(value, max_places)`, `rounded_decimal_string(value, max_places)`, `fixed_point_string(scaled_int, places)` | Decimal answers at a stated precision; take `Fraction` values. `stated_precision_string` raises `VerificationError` if the value is off the grid. |

`VerificationError` is the exception a generator raises when its own check fails.

Related, outside `core`:

- `mathgen.answers.Distractor(text, misconception=None, value=None, credit=None)`: a frozen dataclass for a
  distractor tagged with the error it stands for. `credit` must be at least 0 and below 1, or `ValueError`.
  `mathgen.answers.answer_spec(obj, kind=None)` returns the typed answer of a question.
- `mathgen.figures.register_reader(topic, reader)`: registers a function that recomputes a figure question's
  answer from `question.figure` alone. `mathgen.figures.check(spec)` returns the spec or raises `ValueError`.

## `mathgen.grading`

Entry points only; the individual `grade_*` functions are internal.

| Signature | Description |
|---|---|
| `grade(kind, question, given, *, skill=None) -> GradeResult` | Grades a reply with the grader named by `kind`. If the question has `parts`, grades part by part whatever `kind` is. A reply over 100,000 characters is wrong (the HTTP server and MCP tool refuse one over 2,000 characters before it gets here). If a grader raises, the error is logged and the reply is graded incorrect. A wrong reply that matches a tagged distractor sets `misconception`; `skill` adds that skill's own notes. Raises `KeyError` for an unknown `kind`. |
| `GRADERS` | `dict` of answer kind to grader. Kinds: `column_basis` (any independent columns of a matrix), `expression`, `fraction`, `grid_cells`, `inequality`, `line`, `list`, `matrix`, `mcq`, `number`, `ordered_list`, `pair`, `points`, `polyline`, `set`, `table`, plus `open_response` (never marked: returns a pending result for hand-off) and `step_chain` (typed proof lines). |
| `GradeResult` | See `mathgen.engine`. |
| `mcq_options(question)` | Same function as `mathgen.core.mcq_options`. |
| `spoken.check_answer(question, transcript, fallback=None) -> bool` | Checks a transcript of a spoken answer ("one hundred and eighteen") against `question.answer`. `fallback` is a second transcript tried only if the first fails. A single-number answer must be stated once, not negated or hedged ("not twelve", "twelve or thirteen" are wrong for 12); a value matches within ±0.051 when the key has 0 or 1 decimal places, and within ±0.005 (half a hundredth) when the key is shown to 2 or more places or is a non-terminating fraction below one half, so "zero" is wrong for 0.03. A pair or list answer must state exactly one value per component, none negated or hedged, in order for a pair or an `ordered_list` and in any order for other lists. Ordinal denominators up to "thousandths" are read as fractions ("one fiftieth", "three hundredths"). |

The hand-off for open answers (marking request, marking result, validation, self-marking, prompt text) is in `mathgen.grading.open_response`; see `docs/design/open-response-handoff.md`.

`mathgen.grading.typed` holds the graders themselves. From it, `typed.normalise_reply(kind, given)` and
`typed.grade` are the documented entry points; `grade` is the same function as `mathgen.grading.grade`.

## `mathgen.scheduling`

Spaced-review maths for applications. It holds no learner data and reads no clock: you store state and pass
times in. `mathgen.scheduling` itself exports nothing; import the submodules.

### `mathgen.scheduling.stages`

A review ladder has stages `0..n_stages-1`, then maintenance (stage `n_stages`). A source's `stages` lists the
stages it serves (ints, `"maintenance"`, or `"exam"`); none means every stage.

| Signature | Description |
|---|---|
| `eligible_sources(skill, stage, n_stages) -> list[Source]` | Non-exam sources a review at `stage` draws from. If none names the stage, uses the nearest stage that has some (easier side first). `[]` only if the skill has no non-exam source. |
| `pick_sources(sources, need, start=0) -> list` | `need` sources round-robin from index `start` (wrapped). |
| `non_exam_sources(skill) -> list` | `skill.sources` without exam sources. |
| `is_exam_source(src) -> bool` | True for a source with `stages: [exam]`. |
| `MAINTENANCE`, `EXAM` | The string constants `"maintenance"` and `"exam"`. |

### `mathgen.scheduling.prerequisites`

Prerequisites are advice; nothing here stops you serving any skill. `skills` is a list or a dict by id.

| Signature | Description |
|---|---|
| `prerequisite_closure(skill_id, skills) -> set[str]` | Everything the skill builds on, directly or not, excluding itself. Raises `KeyError` for an unknown id. |
| `ready_to_start(skills, mastered) -> list[Skill]` | Active, unmastered skills whose direct prerequisites are all mastered. |
| `gaps_before(skill_id, skills, mastered) -> list[str]` | Active unmastered skills in the skill's closure, earliest first (ties by id). |

### `mathgen.scheduling.fsrs`

The FSRS-5 memory model, matching py-fsrs 5.1.3: elapsed time counts in whole days, rounded down. Ratings are
`AGAIN, HARD, GOOD, EASY = 1, 2, 3, 4`. Also `DEFAULT_WEIGHTS`, a 19-number tuple.

| Signature | Description |
|---|---|
| `Memory(stability, difficulty)` | Frozen dataclass. Store the one `step` returns. |
| `step(memory, rating, elapsed_days, w=DEFAULT_WEIGHTS) -> (Memory, retrievability \| None)` | Applies one rated review. `memory=None` is a first review; its retrievability is `None`. Raises `ValueError` for a rating outside 1..4 (a bool is not a rating). |
| `retrievability(elapsed_days, stability) -> float` | Predicted probability of recall. Raises `ValueError` if `stability <= 0`. |
| `interval_days(stability, desired_retention) -> float` | Days until recall falls to `desired_retention` (unrounded). Raises `ValueError` unless `0 < desired_retention < 1`. |
| `initial_stability(rating, w=...)`, `initial_difficulty(rating, w=...)`, `next_difficulty(d, rating, w=...)` | The model's building blocks. |

### `mathgen.scheduling.review`

Turns a finished review into an outcome, an FSRS rating, memory and a due date. Outcomes: `"advance"`,
`"maintenance_pass"` (no misses), `"hold"`, `"reset"`. A reset is rated Again, a hold Hard, the others Good.

| Signature | Description |
|---|---|
| `outcome_from_marks(results, *, maintenance=False, reset_misses=2) -> str` | Full marks is no miss, at least half the marks half a miss, less a whole miss. No misses advances (`maintenance_pass` if `maintenance`); misses of `reset_misses` or more reset; otherwise hold. `results` are `GradeResult`s or `(earned, available)` pairs. Raises `ValueError` if empty, if `reset_misses < 1`, or for impossible marks. |
| `score(outcome, memory, last_review_at, now, *, desired_retention=0.9, maximum_interval_days=365, reset_gap_days=1.0) -> dict` | Returns `memory`, `rating`, `retrievability`, `elapsed_days`, `interval_days`, `due_at`. A reset is due after `reset_gap_days`; others when recall falls to `desired_retention`, in whole days from 1 to the maximum. A hold is never due later than the whole days since the last review. Raises `ValueError` if `reset_gap_days < 0`, `maximum_interval_days < 1` or `outcome` is not one of `OUTCOME_RATING`'s keys. A `last_review_at` after `now` counts as no time elapsed. |
| `memory_from_history(log_rows, stage, stage_delays) -> (Memory \| None, datetime \| None)` | Rebuilds memory by replaying rows with `outcome` and `created_at` (oldest first). With no usable history and `stage > 0`, seeds stability from `stage_delays[stage - 1]`. |

Also `OUTCOME_RATING` and the `DEFAULT_*` constants (`DEFAULT_DESIRED_RETENTION`, `DEFAULT_MAXIMUM_INTERVAL_DAYS`,
`DEFAULT_RESET_MISSES`, `DEFAULT_RESET_GAP_DAYS`).

### `mathgen.scheduling.calibration`

Two read-only measurements over plain row dicts you supply.

| Signature | Description |
|---|---|
| `item_difficulty(attempt_rows, skills, min_attempts=20) -> dict` | Rasch difficulty per `(skill, tier)` from rows with `user_id`, `skill_id`, `difficulty`, `correct`. `skills` maps id to `Skill`. Returns `skills`, `items`, `inversions` (tiers answered right more often than an earlier tier), `learners`, `attempts`. |
| `fsrs_calibration(log_rows, bins=10) -> dict` | Predicted against observed recall from rows with `retrievability` (`None` is skipped; otherwise 0 to 1, or `ValueError`) and `rating` (1 means forgot). `bins` is a whole number, 1 or more. Returns `reviews`, `log_loss`, `rmse_bins`, `mean_predicted`, `observed`, `bins`. |

## Services, exports and figures

These modules build on `engine` for platforms that serve questions to people. Their guides have the details:
[server.md](server.md) for the HTTP API and [mcp.md](mcp.md) for the MCP server.

| Module | Entry points |
|---|---|
| `mathgen.service` | `Library.load(root=None, extra_roots=None, *, id_secret=None)`, then `find_skills`, `make_questions`, `question_by_id`, `build_paper` and `grade_reply`. Everything returned is JSON-ready, answers are left out unless asked for (`answers=True`), and grading rebuilds the question from its id. Raises `service.NotFound` and `service.BadRequest`. |
| `mathgen.export` | `paper_html(paper, *, version=None, key=False) -> str` (printable A4 HTML), `paper_moodle_xml(paper, *, version="A") -> str`, `paper_qti21_zip(paper, *, version="A") -> bytes`, each taking a paper from `service.build_paper`. |
| `mathgen.svg` | `figure_svg(spec, *, width=400) -> str` draws a figure spec as a standalone SVG with its alt text as the title; `rendered_kinds()` lists the figure kinds it draws. |
| `mathgen.server` | The HTTP API: `pip install "maths-engine[server]"`, then `python -m mathgen.server`. |
| `mathgen.mcp_server` | The MCP server: `pip install "maths-engine[mcp]"`, then `mathgen-mcp`. |
