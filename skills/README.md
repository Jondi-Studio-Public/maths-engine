# skills/

One skill is one YAML file at `skills/<domain>/<id>.yaml`. Nothing else lists
skills: the loader (`mathgen/skills.py`) scans this tree.

## Licensing

Skill YAML is licensed CC BY 4.0 (`LICENSE-CONTENT`); the code is MIT
(`LICENSE`). The `senior-*` skills adapt Australian Curriculum content
descriptors, which ACARA licenses under CC BY 4.0: their `name`, and in many
files their `description`, are modified descriptor text, and each such file
starts with a comment saying so. F-10 skills (`f10-*`) use ACARA codes as tags;
their names and descriptions are this project's own wording, though they may
share ordinary phrases with the curriculum. `NOTICE` has the required
attribution and lists the edits.

## Layout

```
skills/
  domains.yaml           ordered list of domains, grouped into bands
  capability-gates.yaml  the capability gate registry
  <domain>/
    <id>.yaml            one skill
```

## Fields

```yaml
id: simplify-surds              # equal to the file name without .yaml
name: Simplify surds
domain: secondary-number        # equal to the folder name, listed in domains.yaml
description: One sentence naming the observable operation.
status: active                  # active or deferred
prerequisites: [prime-factorisation]   # skill ids; no cycles (rules below)
answer: expression              # which grader checks replies (below)
sources:
- topic: t1_simplify_surds      # a generator topic name
  params: {}                    # keyword arguments for mathgen.make(topic, **params)
  difficulty: easy              # tier: easy, medium, hard or exam
  stages: [0, 1, 2]             # optional; None means all stages
misconceptions:
- id: drops-the-radical
  description: States only the extracted coefficient and drops the remaining root.
tags: []
```

`skill.schema.yaml` beside this file is the same list as a JSON Schema (draft 2020-12): every field,
which are required, the allowed labels, and the conditions below. An unknown key is an error at every
level, so a misspelt field is caught instead of silently ignored. To check a file in VS Code with the
YAML extension, put `# yaml-language-server: $schema=../skill.schema.yaml` on its first line, or map the
glob `skills/*/*.yaml` to the schema in settings. `tests/public/test_skill_schema.py` checks the whole
library against it and keeps its field lists equal to what `mathgen/skills.py` reads. The schema covers
shape only: it cannot tell whether a topic is registered or a source generates, which `validate()` does.
Quote `verified_on` dates (`'2026-10-01'`) so YAML does not turn them into dates.

- **Required in practice:** `id`, `name`, `domain`, `status`, `answer`, and at least one
  entry in `sources` for an active skill. `description` is one sentence.
- **`prerequisites`:** the skills a learner normally has first: the direct ones, not every ancestor.
  They are advice for ordering and review, never a gate. An active skill may only name active skills,
  since a host that waited for a deferred one would never offer it. An active skill in a senior or
  university domain must name at least one; `validate()` checks both. In this library every other
  active skill also names one or is named by one (`tests/test_prerequisite_graph.py`): only the
  genuine starting points, such as counting to 5, have none, and later skills name them.
- **`sources`:** `topic` must be a registered generator topic (see `python -m mathgen --list`).
  `params` are passed to the generator, whose signature is the authority on which keyword
  arguments exist. `difficulty` is the tier the source stands for, one of `easy`, `medium`, `hard` or `exam`, and an
  active skill's every source must have one (`validate()` reports a missing or other label). It is also
  passed to the generator as a preset name, and it is hashed into the random seed, so changing a
  source's `difficulty` changes which question each seed gives. A generator that has presets must
  register one of each tier name it is labelled with; explicit `params` win over a preset's values.
  `stages` entries come from `0` to `6`, `maintenance` and `exam`.
- **`answer`:** one of the kinds in `GRADERS` (`mathgen/grading/typed.py`): `number`,
  `fraction`, `pair`, `list`, `ordered_list`, `expression`, `mcq`, `table`, `set`, `points`,
  `polyline`, `line`, `inequality`, `matrix`. `list` ignores order and `ordered_list` is positional, so a vector answer
  is an `ordered_list` (it also reads `<5, 6, -5>` and `5i + 6j - 5k`). A `number` reply may group
  thousands with a space (`12 345`), may be a mixed number (`2 1/2`, `2½`) and may use scientific
  notation written `1.2 × 10^3`; calculator notation (`1.2e3`) is refused. A question with a single
  `number` answer asks for one rounding, not two (`tests/test_rounding_asked_once.py`). A `fraction`
  reply may be bracketed, written `\frac{a}{b}` or with another fraction bar (`⁄`, `÷`); when the
  question says "lowest terms", "simplest form" or "simplify", a written a/b must be reduced. A `pair`
  reply's labels must be the answer's own (in any case, without a subscript, or spelled out).
  `grade_expression` accepts a leading `name =`, a derivative label (`f'(x) =`, `dy/dx =`) or
  `Answer:`, and reads `÷`. An `mcq` reply is an option's index, its letter or its whole text; digits
  are always an index. `inequality` (`mathgen/grading/inequality.py`) compares solution sets, so
  `3 < x` and `(3, ∞)` are both `x > 3`. `matrix` (`mathgen/grading/matrix.py`) compares exact
  entries; its `answer_form` `scalar_multiple` or `column_scalar_multiple` accepts any non-zero
  multiple of the answer, or of each of its columns, for answers such as eigenvectors.
- **`misconceptions` and `near_misses`:** both are lists of `id` plus `description`. A
  misconception is wrong method ("subtracts in the wrong order"). A near miss is sound
  reasoning aimed at a different question than the one asked. Ids must not collide across the
  two lists on one skill. A generator links an option to a note by putting the option text and
  the id in `Question.option_notes`.
- **Optional:** `placement_eligible` (default `true`), `placement_order`, `placement_questions`,
  `placement_pass`, `learner_band` (`early`, `primary` or `general`), `framework_areas`, `alignments`,
  `needs_figure`, `capability_gates` and `mental` (below).
- **`alignments`:** entries of `source_id`, `status`, `code`, `framework`, `area`, `verified_on`.
  `status` is `verified`, `candidate` or `unverified`. Only `verified` is a formal claim, and it
  requires `verified_on`. A `framework` entry requires an `area`.

## Status and deferral

A skill is `active` only if every entry in `sources` generates successfully across a range of
seeds and its `answer` kind has a grader. Otherwise it is `deferred` with a `deferred_reason`
from this set: `generator` (the topic or params do not exist yet), `grader` (the answer kind
cannot be graded yet), `hard`, `visual`, `assessment_contract`, `formal_verification`,
`safety_gate`. To activate a deferred skill, confirm what blocked it now exists, set
`status: active`, delete `deferred_reason`, and run the validator.

## Capability gates

`capability_gates` lists ids from `capability-gates.yaml`. Each gate has a `status`
(`not_benchmarked`, `benchmarking`, `passed` or `failed`) and is fail-closed: an unknown gate is
a validation error, and an active skill whose gates have not all passed is rejected. Skills that
need a gate which has not passed stay `deferred`.

A skill whose questions draw a figure must set `needs_figure: true` (the validator checks this) and normally cites the
`figure_stimulus` gate. Such skills may not carry a `mental` block, since a figure has no spoken form.

## Difficulty progression

Give a skill several sources, one per tier, and make the tiers differ in reasoning, not in
range. A hard tier that only uses bigger numbers is not a harder skill. Good escalations add a
step, a new question shape, a sign or form the easier tier never shows, or a situation that needs
a different method. Tune a tier by editing `params`, then check the questions across many seeds.

Write the escalation down as `difficulty_progression`, one entry per tier the sources use:

```yaml
difficulty_progression:
  easy: {steps: 1, demand: Substitute the term number into an explicit linear rule.}
  medium: {steps: 2, demand: Square the term number, then combine the terms in order.}
  hard: {steps: 4, demand: Apply a recursive rule term after term from the first.}
```

`steps` is the number of reasoning steps a learner must chain, and `demand` names them.
`validate()` rejects a tier the sources do not use, a used tier left out, and a harder tier
that asks fewer steps than an easier one. Older skills carry the same thing as one paragraph of
prose, which is accepted and checked by nothing.

## The `mental` block

A skill suited to quick no-paper practice can carry a `mental` block beside `sources`:

```yaml
mental:
  levels:                  # 1 to 5 entries, easiest first
  - topic: addition
    params: {lo: 1, hi: 20}
    target_s: 3            # positive seconds
    limit_s: 8             # at least target_s
  entry_stage: 2           # optional
```

Each level is a `topic`, `params`, optional `difficulty`, and the two timings. Leave the block
off for anything that needs working shown, has more than one unknown, or can have a long answer.

## Adding a skill

1. Pick a domain folder from `domains.yaml`, or add a domain there.
2. Make sure a generator exists for each source (see `mathgen/README.md`), or write one.
3. Create `skills/<domain>/<id>.yaml` with the fields above.
4. If the name or description adapts Australian Curriculum Senior Secondary text, the id
   starts with `senior-` and the file starts with the same two comment lines as every other
   `senior-*` file (copy them from any of those). `tests/public/test_skill_headers.py` checks
   this, and `NOTICE` explains it.
5. Check it loads and generates:

```
python -m mathgen --skills | grep <id>
python -m mathgen --skill <id> --seed 1
python -m mathgen --skill <id> --seed 1 --answer "<a reply>"
python scripts/regrade_sweep.py --ids <id>
python scripts/render_skill_questions.py --ids <id> --out sample.md
```

   `regrade_sweep.py` generates many questions and confirms the grader accepts the canonical
   answer and rejects every distractor. `render_skill_questions.py` writes the questions out so
   you can read what a learner would see.

To run the validator itself:

```python
from mathgen.skills import load_and_validate

domains, skills, report = load_and_validate()
print(report.errors)
```

It checks ids, domains, prerequisites, cycles, answer kinds, gates, stages and that each active
source generates for a few seeds. `MATHS_SKILLS_DIR` points the loader at a different folder, and
`MATHS_EXTRA_SKILLS_DIRS` (or `load_skills(extra_roots=...)`) adds skills from your own folders, laid
out as `<domain>/<id>.yaml`; they join the library's domains, and loading
refuses one that reuses an id already loaded. Entries are split on `os.pathsep` (`;` on Windows).

## Grading facts that shape a skill

- Typed numeric answers are graded to their own stated precision. A rounded answer accepts half a
  unit in its last place; an answer shown exactly is matched exactly, so a question whose stored
  answer is rounded must say so ("to the nearest cent").
- A number reply may give a unit. It must be the unit the canonical shows ("F = 25.00 N") or the one the
  question asks for by name in its asking sentence ("Convert 3 cm to mm", "in km/h"), so name it there.
- `grade_fraction` compares by value, so `2/4` is accepted for `1/2`. Never build a
  "does not simplify" distractor.
- Every distractor and every deliberately wrong test value must clear the grader. Check every
  candidate, not a sample.
