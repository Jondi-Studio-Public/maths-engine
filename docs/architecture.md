# How the engine fits together

This page is the map. It says what each part of the repository does and how a question travels
from a skill file to a marked answer. The how-to detail lives elsewhere: `CONTRIBUTING.md` for
adding a generator or a skill, `mathgen/README.md` for writing generator code, `skills/README.md`
for the skill file format.

## The three layers

| Layer | Where | What it holds |
|---|---|---|
| Skills | `skills/<domain>/<id>.yaml` | What a learner practises: the description, the answer kind, and the generator topics (`sources`) that make its questions at each difficulty. Data only, CC BY 4.0. |
| Generators | `mathgen/generators/*.py` | Code that builds one kind of question from a seed. Each `@generator("topic", ...)` function registers a **topic key**. |
| Graders | `mathgen/grading/` | Code that decides whether a reply matches a question's answer, for each answer kind. |

A skill names topics, never modules or functions. That is the only link between the layers.

## How a question is made and marked

```
skills/secondary-number/simplify-surds.yaml
    sources: [{topic: t1_simplify_surds, difficulty: easy, params: {}}, ...]
        |
        v  mathgen.engine.make_for_skill(skill, seed)
mathgen.core.make("t1_simplify_surds", seed=seed, difficulty="easy", **params)
        |  looks the topic up in core.REGISTRY, seeds random.Random(f"{topic}:{difficulty}:{seed}")
        v
generator function  ->  dict  ->  Question(text, answer, answer_text, distractors, figure, parts, ...)
        |
        v  mathgen.engine.grade(skill, question, reply)
mathgen.grading.grade(kind, question, given, *, skill=None)  ->  GradeResult(correct, ...)
```

The same skill, source and seed always give the same question. `engine.question_id` names a
question as `<skill id>/<source key>/<seed>/<content hash>`, so a host can store attempts and
later rebuild the exact question with `engine.regenerate`. `tests/golden/questions.txt` pins
those ids for the whole library; see "Seed stability" in `CONTRIBUTING.md`.

## Where things live

| Path | Role |
|---|---|
| `mathgen/engine.py` | The public library API: `load_skills`, `make_for_skill`, `make_with_id`, `question_id`, `regenerate`, `grade`. Start here if you are integrating. |
| `mathgen/__main__.py` | The `mathgen` command line, a thin layer over `core` and `engine`. |
| `mathgen/core.py` | `Question`, the `@generator` decorator, `make()`, `REGISTRY`, and the shared helpers generators are built from. |
| `mathgen/answers.py` | The typed answer contract: answer kinds and forms, tagged distractors. |
| `mathgen/misconceptions.py` | The catalogue of misconception ids that distractors are tagged with. |
| `mathgen/skills.py` | Loads and validates skill files. |
| `mathgen/capabilities.py` | Reads `skills/capability-gates.yaml` and blocks skills whose gates have not passed. |
| `mathgen/figures.py`, `mathgen/figure_kinds/` | Figure specs for questions that show a picture. |
| `mathgen/slots.py`, `mathgen/text_quality.py` | Shared word-problem vocabulary, and checks on learner-facing text. |
| `mathgen/service.py` | Skills, questions, worksheets and grading as plain JSON-ready data: the layer the HTTP API, the MCP server and the exports share. |
| `mathgen/export.py`, `mathgen/svg.py` | Printable papers and answer keys, Moodle XML and QTI 2.1 banks; every figure kind drawn as SVG. |
| `mathgen/server/` | The HTTP API (the `server` extra); see `docs/server.md`. |
| `mathgen/mcp_server.py` | The MCP server for AI agents (the `mcp` extra, command `mathgen-mcp`); see `docs/mcp.md`. |
| `mathgen/scheduling/` | Optional spaced-review maths (FSRS, stages, prerequisites, calibration). No database or clock. |
| `skills/domains.yaml`, `skills/capability-gates.yaml`, `skills/skill.schema.yaml` | Domain bands, the gate registry, and the skill file schema. |
| `skills/index.yaml` | Generated map from every skill's topics to the module that registers them. |
| `scripts/` | Maintainer and CI tools (sweeps, golden questions, index, speed check, CI sharding). |
| `tests/public/` | Tests of the integrator-facing surface: CLI, engine API, packaging, skill loading. |
| `tests/test_*.py` | Everything else: per-generator tests, library-wide sweeps and contracts, grading. |

## Why generator file names look like `exd_x_f3_...` or `p19b_...`

The library was built in batches (`t1_` here is one), and each batch's modules kept its code as a prefix. The
prefixes are identifiers, not categories, and topic keys inherited many of them. Topic keys are
public (they appear in question ids and in skill files), so they are not renamed.

To find the code behind a skill or topic:

```
python scripts/build_skill_index.py --where simplify-surds     # every topic of the skill, with file and line
python scripts/build_skill_index.py --where t1_simplify_surds   # one topic
```

## Loading cost

`import mathgen` imports every generator module so that `REGISTRY` is complete. That takes about
two seconds; a long-running host pays it once at start-up. A generator that needs a large table builds it on its
first question, not at import.
