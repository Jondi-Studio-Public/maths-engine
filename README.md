# maths-engine

Maths question generators with verified answers, the graders that check
replies to them, and a library of skill definitions. Pure Python, no website,
no server, no model. The Python package and the command are called `mathgen`.

Every question is built from a seed. The generator computes the answer, checks
it a second way, and builds wrong options that the grader is guaranteed to
reject. So you can generate as many questions as you like and trust the
answer key.

This repository contains code, skill definitions and tests only. It does not
contain a website, curriculum framework data, rubrics or learner data.

## Install

Python 3.12 or newer.

```
git clone https://github.com/Jondi-Studio-Public/maths-engine.git
cd maths-engine
python3 -m venv .venv
. .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Dependencies are SymPy, NumPy and PyYAML. For development add the `dev` extra:
`pip install -e ".[dev]"`.

A plain `pip install .` works too. The built package carries the skill library, so
`mathgen` finds it from any folder. With the editable install above (or when you run Python from
the clone's root), the library is read in place from `skills/`, so edits show up straight away. A
plain install holds the library as it was when you installed it, so reinstall after editing `skills/`.
`MATHS_SKILLS_DIR` points the loader at a different folder either way, and `MATHS_EXTRA_SKILLS_DIRS`
(folders separated by `os.pathsep`: `:`, or `;` on Windows; `~` is expanded) adds skills of your own from folders laid out the same way.
An extra skill joins the library's domains; loading refuses one that reuses an id already loaded.

## Quick start

```
python -m mathgen --skills                 # list every skill: domain/id and active or deferred
python -m mathgen --skill simplify-surds --seed 7
python -m mathgen --skill simplify-surds --seed 7 --answer "6√21"
python -m mathgen --skill simplify-surds --seed 7 --answer "2√3"
```

Output, in the same order (the first command prints one line per skill, only the start is shown):

```
abstract-algebra/cyclic-group-element-order  active
abstract-algebra/integer-polynomial-quotient-remainder  active
abstract-algebra/permutation-composition  active
...

[simplify-surds #7]
  Q: Simplify √756. Give your answer in the form a√b.
  A: 6*sqrt(21)

[simplify-surds #7]
  Q: Simplify √756. Give your answer in the form a√b.
  A: 6*sqrt(21)
  given '6√21': correct

[simplify-surds #7]
  Q: Simplify √756. Give your answer in the form a√b.
  A: 6*sqrt(21)
  given '2√3': wrong
```

The second command prints one question and its answer. The third and fourth also
grade a reply and print `correct` or `wrong`. The same seed always gives the same
question: seed 7 is always √756. Seed 145 gives √12, whose answer is 2√3. A reply
may be written the way you would type it: `6√21`, `6*sqrt(21)` and `6 sqrt(21)` are
all correct.

A multiple-choice skill prints its options with labels. Answer with a label or with
the option's text:

```
python -m mathgen --skill add-within-10 --seed 1 --answer B
```

```
[add-within-10 #1]
  Q: 3 + 5 = ?
     A) 10
     B) 8
     C) 9
     D) 7
  A: B) 8
  given 'B': correct
```

A skill marked `deferred` is defined but not ready for practice. Some still make
questions (`--skill` notes on stderr that the skill is deferred); one with no
question sources, or no grader for its kind of answer, exits with an error that
says what is missing.

Other useful forms:

```
python -m mathgen --list                   # generator topics by level
python -m mathgen --topic quadratic -n 5 --seed 42 --pretty
python -m mathgen --level 9 -n 100 --out questions.jsonl    # prints: wrote 3100 questions to questions.jsonl
```

The `--topic` command prints, for its first question:

```
[L9 quadratic #42]
  Q: Solve x**2 - 5*x - 36 = 0
  A: x = -4 or x = 9
  wrong: x = -9 or x = 4 | x = 4 or x = 9 | x = -9 or x = -4
```

From Python:

```python
from mathgen import engine

skills = engine.load_skills()[1]
skill = next(s for s in skills if s.id == "simplify-surds")
q = engine.make_for_skill(skill, seed=7)
print(q.text, q.answer_text)
print(engine.grade(skill, q, "6√21").correct)
print(engine.grade(skill, q, "2√3").correct)
```

```
Simplify √756. Give your answer in the form a√b. 6*sqrt(21)
True
False
```

`text` and `answer_text` are plain text in Python-style syntax (`x**2`,
`6*sqrt(21)`); `latex` and `answer_latex` are there for rendering. For a
multiple-choice skill (`skill.answer == "mcq"`), `engine.grade` takes the 0-based
index of the chosen option in `mathgen.core.mcq_options(q)`; the command line
does that translation for you.

## Showing a question on a page

`engine.present(skill, q)` returns everything a page shows, as JSON-able data:

```python
page = engine.present(skill, q)
page["prompt"]   # {"text": "Simplify √756. Give your answer in the form a√b.", "tex": None}
page["answer"]   # {"text": "6√21", "tex": "\\(6 \\sqrt{21}\\)"}
page["figure"]   # None, or {"svg": "<svg ...>", "alt": "...", "spec": {...}}
page["options"]  # multiple choice only: one pair per option, in mcq_options order
page["working"]  # the worked solution, one pair per step
```

Every string comes as a pair. `text` is plain Unicode written as on paper
(`x^2`, `2x`, `√21`), for pages without maths rendering, screen readers and
exports. `tex` is the same content with the maths wrapped in `\\(...\\)` and
the prose left as text, or `None` when there is nothing to typeset. Hand it to
KaTeX's auto-render extension or MathJax with the `\\(...\\)` delimiter only
(not `$`, which appears in prose about money). Because the prose is not inside
the maths, a long question wraps on a phone.

Figures come as standalone SVG with no scripts or external files, themeable
through CSS variables, with the alt text as the SVG's title. To draw them
yourself instead, `figure["spec"]` follows a JSON Schema (`python -m mathgen --figure-schema`); see
[docs/figures.md](docs/figures.md). From the command line:

```
python -m mathgen --skill simplify-surds --seed 7 --present      # the page as JSON
python -m mathgen --skill tell-time-to-hour --seed 1 --svg clock.svg
```

## Layout

| Path | What it is |
|---|---|
| `mathgen/` | The package: `core.py` shared helpers, `figures.py`, `slots.py`, `engine.py`, `present.py`, `skills.py` loader and validator |
| `mathgen/render/` | Draws figure specs as standalone SVG |
| `mathgen/generators/` | The question generators, one decorated function each |
| `mathgen/grading/` | Typed-answer graders: number, fraction, expression, list, multiple choice and others |
| `mathgen/scheduling/` | Spaced review: which sources serve each review stage, the FSRS memory model, and calibration from attempt data |
| `skills/<domain>/<id>.yaml` | One skill definition per file |
| `scripts/` | Speed check, regrade sweep and other build tooling |
| `tests/` | The pytest suite. `tests/public/` is a small, fast set to start with |

Generator file names carry historical batch prefixes (such as `p19h_`). They
are identifiers, not categories.

## How determinism and grading work

**Determinism.** A generator takes a seeded random source and nothing else.
The same skill, source and seed always give the same question text, answer and
options, on any machine. That makes failures reproducible and lets tests sweep
many seeds.

**Grading.** The grader that accepts an answer lives next to the generator
that produced it. Typed numeric answers are graded to their own stated
precision. Fractions compare by value, so equivalent forms are accepted.
Expressions are compared by symbolic equivalence. Each generator's tests grade
its own answer as correct and every distractor as wrong, for every candidate.
`scripts/regrade_sweep.py` re-checks this across all skills and reports false
rejections and false acceptances.

The graders take typed replies. One module also handles speech: `mathgen/grading/spoken.py`
turns transcripts of spoken answers ("one hundred and eighteen") into values, and
`spoken.check_answer(question, transcript)` grades one. Its number parsing is shared with the
typed graders, and a `number` skill whose answer is not a plain number (a list or a pair of
values, say) falls back on it, accepting values within 0.051. `tests/public/test_spoken.py`
covers `check_answer`.

## Skills

A skill YAML file names a skill, describes it, says which generator sources
build its questions, and lists prerequisites, common misconceptions and tags.
Some files also carry optional hints, such as stage and placement fields, for
applications built on the library. The loader ignores what it does not need.

## Scheduling reviews

`mathgen.scheduling` is for applications that space practice over time. It holds no learner
data and reads no clock; the application stores state and passes times in.

- `stages.eligible_sources(skill, stage, n_stages)` gives the sources a review at a stage
  draws from (a skill's `stages` hints), using the nearest staged sources when a stage has
  none. `stages.pick_sources` takes questions from them round-robin.
- `fsrs` is the FSRS-5 model (the open scheduler Anki uses), matching the reference
  implementation py-fsrs 5.1.3 (like it, gaps count in whole days, rounded down). `review.score(outcome, memory, last_review_at, now)` turns a
  finished review (`advance`, `hold`, `reset`, `maintenance_pass`) into new memory and a
  due date, and `review.memory_from_history` rebuilds memory from past outcomes.
- `calibration.item_difficulty(rows, skills)` fits a Rasch difficulty per skill and per
  tier from `{user_id, skill_id, difficulty, correct}` rows and flags tiers answered right
  more often than an earlier tier; `calibration.fsrs_calibration` compares predicted with
  observed recall.

## Licensing

- Code: MIT (`LICENSE`).
- Skill definitions in `skills/`: CC BY 4.0 (`LICENSE-CONTENT`).
- The `senior-*` skills (about 40% of the library) adapt content descriptors
  from the Australian Curriculum (Senior Secondary, Version 8.4), modified, and
  some code comments and tests quote descriptors too. ACARA licenses these
  under CC BY 4.0. See `NOTICE` for the required attribution.

Curriculum content (c) Australian Curriculum, Assessment and Reporting
Authority (ACARA) 2010 to present. This project is not affiliated with,
sponsored or approved by ACARA.

## Contributing and security

See `CONTRIBUTING.md` (DCO sign-off, no CLA) and `SECURITY.md`.
