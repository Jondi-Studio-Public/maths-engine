# The HTTP API

`mathgen.server` puts the engine behind a small JSON API, so a school, an LMS plugin or any program that is not
Python can generate questions, build worksheets and tests, and grade replies. It is stateless: nothing is stored,
and any number of copies can run side by side.

## Run it

With Docker:

```
docker build -t maths-engine .
docker run --rm -p 127.0.0.1:8080:8080 -e MATHS_API_KEYS="author:$(openssl rand -hex 24)" maths-engine
```

`compose.example.yaml` is a starting point for a long-running install. Without Docker:

```
pip install "maths-engine[server]"
python -m mathgen.server            # http://127.0.0.1:8080, loopback only unless you pass --host
```

The interactive reference is at `/docs` and the OpenAPI 3.1 description at `/openapi.json`.

## How it works

- **A question is named by its id**, `<skill>/<source key>/<seed>/<hash>`. The id is everything needed to rebuild
  the question, so grading takes an id and a reply and rebuilds the question itself. A caller never sends a
  question back, so it cannot be marked against something it was not shown.
- **The same seed gives the same questions** on every machine running the same library. `GET /v1/health` names
  the library (`library`, a hash of the skill files); a different value means some ids may now give different
  questions. Grading such an id answers `409 question_changed` with the new id, rather than marking against a
  question the learner never saw.
- **Every call that generates or grades runs in a worker process with a deadline** (`MATHS_API_TIMEOUT`, 5 s).
  A few replies make SymPy run for ever; the worker is killed and replaced, the call gets `504`, and the server
  carries on. Run at least two workers so a replacement starting does not hold up other calls.

## Keys and roles

`MATHS_API_KEYS` is a comma-separated list of `role:key` (keys at least 16 characters). Send a key as
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

| Role | May |
|---|---|
| `author` | everything: answers and working, answer keys, Moodle and QTI banks, `reveal` on grading |
| `learner` | questions without answers, grading without the right answer |

Give learners' devices a `learner` key. With no keys set, every caller has `MATHS_API_OPEN_ROLE` (default
`author`), which suits one teacher on their own machine; the server logs a warning. The source is public, so an
`author` key protects convenience, not secrecy: anyone can run the engine and read any answer. Do not use it to
keep a high-stakes test secret.

Rate limiting and TLS belong in the reverse proxy in front of the API (Caddy, nginx, a cloud load balancer).

## Endpoints

| Method and path | What it does |
|---|---|
| `GET /v1/health` | Status, engine version, library hash, worker count. No key needed. |
| `GET /v1/skills` | Find skills. Filters: `text`, `domain`, `year` (`F`, `1` ... `10A`), `code` (an ACARA code or prefix, `AC9M8`), `framework`, `answer_kind`, `difficulty`, `tag`, `gradeable`. |
| `GET /v1/skills/{id}` | One skill, with its description, prerequisites and misconceptions. |
| `POST /v1/questions` | `{"skill", "source"?, "count"? (1-100), "seed"?, "answers"?}` |
| `GET /v1/question?id=` | Rebuild one question by id (`answers=true` for an author). |
| `GET /v1/figure.svg?id=` | A question's figure as SVG. |
| `POST /v1/grade` | `{"id", "reply", "reveal"?}`. `reply` is text; an option index for multiple choice; `{"a": ..., "b": ...}` for a multi-part question. |
| `POST /v1/grade/batch` | `{"items": [{"id", "reply"}, ...]}`, up to 200, as when marking a whole test. |
| `POST /v1/papers` | A worksheet or test: `{"title"?, "seed"?, "versions"? (1-10), "shuffle"?, "sections": [{"title"?, "items": [{"skill", "source"?, "count"?}]}], "format"?}`. `format`: `json` (default), `html` (print), `html-key` (answer key), `moodle` (Moodle XML), `qti` (QTI 2.1 zip, for Canvas and others). |

Moodle and QTI can mark multiple choice and plain numbers themselves. Other answer kinds (expressions,
fractions, pairs, sets...) go in as one accepted text answer, which an LMS compares character for character: an
equivalent form is marked wrong there. For those, grade through `/v1/grade`, or have a teacher review them.

## Settings

| Variable | Default | |
|---|---|---|
| `MATHS_API_KEYS` | none | `role:key,role:key` |
| `MATHS_API_OPEN_ROLE` | `author` | the role of every caller when no keys are set |
| `MATHS_API_WORKERS` | CPU count, at most 8 | worker processes |
| `MATHS_API_TIMEOUT` | `5` | seconds for one question or one grade |
| `MATHS_API_PAPER_TIMEOUT` | `60` | seconds for a paper |
| `MATHS_API_QUEUE_TIMEOUT` | `30` | seconds a call may wait for a free worker before `503` |
| `MATHS_API_MAX_BODY` | `262144` | bytes in a request body |
| `MATHS_API_CORS_ORIGINS` | none | comma-separated origins a browser page may call from |
| `MATHS_SKILLS_DIR`, `MATHS_EXTRA_SKILLS_DIRS` | | your own skills alongside the library (see `mathgen/skills.py`) |

## Licensing in what it returns

Questions, answers and working are generated, and carry no licence terms of their own. Skill names and
descriptions are CC BY 4.0, and senior skill names are adapted from ACARA's curriculum: skill listings for those
carry an `attribution` field, and every paper carries the attribution its skills need, printed in its footer.
Keep it when you republish them.
