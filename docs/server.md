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

A key is 16 or more characters of random. In bash, `openssl rand -hex 24` makes one. In Windows PowerShell:

```
[Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(24))
```

Without any keys, `python -m mathgen.server` serves only this machine: on a loopback host (the default) it starts open, as
role `MATHS_API_OPEN_ROLE`, and logs a warning. On any other host it refuses to start; see `MATHS_API_OPEN` below.

`compose.example.yaml` is a starting point for a long-running install. Without Docker:

```
pip install "maths-engine[server]"
python -m mathgen.server            # http://127.0.0.1:8080, loopback only unless you pass --host
```

The first start takes about 30 seconds while the library loads, so a request in that time is refused; wait for
`/v1/health` to answer.

The interactive reference (`/docs`, and `/redoc`) and the OpenAPI 3.1 description (`/openapi.json`) are off by
default; set `MATHS_API_DOCS=1` to serve them. They need no key, so leave them off on a server other people can reach.

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
  carries on. Run at least two workers so a replacement starting does not hold up other calls. A grade on a
  (skill, source) that has just timed out repeatedly fails fast for a short cool-down instead of killing another worker, and
  a batch that runs past `MATHS_API_BATCH_TIMEOUT` marks its remaining items with the error `timeout`.
- **Load is shed, not queued for ever.** When more calls are in flight than `MATHS_API_MAX_IN_FLIGHT`, the excess
  get `503` with a `Retry-After` header. Retry after that many seconds.
- **Upgrades change ids.** The library hash is part of every id's check, so after an upgrade that changes a skill,
  an old id answers `409 question_changed` with the new id rather than being marked against a different question.
  Finish any test in progress before upgrading, and pin an image tag (or the digest in the `Dockerfile`) so an
  upgrade happens when you choose, not when the container is rebuilt.

## Keys and roles

`MATHS_API_KEYS` is a comma-separated list of `role:key` (keys at least 16 characters). Send a key as
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

| Role | May |
|---|---|
| `author` | everything: answers and working, answer keys, Moodle and QTI banks, `reveal` on grading |
| `learner` | questions without answers, grading without the right answer |

Give learners' devices a `learner` key. **An author key never goes in a browser, a student's device or any
client-side page**: whoever can read the page can read every answer. Keep it on your own server, and have that
server call the API. If a browser page calls the API directly, give it only a `learner` key and list the exact
origin in `MATHS_API_CORS_ORIGINS`, never `*`. The server does not check this for you: `*` is accepted if you set it. Give each key one role; if one key is listed twice, the first role listed wins, with no warning.

With no keys set, `python -m mathgen.server` on a loopback host (`127.0.0.1`, the default) starts open with a
warning, since nothing else can reach it; on any other host, and through `create_app` directly, it refuses to start.
To run with no keys on purpose anyway, set `MATHS_API_OPEN=1`; every caller then has `MATHS_API_OPEN_ROLE`, which defaults to `learner`
(set it to `author` to see answers), and the server logs a warning.

**Learner keys suit practice, not summative tests.** The source is public, so an `author` key protects
convenience, not secrecy: anyone can run the engine and read any answer. A learner id
(`<skill>/<source>/<seed>/<hash>`) gives the answer the same way, because the public engine rebuilds the question
from it. Setting `MATHS_API_ID_SECRET` closes that gap: ids then cannot be regenerated without the server. Even so,
do not rely on it to keep a high-stakes test secret.

### Privacy and logs

The API stores nothing: no replies, no names, no results. It logs the method and path of each request, and the
path includes a question id (so it names a skill and a seed); it never logs a reply.

### Behind a reverse proxy

Rate limiting and TLS belong in the reverse proxy in front of the API (Caddy, nginx, a cloud load balancer). Tell
the server which proxy addresses to trust for `X-Forwarded-For` with `FORWARDED_ALLOW_IPS` (uvicorn's setting; for
a proxy on the same machine, `127.0.0.1`). A minimal Caddyfile:

```
maths.example.org {
    reverse_proxy 127.0.0.1:8080
}
```

The server sends `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer` on every response,
`Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox` on HTML and SVG,
and `Cache-Control: no-store` on every response, so an answer key is not kept by a browser or shared cache.

## Endpoints

| Method and path | What it does |
|---|---|
| `GET /v1/health` | Status, engine version, library hash, worker count. No key needed. |
| `GET /v1/skills` | Find skills. Filters: `text`, `domain`, `year` (`F`, `1` ... `12`, `university`), `code` (an ACARA code or prefix, `AC9M8`), `framework`, `answer_kind`, `difficulty`, `tag`, `gradeable`. |
| `GET /v1/skills/{id}` | One skill, with its description, prerequisites and misconceptions. |
| `POST /v1/questions` | `{"skill", "source"?, "count"? (1-100), "seed"?, "answers"?}` |
| `GET /v1/question?id=` | Rebuild one question by id (`answers=true` for an author). |
| `GET /v1/figure.svg?id=` | A question's figure as SVG. |
| `POST /v1/grade` | `{"id", "reply", "reveal"?}`. `reply` is text; an option index for multiple choice; `{"a": ..., "b": ...}` for a multi-part question (an unknown part label is a 422). A reply may be at most 2000 characters in all (`413` beyond that) and a multi-part reply at most 20 parts. |
| `POST /v1/grade/batch` | `{"items": [{"id", "reply"}, ...]}`, up to 100, as when marking a whole test. |
| `POST /v1/papers` | A worksheet or test: `{"title"?, "seed"?, "versions"? (1-10), "shuffle"?, "sections": [{"title"?, "items": [{"skill", "source"?, "count"?}]}], "format"?, "version"? (A to J), "answers"? (JSON only; needs an author key)}`. `format`: `json` (default), `html` (print), `html-key` (answer key), `moodle` (Moodle XML), `qti` (QTI 2.1 zip, for Canvas and others). A learner key gets `403` for `html-key`, `moodle` and `qti`. |

Moodle and QTI can mark multiple choice and plain numbers themselves. Other answer kinds (expressions,
fractions, pairs, sets...) go in as one accepted text answer, which an LMS compares character for character: an
equivalent form is marked wrong there. For those, grade through `/v1/grade`, or have a teacher review them.

## Settings

| Variable | Default | |
|---|---|---|
| `MATHS_API_KEYS` | none | `role:key,role:key`. Required unless `MATHS_API_OPEN=1`; a value that parses to no keys is an error |
| `MATHS_API_OPEN` | off | `1` lets the server run with no keys; otherwise it refuses to start unless it listens on loopback only |
| `MATHS_API_OPEN_ROLE` | `learner` | the role of every caller when running open |
| `MATHS_API_ID_SECRET` | none | 16 or more characters. Ids become unregeneratable without the server. Changing it invalidates every outstanding id |
| `MATHS_API_WORKERS` | CPUs available to the container, at most 8 | worker processes |
| `MATHS_API_MAX_IN_FLIGHT` | workers x 2 | calls in flight at once; the excess get `503` with `Retry-After` |
| `MATHS_API_WORKER_MEMORY_MB` | `1024` | address-space cap per worker process (Linux); `0` turns it off. A worker uses about 145 MB resident, 340 MB of address space |
| `MATHS_API_COOLDOWN` | `30` | seconds a (skill, source) fails fast after repeated timeouts |
| `MATHS_API_COOLDOWN_STRIKES` | `3` | timeouts within the cool-down window that start one |
| `MATHS_API_LIMIT_CONCURRENCY` | `64` | uvicorn's cap on concurrent connections |
| `MATHS_API_DOCS` | off | `1` serves `/docs`, `/redoc` and `/openapi.json` |
| `MATHS_API_TIMEOUT` | `5` | seconds for one question or one grade |
| `MATHS_API_BATCH_TIMEOUT` | 10 x `MATHS_API_TIMEOUT` | seconds for a whole batch; items still waiting past it get the error `timeout` |
| `MATHS_API_PAPER_TIMEOUT` | `60` | seconds for a paper |
| `MATHS_API_QUEUE_TIMEOUT` | `30` | seconds a call may wait for a free worker before `503` |
| `MATHS_API_MAX_BODY` | `262144` | bytes in a request body |
| `MATHS_API_CORS_ORIGINS` | none | comma-separated exact origins a browser page may call from; never `*` (not enforced) |
| `MATHS_SKILLS_DIR`, `MATHS_EXTRA_SKILLS_DIRS` | | your own skills alongside the library (see `mathgen/skills.py`) |

## Licensing in what it returns

Questions, answers and working are generated, and carry no licence terms of their own. Skill names and
descriptions are CC BY 4.0, and senior skill names are adapted from ACARA's curriculum: skill listings for those
carry an `attribution` field, and every paper carries the attribution its skills need, printed in its footer.
Keep it when you republish them.
