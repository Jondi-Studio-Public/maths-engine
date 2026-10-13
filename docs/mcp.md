# The MCP server

`mathgen-mcp` puts the engine behind the [Model Context Protocol](https://modelcontextprotocol.io), so an AI agent can
find skills, set questions, mark replies and build worksheets. It speaks stdio, the transport desktop agents launch
themselves. It stores nothing. For a web service instead, see `docs/server.md`.

## Install

```
pip install "maths-engine[mcp]"
```

or run it without installing, with [uv](https://docs.astral.sh/uv/):

```
uvx --from "maths-engine[mcp] @ git+https://github.com/Jondi-Studio-Public/maths-engine@main" mathgen-mcp
```

## Run

```
mathgen-mcp                                  # learner role: questions and marking, never an answer
mathgen-mcp --role author                    # adds answers, answer keys and LMS exports
mathgen-mcp --role author --out-dir ./out    # where files are written
```

| Flag | Default | |
|---|---|---|
| `--role learner\|author` | `learner` | what the server may return, below |
| `--out-dir DIR` | `./maths-engine-exports` | where worksheets and exports are written |
| `--skills DIR` | none | an extra folder of skill YAML; repeat the flag for more |
| `--timeout SECONDS` | `30` | how long one generate or mark call may run |
| `--version` | | print the version and exit |

An MCP client normally launches the server itself. A generic configuration:

```json
{
  "mcpServers": {
    "maths-engine": {
      "command": "mathgen-mcp",
      "args": ["--role", "learner", "--out-dir", "/path/to/exports"]
    }
  }
}
```

Without an install, use `"command": "uvx"` and put the `--from ...` arguments from above, then `mathgen-mcp`, in
`args`.

## Roles

| Role | May |
|---|---|
| `learner` (default) | questions without answers, marking without the right answer, worksheets as JSON or printable HTML |
| `author` | everything above, plus answers and working, `reveal` on marking, answer keys, Moodle XML and QTI 2.1 |

In the learner role no tool takes an `answers` or `reveal` flag, so nothing the server returns holds an answer:

- question views carry no answer;
- figures carry the alt text and, if asked for, the SVG drawing, but never the figure spec (for a read-the-picture
  question the spec is the answer);
- a marked reply carries no `canonical`.

**This keeps answers out of the conversation. It is not a lock.** A question id names its seed and the engine is
public, so anyone with the engine installed can rebuild any question and read its answer. Use the learner role for
practice, not for a test that must stay secret. The server also tells the agent not to work answers out and give
them away, but that is an instruction, not a guarantee.

Keep an author server away from learners: it can show every answer.

## Tools

Both roles have these. Replies are JSON.

| Tool | Parameters | Returns |
|---|---|---|
| `engine_info` | none | engine and library versions, the role, the paper formats allowed, and attribution text |
| `find_skills` | `text`, `year` (`F`, `1` ... `10`, `10A`), `curriculum_code` (an ACARA code or prefix, such as `AC9M8`), `domain`, `framework` (`acara-f10-v9`, `acara-senior-v8-4`, `university-core-v1`), `answer_kind`, `gradeable_only` (default false), `limit` (1 to 100, default 25) | `{"total", "skills"}`; every filter given must match |
| `get_skill` | `skill_id` | one skill in full: description, sources, prerequisites, tags, misconceptions |
| `make_questions` | `skill_id`, `count` (1 to 30, default 1), `seed`, `source`, `include_figures` (default false) | `{"questions": [...]}`; each has an `id` |
| `get_question` | `question_id`, `include_figures` | one question, rebuilt from its id |
| `grade_answer` | `question_id`, `reply` | whether the reply is right, a note, any misconception it shows, marks and per-part results |
| `build_paper` | `spec`, `format`, `version` | JSON, or the path of a file written; see below |

`reply` is the learner's text. For multiple choice it is the 0-based option index or the option text; for a
multi-part question an object keyed by part label. The same seed gives the same questions on any machine running the
same library. Skills that are not gradeable (proofs, explanations) are marked by a person and cannot be sent to
`grade_answer`. An id from an older library version is refused with a "no longer generated" message; set a new
question.

`build_paper` takes the paper spec described in `mathgen/service.py` (`build_paper`): a title, a seed, 1 to 10
versions, optional shuffling, and sections of `{skill, source, count}` items. `version` picks one version
(`A`, `B`, ...) of a multi-version paper for the file formats.

| Role | `format` values |
|---|---|
| `learner` | `json` (returned, no answers), `html` (a printable worksheet file) |
| `author` | `json`, `html`, `html-key` (answer key), `moodle` (Moodle XML bank), `qti` (IMS QTI 2.1 zip) |

The author role adds these parameters:

| Tool | Added parameter | Effect |
|---|---|---|
| `make_questions`, `get_question` | `answers` | include the answer and working |
| `grade_answer` | `reveal` | include the right answer as `canonical` |
| `build_paper` | `answers` | JSON only: include answers (default true). The key, Moodle and QTI formats always include them |

## Where files go

JSON comes back in the reply. Every other format is written under the output folder (`--out-dir`, default
`./maths-engine-exports`, created when needed) and the tool returns `{"path", "media_type", "bytes"}`. The server
chooses the name from the paper's title, its version and a hash of the content, so different papers never overwrite
each other and the caller never supplies a path.

## Deadline and workers

Every call that generates or marks runs in a worker process with a deadline (`--timeout`, 30 s), the same pool as the
HTTP API. A generator that never finishes costs one call: the worker is stopped and replaced, the tool returns an
error such as "the engine took longer than 30 s; try another seed or skill", and the session carries on. If the
engine is busy the error says so; try again. The worker pool and the library start on first use, so the handshake is
quick.

## Agent plugins

`plugins/` holds two Agent Plugins that start this server and add skills (instructions) for tutoring, worksheets and
LMS export. See `plugins/README.md`.

## Licensing in what it returns

As in `docs/server.md`: skill names and descriptions are CC BY 4.0, and skills adapted from the Australian
Curriculum carry an ACARA attribution. `engine_info` returns the attribution text and papers carry the attribution
their skills need. Keep it when you share material.
