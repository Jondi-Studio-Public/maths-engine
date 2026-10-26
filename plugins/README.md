# Agent plugins

Two [Agent Plugins](https://agent-plugins.org/specification) (v1.0.0) that give an AI agent the maths engine:
verified questions, deterministic marking and worksheets, aligned to the Australian Curriculum from early years to
second-year university.

| Plugin | For | Skills inside | Server role |
|---|---|---|---|
| `maths-engine-tutor` | a learner practising with an agent | `maths-tutoring`, `maths-skill-finder` | `learner`: never returns an answer |
| `maths-engine-teacher` | a teacher building materials | `maths-worksheets`, `maths-lms-export`, `maths-skill-finder` | `author`: answers, answer keys, Moodle and QTI |

Each plugin runs the `mathgen-mcp` server (see `docs/mcp.md`) as `maths-engine`, and its skills tell the agent how to
use the tools.

## Install

You need [uv](https://docs.astral.sh/uv/) on your `PATH`: the plugins launch the server with `uvx`, which fetches
the release each plugin pins (`maths-engine[mcp]==0.2.0`) from PyPI. That release must be on PyPI before the
plugins can start: until it is, the server fails to start with a bare `CONNECTION_CLOSED`.

**Any Agent Plugins client.** Point the client at the plugin directory (`plugins/maths-engine-tutor` or
`plugins/maths-engine-teacher`). Each holds `plugin.json`, `mcp.json` and `skills/`.

**Claude Code.**

```
/plugin marketplace add Jondi-Studio-Public/maths-engine
/plugin install maths-engine-tutor@maths-engine
/plugin install maths-engine-teacher@maths-engine
```

Install whichever you need. Each plugin has a `.claude-plugin/plugin.json` that carries its MCP server, so the marketplace and
`claude --plugin-dir plugins/maths-engine-tutor` both start the server and load the skills.

**Other MCP clients.** Add the server by hand: see `docs/mcp.md`.

Files the teacher plugin writes (worksheets, keys, banks) go to the plugin's data directory under `exports`, and the
agent reports the path. If you run `mathgen-mcp` by hand, the default folder is `~/maths-engine-exports`; do not share
one output folder between a teacher server and a learner agent, because answer keys land there.

## Why the tutor never sees answers

If the tutor's server could return answers, the agent would have them in front of it, and a model asked to be helpful
will sometimes give them away. So the learner role has no tool or flag that returns an answer, and marking says
whether a reply is right and what misconception it shows, nothing more. The tutor skill also tells the agent to coach
the method rather than work out and state the result, not to use the marker as an oracle (trying options or candidate
answers), and to describe figures from their alt text. Marking always goes through the engine, which accepts
equivalent forms and applies each question's precision rules.

This is a safeguard for practice, not security. The source is public and a question id names its seed, so someone
with the engine installed can rebuild any answer. Do not use it to keep a high-stakes test secret. The teacher
plugin can show every answer: do not give it to learners.

## Licences

- The plugins in this folder (instructions and configuration only): MIT.
- Engine code: MIT (`LICENSE`).
- Skill library definitions under `skills/`: CC BY 4.0 (`LICENSE-CONTENT`).
- Senior secondary content is adapted from the Australian Curriculum (ACARA) under CC BY 4.0 and must keep its
  attribution; see `NOTICE`. ACARA does not endorse this product.
