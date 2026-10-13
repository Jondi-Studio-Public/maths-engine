# Agent plugins

Two [Agent Plugins](https://agent-plugins.org/specification) (v1.0.0) that give an AI agent the maths engine:
verified questions, deterministic marking and worksheets, aligned to the Australian Curriculum from early years to
first-year university.

| Plugin | For | Skills inside | Server role |
|---|---|---|---|
| `maths-engine-tutor` | a learner practising with an agent | `maths-tutoring`, `maths-skill-finder` | `learner`: never returns an answer |
| `maths-engine-teacher` | a teacher building materials | `maths-worksheets`, `maths-lms-export`, `maths-skill-finder` | `author`: answers, answer keys, Moodle and QTI |

Each plugin runs the `mathgen-mcp` server (see `docs/mcp.md`) as `maths-engine`, and its skills tell the agent how to
use the tools.

## Install

You need [uv](https://docs.astral.sh/uv/) on your `PATH`: the plugins launch the server with `uvx`. The server is
fetched from this repository's `main` branch, so it follows the latest code; a later release will pin a tag.

**Any Agent Plugins client.** Point the client at the plugin directory (`plugins/maths-engine-tutor` or
`plugins/maths-engine-teacher`). Each holds `plugin.json`, `mcp.json` and `skills/`.

**Claude Code.**

```
/plugin marketplace add Jondi-Studio-Public/maths-engine
/plugin install maths-engine-tutor@maths-engine
/plugin install maths-engine-teacher@maths-engine
```

Install whichever you need. Claude Code reads each plugin's server settings from `.claude-plugin/marketplace.json` at the repository root.

**Other MCP clients.** Add the server by hand: see `docs/mcp.md`.

Files the teacher plugin writes (worksheets, keys, banks) go to the plugin's data directory under `exports`, and the
agent reports the path.

## Why the tutor never sees answers

If the tutor's server could return answers, the agent would have them in front of it, and a model asked to be helpful
will sometimes give them away. So the learner role has no tool or flag that returns an answer, and marking says
whether a reply is right and what misconception it shows, nothing more. The tutor skill also tells the agent to coach
the method rather than work out and state the result. Marking always goes through the engine, which accepts
equivalent forms and applies each question's precision rules.

This is a safeguard for practice, not security. The source is public and a question id names its seed, so someone
with the engine installed can rebuild any answer. Do not use it to keep a high-stakes test secret. The teacher
plugin can show every answer: do not give it to learners.

## Licences

- Code: MIT (`LICENSE`).
- Skill definitions and these plugin skills: CC BY 4.0 (`LICENSE-CONTENT`).
- Senior secondary content is adapted from the Australian Curriculum (ACARA) under CC BY 4.0 and must keep its
  attribution; see `NOTICE`. ACARA does not endorse this product.
