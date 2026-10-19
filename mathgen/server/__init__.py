"""An HTTP API over the engine, for schools and anyone else who wants questions over the network.

Install with `pip install "maths-engine[server]"` and run `python -m mathgen.server` (or the Docker image, see
`docs/server.md`). `create_app()` gives the ASGI app to mount in another server.

FastAPI is imported only when `create_app` or `Config` is first used, so `import mathgen` never needs it, and
`mathgen.server.pool` (the worker processes with deadlines) works without it: the MCP server uses the pool too.
"""


def __getattr__(name):
    if name == "create_app":
        from .app import create_app

        return create_app
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
