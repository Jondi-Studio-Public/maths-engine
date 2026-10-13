"""An HTTP API over the engine, for schools and anyone else who wants questions over the network.

Install with `pip install "maths-engine[server]"` and run `python -m mathgen.server` (or the Docker image, see
`docs/server.md`). `create_app()` gives the ASGI app to mount in another server.

FastAPI is imported only here, so `import mathgen` never needs it.
"""

from .app import create_app  # noqa: F401
from .config import Config  # noqa: F401
