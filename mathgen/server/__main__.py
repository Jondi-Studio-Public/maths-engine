"""`python -m mathgen.server [--host H] [--port P]`: serve the API with uvicorn. Settings: docs/server.md."""

import argparse
import logging
import os
import sys

LOOPBACK = ("127.0.0.1", "::1", "localhost")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m mathgen.server", description=__doc__)
    # Loopback by default: opening the API to a network is a choice, made with --host 0.0.0.0 (as the image does).
    parser.add_argument(
        "--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1, this machine only)"
    )
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (default 8080)")
    args = parser.parse_args(argv)

    try:
        import uvicorn

        from . import Config, create_app
    except ModuleNotFoundError as exc:
        if exc.name not in ("uvicorn", "fastapi", "starlette"):
            raise
        sys.exit(f'mathgen.server needs {exc.name!r}: install the `server` extra, pip install "maths-engine[server]"')

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = Config.from_env()
        if not config.keys and not config.open and args.host in LOOPBACK:
            logging.getLogger("mathgen.server").warning("no API keys: serving this machine only (%s)", args.host)
            config.open = True  # nobody else can reach a loopback address
        app = create_app(config)
    except ValueError as exc:
        sys.exit(f"mathgen.server: {exc}")
    # Past this many open connections uvicorn answers 503 itself, before a request reaches the app.
    limit = int(os.environ.get("MATHS_API_LIMIT_CONCURRENCY") or 64)
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=True, log_level="info", limit_concurrency=limit)


if __name__ == "__main__":
    main()
