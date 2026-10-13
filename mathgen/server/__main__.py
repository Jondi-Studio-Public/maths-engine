"""`python -m mathgen.server [--host H] [--port P]`: serve the API with uvicorn. Settings: docs/server.md."""

import argparse
import logging


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m mathgen.server", description=__doc__)
    # Loopback by default: opening the API to a network is a choice, made with --host 0.0.0.0 (as the image does).
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    import uvicorn

    from . import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port, proxy_headers=True, log_level="info")


if __name__ == "__main__":
    main()
