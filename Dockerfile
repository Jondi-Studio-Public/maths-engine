# The maths-engine HTTP API as a container: `docker build -t maths-engine .` then
# `docker run -p 8080:8080 -e MATHS_API_KEYS=author:<key>,learner:<key> maths-engine`. Settings: docs/server.md.
# Pinned by digest (the multi-arch index for 3.12-slim, checked 2026-10-13); bump it by hand (see docs/server.md, "Upgrades change ids").
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE LICENSE-CONTENT NOTICE MANIFEST.in ./
COPY .github/requirements-ci.txt ./constraints.txt
COPY mathgen ./mathgen
COPY skills ./skills
# The same exact versions CI tests with; a plain (not editable) install, so the image runs from site-packages.
RUN PIP_CONSTRAINT=constraints.txt PIP_BUILD_CONSTRAINT=constraints.txt pip install ".[server]" \
    && mkdir -p /usr/share/doc/maths-engine && cp LICENSE LICENSE-CONTENT NOTICE /usr/share/doc/maths-engine/ \
    && rm -rf /src

RUN useradd --create-home --uid 10001 engine
USER engine
WORKDIR /home/engine

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/health', timeout=4)"
CMD ["python", "-m", "mathgen.server", "--host", "0.0.0.0", "--port", "8080"]
