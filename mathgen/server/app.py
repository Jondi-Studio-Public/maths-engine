"""The HTTP API: find skills, generate questions and papers, grade replies. Stateless; see `docs/server.md`."""

# No `from __future__ import annotations`: FastAPI reads these annotations at run time, and `Role` is local.

import contextlib
import hashlib
import hmac
import logging
import math
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Any, Literal

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, StrictInt, StringConstraints

from .. import service
from .config import Config
from .jobs import NEEDS_ANSWERS
from .pool import Changed, Timeout, Unavailable, WorkerError, WorkerPool

log = logging.getLogger("mathgen.server")

#: The longest reply the API passes to a grader. The graders' own caps are far higher; nobody types more.
MAX_REPLY_CHARS = 2000
#: Replies in one batch. Each is graded with its own deadline, so a batch can hold a worker for a while.
MAX_BATCH = 100
#: Timeouts one client may cause, as a multiple of `cooldown_strikes`, within the cool-down window before all its
#: grading calls are refused for a while. Replies that hang under many different ids still add up here.
CLIENT_STRIKE_FACTOR = 2
#: Parts (or keys) in a multi-part reply.
MAX_REPLY_PARTS = 20
MAX_SEED = 2**63 - 1
#: A source is a key or difficulty label, or its index among the skill's sources.
SourceKey = Annotated[str, StringConstraints(max_length=100)] | Annotated[int, Field(ge=0, le=10**6)]


class BodyLimit:
    """Refuse a request body over `limit` bytes, counted as it arrives: a `Content-Length` check alone misses a
    chunked upload, which has none."""

    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        length = headers.get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.limit):
            return await _too_large(send)
        seen = 0
        refused = False

        async def counted():
            # Past the limit: answer 413 now and tell the app the client has gone, so it stops reading.
            nonlocal seen, refused
            if refused:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    refused = True
                    await _too_large(send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if not refused:  # the 413 is the response; drop whatever the app tries to send after it
                await send(message)

        await self.app(scope, counted, guarded_send)


async def _too_large(send):
    await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": b'{"detail":"request body too large"}'})


class SecurityHeaders:
    """Headers on every response. `nosniff` and `no-referrer` always; `no-store` always, because a response may
    carry answers or a question id and varies by key; a locked-down Content-Security-Policy on HTML and SVG, which
    a browser may open directly (`html`, `html-key` and `figure.svg`), so nothing inside can run script or load
    anything."""

    CSP = b"default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def with_headers(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])]
                names = {k.lower() for k, _ in headers}
                kind = next((v.lower() for k, v in headers if k.lower() == b"content-type"), b"")
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"cache-control", b"no-store"),
                ]
                if kind.startswith((b"text/html", b"image/svg+xml")):
                    extra.append((b"content-security-policy", self.CSP))
                headers += [(k, v) for k, v in extra if k not in names]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, with_headers)


# --- request bodies ---------------------------------------------------------------------------------------------


class QuestionsRequest(BaseModel):
    skill: str = Field(max_length=200, description="A skill id from GET /v1/skills.")
    source: SourceKey | None = Field(None, description="A source key or difficulty label (`easy`, `exam`...).")
    count: int = Field(1, ge=1, le=service.MAX_COUNT)
    seed: int | None = Field(
        None, ge=0, le=MAX_SEED, description="The same seed gives the same questions. Random if left out."
    )
    answers: bool = Field(False, description="Include answers and working. Needs an author key.")


class GradeRequest(BaseModel):
    id: str = Field(max_length=300, description="The question id, as returned with the question.")
    reply: str | StrictInt | dict[str, str | StrictInt] = Field(
        description="The learner's answer: text; an option index for multiple choice; an object keyed by part "
        "label for a multi-part question."
    )
    reveal: bool = Field(False, description="Include the right answer (`canonical`). Needs an author key.")


class GradeItem(BaseModel):
    id: str = Field(max_length=300)
    reply: str | StrictInt | dict[str, str | StrictInt]


class GradeManyRequest(BaseModel):
    items: list[GradeItem] = Field(min_length=1, max_length=MAX_BATCH)
    reveal: bool = Field(False, description="Include the right answers. Needs an author key.")


class PaperItem(BaseModel):
    skill: str = Field(max_length=200)
    source: SourceKey | None = None
    count: int = Field(1, ge=1, le=service.MAX_COUNT)


class PaperSection(BaseModel):
    title: str | None = Field(None, max_length=200)
    items: list[PaperItem] = Field(min_length=1, max_length=50)


class PaperRequest(BaseModel):
    title: str | None = Field(None, max_length=200)
    seed: int | None = Field(None, ge=0, le=MAX_SEED)
    versions: int = Field(1, ge=1, le=service.MAX_VERSIONS)
    shuffle: bool = False
    sections: list[PaperSection] = Field(min_length=1, max_length=20)
    format: Literal["json", "html", "html-key", "moodle", "qti"] = Field(
        "json", description="`html` prints the paper, `html-key` its answer key; `moodle` and `qti` are LMS banks."
    )
    version: str | None = Field(
        None, pattern="^[A-J]$", description="Only this version (A, B...). Moodle and QTI default to A."
    )
    answers: bool = Field(False, description="JSON only: include answers. Needs an author key.")


# --- app ---------------------------------------------------------------------------------------------------------


def create_app(config: Config | None = None, *, pool: WorkerPool | None = None) -> FastAPI:
    """The API as an ASGI app. `config` defaults to `Config.from_env()`; `pool` lets a test share one pool."""
    config = config or Config.from_env()
    config.check_auth()  # refuses to start with no keys unless MATHS_API_OPEN=1
    state: dict[str, Any] = {}
    # Timeouts per (skill, source): [window start, count, cooled-down-until]; see `run`.
    strikes: dict[str, list[float]] = {}
    strikes_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        state["lib"] = service.Library.load(config.skills_dir, config.extra_skills_dirs, id_secret=config.id_secret)
        state["pool"] = pool or WorkerPool(
            config.workers,
            root=config.skills_dir,
            extra_roots=config.extra_skills_dirs,
            id_secret=config.id_secret,
            memory_mb=config.worker_memory_mb,
        )
        # At most this many requests running or waiting for a worker; the rest are told to retry.
        state["slots"] = threading.BoundedSemaphore(config.max_in_flight or state["pool"].size * 2)
        # Batches may use at most half the workers at once, so single grades always have one left.
        state["batch_slots"] = threading.BoundedSemaphore(max(1, state["pool"].size // 2))
        if not config.keys:
            log.warning("no API keys set (MATHS_API_OPEN): every caller is treated as role %r", config.open_role)
        yield
        if pool is None:
            state["pool"].close()

    app = FastAPI(
        title="maths-engine API",
        version="1",
        summary="Generate maths questions with verified answers, build worksheets and tests, and grade replies.",
        description="Stateless: a question is named by its id and rebuilt from it to be graded. See docs/server.md.",
        lifespan=lifespan,
        docs_url="/docs" if config.docs else None,
        redoc_url="/redoc" if config.docs else None,
        openapi_url="/openapi.json" if config.docs else None,
    )
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=config.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # FastAPI's default handler echoes each error's `input`; a lone surrogate there cannot be encoded (500).
        errors = [{k: v for k, v in e.items() if k not in ("input", "ctx")} for e in exc.errors()]
        return JSONResponse({"detail": jsonable_encoder(errors)}, status_code=422)

    app.add_middleware(BodyLimit, limit=config.max_body_bytes)
    app.add_middleware(SecurityHeaders)  # last added, so outermost: it covers 401s, 413s and CORS answers too

    async def role(request: Request) -> str:
        """The caller's role from its `Authorization: Bearer <key>` (or `X-API-Key`) header."""
        if not config.keys:
            return config.open_role
        header = request.headers.get("authorization", "")
        given = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        for key, key_role in config.keys:
            if given and hmac.compare_digest(given.encode(), key.encode()):
                return key_role
        raise HTTPException(401, "a valid API key is required", headers={"WWW-Authenticate": "Bearer"})

    Role = Annotated[str, Depends(role)]

    def client(request: Request) -> str:
        """Who is calling, for the cool-down: the API key (hashed) or, with no keys, the network address."""
        if config.keys:
            header = request.headers.get("authorization", "")
            given = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
            return "key:" + hashlib.sha256(given.encode()).hexdigest()[:16]
        return "addr:" + (request.client.host if request.client else "?")

    Client = Annotated[str, Depends(client)]

    def need_author(who: str, what: str):
        if who != "author":
            raise HTTPException(403, f"{what} needs an author key")

    @contextmanager
    def slot():
        """One of the `max_in_flight` places; a busy server answers 503 now instead of queueing without limit."""
        slots = state["slots"]
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "the server is busy; try again shortly", headers={"Retry-After": "1"})
        try:
            yield
        finally:
            slots.release()

    def cooled_down(key: str | None) -> int:
        """Seconds left of a cool-down on `key`, else 0."""
        if key is None:
            return 0
        with strikes_lock:
            entry = strikes.get(key)
            return max(0, math.ceil(entry[2] - time.monotonic())) if entry else 0

    def strike(key: str | None, limit: int | None = None):
        """Count a timeout on `key`; `limit` (default `cooldown_strikes`) of them within the window start a cool-down."""
        if key is None:
            return
        now = time.monotonic()
        with strikes_lock:
            if len(strikes) > 1000:  # forget what has expired; the table cannot grow without bound
                for k in [k for k, e in strikes.items() if now - e[0] > config.cooldown and e[2] <= now]:
                    del strikes[k]
            entry = strikes.get(key)
            if entry is None or now - entry[0] > config.cooldown:
                entry = strikes[key] = [now, 0, 0.0]
            entry[1] += 1
            if entry[1] >= (limit or config.cooldown_strikes):
                entry[2] = now + config.cooldown
                entry[0], entry[1] = now, 0

    def run(name: str, timeout: float, *, key: str | None = None, held: bool = False, who: str | None = None, **kwargs):
        """Run a job in the pool. `key` names the (skill, source) for the cool-down and `who` the caller, which
        has a cool-down of its own across all its ids; `held` says the caller already holds a place (a batch
        holds one for all its items)."""
        for name_, cool in ((key, key), (who, "this client")):
            left = cooled_down(name_)
            if left:
                raise HTTPException(
                    503, f"{cool} timed out repeatedly; try again in {left} s", headers={"Retry-After": str(left)}
                )
        try:
            with contextlib.nullcontext() if held else slot():
                return state["pool"].run(name, timeout, wait=config.queue_timeout, **kwargs)
        except service.NotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except service.BadRequest as exc:
            raise HTTPException(422, str(exc)) from None
        except Changed as exc:
            raise HTTPException(409, {"error": "question_changed", "detail": str(exc), "new_id": exc.new_id}) from None
        except Timeout as exc:
            strike(key)
            strike(who, config.cooldown_strikes * CLIENT_STRIKE_FACTOR)
            raise HTTPException(504, str(exc)) from None
        except Unavailable as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from None
        except WorkerError as exc:
            log.error("worker error in %s: %s", name, exc)
            raise HTTPException(500, "the engine failed on this request") from None

    def id_key(qid: str) -> str | None:
        """`skill/source` of a question id, for the cool-down; None for something that is not an id."""
        parts = qid.split("/")
        return "/".join(parts[:2]) if len(parts) == 4 else None

    # The cheap routes are `async`, so they answer on the event loop even while every thread waits on a worker.
    @app.get("/v1/health", tags=["service"])
    async def health():
        pool_ = state["pool"]
        live = pool_.live
        body = {"status": "ok" if live else "down", **state["lib"].version(), "workers": live, "size": pool_.size}
        return JSONResponse(body, status_code=200 if live else 503)

    @app.get("/v1/skills", tags=["skills"])
    async def list_skills(
        _: Role,
        text: str | None = Query(None, max_length=100, description="Matches id, name or description."),
        domain: str | None = None,
        year: str | None = Query(None, description="F, 1 ... 10, 10A, university."),
        code: str | None = Query(None, description="An ACARA code or prefix: AC9M8 is every Year 8 code."),
        framework: str | None = None,
        answer_kind: str | None = None,
        difficulty: str | None = None,
        tag: str | None = None,
        gradeable: bool = False,
    ):
        hits = service.find_skills(
            state["lib"],
            text=text,
            domain=domain,
            year=year,
            code=code,
            framework=framework,
            answer_kind=answer_kind,
            difficulty=difficulty,
            tag=tag,
            gradeable_only=gradeable,
        )
        return {"count": len(hits), "skills": [service.skill_summary(s) for s in hits]}

    @app.get("/v1/skills/{skill_id}", tags=["skills"])
    async def get_skill(skill_id: str, _: Role):
        try:
            return service.skill_summary(state["lib"].skill(skill_id), full=True)
        except service.NotFound as exc:
            raise HTTPException(404, str(exc)) from None

    @app.post("/v1/questions", tags=["questions"])
    def make_questions(body: QuestionsRequest, who: Role):
        if body.answers:
            need_author(who, "answers")
        questions = run(
            "make_questions",
            config.timeout * max(1, body.count / 10),
            key=body.skill if body.source is None else f"{body.skill}/{body.source}",
            skill_id=body.skill,
            source=body.source,
            count=body.count,
            seed=body.seed,
            answers=body.answers,
        )
        return {"questions": questions, "engine": state["lib"].version()}

    @app.get("/v1/question", tags=["questions"])
    def get_question(who: Role, id: str = Query(max_length=300), answers: bool = False):
        if answers:
            need_author(who, "answers")
        return run("question", config.timeout, key=id_key(id), qid=id, answers=answers)

    @app.get("/v1/figure.svg", tags=["questions"], response_class=Response)
    def get_figure(_: Role, id: str = Query(max_length=300)):
        svg = run("figure_svg", config.timeout, key=id_key(id), qid=id)
        return Response(svg, media_type="image/svg+xml")

    def check_reply(reply):
        if isinstance(reply, dict):
            if len(reply) > MAX_REPLY_PARTS:
                raise HTTPException(422, f"a reply may have at most {MAX_REPLY_PARTS} parts")
            texts = [*reply.keys(), *reply.values()]
        else:
            texts = [reply]
        if sum(len(str(t)) for t in texts) > MAX_REPLY_CHARS:
            raise HTTPException(413, f"a reply may be at most {MAX_REPLY_CHARS} characters")

    @app.post("/v1/grade", tags=["grading"])
    def grade(body: GradeRequest, role_: Role, who: Client):
        if body.reveal:
            need_author(role_, "reveal")
        check_reply(body.reply)
        return run(
            "grade", config.timeout, key=id_key(body.id), who=who, qid=body.id, reply=body.reply, reveal=body.reveal
        )

    @app.post("/v1/grade/batch", tags=["grading"])
    def grade_many(body: GradeManyRequest, role_: Role, who: Client):
        """Several replies, as when marking a whole test. Each is graded with its own deadline within one total deadline,
        and replies that time out may use only `batch_budget` worker-seconds between them; one that cannot be marked
        gets an `error` (`question_changed`, `bad_request`, `timeout`) and the rest are still marked."""
        if body.reveal:
            need_author(role_, "reveal")
        for item in body.items:
            check_reply(item.reply)
        left = cooled_down(who)
        if left:
            raise HTTPException(
                503, f"this client timed out repeatedly; try again in {left} s", headers={"Retry-After": str(left)}
            )
        results = []
        deadline = time.monotonic() + config.batch_timeout
        kind = {404: "bad_request", 422: "bad_request", 409: "question_changed", 504: "timeout"}
        spent = 0.0  # worker-seconds given to replies that timed out
        batch_slots = state["batch_slots"]
        if not batch_slots.acquire(
            blocking=False
        ):  # batches share half the workers, so a single grade is never starved
            raise HTTPException(503, "another test is being marked; try again shortly", headers={"Retry-After": "1"})
        try:
            with slot():  # one place for the whole batch
                for item in body.items:
                    if spent >= config.batch_budget:  # hanging replies must not hold a worker for the whole deadline
                        results.append(
                            {"id": item.id, "error": "timeout", "detail": "the batch had too many slow replies"}
                        )
                        continue
                    left = deadline - time.monotonic()
                    if left <= 0:  # past the batch deadline: not run, so a batch cannot hold a worker for long
                        results.append({"id": item.id, "error": "timeout", "detail": "the batch ran out of time"})
                        continue
                    allowed = min(config.timeout, left, config.batch_budget - spent)
                    try:
                        results.append(
                            run(
                                "grade",
                                allowed,
                                key=id_key(item.id),
                                held=True,
                                who=who,
                                qid=item.id,
                                reply=item.reply,
                                reveal=body.reveal,
                            )
                        )
                    except HTTPException as exc:
                        spent += allowed if exc.status_code == 504 else 0
                        detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
                        results.append({"id": item.id, "error": kind.get(exc.status_code, "unavailable"), **detail})
        finally:
            batch_slots.release()
        return {"results": results}

    @app.post("/v1/papers", tags=["papers"])
    def make_paper(body: PaperRequest, who: Role):
        if body.answers or body.format in NEEDS_ANSWERS:
            need_author(who, "a paper with answers")
        spec = body.model_dump(include={"title", "seed", "versions", "shuffle", "sections"}, exclude_none=True)
        out = run(
            "paper",
            config.paper_timeout,
            spec=spec,
            answers=body.answers,
            format=body.format,
            version=body.version,
        )
        if body.format == "json":
            return out
        media_type, content = out
        extension = {"html": "html", "html-key": "html", "moodle": "xml", "qti": "zip"}[body.format]
        name = f"paper-{body.format}.{extension}"
        disposition = "inline" if extension == "html" else "attachment"
        return Response(
            content, media_type=media_type, headers={"Content-Disposition": f'{disposition}; filename="{name}"'}
        )

    return app
