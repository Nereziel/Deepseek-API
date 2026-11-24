"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true supported)
    GET  /healthz

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz is exempt.
"""

from __future__ import annotations

import logging
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import AccountMuted, DeepSeekClient

from .config import (
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    TRUST_PROXY_HEADERS,
    is_known_model,
    resolve_model_type,
)
from .openai_format import (
    completion_response,
    messages_to_prompt,
    stream_chunks,
    stream_error,
)
from .stream_tools import buffered_tool_stream
from .tools import active_tools, parse_tool_calls, prepare_tool_messages
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.1.0")
install_rate_limit(
    app,
    RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0),
    trust_forwarded=TRUST_PROXY_HEADERS,
)

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session.

    Session resolution: cached file → headless capture off the persistent
    profile. If neither works and SERVER_INTERACTIVE_LOGIN is on (the default),
    it opens a visible browser window so you can sign in — the triggering
    request blocks until you finish. If interactive login is off, it raises
    `LoginRequired`, which the endpoint turns into an actionable 503.

    This touches Playwright's sync API, so callers must invoke it OFF the event
    loop (via run_in_threadpool); calling it inside the asyncio loop raises
    "Playwright Sync API inside the asyncio loop"."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


def _muted_details(exc: AccountMuted) -> dict:
    logger.warning("DeepSeek account muted: mute_until=%r", exc.mute_until)
    return {"message": str(exc), "type": "permission_error",
            "code": "account_muted", "mute_until": exc.mute_until,
            "retryable": False}


def _muted_error(exc: AccountMuted):
    return JSONResponse(status_code=403, content={"error": _muted_details(exc)})


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "deepseek"}
            for name in MODEL_MAP
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}",
            status=404, err_type="model_not_found",
        )

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if req.conversation_id else resolve_model_type(req.model)
    try:
        enabled_tools = active_tools(req.tools, req.tool_choice)
    except ValueError as e:
        return _error(str(e), status=400, err_type="invalid_request_error")
    prepared_messages = prepare_tool_messages(req.messages, enabled_tools, req.tool_choice)
    prompt = messages_to_prompt(prepared_messages)
    logger.info("DeepSeek request: model=%s stream=%s thinking=%s messages=%d tools=%d prompt_chars=%d",
                req.model, req.stream, req.thinking_enabled, len(req.messages),
                len(enabled_tools), len(prompt))

    try:
        # Off the event loop: get_client() uses Playwright's sync API, which
        # errors if run inside the asyncio loop.
        client = await run_in_threadpool(get_client)
    except AccountMuted as e:
        return _muted_error(e)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    if enabled_tools:
        # The web API cannot natively stream tools. Buffer the upstream response
        # before classifying it so that we never expose partial tool syntax.
        try:
            reply = await run_in_threadpool(
                client.chat, prompt, req.conversation_id,
                model_type, req.thinking_enabled, req.search,
            )
        except AccountMuted as e:
            return _muted_error(e)
        except LoginRequired as e:
            return _error(str(e), status=503, err_type="login_required")
        except Exception as e:
            return _error(f"DeepSeek request failed: {e}")

        if not reply.text.strip():
            logger.warning("DeepSeek empty completion: reasoning_chars=%d", len(reply.reasoning))
            return _error("DeepSeek returned no final content", status=502, err_type="upstream_error")
        completion = completion_response(
            req.model, reply.text, prompt, reply.conversation_id, reply.reasoning
        )
        calls = parse_tool_calls(reply.text, enabled_tools, req.parallel_tool_calls)
        if calls:
            completion["choices"][0]["message"] = {
                "role": "assistant", "content": None, "tool_calls": calls
            }
            if reply.reasoning:
                completion["choices"][0]["message"]["reasoning_content"] = reply.reasoning
            completion["choices"][0]["finish_reason"] = "tool_calls"
        logger.info("DeepSeek completion: text_chars=%d reasoning_chars=%d tool_calls=%d",
                    len(reply.text), len(reply.reasoning), len(calls))
        if req.stream:
            return StreamingResponse(
                buffered_tool_stream(completion), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return completion

    if req.stream:
        # Open the thread before responding, so failures here (bad session,
        # DeepSeek errors) still get a real HTTP status instead of a 200 stream.
        try:
            stream = await run_in_threadpool(
                client.stream, prompt, req.conversation_id,
                model_type, req.thinking_enabled, req.search,
            )
        except AccountMuted as e:
            return _muted_error(e)
        except LoginRequired as e:
            return _error(str(e), status=503, err_type="login_required")
        except Exception as e:
            return _error(f"DeepSeek request failed: {e}")

        # Prime the stream before sending headers: a JSON restriction from
        # upstream must be an HTTP 403, not a successful HTTP 200 SSE.
        frames = iter(stream_chunks(req.model, stream))
        try:
            first = await run_in_threadpool(next, frames, None)
        except AccountMuted as e:
            return _muted_error(e)
        except LoginRequired as e:
            return _error(str(e), status=503, err_type="login_required")
        except Exception as e:
            logger.warning("DeepSeek stream failed before first frame: %s", type(e).__name__)
            return _error(f"DeepSeek request failed: {e}", status=502, err_type="upstream_error")
        if first is None:
            return _error("DeepSeek returned no stream", status=502, err_type="upstream_error")

        def gen():
            try:
                yield first
                yield from frames
            except AccountMuted as e:
                yield from stream_error(str(e), err_type="permission_error",
                                        details=_muted_details(e))
            except Exception as e:
                logger.warning("DeepSeek stream failed: %s", type(e).__name__)
                yield from stream_error(f"DeepSeek request failed: {e}", err_type="upstream_error")
            finally:
                frames.close()

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        reply = await run_in_threadpool(
            client.chat, prompt, req.conversation_id,
            model_type, req.thinking_enabled, req.search,
        )
    except AccountMuted as e:
        return _muted_error(e)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    if not reply.text.strip():
        logger.warning("DeepSeek empty completion: reasoning_chars=%d", len(reply.reasoning))
        return _error("DeepSeek returned no final content", status=502, err_type="upstream_error")
    logger.info("DeepSeek completion: text_chars=%d reasoning_chars=%d tool_calls=0",
                len(reply.text), len(reply.reasoning))
    return completion_response(req.model, reply.text, prompt, reply.conversation_id,
                               reply.reasoning)
