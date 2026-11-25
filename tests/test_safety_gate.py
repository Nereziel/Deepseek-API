"""Offline guard tests: never contact DeepSeek."""
import asyncio
import json

import httpx
import pytest

from deepseek.client import AccountMuted, Reply
from server import api
from server.safety_gate import SafetyGate, SingleCompletionMiddleware, retry_delay


@pytest.fixture(autouse=True)
def isolated_gate(monkeypatch, tmp_path):
    gate = SafetyGate(tmp_path / "account_muted.json", max_prompt_chars=50)
    monkeypatch.setattr(api, "_gate", gate)
    return gate


def test_oversized_prompt_never_initialises_client(monkeypatch, isolated_gate):
    from fastapi.testclient import TestClient

    def unexpected():
        pytest.fail("Upstream contacted")
    monkeypatch.setattr(api, "get_client", unexpected)
    client = TestClient(api.app)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x" * 51}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "prompt_too_large"
    assert client.get("/healthz").json()["counters"]["prompt_rejected"] == 1


def test_muted_persists_and_blocks_both_models(monkeypatch, isolated_gate):
    from fastapi.testclient import TestClient
    calls = []

    class Client:
        def chat(self, *args):
            calls.append(1)
            raise AccountMuted(None)

    monkeypatch.setattr(api, "get_client", lambda: Client())
    client = TestClient(api.app)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 403
    fresh = SafetyGate(isolated_gate.state_path)
    monkeypatch.setattr(api, "_gate", fresh)
    for model in ("deepseek-chat", "deepseek-expert"):
        r = client.post("/v1/chat/completions", json={**body, "model": model})
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "account_muted"
    assert calls == [1]
    assert client.get("/healthz").json()["account_muted"] is True


def test_bad_state_fails_closed(isolated_gate):
    isolated_gate.state_path.write_text("not JSON", encoding="utf-8")
    gate = SafetyGate(isolated_gate.state_path)
    assert gate.acquire().status_code == 503


def test_http_429_pauses_locally(monkeypatch, isolated_gate):
    from fastapi.testclient import TestClient
    calls = []

    class Client:
        def chat(self, *args):
            calls.append(1)
            response = httpx.Response(429, headers={"Retry-After": "120"},
                                      request=httpx.Request("POST", "https://example.invalid"))
            response.raise_for_status()

    monkeypatch.setattr(api, "get_client", lambda: Client())
    client = TestClient(api.app)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 429 and r.headers["Retry-After"] == "120"
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 429
    assert calls == [1]
    assert client.get("/healthz").json()["cooldown_seconds"] > 0


def test_retry_after():
    assert retry_delay("20", 0) == 20
    assert retry_delay("Thu, 01 Jan 1970 00:02:00 GMT", 60) == 60
    assert retry_delay("invalid", 0) == 60


def test_slot_held_for_entire_response_and_released_on_error(isolated_gate):
    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()
        sent = []

        async def downstream(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            started.set()
            await finish.wait()
            raise RuntimeError("disconnected")

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.disconnect"}

        middleware = SingleCompletionMiddleware(downstream, lambda: isolated_gate)
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}
        first = asyncio.create_task(middleware(scope, receive, send))
        await started.wait()
        await middleware(scope, receive, send)
        assert sent[-2]["status"] == 429
        assert isolated_gate.snapshot()["inflight"] == 1
        finish.set()
        with pytest.raises(RuntimeError):
            await first
        assert isolated_gate.snapshot()["inflight"] == 0
        assert isolated_gate.acquire() is None
        isolated_gate.release()

    asyncio.run(scenario())
