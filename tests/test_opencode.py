"""Offline contract tests: no browser, account or upstream network required."""
import json

import pytest

from server.schemas import ChatCompletionRequest, ChatMessage, ToolDefinition
from server.tools import active_tools, parse_tool_calls, prepare_tool_messages
from server.stream_tools import buffered_tool_stream


def tool():
    return ToolDefinition.model_validate({"type": "function", "function": {
        "name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}
    }})


def test_opencode_messages_and_tool_history():
    req = ChatCompletionRequest.model_validate({"model": "deepseek-chat", "stream": True,
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call_x", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"test.py"}'}}]},
            {"role": "tool", "tool_call_id": "call_x", "content": "hello"},
            {"role": "user", "content": "Summarize the file"}
        ],
        "tools": [tool().model_dump()], "tool_choice": "auto"})
    messages = prepare_tool_messages(req.messages, active_tools(req.tools, req.tool_choice))
    assert "read_file" in messages[0].content
    assert "<tool_call>" in messages[1].content
    assert "call_x" in messages[2].content


def test_only_whitelisted_schema_valid_standalone_calls():
    t = [tool()]
    good = '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>'
    calls = parse_tool_calls(good, t)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"}
    assert not parse_tool_calls('<tool_call>{"name":"evil","arguments":{}}</tool_call>', t)
    assert not parse_tool_calls('<tool_call>{"name":"read_file","arguments":{"wrong":3}}</tool_call>', t)
    assert not parse_tool_calls("explanation: " + good, t)
    assert not parse_tool_calls("~~~" + good + "~~~", t)
    assert not parse_tool_calls(good + good, t, parallel=False)


def test_tool_choice():
    t = [tool()]
    assert active_tools(t, "none") == []
    assert active_tools(t, {"type": "function", "function": {"name": "read_file"}}) == t
    with pytest.raises(ValueError):
        active_tools(t, {"type": "function", "function": {"name": "missing"}})
    with pytest.raises(ValueError):
        active_tools([], "required")


def test_stream_tool_call_format():
    calls = parse_tool_calls('<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>', [tool()])
    completion = {"id": "chatcmpl-1", "created": 1, "model": "deepseek-chat",
                  "conversation_id": "session:1", "choices": [{"index": 0,
                  "message": {"role": "assistant", "content": None, "tool_calls": calls},
                  "finish_reason": "tool_calls"}]}
    chunks = list(buffered_tool_stream(completion))
    assert chunks[-1] == "data: [DONE]\n\n"
    data = [json.loads(chunk[6:]) for chunk in chunks[:-1]]
    assert data[0]["choices"][0]["delta"]["role"] == "assistant"
    assert data[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read_file"
    assert data[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"path":"x"}'
    assert data[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert data[-1]["conversation_id"] == "session:1"


def test_tool_history_is_preserved_with_disabled_tools():
    messages = prepare_tool_messages([
        ChatMessage(role="tool", tool_call_id="call_1", content="result")
    ], [])
    assert messages[0].role == "user"
    assert "call_1" in messages[0].content


def test_external_schema_refs_are_not_resolved():
    unsafe = ToolDefinition.model_validate({"type": "function", "function": {
        "name": "read_file",
        "parameters": {"$ref": "http://127.0.0.1:80/untrusted.json"}
    }})
    payload = '<tool_call>{"name":"read_file","arguments":{}}</tool_call>'
    assert not parse_tool_calls(payload, [unsafe])


def test_thinking_compatibility():
    base = {"messages": [{"role": "user", "content": "hello"}]}
    assert not ChatCompletionRequest.model_validate(base).thinking_enabled
    assert ChatCompletionRequest.model_validate({
        **base, "chat_template_kwargs": {"enable_thinking": True}
    }).thinking_enabled
    assert not ChatCompletionRequest.model_validate({
        **base, "thinking": False, "chat_template_kwargs": {"enable_thinking": True}
    }).thinking_enabled


def test_closed_tools_keep_call_identity_and_request_prose():
    messages = prepare_tool_messages([
        ChatMessage.model_validate({"role": "assistant", "tool_calls": [{
            "id": "call_test", "function": {"name": "read_file", "arguments": '{"path":"a"}'}
        }]}),
        ChatMessage(role="tool", tool_call_id="call_test", content="result")
    ], [], "none")
    assert "do not emit tool calls" in messages[0].content
    assert "call_test" in messages[1].content
    assert "read_file (call_test)" in messages[2].content


def test_replayed_snapshot_emits_only_new_text():
    from deepseek.client import _parse_sse, THINK, RESPONSE
    frames = [
        {"v": {"response": {"fragments": [{"type": THINK, "content": "Think"}]}}},
        {"p": "response/fragments/-1/content", "o": "APPEND", "v": " once"},
        {"v": {"response": {"fragments": [
            {"type": THINK, "content": "Think once"},
            {"type": RESPONSE, "content": "Hello"}
        ]}}},
        {"v": "!"},
    ]
    assert list(_parse_sse("data: " + json.dumps(f) for f in frames)) == [
        (THINK, "Think"), (THINK, " once"), (RESPONSE, "Hello"), (RESPONSE, "!")
    ]


def test_upstream_error_is_not_a_successful_stream():
    from deepseek.client import _parse_sse
    with pytest.raises(RuntimeError, match="reported an error"):
        list(_parse_sse(['data: {"code": 40003}']))


def test_reasoning_only_stream_has_no_success_finish():
    from deepseek.client import THINK
    from server.openai_format import stream_chunks

    class Stream:
        def events(self):
            yield THINK, "thinking"

    frames = stream_chunks("deepseek-chat", Stream())
    assert "assistant" in next(frames)
    assert "reasoning_content" in next(frames)
    with pytest.raises(RuntimeError, match="no final content"):
        next(frames)


@pytest.mark.parametrize("with_tools", [False, True])
def test_empty_completion_reports_502(monkeypatch, with_tools):
    from fastapi.testclient import TestClient
    from deepseek.client import Reply
    from server import api

    class Client:
        def chat(self, *args):
            return Reply(text="", reasoning="thinking", conversation_id="s:2")

    monkeypatch.setattr(api, "get_client", lambda: Client())
    body = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}
    if with_tools:
        body["tools"] = [tool().model_dump()]
    response = TestClient(api.app).post("/v1/chat/completions", json=body)
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


def test_non_sse_business_status_diagnostic():
    from deepseek.client import _completion_json_error
    msg = _completion_json_error({
        "code": 0, "msg": "ok", "data": {
            "biz_code": 123, "biz_msg": "example status",
            "biz_data": {"status": "failed", "token": "SECRET", "prompt": "PRIVATE"}
        }
    })
    assert "biz_code='123'" in msg
    assert "biz_msg='example status'" in msg
    assert "biz_data.status='failed'" in msg
    assert "SECRET" not in msg
    assert "PRIVATE" not in msg
    assert "biz_data_type=NoneType" in _completion_json_error({"code": 0})
