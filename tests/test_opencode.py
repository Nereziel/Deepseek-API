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
