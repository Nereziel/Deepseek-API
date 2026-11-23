"""Buffered OpenAI-compatible SSE for prompt-emulated tool calls."""
import json


def buffered_tool_stream(completion):
    choice = completion["choices"][0]
    message = choice["message"]
    base = {k: completion[k] for k in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"

    def frame(delta, finish=None, extra=None):
        obj = {**base, "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish}
        ]}
        if extra:
            obj.update(extra)
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    yield frame({"role": "assistant", "content": ""})
    if message.get("reasoning_content"):
        yield frame({"reasoning_content": message["reasoning_content"]})
    if message.get("tool_calls"):
        for index, call in enumerate(message["tool_calls"]):
            yield frame({"tool_calls": [{
                "index": index, "id": call["id"], "type": "function",
                "function": {"name": call["function"]["name"], "arguments": ""}
            }]})
            yield frame({"tool_calls": [{
                "index": index,
                "function": {"arguments": call["function"]["arguments"]}
            }]})
    else:
        yield frame({"content": message.get("content") or ""})
    yield frame({}, choice["finish_reason"],
                {"conversation_id": completion.get("conversation_id")})
    yield "data: [DONE]\n\n"
