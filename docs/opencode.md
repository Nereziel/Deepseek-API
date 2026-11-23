# OpenCode integration (experimental tool calling)

This fork adds OpenAI `tools` and `tool_choice` support directly to the existing
FastAPI `POST /v1/chat/completions` endpoint. **No separate proxy is required.**
The DeepSeek consumer web interface does not provide native function calling,
so tool calls are **prompt-emulated and best-effort**. The bridge validates function
names and JSON arguments against the advertised schema and returns OpenAI
`tool_calls`; it does not run tools itself. OpenCode executes tools locally.

## Install on Windows Server

From a PowerShell session in this repo:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m deepseek.auth
.\.venv\Scripts\python.exe app.py
```

Existing virtual environments work too; use the same Python path with
`-m pip install -r requirements.txt`. Run the login interactively once.
Test `http://127.0.0.1:8000/healthz` and `http://127.0.0.1:8000/v1/models`.

### Use installed Microsoft Edge instead of bundled Chromium

Set `PLAYWRIGHT_CHANNEL=msedge` in the repository's `.env` file.
Edge must already be installed. No separate Playwright Chromium installation
is required when Edge is selected. Both manual login and headless session
refresh use Edge; launch failure is reported instead of silently selecting
another browser. By default Edge uses the isolated `session/profile-msedge`
profile, not your regular browser profile. Run `python -m deepseek.auth`
again after switching to replace the cached session, then restart the server.
Unset the setting or set it to `auto` to preserve Chrome-first behavior,
or use `chromium` to select the bundled browser explicitly.

## OpenCode 2.x configuration

In your project `opencode.jsonc` (or global OpenCode config):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "deepseek-local/deepseek-chat",
  "providers": {
    "deepseek-local": {
      "name": "DeepSeek Web (local)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "unused"
      },
      "models": {
        "deepseek-chat": {
          "name": "DeepSeek Instant (web)",
          "capabilities": {"tools": true, "input": ["text"], "output": ["text"]}
        },
        "deepseek-expert": {
          "name": "DeepSeek Expert (web)",
          "capabilities": {"tools": true, "input": ["text"], "output": ["text"]}
        }
      }
    }
  }
}
```

The `apiKey` above is a dummy required by some clients; the local server does not verify it. It is NOT an access control.\n\nUse `/models` inside OpenCode to pick `deepseek-local/deepseek-chat`.
This config targets the current OpenCode 2.x schema. Older OpenCode versions
use `provider` / `npm` / `options.baseURL` instead.

**Important:** `127.0.0.1` works only if OpenCode and the bridge run on the
same host. If OpenCode runs on your PC and the bridge on a Windows Server,
use a private VPN address and protect the HTTP endpoint with authentication.
The base project has no inbound API authentication; do not expose port 8000
to the public Internet, particularly with a reusable DeepSeek login session.

## How tool calls work

1. OpenCode sends OpenAI `tools` and recent `messages`.
2. The bridge prompts the DeepSeek web model to emit `<tool_call>` JSON.
3. The response is validated against the allowed tool names and JSON schemas.
4. The bridge returns OpenAI `tool_calls` and `finish_reason: "tool_calls"`.
5. OpenCode runs the tool and returns `role: "tool"` results on the next turn.

Tool-enabled `stream: true` requests are **buffered** until the model has
finished: the returned SSE stream follows the OpenAI format, but it is not live
token-by-token streaming. Text-only requests retain normal streaming.

The model may return regular text instead of a valid tool call. Malformed,
unrecognized or schema-invalid tool calls are returned as text, never executed.
Tool output is untrusted input. Tool execution and permissions belong to OpenCode.
For coding agents, avoid granting automatic permission for destructive commands
until you've verified the model's behavior on your workload.

The web API does not support native `/v1/responses`, guaranteed tool_choice
compliance, native image input or accurate token usage. Do not assume these
features from the OpenAI-compatible endpoint.

## Offline tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```


## Optional SOCKS5 proxy

For a local proxy such as AdGuard VPN in SOCKS5 mode, install the updated
requirements and set the following in the project-root `.env`:

```ini
DEEPSEEK_PROXY=socks5://127.0.0.1:9090
HOST=127.0.0.1
PORT=8001
PLAYWRIGHT_CHANNEL=chrome
```

The configured proxy is used for HTTPX requests and both interactive browser
login and headless session refresh. Only SOCKS5 URLs without credentials are
supported. With an explicit proxy, HTTPX ignores environment proxy settings
and does not retry over a direct connection. Unset DEEPSEEK_PROXY to retain
the original behavior. Restart the server after changing the settings.

Keep the proxy bound to loopback. When OpenCode runs on the same machine, its
provider base URL is `http://127.0.0.1:8001/v1`; incoming API requests do not
use the SOCKS5 proxy. Test the proxy separately before signing in manually.
