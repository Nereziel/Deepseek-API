"""Offline browser-selection tests; no real browser or session required."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from deepseek import auth


@pytest.mark.parametrize("headless", [False, True])
def test_edge_explicit_channel(monkeypatch, headless):
    monkeypatch.setattr(auth, "PLAYWRIGHT_CHANNEL", "msedge")
    pw = Mock()
    profile = Path("session/profile-msedge")
    assert auth._launch_browser(pw, profile, headless) is pw.chromium.launch_persistent_context.return_value
    pw.chromium.launch_persistent_context.assert_called_once_with(
        str(profile), channel="msedge", headless=headless, args=auth.LAUNCH_ARGS
    )


def test_explicit_edge_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(auth, "PLAYWRIGHT_CHANNEL", "msedge")
    pw = Mock()
    pw.chromium.launch_persistent_context.side_effect = RuntimeError("Edge unavailable")
    with pytest.raises(RuntimeError, match="Edge unavailable"):
        auth._launch_browser(pw, Path("profile"), False)
    assert pw.chromium.launch_persistent_context.call_count == 1


def test_auto_preserves_chrome_fallback(monkeypatch):
    monkeypatch.setattr(auth, "PLAYWRIGHT_CHANNEL", "auto")
    pw = Mock()
    context = Mock()
    pw.chromium.launch_persistent_context.side_effect = [RuntimeError("Chrome unavailable"), context]
    assert auth._launch_browser(pw, Path("profile"), True) is context
    calls = pw.chromium.launch_persistent_context.call_args_list
    assert calls[0].kwargs["channel"] == "chrome"
    assert "channel" not in calls[1].kwargs


def test_bundled_chromium_explicit(monkeypatch):
    monkeypatch.setattr(auth, "PLAYWRIGHT_CHANNEL", "chromium")
    pw = Mock()
    auth._launch_browser(pw, Path("profile"), True)
    pw.chromium.launch_persistent_context.assert_called_once_with(
        "profile", headless=True, args=auth.LAUNCH_ARGS
    )


@pytest.fixture(autouse=True)
def clear_proxy(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_PROXY", raising=False)


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("channel", ["chrome", "msedge", "chromium", "auto"])
def test_browser_proxy(monkeypatch, headless, channel):
    monkeypatch.setenv("DEEPSEEK_PROXY", "socks5://127.0.0.1:9090")
    monkeypatch.setattr(auth, "PLAYWRIGHT_CHANNEL", channel)
    pw = Mock()
    auth._launch_browser(pw, Path("profile"), headless)
    kwargs = pw.chromium.launch_persistent_context.call_args.kwargs
    assert kwargs["proxy"] == {"server": "socks5://127.0.0.1:9090"}
    assert kwargs["headless"] is headless


def test_proxy_preserved_in_auto_fallback(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_PROXY", "socks5://127.0.0.1:9090")
    monkeypatch.setattr(auth, "PLAYWRIGHT_CHANNEL", "auto")
    pw = Mock()
    pw.chromium.launch_persistent_context.side_effect = [RuntimeError("Chrome unavailable"), Mock()]
    auth._launch_browser(pw, Path("profile"), False)
    for call in pw.chromium.launch_persistent_context.call_args_list:
        assert call.kwargs["proxy"] == {"server": "socks5://127.0.0.1:9090"}


@pytest.mark.parametrize("value", ["http://localhost:9090", "socks5://localhost", "socks5://user:password@localhost:9090", "socks5://localhost:0", "socks5://localhost:99999", "socks5://localhost:9090/path"])
def test_invalid_proxy(monkeypatch, value):
    from deepseek.proxy import proxy_url

    monkeypatch.setenv("DEEPSEEK_PROXY", value)
    with pytest.raises(ValueError, match="DEEPSEEK_PROXY"):
        proxy_url()


@pytest.mark.parametrize("configured", [False, True])
def test_httpx_proxy(monkeypatch, configured):
    from deepseek import client

    proxy = "socks5://127.0.0.1:9090"
    if configured:
        monkeypatch.setenv("DEEPSEEK_PROXY", proxy)
    session = auth.Session("test-token", {}, "test-agent", 0)
    monkeypatch.setattr(client, "DeepSeekPow", Mock())
    http = Mock()
    monkeypatch.setattr(client.httpx, "Client", http)
    instance = client.DeepSeekClient(session)
    kwargs = http.call_args.kwargs
    if configured:
        assert kwargs["proxy"] == proxy
        assert kwargs["trust_env"] is False
    else:
        assert "proxy" not in kwargs
        assert "trust_env" not in kwargs
    instance.close()
