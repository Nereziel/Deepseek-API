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
