"""Optional SOCKS5 transport shared by HTTPX and Playwright."""

import os
from urllib.parse import urlsplit


def proxy_url() -> str | None:
    """Return the configured proxy, or None when disabled.

    Chromium does not support SOCKS5 username/password authentication.
    Reject unsupported or malformed URLs without including secrets in errors.
    """
    value = os.getenv("DEEPSEEK_PROXY", "").strip()
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "socks5"
            and bool(parsed.hostname)
            and parsed.port is not None
            and parsed.port > 0
            and parsed.username is None
            and parsed.password is None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() for c in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "DEEPSEEK_PROXY must be a SOCKS5 URL without credentials, "
            "for example socks5://127.0.0.1:9090"
        )
    return value
