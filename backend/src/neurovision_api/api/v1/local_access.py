from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from neurovision_api.core.config import get_allowed_origins


def require_direct_local_client(request: Request, action: str) -> None:
    try:
        client = ipaddress.ip_address(request.client.host if request.client else "")
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=f"{action} is restricted to local clients.") from exc
    host = urlsplit("//" + request.headers.get("host", "")).hostname
    origin = request.headers.get("origin")
    forwarded = any(name in request.headers for name in ("forwarded", "x-forwarded-for", "x-forwarded-host"))
    origin_host = urlsplit(origin).hostname if origin else None
    if (
        not client.is_loopback
        or host not in ("localhost", "127.0.0.1", "::1")
        or forwarded
        or (
            origin is not None
            and (
                origin not in get_allowed_origins()
                or origin_host not in ("localhost", "127.0.0.1", "::1")
            )
        )
    ):
        raise HTTPException(status_code=403, detail=f"{action} is restricted to direct local clients.")
