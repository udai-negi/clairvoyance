"""Sign test webhook requests the way Plivo and Vobiz sign theirs.

Plivo and Vobiz webhooks must carry a valid V3 signature, so tests that drive
those routes sign each request with ``signed_headers`` after ``use_test_keys``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import uuid
from types import SimpleNamespace
from typing import Dict, Optional

import pytest
from plivo.utils.signature_v3 import (
    construct_get_url,
    construct_post_url,
    get_signature_v3,
)

from app.ai.voice.agents.breeze_buddy.services.telephony import security

PLIVO_TEST_TOKEN = "test-plivo-auth-token"
VOBIZ_TEST_TOKEN = "test-vobiz-auth-token"


class _OnceRedis:
    def __init__(self) -> None:
        self.keys: set = set()

    async def set(self, key: str, value: str, nx: bool, ex: int) -> Optional[bool]:
        if key in self.keys:
            return None
        self.keys.add(key)
        return True


def use_test_keys(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    """Pin the URL and tokens the checks read, and give them an in-memory Redis."""
    redis = _OnceRedis()

    async def get_client():
        return redis

    async def get_redis_service():
        return SimpleNamespace(get_client=get_client)

    monkeypatch.setattr(security, "APP_BASE_URL", base_url)
    monkeypatch.setattr(security, "PLIVO_AUTH_TOKEN", PLIVO_TEST_TOKEN)
    monkeypatch.setattr(security, "VOBIZ_AUTH_TOKEN", VOBIZ_TEST_TOKEN)
    monkeypatch.setattr(security, "get_redis_service", get_redis_service)


def signed_headers(
    provider: str,
    base_url: str,
    path: str,
    data: Optional[Dict[str, str]] = None,
    method: str = "POST",
    query: str = "",
) -> Dict[str, str]:
    """V3 signature headers for a request to base_url + path (+ "?" + query)."""
    nonce = uuid.uuid4().hex
    if provider.lower() == "vobiz":
        text = f"{base_url}{path}.{nonce}"
        digest = hmac.new(
            VOBIZ_TEST_TOKEN.encode(), text.encode(), hashlib.sha256
        ).digest()
        return {
            "X-Vobiz-Signature-V3": base64.b64encode(digest).decode(),
            "X-Vobiz-Signature-V3-Nonce": nonce,
        }
    url = base_url + path + (f"?{query}" if query else "")
    if method == "GET":
        signed = construct_get_url(url, {}).decode()
    else:
        signed = construct_post_url(url, dict(data or {})).decode()
    signature = get_signature_v3(PLIVO_TEST_TOKEN.encode(), signed, nonce.encode())
    return {
        "X-Plivo-Signature-V3": signature.decode(),
        "X-Plivo-Signature-V3-Nonce": nonce,
    }
