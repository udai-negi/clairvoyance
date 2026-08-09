"""Telephony webhook and media socket checks.

- Every Plivo and Vobiz webhook route refuses a request with no signature or a
  wrong one (401), and Twilio and Exotel routes are gone (404).
- A Vobiz signature passes once: the same nonce again is refused, and when
  Redis is down a valid signature still passes.
- A Plivo request signed with a template's own Plivo account passes through
  the lead's account; an unknown token does not.
- A stream ticket from the real URL builder passes; a changed query value,
  another provider, an expired or missing ticket do not, and the socket route
  closes a socket without one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from types import SimpleNamespace
from typing import Dict, Optional
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from plivo.utils.signature_v3 import construct_post_url, get_signature_v3
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.ai.voice.agents.breeze_buddy import dispatch as _dispatch  # noqa: F401
from app.ai.voice.agents.breeze_buddy.services.telephony import security
from app.api.routers.breeze_buddy import websocket as websocket_routes
from app.api.routers.breeze_buddy.telephony import router as telephony_router
from app.api.routers.breeze_buddy.telephony.answer import handlers as ans_mod

PREFIX = "/agent/voice/breeze-buddy"
BASE_URL = "https://bb.example.com"
VOBIZ_TOKEN = "test-vobiz-auth-token"
PLIVO_TOKEN = "test-plivo-auth-token"
MERCHANT_PLIVO_TOKEN = "merchant-own-plivo-token"
CHECK_PATH = "/checked"


class FakeRedis:
    def __init__(self) -> None:
        self.keys: set = set()

    async def set(self, key: str, value: str, nx: bool, ex: int) -> Optional[bool]:
        if key in self.keys:
            return None
        self.keys.add(key)
        return True


class BrokenRedis:
    async def set(self, *args, **kwargs) -> None:
        raise ConnectionError("redis down")


def use_redis(monkeypatch: pytest.MonkeyPatch, client) -> None:
    async def get_client():
        return client

    async def get_redis_service():
        return SimpleNamespace(get_client=get_client)

    monkeypatch.setattr(security, "get_redis_service", get_redis_service)


@pytest.fixture(autouse=True)
def secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(security, "APP_BASE_URL", BASE_URL)
    monkeypatch.setattr(security, "VOBIZ_AUTH_TOKEN", VOBIZ_TOKEN)
    monkeypatch.setattr(security, "PLIVO_AUTH_TOKEN", PLIVO_TOKEN)
    monkeypatch.setattr(security, "JWT_SECRET_KEY", "test-jwt-secret")
    use_redis(monkeypatch, FakeRedis())

    async def no_lead(call_id: str) -> None:
        return None

    monkeypatch.setattr(security, "get_lead_by_call_id", no_lead)


def vobiz_headers(path: str, nonce: str = "nonce-1", token: str = VOBIZ_TOKEN):
    text = f"{BASE_URL}{path}.{nonce}"
    digest = hmac.new(token.encode(), text.encode(), hashlib.sha256).digest()
    return {
        "X-Vobiz-Signature-V3": base64.b64encode(digest).decode(),
        "X-Vobiz-Signature-V3-Nonce": nonce,
    }


def plivo_headers(path: str, data: Dict[str, str], token: str) -> Dict[str, str]:
    nonce = "plivo-nonce"
    signed = construct_post_url(BASE_URL + path, dict(data)).decode()
    signature = get_signature_v3(token.encode(), signed, nonce.encode())
    return {
        "X-Plivo-Signature-V3": signature.decode(),
        "X-Plivo-Signature-V3-Nonce": nonce,
    }


def checked(check) -> TestClient:
    app = FastAPI()

    @app.post(CHECK_PATH)
    async def route(request: Request) -> Dict[str, bool]:
        await check(request)
        return {"ok": True}

    return TestClient(app)


def test_forged_webhooks_are_refused_on_every_route() -> None:
    app = FastAPI()
    app.include_router(telephony_router, prefix=PREFIX)
    client = TestClient(app)
    form = {"CallUUID": "call-1", "CallStatus": "busy", "RecordUrl": "https://x.io"}

    signed_routes = [
        ("POST", "/vobiz/answer"),
        ("POST", "/vobiz/callback/status"),
        ("POST", "/vobiz/callback/details"),
        ("POST", "/plivo/answer"),
        ("GET", "/plivo/answer"),
        ("POST", "/plivo/callback/status"),
        ("POST", "/plivo/callback/details"),
        ("POST", "/plivo/callback/transfer/dial-up"),
        ("GET", "/plivo/callback/transfer/dial-up"),
    ]
    for method, path in signed_routes:
        data = form if method == "POST" else None
        r = client.request(method, PREFIX + path, data=data)
        assert r.status_code == 401, (method, path, "unsigned")
        wrong = vobiz_headers(PREFIX + path, token="not-the-token")
        r = client.request(method, PREFIX + path, data=data, headers=wrong)
        assert r.status_code == 401, (method, path, "wrong vobiz signature")
        wrong = plivo_headers(PREFIX + path, form, "not-the-token")
        r = client.request(method, PREFIX + path, data=data, headers=wrong)
        assert r.status_code == 401, (method, path, "wrong plivo signature")

    gone_routes = [
        ("POST", "/twilio/callback/status"),
        ("POST", "/exotel/callback/status"),
        ("POST", "/exotel/answer"),
        ("GET", "/exotel/callback/details"),
        ("POST", "/twilio/callback/twiml-fallback"),
        ("POST", "/vobiz/callback/transfer/dial-up"),
    ]
    for method, path in gone_routes:
        data = form if method == "POST" else None
        r = client.request(method, PREFIX + path, data=data)
        assert r.status_code == 404, (method, path)


def test_vobiz_signature_passes_once() -> None:
    client = checked(security.verify_vobiz_webhook)

    assert client.post(CHECK_PATH, headers=vobiz_headers(CHECK_PATH)).status_code == 200
    assert client.post(CHECK_PATH, headers=vobiz_headers(CHECK_PATH)).status_code == 401
    fresh = vobiz_headers(CHECK_PATH, nonce="nonce-2")
    assert client.post(CHECK_PATH, headers=fresh).status_code == 200


def test_vobiz_signature_covers_the_path() -> None:
    client = checked(security.verify_vobiz_webhook)

    other_path = vobiz_headers("/agent/voice/breeze-buddy/vobiz/answer")
    assert client.post(CHECK_PATH, headers=other_path).status_code == 401


def test_vobiz_valid_signature_passes_when_redis_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_redis(monkeypatch, BrokenRedis())
    client = checked(security.verify_vobiz_webhook)

    assert client.post(CHECK_PATH, headers=vobiz_headers(CHECK_PATH)).status_code == 200
    assert client.post(CHECK_PATH).status_code == 401


def test_vobiz_refuses_everything_without_its_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "VOBIZ_AUTH_TOKEN", "")
    client = checked(security.verify_vobiz_webhook)

    signed_with_empty = vobiz_headers(CHECK_PATH, token="")
    assert client.post(CHECK_PATH, headers=signed_with_empty).status_code == 401


def test_plivo_signature_from_a_template_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leads = {"byo-call": SimpleNamespace(template_id="tmpl-byo")}

    async def get_lead(call_id: str):
        return leads.get(call_id)

    async def lead_account(lead):
        return SimpleNamespace(auth_token=MERCHANT_PLIVO_TOKEN)

    monkeypatch.setattr(security, "get_lead_by_call_id", get_lead)
    monkeypatch.setattr(security, "lead_plivo_account", lead_account)
    client = checked(security.verify_plivo_webhook)

    env = {"CallUUID": "env-call"}
    r = client.post(
        CHECK_PATH, data=env, headers=plivo_headers(CHECK_PATH, env, PLIVO_TOKEN)
    )
    assert r.status_code == 200

    byo = {"CallUUID": "byo-call"}
    headers = plivo_headers(CHECK_PATH, byo, MERCHANT_PLIVO_TOKEN)
    assert client.post(CHECK_PATH, data=byo, headers=headers).status_code == 200

    unknown = {"CallUUID": "env-call"}
    headers = plivo_headers(CHECK_PATH, unknown, MERCHANT_PLIVO_TOKEN)
    assert client.post(CHECK_PATH, data=unknown, headers=headers).status_code == 401

    headers = plivo_headers(CHECK_PATH, byo, "attacker-token")
    assert client.post(CHECK_PATH, data=byo, headers=headers).status_code == 401


async def _never(*args, **kwargs):
    raise AssertionError("not used")


def socket_for(url: str) -> WebSocket:
    parts = urlsplit(url)
    scope = {
        "type": "websocket",
        "path": parts.path,
        "headers": [],
        "query_string": parts.query.encode(),
    }
    return WebSocket(scope, receive=_never, send=_never)


def test_stream_ticket_from_the_url_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ans_mod, "APP_BASE_URL", BASE_URL)
    url = ans_mod._build_websocket_url(
        "vobiz", "tmpl-1", "+918000000901", to_number="919000000001", ivr_mode=True
    )
    assert "&ticket=" in url
    assert security.stream_ticket_ok(socket_for(url), "vobiz")
    assert not security.stream_ticket_ok(socket_for(url), "plivo")

    for name, value in [
        ("template_id", "tmpl-2"),
        ("from_number", "918000000999"),
        ("to_number", "919000000999"),
    ]:
        query = dict(socket_for(url).query_params)
        query[name] = value
        changed = "wss://x/p?" + "&".join(f"{k}={v}" for k, v in query.items())
        assert not security.stream_ticket_ok(socket_for(changed), "vobiz"), name

    without_ivr = url.replace("&ivr_mode=true", "")
    assert not security.stream_ticket_ok(socket_for(without_ivr), "vobiz")

    pod_url = ans_mod._build_websocket_url(
        "plivo",
        "tmpl-1",
        "918000000901",
        pod_ws_url="wss://bb.example.com/ws/pod/voice-agent-7/agent/voice/"
        "breeze-buddy/plivo/callback/ws/v2",
    )
    fallback_pod = pod_url.replace("/ws/pod/voice-agent-7", "")
    assert security.stream_ticket_ok(socket_for(fallback_pod), "plivo")


def test_stream_ticket_expiry_and_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    ticket = security.stream_ticket("plivo", "tmpl-1", "9180", "", "")
    url = f"wss://x/p?template_id=tmpl-1&from_number=9180&ticket={ticket}"
    assert security.stream_ticket_ok(socket_for(url), "plivo")

    expires_at = int(ticket.split(".")[0])
    monkeypatch.setattr(security, "time", SimpleNamespace(time=lambda: expires_at + 1))
    assert not security.stream_ticket_ok(socket_for(url), "plivo")
    monkeypatch.setattr(security, "time", SimpleNamespace(time=lambda: expires_at))
    assert security.stream_ticket_ok(socket_for(url), "plivo")

    no_ticket = "wss://x/p?template_id=tmpl-1&from_number=9180"
    assert not security.stream_ticket_ok(socket_for(no_ticket), "plivo")
    junk = "wss://x/p?template_id=tmpl-1&from_number=9180&ticket=abc"
    assert not security.stream_ticket_ok(socket_for(junk), "plivo")

    monkeypatch.setattr(security, "JWT_SECRET_KEY", "")
    assert not security.stream_ticket_ok(socket_for(url), "plivo")


def test_socket_without_a_ticket_is_closed() -> None:
    app = FastAPI()
    app.include_router(websocket_routes.router, prefix=PREFIX)
    client = TestClient(app)

    for provider in ("vobiz", "plivo", "twilio"):
        path = f"{PREFIX}/{provider}/callback/ws/v2?template_id=t&from_number=1"
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(path):
                pass
        assert closed.value.code == 1008, provider
