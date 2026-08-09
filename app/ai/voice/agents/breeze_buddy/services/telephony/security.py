"""Telephony webhook and media socket checks (PT-02/05/12/23).

Plivo and Vobiz webhooks must carry a valid V3 signature. Media sockets must
carry the stream ticket that the answer handler put in the stream URL.
"""

import base64
import hashlib
import hmac
import time

from fastapi import HTTPException, Request, WebSocket
from plivo.utils import validate_v3_signature

from app.ai.voice.agents.breeze_buddy.accounts import AccountRefused
from app.ai.voice.agents.breeze_buddy.services.telephony.plivo.account import (
    lead_plivo_account,
)
from app.core.config.static import (
    APP_BASE_URL,
    JWT_SECRET_KEY,
    PLIVO_AUTH_TOKEN,
    VOBIZ_AUTH_TOKEN,
)
from app.core.logger import logger
from app.database.accessor import get_lead_by_call_id
from app.services.redis import get_redis_service

STREAM_TICKET_TTL_SECONDS = 5 * 60
VOBIZ_NONCE_TTL_SECONDS = 24 * 60 * 60


def _plivo_v3_ok(
    method: str, url: str, nonce: str, token: str, signature: str, params: dict
) -> bool:
    try:
        return bool(
            validate_v3_signature(method, url, nonce, token, signature, params or None)
        )
    except Exception:
        return False


async def verify_plivo_webhook(request: Request) -> None:
    """401 unless a Plivo webhook carries a valid X-Plivo-Signature-V3.

    V2 is not accepted: it signs only the URL path, so a replayed V2 header
    would authenticate any body. The URL is rebuilt from APP_BASE_URL because
    behind a load balancer request.url is an internal host.

    The environment's token is tried first. A call placed on a template's own
    Plivo account is signed with that account's token, so the token of the
    lead's account is tried next, the lead found by the request's CallUUID.
    """
    url = APP_BASE_URL.rstrip("/") + request.url.path
    if request.url.query:
        url += f"?{request.url.query}"
    params = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    signature = request.headers.get("X-Plivo-Signature-V3", "")
    nonce = request.headers.get("X-Plivo-Signature-V3-Nonce", "")
    method = request.method

    if signature and nonce:
        if PLIVO_AUTH_TOKEN and _plivo_v3_ok(
            method, url, nonce, PLIVO_AUTH_TOKEN, signature, params
        ):
            return
        call_uuid = params.get("CallUUID") or request.query_params.get("CallUUID")
        lead = await get_lead_by_call_id(call_uuid) if call_uuid else None
        if lead is not None:
            try:
                account = await lead_plivo_account(lead)
            except AccountRefused:
                account = None
            if (
                account is not None
                and account.auth_token != PLIVO_AUTH_TOKEN
                and _plivo_v3_ok(
                    method, url, nonce, account.auth_token, signature, params
                )
            ):
                return

    logger.warning(
        f"Rejected unsigned/forged plivo webhook: expected a Plivo signature "
        f"for {APP_BASE_URL.rstrip('/')}{request.url.path}; got "
        f"signature={'present' if signature else 'missing'}, "
        f"nonce={'present' if nonce else 'missing'}"
        + ("" if PLIVO_AUTH_TOKEN else ", PLIVO_AUTH_TOKEN unset")
    )
    raise HTTPException(status_code=401, detail="Webhook signature verification failed")


async def verify_vobiz_webhook(request: Request) -> None:
    """401 unless a Vobiz webhook carries a valid, unused X-Vobiz-Signature-V3.

    V3 = base64(HMAC-SHA256(VOBIZ_AUTH_TOKEN, URL without query + "." + nonce)).
    It signs neither the query nor the body, so each nonce is accepted once
    (Redis SET NX, 24 h), as Vobiz advises. When Redis fails, the valid
    signature alone lets the request through and a warning is logged.
    """
    url = APP_BASE_URL.rstrip("/") + request.url.path
    signature = request.headers.get("X-Vobiz-Signature-V3", "")
    nonce = request.headers.get("X-Vobiz-Signature-V3-Nonce", "")

    valid = False
    if signature and nonce and VOBIZ_AUTH_TOKEN:
        digest = hmac.new(
            VOBIZ_AUTH_TOKEN.encode(), f"{url}.{nonce}".encode(), hashlib.sha256
        ).digest()
        valid = hmac.compare_digest(base64.b64encode(digest).decode(), signature)
    if not valid:
        logger.warning(
            f"Rejected unsigned/forged vobiz webhook: expected a Vobiz signature "
            f"for {url}; got signature={'present' if signature else 'missing'}, "
            f"nonce={'present' if nonce else 'missing'}"
            + ("" if VOBIZ_AUTH_TOKEN else ", VOBIZ_AUTH_TOKEN unset")
        )
        raise HTTPException(
            status_code=401, detail="Webhook signature verification failed"
        )

    try:
        client = await (await get_redis_service()).get_client()
        first_use = await client.set(
            f"telephony_nonce:vobiz:{nonce}", "1", nx=True, ex=VOBIZ_NONCE_TTL_SECONDS
        )
    except Exception as e:
        logger.warning(
            f"Vobiz webhook nonce check skipped, Redis failed: {type(e).__name__}"
        )
        return
    if not first_use:
        logger.warning(f"Rejected replayed vobiz webhook for {url}")
        raise HTTPException(
            status_code=401, detail="Webhook signature verification failed"
        )


def _stream_ticket_signature(
    provider: str,
    template_id: str,
    from_number: str,
    to_number: str,
    ivr_mode: str,
    expires_at: str,
) -> str:
    key = hmac.new(
        JWT_SECRET_KEY.encode(), b"telephony-stream-ticket", hashlib.sha256
    ).digest()
    message = "|".join(
        [provider.lower(), template_id, from_number, to_number, ivr_mode, expires_at]
    )
    return hmac.new(key, message.encode(), hashlib.sha256).hexdigest()


def stream_ticket(
    provider: str, template_id: str, from_number: str, to_number: str, ivr_mode: str
) -> str:
    """The ticket the answer handler puts in a stream URL: "<expiry>.<signature>".

    It signs the provider and the stream URL's query values, so a socket can
    neither be opened without it nor point a ticket at another template or
    caller. It does not sign the path: Smart Router's /ws/pod/<pod> prefix and
    the ingress fallback pod leave it valid.
    """
    expires_at = str(int(time.time()) + STREAM_TICKET_TTL_SECONDS)
    signature = _stream_ticket_signature(
        provider, template_id, from_number, to_number, ivr_mode, expires_at
    )
    return f"{expires_at}.{signature}"


def stream_ticket_ok(websocket: WebSocket, provider: str) -> bool:
    """True if a media socket's URL carries a valid, unexpired stream ticket."""
    query = websocket.query_params
    expires_at, _, signature = query.get("ticket", "").partition(".")
    if not JWT_SECRET_KEY:
        logger.warning("Media websocket ticket: JWT_SECRET_KEY is unset")
        return False
    if not (expires_at.isdigit() and signature):
        logger.warning(f"Media websocket ticket: missing on a {provider} socket")
        return False
    if int(expires_at) < time.time():
        logger.warning(f"Media websocket ticket: expired on a {provider} socket")
        return False
    expected = _stream_ticket_signature(
        provider,
        query.get("template_id", ""),
        query.get("from_number", ""),
        query.get("to_number", ""),
        query.get("ivr_mode", ""),
        expires_at,
    )
    if not hmac.compare_digest(expected, signature):
        logger.warning(
            f"Media websocket ticket: wrong signature on a {provider} socket"
        )
        return False
    return True
