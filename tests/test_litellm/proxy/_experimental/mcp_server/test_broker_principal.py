"""Broker principal resolution: a valid broker token must resolve the inbound
principal so the existing per-user OAuth inject path fires.

The broker mints an HS256 master-key JWT carrying ``user_id=<principal>``.
``user_api_key_auth`` does NOT accept that token (it only accepts ``sk-``
virtual keys or asymmetric/OIDC JWTs), so the inbound flow falls through to an
anonymous ``UserAPIKeyAuth()`` (``user_id=None``). Without the fix here the
audience check passes (the token IS aud-bound) but
``server.py::_get_user_oauth_extra_headers_from_db`` returns ``None`` because
``user_id`` is falsy → the stored upstream token is never injected and the
backend MCP call is anonymous.

The fix: ``_validate_broker_token_audience`` ALSO returns the resolved
principal (the ``user_id`` claim it just validated, signature + ``aud`` + ``exp``),
and the call site sets ``UserAPIKeyAuth(user_id=principal)`` when the current
auth is anonymous. These tests prove that seam: minted-token → user_id ==
principal, while preserving the Task-6 negatives (foreign aud → 401, non-broker
target → no principal).
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.broker import mint_broker_token
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer

# ≥32 bytes — keeps PyJWT's InsecureKeyLengthWarning quiet (RFC 7518 §3.2).
_MASTER_KEY = "test-master-key-0123456789abcdef0123456789abcdef"
_BASE_URL = "https://llm.example.com"
_PRINCIPAL = "mcp-oauth:deadbeef0123456789abcdef01234567"

_GET_BASE_URL = (
    "litellm.proxy._experimental.mcp_server.oauth_utils.get_request_base_url"
)
_GET_BY_NAME = (
    "litellm.proxy._experimental.mcp_server.mcp_server_manager."
    "global_mcp_server_manager.get_mcp_server_by_name"
)
_MASTER_KEY_ATTR = "litellm.proxy.proxy_server.master_key"


def _server(**kw) -> MCPServer:
    base = dict(
        server_id="s1",
        name="gitlab",
        transport="http",
        auth_type="oauth2",
        broker=True,
    )
    base.update(kw)
    return MCPServer(**base)


def _resource(server: MCPServer) -> str:
    name = server.server_name or server.name
    return f"{_BASE_URL}/mcp/{name}"


def _token_for(
    server: MCPServer, *, principal: str = _PRINCIPAL, resource: str | None = None
) -> str:
    return mint_broker_token(
        principal=principal,
        server_id=server.server_id,
        resource=resource if resource is not None else _resource(server),
        master_key=_MASTER_KEY,
        ttl_seconds=3600,
    )


def _call(token: str, server, *, path: str = "/mcp/gitlab", mcp_servers=None):
    """Invoke the helper with the manager + base-url + master_key mocked.
    ``server`` may be a single MCPServer or a name→server callable for mixed
    targets."""
    request = MagicMock()  # only consumed by the (patched) get_request_base_url
    get_by_name_kw = (
        {"side_effect": server} if callable(server) else {"return_value": server}
    )
    with patch(_GET_BASE_URL, return_value=_BASE_URL), patch(
        _GET_BY_NAME, **get_by_name_kw
    ), patch(_MASTER_KEY_ATTR, _MASTER_KEY):
        return MCPRequestHandler._validate_broker_token_audience(
            request=request,
            mcp_servers=mcp_servers,
            request_route=path,
            token=token,
        )


# ── (1) Helper returns the principal for a VALID broker token ────────────────
def test_valid_broker_token_resolves_principal():
    """A valid minted broker token (aud matches the broker target) → the helper
    returns the principal (the ``user_id`` claim), no raise. This is the value
    the call site uses to build ``UserAPIKeyAuth(user_id=principal)``."""
    server = _server()
    principal = _call(_token_for(server), server)
    assert principal == _PRINCIPAL


def test_valid_broker_token_with_bearer_prefix_resolves_principal():
    """The inbound Authorization header carries a ``Bearer `` prefix; the helper
    strips it before decoding, so the principal still resolves."""
    server = _server()
    principal = _call("Bearer " + _token_for(server), server)
    assert principal == _PRINCIPAL


def test_principal_matches_inject_lookup_key():
    """The principal the helper returns is exactly the ``user_id`` claim the
    broker minted — the same key the upstream token is stored under server-side
    (``(user_id, server_id)``), so ``_get_user_oauth_extra_headers_from_db``
    resolves it. Cross-check by decoding the same way that fn's owner does."""
    import jwt

    server = _server()
    token = _token_for(server, principal="mcp-oauth:abc123")
    resolved = _call(token, server)
    claims = jwt.decode(
        token, _MASTER_KEY, algorithms=["HS256"], options={"verify_aud": False}
    )
    assert resolved == claims["user_id"] == "mcp-oauth:abc123"


# ── (2) The call-site override builds a principal-bearing auth ───────────────
def test_call_site_override_sets_user_id_when_anonymous():
    """Mirror the call-site override: when the helper resolves a principal AND
    the current auth is the anonymous fallback (user_id falsy), the override
    must produce a ``UserAPIKeyAuth(user_id=principal)`` so the inject path
    fires. Drives the real helper for the principal, then applies the exact
    override predicate."""
    server = _server()
    validated = UserAPIKeyAuth()  # anonymous fallback (user_id=None)
    assert not validated.user_id

    principal = _call(_token_for(server), server)
    # The call-site predicate: resolved principal + falsy current user_id.
    if principal and not validated.user_id:
        validated = UserAPIKeyAuth(user_id=principal)

    assert validated.user_id == _PRINCIPAL


def test_call_site_override_does_not_clobber_explicit_key_auth():
    """If an explicit ``x-litellm-api-key`` already resolved a real user, the
    override must NOT overwrite it even though a broker token resolves a
    principal (do-not-clobber guard: only act on the anonymous fallback)."""
    server = _server()
    validated = UserAPIKeyAuth(user_id="real-litellm-user")

    principal = _call(_token_for(server), server)
    if principal and not validated.user_id:
        validated = UserAPIKeyAuth(user_id=principal)

    # Untouched — the explicit-key user wins.
    assert validated.user_id == "real-litellm-user"


# ── (3) Negatives preserved (Task-6 behaviour) ──────────────────────────────
def test_foreign_audience_still_rejected_no_principal_leak():
    """A token whose aud is for a DIFFERENT broker server must still 401 — the
    helper must not leak a principal for a foreign-aud token."""
    server = _server()
    foreign = _token_for(server, resource=f"{_BASE_URL}/mcp/other")
    with pytest.raises(HTTPException) as exc:
        _call(foreign, server)
    assert exc.value.status_code == 401
    assert exc.value.detail["error"] == "invalid_token"


def test_non_broker_target_resolves_no_principal():
    """A non-broker (relay/api_key/byok) target → the helper is a no-op and
    resolves NO principal (returns None), so the call-site override never fires
    and the flow stays byte-unchanged."""
    non_broker = _server(broker=False)
    assert non_broker.is_oauth_broker is False
    # Even a valid-looking token resolves nothing for a non-broker target.
    assert _call("totally-not-a-valid-token", non_broker) is None


def test_unresolvable_target_resolves_no_principal():
    """If the target name resolves to no server, there is no broker → no
    principal resolved (None)."""
    request = MagicMock()
    with patch(_GET_BASE_URL, return_value=_BASE_URL), patch(
        _GET_BY_NAME, return_value=None
    ), patch(_MASTER_KEY_ATTR, _MASTER_KEY):
        assert (
            MCPRequestHandler._validate_broker_token_audience(
                request=request,
                mcp_servers=None,
                request_route="/mcp/gitlab",
                token="anything",
            )
            is None
        )


def test_mixed_targets_valid_broker_token_resolves_principal():
    """When multiple servers are targeted and one is a broker, a token valid for
    the broker resolves that broker's principal (the inject path will then look
    up the stored token for that broker's server_id)."""
    broker = _server(server_id="s1", name="gitlab", broker=True)
    relay = _server(server_id="s2", name="relay", broker=False)

    def _by_name(name, client_ip=None):
        return {"gitlab": broker, "relay": relay}.get(name)

    correct = _token_for(broker, resource=f"{_BASE_URL}/mcp/gitlab")
    principal = _call(
        correct,
        _by_name,
        path="/some/non-mcp-path",
        mcp_servers=["gitlab", "relay"],
    )
    assert principal == _PRINCIPAL
