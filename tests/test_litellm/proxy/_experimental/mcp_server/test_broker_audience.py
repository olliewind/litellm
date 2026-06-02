"""Inbound audience validation for broker-mode MCP servers (RFC 8707 §2 /
OAuth 2.1 §5.2).

A broker MCP server must reject a token whose ``aud`` is for a different
server (or a non-broker / raw token). These tests drive
``MCPRequestHandler._validate_broker_token_audience`` directly so the check
is exercised without building a full ASGI scope through
``process_mcp_request``.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    MCPRequestHandler,
)
from litellm.proxy._experimental.mcp_server.broker import mint_broker_token
from litellm.types.mcp_server.mcp_server_manager import MCPServer

# ≥32 bytes — keeps PyJWT's InsecureKeyLengthWarning quiet (RFC 7518 §3.2).
_MASTER_KEY = "test-master-key-0123456789abcdef0123456789abcdef"
_BASE_URL = "https://llm.example.com"

# Where the helper looks these up (lazy imports inside the function resolve to
# the defining modules, so patch them at their source).
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


def _token_for(server: MCPServer, *, resource: str | None = None) -> str:
    return mint_broker_token(
        principal="mcp-oauth:abc",
        server_id=server.server_id,
        resource=resource if resource is not None else _resource(server),
        master_key=_MASTER_KEY,
        ttl_seconds=3600,
    )


def _call(token: str, server: MCPServer, *, path: str = "/mcp/gitlab"):
    """Invoke the helper with the manager + base-url + master_key mocked."""
    request = MagicMock()  # only consumed by the (patched) get_request_base_url
    with patch(_GET_BASE_URL, return_value=_BASE_URL), patch(
        _GET_BY_NAME, return_value=server
    ), patch(_MASTER_KEY_ATTR, _MASTER_KEY):
        return MCPRequestHandler._validate_broker_token_audience(
            request=request,
            mcp_servers=None,
            request_route=path,
            token=token,
        )


# (a) matching aud → no raise, resolves principal -----------------------------
def test_matching_audience_passes():
    server = _server()
    # Does not raise; on success the helper resolves the principal (the
    # ``user_id`` claim ``_token_for`` minted) so the inject path can fire.
    assert _call(_token_for(server), server) == "mcp-oauth:abc"


def test_matching_audience_with_bearer_prefix_passes():
    """The inbound Authorization header value carries a ``Bearer `` prefix —
    the helper must strip it before decoding (jwt.decode would otherwise fail
    and reject a legitimate token)."""
    server = _server()
    assert _call("Bearer " + _token_for(server), server) == "mcp-oauth:abc"


# (b) foreign aud → 401 -------------------------------------------------------
def test_foreign_audience_rejected():
    server = _server()
    foreign = _token_for(server, resource=f"{_BASE_URL}/mcp/other")
    with pytest.raises(HTTPException) as exc:
        _call(foreign, server)
    assert exc.value.status_code == 401
    assert exc.value.detail["error"] == "invalid_token"


# (c) non-broker target → helper is a no-op ----------------------------------
def test_non_broker_target_is_noop():
    """A relay/delegate/api_key/byok server is NOT is_oauth_broker, so the
    helper must return without inspecting the token at all — even a clearly
    foreign / garbage token passes through untouched here."""
    non_broker = _server(broker=False)
    assert non_broker.is_oauth_broker is False
    assert _call("totally-not-a-valid-token", non_broker) is None


def test_non_oauth2_broker_flag_is_noop():
    """broker=True but auth_type != oauth2 is not is_oauth_broker."""
    server = _server(broker=True, auth_type="bearer_token")
    assert server.is_oauth_broker is False
    assert _call("garbage", server) is None


# (d) raw / non-JWT token presented to a broker target → 401 -----------------
def test_raw_token_to_broker_rejected():
    server = _server()
    with pytest.raises(HTTPException) as exc:
        _call("ghp_rawUpstreamGithubToken", server)
    assert exc.value.status_code == 401
    assert exc.value.detail["error"] == "invalid_token"


def test_token_signed_with_wrong_key_rejected():
    """A JWT with the right aud but signed by something other than master_key
    must fail closed (an attacker cannot forge audience binding)."""
    server = _server()
    forged = mint_broker_token(
        principal="p",
        server_id=server.server_id,
        resource=_resource(server),
        master_key="a-different-key-0123456789abcdef0123456789ab",
        ttl_seconds=3600,
    )
    with pytest.raises(HTTPException) as exc:
        _call(forged, server)
    assert exc.value.status_code == 401


# server_name preferred over name when computing the resource ----------------
def test_resource_uses_server_name_when_present():
    server = _server(name="display-name", server_name="gitlab")
    # Token aud must match server_name-derived resource, not name-derived.
    good = _token_for(server, resource=f"{_BASE_URL}/mcp/gitlab")
    # Valid aud → resolves the minted principal (no raise).
    assert _call(good, server, path="/mcp/gitlab") == "mcp-oauth:abc"

    bad = _token_for(server, resource=f"{_BASE_URL}/mcp/display-name")
    with pytest.raises(HTTPException) as exc:
        _call(bad, server, path="/mcp/gitlab")
    assert exc.value.status_code == 401


# mixed targets: one broker among several → broker still enforced ------------
def test_mixed_targets_broker_enforced():
    """When the request targets multiple servers and ANY is a broker, the
    presented token must be valid for that broker server."""
    broker = _server(server_id="s1", name="gitlab", broker=True)
    relay = _server(server_id="s2", name="relay", broker=False)

    def _by_name(name, client_ip=None):
        return {"gitlab": broker, "relay": relay}.get(name)

    foreign = _token_for(broker, resource=f"{_BASE_URL}/mcp/other")
    request = MagicMock()
    with patch(_GET_BASE_URL, return_value=_BASE_URL), patch(
        _GET_BY_NAME, side_effect=_by_name
    ), patch(_MASTER_KEY_ATTR, _MASTER_KEY):
        with pytest.raises(HTTPException) as exc:
            MCPRequestHandler._validate_broker_token_audience(
                request=request,
                mcp_servers=["gitlab", "relay"],
                request_route="/some/non-mcp-path",
                token=foreign,
            )
    assert exc.value.status_code == 401


def test_valid_broker_token_with_mixed_targets_passes():
    """A token with the correct audience for the broker server must NOT raise,
    even when another target in the same request is a non-broker (relay) server.
    On success the helper resolves the broker's principal."""
    broker = _server(server_id="s1", name="gitlab", broker=True)
    relay = _server(server_id="s2", name="relay", broker=False)

    def _by_name(name, client_ip=None):
        return {"gitlab": broker, "relay": relay}.get(name)

    # Mint a token whose aud matches the broker server's expected resource.
    correct_token = _token_for(broker, resource=f"{_BASE_URL}/mcp/gitlab")
    request = MagicMock()
    with patch(_GET_BASE_URL, return_value=_BASE_URL), patch(
        _GET_BY_NAME, side_effect=_by_name
    ), patch(_MASTER_KEY_ATTR, _MASTER_KEY):
        result = MCPRequestHandler._validate_broker_token_audience(
            request=request,
            mcp_servers=["gitlab", "relay"],
            request_route="/some/non-mcp-path",
            token=correct_token,
        )
    assert result == "mcp-oauth:abc"


def test_unresolvable_target_is_noop():
    """If the targeted name resolves to no server, there is no broker to
    enforce against — the helper is a no-op (fails open to the rest of the
    flow, which fails closed elsewhere)."""
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
