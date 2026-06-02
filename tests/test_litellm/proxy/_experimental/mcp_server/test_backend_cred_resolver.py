import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.proxy._types import UserAPIKeyAuth

S = "litellm.proxy._experimental.mcp_server.server"


def _srv(**kw):
    base = dict(server_id="s1", name="gitlab", server_name="gitlab", transport="http", auth_type="oauth2")
    base.update(kw)
    return MCPServer(**base)


@pytest.mark.asyncio
async def test_oauth2_user_replaces_client_header_with_stored_token():
    srv = _srv(broker=True)  # is_oauth_broker + needs_user_oauth_token
    auth = UserAPIKeyAuth(api_key="x", user_id="mcp-oauth:abc")
    with patch(f"{S}._get_user_oauth_extra_headers_from_db",
               new=AsyncMock(return_value={"Authorization": "Bearer STORED"})):
        from litellm.proxy._experimental.mcp_server.server import _apply_user_oauth_auth
        out = await _apply_user_oauth_auth(srv, auth, {"Authorization": "Bearer CLIENT-JWT"})
    assert out == {"Authorization": "Bearer STORED"}   # client header replaced


@pytest.mark.asyncio
async def test_broker_with_no_stored_token_raises_401_not_passthrough():
    srv = _srv(broker=True)
    auth = UserAPIKeyAuth(api_key="x", user_id="mcp-oauth:abc")
    with patch(f"{S}._get_user_oauth_extra_headers_from_db", new=AsyncMock(return_value=None)):
        from litellm.proxy._experimental.mcp_server.server import _apply_user_oauth_auth
        with pytest.raises(HTTPException) as ei:
            await _apply_user_oauth_auth(srv, auth, {"Authorization": "Bearer CLIENT-JWT"})
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_relay_no_token_keeps_client_header():
    srv = _srv(broker=False, delegate_auth_to_upstream=True)  # relay: not a broker -> no 401 (function gates on is_oauth_broker, not the delegate flag)
    auth = UserAPIKeyAuth(api_key="x", user_id="u1")
    with patch(f"{S}._get_user_oauth_extra_headers_from_db", new=AsyncMock(return_value=None)):
        from litellm.proxy._experimental.mcp_server.server import _apply_user_oauth_auth
        out = await _apply_user_oauth_auth(srv, auth, {"Authorization": "Bearer CLIENT-TOKEN"})
    assert out == {"Authorization": "Bearer CLIENT-TOKEN"}   # relay forwards client (unchanged)


@pytest.mark.asyncio
async def test_non_oauth2_unchanged():
    srv = _srv(auth_type="bearer_token")
    auth = UserAPIKeyAuth(api_key="x", user_id="u1")
    from litellm.proxy._experimental.mcp_server.server import _apply_user_oauth_auth
    out = await _apply_user_oauth_auth(srv, auth, {"Authorization": "static"})
    assert out == {"Authorization": "static"}


@pytest.mark.asyncio
async def test_user_api_key_auth_none_returns_unchanged():
    srv = _srv(broker=True)
    from litellm.proxy._experimental.mcp_server.server import _apply_user_oauth_auth
    out = await _apply_user_oauth_auth(srv, None, {"Authorization": "Bearer CLIENT"})
    assert out == {"Authorization": "Bearer CLIENT"}   # no principal -> no override, no 401


@pytest.mark.asyncio
async def test_m2m_with_extra_headers_unchanged():
    srv = _srv(oauth2_flow="client_credentials")  # has_client_credentials -> needs_user_oauth_token False
    auth = UserAPIKeyAuth(api_key="x", user_id="u1")
    from litellm.proxy._experimental.mcp_server.server import _apply_user_oauth_auth
    out = await _apply_user_oauth_auth(srv, auth, {"Authorization": "Bearer M2M"})
    assert out == {"Authorization": "Bearer M2M"}


@pytest.mark.asyncio
async def test_m2m_no_extra_headers_uses_fallback_fetch():
    srv = _srv(oauth2_flow="client_credentials")
    auth = UserAPIKeyAuth(api_key="x", user_id="u1")
    with patch(f"{S}._get_user_oauth_extra_headers_from_db", new=AsyncMock(return_value=None)) as f:
        from litellm.proxy._experimental.mcp_server.server import _apply_user_oauth_auth
        out = await _apply_user_oauth_auth(srv, auth, None)
    assert out is None
    f.assert_awaited()   # pre-existing fallback fetch preserved
