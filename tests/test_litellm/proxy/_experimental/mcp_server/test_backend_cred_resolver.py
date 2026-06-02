import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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


@pytest.mark.asyncio
async def test_resolver_applies_override_after_prepare():
    srv = _srv(broker=True)
    auth = UserAPIKeyAuth(api_key="x", user_id="mcp-oauth:abc")
    with (
        patch(f"{S}._prepare_mcp_server_headers", return_value=("SAH", {"Authorization": "Bearer CLIENT"})),
        patch(f"{S}._get_user_oauth_extra_headers_from_db", new=AsyncMock(return_value={"Authorization": "Bearer STORED"})),
    ):
        from litellm.proxy._experimental.mcp_server.server import resolve_mcp_server_headers
        sah, eh = await resolve_mcp_server_headers(
            server=srv, user_api_key_auth=auth, mcp_server_auth_headers=None,
            mcp_auth_header=None, oauth2_headers={"Authorization": "Bearer CLIENT"}, raw_headers=None,
        )
    assert sah == "SAH"                                   # base server_auth_header preserved
    assert eh == {"Authorization": "Bearer STORED"}       # override applied


@pytest.mark.asyncio
async def test_execute_mcp_tool_injects_stored_token_for_broker():
    srv = _srv(broker=True)
    auth = UserAPIKeyAuth(api_key="x", user_id="mcp-oauth:abc")
    captured = {}

    async def _managed(**kwargs):
        captured.update(kwargs)
        return MagicMock(content=[], isError=False)

    with (
        patch(f"{S}._handle_managed_mcp_tool", new=_managed),
        patch(f"{S}._get_user_oauth_extra_headers_from_db",
              new=AsyncMock(return_value={"Authorization": "Bearer STORED"})),
        patch(f"{S}.global_mcp_tool_registry") as reg,
        patch(f"{S}.global_mcp_server_manager") as mgr,
        patch(f"{S}.MCPRequestHandler") as rh,
    ):
        reg.get_tool.return_value = None  # not a local tool -> managed path
        mgr._get_mcp_server_from_tool_name.return_value = srv
        rh.is_tool_allowed.return_value = True
        from litellm.proxy._experimental.mcp_server.server import execute_mcp_tool
        await execute_mcp_tool(
            name="gitlab/whoami", arguments={},
            allowed_mcp_servers=[srv], start_time=MagicMock(),
            user_api_key_auth=auth,
            mcp_auth_header=None, oauth2_headers={"Authorization": "Bearer CLIENT-JWT"},
            raw_headers=None,
        )
    # the managed dispatch must receive the STORED token, never the client's JWT
    assert captured["oauth2_headers"] == {"Authorization": "Bearer STORED"}


@pytest.mark.asyncio
async def test_broker_never_forwards_client_bearer():
    srv = _srv(broker=True)
    auth = UserAPIKeyAuth(api_key="x", user_id="mcp-oauth:abc")
    client = {"Authorization": "Bearer CLIENT-JWT"}
    with patch(f"{S}._get_user_oauth_extra_headers_from_db",
               new=AsyncMock(return_value={"Authorization": "Bearer STORED"})):
        from litellm.proxy._experimental.mcp_server.server import _apply_user_oauth_auth
        out = await _apply_user_oauth_auth(srv, auth, client)
    assert out != client and out == {"Authorization": "Bearer STORED"}
