import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.types.mcp_server.mcp_server_manager import MCPServer

D = "litellm.proxy._experimental.mcp_server.discoverable_endpoints"


def _broker_server():
    return MCPServer(server_id="s1", name="gitlab", server_name="gitlab", transport="http",
                     auth_type="oauth2", broker=True, url="http://backend/mcp",
                     authorization_url="https://up/authorize", token_url="https://up/token",
                     client_id="cid", client_secret="csecret")


@pytest.mark.asyncio
async def test_broker_authorize_uses_own_pkce_not_clients():
    server = _broker_server()
    from unittest.mock import MagicMock
    req = MagicMock()
    with (
        patch(f"{D}.validate_trusted_redirect_uri"),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}.encrypt_value_helper", return_value="mocked_encrypted_state"),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import broker_authorize
        resp = await broker_authorize(request=req, mcp_server=server, client_id="claude",
                                      redirect_uri="https://claude/cb", state="client-state",
                                      code_challenge="CLIENT-CHALLENGE", code_challenge_method="S256",
                                      scope="api", consented=True)
    loc = resp.headers["location"]
    assert loc.startswith("https://up/authorize")
    assert "client_id=cid" in loc
    assert "CLIENT-CHALLENGE" not in loc            # client's PKCE not forwarded upstream
    assert "code_challenge=" in loc                 # LiteLLM's own challenge present
    assert "redirect_uri=https%3A%2F%2Fllm.example.com%2Fcallback" in loc


@pytest.mark.asyncio
async def test_broker_authorize_shows_consent_first():
    server = _broker_server()
    from unittest.mock import MagicMock
    with patch(f"{D}.validate_trusted_redirect_uri"):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import broker_authorize
        resp = await broker_authorize(request=MagicMock(), mcp_server=server, client_id="claude",
                                      redirect_uri="https://claude/cb", state="s",
                                      code_challenge="C", code_challenge_method="S256",
                                      scope="api", consented=False)
    assert resp.status_code == 200
    body = resp.body.decode()
    assert "Allow" in body          # user-visible action button
    assert "Authorize" in body      # user-visible heading
    assert "consented" in body      # hidden field still present


@pytest.mark.asyncio
async def test_broker_authorize_rejects_missing_client_id():
    from fastapi import HTTPException
    from unittest.mock import MagicMock
    server = _broker_server()
    server.client_id = None
    with patch(f"{D}.validate_trusted_redirect_uri"):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import broker_authorize
        with pytest.raises(HTTPException) as ei:
            await broker_authorize(request=MagicMock(), mcp_server=server, client_id="claude",
                                   redirect_uri="https://claude/cb", state="s", code_challenge="C",
                                   code_challenge_method="S256", scope="api", consented=True)
    assert ei.value.status_code == 500


@pytest.mark.asyncio
async def test_broker_authorize_rejects_missing_authorization_url():
    from fastapi import HTTPException
    from unittest.mock import MagicMock
    server = _broker_server()
    server.authorization_url = None
    with patch(f"{D}.validate_trusted_redirect_uri"):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import broker_authorize
        with pytest.raises(HTTPException) as ei:
            await broker_authorize(request=MagicMock(), mcp_server=server, client_id="claude",
                                   redirect_uri="https://claude/cb", state="s", code_challenge="C",
                                   code_challenge_method="S256", scope="api", consented=False)
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_broker_callback_stores_upstream_and_issues_litellm_code():
    server = _broker_server()
    state_data = {
        "base_url": "https://claude/cb", "original_state": "client-state",
        "code_challenge": "CLIENT-CHALLENGE", "code_challenge_method": "S256",
        "client_redirect_uri": "https://claude/cb",
        "broker_server_id": "s1", "broker_code_verifier": "litellm-verifier",
    }
    stored = {}
    async def _store(server, user_id, token_response, raise_on_error=False):
        stored[(user_id, server.server_id)] = token_response["access_token"]

    with (
        patch(f"{D}.decode_state_hash", return_value=state_data),
        patch(f"{D}._get_validated_client_redirect_uri", return_value="https://claude/cb"),
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}._exchange_code_for_token_dict",
              new=AsyncMock(return_value={"access_token": "UPSTREAM-RAW", "refresh_token": "R"})) as exch,
        patch(f"{D}._store_per_user_token_server_side", new=_store),
        patch(f"{D}.encrypt_value_helper", side_effect=lambda s: "ENC(" + s + ")"),
        patch(f"{D}.establish_principal", return_value="mcp-oauth:abc"),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import callback
        resp = await callback(request=MagicMock(), code="UPSTREAM-CODE", state="enc-state")

    assert exch.await_args.kwargs.get("code_verifier") == "litellm-verifier"  # LiteLLM's verifier, not client's
    assert stored[("mcp-oauth:abc", "s1")] == "UPSTREAM-RAW"                   # stored server-side under principal
    loc = resp.headers["location"]
    assert loc.startswith("https://claude/cb")
    assert "state=client-state" in loc
    assert "UPSTREAM-RAW" not in loc and "UPSTREAM-CODE" not in loc            # NO passthrough
    assert "code=ENC" in loc                                                  # a LiteLLM-minted code


@pytest.mark.asyncio
async def test_broker_callback_redirects_error_when_exchange_fails():
    server = _broker_server()
    state_data = {
        "base_url": "https://claude/cb", "original_state": "client-state",
        "code_challenge": "CLIENT-CHALLENGE", "code_challenge_method": "S256",
        "client_redirect_uri": "https://claude/cb",
        "broker_server_id": "s1", "broker_code_verifier": "litellm-verifier",
    }
    stored = {}
    async def _store(server, user_id, token_response, raise_on_error=False):
        stored[(user_id, server.server_id)] = token_response["access_token"]
    with (
        patch(f"{D}.decode_state_hash", return_value=state_data),
        patch(f"{D}._get_validated_client_redirect_uri", return_value="https://claude/cb"),
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}._exchange_code_for_token_dict", new=AsyncMock(side_effect=Exception("upstream 400"))),
        patch(f"{D}._store_per_user_token_server_side", new=_store),
        patch(f"{D}.encrypt_value_helper", side_effect=lambda s: "ENC(" + s + ")"),
        patch(f"{D}.establish_principal", return_value="mcp-oauth:abc"),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import callback
        resp = await callback(request=MagicMock(), code="UPSTREAM-CODE", state="enc-state")
    loc = resp.headers["location"]
    assert "error=server_error" in loc
    assert "state=client-state" in loc
    assert stored == {}                  # nothing stored when exchange fails
    assert "code=ENC" not in loc         # no LiteLLM code minted


@pytest.mark.asyncio
async def test_broker_callback_redirects_error_when_store_fails():
    server = _broker_server()
    state_data = {
        "base_url": "https://claude/cb", "original_state": "client-state",
        "code_challenge": "CLIENT-CHALLENGE", "code_challenge_method": "S256",
        "client_redirect_uri": "https://claude/cb",
        "broker_server_id": "s1", "broker_code_verifier": "litellm-verifier",
    }
    async def _store(server, user_id, token_response, raise_on_error=False):
        if raise_on_error:
            raise Exception("DB down")
    with (
        patch(f"{D}.decode_state_hash", return_value=state_data),
        patch(f"{D}._get_validated_client_redirect_uri", return_value="https://claude/cb"),
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}._exchange_code_for_token_dict",
              new=AsyncMock(return_value={"access_token": "UPSTREAM-RAW"})),
        patch(f"{D}._store_per_user_token_server_side", new=_store),
        patch(f"{D}.encrypt_value_helper", side_effect=lambda s: "ENC(" + s + ")"),
        patch(f"{D}.establish_principal", return_value="mcp-oauth:abc"),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import callback
        resp = await callback(request=MagicMock(), code="UPSTREAM-CODE", state="enc-state")
    loc = resp.headers["location"]
    assert "error=server_error" in loc       # store failure surfaces as OAuth error
    assert "code=ENC" not in loc             # no code minted when store failed
