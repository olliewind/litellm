import pytest
from unittest.mock import patch
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
