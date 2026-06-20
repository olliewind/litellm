import base64
import hashlib
import json
import pytest
import jwt
from fastapi import HTTPException
from fastapi.responses import JSONResponse
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


@pytest.mark.asyncio
async def test_broker_token_endpoint_mints_for_valid_verifier():
    server = _broker_server()
    verifier = "the-verifier"
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    data = {"broker_code": True, "principal": "mcp-oauth:abc", "server_id": "s1",
            "client_code_challenge": challenge, "client_redirect_uri": "https://claude/cb"}
    with (
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}._get_broker_master_key", return_value="test-master-key-0123456789abcdef0123456789abcdef"),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import broker_token_mint
        result = await broker_token_mint(request=MagicMock(), data=data, code_verifier=verifier)
    claims = jwt.decode(result["access_token"], "test-master-key-0123456789abcdef0123456789abcdef",
                        algorithms=["HS256"], options={"verify_aud": False})
    assert claims["token_type"] == "mcp_broker"
    assert claims["aud"] == "https://llm.example.com/mcp/gitlab"
    assert claims["user_id"] == "mcp-oauth:abc"
    assert result["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_broker_token_endpoint_rejects_bad_verifier():
    server = _broker_server()
    data = {"broker_code": True, "principal": "p", "server_id": "s1",
            "client_code_challenge": "OTHER-CHALLENGE", "client_redirect_uri": "https://claude/cb"}
    with patch(f"{D}.get_mcp_server_by_id", return_value=server):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import broker_token_mint
        with pytest.raises(HTTPException) as ei:
            await broker_token_mint(request=MagicMock(), data=data, code_verifier="wrong-verifier")
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_broker_token_mint_rejects_blob_missing_ids():
    verifier = "v"
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    data = {"broker_code": True, "client_code_challenge": challenge}  # PKCE ok, but no server_id/principal
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import broker_token_mint
    with pytest.raises(HTTPException) as ei:
        await broker_token_mint(request=MagicMock(), data=data, code_verifier=verifier)
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_token_endpoint_routes_broker_code_to_mint():
    server = _broker_server()
    verifier = "the-verifier"
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    blob = json.dumps({"broker_code": True, "principal": "mcp-oauth:abc", "server_id": "s1",
                       "client_code_challenge": challenge, "client_redirect_uri": "https://claude/cb"})
    with (
        patch(f"{D}.decrypt_value_helper", return_value=blob),
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}._get_broker_master_key", return_value="test-master-key-0123456789abcdef0123456789abcdef"),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import token_endpoint
        resp = await token_endpoint(request=MagicMock(), grant_type="authorization_code",
                                    code="enc-broker-code", redirect_uri=None, client_id="claude",
                                    client_secret=None, code_verifier=verifier, refresh_token=None,
                                    scope=None, mcp_server_name=None)
    body = json.loads(resp.body)
    assert body["token_type"] == "bearer"
    assert "access_token" in body


def test_try_decode_broker_code_returns_none_for_relay_code():
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import _try_decode_broker_code
    with patch(f"{D}.decrypt_value_helper", return_value=None):           # relay/upstream code can't decrypt
        assert _try_decode_broker_code("random-upstream-code") is None
    with patch(f"{D}.decrypt_value_helper", return_value=json.dumps({"foo": "bar"})):  # decrypts but not a broker code
        assert _try_decode_broker_code("something") is None


def test_prm_advertises_litellm_as_as_and_canonical_resource():
    """Regression-lock: the PRM must advertise LiteLLM itself as the authorization
    server (not the upstream IdP), and the resource must be LiteLLM's /mcp/{name} URL.
    A client discovering these metadata values will send its authorization request
    to LiteLLM, which will then broker to the upstream — the foundational invariant
    of the broker design."""
    from unittest.mock import MagicMock
    with (
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}.IPAddressUtils"),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mgr,
    ):
        mgr.get_mcp_server_by_name.return_value = _broker_server()
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import _build_oauth_protected_resource_response
        prm = _build_oauth_protected_resource_response(MagicMock(), "gitlab", True)
    assert prm["resource"] == "https://llm.example.com/mcp/gitlab"            # canonical RS URI
    assert prm["authorization_servers"] == ["https://llm.example.com/gitlab"]  # LiteLLM is the AS (not upstream)


@pytest.mark.asyncio
async def test_broker_token_mint_includes_refresh_token():
    server = _broker_server()
    verifier = "the-verifier"
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    data = {"broker_code": True, "principal": "mcp-oauth:abc", "server_id": "s1",
            "client_code_challenge": challenge, "client_redirect_uri": "https://claude/cb"}
    key = "test-master-key-0123456789abcdef0123456789abcdef"
    with (
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}._get_broker_master_key", return_value=key),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import broker_token_mint
        result = await broker_token_mint(request=MagicMock(), data=data, code_verifier=verifier)
    assert "refresh_token" in result
    claims = jwt.decode(result["refresh_token"], key, algorithms=["HS256"],
                        audience="https://llm.example.com/mcp/gitlab")
    assert claims["token_type"] == "mcp_broker_refresh"
    assert claims["user_id"] == "mcp-oauth:abc"
    assert claims["server_id"] == "s1"


def _mint_refresh(server_id="s1", resource="https://llm.example.com/mcp/gitlab",
                  principal="mcp-oauth:abc", key="test-master-key-0123456789abcdef0123456789abcdef"):
    from litellm.proxy._experimental.mcp_server.broker import mint_broker_refresh_token
    return mint_broker_refresh_token(principal=principal, server_id=server_id, resource=resource,
                                     master_key=key, ttl_seconds=3600)


@pytest.mark.asyncio
async def test_token_endpoint_broker_refresh_issues_new_pair():
    server = _broker_server()
    key = "test-master-key-0123456789abcdef0123456789abcdef"
    tok = _mint_refresh()
    with (
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}._get_broker_master_key", return_value=key),
        patch(f"{D}.get_prisma_client_or_throw", return_value=MagicMock()),
        patch("litellm.proxy._experimental.mcp_server.db.get_user_oauth_credential",
              new=AsyncMock(return_value={"access_token": "UPSTREAM"})),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import token_endpoint
        resp = await token_endpoint(request=MagicMock(), grant_type="refresh_token",
                                    code=None, redirect_uri=None, client_id="claude",
                                    client_secret=None, code_verifier=None, refresh_token=tok,
                                    scope=None, mcp_server_name=None)
    body = json.loads(resp.body)
    assert body["token_type"] == "bearer"
    assert "access_token" in body and "refresh_token" in body
    access = jwt.decode(body["access_token"], key, algorithms=["HS256"],
                        audience="https://llm.example.com/mcp/gitlab")
    assert access["token_type"] == "mcp_broker"
    refresh = jwt.decode(body["refresh_token"], key, algorithms=["HS256"],
                         audience="https://llm.example.com/mcp/gitlab")
    assert refresh["token_type"] == "mcp_broker_refresh"


@pytest.mark.asyncio
async def test_token_endpoint_broker_refresh_invalid_grant_when_cred_missing():
    server = _broker_server()
    key = "test-master-key-0123456789abcdef0123456789abcdef"
    tok = _mint_refresh()
    with (
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}._get_broker_master_key", return_value=key),
        patch(f"{D}.get_prisma_client_or_throw", return_value=MagicMock()),
        patch("litellm.proxy._experimental.mcp_server.db.get_user_oauth_credential",
              new=AsyncMock(return_value=None)),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import token_endpoint
        with pytest.raises(HTTPException) as ei:
            await token_endpoint(request=MagicMock(), grant_type="refresh_token",
                                 code=None, redirect_uri=None, client_id="claude",
                                 client_secret=None, code_verifier=None, refresh_token=tok,
                                 scope=None, mcp_server_name=None)
    assert ei.value.status_code == 400
    assert ei.value.detail == {"error": "invalid_grant"}


@pytest.mark.asyncio
async def test_token_endpoint_broker_refresh_invalid_grant_when_access_token_absent():
    server = _broker_server()
    key = "test-master-key-0123456789abcdef0123456789abcdef"
    tok = _mint_refresh()
    with (
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}._get_broker_master_key", return_value=key),
        patch(f"{D}.get_prisma_client_or_throw", return_value=MagicMock()),
        patch("litellm.proxy._experimental.mcp_server.db.get_user_oauth_credential",
              new=AsyncMock(return_value={"refresh_token": "only-refresh"})),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import token_endpoint
        with pytest.raises(HTTPException) as ei:
            await token_endpoint(request=MagicMock(), grant_type="refresh_token",
                                 code=None, redirect_uri=None, client_id="claude",
                                 client_secret=None, code_verifier=None, refresh_token=tok,
                                 scope=None, mcp_server_name=None)
    assert ei.value.status_code == 400
    assert ei.value.detail == {"error": "invalid_grant"}


@pytest.mark.asyncio
async def test_token_endpoint_access_token_as_refresh_falls_through_to_upstream():
    # A broker ACCESS token (token_type=mcp_broker) presented as refresh_token must NOT
    # renew via the broker path — it fails the type guard and falls through to the
    # upstream relay (exchange_token_with_server).
    server = _broker_server()
    key = "test-master-key-0123456789abcdef0123456789abcdef"
    from litellm.proxy._experimental.mcp_server.broker import mint_broker_token
    access = mint_broker_token(principal="mcp-oauth:abc", server_id="s1",
                               resource="https://llm.example.com/mcp/gitlab",
                               master_key=key, ttl_seconds=3600)
    relay = AsyncMock(return_value=JSONResponse({"relayed": True}))
    with (
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}._get_broker_master_key", return_value=key),
        patch(f"{D}.exchange_token_with_server", new=relay),
        patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager") as mgr,
        patch(f"{D}.IPAddressUtils"),
    ):
        mgr.get_mcp_server_by_name.return_value = server
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import token_endpoint
        await token_endpoint(request=MagicMock(), grant_type="refresh_token",
                             code=None, redirect_uri=None, client_id="claude",
                             client_secret=None, code_verifier=None, refresh_token=access,
                             scope=None, mcp_server_name=None)
    relay.assert_awaited_once()   # fell through to the upstream relay
