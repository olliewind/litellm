"""End-to-end broker flow test: authorize → callback → token_endpoint.

Uses reversible crypto mocks (ENC:/strip prefix) so encrypted state and code
round-trip correctly through the real production logic.  No production code
is modified; this exercises Tasks 1-6 behaviour with injected boundaries.

Three headline properties verified:
1. The client NEVER receives the upstream access token.
2. The minted token is audience-bound to LiteLLM's /mcp/{name} resource.
3. The principal in the minted JWT is exactly the key under which the upstream
   token was stored server-side (so the inject path resolves correctly).
"""
import base64
import hashlib
import json

import jwt
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse, parse_qs

from litellm.types.mcp_server.mcp_server_manager import MCPServer

D = "litellm.proxy._experimental.mcp_server.discoverable_endpoints"
MASTER = "test-master-key-0123456789abcdef0123456789abcdef"


def _server():
    return MCPServer(
        server_id="s1",
        name="gitlab",
        server_name="gitlab",
        transport="http",
        auth_type="oauth2",
        broker=True,
        url="http://backend/mcp",
        authorization_url="https://up/authorize",
        token_url="https://up/token",
        client_id="cid",
        client_secret="csec",
    )


@pytest.mark.asyncio
async def test_broker_e2e_no_passthrough_then_inject_key_matches():
    """Full three-leg chain with reversible crypto mocks.

    authorize  → 302 to upstream (leg 1)
    callback   → back-channel exchange, store upstream token, issue LiteLLM code (leg 2)
    token      → PKCE-verify, mint audience-bound JWT (leg 3)
    """
    server = _server()
    cred_store = {}

    # PKCE: client generates verifier + challenge
    verifier = "client-verifier"
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )

    # Reversible crypto mocks: "ENC:" prefix is the ciphertext; strip to decrypt.
    def _enc(s):
        return "ENC:" + s

    def _dec(s, key="oauth_state"):
        if isinstance(s, str) and s.startswith("ENC:"):
            return s[4:]
        return None

    async def _store(server, user_id, token_response, raise_on_error=False):
        cred_store[(user_id, server.server_id)] = token_response["access_token"]

    with (
        patch(f"{D}.validate_trusted_redirect_uri"),
        patch(f"{D}._get_validated_client_redirect_uri", return_value="https://claude/cb"),
        patch(f"{D}.get_request_base_url", return_value="https://llm.example.com"),
        patch(f"{D}.encrypt_value_helper", side_effect=_enc),
        patch(f"{D}.decrypt_value_helper", side_effect=_dec),
        patch(f"{D}.get_mcp_server_by_id", return_value=server),
        patch(
            f"{D}._exchange_code_for_token_dict",
            new=AsyncMock(
                return_value={"access_token": "UPSTREAM-RAW", "refresh_token": "R"}
            ),
        ),
        patch(f"{D}._store_per_user_token_server_side", new=_store),
        patch(f"{D}._get_broker_master_key", return_value=MASTER),
    ):
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            broker_authorize,
            callback,
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.broker import token_audience_ok

        # --- Leg 1: client → LiteLLM authorize → 302 to upstream ---
        # broker_authorize generates its own PKCE pair; the client's challenge
        # must be stored in state but NOT forwarded upstream.
        a = await broker_authorize(
            request=MagicMock(),
            mcp_server=server,
            client_id="claude",
            redirect_uri="https://claude/cb",
            state="client-state",
            code_challenge=challenge,
            code_challenge_method="S256",
            scope="api",
            consented=True,
        )
        assert a.status_code in (302, 307)
        loc = a.headers["location"]
        assert loc.startswith("https://up/authorize")
        assert "UPSTREAM-RAW" not in loc  # no token at this stage
        enc_state = parse_qs(urlparse(loc).query)["state"][0]

        # --- Leg 2: upstream → LiteLLM /callback → store + issue LiteLLM code ---
        # The upstream token must NEVER appear in the redirect to the client.
        c = await callback(request=MagicMock(), code="UPSTREAM-CODE", state=enc_state)
        assert c.status_code in (302, 307)
        cb_loc = c.headers["location"]
        assert "UPSTREAM-RAW" not in cb_loc   # property 1: no passthrough
        assert "UPSTREAM-CODE" not in cb_loc
        assert "error" not in parse_qs(urlparse(cb_loc).query)
        litellm_code = parse_qs(urlparse(cb_loc).query)["code"][0]

        # --- Leg 3: client redeems LiteLLM code at token endpoint ---
        resp = await token_endpoint(
            request=MagicMock(),
            grant_type="authorization_code",
            code=litellm_code,
            redirect_uri=None,
            client_id="claude",
            client_secret=None,
            code_verifier=verifier,
            refresh_token=None,
            scope=None,
            mcp_server_name=None,
        )
        tok_body = json.loads(resp.body)
        assert tok_body["token_type"] == "bearer"
        tok = tok_body["access_token"]

        # Property 1: client never gets the upstream token
        assert tok != "UPSTREAM-RAW"

        # Property 2: minted token is audience-bound to LiteLLM's resource
        assert token_audience_ok(
            tok,
            expected_resource="https://llm.example.com/mcp/gitlab",
            master_key=MASTER,
        )

        # Property 3: inject key == store key
        # The principal embedded in the JWT must match the key the upstream
        # token was stored under, so the existing inject lookup resolves.
        claims = jwt.decode(
            tok, MASTER, algorithms=["HS256"], options={"verify_aud": False}
        )
        principal = claims["user_id"]
        assert (principal, "s1") in cred_store, (
            f"upstream token not stored under ({principal!r}, 's1'); "
            f"stored keys: {list(cred_store)}"
        )
        assert cred_store[(principal, "s1")] == "UPSTREAM-RAW"
