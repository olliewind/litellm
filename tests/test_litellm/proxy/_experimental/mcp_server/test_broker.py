import jwt as _jwt
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.proxy._experimental.mcp_server.broker import (
    mint_broker_token, establish_principal, token_audience_ok,
)

_MASTER_KEY = "test-master-key-0123456789abcdef0123456789abcdef"  # ≥32 bytes; HS256 RFC 7518 §3.2


def _server(**kw):
    base = dict(server_id="s1", name="gitlab", transport="http", auth_type="oauth2")
    base.update(kw)
    return MCPServer(**base)


def test_broker_defaults_false():
    assert _server().broker is False


def test_is_oauth_broker_true_only_for_oauth2_plus_broker():
    assert _server(broker=True).is_oauth_broker is True
    assert _server(broker=True, auth_type="bearer_token").is_oauth_broker is False
    assert _server(broker=False).is_oauth_broker is False


def test_mint_is_aud_bound_and_carries_principal():
    tok = mint_broker_token(principal="mcp-oauth:abc", server_id="s1",
                            resource="https://llm.example.com/mcp/gitlab",
                            master_key=_MASTER_KEY, ttl_seconds=3600)
    claims = _jwt.decode(tok, _MASTER_KEY, algorithms=["HS256"],
                         audience="https://llm.example.com/mcp/gitlab")
    assert claims["user_id"] == "mcp-oauth:abc"   # becomes user_api_key_auth.user_id → inject key
    assert claims["server_id"] == "s1"
    assert "exp" in claims


def test_principal_stable_per_grant_and_opaque():
    a = establish_principal(server_id="s1", state="xyz")
    assert a == establish_principal(server_id="s1", state="xyz")  # callback↔token agree
    assert a != establish_principal(server_id="s1", state="other")
    assert a.startswith("mcp-oauth:")


def test_audience_check():
    good = mint_broker_token(principal="p", server_id="s1",
                             resource="https://llm.example.com/mcp/gitlab",
                             master_key=_MASTER_KEY, ttl_seconds=60)
    bad = mint_broker_token(principal="p", server_id="s1",
                            resource="https://llm.example.com/mcp/other",
                            master_key=_MASTER_KEY, ttl_seconds=60)
    assert token_audience_ok(good, expected_resource="https://llm.example.com/mcp/gitlab", master_key=_MASTER_KEY) is True
    assert token_audience_ok(bad, expected_resource="https://llm.example.com/mcp/gitlab", master_key=_MASTER_KEY) is False
    assert token_audience_ok("not-a-jwt", expected_resource="https://llm.example.com/mcp/gitlab", master_key=_MASTER_KEY) is False
