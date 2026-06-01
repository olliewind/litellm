"""OAuth broker helpers for MCP servers: mint the client-facing token, establish the
per-grant principal, and validate token audience. Pure functions — no I/O — so they
are unit-testable in isolation. See spec 2026-06-01-litellm-mcp-oauth-broker-design."""
import hashlib
import time

import jwt


def mint_broker_token(*, principal: str, server_id: str, resource: str,
                      master_key: str, ttl_seconds: int = 3600) -> str:
    """Client-facing access token: master-key-signed JWT (same primitive SSO/BYOK use,
    so user_api_key_auth accepts it and sets user_api_key_auth.user_id = principal),
    audience-bound to the MCP server's canonical resource URI (RFC 8707 §2). It is a
    reference to the upstream credential stored server-side (RFC 6749 §1.4)."""
    now = int(time.time())
    payload = {
        "user_id": principal,
        "server_id": server_id,
        "aud": resource,
        "iat": now,
        "exp": now + ttl_seconds,
        "token_type": "mcp_broker",
    }
    return jwt.encode(payload, master_key, algorithm="HS256")


def establish_principal(*, server_id: str, state: str) -> str:
    """Opaque, stable per-grant principal id. Deterministic from (server_id, state) so
    the callback (store) and token endpoint (mint) agree without extra storage.
    Provider-agnostic; named-identity (OIDC sub) derivation is a deferred enhancement."""
    digest = hashlib.sha256(f"{len(server_id)}:{server_id}:{state}".encode()).hexdigest()[:32]
    return f"mcp-oauth:{digest}"


def token_audience_ok(token: str, *, expected_resource: str, master_key: str) -> bool:
    """RFC 8707 §2 / OAuth 2.1 §5.2: a broker token is valid only for the MCP server in
    its `aud`. Rejects foreign-audience tokens and any raw upstream token."""
    try:
        jwt.decode(token, master_key, algorithms=["HS256"], audience=expected_resource)
        return True
    except Exception:
        return False
