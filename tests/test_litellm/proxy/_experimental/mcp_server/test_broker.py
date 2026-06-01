from litellm.types.mcp_server.mcp_server_manager import MCPServer


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
