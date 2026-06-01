import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from litellm.proxy._experimental.mcp_server.server import handle_streamable_http_mcp
from litellm.proxy._types import ProxyException
from starlette.requests import Request

@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_proxy_exception():
    """Test that handle_streamable_http_mcp re-raises ProxyException instead of converting to 500."""
    # Create mock scope, receive, send
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/test-server",
        "headers": [(b"host", b"localhost"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
    }
    
    async def mock_receive():
        return {"type": "http.request", "body": b""}
        
    mock_send = AsyncMock()

    # Mock extract_mcp_auth_context to raise a ProxyException
    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        side_effect=ProxyException(message="Custom Error", type="auth_error", param=None, code=403)
    ):
        # Should raise the ProxyException, not convert to 500
        with pytest.raises(ProxyException) as excinfo:
            await handle_streamable_http_mcp(scope, mock_receive, mock_send)
            
        assert excinfo.value.code == "403"
        assert excinfo.value.message == "Custom Error"
        
        # Verify it didn't try to send a 500 response
        mock_send.assert_not_called()

from litellm.proxy.proxy_server import toolset_mcp_route, dynamic_mcp_route

@pytest.mark.asyncio
async def test_toolset_mcp_route_proxy_exception():
    request = AsyncMock(spec=Request)
    request.scope = {"type": "http"}
    
    with patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager.get_toolset_by_name_cached",
        side_effect=ProxyException(message="Toolset Error", type="auth_error", param=None, code=401)
    ), patch(
        "litellm.proxy.proxy_server.prisma_client", new=AsyncMock()
    ):
        with pytest.raises(ProxyException) as excinfo:
            await toolset_mcp_route("test_toolset", request)
            
        assert excinfo.value.code == "401"
