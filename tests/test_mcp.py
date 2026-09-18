import json
from src.mcp.server import MCPServer


def test_mcp_tool_groups():
    server = MCPServer()
    tools = server.list_tools()
    
    groups = {t["group"] for t in tools}
    expected_groups = {"Incident", "Deployment", "Service", "Runbook", "Metrics", "Approval"}
    
    assert expected_groups.issubset(groups)
    assert len(tools) >= 7


def test_mcp_list_tools_by_group():
    server = MCPServer()
    incident_tools = server.list_tools(group="Incident")
    assert len(incident_tools) == 1
    assert incident_tools[0]["name"] == "incident.search"

    approval_tools = server.list_tools(group="Approval")
    assert len(approval_tools) == 2
    names = {t["name"] for t in approval_tools}
    assert names == {"approval.request", "approval.check"}


def test_mcp_incident_search():
    server = MCPServer()
    result = server.call_tool("incident.search", {"service": "payment-service"})
    
    assert result["success"] is True
    assert result["mcp_tool"] == "incident.search"
    assert "results" in result


def test_mcp_deployment_recent():
    server = MCPServer()
    result = server.call_tool("deployment.recent", {"service": "payment-service"})
    
    assert result["success"] is True
    assert result["mcp_tool"] == "deployment.recent"
    assert "results" in result


def test_mcp_service_status():
    server = MCPServer()
    result = server.call_tool("service.status", {"service": "fraud-service"})
    
    assert result["success"] is True
    assert result["mcp_tool"] == "service.status"
    assert result["status"]["status"] == "degraded"


def test_mcp_runbook_retrieve():
    server = MCPServer()
    result = server.call_tool("runbook.retrieve", {"topic": "payment"})
    
    assert result["success"] is True
    assert result["mcp_tool"] == "runbook.retrieve"
    assert result["count"] > 0


def test_mcp_metrics_calculate():
    server = MCPServer()
    result = server.call_tool("metrics.calculate", {"failures": 10, "total_transactions": 1000})
    
    assert result["success"] is True
    assert result["mcp_tool"] == "metrics.calculate"
    assert result["failure_rate_percent"] == 1.0


def test_mcp_approval_workflow():
    server = MCPServer()
    
    # 1. Request approval for service restart
    req_res = server.call_tool(
        "approval.request",
        {"action": "restart_service", "service": "payment-service", "reason": "High error rate"}
    )
    assert req_res["success"] is True
    approval_id = req_res["approval_id"]
    assert approval_id.startswith("APP-")
    
    # 2. Check approval status before human approval
    check_res = server.call_tool("approval.check", {"approval_id": approval_id})
    assert check_res["success"] is True
    assert check_res["approval_requests"][0]["status"] == "pending"
    
    # 3. Attempt restart without approval flag or approved request
    restart_res = server.call_tool("service.restart", {"service": "payment-service", "approval_id": approval_id})
    assert restart_res["success"] is False
    assert restart_res["requires_approval"] is True
    
    # 4. Approve request externally
    approve_res = server.approve_request(approval_id, approved_by="admin@nexusops.ai")
    assert approve_res["success"] is True
    
    # 5. Retry restart with approval_id now approved
    restart_approved = server.call_tool("service.restart", {"service": "payment-service", "approval_id": approval_id})
    assert restart_approved["success"] is True
    assert restart_approved["simulated"] is True


def test_mcp_jsonrpc_protocol():
    server = MCPServer()
    
    # Initialize
    init_res = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_res["result"]["serverInfo"]["name"] == "nexusops-mcp"
    
    # Tools list
    list_res = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"group": "Metrics"}})
    tools = list_res["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "metrics.calculate"
    
    # Tools call
    call_res = server.handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "metrics.calculate",
            "arguments": {"failures": 5, "total_transactions": 50}
        }
    })
    assert call_res["result"]["isError"] is False
    struct_res = call_res["result"]["structured_result"]
    assert struct_res["failure_rate_percent"] == 10.0
