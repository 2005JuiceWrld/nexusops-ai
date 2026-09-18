from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, Callable

from src.agent.tools import (
    calculate_metrics,
    get_production_policy,
    get_recent_deployments,
    get_service_status,
    restart_service,
    retrieve_runbook,
    search_incidents,
)


class MCPTool:
    """Model Context Protocol Tool definition."""

    def __init__(
        self,
        name: str,
        group: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        self.name = name
        self.group = group
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class MCPServer:
    """
    Model Context Protocol (MCP) tool server for NexusOps AI.

    Wraps existing operational tools into logical MCP tool groups:
    - Incident
    - Deployment
    - Service
    - Runbook
    - Metrics
    - Approval
    """

    def __init__(self) -> None:
        self.tools: dict[str, MCPTool] = {}
        self.approval_store: dict[str, dict[str, Any]] = {}
        self._register_default_tools()

    def register_tool(
        self,
        name: str,
        group: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        tool = MCPTool(
            name=name,
            group=group,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        self.tools[name] = tool

    def _register_default_tools(self) -> None:
        # Group: Incident
        self.register_tool(
            name="incident.search",
            group="Incident",
            description="Find operational incidents by service name or list all.",
            input_schema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Optional service name filter",
                    }
                },
            },
            handler=self._handle_incident_search,
        )

        # Group: Deployment
        self.register_tool(
            name="deployment.recent",
            group="Deployment",
            description="Return recent deployment history for services.",
            input_schema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Optional service name filter",
                    }
                },
            },
            handler=self._handle_deployment_recent,
        )

        # Group: Service
        self.register_tool(
            name="service.status",
            group="Service",
            description="Return current service operational status and metrics.",
            input_schema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Target service name",
                    }
                },
                "required": ["service"],
            },
            handler=self._handle_service_status,
        )

        self.register_tool(
            name="service.restart",
            group="Service",
            description="Simulate restarting a service (requires human approval).",
            input_schema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Target service name",
                    },
                    "approved": {
                        "type": "boolean",
                        "description": "Human approval status flag",
                        "default": False,
                    },
                    "approval_id": {
                        "type": "string",
                        "description": "Optional approval request ID",
                    },
                },
                "required": ["service"],
            },
            handler=self._handle_service_restart,
        )

        # Group: Runbook
        self.register_tool(
            name="runbook.retrieve",
            group="Runbook",
            description="Retrieve operational runbooks matching a keyword or topic.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or keyword for runbook search",
                    }
                },
                "required": ["topic"],
            },
            handler=self._handle_runbook_retrieve,
        )

        # Group: Metrics
        self.register_tool(
            name="metrics.calculate",
            group="Metrics",
            description="Calculate operational failure rates and transactional metrics.",
            input_schema={
                "type": "object",
                "properties": {
                    "failures": {
                        "type": "integer",
                        "description": "Number of recorded failures",
                    },
                    "total_transactions": {
                        "type": "integer",
                        "description": "Total count of processed transactions",
                    },
                },
                "required": ["failures", "total_transactions"],
            },
            handler=self._handle_metrics_calculate,
        )

        # Group: Approval
        self.register_tool(
            name="approval.request",
            group="Approval",
            description="Request human approval for high-risk production actions.",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "High-risk action to be executed",
                    },
                    "service": {
                        "type": "string",
                        "description": "Target service name",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for requesting production change",
                    },
                    "requested_by": {
                        "type": "string",
                        "description": "Identifier of requester",
                        "default": "agent",
                    },
                },
                "required": ["action", "service"],
            },
            handler=self._handle_approval_request,
        )

        self.register_tool(
            name="approval.check",
            group="Approval",
            description="Check approval request status and view production rollback policies.",
            input_schema={
                "type": "object",
                "properties": {
                    "approval_id": {
                        "type": "string",
                        "description": "Specific approval request ID to check",
                    },
                    "service": {
                        "type": "string",
                        "description": "Service name filter for pending requests",
                    },
                },
            },
            handler=self._handle_approval_check,
        )

    # Handlers calling existing tools (zero duplicated business logic)
    def _handle_incident_search(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        service = kwargs.get("service")
        res = search_incidents(service=service)
        res["mcp_tool"] = "incident.search"
        return res

    def _handle_deployment_recent(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        service = kwargs.get("service")
        res = get_recent_deployments(service=service)
        res["mcp_tool"] = "deployment.recent"
        return res

    def _handle_service_status(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        service = kwargs.get("service", "")
        res = get_service_status(service=service)
        res["mcp_tool"] = "service.status"
        return res

    def _handle_service_restart(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        service = kwargs.get("service", "")
        approved = kwargs.get("approved", False)
        approval_id = kwargs.get("approval_id")

        if approval_id and approval_id in self.approval_store:
            req = self.approval_store[approval_id]
            if req.get("status") == "approved":
                approved = True

        res = restart_service(service=service, approved=approved)
        res["mcp_tool"] = "service.restart"
        if approval_id:
            res["approval_id"] = approval_id
        return res

    def _handle_runbook_retrieve(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        topic = kwargs.get("topic", "")
        res = retrieve_runbook(topic=topic)
        res["mcp_tool"] = "runbook.retrieve"
        return res

    def _handle_metrics_calculate(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        failures = kwargs.get("failures", 0)
        total_transactions = kwargs.get("total_transactions", 0)
        res = calculate_metrics(
            failures=failures, total_transactions=total_transactions
        )
        res["mcp_tool"] = "metrics.calculate"
        return res

    def _handle_approval_request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        action = kwargs.get("action", "unknown")
        service = kwargs.get("service", "unknown")
        reason = kwargs.get("reason", "Automated remediation request")
        requested_by = kwargs.get("requested_by", "agent")

        approval_id = f"APP-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        req_data = {
            "approval_id": approval_id,
            "status": "pending",
            "action": action,
            "service": service,
            "reason": reason,
            "requested_by": requested_by,
            "created_at": now,
        }
        self.approval_store[approval_id] = req_data

        return {
            "tool": "approval.request",
            "mcp_tool": "approval.request",
            "success": True,
            "approval_id": approval_id,
            "request": req_data,
        }

    def _handle_approval_check(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        approval_id = kwargs.get("approval_id")
        service = kwargs.get("service")

        policy_info = get_production_policy()

        requests: list[dict[str, Any]] = []
        if approval_id:
            if approval_id in self.approval_store:
                requests.append(self.approval_store[approval_id])
        else:
            for req in self.approval_store.values():
                if service is None or req.get("service") == service:
                    requests.append(req)

        return {
            "tool": "approval.check",
            "mcp_tool": "approval.check",
            "success": True,
            "production_policy": policy_info.get("results", []),
            "approval_requests": requests,
        }

    def approve_request(
        self, approval_id: str, approved_by: str = "human_operator"
    ) -> dict[str, Any]:
        """Utility method to update an approval status."""
        if approval_id not in self.approval_store:
            return {
                "success": False,
                "error": f"Approval request {approval_id} not found.",
            }
        req = self.approval_store[approval_id]
        req["status"] = "approved"
        req["approved_by"] = approved_by
        req["approved_at"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        return {"success": True, "request": req}

    def list_tools(self, group: str | None = None) -> list[dict[str, Any]]:
        """List registered tools optionally filtered by logical group."""
        result = []
        for tool in self.tools.values():
            if group is None or tool.group.lower() == group.lower():
                result.append(tool.to_dict())
        return result

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a tool by MCP tool name."""
        if name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown MCP tool: {name}",
                "available_tools": list(self.tools.keys()),
            }
        tool = self.tools[name]
        args = arguments or {}
        try:
            return tool.handler(args)
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error in {name}: {str(e)}",
            }

    def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle MCP standard JSON-RPC 2.0 requests.
        Supports: initialize, ping, tools/list, tools/call
        """
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "nexusops-mcp",
                        "version": "1.0.0",
                    },
                },
            }
        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        elif method == "tools/list":
            group = params.get("group")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.list_tools(group=group)},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            res = self.call_tool(tool_name, tool_args)
            is_error = not res.get("success", True)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(res, indent=2),
                        }
                    ],
                    "isError": is_error,
                    "structured_result": res,
                },
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }


# Global singleton MCP server instance
mcp_server = MCPServer()
