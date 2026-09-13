from __future__ import annotations

import networkx as nx


class ServiceDependencyGraph:
    """Deterministic graph of services, deployments, incidents, and dependencies."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        # Services
        self.graph.add_node(
            "payment-service",
            node_type="service",
        )
        self.graph.add_node(
            "fraud-service",
            node_type="service",
        )

        # Deployment
        self.graph.add_node(
            "DEP-001",
            node_type="deployment",
            version="2.4.1",
        )

        # Incident
        self.graph.add_node(
            "INC-001",
            node_type="incident",
        )

        # Dependency relationship
        self.graph.add_edge(
            "payment-service",
            "fraud-service",
            relationship="depends_on",
        )

        # Deployment relationship
        self.graph.add_edge(
            "DEP-001",
            "payment-service",
            relationship="deployed_to",
        )

        # Incident relationship
        self.graph.add_edge(
            "INC-001",
            "payment-service",
            relationship="affected",
        )

    def get_dependencies(self, service: str) -> list[str]:
        """Return direct service dependencies."""
        if service not in self.graph:
            return []

        return [
            node
            for node in self.graph.successors(service)
            if self.graph.nodes[node].get("node_type") == "service"
        ]

    def get_dependents(self, service: str) -> list[str]:
        """Return services that directly depend on the given service."""
        if service not in self.graph:
            return []

        return [
            node
            for node in self.graph.predecessors(service)
            if self.graph.nodes[node].get("node_type") == "service"
        ]

    def get_related_nodes(self, service: str) -> list[str]:
        """Return nodes directly connected to a service."""
        if service not in self.graph:
            return []

        return sorted(
            set(self.graph.predecessors(service))
            | set(self.graph.successors(service))
        )

    def dependency_chain(self, service: str) -> list[dict[str, str]]:
        """Return deterministic dependency relationships."""
        if service not in self.graph:
            return []

        relationships = []

        for dependency in self.get_dependencies(service):
            relationships.append(
                {
                    "source": service,
                    "relationship": "depends_on",
                    "target": dependency,
                }
            )

        return relationships

    def explain_service(self, service: str) -> dict:
        """Return a compact deterministic explanation of service relationships."""
        return {
            "service": service,
            "dependencies": self.get_dependencies(service),
            "dependents": self.get_dependents(service),
            "related_nodes": self.get_related_nodes(service),
            "dependency_chain": self.dependency_chain(service),
        }


def build_service_graph() -> ServiceDependencyGraph:
    """Create the NexusOps service dependency graph."""
    return ServiceDependencyGraph()


if __name__ == "__main__":
    graph = build_service_graph()

    print("PAYMENT SERVICE GRAPH")
    print("=" * 60)

    print(graph.explain_service("payment-service"))