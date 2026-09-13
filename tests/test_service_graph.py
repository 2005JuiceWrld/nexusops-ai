from src.graph.service_graph import ServiceDependencyGraph


def test_payment_service_dependency():
    graph = ServiceDependencyGraph()

    dependencies = graph.get_dependencies("payment-service")

    assert "fraud-service" in dependencies


def test_payment_service_related_nodes():
    graph = ServiceDependencyGraph()

    related = graph.get_related_nodes("payment-service")

    assert "fraud-service" in related
    assert "DEP-001" in related
    assert "INC-001" in related


def test_dependency_chain():
    graph = ServiceDependencyGraph()

    chain = graph.dependency_chain("payment-service")

    assert {
        "source": "payment-service",
        "relationship": "depends_on",
        "target": "fraud-service",
    } in chain


def test_unknown_service():
    graph = ServiceDependencyGraph()

    assert graph.get_dependencies("unknown-service") == []
    assert graph.get_dependents("unknown-service") == []
    assert graph.get_related_nodes("unknown-service") == []
    assert graph.dependency_chain("unknown-service") == []