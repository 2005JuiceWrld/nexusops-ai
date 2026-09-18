# NexusOps AI — Architecture

## 1. System Objective

NexusOps AI investigates operational incidents using evidence retrieval, service dependency reasoning, LLM analysis, tool execution, verification, and safety controls.

The architecture deliberately separates:

```text
AI reasoning
     ↓
evidence
     ↓
verification
     ↓
policy
     ↓
approval
     ↓
action
2. Request Lifecycle
User Query
    ↓
Prompt Injection Check
    ↓
Plan
    ↓
Retrieve Evidence
    ↓
Execute Required Tools
    ↓
Analyze
    ↓
Evidence Verification
    ↓
Safety / Policy Check
    ↓
Human Approval if Required
    ↓
Final Answer
3. Retrieval Layer

The retrieval layer combines multiple strategies.

BM25

Provides lexical matching for operational terms such as:

payment-service
INC-001
DEP-001
rollback
latency
fraud-service
Dense Retrieval

Provides semantic matching when the wording differs from the indexed evidence.

Reciprocal Rank Fusion

Combines lexical and semantic rankings.

Reranking

A reranking stage improves the ordering of candidate evidence before it reaches the reasoning layer.

4. Dependency Graph

NetworkX represents relationships between services.

Example:

payment-service
       │
       │ depends_on
       ▼
fraud-service

The graph provides structural evidence that cannot reliably be inferred from free-text retrieval alone.

5. Agent Workflow

The workflow is explicitly represented as:

PLAN
RETRIEVE
TOOL_CALL
ANALYZE
VERIFY
ANSWER

Each stage has a specific responsibility.

PLAN

Determines which information is required.

RETRIEVE

Obtains relevant operational evidence.

TOOL_CALL

Calls bounded tools rather than allowing arbitrary infrastructure interaction.

ANALYZE

Combines retrieved evidence, tool output, graph relationships, and model reasoning.

VERIFY

Checks whether the generated conclusion is grounded and policy-consistent.

ANSWER

Returns the verified result to the user.

6. Tool Boundary

Available capabilities include:

search_incidents()
get_recent_deployments()
get_service_status()
retrieve_runbook()
get_production_policy()
calculate_metrics()
restart_service()

The tools provide a controlled interface between the agent and operational data/actions.

restart_service() is simulated and cannot directly restart a production service.

7. Verification Layer

EvidenceVerifier checks:

Cause grounding
Supporting evidence
Recommendation
Confidence
Policy consistency
Grounded claims

This creates a deterministic control layer around probabilistic model output.

8. Safety Layer

SafetyPolicy determines whether a proposed action is permitted.

High-risk actions require approval.

Example:

Agent proposes rollback
        ↓
Policy check
        ↓
High-risk action
        ↓
Human approval required
9. Prompt Injection Defense

Input is classified before normal workflow execution.

Categories include:

direct_injection
authority_spoofing
action_injection
indirect_injection

Detected malicious instructions are blocked.

10. Evaluation Architecture

The evaluation framework measures five dimensions:

Retrieval
Reasoning
Agent
Safety
System

Agent trajectories are separately checked for:

WrongTool
WrongArgs
WrongOrder
MissingStep
UnexpectedStep

This evaluates not only the final answer but also the behavior of the agent.

11. Design Principles
Evidence over confidence

A confident LLM response is not automatically considered correct.

Least privilege

The agent receives bounded capabilities rather than unrestricted infrastructure access.

Human control

High-risk operations require explicit approval.

Reproducibility

Deterministic fallbacks allow evaluation without depending entirely on external model availability.

Explicit workflow

Agent behavior follows a controlled state sequence rather than unrestricted autonomous execution.

12. Operational Scenario

Payment failures are investigated using:

INC-001
DEP-001
payment-service 2.4.1
fraud-service p95 = 620 ms
fraud-service errors = 8.4%
payment-service → fraud-service

The reasoning chain associates increased fraud-service latency and the recent payment-service deployment with the incident.

Rollback to payment-service 2.4.0 is proposed as remediation.

Because rollback is operationally sensitive, approval is required.

13. Evaluation Result

Current benchmark:

119 / 125
95.2%

The system achieves perfect scores in:

Retrieval
Agent
System
Trajectory
Injection resistance

Remaining failures are documented evaluation edge cases rather than additional architectural requirements.
