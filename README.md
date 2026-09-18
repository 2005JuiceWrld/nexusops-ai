# NexusOps AI

## Enterprise AI Reasoning & Autonomous Operations Platform

NexusOps AI is an enterprise-oriented AI operations reasoning platform that investigates production incidents using retrieval, service dependency reasoning, tool execution, evidence verification, safety policies, and human approval controls.

The system is designed around one principle:

> AI should reason from evidence and operate within explicit safety boundaries.

It does not perform real production infrastructure changes. High-risk actions such as service restart and rollback are simulated and require approval.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │      FastAPI API     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Agent Workflow    │
                         │ PLAN → RETRIEVE     │
                         │ → TOOL → ANALYZE    │
                         │ → VERIFY → ANSWER   │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
      │ Hybrid        │     │ Service       │     │ LLM Reasoner  │
      │ Retrieval     │     │ Dependency    │     │               │
      │               │     │ Graph         │     │ Ollama /      │
      │ BM25          │     │               │     │ deterministic │
      │ Dense         │     │ NetworkX      │     │ fallback      │
      │ RRF           │     │               │     │               │
      │ Reranking     │     │               │     │               │
      └───────┬───────┘     └───────────────┘     └───────────────┘
              │
              ▼
      ┌──────────────────────────────────────────┐
      │              Tool Boundary               │
      │ incidents / deployments / status /       │
      │ runbooks / metrics / approval / restart  │
      └────────────────────┬─────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ EvidenceVerifier        │
              │ groundedness            │
              │ evidence validity       │
              │ recommendation          │
              │ policy consistency       │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ SafetyPolicy            │
              │ + Prompt Injection      │
              │ Detection               │
              │ + Human Approval        │
              └────────────┬────────────┘
                           │
                           ▼
                    Final Answer / Action
Core Capabilities
1. Incident Investigation

The system can investigate operational incidents by combining:

incident records
recent deployments
service health information
runbooks
production policies
dependency relationships
calculated operational metrics
2. Hybrid Retrieval

Retrieval combines:

BM25 lexical search
dense retrieval
Reciprocal Rank Fusion (RRF)
reranking

This allows both exact operational terminology and semantic matching.

3. Service Dependency Reasoning

NexusOps maintains service relationships using a dependency graph.

Example:

payment-service
       │
       ▼
fraud-service

The graph can be queried during incident investigation to reason about downstream dependencies.

4. Tool-Using Agent

The agent operates through explicit tool boundaries:

incident.search
deployment.recent
service.status
runbook.retrieve
metrics.calculate
approval.check
service.restart

Tool selection and arguments are evaluated independently.

5. Evidence Verification

Before accepting an AI-generated operational conclusion, the verifier checks:

evidence grounding
supporting evidence
confidence validity
recommendation presence
policy consistency
grounded claims

Unsupported conclusions can therefore be rejected rather than blindly returned.

6. Safety Controls

High-risk operational actions require authorization.

For example:

restart_service

does not actually restart infrastructure.

Without approval:

requires_approval

is returned.

7. Prompt-Injection Defense

The system detects several classes of malicious instructions:

direct_injection
authority_spoofing
action_injection
indirect_injection

Detected injections are blocked before normal workflow execution.

8. Deterministic Evaluation

The platform includes a deterministic evaluation suite covering:

Retrieval
Reasoning
Agent
Safety
System

This provides repeatable measurements rather than relying only on subjective demonstrations.

Operational Scenario

The primary investigation scenario involves payment failures following a deployment.

Incident:
INC-001

Deployment:
DEP-001

payment-service:
2.4.1

fraud-service:
p95 latency = 620 ms
error rate   = 8.4%

Dependency:
payment-service → fraud-service

The evidence indicates that increased fraud-service latency is associated with the payment failures and that payment-service 2.4.1 changed fraud-verification request handling.

The proposed remediation is rollback to:

payment-service 2.4.0

Because rollback is a high-risk production action, the system requires human approval.

Evaluation Results

Latest deterministic evaluation:

CategoryResult
Retrieval35/35
Reasoning24/25
Agent25/25
Safety20/25
System15/15
Overall119/125 (95.2%)
Retrieval
Recall@5       0.943
Recall@10      1.000
MRR            0.911
NDCG@10        0.923
Reasoning
Groundedness            0.960
Citation correctness    0.633
Cause accuracy          0.820
Agent
Task success     1.000
Tool selection   1.000
Tool arguments   1.000
Trajectory
Success          1.000
Plan             1.000
Retrieval        1.000
Tool             1.000
Args             1.000
Verify           1.000
Answer           1.000

Trajectory anomalies:

WrongTool        0
WrongArgs        0
WrongOrder       0
MissingStep      0
UnexpectedStep   0
Safety
Unsafe refusal       1.000
Approval accuracy    0.800
Injection resistance 1.000
System
Mean latency    6.88 ms
P95 latency    11.07 ms

Token usage is not reported because it is unavailable in the deterministic evaluation environment.

Evaluation Design

The suite contains:

125 deterministic cases

Retrieval    35
Reasoning    25
Agent        25
Safety       25
System       15

Evaluation artifacts:

evaluation/results.json
evaluation/summary.json
Safety Philosophy

NexusOps intentionally separates:

Reasoning
    ↓
Evidence verification
    ↓
Safety policy
    ↓
Approval
    ↓
Action

This prevents an LLM from directly controlling sensitive operational actions.

The platform demonstrates a safer agent architecture where the model proposes reasoning and actions, while deterministic controls enforce operational boundaries.

Technology Stack
Python 3.12
FastAPI
NetworkX
BM25
Dense Retrieval
RRF
Cross-Encoder Reranking
Ollama
Llama 3.2 3B
Pytest
Git / GitHub
Project Structure
nexusops-ai/
│
├── src/
│   ├── agent/
│   │   ├── workflow.py
│   │   ├── tools.py
│   │   ├── llm.py
│   │   ├── service_graph.py
│   │   ├── injection.py
│   │   └── ...
│   │
│   ├── evaluation/
│   │   ├── cases.py
│   │   ├── metrics.py
│   │   ├── runner.py
│   │   ├── trajectory.py
│   │   └── ...
│   │
│   └── api/
│
├── tests/
│   ├── test_evaluation.py
│   └── test_injection.py
│
├── evaluation/
│   ├── results.json
│   └── summary.json
│
└── README.md
Running the Project

Activate the environment:

.\.venv\Scripts\Activate.ps1

Run the API:

python -m uvicorn src.api.app:app --reload

Run tests:

python -m pytest -q

Run the deterministic evaluation:

python -m src.evaluation
Engineering Decisions
Deterministic fallback

The project includes deterministic reasoning fallbacks so evaluation remains reproducible when a local LLM is unavailable.

Simulated operational actions

Production actions are simulated. This makes it possible to demonstrate safety and approval workflows without creating real infrastructure risk.

Evidence-first reasoning

The system does not treat an LLM response as authoritative. Operational conclusions must pass evidence and policy checks.

Explicit agent trajectory

The workflow has a defined execution sequence:

PLAN
→ RETRIEVE
→ TOOL_CALL
→ ANALYZE
→ VERIFY
→ ANSWER

The trajectory itself is evaluated.

Current Limitations

The latest evaluation contains 6 failed cases.

These are concentrated in:

one dependency-query reasoning edge case
five safety/injection taxonomy cases

The failures do not indicate a missing core architecture component. They represent remaining evaluation edge cases in the current deterministic benchmark.

The project is intentionally frozen at:

119/125
95.2%

rather than introducing unnecessary complexity solely to achieve a perfect benchmark score.

What This Project Demonstrates

NexusOps demonstrates practical understanding of:

AI agents
LLM reasoning
RAG
hybrid retrieval
reranking
knowledge/dependency graphs
tool calling
MCP-style tool boundaries
evidence grounding
LLM evaluation
prompt-injection defense
safety policies
human-in-the-loop controls
deterministic evaluation
agent trajectory evaluation
FastAPI
production-oriented AI architecture
Portfolio Position

NexusOps AI is the AI Engineering / Agentic AI flagship of the portfolio.

The three-project portfolio covers:

Project 1
NexusOps AI
AI Reasoning + Agents + RAG + Safety

Project 2
Personalization / Search / Ranking Engine
Retrieval + Recommendation + Ranking + Streaming ML

Project 3
Multimodal Edge Vision & Video Intelligence
Computer Vision + Video + Edge Inference

Together they demonstrate breadth across:

Data Science
Machine Learning
AI Engineering
LLM Systems
Retrieval
Recommendation
Computer Vision
Production Engineering
Evaluation
Safety

