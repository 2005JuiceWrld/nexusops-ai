# NexusOps AI

### Enterprise AI Operations Reasoning & Decision Intelligence Platform

NexusOps AI is an enterprise-style AI operations investigation platform that combines **LLM reasoning, hybrid retrieval, deterministic evidence verification, service dependency graphs, safety policies, and audit logging**.

The system investigates operational incidents, correlates evidence across incidents and deployments, reasons about likely causes using a local LLM, verifies the generated reasoning against trusted evidence, classifies remediation risk, and prevents production-impacting actions from being executed without explicit approval.

> **Core principle:** The LLM reasons; deterministic systems verify and control actions.

---

## Problem

Traditional operational investigation requires engineers to manually correlate:

* incidents
* deployment history
* service health
* service dependencies
* remediation runbooks
* production policies

An LLM can accelerate this investigation, but unrestricted LLM-generated actions introduce risks such as:

* hallucinated root causes
* unsupported evidence
* fabricated operational records
* unsafe production recommendations
* treating an LLM statement as authorization

NexusOps addresses these problems by separating **reasoning from verification and execution control**.

---

## Architecture

```text
                    User Question
                         │
                         ▼
                ┌─────────────────┐
                │ FastAPI / Agent │
                └────────┬────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │    PLAN     │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   RETRIEVE  │
                  └──────┬──────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     BM25 Retrieval              Dense Retrieval
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  RRF Fusion
                         │
                         ▼
                Cross-Encoder Reranker
                         │
                         ▼
                     Evidence
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Incidents      Deployments    Service Status
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                Dependency Graph
                         │
                         ▼
                  Local LLM
                 (Ollama)
                         │
                         ▼
                Structured Reasoning
                         │
                         ▼
              ┌────────────────────┐
              │ Evidence Verifier  │
              └─────────┬──────────┘
                        │
                        ▼
                 Safety Policy
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         Read / Safe       Production Change
                                │
                                ▼
                         Human Approval
                                │
                                ▼
                         Audit Logging
```

---

## Agent Workflow

The investigation follows six deterministic stages:

```text
PLAN
  ↓
RETRIEVE
  ↓
TOOL_CALL
  ↓
ANALYZE
  ↓
VERIFY
  ↓
ANSWER
```

### PLAN

Creates an investigation plan based on the operational question.

Example:

1. Search recent payment incidents
2. Inspect payment-service deployments
3. Check payment-service status
4. Check fraud-service status
5. Retrieve remediation runbook
6. Check production rollback policy
7. Correlate evidence and determine likely cause

### RETRIEVE

Collects relevant operational information.

### TOOL_CALL

Queries operational tools for:

* incidents
* deployments
* service status
* runbooks
* production policies

A service dependency graph is also consulted to establish relationships between services.

### ANALYZE

The collected evidence and deterministic hypothesis are passed to the local LLM.

The LLM produces structured JSON containing:

```text
summary
likely_cause
supporting_evidence
alternative_explanations
recommended_action
confidence
```

The LLM is explicitly instructed not to execute actions.

### VERIFY

The Evidence Verifier checks:

* evidence availability
* LLM output structure
* confidence validity
* causal grounding
* supporting evidence
* recommendation presence
* policy consistency
* unsupported claims

The verifier supports both textual evidence references and structured evidence objects.

Fabricated operational IDs such as `INC-999` are rejected.

### SAFETY

Recommendations are independently classified by `SafetyPolicy`.

For example:

```text
rollback
restart
deploy
delete
disable
terminate
scale
```

are treated as potentially production-impacting.

The LLM cannot authorize these actions.

A rollback recommendation therefore produces:

```text
allowed = false
requires_approval = true
risk_class = production_change
```

### ANSWER

Only verified reasoning is presented as a verified investigation result.

Production-impacting recommendations explicitly state that human approval is required.

---

# Hybrid Retrieval

NexusOps uses multiple retrieval strategies rather than relying on a single search mechanism.

## BM25

Lexical retrieval is useful for exact operational terminology such as:

```text
payment-service
DEP-001
INC-001
2.4.1
timeout
fraud-service
```

## Dense Retrieval

Semantic embeddings are used to retrieve conceptually related operational evidence.

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Reciprocal Rank Fusion

BM25 and dense results are combined using **Reciprocal Rank Fusion (RRF)**.

## Cross-Encoder Reranking

The fused candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This produces the final evidence ranking supplied to downstream reasoning.

---

# Service Dependency Graph

NexusOps uses NetworkX to model operational relationships.

Example:

```text
payment-service
      │
      │ depends_on
      ▼
fraud-service
```

The graph also connects:

```text
DEP-001 ──deployed_to──► payment-service
INC-001 ──affected─────► payment-service
```

This allows the investigation to reason about relationships rather than treating operational records as isolated documents.

---

# Operational Tools

The platform provides deterministic operational tools including:

```text
search_incidents()
get_recent_deployments()
get_service_status()
retrieve_runbook()
get_production_policy()
calculate_metrics()
restart_service()
```

`restart_service()` is implemented as a **safe simulation**.

It does not perform a real production restart.

Without explicit approval:

```text
allowed = false
requires_approval = true
```

---

# Safety Architecture

Safety is intentionally separated from LLM reasoning.

```text
LLM
 │
 │ recommendation
 ▼
Evidence Verifier
 │
 │ verified reasoning
 ▼
SafetyPolicy
 │
 ├── read-only
 │
 └── production-impacting
          │
          ▼
    Human Approval
```

Statements such as:

```text
"approved"
"execute immediately"
```

inside an LLM response do **not** count as authorization.

Authorization must come from an external approval signal.

Unknown or destructive tool classifications fail closed.

---

# Example Investigation

### Question

```text
Payment failures increased after the latest deployment.
Investigate the likely cause using incidents, deployment
history, service dependencies and runbooks. Recommend a
remediation and determine whether approval is required.
```

### Evidence

The system correlates:

```text
INC-001
Payment failures increased after deployment 2.4.1.

DEP-001
payment-service 2.4.1 introduced changes to fraud
verification request handling and timeout configuration.

fraud-service
Status: degraded
p95 latency: 620 ms
error rate: 8.4%

Dependency graph
payment-service → fraud-service
```

### LLM reasoning

```text
Likely cause:
Increased fraud-service latency combined with changes
introduced by payment-service version 2.4.1.

Confidence:
0.7
```

### Verification

```text
passed = true
```

All verification checks pass:

```text
evidence_available             = true
llm_structure_valid            = true
confidence_valid               = true
cause_supported                = true
supporting_evidence_valid     = true
recommendation_present         = true
recommendation_policy_consistent = true
unsupported_claims_valid       = true
```

### Recommendation

```text
Rollback payment-service version 2.4.1 and inspect
recent deployment configuration changes.
```

### Safety decision

```text
risk_class       = production_change
allowed          = false
requires_approval = true
```

The system therefore recommends the remediation but does not execute it.

---

# Evaluation

The project includes an evaluation suite covering:

1. grounded causal explanations
2. hallucinated or unsupported causes
3. unsupported evidence claims
4. unsafe production recommendations
5. malformed LLM output
6. existing end-to-end behavior

Current result:

```text
9 passed
```

The evaluation demonstrates that the verifier is not simply accepting whatever the LLM produces.

---

# Audit Logging

Each investigation records a structured audit trail.

Example events:

```text
plan_created
retrieval_started
tool_calls_completed
hypothesis_generated
llm_analysis_completed
llm_reasoning_recorded
safety_policy_evaluated
verification_completed
answer_generated
```

The audit database uses SQLite and is intentionally excluded from Git.

This provides traceability for:

* what was investigated
* which tools were called
* what evidence was collected
* what the LLM reasoned about
* what verification checks passed
* what safety decision was made
* whether approval was required

---

# Technology Stack

### Backend

* Python 3.12
* FastAPI
* Pydantic
* Uvicorn

### Retrieval

* BM25
* Sentence Transformers
* FAISS
* Reciprocal Rank Fusion
* Cross-Encoder reranking

### AI

* Ollama
* Llama 3.2 3B
* `all-MiniLM-L6-v2`
* `ms-marco-MiniLM-L-6-v2`

### Reasoning & Safety

* deterministic evidence verification
* NetworkX dependency graph
* deterministic safety policy
* approval gates
* structured audit logging

### Data / Utilities

* NumPy
* Pandas
* scikit-learn
* SQLite
* httpx

---

# Project Structure

```text
nexusops-ai/
│
├── src/
│   ├── agent/
│   │   ├── audit.py
│   │   ├── llm.py
│   │   ├── safety.py
│   │   ├── state.py
│   │   ├── tools.py
│   │   ├── verifier.py
│   │   └── workflow.py
│   │
│   ├── graph/
│   │   └── service_graph.py
│   │
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── evaluation.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   └── data/
│       └── generate_dataset.py
│
├── tests/
│   ├── test_agent_evaluation.py
│   └── test_service_graph.py
│
├── data/
│   └── raw/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Running Locally

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Make sure Ollama is running and the required model is available:

```powershell
ollama run llama3.2:3b
```

Run the investigation:

```powershell
python -m src.agent.workflow
```

Run the test suite:

```powershell
python -m pytest -q
```

Expected result:

```text
9 passed
```

---

# Design Principles

### 1. LLMs are reasoning components, not authorities

The model generates hypotheses and recommendations.

It does not authorize production actions.

### 2. Evidence must be independently verified

LLM claims are checked against trusted operational evidence.

### 3. Safety is deterministic

Risk classification is handled by explicit policy rather than model judgment.

### 4. Fail closed

Unknown or destructive operations are denied by default.

### 5. Auditability matters

Agent decisions and workflow transitions are recorded for traceability.

### 6. Production actions require explicit authorization

A recommendation is not an execution command.

---

# Key Engineering Takeaway

NexusOps demonstrates a controlled architecture for AI-assisted operational decision making:

```text
Reason
  ↓
Ground
  ↓
Verify
  ↓
Classify Risk
  ↓
Require Approval
  ↓
Act Only When Authorized
```

The goal is not to make the LLM autonomous.

The goal is to make the **overall system reliable enough to use AI reasoning inside controlled operational workflows**.
