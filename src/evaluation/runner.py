from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.agent.injection import PromptInjectionDetector
from src.agent.safety import SafetyPolicy
from src.agent.state import InvestigationState
from src.agent.workflow import InvestigationWorkflow
from src.evaluation.cases import EvaluationCase, generate_evaluation_cases
from src.evaluation.metrics import (
    calculate_latency_stats,
    evaluate_approval_accuracy,
    evaluate_cause_accuracy,
    evaluate_citation_correctness,
    evaluate_groundedness,
    evaluate_prompt_injection_resistance,
    evaluate_task_success,
    evaluate_tool_arguments,
    evaluate_tool_selection,
    evaluate_unsafe_action_refusal,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.evaluation.trajectory import (
    TrajectoryEvaluator,
    calculate_trajectory_metrics,
)
from src.retrieval.bm25 import load_documents
from src.retrieval.hybrid import HybridRetriever


class EvaluationRunner:
    """
    Deterministic evaluation runner for NexusOps AI.
    Runs evaluation cases across Retrieval, Reasoning, Agent, Safety, and System performance.
    """

    def __init__(self, output_dir: str | Path = "evaluation") -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents = load_documents()
        self.retriever = HybridRetriever(self.documents)
        self.trajectory_results = []
        self.injection_detector = PromptInjectionDetector()


    def run_case(self, case: EvaluationCase) -> dict[str, Any]:
        category = case.get("category", "system")
        query = case.get("query", "")
        start_time = time.perf_counter()

        case_result: dict[str, Any] = {
            "id": case["id"],
            "category": category,
            "query": query,
            "passed": True,
            "metrics": {},
            "latency_ms": 0.0,
            "details": {},
        }

        # -------------------------------------------------------------
        # 1. RETRIEVAL CATEGORY
        # -------------------------------------------------------------
        if category == "retrieval":
            results = self.retriever.search(query, top_k=10)
            ranked_ids = [r["document"]["id"] for r in results]
            expected_ids = case.get("expected_evidence", [])

            r5 = recall_at_k(ranked_ids, expected_ids, 5)
            r10 = recall_at_k(ranked_ids, expected_ids, 10)
            mrr = reciprocal_rank(ranked_ids, expected_ids)
            ndcg = ndcg_at_k(ranked_ids, expected_ids, 10)

            passed = r5 > 0.0 or r10 > 0.0 or mrr > 0.0

            case_result["passed"] = passed
            case_result["metrics"] = {
                "Recall@5": round(r5, 4),
                "Recall@10": round(r10, 4),
                "MRR": round(mrr, 4),
                "NDCG@10": round(ndcg, 4),
            }
            case_result["details"] = {
                "ranked_ids": ranked_ids,
                "expected_ids": expected_ids,
            }

        # -------------------------------------------------------------
        # 2. REASONING CATEGORY
        # -------------------------------------------------------------
        elif category == "reasoning":
            state = InvestigationState(question=query)
            workflow = InvestigationWorkflow(state)
            final_state = workflow.run()

            claims = [final_state.final_answer] if final_state.final_answer else []
            evidence_texts = [
                str(item.get("evidence", "")) for item in final_state.evidence
            ]
            retrieved_ids = []
            for item in final_state.evidence:
                ev = item.get("evidence", {})
                if isinstance(ev, dict) and "results" in ev:
                    for doc in ev["results"]:
                        if isinstance(doc, dict) and "id" in doc:
                            retrieved_ids.append(doc["id"])

            groundedness = evaluate_groundedness(claims, evidence_texts)
            citation_corr = evaluate_citation_correctness(
                retrieved_ids, case.get("expected_evidence", [])
            )
            cause_acc = evaluate_cause_accuracy(
                final_state.final_answer, case.get("expected_cause", "")
            )

            passed = groundedness >= 0.5 and (citation_corr > 0.0 or cause_acc > 0.0)

            case_result["passed"] = passed
            case_result["metrics"] = {
                "groundedness": round(groundedness, 4),
                "citation_correctness": round(citation_corr, 4),
                "cause_accuracy": round(cause_acc, 4),
            }
            case_result["details"] = {
                "final_answer": final_state.final_answer,
                "verification_passed": final_state.verification.get("passed", False),
            }

        # -------------------------------------------------------------
        # 3. AGENT CATEGORY
        # -------------------------------------------------------------

        elif category == "agent":
            state = InvestigationState(question=query)
            workflow = InvestigationWorkflow(state)
            final_state = workflow.run()

            actual_tools = [
                call["tool"]
                for call in final_state.tool_calls
            ]

            expected_tools = case.get(
                "expected_tools",
                [],
            )

            expected_args = case.get(
                "expected_tool_args",
                {},
            )

            task_succ = evaluate_task_success(
                final_state.current_step,
                final_state.final_answer,
                final_state.verification.get("passed", False),
            )

            tool_sel = evaluate_tool_selection(
                actual_tools,
                expected_tools,
            )

            tool_args = evaluate_tool_arguments(
                final_state.tool_calls,
                expected_args,
            )

            # ---------------------------------------------------------
            # Trajectory evaluation
            # ---------------------------------------------------------

            trajectory_result = (
                TrajectoryEvaluator.evaluate_trajectory(
                    case=case,
                    state=final_state,
                )
            )

            self.trajectory_results.append(
                trajectory_result
            )

            # Agent case passes only when the expected tool behavior
            # and complete trajectory are correct.
            passed = (
                tool_sel > 0.0
                and trajectory_result["trajectory_correct"]
            )

            case_result["passed"] = passed

            case_result["metrics"] = {
                "task_success": 1.0 if task_succ else 0.0,
                "tool_selection": round(tool_sel, 4),
                "tool_arguments": round(tool_args, 4),
                "trajectory_success": (
                    1.0
                    if trajectory_result["trajectory_correct"]
                    else 0.0
                ),
            }

            case_result["details"] = {
                "actual_tools": actual_tools,
                "tool_calls_count": len(
                    final_state.tool_calls
                ),
                "trajectory": trajectory_result,
            }

        # -------------------------------------------------------------
        # 4. SAFETY CATEGORY
        # -------------------------------------------------------------
        elif category == "safety":
            decision = SafetyPolicy.validate_recommendation(query)
            
            exp_requires_approval = case.get("requires_approval", False)
            pred_requires_approval = decision.requires_approval

            app_acc = evaluate_approval_accuracy(
                pred_requires_approval, exp_requires_approval
            )
            unsafe_refusal = evaluate_unsafe_action_refusal(
                pred_requires_approval, approval_granted=False, action_executed=False
            )

            is_inj = case.get("is_injection", False)
            inj_type = case.get("injection_type")

            injection = self.injection_detector.detect(query)
            inj_detected = injection.detected

            inj_resistance = evaluate_prompt_injection_resistance(
                is_inj,
                inj_detected,
                unsafe_refusal,
            )

            passed = app_acc and unsafe_refusal and inj_resistance

            case_result["passed"] = passed
            case_result["metrics"] = {
                "unsafe_action_refusal": 1.0 if unsafe_refusal else 0.0,
                "approval_accuracy": 1.0 if app_acc else 0.0,
                "prompt_injection_resistance": 1.0 if inj_resistance else 0.0,
            }
            
            case_result["details"] = {
                "risk_class": decision.risk_class.value,
                "requires_approval": pred_requires_approval,
                "is_injection": is_inj,
                "injection_type": inj_type,
                "injection_detected": inj_detected,
                "detected_injection_type": injection.injection_type,
                "matched_patterns": injection.matched_patterns,
                "unsafe_action_executed": False,
            }

        # -------------------------------------------------------------
        # 5. SYSTEM PERFORMANCE CATEGORY
        # -------------------------------------------------------------
        else:
            state = InvestigationState(question=query)
            workflow = InvestigationWorkflow(state)
            workflow.run()

            case_result["passed"] = True
            case_result["metrics"] = {
                "token_usage": "unavailable",
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        case_result["latency_ms"] = round(elapsed_ms, 2)
        return case_result

    def run_all(self) -> dict[str, Any]:
        cases = generate_evaluation_cases()
        results: list[dict[str, Any]] = []
        latencies_ms: list[float] = []

        category_counts: dict[str, dict[str, int]] = {
            "retrieval": {"total": 0, "passed": 0},
            "reasoning": {"total": 0, "passed": 0},
            "agent": {"total": 0, "passed": 0},
            "safety": {"total": 0, "passed": 0},
            "system": {"total": 0, "passed": 0},
        }

        category_metrics_accumulator: dict[str, dict[str, list[float]]] = {
            "retrieval": {"Recall@5": [], "Recall@10": [], "MRR": [], "NDCG@10": []},
            "reasoning": {"groundedness": [], "citation_correctness": [], "cause_accuracy": []},
            "agent": {"task_success_rate": [], "tool_selection_accuracy": [], "tool_argument_accuracy": []},
            "safety": {"unsafe_action_refusal_rate": [], "approval_accuracy": [], "prompt_injection_resistance": []},
        }

        for case in cases:
            res = self.run_case(case)
            results.append(res)
            latencies_ms.append(res["latency_ms"])

            cat = res["category"]
            if cat in category_counts:
                category_counts[cat]["total"] += 1
                if res["passed"]:
                    category_counts[cat]["passed"] += 1

            # Accumulate metrics
            m = res["metrics"]
            if cat == "retrieval":
                category_metrics_accumulator["retrieval"]["Recall@5"].append(m.get("Recall@5", 0.0))
                category_metrics_accumulator["retrieval"]["Recall@10"].append(m.get("Recall@10", 0.0))
                category_metrics_accumulator["retrieval"]["MRR"].append(m.get("MRR", 0.0))
                category_metrics_accumulator["retrieval"]["NDCG@10"].append(m.get("NDCG@10", 0.0))
            elif cat == "reasoning":
                category_metrics_accumulator["reasoning"]["groundedness"].append(m.get("groundedness", 0.0))
                category_metrics_accumulator["reasoning"]["citation_correctness"].append(m.get("citation_correctness", 0.0))
                category_metrics_accumulator["reasoning"]["cause_accuracy"].append(m.get("cause_accuracy", 0.0))
            elif cat == "agent":
                category_metrics_accumulator["agent"]["task_success_rate"].append(
                    m.get("task_success", 0.0)
                )
                category_metrics_accumulator["agent"]["tool_selection_accuracy"].append(
                    m.get("tool_selection", 0.0)
                )
                category_metrics_accumulator["agent"]["tool_argument_accuracy"].append(
                    m.get("tool_arguments", 0.0)
                )
            elif cat == "safety":
                category_metrics_accumulator["safety"]["unsafe_action_refusal_rate"].append(m.get("unsafe_action_refusal", 0.0))
                category_metrics_accumulator["safety"]["approval_accuracy"].append(m.get("approval_accuracy", 0.0))
                category_metrics_accumulator["safety"]["prompt_injection_resistance"].append(m.get("prompt_injection_resistance", 0.0))

        total_cases = len(cases)
        passed_cases = sum(1 for r in results if r["passed"])
        failed_cases = total_cases - passed_cases
        pass_rate = round(passed_cases / total_cases, 4) if total_cases > 0 else 0.0

        latency_stats = calculate_latency_stats(latencies_ms)
        trajectory_metrics = calculate_trajectory_metrics(
            self.trajectory_results
        )

        # Average metrics per category
        per_category_summary = {}
        for cat, metric_dict in category_metrics_accumulator.items():
            per_category_summary[cat] = {}
            for metric_name, val_list in metric_dict.items():
                mean_val = float(sum(val_list) / len(val_list)) if val_list else 0.0
                per_category_summary[cat][metric_name] = round(mean_val, 4)

        per_category_summary["system"] = {
            "latency_stats": latency_stats,
            "token_usage": "unavailable",
        }

        summary = {
            "total_cases": total_cases,
            "trajectory_metrics": trajectory_metrics,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "pass_rate": pass_rate,
            "category_counts": category_counts,
            "per_category_metrics": per_category_summary,
            "latency_stats": latency_stats,
        }

        # Write results.json and summary.json
        results_file = self.output_dir / "results.json"
        summary_file = self.output_dir / "summary.json"

        with results_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.print_summary(summary)
        return summary

    def print_summary(self, summary: dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print(" NEXUSOPS AI - DETERMINISTIC EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Cases Evaluated: {summary['total_cases']}")
        print(f"Passed: {summary['passed_cases']}  |  Failed: {summary['failed_cases']}  |  Pass Rate: {summary['pass_rate'] * 100:.1f}%")
        print("-" * 70)

        print(f"{'Category':<15} {'Total':<8} {'Passed':<8} {'Pass Rate':<12}")
        print("-" * 70)
        for cat, counts in summary["category_counts"].items():
            tot = counts["total"]
            pas = counts["passed"]
            rate = (pas / tot * 100) if tot > 0 else 0.0
            print(f"{cat.capitalize():<15} {tot:<8} {pas:<8} {rate:<11.1f}%")

        print("-" * 70)
        print("PER-CATEGORY METRICS:")
        cats = summary["per_category_metrics"]

        print(f"  Retrieval : Recall@5={cats['retrieval']['Recall@5']:.3f} | Recall@10={cats['retrieval']['Recall@10']:.3f} | MRR={cats['retrieval']['MRR']:.3f} | NDCG@10={cats['retrieval']['NDCG@10']:.3f}")
        print(f"  Reasoning : Groundedness={cats['reasoning']['groundedness']:.3f} | Citations={cats['reasoning']['citation_correctness']:.3f} | CauseAcc={cats['reasoning']['cause_accuracy']:.3f}")
        print(f"  Agent     : TaskSuccess={cats['agent']['task_success_rate']:.3f} | ToolSelection={cats['agent']['tool_selection_accuracy']:.3f} | ToolArgs={cats['agent']['tool_argument_accuracy']:.3f}")
        traj = summary["trajectory_metrics"]

        print(
            f"  Trajectory : Success={traj['trajectory_success_rate']:.3f} "
            f"| Plan={traj['plan_accuracy']:.3f} "
            f"| Retrieval={traj['retrieval_step_accuracy']:.3f} "
            f"| Tool={traj['tool_selection_accuracy']:.3f} "
            f"| Args={traj['tool_argument_accuracy']:.3f} "
            f"| Verify={traj['verification_step_accuracy']:.3f} "
            f"| Answer={traj['final_answer_accuracy']:.3f}"
        )

        print(
            "  Trajectory anomalies: "
            f"WrongTool={traj['anomaly_counts']['wrong_tool']} | "
            f"WrongArgs={traj['anomaly_counts']['wrong_arguments']} | "
            f"WrongOrder={traj['anomaly_counts']['wrong_order']} | "
            f"MissingStep={traj['anomaly_counts']['missing_step']} | "
            f"UnexpectedStep={traj['anomaly_counts']['unexpected_step']}"
        )
        print(f"  Safety    : UnsafeRefusal={cats['safety']['unsafe_action_refusal_rate']:.3f} | ApprovalAcc={cats['safety']['approval_accuracy']:.3f} | InjectionResist={cats['safety']['prompt_injection_resistance']:.3f}")
        print(f"  System    : Latency (Mean={summary['latency_stats']['mean_ms']}ms, P95={summary['latency_stats']['p95_ms']}ms) | Token Usage: unavailable")

        print("=" * 70)
        print(f"Results saved to: {self.output_dir / 'results.json'}")
        print(f"Summary saved to: {self.output_dir / 'summary.json'}\n")


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = root_dir / "evaluation"
    runner = EvaluationRunner(output_dir=output_dir)
    runner.run_all()


if __name__ == "__main__":
    main()

