from __future__ import annotations

import json
from typing import Any, TypedDict

from src.agent.state import InvestigationState
from src.evaluation.cases import EvaluationCase

WORKFLOW_CANONICAL_STEPS: list[str] = [
    "PLAN",
    "RETRIEVE",
    "TOOL_CALL",
    "ANALYZE",
    "VERIFY",
    "ANSWER",
]


class StepEvaluation(TypedDict):
    step: str
    passed: bool
    details: dict[str, Any]


class TrajectoryResult(TypedDict):
    case_id: str
    steps: list[StepEvaluation]
    checks: dict[str, bool]
    anomalies: dict[str, bool]
    trajectory_correct: bool
    details: dict[str, Any]


class TrajectoryEvaluator:
    """
    Evaluates agent execution trajectories across the intermediate execution path:
    PLAN -> RETRIEVE -> TOOL_CALL -> ANALYZE -> VERIFY -> ANSWER.
    """

    @staticmethod
    def evaluate_trajectory(
        case: EvaluationCase | dict[str, Any],
        state: InvestigationState,
        required_steps: list[str] | None = None,
        optional_steps: list[str] | None = None,
    ) -> TrajectoryResult:
        case_id = case.get("id", "UNKNOWN")
        expected_tools = case.get("expected_tools", [])
        expected_args = case.get("expected_tool_args", {})
        expected_cause = case.get("expected_cause", "")
        prohibited_tools = set(case.get("prohibited_tools", []))
        strict_tools = case.get("strict_tools", False)

        # -------------------------------------------------------------
        # 1. Extract Trajectory Step Sequence
        # -------------------------------------------------------------
        steps_in_log: list[str] = []
        for entry in state.audit_log:
            step_name = entry.get("step")
            if step_name and (not steps_in_log or steps_in_log[-1] != step_name):
                steps_in_log.append(step_name)

        if not steps_in_log and state.current_step:
            steps_in_log = [state.current_step]

        # 2. Extract tools and arguments
        actual_tools = [call.get("tool", "") for call in state.tool_calls]
        recorded_calls = state.tool_calls

        # 3. Determine applicable steps
        if required_steps is None:
            req_steps = list(case.get("required_steps", WORKFLOW_CANONICAL_STEPS))
        else:
            req_steps = list(required_steps)

        opt_steps = set(optional_steps or case.get("optional_steps", []))

        # -------------------------------------------------------------
        # 4. Evaluate Required Checks
        # -------------------------------------------------------------
        # 1) plan_correct: non-empty plan with actionable steps
        plan_correct = bool(state.plan and len(state.plan) > 0)

        # 2) retrieval_correct: retrieval logged or evidence populated
        retrieval_logged = any(
            e.get("event") == "retrieval_started" or e.get("step") == "RETRIEVE"
            for e in state.audit_log
        )
        retrieval_correct = retrieval_logged or bool(state.evidence)

        # 3) tool_selected_correctly: all expected tools are present
        if expected_tools:
            tool_selected_correctly = all(t in actual_tools for t in expected_tools)
        else:
            tool_selected_correctly = True

        # 4) tool_arguments_correct: expected arguments match the
        # correct recorded call for each expected tool.
        if expected_args and recorded_calls:
            tool_arguments_correct = True

            # New format:
            # {
            #     "tool_a": {"arg": "value"},
            #     "tool_b": {"other_arg": "value"},
            # }
            #
            # Legacy format:
            # {
            #     "service": "payment-service"
            # }
            #
            # Legacy format is applied to each expected tool for
            # backwards compatibility.

            is_per_tool_args = (
                isinstance(expected_args, dict)
                and bool(expected_args)
                and all(
                    isinstance(value, dict)
                    for key, value in expected_args.items()
                    if key in expected_tools
                )
                and any(
                    key in expected_tools
                    for key in expected_args
                )
            )

            for expected_tool in expected_tools:
                matching_calls = [
                    call
                    for call in recorded_calls
                    if call.get("tool") == expected_tool
                ]

                if not matching_calls:
                    tool_arguments_correct = False
                    break

                if is_per_tool_args:
                    tool_expected_args = expected_args.get(
                        expected_tool,
                        {},
                    )
                else:
                    tool_expected_args = expected_args

                tool_has_correct_args = any(
                    all(
                        call.get("arguments", {}).get(key)
                        == expected_val
                        for key, expected_val
                        in tool_expected_args.items()
                    )
                    for call in matching_calls
                )

                if not tool_has_correct_args:
                    tool_arguments_correct = False
                    break

        elif expected_args and not recorded_calls:
            tool_arguments_correct = False
        else:
            tool_arguments_correct = True

        # 5) analysis_present: hypotheses formed or analysis recorded
        analysis_present = bool(
            state.hypotheses
            or state.recommendation
            or any(e.get("step") == "ANALYZE" for e in state.audit_log)
        )

        # 6) verification_correct: verification step executed and passed
        verification_passed = False
        if isinstance(state.verification, dict):
            verification_passed = state.verification.get("passed", False)
        elif isinstance(state.verification, list) and state.verification:
            verification_passed = any(
                v.get("passed", False) for v in state.verification if isinstance(v, dict)
            )
        verification_logged = any(e.get("step") == "VERIFY" for e in state.audit_log)
        verification_correct = verification_logged and verification_passed

        # 7) final_answer_correct: terminal step reached with valid answer
        has_answer = bool(state.final_answer and state.final_answer.strip())
        is_answer_step = (state.current_step == "ANSWER") or any(
            e.get("step") == "ANSWER" for e in state.audit_log
        )
        if expected_cause and expected_cause != "N/A":
            cause_lower = expected_cause.lower()
            ans_lower = state.final_answer.lower()
            cause_match = (
                cause_lower in ans_lower
                or any(w in ans_lower for w in cause_lower.split() if len(w) > 4)
            )
            final_answer_correct = is_answer_step and has_answer and cause_match
        else:
            final_answer_correct = is_answer_step and has_answer

        # -------------------------------------------------------------
        # 5. Detect Trajectory Anomalies
        # -------------------------------------------------------------
        # a) unexpected_tool
        unexpected_tool = False
        if prohibited_tools:
            unexpected_tool = any(t in prohibited_tools for t in actual_tools)
        elif strict_tools and expected_tools:
            unexpected_tool = any(t not in expected_tools for t in actual_tools)

        # b) missing_tool
        missing_tool = False
        if expected_tools:
            missing_tool = any(t not in actual_tools for t in expected_tools)

        # c) wrong_tool_order
        wrong_tool_order = False
        # Tool sequence check
        if len(expected_tools) > 1:
            matched_indices = [
                actual_tools.index(t) for t in expected_tools if t in actual_tools
            ]
            if matched_indices != sorted(matched_indices):
                wrong_tool_order = True

        # Workflow step sequence check
        canonical_indices = [
            WORKFLOW_CANONICAL_STEPS.index(s)
            for s in steps_in_log
            if s in WORKFLOW_CANONICAL_STEPS
        ]
        if canonical_indices != sorted(canonical_indices):
            wrong_tool_order = True

        # d) invalid_arguments
        invalid_arguments = False
        if expected_args and not tool_arguments_correct:
            invalid_arguments = True

        for call in recorded_calls:
            res = call.get("result", {})
            if isinstance(res, dict) and "error" in res:
                err_text = str(res["error"]).lower()
                if (
                    "argument" in err_text
                    or "parameter" in err_text
                    or "must be greater" in err_text
                    or "invalid" in err_text
                ):
                    invalid_arguments = True

        # e) failed_recovery
        failed_recovery = False
        saw_tool_error = False
        for call in recorded_calls:
            res = call.get("result", {})
            if isinstance(res, dict):
                if res.get("success") is False or "error" in res:
                    saw_tool_error = True
        if saw_tool_error and not (verification_correct and final_answer_correct):
            failed_recovery = True

        # f) missing_step / unexpected_step
        missing_step = any(
            s not in steps_in_log for s in req_steps if s not in opt_steps
        )
        unexpected_step = any(
            s not in req_steps and s not in opt_steps and s not in WORKFLOW_CANONICAL_STEPS
            for s in steps_in_log
        )

        # -------------------------------------------------------------
        # 6. Build Step Evaluations
        # -------------------------------------------------------------
        step_eval_map = {
            "PLAN": plan_correct,
            "RETRIEVE": retrieval_correct,
            "TOOL_CALL": (
                tool_selected_correctly
                and tool_arguments_correct
                and not unexpected_tool
                and not missing_tool
            ),
            "ANALYZE": analysis_present,
            "VERIFY": verification_correct,
            "ANSWER": final_answer_correct,
        }

        steps: list[StepEvaluation] = []
        for s in req_steps:
            if s in steps_in_log:
                step_passed = step_eval_map.get(s, True)
            else:
                step_passed = s in opt_steps

            steps.append(
                {
                    "step": s,
                    "passed": step_passed,
                    "details": {
                        "executed": s in steps_in_log,
                        "optional": s in opt_steps,
                    },
                }
            )

        checks = {
            "plan_correct": plan_correct,
            "retrieval_correct": retrieval_correct,
            "tool_selected_correctly": tool_selected_correctly,
            "tool_arguments_correct": tool_arguments_correct,
            "analysis_present": analysis_present,
            "verification_correct": verification_correct,
            "final_answer_correct": final_answer_correct,
        }

        anomalies = {
            "unexpected_tool": unexpected_tool,
            "missing_tool": missing_tool,
            "wrong_tool_order": wrong_tool_order,
            "invalid_arguments": invalid_arguments,
            "failed_recovery": failed_recovery,
            "missing_step": missing_step,
            "unexpected_step": unexpected_step,
        }

        # Overall trajectory correct:
        # All required steps passed, with no ordering, step omission, or recovery failures
        trajectory_correct = (
            all(s["passed"] for s in steps)
            and not wrong_tool_order
            and not missing_step
            and not unexpected_step
            and not failed_recovery
        )

        return {
            "case_id": case_id,
            "steps": steps,
            "checks": checks,
            "anomalies": anomalies,
            "trajectory_correct": trajectory_correct,
            "details": {
                "steps_in_log": steps_in_log,
                "actual_tools": actual_tools,
            },
        }


def calculate_trajectory_metrics(
    trajectory_results: list[TrajectoryResult],
) -> dict[str, Any]:
    """
    Aggregate metrics and anomaly counts across a collection of trajectory results.
    """
    total = len(trajectory_results)
    if total == 0:
        return {
            "total_trajectories": 0,
            "trajectory_success_rate": 0.0,
            "plan_accuracy": 0.0,
            "retrieval_step_accuracy": 0.0,
            "tool_selection_accuracy": 0.0,
            "tool_argument_accuracy": 0.0,
            "verification_step_accuracy": 0.0,
            "final_answer_accuracy": 0.0,
            "anomaly_counts": {
                "wrong_tool": 0,
                "wrong_arguments": 0,
                "wrong_order": 0,
                "missing_step": 0,
                "unexpected_step": 0,
            },
        }

    succ_count = sum(1 for t in trajectory_results if t["trajectory_correct"])
    plan_count = sum(1 for t in trajectory_results if t["checks"]["plan_correct"])
    ret_count = sum(1 for t in trajectory_results if t["checks"]["retrieval_correct"])
    tool_sel_count = sum(1 for t in trajectory_results if t["checks"]["tool_selected_correctly"])
    tool_arg_count = sum(1 for t in trajectory_results if t["checks"]["tool_arguments_correct"])
    ver_count = sum(1 for t in trajectory_results if t["checks"]["verification_correct"])
    ans_count = sum(1 for t in trajectory_results if t["checks"]["final_answer_correct"])

    wrong_tool_count = sum(1 for t in trajectory_results if t["anomalies"]["unexpected_tool"])
    wrong_args_count = sum(1 for t in trajectory_results if t["anomalies"]["invalid_arguments"])
    wrong_order_count = sum(1 for t in trajectory_results if t["anomalies"]["wrong_tool_order"])
    missing_step_count = sum(1 for t in trajectory_results if t["anomalies"]["missing_step"])
    unexp_step_count = sum(1 for t in trajectory_results if t["anomalies"]["unexpected_step"])

    return {
        "total_trajectories": total,
        "trajectory_success_rate": round(succ_count / total, 4),
        "plan_accuracy": round(plan_count / total, 4),
        "retrieval_step_accuracy": round(ret_count / total, 4),
        "tool_selection_accuracy": round(tool_sel_count / total, 4),
        "tool_argument_accuracy": round(tool_arg_count / total, 4),
        "verification_step_accuracy": round(ver_count / total, 4),
        "final_answer_accuracy": round(ans_count / total, 4),
        "anomaly_counts": {
            "wrong_tool": wrong_tool_count,
            "wrong_arguments": wrong_args_count,
            "wrong_order": wrong_order_count,
            "missing_step": missing_step_count,
            "unexpected_step": unexp_step_count,
        },
    }
