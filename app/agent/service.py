from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai import DeferredToolRequests, DeferredToolResults

from app.agent.errors import (
    AgentExecutionError,
    FailureKind,
    contains_unexpected_failure,
    select_execution_error,
)
from app.agent.generation import AgentGeneration
from app.agent.models import AgentDependencies, AgentOutput, ApprovalStore, PendingApproval, RunState


class AgentService:
    def __init__(self) -> None:
        self.approvals = ApprovalStore()

    async def handle_message(
        self, generation: AgentGeneration, text: str, *, update_id: int, user_id: int, chat_id: int
    ) -> str:
        deps = AgentDependencies(generation, user_id, chat_id, update_id, f"telegram-{update_id}")
        generation.logger.debug(
            "agent_request_start",
            generation_id=generation.id,
            request_id=deps.request_id,
            update_id=update_id,
            user_id=user_id,
            chat_id=chat_id,
            prompt=text,
        )
        async with generation.run():
            result, failure_text = await self._run_agent(
                generation,
                deps,
                lambda: generation.agent.run(text, deps=deps),
                event="agent_request_failed",
                update_id=update_id,
            )
        if failure_text is not None:
            return failure_text
        assert result is not None
        self._log_result(generation, deps.request_id, result)
        return await self._result_text(generation, result.output, result.all_messages(), user_id, chat_id)

    async def confirm(
        self, generations: dict[str, AgentGeneration], approval_id: str, *, user_id: int, chat_id: int
    ) -> str:
        approval = await self.approvals.consume(approval_id, user_id, chat_id)
        if approval is None:
            return "Confirmation is unavailable or has expired."
        generation = generations.get(approval.generation_id)
        if generation is None:
            return "Confirmation is unavailable or has expired."
        generation.pending_approvals -= 1
        encoded = json.dumps(approval.arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(encoded.encode()).hexdigest() != approval.arguments_hash:
            return "Confirmation is invalid."
        deps = AgentDependencies(
            generation,
            user_id,
            chat_id,
            0,
            f"confirmation-{approval.approval_id}",
            RunState(approval_resume_only=True, approved_tool_call_id=approval.tool_call_id),
        )
        deferred = DeferredToolResults(approvals={approval.tool_call_id: True})
        async with generation.run():
            result, failure_text = await self._run_agent(
                generation,
                deps,
                lambda: generation.agent.run(
                    message_history=approval.message_history,
                    deferred_tool_results=deferred,
                    deps=deps,
                ),
                event="agent_confirmation_failed",
                approval_id=approval.approval_id,
            )
        if failure_text is not None:
            return failure_text
        assert result is not None
        self._log_result(generation, deps.request_id, result)
        return await self._result_text(generation, result.output, result.all_messages(), user_id, chat_id)

    async def purge_expired(self, generations: dict[str, AgentGeneration]) -> None:
        for approval in await self.approvals.purge():
            generation = generations.get(approval.generation_id)
            if generation is not None:
                generation.pending_approvals -= 1
        for generation in list(generations.values()):
            if generation.retired and generation.active_runs == 0 and generation.pending_approvals <= 0:
                await generation.close()
                generations.pop(generation.id, None)

    async def _run_agent(
        self,
        generation: AgentGeneration,
        deps: AgentDependencies,
        operation: Callable[[], Awaitable[Any]],
        *,
        event: str,
        **log_fields: Any,
    ) -> tuple[Any | None, str | None]:
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(
                operation(), generation.settings.request_timeout_seconds
            )
        except asyncio.CancelledError as exc:
            if deps.run_state.dispatched_tool_calls:
                self._handle_failure(
                    generation,
                    deps,
                    exc,
                    event=event,
                    duration_ms=_duration_ms(started),
                    **log_fields,
                )
            raise
        except asyncio.TimeoutError as exc:
            return None, self._handle_failure(
                generation,
                deps,
                exc,
                event=event,
                overall_timeout=True,
                duration_ms=_duration_ms(started),
                **log_fields,
            )
        except Exception as exc:
            execution_error = select_execution_error(exc)
            if execution_error is None and deps.run_state.dispatched_tool_calls == 0:
                raise
            return None, self._handle_failure(
                generation,
                deps,
                exc,
                event=event,
                execution_error=execution_error,
                unexpected=contains_unexpected_failure(exc),
                duration_ms=_duration_ms(started),
                **log_fields,
            )
        return result, None

    @staticmethod
    def _handle_failure(
        generation: AgentGeneration,
        deps: AgentDependencies,
        exc: BaseException,
        *,
        event: str,
        execution_error: AgentExecutionError | None = None,
        overall_timeout: bool = False,
        unexpected: bool = False,
        duration_ms: int,
        **log_fields: Any,
    ) -> str:
        state = deps.run_state
        execution_error = execution_error or select_execution_error(exc)
        has_unknown_outcome = bool(
            state.unknown_outcomes
            or state.in_flight_calls
            or (
                unexpected
                and state.dispatched_tool_calls > state.completed_tool_calls
            )
        )
        failure_kind = _failure_kind(
            execution_error, has_unknown_outcome, overall_timeout, unexpected
        )
        outcome_known = not has_unknown_outcome and (
            execution_error is None or execution_error.outcome_known
        )
        fields = dict(
            generation_id=generation.id,
            request_id=deps.request_id,
            failure_kind=failure_kind,
            exception_type=type(exc).__name__,
            server=(execution_error.server if execution_error else state.last_server),
            tool_name=(execution_error.tool_name if execution_error else state.last_tool_name),
            failure_dispatched=(
                execution_error.dispatched
                if execution_error
                else bool(state.dispatched_tool_calls)
            ),
            attempted_tool_calls=state.attempted_tool_calls,
            dispatched_tool_calls=state.dispatched_tool_calls,
            completed_tool_calls=state.completed_tool_calls,
            in_flight_tool_calls=len(state.in_flight_calls),
            unknown_outcomes=len(state.unknown_outcomes),
            outcome_known=outcome_known,
            duration_ms=duration_ms,
            **log_fields,
        )
        if unexpected:
            fields["error"] = str(exc)
            generation.logger.error(event, **fields)
        else:
            generation.logger.warning(event, **fields)
        return _failure_text(state, execution_error, has_unknown_outcome, unexpected)

    async def _result_text(self, generation: AgentGeneration, output: object, history: list, user_id: int, chat_id: int) -> str:
        if isinstance(output, AgentOutput):
            return output.text
        if not isinstance(output, DeferredToolRequests):
            return "No response was produced."
        if len(output.approvals) != 1 or output.calls:
            return "No action was performed. Submit one confirmation-required action at a time."
        call = output.approvals[0]
        approval = PendingApproval.create(
            generation_id=generation.id,
            user_id=user_id,
            chat_id=chat_id,
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            arguments=call.args if isinstance(call.args, dict) else json.loads(call.args or "{}"),
            message_history=history,
        )
        await self.approvals.add(approval)
        generation.pending_approvals += 1
        return f"Confirmation required. Run /confirm {approval.approval_id}."

    @staticmethod
    def _log_result(generation: AgentGeneration, request_id: str, result: object) -> None:
        output = getattr(result, "output", None)
        if isinstance(output, AgentOutput):
            output_data: object = output.model_dump()
        else:
            output_data = str(output)
        try:
            history = json.loads(getattr(result, "all_messages_json")().decode("utf-8"))
        except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            history = "[unavailable]"
        generation.logger.debug(
            "agent_request_complete",
            generation_id=generation.id,
            request_id=request_id,
            output=output_data,
            message_history=history,
        )


def _failure_kind(
    execution_error: AgentExecutionError | None,
    has_unknown_outcome: bool,
    overall_timeout: bool,
    unexpected: bool,
) -> str:
    if unexpected:
        return "unexpected_after_tool_dispatch"
    if has_unknown_outcome:
        return FailureKind.TOOL_OUTCOME_UNKNOWN.value
    if execution_error is not None:
        return execution_error.kind.value
    if overall_timeout:
        return "agent_timeout"
    return "unexpected_after_tool_dispatch"


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _failure_text(
    state: RunState,
    execution_error: AgentExecutionError | None,
    has_unknown_outcome: bool,
    unexpected: bool,
) -> str:
    if has_unknown_outcome:
        if unexpected:
            return (
                "A tool call failed unexpectedly after it was dispatched. "
                "Its outcome is unknown; check the current state before retrying."
            )
        return (
            "A tool call timed out or was interrupted after it was dispatched. "
            "Its outcome is unknown; check the current state before retrying."
        )
    if execution_error is not None and execution_error.kind is FailureKind.TOOL_REPORTED_FAILURE:
        if state.completed_tool_calls:
            return (
                "One or more tool calls completed, then another tool reported a failure. "
                "Check the current state before retrying."
            )
        return (
            "The tool reported a failure, so completion was not confirmed. "
            "Check the current state before retrying."
        )
    if state.completed_tool_calls:
        if execution_error is not None and execution_error.kind is FailureKind.POLICY_BLOCKED:
            return (
                "One or more tool calls completed, but a later call was blocked by the bot's safety policy. "
                "Check the current state before retrying."
            )
        if execution_error is not None and execution_error.kind is FailureKind.TOOL_UNAVAILABLE:
            return (
                "One or more tool calls completed, but a required service later became unavailable. "
                "Check the current state before retrying."
            )
        return (
            "One or more tool calls completed, but the agent could not produce a final response. "
            "Check the current state before retrying."
        )
    if state.dispatched_tool_calls:
        return (
            "The agent request failed after a tool was dispatched, so completion was not confirmed. "
            "Check the current state before retrying."
        )
    if execution_error is not None and execution_error.kind is FailureKind.POLICY_BLOCKED:
        return "The requested action was blocked by the bot's safety policy. No tool action was performed."
    if execution_error is not None and execution_error.kind is FailureKind.TOOL_UNAVAILABLE:
        return "The required service is temporarily unavailable. No tool action was performed."
    return "The agent request timed out before any tool was dispatched. No tool action was performed."
