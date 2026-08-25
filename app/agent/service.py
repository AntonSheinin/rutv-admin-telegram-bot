from __future__ import annotations

import asyncio
import hashlib
import json

from pydantic_ai import DeferredToolRequests, DeferredToolResults

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
            try:
                result = await asyncio.wait_for(
                    generation.agent.run(text, deps=deps), generation.settings.request_timeout_seconds
                )
            except Exception as exc:
                generation.logger.debug(
                    "agent_request_failed",
                    generation_id=generation.id,
                    request_id=deps.request_id,
                    update_id=update_id,
                    error=str(exc),
                )
                raise
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
            try:
                result = await asyncio.wait_for(
                    generation.agent.run(
                        message_history=approval.message_history,
                        deferred_tool_results=deferred,
                        deps=deps,
                    ),
                    generation.settings.request_timeout_seconds,
                )
            except Exception as exc:
                generation.logger.debug(
                    "agent_confirmation_failed",
                    generation_id=generation.id,
                    request_id=deps.request_id,
                    approval_id=approval.approval_id,
                    error=str(exc),
                )
                raise
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
