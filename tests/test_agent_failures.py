import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

from app.agent.errors import (
    AgentExecutionError,
    FailureKind,
    contains_unexpected_failure,
    select_execution_error,
)
from app.agent.generation import AGENT_INSTRUCTIONS, PolicyToolset
from app.agent.models import AgentDependencies, PendingApproval, RunState, ToolCallMetadata
from app.agent.service import AgentService


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def debug(self, event: str, **fields) -> None:
        self.events.append(("DEBUG", event, fields))

    def info(self, event: str, **fields) -> None:
        self.events.append(("INFO", event, fields))

    def warning(self, event: str, **fields) -> None:
        self.events.append(("WARNING", event, fields))

    def error(self, event: str, **fields) -> None:
        self.events.append(("ERROR", event, fields))


class FakeWrappedToolset:
    def __init__(self, behavior) -> None:
        self.behavior = behavior

    async def call_tool(self, name, tool_args, ctx, tool):
        return await self.behavior()


def make_policy(
    behavior, *, state: RunState | None = None, timeout: float = 1.0, concurrency: int = 1
):
    server = SimpleNamespace(name="playlist", tool_timeout_seconds=timeout)
    logger = RecordingLogger()
    settings = SimpleNamespace(
        max_tool_calls=5,
        max_tool_argument_bytes=1024,
        max_tool_result_bytes=1024,
    )
    generation = SimpleNamespace(
        id="generation",
        settings=settings,
        semaphores={"playlist": asyncio.Semaphore(concurrency)},
        logger=logger,
        server_for_tool=lambda _name: server,
    )
    deps = AgentDependencies(generation, 1, 2, 3, "request", state or RunState())
    ctx = SimpleNamespace(deps=deps, tool_call_approved=False)
    return PolicyToolset(FakeWrappedToolset(behavior)), ctx, logger


@pytest.mark.asyncio
async def test_policy_toolset_classifies_tool_error_and_hashes_failed_signature():
    async def fail():
        raise ToolError("Playlist Service returned 404: User not found")

    policy, ctx, logger = make_policy(fail)

    with pytest.raises(AgentExecutionError) as exc_info:
        await policy.call_tool("playlist_find_user", {"q": "secret customer"}, ctx, None)

    error = exc_info.value
    state = ctx.deps.run_state
    expected_hash = hashlib.sha256(
        json.dumps({"q": "secret customer"}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert error.kind is FailureKind.TOOL_REPORTED_FAILURE
    assert error.dispatched is True
    assert state.dispatched_tool_calls == 1
    assert state.completed_tool_calls == 0
    assert state.failed_signatures == {("playlist_find_user", expected_hash)}
    assert not state.in_flight_calls
    failure_event = next(fields for _, event, fields in logger.events if event == "mcp_tool_call_failed")
    assert "arguments" not in failure_event
    assert "secret customer" not in str(failure_event)
    assert failure_event["outcome_known"] is False


@pytest.mark.asyncio
async def test_policy_toolset_timeout_records_unknown_outcome_and_blocks_retry():
    async def wait_forever():
        await asyncio.Event().wait()

    policy, ctx, _logger = make_policy(wait_forever, timeout=0.01)
    args = {"user_id": 10}

    with pytest.raises(AgentExecutionError) as exc_info:
        await policy.call_tool("playlist_get_user", args, ctx, None)

    assert exc_info.value.kind is FailureKind.TOOL_OUTCOME_UNKNOWN
    assert ctx.deps.run_state.unknown_outcomes
    assert not ctx.deps.run_state.in_flight_calls
    with pytest.raises(AgentExecutionError) as retry_info:
        await policy.call_tool("playlist_get_user", args, ctx, None)
    assert retry_info.value.kind is FailureKind.POLICY_BLOCKED
    assert ctx.deps.run_state.dispatched_tool_calls == 1


@pytest.mark.asyncio
async def test_policy_toolset_cancellation_records_unknown_and_propagates():
    started = asyncio.Event()

    async def wait_forever():
        started.set()
        await asyncio.Event().wait()

    policy, ctx, _logger = make_policy(wait_forever)
    task = asyncio.create_task(policy.call_tool("playlist_get_user", {"user_id": 10}, ctx, None))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert ctx.deps.run_state.unknown_outcomes
    assert not ctx.deps.run_state.in_flight_calls


@pytest.mark.asyncio
async def test_start_event_is_not_emitted_before_semaphore_dispatch():
    waiting = asyncio.Event()

    class WaitingSemaphore:
        def locked(self):
            return False

        async def __aenter__(self):
            waiting.set()
            await asyncio.Event().wait()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def succeed():
        return {"ok": True}

    policy, ctx, logger = make_policy(succeed)
    ctx.deps.generation.semaphores["playlist"] = WaitingSemaphore()
    task = asyncio.create_task(
        policy.call_tool("playlist_get_user", {"user_id": 10}, ctx, None)
    )
    await waiting.wait()

    assert not any(event == "mcp_tool_call_start" for _, event, _ in logger.events)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ctx.deps.run_state.dispatched_tool_calls == 0
    assert not ctx.deps.run_state.unknown_outcomes


@pytest.mark.asyncio
async def test_policy_toolset_tracks_identical_parallel_calls_separately():
    both_started = asyncio.Event()
    release = asyncio.Event()
    starts = 0

    async def wait_for_release():
        nonlocal starts
        starts += 1
        if starts == 2:
            both_started.set()
        await release.wait()
        return {"ok": True}

    policy, ctx, _logger = make_policy(wait_for_release, concurrency=2)
    first = asyncio.create_task(policy.call_tool("playlist_get_user", {"user_id": 10}, ctx, None))
    second = asyncio.create_task(policy.call_tool("playlist_get_user", {"user_id": 10}, ctx, None))
    await both_started.wait()

    assert len(ctx.deps.run_state.in_flight_calls) == 2
    release.set()
    await asyncio.gather(first, second)
    assert not ctx.deps.run_state.in_flight_calls
    assert ctx.deps.run_state.dispatched_tool_calls == 2
    assert ctx.deps.run_state.completed_tool_calls == 2


def test_parallel_failure_selection_uses_safest_category():
    grouped = ExceptionGroup(
        "parallel calls",
        [
            AgentExecutionError(FailureKind.POLICY_BLOCKED),
            AgentExecutionError(FailureKind.TOOL_REPORTED_FAILURE, dispatched=True),
            AgentExecutionError(
                FailureKind.TOOL_OUTCOME_UNKNOWN,
                dispatched=True,
                outcome_known=False,
            ),
        ],
    )

    selected = select_execution_error(grouped)

    assert selected is not None
    assert selected.kind is FailureKind.TOOL_OUTCOME_UNKNOWN


def test_mixed_parallel_failure_is_also_marked_unexpected():
    grouped = ExceptionGroup(
        "parallel calls",
        [
            AgentExecutionError(FailureKind.TOOL_REPORTED_FAILURE, dispatched=True),
            RuntimeError("programming defect"),
        ],
    )

    assert contains_unexpected_failure(grouped) is True
    assert contains_unexpected_failure(
        AgentExecutionError(FailureKind.TOOL_REPORTED_FAILURE, dispatched=True)
    ) is False


@pytest.mark.asyncio
async def test_tool_error_propagates_through_real_pydantic_agent():
    async def playlist_fail(q: str):
        raise ToolError("Playlist Service returned 404: User not found")

    base = FunctionToolset([playlist_fail])
    policy, ctx, _logger = make_policy(lambda: None)
    policy = PolicyToolset(base)
    agent = Agent(
        TestModel(call_tools="all"),
        deps_type=AgentDependencies,
        toolsets=[policy],
    )

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run("find a user", deps=ctx.deps)

    assert exc_info.value.kind is FailureKind.TOOL_REPORTED_FAILURE
    assert exc_info.value.dispatched is True
    assert exc_info.value.outcome_known is False


class FakeAgent:
    def __init__(self, behavior) -> None:
        self.behavior = behavior

    async def run(self, *args, **kwargs):
        return await self.behavior(kwargs["deps"])


class FakeGeneration:
    def __init__(self, behavior, *, timeout: float = 1.0) -> None:
        self.id = "generation"
        self.agent = FakeAgent(behavior)
        self.logger = RecordingLogger()
        self.settings = SimpleNamespace(request_timeout_seconds=timeout)
        self.pending_approvals = 0

    @asynccontextmanager
    async def run(self):
        yield


@pytest.mark.asyncio
async def test_service_returns_safe_no_dispatch_policy_message():
    async def blocked(_deps):
        raise AgentExecutionError(FailureKind.POLICY_BLOCKED, tool_name="playlist_find_user")

    generation = FakeGeneration(blocked)
    response = await AgentService().handle_message(generation, "hello", update_id=1, user_id=2, chat_id=3)

    assert response == "The requested action was blocked by the bot's safety policy. No tool action was performed."


@pytest.mark.asyncio
async def test_service_returns_safe_no_dispatch_unavailable_message():
    async def unavailable(_deps):
        raise AgentExecutionError(FailureKind.TOOL_UNAVAILABLE, tool_name="playlist_find_user")

    generation = FakeGeneration(unavailable)
    response = await AgentService().handle_message(generation, "hello", update_id=1, user_id=2, chat_id=3)

    assert response == "The required service is temporarily unavailable. No tool action was performed."


@pytest.mark.asyncio
async def test_service_does_not_claim_no_action_for_reported_tool_failure():
    async def fail(deps):
        deps.run_state.dispatched_tool_calls = 1
        raise AgentExecutionError(
            FailureKind.TOOL_REPORTED_FAILURE,
            server="playlist",
            tool_name="playlist_find_user",
            dispatched=True,
            outcome_known=False,
        )

    generation = FakeGeneration(fail)
    response = await AgentService().handle_message(generation, "missing", update_id=1, user_id=2, chat_id=3)

    assert "completion was not confirmed" in response
    assert "No action was performed" not in response
    failure_level, _, failure = next(
        item for item in generation.logger.events if item[1] == "agent_request_failed"
    )
    assert failure_level == "WARNING"
    assert failure["outcome_known"] is False
    assert failure["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_outer_timeout_with_in_flight_tool_reports_unknown_outcome():
    async def time_out(deps):
        signature = ("playlist_disable_user", "hash")
        deps.run_state.dispatched_tool_calls = 1
        deps.run_state.in_flight_calls["dispatch-id"] = ToolCallMetadata(
            "playlist", "playlist_disable_user", signature
        )
        await asyncio.Event().wait()

    generation = FakeGeneration(time_out, timeout=0.01)
    response = await AgentService().handle_message(generation, "disable", update_id=1, user_id=2, chat_id=3)

    assert "outcome is unknown" in response
    assert "before any tool" not in response


@pytest.mark.asyncio
async def test_outer_timeout_before_dispatch_reports_no_tool_action():
    async def time_out(_deps):
        await asyncio.Event().wait()

    generation = FakeGeneration(time_out, timeout=0.01)
    response = await AgentService().handle_message(generation, "hello", update_id=1, user_id=2, chat_id=3)

    assert response == "The agent request timed out before any tool was dispatched. No tool action was performed."


@pytest.mark.asyncio
async def test_service_cancellation_after_dispatch_logs_unknown_and_propagates():
    started = asyncio.Event()

    async def wait_for_cancellation(deps):
        signature = ("playlist_disable_user", "hash")
        deps.run_state.dispatched_tool_calls = 1
        deps.run_state.unknown_outcomes["dispatch-id"] = ToolCallMetadata(
            "playlist", "playlist_disable_user", signature
        )
        started.set()
        await asyncio.Event().wait()

    generation = FakeGeneration(wait_for_cancellation)
    task = asyncio.create_task(
        AgentService().handle_message(
            generation, "disable", update_id=1, user_id=2, chat_id=3
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    failure = next(
        fields for _, event, fields in generation.logger.events if event == "agent_request_failed"
    )
    assert failure["failure_kind"] == FailureKind.TOOL_OUTCOME_UNKNOWN.value


@pytest.mark.asyncio
async def test_model_failure_after_completed_tool_reports_partial_completion():
    async def fail_after_tool(deps):
        deps.run_state.dispatched_tool_calls = 1
        deps.run_state.completed_tool_calls = 1
        raise RuntimeError("model failed")

    generation = FakeGeneration(fail_after_tool)
    response = await AgentService().handle_message(generation, "change", update_id=1, user_id=2, chat_id=3)

    assert "tool calls completed" in response
    assert "No action was performed" not in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (FailureKind.TOOL_REPORTED_FAILURE, "another tool reported a failure"),
        (FailureKind.POLICY_BLOCKED, "later call was blocked"),
        (FailureKind.TOOL_UNAVAILABLE, "service later became unavailable"),
    ],
)
async def test_known_failure_after_completed_tool_mentions_earlier_completion(kind, expected):
    async def fail_after_tool(deps):
        deps.run_state.dispatched_tool_calls = 1
        deps.run_state.completed_tool_calls = 1
        raise AgentExecutionError(kind, server="playlist", tool_name="playlist_disable_user")

    generation = FakeGeneration(fail_after_tool)
    response = await AgentService().handle_message(
        generation, "change", update_id=1, user_id=2, chat_id=3
    )

    assert expected in response
    assert "No action was performed" not in response


@pytest.mark.asyncio
async def test_unexpected_failure_after_dispatch_uses_conservative_response():
    async def fail_after_dispatch(deps):
        deps.run_state.dispatched_tool_calls = 1
        raise RuntimeError("unexpected transport failure")

    generation = FakeGeneration(fail_after_dispatch)
    response = await AgentService().handle_message(
        generation, "change", update_id=1, user_id=2, chat_id=3
    )

    assert "failed unexpectedly after it was dispatched" in response
    assert "No action was performed" not in response
    failure_level, _, failure = next(
        item for item in generation.logger.events if item[1] == "agent_request_failed"
    )
    assert failure_level == "ERROR"
    assert failure["failure_kind"] == "unexpected_after_tool_dispatch"
    assert failure["outcome_known"] is False
    assert failure["error"] == "unexpected transport failure"


@pytest.mark.asyncio
async def test_mixed_parallel_failure_is_logged_as_unexpected():
    async def fail_in_parallel(deps):
        deps.run_state.dispatched_tool_calls = 2
        raise ExceptionGroup(
            "parallel calls",
            [
                AgentExecutionError(
                    FailureKind.TOOL_REPORTED_FAILURE,
                    dispatched=True,
                    outcome_known=False,
                ),
                RuntimeError("programming defect"),
            ],
        )

    generation = FakeGeneration(fail_in_parallel)
    response = await AgentService().handle_message(
        generation, "change", update_id=1, user_id=2, chat_id=3
    )

    assert "outcome is unknown" in response
    failure_level, _, failure = next(
        item for item in generation.logger.events if item[1] == "agent_request_failed"
    )
    assert failure_level == "ERROR"
    assert failure["failure_kind"] == "unexpected_after_tool_dispatch"


@pytest.mark.asyncio
async def test_unexpected_failure_before_tool_dispatch_still_propagates():
    async def fail(_deps):
        raise RuntimeError("programming defect")

    generation = FakeGeneration(fail)

    with pytest.raises(RuntimeError, match="programming defect"):
        await AgentService().handle_message(generation, "hello", update_id=1, user_id=2, chat_id=3)


@pytest.mark.asyncio
async def test_confirmation_uses_failure_mapping_and_remains_single_use():
    async def blocked(_deps):
        raise AgentExecutionError(FailureKind.POLICY_BLOCKED, tool_name="playlist_disable_user")

    service = AgentService()
    generation = FakeGeneration(blocked)
    generation.pending_approvals = 1
    approval = PendingApproval.create(
        generation_id=generation.id,
        user_id=2,
        chat_id=3,
        tool_call_id="call",
        tool_name="playlist_disable_user",
        arguments={"user_id": 10},
        message_history=[],
    )
    await service.approvals.add(approval)

    first = await service.confirm({generation.id: generation}, approval.approval_id, user_id=2, chat_id=3)
    second = await service.confirm({generation.id: generation}, approval.approval_id, user_id=2, chat_id=3)

    assert first == "The requested action was blocked by the bot's safety policy. No tool action was performed."
    assert second == "Confirmation is unavailable or has expired."
    assert generation.pending_approvals == 0


def test_agent_instructions_cover_stateless_lookup_and_outcome_safety():
    assert "without looking up a user" in AGENT_INSTRUCTIONS
    assert "Never invent" in AGENT_INSTRUCTIONS
    assert "ask for an explicit user identifier" in AGENT_INSTRUCTIONS
    assert "Never claim success" in AGENT_INSTRUCTIONS
    assert "Never claim that a failure had no side effects" in AGENT_INSTRUCTIONS
    assert "stateless" in AGENT_INSTRUCTIONS
