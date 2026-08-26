from __future__ import annotations

from enum import StrEnum


class FailureKind(StrEnum):
    TOOL_REPORTED_FAILURE = "tool_reported_failure"
    TOOL_OUTCOME_UNKNOWN = "tool_outcome_unknown"
    TOOL_UNAVAILABLE = "tool_unavailable"
    POLICY_BLOCKED = "policy_blocked"


class AgentExecutionError(RuntimeError):
    def __init__(
        self,
        kind: FailureKind,
        *,
        server: str | None = None,
        tool_name: str | None = None,
        dispatched: bool = False,
        outcome_known: bool = True,
    ) -> None:
        self.kind = kind
        self.server = server
        self.tool_name = tool_name
        self.dispatched = dispatched
        self.outcome_known = outcome_known
        super().__init__(kind.value)


_FAILURE_PRIORITY = {
    FailureKind.TOOL_OUTCOME_UNKNOWN: 0,
    FailureKind.TOOL_REPORTED_FAILURE: 1,
    FailureKind.TOOL_UNAVAILABLE: 2,
    FailureKind.POLICY_BLOCKED: 3,
}


def select_execution_error(exc: BaseException) -> AgentExecutionError | None:
    """Return the safest typed failure from nested/parallel exceptions."""
    found: list[AgentExecutionError] = []
    visited: set[int] = set()

    def visit(current: BaseException | None) -> None:
        if current is None or id(current) in visited:
            return
        visited.add(id(current))
        if isinstance(current, AgentExecutionError):
            found.append(current)
        if isinstance(current, BaseExceptionGroup):
            for nested in current.exceptions:
                visit(nested)
        visit(current.__cause__)
        visit(current.__context__)

    visit(exc)
    return min(found, key=lambda item: _FAILURE_PRIORITY[item.kind], default=None)


def contains_unexpected_failure(exc: BaseException) -> bool:
    """Detect an unexpected branch without treating a typed error's cause as a defect."""
    if isinstance(exc, AgentExecutionError):
        return False
    if isinstance(exc, BaseExceptionGroup):
        return any(contains_unexpected_failure(nested) for nested in exc.exceptions)
    return select_execution_error(exc) is None
