"""Smoke test for the observability fake itself (Task 6)."""

from odwi_llm.observability.port import ObservabilityPort

from ._fake_observability import FakeObservability


def test_satisfies_the_port_and_starts_empty() -> None:
    obs: ObservabilityPort = FakeObservability()
    obs.emit("noop")
    assert isinstance(obs, FakeObservability)
    assert FakeObservability().events == []


def test_records_events_in_order_with_fields() -> None:
    obs = FakeObservability()
    obs.emit("execution_started", session_id="s1")
    obs.emit("policy_decision", phase="input", decision="allow")
    obs.emit("policy_decision", phase="output", decision="deny")
    obs.emit("result", finish_reason="stop")

    assert obs.names == [
        "execution_started",
        "policy_decision",
        "policy_decision",
        "result",
    ]
    assert obs.events[0] == ("execution_started", {"session_id": "s1"})
    assert obs.fields_for("policy_decision") == [
        {"phase": "input", "decision": "allow"},
        {"phase": "output", "decision": "deny"},
    ]
