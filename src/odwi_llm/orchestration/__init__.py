"""Orchestration — design §8-§11 (Stage 2, v0.5).

The orchestrator is mechanism, not policy: it knows it must ask Policy at
defined points, not what the rules are. `Orchestrator` is a substitutable
port; `Workflow` is the thin reference implementation, and a future
agentic engine would be another adapter of the same port.
"""
