"""Observability port — design §5 (Stage 2, v0.5).

Interface only, no real backend. `NullObservability` (no-op) is the
default, so no app is forced to wire OpenTelemetry / Langfuse / anything
just to use `Composer`.
"""
