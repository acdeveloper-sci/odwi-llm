# odwi-llm

Provider-agnostic LLM access layer. Applications depend only on `LLMPort`
and its types; LiteLLM, Any-LLM or a native SDK sit behind the port as
interchangeable adapters.

- Design: [`.docs/llm_agnostic_connector_design.md`](.docs/llm_agnostic_connector_design.md)
- Stage-1 lab (separate `uv` project): [`llm_lab/`](llm_lab/) — raw provider
  behaviour, `experiments/FINDINGS.md`, filled test matrix (§9.3).
- This package's task plan: [`.docs/odwi_llm_core_task_plan.md`](.docs/odwi_llm_core_task_plan.md)

Package manager: `uv`. Python: 3.13+.
