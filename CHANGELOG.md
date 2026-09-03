# odwi-llm — estado y punto de retome

Resumen para quien **no** siguió las fases de implementación. Diseño
completo: [`.docs/llm_agnostic_connector_design.md`](.docs/llm_agnostic_connector_design.md).
Evidencia del laboratorio: [`llm_lab/experiments/FINDINGS.md`](llm_lab/experiments/FINDINGS.md).

---

## 0.1.0 — Etapa 1: contrato + adapters + suite (2026-09)

Planes ejecutados: `.docs/odwi_llm_experiments_task_plan.md` (laboratorio) y
`.docs/odwi_llm_core_task_plan.md` (este código).

### Qué quedó implementado

- **`odwi_llm.core`** — el contrato de §4, sin ninguna librería de proveedor:
  - `types.py` — `Role`, `Message`, `Intent`, `LLMRequest`, `Usage`,
    `FinishReason`, `LLMResponse`, `StructuredResponse[T]`, `ToolSpec`,
    `ToolCall`, `ToolResult`, `ChatTurnResponse`, `StreamChunk`.
  - `requirements.py` — `LLMRequirements`, `LLMCapabilities`,
    `CapabilityError` (fail-fast al construir el adapter, §4.4).
  - `errors.py` — jerarquía `LLMError` (§4.5). El dominio nunca ve un
    error crudo de `litellm`/`any_llm`.
  - `port.py` — `LLMPort` (ABC): `generate` · `structured` · `stream` ·
    `chat_with_tools` + `capabilities`. Núcleo async.
  - `sync.py` — `SyncLLM` + `_LoopRunner`: fachada bloqueante sobre
    **cualquier** `LLMPort`, funciona aunque ya haya un event loop
    corriendo (Streamlit/Jupyter). No usa `asyncio.run()`.
  - `composition.py` — `FallbackLLM(primary, fallback=None)` (§5.4).
- **`odwi_llm.adapters`** — la única capa que importa una librería de proveedor:
  - `litellm_adapter.LiteLLMAdapter` — **primario** (decisión #4).
  - `anyllm_adapter.AnyLLMAdapter` — **respaldo**. No puede tool calling
    contra LM Studio → `CapabilityError` en construcción.
  - `config.ProviderConfig` — API key por env var, `base_url`, timeout.
  - `pricing.py` — tabla de costo estática (decisión #3). Ninguna
    librería es fuente fiable de costo; el paquete la mantiene.
  - `_shared.py` — helpers + `call_with_retry` (§5.3: 429/5xx/timeout,
    acotado a 3, no reintenta auth/context/filter/schema).
- **Tests** — `uv run pytest`: 109 passed, 4-6 skipped, **sin red ni cuota**.
  - `tests/contract/` — suite exacta contra `FakeAdapter`, offline siempre.
  - `tests/contract_shape/` — suite estructural; por defecto reproduce
    `tests/cassettes/` (31 JSON reales grabados), `--live` va contra
    proveedores, `--live --record` regraba.
  - `tests/architecture/test_import_boundaries.py` — verifica §3.2.
  - `mypy --strict` limpio.

### API que expone el paquete a una app consumidora

```python
from odwi_llm.adapters.litellm_adapter import LiteLLMAdapter
from odwi_llm.adapters.anyllm_adapter import AnyLLMAdapter
from odwi_llm.core.composition import FallbackLLM
from odwi_llm.core.requirements import LLMRequirements
from odwi_llm.core.sync import SyncLLM
from odwi_llm.core.types import LLMRequest, Message, Role, Intent

adapter = LiteLLMAdapter(
    "groq", "openai/gpt-oss-120b",
    LLMRequirements(structured_output=True),   # fail-fast si no se cumple
)
llm = FallbackLLM(primary=adapter)             # fallback=None en Etapa 1

# app async  -> usar el LLMPort directo:      await llm.generate(req)
# app síncrona -> envolver una vez por proceso:
sync = SyncLLM(llm)
resp = sync.generate(LLMRequest(messages=[Message(role=Role.USER, content="…")]))
```

La app depende **solo** de `LLMPort` / `SyncLLM` / `FallbackLLM` y los tipos.
Cambiar `LiteLLMAdapter` por `AnyLLMAdapter` no toca ninguna app.
Credenciales por variable de entorno (`GEMINI_API_KEY`, `GROQ_API_KEY`);
Ollama/LM Studio por `ProviderConfig(base_url=…)`.

### Qué falta explícitamente para la Etapa 2

- **`odwi_llm.guardrails`** — input policy (minimización, allowlists),
  tool policy (validación de args, read-only, límites de fila), output
  policy (esquema, citación). Opcionalmente `any-guardrail` para
  toxicidad/jailbreak (decisión #5). Carpeta aún no creada.
- **`odwi_llm.orchestration`** — loop multi-turno, ejecución de tools,
  retrieval (`search_docs`), historial. Carpeta aún no creada.
- **Activar el fallback:** hoy todas las apps construyen
  `FallbackLLM(primary=…)` sin `fallback`. La Etapa 2 pasa
  `FallbackLLM(primary=LiteLLMAdapter(...), fallback=AnyLLMAdapter(...))`
  cuando la disponibilidad lo justifique (verificado en
  `tests/adapters/test_fallback_real.py`).
- **App de AI Insights** — primera app consumidora (roadmap §10, hito `l7`).
- **`NativeAnthropicAdapter`** — sin API key; se agrega cuando exista,
  sin tocar lo demás (criterio en §5.2.1).
- **Etapa 2 propiamente** — `stream_chat_with_tools` (si la evaluación lo
  justifica), RAG, catálogo de tablas sobre DuckDB. Cada uno arranca con
  su propio `Specify` corto antes del diseño.

### Cómo re-grabar cassettes (§9.5)

Cuando cambie la versión pinneada de `litellm` / `any-llm-sdk`, o el
comportamiento observado derive:

```bash
uv run pytest tests/contract_shape/ --live --record --adapter litellm
uv run pytest tests/contract_shape/ --live --record --adapter anyllm
```

Necesita `GEMINI_API_KEY` / `GROQ_API_KEY` en el entorno y Ollama +
LM Studio corriendo local.
