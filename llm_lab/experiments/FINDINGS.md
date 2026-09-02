# FINDINGS — Laboratorio Etapa 1 (Tareas 6–11)

Resumen **operativo**, no el diseño. Detalle y evidencia:
`experiments/0*.py` (bloques `FIELD MAP` / `OUTCOMES` / `COMPARISON`),
`experiments/10_lib_matrix.py`, `experiments/dumps/`, y la matriz §9.3 del
documento de diseño.

Proveedores del laboratorio: **Gemini** `gemini-3.5-flash-lite`, **Groq**
`openai/gpt-oss-120b`, **Ollama** `qwen3:0.6b`, **LM Studio**
`llama-3.2-3b-instruct`. Anthropic fuera de esta fase.

---

## 1. Qué debe absorber cada adapter — §5.1 pasos 4 y 5

### 1.1 Descartar reasoning (paso 4)

El reasoning **nunca** desaparece solo; el modo estructurado tampoco lo
silencia (Tarea 7). Aparece en sitios distintos según la vía:

| Vía | Dónde llega el reasoning |
|---|---|
| SDK nativo | `message.reasoning` (Groq, Ollama) · `message.reasoning_content` (LM Studio) · `part.thought_signature` bytes (Gemini) |
| LiteLLM | `message.reasoning` (ruta `groq/`) · `message.reasoning_content` (ruta `ollama_chat/`) · `message.provider_specific_fields.thought_signatures` (ruta `gemini/`) |
| Any-LLM | `message.reasoning` como **objeto** `Reasoning(content=…)`, en todos los proveedores; en stream, deltas `Reasoning(...)` |

El adapter borra todos estos antes de construir `LLMResponse`. En stream
hay que filtrar además `delta.reasoning` / `delta.reasoning_content`
(Groq 30 chunks, Ollama 160 chunks en la Tarea 9) o se cuelan como texto.

### 1.2 Parsear tool args (paso 5)

`ToolCall.arguments` del contrato (§4.3) es un dict ya parseado. Realidad:

- **SDK nativo**: Gemini entrega `args` como **dict** (`{'city': 'Paris'}`); Groq / Ollama / LM Studio como **string JSON** (`'{"city":"Paris"}'`).
- **LiteLLM y Any-LLM**: **siempre string JSON**, incluido Gemini (litellm hace `json.dumps` del dict).
- ⇒ el adapter hace `json.loads` salvo cuando ya es dict.
- **LiteLLM + Gemini + tools**: mete la `thought_signature` **dentro del `id`** del tool call: `id = "call_1984494__thought__El4K…"`. El adapter debe cortar el sufijo `__thought__…` para recuperar el `id` real.

### 1.3 Normalizar `finish_reason`

| Situación | Gemini nativo | OpenAI-shaped nativo | LiteLLM | Any-LLM |
|---|---|---|---|---|
| fin normal | `STOP` | `"stop"` | `"stop"` | `"stop"` |
| salida truncada | `MAX_TOKENS` | `"length"` | `"length"` | `"length"` (Any-LLM además **lanza** `LengthFinishReasonError` si se le pide `response_format`) |
| turno con tool call | **`STOP`** (¡sin valor propio!) | `"tool_calls"` | `"tool_calls"` (litellm ya lo deduce) | `"tool_calls"` |

⇒ El adapter mapea `STOP`/`MAX_TOKENS` → `FinishReason.STOP`/`LENGTH`, y
para detectar un turno-de-tools con el SDK nativo de Gemini **no** puede
mirar `finish_reason`: comprueba si hay `function_call` en las `parts`.
Texto vacío + `LENGTH` es un resultado válido (los modelos de reasoning
gastan el presupuesto pensando), no un error.

### 1.4 Mapear errores → jerarquía §4.5

| Caso observado | Origen | Mapear a |
|---|---|---|
| `exceed_context_size_error` HTTP 400 | LM Studio (Tarea 9b) | `LLMContextLengthError` |
| HTTP 400 `INVALID_ARGUMENT` "input token count exceeds…" | Gemini (doc) | `LLMContextLengthError` |
| HTTP 400 `context_length_exceeded` | Groq (doc) | `LLMContextLengthError` |
| **prompt truncado en silencio, HTTP 200** | Ollama (Tarea 9b) | *sin señal* — el adapter Ollama **no puede** detectar desbordamiento de entrada; documentar el gap |
| HTTP 400 `peg-native format` (tool call malformada) | LM Studio (Tarea 8) | `LLMProviderError` (falla de generación server-side, **no** `LLMSchemaError`) |
| HTTP 503 `UNAVAILABLE` "high demand" | Gemini free tier (recurrente) | `LLMProviderError` + reintento técnico acotado (§5.3) — implementado en `_dump_utils.call_with_retry` |
| `NotImplementedError` LM Studio tools | Any-LLM (Tarea 10) | `CapabilityError` al construir el adapter (fail-fast, §4.4) |

Detalle no obvio: el cuerpo de error de LM Studio es **JSON doble-envuelto
y no estándar** — `exc.code` queda `None`; hay que parsear `exc.body` /
el texto del mensaje. Any-LLM ya trae su propia jerarquía tipada
(`ContextLengthExceededError`, `RateLimitError`, `AuthenticationError`,
`LengthFinishReasonError`, …) que se mapea casi 1:1 a la de §4.5.

### 1.5 Streaming → `StreamChunk` (§4.3)

- `text_delta`: `parts[].text` no vacío (Gemini) / `delta.content` (resto). Descartar deltas `""`/`None`; la delta final vacía es el marcador de `done`, no texto.
- `done`: en `finish_reason`; normalizar `STOP` → `stop`.
- `usage`: **SDK nativo** → Groq/Ollama/LM Studio lo mandan en un chunk final `choices:[]` (Groq además en el chunk `stop`); Gemini lo pone acumulativo en **cada** chunk (no hay chunk de usage separado). **LiteLLM** descarta el chunk `choices:[]` → **no hay usage en stream** salvo pasar `stream_options={"include_usage": True}` explícito. **Any-LLM** sí preserva el usage.
- El chunk de usage trae `choices:[]` → acceder a `chunk.choices[0]` revienta; guardar siempre.

---

## 2. Ajustes al mapa de `Intent` — §5.2

Probe `experiments/06_intent_probe.py` (`dumps/intent_probe.txt`): misma
petición dos veces a `temperature=0` y una a `temperature=1`, por SDK
nativo.

| Proveedor | `temperature=0` aceptado | 2× temp=0 idénticas | Nota |
|---|---|---|---|
| Gemini `gemini-3.5-flash-lite` | sí | **no** | flash-lite no es determinista ni a temp=0; el free tier no expone `seed` |
| Groq `openai/gpt-oss-120b` | sí | **sí** | es modelo de reasoning y aun así honra `temperature=0` y reproduce |
| Ollama `qwen3:0.6b` | sí | **no** | varía; necesitaría `options.seed` fijo (y sin thinking) para reproducir |
| LM Studio `llama-3.2-3b-instruct` | sí | **sí** | |

**Conclusión:**

- El borrador de §5.2 se mantiene: `DETERMINISTIC → temperature=0` para
  los 4. **Ninguno rechaza ni ignora `temperature=0`**, y no apareció
  ningún modelo con "adaptive thinking" que exija `temperature=1` en este
  set (gpt-oss aceptó 0) — la nota de §5.2 sobre Fable sigue siendo
  hipotética aquí.
- Matiz importante: **`temperature=0` ≠ reproducible**. Gemini y Ollama
  siguen variando salida a salida. Si la Etapa 1 necesita reproducibilidad
  real (grabar cassettes, §9.5), el adapter debe además fijar `seed` donde
  exista (`seed` en Groq, `options.seed` en Ollama) y asumir que Gemini
  free tier no lo permite.
- `BALANCED` = omitir el parámetro; `CREATIVE` = `temperature=1`. Sin
  cambios: en el probe `temperature=1` siempre difirió de `0`, o sea que
  el parámetro llega y surte efecto en los 4.

---

## 3. Adapter primario para Etapa 1: **LiteLLM**

| Criterio | LiteLLM | Any-LLM |
|---|---|---|
| Celdas de la matriz que pasan | **16 / 16** | 15 / 16 |
| Bloqueos duros | — | **LM Studio + tools = `NotImplementedError`** (envuelve el SDK nativo `lmstudio-python`; tools solo por su API agéntica `.act()`) |
| Extras por proveedor | no | sí (`any-llm-sdk[ollama,lmstudio]`) |
| `structured` | solo string JSON en `content` (el adapter parsea) | **instancia Pydantic** en `message.parsed` |
| `usage` en stream | no, salvo `stream_options` explícito | sí |
| Normalización | agresiva e **inconsistente** (reasoning en 3 campos según ruta; `thought_signature` dentro del `id`) | más limpia; jerarquía de errores propia |
| Peso de dependencias | alto (boto3, tiktoken, tokenizers…) | menor |

**Recomendación:** primario **LiteLLM**. Motivo decisivo: la Etapa 2
necesita tool calling contra los modelos locales, y Any-LLM no puede
hacerlo con LM Studio. Las miserias de normalización de LiteLLM (§1.1–§1.5)
son todas absorbibles en el adapter; un bloqueo de capacidad no.

Any-LLM queda como **candidato a `fallback`** (§5.4, decisión #4) para
Gemini/Groq, donde sí cubre y aporta `message.parsed` y errores tipados.
Ambos adapters se construyen sobre el mismo contrato §4; cambiar de uno a
otro no toca ninguna app.

---

## 4. Insumo para §11 — decisiones abiertas

### Decisión #2 — ¿`stream` + tool calling entra al contrato de Etapa 2?

**Recomendación: NO en el contrato base.** `stream()` se queda solo-texto
como en §4.6; `chat_with_tools()` devuelve el turno completo (no streamea).

- No se midió stream+tools en el laboratorio (Tareas 9 y 10 son solo
  texto, a propósito — §9.4).
- Lo que sí se vio: el stream de **solo texto** ya diverge fuerte —
  Gemini manda chunks gruesos con `usage` acumulativo por chunk; Groq
  manda el `usage` en dos chunks; LiteLLM se come el chunk de `usage`;
  el reasoning llega como su propio flujo de deltas. Sumar deltas de
  tool call encima multiplica esa divergencia.
- Además `gpt-oss-120b` no emite tool calls en paralelo ni siquiera sin
  stream, y LM Studio revienta con dos tool calls (Tarea 8). El terreno
  no está estable ni para el caso no-stream.
- Propuesta concreta para §11 #2: dejar `stream()` texto-only en el
  contrato de Etapa 2. Si una app necesita "primer token rápido" en el
  chat con herramientas, se evalúa una operación aparte
  (`stream_chat_with_tools`) en un laboratorio posterior; no se compromete
  en el contrato ahora.

### Decisión #3 — ¿quién calcula `estimated_cost_usd`?

**Recomendación: una tabla de precios propia del paquete.** El adapter
llena `Usage.estimated_cost_usd` solo si el modelo está en esa tabla; si
no, `None` (el tipo ya lo permite, §4.1).

- **Any-LLM**: no reporta costo en absoluto (sin campo, `model_extra` vacío).
- **LiteLLM**: `response._hidden_params["response_cost"]` da un float
  (`0.0` para local, correcto), pero (a) está **fuera** de `model_dump()`
  y `repr()`, (b) `litellm.completion_cost()` **lanza** "model isn't
  mapped yet" para `openai/gpt-oss-120b` — la tabla de precios que trae
  LiteLLM se queda atrás con modelos nuevos.
- Ningún SDK nativo devuelve costo (Groq da *timings*, no precio).
- ⇒ No delegar el costo a la librería. El paquete mantiene su propia
  tabla `{provider, model, input_usd_per_1M, output_usd_per_1M}` y el
  adapter calcula `input_tokens·pᵢ + output_tokens·pₒ`. Modelos sin
  entrada → `estimated_cost_usd = None`.
