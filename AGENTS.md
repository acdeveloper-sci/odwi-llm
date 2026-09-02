# odwi-llm — Contexto para Claude Code

## Qué es este proyecto
Laboratorio + paquete interno agnóstico de LLM. Ver documentos de diseño en la raíz:
- `llm_agnostic_connector_design.md` — arquitectura, contrato, decisiones cerradas
- `odwi_llm_experiments_task_plan.md` — plan de tareas de esta fase (experiments/)

Lee ambos antes de escribir código. No hay código escrito todavía.

## Reglas del proyecto
- Todo el código (nombres, comentarios, docstrings, strings) se escribe en inglés,
  sin importar el idioma de la conversación.
- Gestor de paquetes: `uv`. No usar pip directo.
- Sigue el plan de tareas en orden. No saltes a `core/`/adapters hasta que
  `odwi_llm_experiments_task_plan.md` esté completo (Tarea 13).
- Cada tarea tiene un criterio "Hecho cuando" — verifícalo antes de pasar a la siguiente.
- Anthropic está excluido de este laboratorio (sin API key). No lo agregues sin que se indique.

## Al terminar cada tarea
Resume en 2-3 líneas qué se hizo y qué archivo(s) se generaron, antes de continuar con la siguiente.
