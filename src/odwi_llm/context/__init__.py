"""Context retrieval port — design §6-§7 (Stage 2, v0.5).

How context is retrieved (history, RAG, aggregated data) is independent of
how the execution flow is controlled. Types and the port only; no real
selection logic — that belongs to each app.
"""
