"""Track how entity embeddings drift across LLM/encoder model versions.

Public API:

- ``Store`` — SQLite-backed storage of entities and embedding samples.
- ``cosine_distance`` — vector math used by the store to compute drift.
- ``schema`` — Strawberry GraphQL schema for query and mutation access.

Specification of the data model lives in README.md.
"""
from embedding_drift.store import Embedding, Entity, Store, cosine_distance, DriftEvent
from embedding_drift.schema import schema

__all__ = [
    "Embedding",
    "Entity",
    "Store",
    "cosine_distance",
    "DriftEvent",
    "schema",
]

__version__ = "0.1.0"
