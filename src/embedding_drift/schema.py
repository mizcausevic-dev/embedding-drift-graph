"""Strawberry GraphQL schema for the embedding-drift store."""
from __future__ import annotations

from typing import Optional

import strawberry

from embedding_drift import store as store_module


_active_store: Optional[store_module.Store] = None


def get_active_store() -> store_module.Store:
    """Return the per-process active store, creating one in-memory if absent."""
    global _active_store
    if _active_store is None:
        _active_store = store_module.Store()
    return _active_store


def set_active_store(store: store_module.Store) -> None:
    """Override the active store (used by tests and explicit deployments)."""
    global _active_store
    _active_store = store


# --- types ----------------------------------------------------------------


@strawberry.type
class Entity:
    id: str
    name: str
    created_at: float


@strawberry.type
class Embedding:
    entity_id: str
    model_version: str
    vector: list[float]
    recorded_at: float


@strawberry.type
class DriftEvent:
    entity_id: str
    from_model: str
    to_model: str
    cosine_distance: float
    recorded_at: float


@strawberry.type
class Stats:
    entity_count: int
    embedding_count: int
    drift_event_count: int


# --- conversions ----------------------------------------------------------


def _to_gql_entity(e: store_module.Entity) -> Entity:
    return Entity(id=e.id, name=e.name, created_at=e.created_at)


def _to_gql_embedding(e: store_module.Embedding) -> Embedding:
    return Embedding(
        entity_id=e.entity_id,
        model_version=e.model_version,
        vector=list(e.vector),
        recorded_at=e.recorded_at,
    )


def _to_gql_drift(e: store_module.DriftEvent) -> DriftEvent:
    return DriftEvent(
        entity_id=e.entity_id,
        from_model=e.from_model,
        to_model=e.to_model,
        cosine_distance=e.cosine_distance,
        recorded_at=e.recorded_at,
    )


# --- query / mutation -----------------------------------------------------


@strawberry.type
class Query:
    @strawberry.field
    def entities(self) -> list[Entity]:
        return [_to_gql_entity(e) for e in get_active_store().list_entities()]

    @strawberry.field
    def embeddings(self, entity_id: str) -> list[Embedding]:
        return [
            _to_gql_embedding(e)
            for e in get_active_store().embeddings_for_entity(entity_id)
        ]

    @strawberry.field
    def drift(
        self,
        entity_id: Optional[str] = None,
        min_distance: float = 0.0,
    ) -> list[DriftEvent]:
        return [
            _to_gql_drift(e)
            for e in get_active_store().drift_events(
                entity_id=entity_id, min_distance=min_distance
            )
        ]

    @strawberry.field
    def stats(self) -> Stats:
        s = get_active_store().stats()
        return Stats(
            entity_count=s["entity_count"],
            embedding_count=s["embedding_count"],
            drift_event_count=s["drift_event_count"],
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    def upsert_entity(self, id: str, name: str) -> Entity:
        return _to_gql_entity(get_active_store().upsert_entity(id, name))

    @strawberry.mutation
    def record_embedding(
        self,
        entity_id: str,
        model_version: str,
        vector: list[float],
    ) -> list[DriftEvent]:
        _, events = get_active_store().record_embedding(
            entity_id, model_version, vector
        )
        return [_to_gql_drift(e) for e in events]


schema = strawberry.Schema(query=Query, mutation=Mutation)
