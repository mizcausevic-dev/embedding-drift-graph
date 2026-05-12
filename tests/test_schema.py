"""Tests for the GraphQL schema layer."""
import pytest

from embedding_drift import Store
from embedding_drift.schema import schema, set_active_store


@pytest.fixture
def fresh_store():
    store = Store()
    set_active_store(store)
    yield store


def test_upsert_entity_and_list(fresh_store):
    result = schema.execute_sync(
        """
        mutation {
            upsertEntity(id: "ent_a", name: "Entity A") { id name }
        }
        """
    )
    assert result.errors is None
    assert result.data["upsertEntity"]["id"] == "ent_a"

    result = schema.execute_sync("{ entities { id name } }")
    assert result.errors is None
    assert result.data["entities"] == [{"id": "ent_a", "name": "Entity A"}]


def test_record_embedding_emits_drift(fresh_store):
    schema.execute_sync(
        'mutation { upsertEntity(id: "x", name: "X") { id } }'
    )
    schema.execute_sync(
        'mutation { recordEmbedding(entityId: "x", modelVersion: "v1", vector: [1.0, 0.0]) { fromModel toModel cosineDistance } }'
    )
    result = schema.execute_sync(
        'mutation { recordEmbedding(entityId: "x", modelVersion: "v2", vector: [0.0, 1.0]) { fromModel toModel cosineDistance } }'
    )
    assert result.errors is None
    drifts = result.data["recordEmbedding"]
    assert len(drifts) == 1
    assert drifts[0]["fromModel"] == "v1"
    assert drifts[0]["toModel"] == "v2"
    assert drifts[0]["cosineDistance"] == pytest.approx(1.0)


def test_drift_query_filters_by_min_distance(fresh_store):
    schema.execute_sync('mutation { upsertEntity(id: "x", name: "X") { id } }')
    schema.execute_sync(
        'mutation { recordEmbedding(entityId: "x", modelVersion: "v1", vector: [1.0, 0.0]) { fromModel } }'
    )
    schema.execute_sync(
        'mutation { recordEmbedding(entityId: "x", modelVersion: "v2", vector: [0.99, 0.01]) { fromModel } }'
    )
    schema.execute_sync(
        'mutation { recordEmbedding(entityId: "x", modelVersion: "v3", vector: [0.0, 1.0]) { fromModel } }'
    )

    result = schema.execute_sync(
        "{ drift(minDistance: 0.5) { fromModel toModel cosineDistance } }"
    )
    assert result.errors is None
    rows = result.data["drift"]
    assert len(rows) == 2
    assert all(r["cosineDistance"] >= 0.5 for r in rows)


def test_stats_query_reflects_writes(fresh_store):
    schema.execute_sync('mutation { upsertEntity(id: "a", name: "A") { id } }')
    schema.execute_sync(
        'mutation { recordEmbedding(entityId: "a", modelVersion: "v1", vector: [1.0]) { fromModel } }'
    )
    result = schema.execute_sync(
        "{ stats { entityCount embeddingCount driftEventCount } }"
    )
    assert result.errors is None
    assert result.data["stats"] == {
        "entityCount": 1,
        "embeddingCount": 1,
        "driftEventCount": 0,
    }
