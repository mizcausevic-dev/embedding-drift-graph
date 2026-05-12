"""Tests for the SQLite-backed Store and drift math."""
import math

import pytest

from embedding_drift import Store, cosine_distance


def test_cosine_distance_identical_vectors_is_zero():
    assert cosine_distance([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_cosine_distance_orthogonal_is_one():
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_cosine_distance_opposite_is_two():
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)


def test_cosine_distance_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_distance([1.0, 2.0], [1.0, 2.0, 3.0])


def test_store_upsert_entity():
    store = Store()
    e = store.upsert_entity("ent_a", "Entity A")
    assert e.id == "ent_a"
    assert e.name == "Entity A"
    assert store.get_entity("ent_a") is not None


def test_record_embedding_unknown_entity_raises():
    store = Store()
    with pytest.raises(KeyError):
        store.record_embedding("missing", "encoder-v1", [0.1, 0.2, 0.3])


def test_record_embedding_creates_drift_against_prior():
    store = Store()
    store.upsert_entity("ent_a", "Entity A")
    store.record_embedding("ent_a", "encoder-v1", [1.0, 0.0, 0.0])
    _, drifts = store.record_embedding("ent_a", "encoder-v2", [0.0, 1.0, 0.0])
    assert len(drifts) == 1
    assert drifts[0].from_model == "encoder-v1"
    assert drifts[0].to_model == "encoder-v2"
    assert math.isclose(drifts[0].cosine_distance, 1.0, abs_tol=1e-9)


def test_recording_third_version_yields_two_drift_events():
    store = Store()
    store.upsert_entity("ent_a", "Entity A")
    store.record_embedding("ent_a", "encoder-v1", [1.0, 0.0, 0.0])
    store.record_embedding("ent_a", "encoder-v2", [1.0, 0.0, 0.0])
    _, drifts = store.record_embedding("ent_a", "encoder-v3", [0.0, 1.0, 0.0])
    # Drift against both v1 and v2; both are 1.0.
    assert len(drifts) == 2
    pairs = {(d.from_model, d.to_model) for d in drifts}
    assert pairs == {("encoder-v1", "encoder-v3"), ("encoder-v2", "encoder-v3")}


def test_drift_events_filter_by_min_distance():
    store = Store()
    store.upsert_entity("ent_a", "Entity A")
    store.record_embedding("ent_a", "encoder-v1", [1.0, 0.0])
    store.record_embedding("ent_a", "encoder-v2", [0.99, 0.01])  # tiny drift
    store.record_embedding("ent_a", "encoder-v3", [0.0, 1.0])    # big drift
    big = store.drift_events(min_distance=0.5)
    assert all(d.cosine_distance >= 0.5 for d in big)
    assert len(big) == 2  # v1->v3 and v2->v3 both above 0.5


def test_stats_counts_correctly():
    store = Store()
    store.upsert_entity("a", "A")
    store.upsert_entity("b", "B")
    store.record_embedding("a", "v1", [1.0, 0.0])
    store.record_embedding("a", "v2", [0.0, 1.0])
    store.record_embedding("b", "v1", [1.0, 1.0])
    s = store.stats()
    assert s["entity_count"] == 2
    assert s["embedding_count"] == 3
    assert s["drift_event_count"] == 1
