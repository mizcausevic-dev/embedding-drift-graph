"""SQLite-backed store for entity embeddings and computed drift events.

The store records every embedding sample tagged with the model version
that produced it. Each time a new embedding lands for an entity that
already has a sample on a previous model version, a drift event is
computed and recorded automatically.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    created_at: float


@dataclass(frozen=True)
class Embedding:
    entity_id: str
    model_version: str
    vector: tuple[float, ...]
    recorded_at: float


@dataclass(frozen=True)
class DriftEvent:
    entity_id: str
    from_model: str
    to_model: str
    cosine_distance: float
    recorded_at: float


def cosine_distance(a: Iterable[float], b: Iterable[float]) -> float:
    """Return the cosine distance (1 - cosine similarity) between two vectors.

    Inputs may be any sequence of floats. ``a`` and ``b`` must have the
    same dimensionality.
    """
    av = np.asarray(list(a), dtype=np.float64)
    bv = np.asarray(list(b), dtype=np.float64)
    if av.shape != bv.shape:
        raise ValueError(f"shape mismatch: {av.shape} vs {bv.shape}")
    norm = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if norm == 0.0:
        return 0.0
    similarity = float(np.dot(av, bv) / norm)
    # Clamp to [-1, 1] to defend against floating-point overflow.
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    entity_id      TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    vector_json    TEXT NOT NULL,
    recorded_at    REAL NOT NULL,
    PRIMARY KEY (entity_id, model_version),
    FOREIGN KEY (entity_id) REFERENCES entities (id)
);

CREATE TABLE IF NOT EXISTS drift_events (
    entity_id        TEXT NOT NULL,
    from_model       TEXT NOT NULL,
    to_model         TEXT NOT NULL,
    cosine_distance  REAL NOT NULL,
    recorded_at      REAL NOT NULL,
    PRIMARY KEY (entity_id, from_model, to_model),
    FOREIGN KEY (entity_id) REFERENCES entities (id)
);
"""


class Store:
    """SQLite-backed storage for the embedding-drift graph."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- writes --------------------------------------------------------

    def upsert_entity(self, entity_id: str, name: str) -> Entity:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO entities (id, name, created_at) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name
            """,
            (entity_id, name, now),
        )
        self._conn.commit()
        return self.get_entity(entity_id)  # type: ignore[return-value]

    def record_embedding(
        self,
        entity_id: str,
        model_version: str,
        vector: Iterable[float],
    ) -> tuple[Embedding, list[DriftEvent]]:
        """Insert an embedding and compute drift events vs every prior
        embedding of the same entity. Returns the inserted embedding
        plus the drift events created.
        """
        if self.get_entity(entity_id) is None:
            raise KeyError(f"unknown entity: {entity_id}")

        v = tuple(float(x) for x in vector)
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO embeddings (entity_id, model_version, vector_json, recorded_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_id, model_version)
                DO UPDATE SET vector_json = excluded.vector_json,
                              recorded_at = excluded.recorded_at
            """,
            (entity_id, model_version, json.dumps(v), now),
        )

        # Compute drift against every other embedding of this entity.
        priors = self.embeddings_for_entity(entity_id)
        drift_events: list[DriftEvent] = []
        for prior in priors:
            if prior.model_version == model_version:
                continue
            dist = cosine_distance(prior.vector, v)
            self._conn.execute(
                """
                INSERT INTO drift_events
                    (entity_id, from_model, to_model, cosine_distance, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(entity_id, from_model, to_model)
                    DO UPDATE SET cosine_distance = excluded.cosine_distance,
                                  recorded_at = excluded.recorded_at
                """,
                (entity_id, prior.model_version, model_version, dist, now),
            )
            drift_events.append(
                DriftEvent(
                    entity_id=entity_id,
                    from_model=prior.model_version,
                    to_model=model_version,
                    cosine_distance=dist,
                    recorded_at=now,
                )
            )

        self._conn.commit()
        emb = Embedding(
            entity_id=entity_id,
            model_version=model_version,
            vector=v,
            recorded_at=now,
        )
        return emb, drift_events

    # ---- reads ---------------------------------------------------------

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self._conn.execute(
            "SELECT id, name, created_at FROM entities WHERE id = ?",
            (entity_id,),
        ).fetchone()
        if row is None:
            return None
        return Entity(id=row["id"], name=row["name"], created_at=row["created_at"])

    def list_entities(self) -> list[Entity]:
        rows = self._conn.execute(
            "SELECT id, name, created_at FROM entities ORDER BY created_at ASC"
        ).fetchall()
        return [
            Entity(id=r["id"], name=r["name"], created_at=r["created_at"]) for r in rows
        ]

    def embeddings_for_entity(self, entity_id: str) -> list[Embedding]:
        rows = self._conn.execute(
            """
            SELECT entity_id, model_version, vector_json, recorded_at
              FROM embeddings
             WHERE entity_id = ?
          ORDER BY recorded_at ASC
            """,
            (entity_id,),
        ).fetchall()
        return [
            Embedding(
                entity_id=r["entity_id"],
                model_version=r["model_version"],
                vector=tuple(json.loads(r["vector_json"])),
                recorded_at=r["recorded_at"],
            )
            for r in rows
        ]

    def drift_events(
        self,
        *,
        entity_id: str | None = None,
        min_distance: float = 0.0,
    ) -> list[DriftEvent]:
        query = (
            "SELECT entity_id, from_model, to_model, cosine_distance, recorded_at"
            "  FROM drift_events"
            " WHERE cosine_distance >= ?"
        )
        params: list = [min_distance]
        if entity_id is not None:
            query += " AND entity_id = ?"
            params.append(entity_id)
        query += " ORDER BY cosine_distance DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            DriftEvent(
                entity_id=r["entity_id"],
                from_model=r["from_model"],
                to_model=r["to_model"],
                cosine_distance=r["cosine_distance"],
                recorded_at=r["recorded_at"],
            )
            for r in rows
        ]

    def stats(self) -> dict[str, int]:
        return {
            "entity_count": self._conn.execute(
                "SELECT COUNT(*) AS n FROM entities"
            ).fetchone()["n"],
            "embedding_count": self._conn.execute(
                "SELECT COUNT(*) AS n FROM embeddings"
            ).fetchone()["n"],
            "drift_event_count": self._conn.execute(
                "SELECT COUNT(*) AS n FROM drift_events"
            ).fetchone()["n"],
        }
