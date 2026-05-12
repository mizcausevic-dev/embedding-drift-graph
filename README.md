# embedding-drift-graph

Track how entity embeddings drift across model/encoder versions. Every embedding you record is automatically compared to every prior embedding for the same entity, with the cosine distance materialized as a drift event you can query through a GraphQL API.

Use it when:
- You re-encoded your knowledge base after an encoder upgrade and need to see which entities moved the most
- Your RAG pipeline switched models and quality regressed; you want to know which concepts the new model "thinks differently" about
- You're benchmarking encoders and want a real metric for "semantic stability per entity"

## Install

```bash
pip install -e .[dev]
```

## Quickstart

```bash
embedding-drift seed       # build an in-memory store with 3 entities × 3 model versions
embedding-drift query      # run a sample GraphQL query against the seeded store
embedding-drift schema     # print the GraphQL SDL
```

`embedding-drift query` output (deterministic, seed 42):

```json
{
  "stats": { "entityCount": 3, "embeddingCount": 9, "driftEventCount": 9 },
  "drift": [
    {
      "entityId": "entity_a", "fromModel": "encoder-v1", "toModel": "encoder-v3",
      "cosineDistance": 0.2134830375679505
    },
    {
      "entityId": "entity_a", "fromModel": "encoder-v2", "toModel": "encoder-v3",
      "cosineDistance": 0.18909304125940885
    },
    {
      "entityId": "entity_c", "fromModel": "encoder-v2", "toModel": "encoder-v3",
      "cosineDistance": 0.12445277619347739
    }
    ...
  ]
}
```

Note that `entity_a` shows substantially more drift between `encoder-v2` and `encoder-v3` than `entity_b` or `entity_c` — the seed simulates a "concept rename" for entity_a during the v3 transition, which is exactly the kind of regression you want detected.

## Library usage

```python
from embedding_drift import Store

store = Store("drift.db")          # or omit path for in-memory
store.upsert_entity("concept_x", "Concept X")

emb_v1, drifts = store.record_embedding("concept_x", "encoder-v1", [0.12, -0.04, ...])
# drifts is empty (no prior versions)

emb_v2, drifts = store.record_embedding("concept_x", "encoder-v2", [0.10, -0.03, ...])
# drifts has one DriftEvent comparing v1 -> v2

# Query all drift events above a threshold
for d in store.drift_events(min_distance=0.10):
    print(d.entity_id, d.from_model, "->", d.to_model, "cos_dist", d.cosine_distance)
```

## GraphQL

The package exposes a [Strawberry GraphQL](https://strawberry.rocks) schema. Embed it in any ASGI server:

```python
from strawberry.asgi import GraphQL
from embedding_drift import schema, Store
from embedding_drift.schema import set_active_store

set_active_store(Store("drift.db"))
app = GraphQL(schema)
```

Then run with `uvicorn yourmodule:app --reload` and POST queries to `/`.

### Available queries

```graphql
{
  stats { entityCount embeddingCount driftEventCount }
  entities { id name createdAt }
  embeddings(entityId: "concept_x") { modelVersion vector recordedAt }
  drift(entityId: "concept_x", minDistance: 0.10) {
    fromModel toModel cosineDistance
  }
}
```

### Available mutations

```graphql
mutation {
  upsertEntity(id: "concept_x", name: "Concept X") { id name }
  recordEmbedding(
    entityId: "concept_x",
    modelVersion: "encoder-v2",
    vector: [0.10, -0.03]
  ) { fromModel toModel cosineDistance }
}
```

## Why SQLite (and not pgvector)?

This is a reference implementation that runs anywhere Python runs, with no extra infrastructure. The drift math is pure numpy. For production scale you can swap the storage backend without touching the GraphQL surface — `Store` is a thin layer over four SQL tables you'd recognize in any RDBMS.

## Data model

| Table | Purpose |
|---|---|
| `entities` | Canonical entities (id + display name + created_at) |
| `embeddings` | One row per `(entity_id, model_version)`; vector stored as JSON |
| `drift_events` | Computed on insert: cosine distance between every pair of model versions per entity |

## Development

```bash
pip install -e .[dev]
pytest -v
python -m embedding_drift seed
python -m embedding_drift query
python -m embedding_drift schema   # prints SDL
```

## Dependencies

- [numpy](https://numpy.org/) ≥ 1.26 — vector math
- [strawberry-graphql](https://strawberry.rocks/) ≥ 0.220 — schema definition
- Python `sqlite3` (stdlib) — storage

## License

AGPL-3.0.

---

**Connect:** [LinkedIn](https://www.linkedin.com/in/mirzacausevic/) · [Kinetic Gain](https://kineticgain.com) · [Medium](https://medium.com/@mizcausevic/) · [Skills](https://mizcausevic.com/skills/)
