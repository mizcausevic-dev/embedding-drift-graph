"""CLI entry point for embedding-drift-graph.

Usage:
    embedding-drift seed       # populate an in-memory store with synthetic data
    embedding-drift query      # run a sample GraphQL query against it
    embedding-drift schema     # print the GraphQL SDL
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np

from embedding_drift import Store
from embedding_drift.schema import schema, set_active_store

_RNG = np.random.default_rng(seed=42)


def _seed_store(store: Store) -> None:
    """Populate the store with three entities and three model versions each.

    Drift is constructed deterministically:
      - v1 -> v2 has small drift (~0.05)
      - v2 -> v3 has larger drift (~0.20) for entity_a only (concept rename)
    """
    for entity_id, name in [
        ("entity_a", "Concept A"),
        ("entity_b", "Concept B"),
        ("entity_c", "Concept C"),
    ]:
        store.upsert_entity(entity_id, name)
        base = _RNG.normal(size=8).tolist()
        # v1: base vector
        store.record_embedding(entity_id, "encoder-v1", base)
        # v2: small perturbation
        small_perturb = (np.array(base) + _RNG.normal(scale=0.30, size=8)).tolist()
        store.record_embedding(entity_id, "encoder-v2", small_perturb)
        # v3: dramatic perturbation for entity_a (simulates a concept rename),
        # small perturbation for everyone else.
        if entity_id == "entity_a":
            large_perturb = (np.array(base) + _RNG.normal(scale=3.0, size=8)).tolist()
        else:
            large_perturb = (np.array(base) + _RNG.normal(scale=0.30, size=8)).tolist()
        store.record_embedding(entity_id, "encoder-v3", large_perturb)


def _run_sample_query(store: Store) -> None:
    set_active_store(store)
    result = schema.execute_sync(
        """
        {
            stats { entityCount embeddingCount driftEventCount }
            drift {
                entityId
                fromModel
                toModel
                cosineDistance
            }
        }
        """
    )
    if result.errors:
        print("errors:", result.errors, file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result.data, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embedding-drift")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed", help="seed an in-memory store and print stats")
    sub.add_parser("query", help="seed and run a sample drift query")
    sub.add_parser("schema", help="print the GraphQL schema SDL")
    args = parser.parse_args(argv)

    if args.cmd == "schema":
        print(schema.as_str())
        return 0

    store = Store()
    _seed_store(store)
    set_active_store(store)

    if args.cmd == "seed":
        s = store.stats()
        print(json.dumps(s, indent=2))
        return 0

    if args.cmd == "query":
        _run_sample_query(store)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
