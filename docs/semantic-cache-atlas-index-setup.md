# Setup: Atlas Vector Search index for semantic response caching

One-time setup needed in MongoDB Atlas before the semantic response cache
(retrieval-system, `packages/semantic_cache`) will actually return cache
hits. No new database or cluster is needed — this reuses the same Mongo
deployment/database already used by `packages/persona` and `packages/auth`
(same `MONGO_URI`/`MONGO_DB`), just a new collection (`semantic_cache`)
inside it.

**If this index is missing, nothing breaks.** The application fails open —
every cache lookup that can't reach the index is caught, logged, and the
app falls back to running the query normally, exactly as it did before this
feature existed. This setup only needs to happen for the caching to
actually start working; it is not a blocker for deploying the code.

## What to create

A **Vector Search index** (not a regular/Atlas Search index — the "Vector
Search" index type specifically) on:

- **Database:** the value of `MONGO_DB` in this app's environment (same
  database `persona`/`auth` already use — ask if unsure)
- **Collection:** `semantic_cache`
- **Index name:** `semantic_cache_vector_index` — must be exactly this
  string, since the application code (`packages/semantic_cache/src/semantic_cache/repository.py`)
  references it by this literal name.

## Index definition

Via Atlas UI (Database → your cluster → Search tab → Create Search Index →
choose "Vector Search" → JSON editor), or via `mongosh`/Atlas CLI — use this
definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "query_embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "mode"
    }
  ]
}
```

### About `numDimensions: 1024`

This must exactly match the output vector size of whatever embedding model
is configured as `VOYAGE_EMBED_MODEL` in this app's environment (currently
`voyage-4-large` per `.env.example`). **1024 has not been independently
verified against Voyage's live API for this specific model** — please
confirm the actual output dimension of `voyage-4-large` (check Voyage AI's
model documentation, or ask the app team to report the length of a real
embedding vector returned by `POST /v1/embed` with `role: "query_embed"`)
before creating the index. If the app team gives you a different number,
use that number instead of 1024.

A mismatched `numDimensions` will make every vector insert/query against
this index fail (not crash the app — see "fails open" above — but the
cache will never work).

## After creating the index

Nothing else is needed — no code deploy, no restart. The next query the
app makes will start populating the `semantic_cache` collection, and
subsequent semantically-similar queries will start hitting cache once the
index has finished building (Atlas indexes build asynchronously; check the
index status in the Atlas UI shows "Active" before expecting hits).

## Reference

Full design context: `docs/superpowers/specs/2026-08-18-semantic-response-caching-design.md`
(see "Operational setup" section).
