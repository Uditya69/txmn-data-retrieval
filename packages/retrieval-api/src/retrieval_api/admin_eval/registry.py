import logging

from retrieval_api.admin_eval.adapters import collection_routing, intent, retrieval, slm_intent

# The Langfuse SDK's own logger emits "Authentication error"/"Context error"
# warnings on every traced call when LANGFUSE_PUBLIC_KEY isn't set - third-party
# log output, not something the eval scripts print themselves. Silenced here
# (only for runs started through the admin path) rather than touching the eval
# scripts or their CLI behavior.
logging.getLogger("langfuse").setLevel(logging.CRITICAL)

SUITES: dict[str, dict] = {
    "slm_intent": {"name": "SLM Intent, Filters & Rewrite", "run": slm_intent.run},
    "intent": {"name": "Intent + Filters (exact-match)", "run": intent.run},
    "collection_routing": {"name": "Collection Routing", "run": collection_routing.run},
    "retrieval": {"name": "Retrieval Pipeline", "run": retrieval.run},
}
