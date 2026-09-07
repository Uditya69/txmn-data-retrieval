"""Server-readiness check for a long unattended eval run.

Before kicking off a 5-10 hour eval run against a local/self-hosted LLM, confirm the
things that would otherwise fail loudly (or silently degrade) hours in: ES reachable
and holding the configured index, every Milvus collection retrieval_eval.py expects
present, and the model-gateway resolving + actually able to serve each role it needs
(slm, reranker, synthesis, query_embed). The slm role additionally gets one real chat
call (not just role resolution) since that's the role most likely to be sitting behind
a flaky local vLLM endpoint - config resolving cleanly doesn't mean the model behind it
is actually answering.

Usage (from repo root):
    uv run python evals/scripts/preflight.py --gateway-url http://localhost:8001

Exits 0 if every check passes, 1 otherwise - safe to gate a headless run script on.
"""
import argparse
import asyncio
import time

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.gateway_client import GatewayClient

_GATEWAY_ROLES = ["slm", "reranker", "synthesis", "query_embed"]


async def check_elasticsearch(settings) -> tuple[bool, str]:
    client = get_es_client(settings)
    try:
        if not await client.ping():
            return False, "ES ping failed"
        count = await client.count(index=settings.es_index)
        return True, f"index {settings.es_index!r} reachable, {count['count']} docs"
    except Exception as exception:
        return False, f"{type(exception).__name__}: {exception}"
    finally:
        await client.close()


async def check_milvus(settings) -> tuple[bool, str]:
    client = get_milvus_client(settings)
    try:
        present = set(await asyncio.to_thread(client.list_collections))
        missing = [name for name in MILVUS_COLLECTIONS if name not in present]
        if missing:
            return False, f"missing collections: {missing}"
        return True, f"all {len(MILVUS_COLLECTIONS)} collections present"
    except Exception as exception:
        return False, f"{type(exception).__name__}: {exception}"
    finally:
        client.close()


async def check_gateway_roles(gateway: GatewayClient) -> tuple[bool, str]:
    resolved = {}
    for role in _GATEWAY_ROLES:
        try:
            resolved[role] = await gateway.get_model(role=role)
        except Exception as exception:
            return False, f"role {role!r} failed to resolve: {type(exception).__name__}: {exception}"
    return True, ", ".join(f"{role}={model}" for role, model in resolved.items())


async def check_slm_live_call(gateway: GatewayClient) -> tuple[bool, str]:
    started = time.perf_counter()
    try:
        response = await gateway.chat(role="slm", messages=[{"role": "user", "content": "Reply with the single word: ready"}])
    except Exception as exception:
        return False, f"{type(exception).__name__}: {exception}"
    elapsed = time.perf_counter() - started
    if not response or not response.strip():
        return False, f"empty response after {elapsed:.1f}s"
    return True, f"responded in {elapsed:.1f}s: {response.strip()[:80]!r}"


async def run(gateway_url: str, skip_llm_call: bool) -> bool:
    settings = get_settings()
    gateway = GatewayClient(gateway_url or settings.gateway_url, trace_enabled=False)

    checks = [
        ("elasticsearch", check_elasticsearch(settings)),
        ("milvus", check_milvus(settings)),
        ("gateway roles", check_gateway_roles(gateway)),
    ]
    if not skip_llm_call:
        checks.append(("slm live call", check_slm_live_call(gateway)))

    all_ok = True
    for name, coro in checks:
        ok, detail = await coro
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ES/Milvus/gateway readiness before a long eval run")
    parser.add_argument("--gateway-url", help="override GATEWAY_URL (useful when running outside Docker)")
    parser.add_argument(
        "--skip-llm-call", action="store_true",
        help="skip the one live chat call to the slm role (faster, but doesn't prove the model is actually answering)",
    )
    args = parser.parse_args()
    ok = asyncio.run(run(args.gateway_url, args.skip_llm_call))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
