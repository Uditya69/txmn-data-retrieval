from datetime import datetime, timezone

from persona.merge import merge_category_affinity, merge_expertise_patch


async def get_persona(personas, user_id: str) -> dict | None:
    return await personas.find_one({"user_id": user_id})


async def record_signal(
    personas, user_id: str, categories: list[str], expertise_patch: dict | None,
) -> dict:
    existing = await personas.find_one({"user_id": user_id}) or {}
    existing_count = existing.get("query_count", 0)

    merged = merge_expertise_patch(existing, expertise_patch)
    merged["user_id"] = user_id
    merged["category_affinity"] = merge_category_affinity(
        existing.get("category_affinity", {}), existing_count, categories,
    )
    merged["query_count"] = existing_count + 1
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()

    # NOTE: this is a read-modify-write (find_one above, replace_one here), not an
    # atomic update - two concurrent record_signal calls for the same user_id can
    # race: both read the same `existing` snapshot, and whichever replace_one lands
    # last silently overwrites the other's merged result (including a fresh
    # expertise_level). Not fixed here - a real fix needs MongoDB update-pipeline
    # operators ($inc/$mergeObjects etc.) to make the whole read-modify-write
    # atomic, which is a bigger design change than this finding calls for.
    await personas.replace_one({"user_id": user_id}, merged, upsert=True)
    return merged
