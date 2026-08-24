from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from auth.config import AuthSettings


@lru_cache
def get_mongo_client(settings: AuthSettings) -> AsyncIOMotorClient:
    # tz_aware=True - without it PyMongo returns naive datetimes from find(), but
    # service.py compares expires_at against datetime.now(timezone.utc) (aware);
    # mixing the two raises TypeError. Needed now that expires_at is stored as a
    # real BSON Date instead of an ISO string (see service.py::create_refresh_token).
    return AsyncIOMotorClient(settings.mongo_uri, tz_aware=True)


def get_users_collection(client: AsyncIOMotorClient, settings: AuthSettings) -> AsyncIOMotorCollection:
    # NOTE: a real deployment needs `await collection.create_index("email", unique=True)`
    # at startup - service.py's signup is check-then-insert, which has a known race
    # window (two concurrent signups for the same email can both succeed) until that
    # unique index exists. Not added here - see finding write-up for scope rationale.
    return client[settings.mongo_db]["users"]


def get_refresh_tokens_collection(client: AsyncIOMotorClient, settings: AuthSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["refresh_tokens"]


async def ensure_refresh_token_indexes(refresh_tokens: AsyncIOMotorCollection) -> None:
    """Call once at app startup (create_index is idempotent - safe to call every
    startup, a no-op once the index already exists). Two indexes:
    - unique on token_hash: guards against a hash collision ever creating two live
      records for the same secret.
    - TTL on expires_at (expireAfterSeconds=0 - the field's own stored value IS the
      deletion time): Mongo's background TTL monitor deletes a document once its
      expires_at has passed, so a token that's never used to refresh (abandoned
      session, browser tab closed without logout) self-cleans instead of
      accumulating in the collection forever. Requires expires_at to be a real BSON
      Date, not an ISO string - see service.py::create_refresh_token."""
    await refresh_tokens.create_index("token_hash", unique=True)
    await refresh_tokens.create_index("expires_at", expireAfterSeconds=0)
