import uuid
from datetime import datetime, timedelta, timezone

from auth.config import AuthSettings
from auth.security import generate_refresh_token, hash_password, hash_refresh_token, verify_password


class EmailAlreadyRegistered(Exception):
    pass


class InvalidCredentials(Exception):
    pass


class InvalidRefreshToken(Exception):
    pass


async def signup(users, email: str, password: str) -> str:
    email = email.strip().lower()
    existing = await users.find_one({"email": email})
    if existing is not None:
        raise EmailAlreadyRegistered(email)

    user_id = str(uuid.uuid4())
    await users.insert_one({
        "_id": user_id,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return user_id


async def login(users, email: str, password: str) -> str:
    email = email.strip().lower()
    user = await users.find_one({"email": email})
    if user is None or not verify_password(password, user["password_hash"]):
        raise InvalidCredentials(email)
    return user["_id"]


async def create_refresh_token(refresh_tokens, user_id: str, settings: AuthSettings) -> str:
    token = generate_refresh_token()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expiry_days)
    await refresh_tokens.insert_one({
        "token_hash": hash_refresh_token(token),
        "user_id": user_id,
        # Stored as a real datetime (BSON Date), not .isoformat() - a Mongo TTL
        # index (see db.py::ensure_indexes) only fires on a Date-typed field, so an
        # ISO string here would silently never self-clean.
        "expires_at": expires_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return token


async def rotate_refresh_token(refresh_tokens, token: str, settings: AuthSettings) -> tuple[str, str]:
    """Validates `token`, consumes it (deletes the old record), and issues a
    replacement - rotation on every use, not just on expiry. A stolen-and-reused
    refresh token is thereby limited to a single use: whichever party (attacker
    or legitimate client) redeems it first invalidates it for the other, making
    reuse detectable (the second redemption attempt fails outright) rather than
    silently allowing indefinite reuse of one static long-lived secret. Returns
    (user_id, new_refresh_token); raises InvalidRefreshToken if `token` doesn't
    match a live, unexpired record."""
    token_hash = hash_refresh_token(token)
    doc = await refresh_tokens.find_one({"token_hash": token_hash})
    if doc is None:
        raise InvalidRefreshToken(token)
    await refresh_tokens.delete_one({"token_hash": token_hash})
    expires_at = doc["expires_at"]
    if expires_at < datetime.now(timezone.utc):
        raise InvalidRefreshToken(token)
    new_token = await create_refresh_token(refresh_tokens, doc["user_id"], settings)
    return doc["user_id"], new_token


async def revoke_refresh_token(refresh_tokens, token: str) -> None:
    await refresh_tokens.delete_one({"token_hash": hash_refresh_token(token)})
