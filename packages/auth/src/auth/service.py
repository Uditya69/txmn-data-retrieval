import uuid
from datetime import datetime, timezone

from auth.security import hash_password, verify_password


class EmailAlreadyRegistered(Exception):
    pass


class InvalidCredentials(Exception):
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
