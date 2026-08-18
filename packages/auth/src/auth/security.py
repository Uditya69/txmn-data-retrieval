import hashlib
import secrets
import time

import bcrypt
import jwt

from auth.config import AuthSettings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, settings: AuthSettings) -> str:
    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + settings.jwt_expiry_minutes * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str, settings: AuthSettings) -> str | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    return payload.get("user_id")


def generate_refresh_token() -> str:
    """Opaque random token, not a JWT - unlike an access token it must be
    revocable (logout, rotation-on-use), which requires a server-side record to
    delete. A JWT refresh token would just re-decode as valid until its own
    expiry with no way to invalidate it early."""
    return secrets.token_urlsafe(32)


def hash_refresh_token(token: str) -> str:
    """Only the hash is ever stored in Mongo - a DB read/leak must not hand out
    usable refresh tokens directly, mirroring why passwords are hashed and not
    stored raw. SHA-256 (not bcrypt) is deliberate here: the token itself is
    already 32 bytes of CSPRNG output (not a low-entropy human password), so
    there's nothing for bcrypt's slow, salted hashing to defend against that a
    fast, deterministic hash doesn't already cover for an unguessable input -
    and determinism is required anyway, to look the token up by exact hash
    match on refresh."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
