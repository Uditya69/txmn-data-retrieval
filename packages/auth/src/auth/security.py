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
