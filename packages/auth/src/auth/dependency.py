from fastapi import Header

from auth.config import get_auth_settings
from auth.security import decode_access_token


async def get_current_user_id(authorization: str | None = Header(default=None)) -> str | None:
    if authorization is None or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ")
    settings = get_auth_settings()
    return decode_access_token(token, settings)
