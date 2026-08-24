import pytest

from auth.config import get_auth_settings
from auth.dependency import get_current_user_id
from auth.security import create_access_token


@pytest.mark.asyncio
async def test_returns_user_id_for_valid_bearer_token():
    settings = get_auth_settings()
    token = create_access_token("user-123", settings)
    result = await get_current_user_id(authorization=f"Bearer {token}")
    assert result == "user-123"


@pytest.mark.asyncio
async def test_returns_none_for_missing_header():
    result = await get_current_user_id(authorization=None)
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_for_malformed_header():
    result = await get_current_user_id(authorization="not-a-bearer-token")
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_for_invalid_token():
    result = await get_current_user_id(authorization="Bearer garbage-token")
    assert result is None
