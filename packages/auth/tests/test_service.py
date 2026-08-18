from datetime import datetime, timedelta, timezone

import pytest

from auth.config import get_auth_settings
from auth.security import hash_password
from auth.service import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    InvalidRefreshToken,
    create_refresh_token,
    login,
    revoke_refresh_token,
    rotate_refresh_token,
    signup,
)


@pytest.mark.asyncio
async def test_signup_creates_user_and_returns_user_id(fake_users_collection):
    users = fake_users_collection
    user_id = await signup(users, "alice@example.com", "hunter2")
    assert isinstance(user_id, str) and user_id
    stored = users.documents[0]
    assert stored["email"] == "alice@example.com"
    assert stored["_id"] == user_id
    assert stored["password_hash"] != "hunter2"


@pytest.mark.asyncio
async def test_signup_rejects_duplicate_email(fake_users_collection):
    users = fake_users_collection
    await signup(users, "alice@example.com", "hunter2")
    with pytest.raises(EmailAlreadyRegistered):
        await signup(users, "alice@example.com", "different-password")


@pytest.mark.asyncio
async def test_login_returns_user_id_for_correct_credentials(fake_users_collection):
    users = fake_users_collection
    user_id = await signup(users, "alice@example.com", "hunter2")
    result = await login(users, "alice@example.com", "hunter2")
    assert result == user_id


@pytest.mark.asyncio
async def test_login_rejects_unknown_email(fake_users_collection):
    users = fake_users_collection
    with pytest.raises(InvalidCredentials):
        await login(users, "nobody@example.com", "hunter2")


@pytest.mark.asyncio
async def test_login_rejects_wrong_password(fake_users_collection):
    users = fake_users_collection
    await signup(users, "alice@example.com", "hunter2")
    with pytest.raises(InvalidCredentials):
        await login(users, "alice@example.com", "wrong-password")


@pytest.mark.asyncio
async def test_signup_normalizes_email_case_and_whitespace(fake_users_collection):
    users = fake_users_collection
    await signup(users, "Alice@X.com", "hunter2")
    with pytest.raises(EmailAlreadyRegistered):
        await signup(users, "alice@x.com ", "different-password")


@pytest.mark.asyncio
async def test_login_normalizes_email_case_and_whitespace(fake_users_collection):
    users = fake_users_collection
    user_id = await signup(users, "Alice@X.com", "hunter2")
    result = await login(users, " alice@x.com", "hunter2")
    assert result == user_id


@pytest.mark.asyncio
async def test_create_refresh_token_stores_only_the_hash(fake_refresh_tokens_collection):
    refresh_tokens = fake_refresh_tokens_collection
    settings = get_auth_settings()

    token = await create_refresh_token(refresh_tokens, "user-123", settings)

    stored = refresh_tokens.documents[0]
    assert stored["user_id"] == "user-123"
    assert stored["token_hash"] != token
    assert "expires_at" in stored


@pytest.mark.asyncio
async def test_rotate_refresh_token_returns_user_id_and_issues_a_new_token(fake_refresh_tokens_collection):
    refresh_tokens = fake_refresh_tokens_collection
    settings = get_auth_settings()
    token = await create_refresh_token(refresh_tokens, "user-123", settings)

    user_id, new_token = await rotate_refresh_token(refresh_tokens, token, settings)

    assert user_id == "user-123"
    assert new_token != token
    assert len(refresh_tokens.documents) == 1  # old one consumed, exactly one new one


@pytest.mark.asyncio
async def test_rotate_refresh_token_rejects_reuse_of_an_already_rotated_token(fake_refresh_tokens_collection):
    """A rotated-away token must not work a second time - this is what makes a
    stolen-and-reused token detectable instead of silently accepted forever."""
    refresh_tokens = fake_refresh_tokens_collection
    settings = get_auth_settings()
    token = await create_refresh_token(refresh_tokens, "user-123", settings)
    await rotate_refresh_token(refresh_tokens, token, settings)

    with pytest.raises(InvalidRefreshToken):
        await rotate_refresh_token(refresh_tokens, token, settings)


@pytest.mark.asyncio
async def test_rotate_refresh_token_rejects_unknown_token(fake_refresh_tokens_collection):
    refresh_tokens = fake_refresh_tokens_collection
    settings = get_auth_settings()

    with pytest.raises(InvalidRefreshToken):
        await rotate_refresh_token(refresh_tokens, "never-issued", settings)


@pytest.mark.asyncio
async def test_rotate_refresh_token_rejects_expired_token(fake_refresh_tokens_collection):
    refresh_tokens = fake_refresh_tokens_collection
    settings = get_auth_settings()
    token = await create_refresh_token(refresh_tokens, "user-123", settings)
    refresh_tokens.documents[0]["expires_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    with pytest.raises(InvalidRefreshToken):
        await rotate_refresh_token(refresh_tokens, token, settings)


@pytest.mark.asyncio
async def test_revoke_refresh_token_removes_it(fake_refresh_tokens_collection):
    refresh_tokens = fake_refresh_tokens_collection
    settings = get_auth_settings()
    token = await create_refresh_token(refresh_tokens, "user-123", settings)

    await revoke_refresh_token(refresh_tokens, token)

    assert refresh_tokens.documents == []


@pytest.mark.asyncio
async def test_revoke_refresh_token_is_a_no_op_for_an_unknown_token(fake_refresh_tokens_collection):
    refresh_tokens = fake_refresh_tokens_collection
    await revoke_refresh_token(refresh_tokens, "never-issued")  # must not raise
    assert refresh_tokens.documents == []
